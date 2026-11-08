#!/bin/bash
# Patch 0051 window on the Spark (2026-11-08): KV + indexer cache in device allocations vs registered host memory,
# same binary (~/llama-k3-kh/build-kh = production tree + 0050 + 0051). glm53-spark must be at 0 replicas.
# Both legs: production env and args except LLAMA_ELASTIC_KV off (0051 replaces it) and --ctx-size 262144 (fixed
# KV of 4.9 GB that fits next to the 78000 MiB expert cache; the memory valve keeps 6 GiB free either way).
#   kvprobe  microbenchmark of the access patterns in the three placements (dev / reg / halloc)
#   perf     standard request x(1 warm + 3), then decode after 16k, 100k and 200k-token prompts (md5 each)
#   prof     GGML_CUDA_OP_PROFILE=16 + LLAMA_LOGITS_FNV=1, 16k and 100k prompts, 64 decode tokens each, then a
#            needle (fact at the start of a 32k prompt, question at the end)
# Usage: kh-prof.sh [legs...]   default: D-prof H-prof (kh-window.sh with -lv 4, so the op profile lines are logged)
set -u
cd /srv/glm-5.3-flash; OUT=legs-kh; mkdir -p $OUT
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
BD=$HOME/llama-k3-kh/build-kh; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
CTX=262144
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
ARGS="-m $M --alias glm-5.3-flash --jinja --ctx-size $CTX --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --metrics --no-warmup --fit off --load-mode mmap --host 127.0.0.1 --port 18080 -lv 4"

leg() { # leg TAG placement(D|H) mode(perf|prof)
  local TAG=$1 PL=$2 MODE=$3 EXTRA=""
  [ $PL = H ] && EXTRA="LLAMA_KV_HOST=1"
  [ $MODE = prof ] && EXTRA="$EXTRA GGML_CUDA_OP_PROFILE=16 LLAMA_LOGITS_FNV=1"
  pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; return 1; }
  cp snapshot/sm-seed.k3sn snapshot/kh-$TAG.k3sn
  echo "=== LEG $TAG placement=$PL mode=$MODE extra=[$EXTRA] $(date +%T)"
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/kh-$TAG.k3sn $EXTRA $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & local pid=$!
  ( low=999999; while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      [ "$a" -lt "$low" ] && low=$a && echo "$low" > $OUT/lowmem-$TAG.txt
      if [ "$a" -lt 4096 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) & local WD=$!
  local ok=0 i
  for i in $(seq 1 300); do
    curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s"; ok=1; break; }
    kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -5 | cut -c1-300; break; }
    sleep 2
  done
  if [ $ok = 1 ]; then
    grep -a -E "LLAMA_KV_HOST|host_kv_buffer|KV buffer size|elastic" $OUT/server-$TAG.log | cut -c1-260
    if [ $MODE = perf ]; then
      python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $TAG --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
      python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $TAG-depth --lens 16384,100000,200000 --n-predict 128 | tee -a $OUT/results.jsonl
    else
      python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $TAG-depth --lens 16384,100000 --n-predict 64 | tee -a $OUT/results.jsonl
      python3 -I needle.py http://127.0.0.1:18080 $TAG-needle 32000 | tee -a $OUT/results.jsonl
    fi
  fi
  kill -INT $pid 2>/dev/null; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
  kill $WD 2>/dev/null
  [ $MODE = prof ] && { grep -a -o "LOGITS_FNV.*" $OUT/server-$TAG.log > $OUT/$TAG.fnv; echo "fnv lines $(wc -l < $OUT/$TAG.fnv)"; }
  echo "lowest MemAvailable $(cat $OUT/lowmem-$TAG.txt) MiB; now $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB; END $TAG $(date +%T)"
  rm -f snapshot/kh-$TAG.k3sn
}

LEGS="${*:-D-prof H-prof}"
for L in $LEGS; do
  case $L in
    probe) pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT probe: a llama-server is running"; continue; }
           echo "=== kvprobe $(date +%T)"; ./kvprobe 12 16384,100000,262144,1000000 dev,reg,regnothp,halloc 2>&1 | tee $OUT/kvprobe.txt ;;
    D-perf|H-perf|D-prof|H-prof) leg ${L/-/} ${L%%-*} ${L##*-} ;;
    *) echo "unknown leg $L" ;;
  esac
done
echo "KH-PROF-DONE $(date +%T)"
