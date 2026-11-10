#!/bin/bash
# No-compromise windows (nc) on the Spark, 2026-11-09/10, from bi-leg.sh.
# Binary $BD (default ~/nc/a121).  glm53-spark must be at 0 replicas.
# Usage: ek-window.sh LEG...   (legs below; each starts and stops its own server)
set -u
cd /srv/glm-5.3-flash; OUT=legs-nc; mkdir -p $OUT
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
BD=${BD:-$HOME/nc/a121}; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@} ${!LLAMA_PREFILL_@} ${!LLAMA_BI_@} ${!LLAMA_SLOT_@} ${!LLAMA_CLIENT_@} ${!LLAMA_PRIORITY_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
EXACT6="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXACT_ATTN_COLS=1 GGML_CUDA_ARGSORT_EXACT_ROWS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
base_args() { # base_args PARALLEL CTX
  echo "-m $M --alias glm-5.3-flash --jinja --ctx-size $2 --parallel $1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --metrics --no-warmup --fit off --load-mode mmap -lv 4 --host 127.0.0.1 --port 18080"
}
B="${CLIENT:-python3 -I}"
PID=""; WD=""
start() { # start TAG "ENV" PARALLEL CTX
  local TAG=$1 EXTRA=$2
  pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; return 1; }
  cp snapshot/el-seed.k3sn snapshot/nc-$TAG.k3sn
  echo "=== LEG $TAG parallel=$3 ctx=$4 extra=[$EXTRA] $(date -u +%T)"
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/nc-$TAG.k3sn $EXTRA $BIN $(base_args $3 $4) ${XARGS:-} > $OUT/server-$TAG.log 2>&1 &
  PID=$!
  ( low=999999; while kill -0 $PID 2>/dev/null; do a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
      [ "$a" -lt "$low" ] && low=$a && echo "$low" > $OUT/lowmem-$TAG.txt
      if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $PID; break; fi; sleep 1; done ) &
  WD=$!
  local i
  for i in $(seq 1 300); do
    curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s"; grep -a -E "elastic|backed|n_stream|n_slots" $OUT/server-$TAG.log | head -8 | cut -c1-250; return 0; }
    kill -0 $PID 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -8 | cut -c1-300; return 1; }
    sleep 2
  done
  echo "LEG-FAIL $TAG not ready"; return 1
}
stop() { # stop TAG
  local TAG=$1
  kill -INT $PID 2>/dev/null; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -KILL $PID 2>/dev/null; wait $PID 2>/dev/null
  kill $WD 2>/dev/null
  echo "elastic lines: $(grep -a -c "elastic KV" $OUT/server-$TAG.log); refused: $(grep -a -c REFUSED $OUT/server-$TAG.log)"
  echo "lowest MemAvailable $(cat $OUT/lowmem-$TAG.txt) MiB; now $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) MiB; END $TAG $(date -u +%T)"
  rm -f snapshot/nc-$TAG.k3sn
}
R=$OUT/results.jsonl
# bi-leg.sh TAG PARALLEL CTX "EXTRA_ENV" "ekbench args" ["ekbench args" ...]   (extra server args via XARGS env)
TAG=$1; PAR=$2; CTX=$3; EXTRA=$4; shift 4
start $TAG "$EXTRA" $PAR $CTX || { stop $TAG; exit 1; }
for cmd in "$@"; do
  $B $cmd --tag $TAG | tee -a $R
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED during [$cmd]"; break; }
done
stop $TAG
echo "NC-LEG-DONE $TAG $(date -u +%T)"
