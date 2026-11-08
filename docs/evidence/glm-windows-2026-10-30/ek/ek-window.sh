#!/bin/bash
# Patch 0052 window on the Spark (2026-11-08): elastic KV per stream, multi-slot serving.
# Binary ~/llama-k3-ek/build-ek = production tree + 0050 + 0051 + 0052. glm53-spark must be at 0 replicas.
# Usage: ek-window.sh LEG...   (legs below; each starts and stops its own server)
set -u
cd /srv/glm-5.3-flash; OUT=legs-ek; mkdir -p $OUT
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
BD=$HOME/llama-k3-ek/build-ek; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@} ${!LLAMA_PREFILL_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
EXACT6="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXACT_ATTN_COLS=1 GGML_CUDA_ARGSORT_EXACT_ROWS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
base_args() { # base_args PARALLEL CTX
  echo "-m $M --alias glm-5.3-flash --jinja --ctx-size $2 --parallel $1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --metrics --no-warmup --fit off --load-mode mmap -lv 4 --host 127.0.0.1 --port 18080"
}
B="python3 -I ekbench.py"
PID=""; WD=""
start() { # start TAG "ENV" PARALLEL CTX
  local TAG=$1 EXTRA=$2
  pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; return 1; }
  cp snapshot/el-seed.k3sn snapshot/ek-$TAG.k3sn
  echo "=== LEG $TAG parallel=$3 ctx=$4 extra=[$EXTRA] $(date -u +%T)"
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/ek-$TAG.k3sn $EXTRA $BIN $(base_args $3 $4) > $OUT/server-$TAG.log 2>&1 &
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
  rm -f snapshot/ek-$TAG.k3sn
}
R=$OUT/results.jsonl
leg_P1() { # production shape with the 0052 binary: n_stream == 1 must be unchanged
  start P1 "" 1 1048576 || return
  python3 -I bench.py conc --url http://127.0.0.1:18080 --tag P1 --n 1 --rounds 3 --n-predict 256 | tee -a $R
  python3 -I bench.py depth --url http://127.0.0.1:18080 --tag P1-d16k --lens 16384 --n-predict 128 | tee -a $R
  stop P1
}
leg_P4() { # four slots of 1M each, prompt isolation; $1 extra env, $2 tag suffix
  local T=P4$2
  start $T "LLAMA_PREFILL_ISOLATE=1 $1" 4 4194304 || return
  $B solo p0 --tag $T-warm | tee -a $R
  $B solo p0 p1 p2 p3 L16384 --tag $T-solo | tee -a $R
  $B multi p0 p1 --tag $T-n2 --rounds 2 | tee -a $R
  $B multi p0 p1 p2 p3 --tag $T-n4 --rounds 2 | tee -a $R
  $B multi p0 L16384 --tag $T-n2mix --rounds 1 | tee -a $R
  $B multi p1 p2 p3 L16384 --tag $T-n4mix --rounds 1 | tee -a $R
  $B solo p0 --tag $T-after | tee -a $R
  grep -a "elastic KV" $OUT/server-$T.log | tail -12 | cut -c1-220
  stop $T
}
leg_P8() {
  start P8 "LLAMA_PREFILL_ISOLATE=1" 8 8388608 || return
  $B solo p0 --tag P8-warm | tee -a $R
  $B solo p0 p1 p2 p3 p4 p5 p6 p7 --tag P8-solo | tee -a $R
  $B multi p0 p1 p2 p3 p4 p5 p6 p7 --tag P8-n8 --rounds 2 | tee -a $R
  $B multi p0 p1 p2 p3 --tag P8-n4 --rounds 1 | tee -a $R
  stop P8
}
leg_PR() { # refuse policy: fixed share of 8192 cells per stream, then a memory floor that nothing can meet
  start PRs "LLAMA_PREFILL_ISOLATE=1 LLAMA_ELASTIC_KV_REFUSE=1 LLAMA_ELASTIC_KV_STREAM_MAX=8192" 4 4194304 || return
  $B multi p0 L16384 --tag PRs-mix --rounds 0 | tee -a $R
  $B solo p0 --tag PRs-after | tee -a $R
  stop PRs
  start PRf "LLAMA_PREFILL_ISOLATE=1 LLAMA_ELASTIC_KV_REFUSE=1 LLAMA_ELASTIC_KV_FLOOR_MIB=120000" 4 4194304 || return
  $B multi p0 L16384 --tag PRf-mix --rounds 0 | tee -a $R
  $B solo p1 --tag PRf-after | tee -a $R
  curl -fsS -m 2 127.0.0.1:18080/health; echo
  stop PRf
}
for L in "$@"; do
  case $L in
    P1) leg_P1 ;;
    P4) leg_P4 "" "" ;;
    P4x) leg_P4 "$EXACT6" "x" ;;
    P8) leg_P8 ;;
    PR) leg_PR ;;
    *) echo "unknown leg $L" ;;
  esac
done
echo "EK-WINDOW-DONE $(date -u +%T)"
