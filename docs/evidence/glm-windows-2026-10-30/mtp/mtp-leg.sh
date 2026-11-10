#!/bin/bash
# Spark legs for patch 0068 (exact speculative decoding with the MTP head). From nuc5-leg.sh. Binary $BD (default
# ~/nt/d121: stack through 0067 + 0068). glm53-spark must be at 0 replicas and the caller holds
# /srv/glm-5.3-flash/window.lock (spark-mtp1.sh).
# Usage: mtp-leg.sh TAG PARALLEL CTX "EXTRA_ENV" "client cmd" ["client cmd" ...]
# Env: XARGS = extra server arguments (the MTP flags); SEED = snapshot to start from (a copy is used; "none" = cold);
#      SAVE = keep the leg's snapshot under this name (an MTP server and a plain one do not share snapshots).
set -u
cd /srv/glm-5.3-flash; OUT=legs-mtp; mkdir -p $OUT
BD=${BD:-$HOME/nt/d121}; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@} ${!LLAMA_PREFILL_@} ${!LLAMA_BI_@} ${!LLAMA_SLOT_@} ${!LLAMA_CLIENT_@} ${!LLAMA_PRIORITY_@} ${!LLAMA_SPEC_@} ${!LLAMA_RS_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
base_args() { # base_args PARALLEL CTX
  echo "-m $M --alias glm-5.3-flash --jinja --ctx-size $2 --parallel $1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --metrics --no-warmup --fit off --load-mode mmap -lv 4 --host 127.0.0.1 --port 18080"
}
B="python3 -I"
PID=""; WD=""; PW=""
SEED=${SEED:-el-seed.k3sn}
start() { # start TAG "ENV" PARALLEL CTX
  local TAG=$1 EXTRA=$2
  [ -s $OUT/server.pid ] && kill -0 "$(cat $OUT/server.pid)" 2>/dev/null && { echo "ABORT $TAG: the previous test server is running"; return 1; }
  pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; return 1; }
  rm -f snapshot/mtp-$TAG.k3sn; [ "$SEED" = none ] || cp snapshot/$SEED snapshot/mtp-$TAG.k3sn
  echo "=== LEG $TAG parallel=$3 ctx=$4 seed=$SEED extra=[$EXTRA] xargs=[${XARGS:-}] $(date -u +%T)"
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/mtp-$TAG.k3sn $EXTRA $BIN $(base_args $3 $4) ${XARGS:-} > $OUT/server-$TAG.log 2>&1 &
  PID=$!; echo $PID > $OUT/server.pid
  ( low=999999; while kill -0 $PID 2>/dev/null; do a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo)
      [ "$a" -lt "$low" ] && low=$a && echo "$low" > $OUT/lowmem-$TAG.txt
      if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $PID; break; fi; sleep 1; done ) &
  WD=$!
  ( while kill -0 $PID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null)"; sleep 2; done > $OUT/power-$TAG.txt ) &
  PW=$!
  local i
  for i in $(seq 1 300); do
    curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s; MemAvailable $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) MiB"; grep -a -i -E "snapshot|n_slots|n_rs_seq|speculative|mtp" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | head -12 | cut -c1-260; return 0; }
    kill -0 $PID 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -8 | cut -c1-300; return 1; }
    sleep 2
  done
  echo "LEG-FAIL $TAG not ready"; return 1
}
stop() { # stop TAG
  local TAG=$1
  kill -INT $PID 2>/dev/null; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -KILL $PID 2>/dev/null; wait $PID 2>/dev/null
  kill $WD $PW 2>/dev/null; rm -f $OUT/server.pid
  grep -a "^SPEC_TRACE " $OUT/server-$TAG.log > $OUT/trace-$TAG.txt
  grep -a -i "draft acceptance\|hit rate" $OUT/server-$TAG.log | tail -4 | cut -c1-220
  grep -a -i -E "GGML_ASSERT|abort|refus" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | tail -4 | cut -c1-300
  echo "trace lines: $(wc -l < $OUT/trace-$TAG.txt); lowest MemAvailable $(cat $OUT/lowmem-$TAG.txt) MiB; now $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) MiB; END $TAG $(date -u +%T)"
  [ -n "${SAVE:-}" ] && [ -s snapshot/mtp-$TAG.k3sn ] && cp snapshot/mtp-$TAG.k3sn snapshot/$SAVE
  rm -f snapshot/mtp-$TAG.k3sn snapshot/mtp-$TAG.k3sn.tmp
}
R=$OUT/results.jsonl
TAG=$1; PAR=$2; CTX=$3; EXTRA=$4; shift 4
start $TAG "$EXTRA" $PAR $CTX || { [ -n "$PID" ] && stop $TAG; exit 1; }
for cmd in "$@"; do
  $B $cmd --tag $TAG | tee -a $R | cut -c1-420
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED during [$cmd]"; grep -a -i -E "GGML_ASSERT|abort|error" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | tail -5 | cut -c1-300; break; }
done
stop $TAG
echo "MTP-LEG-DONE $TAG $(date -u +%T)"
