#!/bin/bash
# GH200 legs for patch 0070 (exact speculative decoding with the MTP head; 0068 was its working number). From nc/nc-leg.sh. One leg = one test server
# in docker: the production image with a dev build mounted over /opt/llama (OPT, default ~/nt/out-c90: stack through
# 0069 + 0070), production env and args (argo-things configs/glm53-gh200/llama-server.yaml at g68d4166) without the
# client classes, the slot reserve, the hold rule and the API key. glm53-gh200 must be at 0 replicas and the caller
# holds /srv/glm-5.3-flash/window.lock (gh200-mtp1.sh).
# Usage: gmtp-leg.sh TAG PARALLEL "EXTRA_ENV" "client cmd" ["client cmd" ...]   (clients in this directory)
# Env: XARGS = extra server arguments (the MTP flags); SEED = snapshot to start from: "prod" (a copy of production's,
#      the default), "none" (cold), or a name under snap/ (an MTP server and a plain one do not share snapshots);
#      SAVE = keep the leg's snapshot under this name in snap/; CMDTIMEOUT = seconds allowed per client command (legs
#      that are expected to hang or die); STOPWAIT = seconds to wait for the server to exit after SIGINT (default 90).
set -u
G=/srv/glm-5.3-flash/mtp; cd $G; OUT=$G/legs; mkdir -p $OUT $G/snap
IMG=${IMG:-ghcr.io/xerootg/llamacpp-spark:547208dca250-g68d4166}
OPT=${OPT:-$HOME/nt/out-c90}
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/store/glm53flash-q4.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=40960 LLAMA_EXPERT_STORE_VALVE_FLOOR=10 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_HBM_SPARE=1 LLAMA_EXPERT_STORE_MAX_BATCH=4096 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
TAG=$1; PAR=$2; EXTRA=$3; shift 3
CTX=$((PAR * 1048576)); NAME=gmtp-$TAG; R=$OUT/results.jsonl; SEED=${SEED:-prod}
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running (is glm53-gh200 at 0?)"; exit 1; }
sudo docker ps -q --filter name=gmtp- | grep -q . && { echo "ABORT $TAG: a gmtp container is running"; exit 1; }
sudo rm -f $G/snap/leg-$TAG.k3sn
case $SEED in
  none) ;;
  prod) sudo cp /srv/glm-5.3-flash/snapshot/glm53flash-q4-full.k3sn $G/snap/leg-$TAG.k3sn ;;
  *)    sudo cp $G/snap/$SEED $G/snap/leg-$TAG.k3sn ;;
esac
EOPTS=""; for kv in $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=/snapshot/leg-$TAG.k3sn $EXTRA; do EOPTS="$EOPTS -e $kv"; done
ARGS="-m /models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --mmproj /models/mmproj-BF16.gguf --alias glm-5.3-flash --host 127.0.0.1 --port 18104 --ctx-size $CTX --parallel $PAR --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 189360 --cache-ram 0 --temp 1.0 --top-p 0.95 --threads 16 --threads-batch 32 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 ${XARGS:-}"
echo "=== LEG $TAG parallel=$PAR ctx=$CTX img=${IMG##*:} opt=$OPT seed=$SEED extra=[$EXTRA] xargs=[${XARGS:-}] $(date -u +%T)"
(while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits | tr -d ,) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem-$TAG.txt & MON=$!
sudo docker run --rm --name $NAME --gpus all --network host --cap-add IPC_LOCK --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /srv/glm-5.3-flash/q4:/models:ro -v /srv/glm-5.3-flash/store:/store:ro -v $G/snap:/snapshot -v $OPT:/opt/llama:ro \
  $EOPTS $IMG $ARGS > $OUT/server-$TAG.log 2>&1 &
DPID=$!   # the sudo process (root): liveness via /proc, kill -0 from this user gets EPERM
( while [ -d /proc/$DPID ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 8192 ]; then echo "WATCHDOG kill $TAG ${a} MiB $(date -u +%T)" | tee -a $OUT/watchdog.txt; sudo docker kill -s KILL $NAME; break; fi; sleep 1; done ) & WD=$!
stop() {
  sudo docker kill -s INT $NAME >/dev/null 2>&1; local i
  for i in $(seq 1 ${STOPWAIT:-90}); do [ -d /proc/$DPID ] || break; sleep 1; done
  [ -d /proc/$DPID ] && { echo "stop: SIGKILL $NAME"; sudo docker kill -s KILL $NAME >/dev/null 2>&1; }
  wait $DPID 2>/dev/null; kill $WD $MON 2>/dev/null
  grep -a "^SPEC_TRACE " $OUT/server-$TAG.log > $OUT/trace-$TAG.txt
  grep -a -i "draft acceptance\|hit rate" $OUT/server-$TAG.log | tail -4 | cut -c1-220
  grep -a -i -E "GGML_ASSERT|abort|refus" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | tail -4 | cut -c1-300
  echo "trace lines: $(wc -l < $OUT/trace-$TAG.txt); PEAK GPU $(awk '{print $2}' $OUT/mem-$TAG.txt | sort -n | tail -1) MiB, LOWEST MemAvailable $(awk '{print $4}' $OUT/mem-$TAG.txt | sort -n | head -1) MiB; END $TAG $(date -u +%T)"
  [ -n "${SAVE:-}" ] && [ -s $G/snap/leg-$TAG.k3sn ] && sudo cp $G/snap/leg-$TAG.k3sn $G/snap/$SAVE
  sudo rm -f $G/snap/leg-$TAG.k3sn $G/snap/leg-$TAG.k3sn.tmp
}
ok=0
for i in $(seq 1 200); do
  curl -fsS -m 2 127.0.0.1:18104/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*3))s"; ok=1; break; }
  [ -d /proc/$DPID ] || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -8 | cut -c1-300; break; }
  sleep 3
done
[ $ok = 1 ] || { echo "LEG-FAIL $TAG not ready"; stop; exit 1; }
grep -a -i -E "snapshot|n_slots|n_rs_seq|speculative|mtp|BATCH_INVARIANT|PREFILL_ISOLATE|SPEC_EXACT" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | head -14 | cut -c1-260
grep -a -m1 -i "device_info\|CUDA0" $OUT/server-$TAG.log | cut -c1-200
echo "GPU after load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for cmd in "$@"; do
  ${CMDTIMEOUT:+timeout $CMDTIMEOUT} python3 -I $G/$cmd --url http://127.0.0.1:18104 --tag $TAG | tee -a $R | cut -c1-420
  [ -d /proc/$DPID ] || { echo "SERVER DIED during [$cmd]"; grep -a -i -E "GGML_ASSERT|abort|error" $OUT/server-$TAG.log | grep -a -v SPEC_TRACE | tail -5 | cut -c1-300; break; }
done
stop
echo "GMTP-LEG-DONE $TAG $(date -u +%T)"
