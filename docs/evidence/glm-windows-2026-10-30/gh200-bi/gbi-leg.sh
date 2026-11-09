#!/bin/bash
# GH200 batch-invariant multi-slot window (gh200-bi), 2026-11-09. One leg = one test server in docker (the CI image that
# would ship), production env, mounts and args (argo-things configs/glm53-gh200/llama-server.yaml, image g7f117e1) except
# --parallel/--ctx-size and the leg's extra env. glm53-gh200 must be at 0 replicas.
# Usage: gbi-leg.sh TAG PARALLEL "EXTRA_ENV" "script.py args" ["script.py args" ...]   (scripts in this directory)
#   env IMG (default the 0054+0055 image), CACHE_RAM (default 98304), XARGS (extra server args)
set -u
G=/srv/glm-5.3-flash/gbi; cd $G; OUT=$G/legs; mkdir -p $OUT
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
IMG=${IMG:-ghcr.io/xerootg/llamacpp-spark:547208dca250-g64500e2}
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/store/glm53flash-q4.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=40960 LLAMA_EXPERT_STORE_VALVE_FLOOR=10 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_HBM_SPARE=1 LLAMA_EXPERT_STORE_MAX_BATCH=4096 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0"
TAG=$1; PAR=$2; EXTRA=$3; shift 3
CTX=$((PAR * 1048576)); NAME=gbi-$TAG; R=$OUT/results.jsonl
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running (is glm53-gh200 at 0?)"; exit 1; }
sudo docker ps -q --filter name=gbi- | grep -q . && { echo "ABORT $TAG: a gbi container is running"; exit 1; }
mkdir -p $G/snap; sudo cp /srv/glm-5.3-flash/snapshot/glm53flash-q4-full.k3sn $G/snap/snap-$TAG.k3sn
EOPTS=""; for kv in $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=/snapshot/snap-$TAG.k3sn $EXTRA; do EOPTS="$EOPTS -e $kv"; done
ARGS="-m /models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --mmproj /models/mmproj-BF16.gguf --alias glm-5.3-flash --host 127.0.0.1 --port 18104 --ctx-size $CTX --parallel $PAR --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 189360 --cache-ram ${CACHE_RAM:-98304} --temp 1.0 --top-p 0.95 --threads 16 --threads-batch 32 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 ${XARGS:-}"
echo "=== LEG $TAG parallel=$PAR ctx=$CTX img=${IMG##*:} extra=[$EXTRA] xargs=[${XARGS:-}] cache_ram=${CACHE_RAM:-98304} $(date -u +%T)"
(while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem-$TAG.txt & MON=$!
sudo docker run --rm --name $NAME --gpus all --network host --cap-add IPC_LOCK --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /srv/glm-5.3-flash/q4:/models:ro -v /srv/glm-5.3-flash/store:/store:ro -v $G/snap:/snapshot -v $G/keys:/keys:ro \
  $EOPTS $IMG $ARGS > $OUT/server-$TAG.log 2>&1 &
DPID=$!   # the sudo process (root): liveness via /proc, kill -0 from this user gets EPERM
( while [ -d /proc/$DPID ]; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 8192 ]; then echo "WATCHDOG kill $TAG ${a} MiB $(date -u +%T)" | tee -a $OUT/watchdog.txt; sudo docker kill -s KILL $NAME; break; fi; sleep 1; done ) & WD=$!
stop() {
  sudo docker kill -s INT $NAME >/dev/null 2>&1; local i
  for i in $(seq 1 90); do [ -d /proc/$DPID ] || break; sleep 1; done
  [ -d /proc/$DPID ] && { echo "stop: SIGKILL $NAME"; sudo docker kill -s KILL $NAME >/dev/null 2>&1; }
  wait $DPID 2>/dev/null; kill $WD $MON 2>/dev/null
  echo "PEAK GPU $(awk '{print $2}' $OUT/mem-$TAG.txt | sort -n | tail -1) MiB, LOWEST MemAvailable $(awk '{print $3}' $OUT/mem-$TAG.txt | sort -n | head -1) MiB; END $TAG $(date -u +%T)"
  sudo rm -f $G/snap/snap-$TAG.k3sn
}
ok=0
for i in $(seq 1 200); do
  curl -fsS -m 2 127.0.0.1:18104/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*3))s"; ok=1; break; }
  [ -d /proc/$DPID ] || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -8 | cut -c1-300; break; }
  sleep 3
done
[ $ok = 1 ] || { echo "LEG-FAIL $TAG not ready"; stop; exit 1; }
grep -a -E "LLAMA_KV_HOST|host_kv_buffer|KV buffer size|elastic KV|BATCH_INVARIANT|batch.invariant|client classes|CLIENT_CLASSES|n_slots|SLOT_SERIAL|PREFILL_ISOLATE" $OUT/server-$TAG.log | head -14 | cut -c1-260
echo "GPU after load: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for cmd in "$@"; do
  echo "--- $cmd $(date -u +%T)"
  python3 -I $G/$cmd --url http://127.0.0.1:18104 --tag $TAG | tee -a $R
  [ -d /proc/$DPID ] || { echo "SERVER DIED during [$cmd]"; break; }
done
stop
echo "GBI-LEG-DONE $TAG $(date -u +%T)"
