#!/bin/bash
# Container canary for patch 0079 (kimi-k3-spark PR #5): the CI image with the production env, args and mounts of
# argo-things glm53-spark (parallel 5, ctx 1024000), port 18100, memory watchdog. glm53-spark must be at 0 replicas.
# Usage: canary79.sh IMAGE_TAG
set -u
IMG=ghcr.io/xerootg/llamacpp-spark:$1
OUT=/srv/glm-5.3-flash/legs-canary79; mkdir -p $OUT
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
sudo -n docker pull -q $IMG > $OUT/pull.txt 2>&1 || { echo "PULL FAILED"; cat $OUT/pull.txt; exit 1; }
E=""
for kv in LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/snapshot/canary79.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_ELASTIC_KV_REFUSE=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 \
  LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 \
  LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 \
  LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 \
  GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 \
  LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_SLOT_RESERVE=1 LLAMA_PRIORITY_HOLD_PROMPTS=1; do E="$E -e $kv"; done
cp /srv/glm-5.3-flash/snapshot/el-seed.k3sn /srv/glm-5.3-flash/snapshot/canary79.k3sn
sudo -n docker rm -f glm-canary >/dev/null 2>&1
sudo -n docker run -d --name glm-canary --gpus all --cap-add IPC_LOCK --ulimit memlock=-1 -p 127.0.0.1:18100:8080 $E \
  -v /srv/glm-5.3-flash/q2:/models:ro -v /srv/glm-5.3-flash/store:/store:ro -v /srv/glm-5.3-flash/snapshot:/snapshot \
  $IMG -m /models/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf --alias glm-5.3-flash --host 0.0.0.0 --port 8080 \
  --ctx-size 1024000 --parallel 5 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 \
  --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $OUT/run.txt 2>&1
( low=999999; while sudo -n docker inspect -f "{{.State.Running}}" glm-canary 2>/dev/null | grep -q true; do
    a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo); [ $a -lt $low ] && low=$a && echo $low > $OUT/lowmem.txt
    if [ $a -lt 3072 ]; then echo "WATCHDOG KILL at $a MiB" >> $OUT/lowmem.txt; sudo -n docker kill glm-canary; fi; sleep 1; done ) &
ready=0
for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:18100/health > /dev/null && { ready=1; break; }; sleep 2; done
rc=0
if [ $ready = 1 ]; then
  echo "health after ${i}x2s" > $OUT/result.txt
  sudo -n docker logs glm-canary 2>&1 | grep -a -c "CUDA0" | sed "s/^/CUDA0 lines: /" >> $OUT/result.txt
  python3 /srv/glm-5.3-flash/bench.py conc --url http://127.0.0.1:18100 --tag C79 --n 1 --rounds 3 --n-predict 256 >> $OUT/result.txt 2>&1 || rc=1
  python3 /srv/glm-5.3-flash/bench.py depth --url http://127.0.0.1:18100 --tag C79-d16k --lens 16384 --n-predict 128 >> $OUT/result.txt 2>&1 || rc=1
else
  echo "NOT HEALTHY after 480 s" > $OUT/result.txt; rc=1
fi
sudo -n docker logs glm-canary > $OUT/server.log 2>&1
sudo -n docker rm -f glm-canary > /dev/null
rm -f /srv/glm-5.3-flash/snapshot/canary79.k3sn /srv/glm-5.3-flash/snapshot/canary79.k3sn.tmp
echo "lowest MemAvailable $(cat $OUT/lowmem.txt 2>/dev/null) MiB; $([ $rc = 0 ] && echo CANARY-DONE || echo CANARY-FAILED)" >> $OUT/result.txt
cat $OUT/result.txt
exit $rc
