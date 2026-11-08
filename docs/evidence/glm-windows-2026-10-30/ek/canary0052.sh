#!/bin/bash
# Container canary for the multi-slot candidate (patch 0052): production image/env/mounts of argo-things
# configs/kimi-k3/glm53-spark.yaml (#150) with the 0052 image and four resident slots of 1M each, served one request
# at a time (LLAMA_SLOT_SERIAL=1, bytes equal to the single-slot gold), refuse-on-full memory policy. Port 18100,
# memory watchdog. glm53-spark must be at 0 replicas.
set -u
IMG=${IMG:-ghcr.io/xerootg/llamacpp-spark:547208dca250-g1bbc8bb}
OUT=/srv/glm-5.3-flash/legs-canary0052; mkdir -p $OUT
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
sudo -n docker pull -q $IMG > $OUT/pull.txt 2>&1
E=""
for kv in LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/snapshot/glm53flash-q2.k3sn \
  LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 \
  LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 \
  LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 \
  LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 \
  GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1 \
  LLAMA_SLOT_SERIAL=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_ELASTIC_KV_REFUSE=1; do E="$E -e $kv"; done
sudo -n docker rm -f glm-canary >/dev/null 2>&1
sudo -n docker run -d --name glm-canary --gpus all --cap-add IPC_LOCK --ulimit memlock=-1 -p 18100:8080 $E \
  -v /srv/glm-5.3-flash/q2:/models:ro -v /srv/glm-5.3-flash/store:/store:ro -v /srv/glm-5.3-flash/snapshot:/snapshot \
  $IMG -m /models/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf --alias glm-5.3-flash --host 0.0.0.0 --port 8080 \
  --ctx-size 4194304 --parallel 4 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 \
  --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $OUT/run.txt 2>&1
( low=999999; while sudo -n docker inspect -f '{{.State.Running}}' glm-canary 2>/dev/null | grep -q true; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); [ $a -lt $low ] && low=$a && echo $low > $OUT/lowmem.txt
    if [ $a -lt 3072 ]; then echo "WATCHDOG KILL at $a MiB" >> $OUT/lowmem.txt; sudo -n docker kill glm-canary; fi; sleep 1; done ) &
for i in $(seq 1 900); do curl -sf -m 2 http://127.0.0.1:18100/health > /dev/null && break; sleep 2; done
echo "health after ${i}x2s" > $OUT/result.txt
U=http://127.0.0.1:18100
python3 -I bench.py conc --url $U --tag C0052 --n 1 --rounds 3 >> $OUT/result.txt 2>&1
python3 -I bench.py depth --url $U --tag C0052-d16k --lens 16384 --n-predict 128 >> $OUT/result.txt 2>&1
python3 -I ekbench.py multi p0 p1 p2 p3 --url $U --tag C0052-serial4 --rounds 1 >> $OUT/result.txt 2>&1
python3 -I ekbench.py solo L16384 p1 L16384 --cache --url $U --tag C0052-resident >> $OUT/result.txt 2>&1
sudo -n docker logs glm-canary > $OUT/server.log 2>&1
sudo -n docker rm -f glm-canary > /dev/null
echo CANARY-DONE >> $OUT/result.txt
