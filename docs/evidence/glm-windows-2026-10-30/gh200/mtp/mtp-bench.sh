#!/bin/bash
# MTP speculative decoding on the GH200, production 512k settings (ship build). glm53-gh200 must be at 0.
cd /srv/glm-5.3; out=prof/mtp; mkdir -p $out
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/snapshot/glm53-q2-full.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=16384 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
for cfg in "none" "mtp 1" "mtp 2" "mtp 3"; do
  set -- $cfg; tag=$1${2:+-n$2}
  spec=""; [ "$1" = mtp ] && spec="--spec-type draft-mtp --spec-draft-n-max $2"
  echo "=== $tag $(date +%T)"
  (while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > $out/gpumem-$tag.txt & m=$!
  $HOME/llama-ship/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18102 --ctx-size 524288 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap --temp 1.0 --top-p 0.95 $spec > $out/server-$tag.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:18102/health | grep -q ok && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 3; done
  if [ $ok = 1 ]; then python3 -I mtp_client.py $tag || echo "CLIENT-FAIL $tag"; else echo "SERVER-FAIL $tag"; grep -a -i -E "error|fail|abort" $out/server-$tag.log | tail -4 | cut -c1-200; fi
  kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null; kill $m
  echo "PEAK $tag $(sort -n $out/gpumem-$tag.txt | tail -1) MiB"
  grep -a -E "draft acceptance|acc per pos" $out/server-$tag.log | tail -6 | cut -c1-160
done
echo MTP-DONE
