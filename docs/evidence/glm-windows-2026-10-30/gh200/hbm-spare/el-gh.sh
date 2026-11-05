#!/bin/bash
# Patch 0047 window (LLAMA_EXPERT_STORE_HBM_SPARE): production env/args (U=4096, HBM tier 8192), llama-el build
# (stack + 0045 + 0047). A = 0047 off, B = 0047 on. pf_client, then 3 x 256-token decodes, then cache stats.
cd /srv/glm-5.3; out=pfub/el; mkdir -p $out
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT a llama-server is running"; exit 1; }
BASE="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=8192 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0 LLAMA_EXPERT_STORE_MAX_BATCH=4096"
for leg in "A LLAMA_EXPERT_STORE_HBM_SPARE=0" "B LLAMA_EXPERT_STORE_HBM_SPARE=1"; do
  set -- $leg; tag=$1; shift
  cp snapshot/glm53-q2-full.k3sn $out/snap-$tag.k3sn
  (while sleep 2; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > $out/gpumem-$tag.txt & m=$!
  echo "=== $tag $* $(date +%T)"
  env $BASE "$@" LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/$out/snap-$tag.k3sn $HOME/llama-el/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18104 --ctx-size 524288 --parallel 1 --batch-size 4096 --ubatch-size 4096 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server-$tag.log 2>&1 &
  pid=$!; ok=0
  for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:18104/health | grep -q ok && { ok=1; break; }; kill -0 $pid 2>/dev/null || break; sleep 3; done
  grep -a -E "slots \(|snapshot restored|memory valve:.*released [0-9]+ slots" $out/server-$tag.log | head -6 | cut -c1-200
  if [ $ok = 1 ]; then
    python3 -I pf_client.py $tag || echo "CLIENT-FAIL $tag"
    for r in 1 2 3; do curl -s -m 600 http://127.0.0.1:18104/completion -H 'Content-Type: application/json' -d '{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":256,"cache_prompt":false,"ignore_eos":true}' | python3 -c "import json,sys,hashlib;d=json.load(sys.stdin);print('DEC $tag r$r',round(d['timings']['predicted_per_second'],2),'tok/s md5',hashlib.md5(d['content'].encode()).hexdigest()[:10])"; done
    grep -a -E "hit rate" $out/server-$tag.log | tail -1 | cut -c1-200
    grep -a -E "memory valve: host MemAvailable" $out/server-$tag.log | tail -1 | cut -c1-260
  else echo "SERVER-FAIL $tag"; grep -a -i -E "error|abort|fail|out of memory" $out/server-$tag.log | tail -3 | cut -c1-200; fi
  kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; kill $m
  rm -f $out/snap-$tag.k3sn
  echo "PEAK $tag $(sort -n $out/gpumem-$tag.txt | tail -1) MiB"
done
echo EL-GH-DONE
