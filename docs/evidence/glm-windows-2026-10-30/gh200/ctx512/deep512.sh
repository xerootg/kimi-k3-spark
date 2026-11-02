#!/bin/bash
cd /srv/glm-5.3; out=prof/deep512; mkdir -p $out
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/snapshot/glm53-q2-full.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=16384 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
(while sleep 2; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo)"; done) > $out/mem.txt & m=$!
$HOME/llama-ship/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18101 --ctx-size 524288 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 98304 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap --temp 1.0 --top-p 0.95 > $out/server.log 2>&1 &
pid=$!
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:18101/health | grep -q ok && break; kill -0 $pid 2>/dev/null || { echo "DEEP server died"; tail -3 $out/server.log; kill $m; exit 1; }; sleep 3; done
python3 -I deep512.py 450000 | tee $out/results.jsonl || echo "DEEP client failed"
kill -0 $pid 2>/dev/null || { echo "DEEP server died during run"; grep -a -i -E "out of memory|alloc|error" $out/server.log | tail -5; }
kill -INT $pid; wait $pid 2>/dev/null; kill $m
echo "PEAK-HBM-MiB $(awk "{print \$2}" $out/mem.txt | sort -n | tail -1) of 97871; MIN-HOST-MemAvailable-GiB $(awk "{print \$3}" $out/mem.txt | sort -n | head -1)"
grep -a -E "prompt cache|cache state|saved|restored" $out/server.log | tail -6 | cut -c1-180
echo DEEP-DONE
