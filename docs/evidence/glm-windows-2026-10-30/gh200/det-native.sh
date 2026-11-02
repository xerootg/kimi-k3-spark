#!/bin/bash
# Native determinism probe with the debug tree: one process, N short requests, tensor digests.
# Usage: det-native.sh TAG N NPRED [ENV=VAL ...]
set -u
tag=$1; n=$2; np=$3; shift 3
out=/srv/glm-5.3/det/$tag; mkdir -p $out
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=0 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=0 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
for kv in "$@"; do export "$kv"; done
env | grep -E "^(LLAMA|GGML)_" | sort > $out/env.txt
~/llama-det/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18097 --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib ${CACHE_MIB:-100000} --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server.log 2>&1 &
pid=$!
for i in $(seq 1 240); do curl -s -m 2 http://127.0.0.1:18097/health 2>/dev/null | grep -q ok && break; kill -0 $pid 2>/dev/null || { echo "DET $tag server died"; tail -5 $out/server.log; exit 1; }; sleep 5; done
P="{\"prompt\":\"Write a Python quicksort function and explain it.\",\"temperature\":0,\"seed\":1,\"n_predict\":$np,\"cache_prompt\":false}"
for i in $(seq 1 $n); do
  curl -s -m 1800 -H "Content-Type: application/json" -d "$P" http://127.0.0.1:18097/completion > $out/r$i.json
  python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));t=d[\"timings\"];print(\"DET $tag r$i\",round(t[\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])" $out/r$i.json
done
kill $pid; wait $pid 2>/dev/null
echo "DET $tag done"
