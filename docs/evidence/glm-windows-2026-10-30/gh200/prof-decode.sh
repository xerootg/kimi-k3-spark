#!/bin/bash
# Nsight Systems profile of steady-state decode, production env, full residency (pod must be scaled to 0).
# Usage: prof-decode.sh TAG [ENV=VAL ...]
set -u
tag=$1; shift
out=/srv/glm-5.3/prof/$tag; mkdir -p $out
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/snapshot/glm53-q2-full.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=0 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
for kv in "$@"; do export "$kv"; done
env | grep -E "^(LLAMA|GGML)_" | sort > $out/env.txt
LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-sbsa-armv8${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} nsys profile --kill none -o $out/decode --force-overwrite true --trace=cuda,osrt --cuda-graph-trace=node --sample=none --cpuctxsw=none -y ${PROF_DELAY:-120} -d ${PROF_DUR:-8} \
  ~/llama-det/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18098 --ctx-size 131072 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18098/health 2>/dev/null | grep -q ok && break; sleep 2; done
echo "ready after $((i*2))s"
P="{\"prompt\":\"Write a Python quicksort function and explain it.\",\"temperature\":0,\"seed\":1,\"n_predict\":${NPRED:-1500},\"cache_prompt\":false}"
curl -s -m 900 -H "Content-Type: application/json" -d "$P" http://127.0.0.1:18098/completion > $out/r1.json
python3 -c "import json,sys;d=json.load(open(sys.argv[1]));t=d[\"timings\"];print(\"PROF $tag\",round(t[\"predicted_per_second\"],2),\"tok/s\")" $out/r1.json
pkill -INT -f "port 18098"; wait $pid 2>/dev/null
ls -la $out
echo "PROF $tag done"
