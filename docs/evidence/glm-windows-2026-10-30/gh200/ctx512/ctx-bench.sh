#!/bin/bash
# Prefill benchmark: one server per config, a ~20k-token prompt twice (cache_prompt false), then the standard request.
# Usage: prefill-bench.sh TAG [ENV=VAL ...]
set -u
tag=$1; shift
out=/srv/glm-5.3/prof/$tag; mkdir -p $out
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/snapshot/glm53-q2-full.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=0 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
for kv in "$@"; do export "$kv"; done
env | grep -E "^(LLAMA|GGML)_" | sort > $out/env.txt
${BIN:-$HOME/llama-det/build/bin/llama-server} -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18099 --ctx-size ${CTX:-131072} --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18099/health 2>/dev/null | grep -q ok && break; kill -0 $pid 2>/dev/null || { echo "BENCH $tag server died"; grep -a -E "error|failed|Invalid" $out/server.log | tail -3; exit 1; }; sleep 3; done
python3 - <<PY > $out/long.json
import json, random
random.seed(7)
words = "kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration affinity scheduler controller operator webhook certificate issuer acme dns".split()
text = " ".join(random.choice(words) for _ in range(int(${LONG_TOK:-24000}/1.29)))
print(json.dumps({"prompt": "Summarize this log:\n" + text, "temperature": 0, "n_predict": 64, "ignore_eos": True, "cache_prompt": False}))
PY
for r in 1; do
  curl -s -m 900 -H "Content-Type: application/json" -d @$out/long.json http://127.0.0.1:18099/completion > $out/long$r.out
  python3 -c "import json,sys;t=json.load(open(sys.argv[1]))[\"timings\"];print(\"BENCH $tag prefill\",t[\"prompt_n\"],\"tok\",round(t[\"prompt_per_second\"],1),\"tok/s decode-at-depth\",round(t[\"predicted_per_second\"],2))" $out/long$r.out
done
P="{\"prompt\":\"Write a Python quicksort function and explain it.\",\"temperature\":0,\"seed\":1,\"n_predict\":128,\"cache_prompt\":false}"
curl -s -m 600 -H "Content-Type: application/json" -d "$P" http://127.0.0.1:18099/completion > $out/std.json
python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));t=d[\"timings\"];print(\"BENCH $tag decode\",round(t[\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])" $out/std.json
kill $pid; wait $pid 2>/dev/null
