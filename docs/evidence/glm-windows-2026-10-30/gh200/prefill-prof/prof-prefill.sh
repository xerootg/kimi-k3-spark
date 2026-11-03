#!/bin/bash
# Nsight profile of GLM-5.3 prefill on the GH200, production 512k env, one 200k-token prompt (cache off),
# two capture windows: shallow (~10-20k tokens in) and deep (~170k in). glm53-gh200 must be at 0.
cd /srv/glm-5.3; out=prof/prefill; mkdir -p $out
cp snapshot/glm53-q2-full.k3sn $out/snapcopy.k3sn
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-sbsa-armv8${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/$out/snapcopy.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=16384 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0
nsys launch --session-new=pf --trace=cuda,osrt --cuda-graph-trace=node \
  $HOME/llama-ship/build/bin/llama-server -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 --host 127.0.0.1 --port 18103 --ctx-size 524288 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > $out/server.log 2>&1 &
for i in $(seq 1 200); do curl -s -m 2 http://127.0.0.1:18103/health | grep -q ok && break; sleep 3; done; echo "ready after $((i*3)) s"
python3 - > $out/req.json <<PY
import json, random
r = random.Random(7); W = "kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration affinity scheduler".split()
print(json.dumps({"prompt": "Summarize this log:\n" + " ".join(r.choice(W) for _ in range(int(200000/1.29))), "temperature": 0, "n_predict": 8, "cache_prompt": False}))
PY
( curl -s -m 3600 -H "Content-Type: application/json" -d @$out/req.json http://127.0.0.1:18103/completion > $out/resp.json ) & c=$!
sleep 25; nsys start --session=pf -o $out/shallow --force-overwrite true; sleep 15; nsys stop --session=pf; echo "shallow captured $(date +%T)"
sleep 255; nsys start --session=pf -o $out/deep --force-overwrite true; sleep 15; nsys stop --session=pf; echo "deep captured $(date +%T)"
wait $c
python3 -c "import json;t=json.load(open(\"$out/resp.json\"))[\"timings\"];print(\"PREFILL\",t[\"prompt_n\"],\"tok\",round(t[\"prompt_per_second\"],1),\"tok/s\",round(t[\"prompt_ms\"]/1e3,1),\"s\")"
grep -a "prompt processing, n_tokens" $out/server.log | awk "NR%10==0" | cut -c1-150 | tail -12
nsys shutdown --session=pf --kill sigint 2>/dev/null; sleep 5; pkill -INT -f "port 18103"
ls -la $out
echo PREFILL-PROF-DONE
