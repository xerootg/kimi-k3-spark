#!/bin/bash
# Window flash3 (expert cache 58000 MiB after flash2 hit the watchdog at 70000): GLM-5.3-Flash UD-Q2_K_XL on the Spark through the expert store (SSD streamer), full 1M context.
# Experts live in a host-RAM cache (--moe-cache-mib) filled from store/glm53flash-q2.k3xs; the dense part and the KV
# are on the GPU. Watchdog kills the server below 4 GiB MemAvailable. Production glm53-spark must be at 0.
set -u
cd /srv/glm-5.3-flash
BIN=$HOME/llama-k3-fdq/build-fdq/bin/llama-server
export LD_LIBRARY_PATH="$HOME/llama-k3-fdq/build-fdq/bin:$HOME/llama-k3-fdq/build-fdq/lib"
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
OUT=legs-flash3; mkdir -p $OUT
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
(while sleep 2; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem.txt & MEMP=$!
trap 'kill $MEMP 2>/dev/null; [ -n "${pid:-}" ] && kill -KILL $pid 2>/dev/null' EXIT
STORE_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1"
start() { # tag ctx parallel cache_mib
  echo "=== LEG $1 ctx=$2 parallel=$3 cache_MiB=$4 $(date +%T)"
  env $STORE_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/flash-$1.k3sn "$BIN" -m $M --alias glm-5.3-flash --jinja -ngl 999 --n-cpu-moe 999 --moe-cache-mib $4 \
    --ctx-size $2 --parallel $3 --flash-attn on --batch-size 2048 --ubatch-size 2048 --cache-ram 0 \
    --threads 8 --threads-batch 20 --host 127.0.0.1 --port 18080 --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $OUT/server-$1.log 2>&1 & pid=$!
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 4096 ]; then echo "WATCHDOG kill $1 MemAvailable ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) &
  for i in $(seq 1 450); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $1 after $((i*2))s"; grep -a -h -E "KV buffer size|n_ctx_slot|host cache|expert store: .*MiB|model buffer" $OUT/server-$1.log | head -8 | cut -c1-170; return 0; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $1 server died"; grep -a -v "unused tensor" $OUT/server-$1.log | tail -4 | cut -c1-200; pid=; return 1; }; sleep 2; done
  echo "LEG-FAIL $1 health timeout"; return 1
}
stop() { [ -n "${pid:-}" ] || return 0; kill -INT $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; }
if start S1M 1048576 1 58000; then
  python3 -I bench.py conc --url http://127.0.0.1:18080 --tag S1M --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
  python3 -I bench.py depth --url http://127.0.0.1:18080 --tag S1M-depth --lens 2048,16384,65536 --n-predict 64 | tee -a $OUT/results.jsonl
  curl -s -m 900 http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What pods are running in namespace garage? Use the tool."}],"tools":[{"type":"function","function":{"name":"kubectl_get","description":"Read Kubernetes resources","parameters":{"type":"object","properties":{"kind":{"type":"string"},"namespace":{"type":"string"}},"required":["kind"]}}}],"max_tokens":800,"temperature":1.0,"top_p":0.95}' > $OUT/toolcall.json
  python3 -c "import json;d=json.load(open('$OUT/toolcall.json'));c=d['choices'][0];print('TOOLCALL finish',c.get('finish_reason'),'calls',json.dumps(c['message'].get('tool_calls'))[:200],'decode',round(d.get('timings',{}).get('predicted_per_second',0),2))"
  grep -a -E "hit rate" $OUT/server-S1M.log | tail -1 | cut -c1-180
  stop
fi
echo "MEMORY S1M lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"
if start S4 1048576 4 58000; then python3 -I bench.py conc --url http://127.0.0.1:18080 --tag S4 --n 4 --rounds 2 --n-predict 256 | tee -a $OUT/results.jsonl; stop; fi
cat $OUT/watchdog.txt 2>/dev/null
echo "MEMORY lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"
echo WIN2FLASH3-DONE
