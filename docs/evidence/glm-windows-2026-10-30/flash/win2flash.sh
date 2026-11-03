#!/bin/bash
# Window flash: GLM-5.3-Flash UD-Q2_K_XL fully resident on the Spark (no expert store), production pod down.
# Legs: F1 one slot at 524288, then F1M at the full 1048576; weights read in place (--load-mode mmap); a watchdog kills the server below 4 GiB MemAvailable (standard x2, depth 2k/16k/64k, tool call); F4 four slots (ctx 524288 total);
# F8 eight slots (ctx 524288 total). MemAvailable sampled every 2 s.
set -u
cd /srv/glm-5.3-flash
BIN=$HOME/llama-k3-fdq/build-fdq/bin/llama-server
export LD_LIBRARY_PATH="$HOME/llama-k3-fdq/build-fdq/bin:$HOME/llama-k3-fdq/build-fdq/lib"
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
OUT=legs-flash; mkdir -p $OUT
pgrep -f "llama-serve[r] -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
(while sleep 2; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem.txt & MEMP=$!
trap 'kill $MEMP 2>/dev/null; [ -n "${pid:-}" ] && kill -KILL $pid 2>/dev/null' EXIT
start() { # tag ctx parallel
  echo "=== LEG $1 ctx=$2 parallel=$3 $(date +%T)"
  "$BIN" -m $M --alias glm-5.3-flash --jinja -ngl 999 --ctx-size $2 --parallel $3 --flash-attn on --batch-size 2048 --ubatch-size 2048 \
    --threads 8 --threads-batch 12 --host 127.0.0.1 --port 18080 --metrics --no-warmup --fit off --load-mode mmap > $OUT/server-$1.log 2>&1 & pid=$!
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 4096 ]; then echo "WATCHDOG kill $1 MemAvailable ${a} MiB" >> $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) &
  for i in $(seq 1 300); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $1 after $((i*2))s"; return 0; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $1 server died"; tail -5 $OUT/server-$1.log | cut -c1-200; pid=; return 1; }; sleep 2; done
  echo "LEG-FAIL $1 health timeout"; return 1
}
stop() { [ -n "${pid:-}" ] || return 0; kill -INT $pid; for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; }
F1CTX=524288
if ! start F1 $F1CTX 1; then stop; F1CTX=0; fi
if [ "$F1CTX" != 0 ]; then
  echo "F1 running with ctx $F1CTX"
  python3 -I bench.py conc --url http://127.0.0.1:18080 --tag F1 --n 1 --rounds 2 --n-predict 256 | tee -a $OUT/results.jsonl
  python3 -I bench.py depth --url http://127.0.0.1:18080 --tag F1-depth --lens 2048,16384,65536 --n-predict 64 | tee -a $OUT/results.jsonl
  curl -s -m 600 http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What pods are running in namespace garage? Use the tool."}],"tools":[{"type":"function","function":{"name":"kubectl_get","description":"Read Kubernetes resources","parameters":{"type":"object","properties":{"kind":{"type":"string"},"namespace":{"type":"string"}},"required":["kind"]}}}],"max_tokens":800,"temperature":1.0,"top_p":0.95}' > $OUT/toolcall.json
  python3 -c "import json;d=json.load(open('$OUT/toolcall.json'));c=d['choices'][0];print('TOOLCALL finish',c.get('finish_reason'),'calls',json.dumps(c['message'].get('tool_calls'))[:200],'decode',round(d.get('timings',{}).get('predicted_per_second',0),2))"
  grep -a -h -E "CUDA0 model buffer|CPU_Mapped model buffer|KV buffer size|compute buffer size|n_ctx_slot" $OUT/server-F1*.log | head -6 | cut -c1-160
  stop
fi
if [ "$F1CTX" != 0 ] && start F1M 1048576 1; then grep -a -h -E "KV buffer size|n_ctx_slot" $OUT/server-F1M.log | head -3 | cut -c1-160; python3 -I bench.py conc --url http://127.0.0.1:18080 --tag F1M --n 1 --rounds 1 --n-predict 128 | tee -a $OUT/results.jsonl; stop; else stop; echo "F1M 1M context did not load"; fi
cat $OUT/watchdog.txt 2>/dev/null
for n in 4 8; do
  if start F$n 524288 $n; then python3 -I bench.py conc --url http://127.0.0.1:18080 --tag F$n --n $n --rounds 2 --n-predict 256 | tee -a $OUT/results.jsonl; stop; fi
done
echo "MEMORY lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"
echo WIN2FLASH-DONE
