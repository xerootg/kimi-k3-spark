#!/bin/bash
# GH200 GLM-5.3 throughput workup. The glm53-gh200 pod must be scaled to 0 (needs the host memory).
# Binary: ~/llama-ship (production patches 0001-0038 + 0042 + 0043, sm_90) = image gea5e991's stack.
# Env and args match the production manifest (argo-things 71e4639) except --parallel and --cache-ram 0.
# Output: results.jsonl (one JSON line per measurement), power.csv (unix_ts, gpu W, module W, util %), server logs.
set -u
cd "$(dirname "$0")"
OUT=${OUT:-run-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"
BIN=$HOME/llama-ship/build/bin/llama-server
PORT=18100
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=/srv/glm-5.3/store/glm53-q2-v1.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=/srv/glm-5.3/snapshot/glm53-q2-full.k3sn LLAMA_EXPERT_STORE_THREADS=32 LLAMA_EXPERT_STORE_URING=0 LLAMA_EXPERT_STORE_GPU_SLOTS=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_HBM_MIB=49152 LLAMA_EXPERT_STORE_HBM_DECAY=512 LLAMA_EXPERT_STORE_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_HBM_RESERVE_MIB=8192 LLAMA_EXPERT_STORE_MAX_BATCH=2048 LLAMA_EXPERT_STORE_PREFILL_BANK=0 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SELF_PREFETCH=0 LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=0 LLAMA_EXPERT_STORE_THP_BANK=0"
EXACT_ENV="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXACT_ATTN_COLS=1 GGML_CUDA_ARGSORT_EXACT_ROWS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_PREFILL_ISOLATE=1"

# power sampler: unix time + GPU and module (Grace CPU + Hopper GPU + memory) instantaneous watts
(nvidia-smi --query-gpu=power.draw.instant,module.power.draw.instant,utilization.gpu --format=csv,noheader,nounits -lms 500 |
  while IFS= read -r l; do echo "$(date +%s.%N),$l"; done) > "$OUT/power.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; pkill -f "llama-server.*--port $PORT" 2>/dev/null' EXIT

start_server() {  # $1 tag, $2 parallel, $3 extra env
  local tag=$1 par=$2 extra=$3
  env $PROD_ENV $extra "$BIN" -m /srv/glm-5.3/q2/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf --alias glm-5.3 \
    --host 127.0.0.1 --port $PORT --ctx-size 131072 --parallel "$par" --batch-size 2048 --ubatch-size 2048 \
    --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 260000 --cache-ram 0 --threads 16 --threads-batch 32 \
    --flash-attn on --jinja --no-warmup --fit off --load-mode mmap > "$OUT/server-$tag.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 200); do
    curl -s -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ok && { echo "SERVER $tag ready ${i}x3s"; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "SERVER $tag died"; tail -3 "$OUT/server-$tag.log"; return 1; }
    sleep 3
  done
  echo "SERVER $tag timeout"; return 1
}
stop_server() { kill -INT $SPID 2>/dev/null; wait $SPID 2>/dev/null; }

leg() {  # $1 tag, $2 parallel, $3 extra env, rest: bench.py args
  local tag=$1 par=$2 extra=$3; shift 3
  echo "=== LEG $tag parallel=$par $(date +%T)"
  if start_server "$tag" "$par" "$extra"; then
    python3 -I bench.py "$@" --tag "$tag" --url http://127.0.0.1:$PORT | tee -a "$OUT/results.jsonl" || echo "LEG-FAIL $tag"
  else
    echo "LEG-FAIL $tag"
  fi
  stop_server
}

leg p1-depth 1 "" depth --lens 0,1024,4096,16384,32768,65536,120000 --n-predict 256
leg p1 1 "" conc --n 1 --rounds 3 --n-predict 256
for n in 2 4 8 16; do leg plain-p$n $n "" conc --n $n --rounds 2 --n-predict 256; done
leg exact-p1 1 "$EXACT_ENV" conc --n 1 --rounds 2 --n-predict 256
for n in 2 4 8; do leg exact-p$n $n "$EXACT_ENV" conc --n $n --rounds 2 --n-predict 256; done
echo "BENCH-DONE $OUT"
