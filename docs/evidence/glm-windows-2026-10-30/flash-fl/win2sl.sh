#!/bin/bash
# Window fl (Spark, GLM-5.3-Flash, production env incl. elastic KV, 78000 MiB cache, 1M ctx): where does a decode token go,
# and what does MTP speculative decoding give? P = nsys trace of one 64-token decode; M0 = baseline (md5 53816a69bc);
# M1 = MTP (draft-mtp, n-max 3, p-min 0.7), its own snapshot (an MTP server rejects the non-MTP one), warmed first;
# M2 = MTP plus the six exact-columns knobs (batched verify columns bit-identical to single-token decode).
# Production glm53-spark must be at 0. Watchdog kills the server below 3 GiB MemAvailable.
set -u
cd /srv/glm-5.3-flash; OUT=legs-sl; mkdir -p $OUT nsys
BIN=$HOME/llama-k3-el/build-el/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-el/build-el/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
[ -s snapshot/fl-seed.k3sn ] || { echo "ABORT-WINDOW no seed snapshot"; exit 1; }
(while sleep 2; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem.txt & MEMP=$!
trap 'kill $MEMP 2>/dev/null; [ -n "${pid:-}" ] && kill -KILL $pid 2>/dev/null' EXIT
ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 LLAMA_ELASTIC_KV=1"
EXACT="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXACT_ATTN_COLS=1 GGML_CUDA_ARGSORT_EXACT_ROWS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_PREFILL_ISOLATE=1"
ARGS="-m $M --alias glm-5.3-flash --jinja -ngl 999 --n-cpu-moe 999 --moe-cache-mib 78000 --ctx-size 1048576 --parallel 1 --flash-attn on --batch-size 2048 --ubatch-size 2048 --cache-ram 0 --threads 8 --threads-batch 20 --host 127.0.0.1 --port 18080 --metrics --no-warmup --fit off --load-mode mmap -lv 4"
MTP="--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.7"
P64='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":64,"cache_prompt":false}'
start() { # tag snapshot nsys(0/1) extra-env... -- extra-args...
  local tag=$1 sn=$2 ns=$3; shift 3; local envx=() argx=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do envx+=("$1"); shift; done; [ "${1:-}" = -- ] && shift; argx=("$@")
  echo "=== LEG $tag $(date +%T) env: ${envx[*]:-} args: ${argx[*]:-}"
  if [ $ns = 1 ]; then nsys sessions list 2>/dev/null | grep -q fl$tag && nsys stop --session=fl$tag >/dev/null 2>&1
    env $ENV "${envx[@]}" LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/$sn nsys launch --session-new=fl$tag --trace=cuda,nvtx,osrt --cuda-graph-trace=node $BIN $ARGS "${argx[@]}" > $OUT/server-$tag.log 2>&1 & pid=$!
  else env $ENV "${envx[@]}" LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/$sn $BIN $ARGS "${argx[@]}" > $OUT/server-$tag.log 2>&1 & pid=$!; fi
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $tag ${a} MiB" | tee -a $OUT/watchdog.txt; pkill -KILL -P $pid; kill -KILL $pid; break; fi; sleep 1; done ) &
  for i in $(seq 1 300); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $tag after $((i*2))s"; return 0; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $tag died"; grep -a -i -E "error|abort|assert" $OUT/server-$tag.log | tail -3 | cut -c1-200; pid=; return 1; }; sleep 2; done; echo "LEG-FAIL $tag timeout"; return 1
}
stop() { [ -n "${pid:-}" ] || return 0; kill -INT $pid; pkill -INT -P $pid 2>/dev/null; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; sleep 3; }
bench() { python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $1 --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
          python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $1-d16k --lens 16384 --n-predict 128 | tee -a $OUT/results.jsonl; }
for L in 8 1 2; do
  cp snapshot/fl-seed.k3sn snapshot/sl-$L.k3sn
  if start SL$L sl-$L.k3sn 0 LLAMA_EXPERT_STORE_SPLIT_LATE=$L; then bench SL$L; grep -a -E "split GEMV" $OUT/server-SL$L.log | tail -1 | cut -c1-330; stop; fi
done
cat $OUT/watchdog.txt 2>/dev/null
echo "MEMORY lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"
echo WIN2SL-DONE
