#!/bin/bash
# GDN state-fusion fault reproduction (window ek, 2026-11-08): a two-slot server, p0 (256 tokens) next to the 16k
# request, then p0 alone (the first one-sequence decode after the other slot finished). Debug build ~/kf/src with
# GGML_CUDA_FUSE_GDN_STATE_MULTI=1 (guard lifted) and GGML_CUDA_GDN_STATE_CHECK=1 (ids and pointers logged).
# Usage: gdn-leg.sh TAG "EXTRA_ENV"
set -u
cd /srv/glm-5.3-flash; OUT=legs-gdn; mkdir -p $OUT
BD=${BD:-$HOME/kf/src/build}; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@} ${!LLAMA_PREFILL_@} ${!LLAMA_BI_@} ${!LLAMA_SLOT_@} ${!LLAMA_CLIENT_@} ${!LLAMA_PRIORITY_@} ${!LLAMA_KPOOL_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_ELASTIC_KV_REFUSE=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_KPOOL_OWN_SEQS=1"
ARGS="-m $M --alias glm-5.3-flash --host 127.0.0.1 --port 18080 --ctx-size 1024000 --parallel 5 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4"
TAG=$1; EXTRA=${2:-}; PAR=${3:-5}; CTX=${4:-1024000}
ARGS=${ARGS/--ctx-size 1024000 --parallel 5/--ctx-size $CTX --parallel $PAR}
if [ "${NO_ELASTIC:-0}" = 1 ]; then PROD_ENV=${PROD_ENV/LLAMA_ELASTIC_KV=1 LLAMA_ELASTIC_KV_REFUSE=1 /}; fi
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; exit 1; }
cp snapshot/el-seed.k3sn snapshot/gdn-$TAG.k3sn
echo "=== LEG $TAG extra=[$EXTRA] $(date -u +%T)"
env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/gdn-$TAG.k3sn $EXTRA $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & pid=$!
( while kill -0 $pid 2>/dev/null; do a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo); if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; pkill -KILL -P $pid; kill -KILL $pid; break; fi; sleep 1; done ) & WD=$!
ok=0; for i in $(seq 1 150); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s"; ok=1; break; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $TAG died"; break; }; sleep 2; done
if [ $ok = 1 ]; then
  B="python3 -I ekbench.py --url http://127.0.0.1:18080"
  run() { echo "--- $*"; "$@" 2>&1 | grep -v "^  File\\|^    \\|^Traceback\\|^urllib\\|^http\\.\\|^Connection\\|^During\\|^Exception\\|^\\s*\\^" | tee -a $OUT/results-$TAG.jsonl | cut -c1-200; kill -0 $pid 2>/dev/null || echo "SERVER DIED after [$*]"; }
  run $B solo p0 p1
  run $B multi p0 p1 p2 p3 p4 --rounds 2                 # five slots at once, cells reassigned between rounds
  run $B multi p0 p1 p2 --cache --rounds 2                # prompt cache on: slots keep and reuse their cells
  run $B solo p0 --cache
  run python3 -I stag.py p0 p1 --delays 1.0 0.01 --reps 3 --url http://127.0.0.1:18080   # a slot starts while another prefills / generates
  run $B multi p0 L16384 --rounds 1                        # the ek mix
  run $B solo p0 p1 p2
  kill -0 $pid 2>/dev/null && echo "SERVER ALIVE at the end" || echo "SERVER DIED"
fi
kill -INT $pid 2>/dev/null; for i in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; kill $WD 2>/dev/null
grep -a "gdn-state-check" $OUT/server-$TAG.log | sort | uniq -c | sort -rn | head -6 | cut -c1-240
grep -a -i "illegal\|abort\|CUDA error\|OUT OF RANGE" $OUT/server-$TAG.log | head -5 | cut -c1-240
rm -f snapshot/gdn-$TAG.k3sn snapshot/gdn-$TAG.k3sn.tmp
echo "GDN-LEG-DONE $TAG $(date -u +%T)"
