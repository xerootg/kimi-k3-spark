# window nd helpers (source me). Needs: OUT, BIN (llama-server), LD_LIBRARY_PATH. Run under flock window.lock.
cd /srv/glm-5.3-flash
export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_PREFILL_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
mkdir -p $OUT
ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1"
ARGS="-m $M --alias glm-5.3-flash --jinja -ngl 999 --n-cpu-moe 999 --moe-cache-mib 78000 --ctx-size ${CTX:-1048576} --flash-attn on --batch-size 2048 --ubatch-size 2048 --cache-ram 0 --threads 8 --threads-batch 20 --host 127.0.0.1 --port 18080 --metrics --no-warmup --fit off --load-mode mmap -lv 4"
EXACTN="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
SEED=snapshot/nd-seed.k3sn
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
[ -s $SEED ] || { echo "ABORT-WINDOW no seed snapshot"; exit 1; }
(while sleep 2; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) >> $OUT/mem.txt & MEMP=$!
trap 'kill $MEMP 2>/dev/null; [ -n "${pid:-}" ] && kill -KILL $pid 2>/dev/null' EXIT
start() { # tag parallel extra-env...
  local tag=$1 par=$2; shift 2
  cp $SEED snapshot/nd-$tag.k3sn
  echo "=== LEG $tag $(date +%T) parallel $par env: $*"
  env $ENV "$@" LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/nd-$tag.k3sn $BIN $ARGS --parallel $par > $OUT/server-$tag.log 2>&1 & pid=$!
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 4096 ]; then echo "WATCHDOG kill $tag ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) &
  for i in $(seq 1 300); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $tag after $((i*2))s"; return 0; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $tag died"; grep -a -i -E "error|abort|assert" $OUT/server-$tag.log | tail -3 | cut -c1-200; pid=; return 1; }; sleep 2; done; echo "LEG-FAIL $tag timeout"; return 1
}
stop() { [ -n "${pid:-}" ] || return 0; kill -INT $pid; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; rm -f snapshot/nd-*.k3sn.tmp; sleep 3; }
mb() { python3 -I $HOME/nd/mbench.py "$@" | tee -a $OUT/results.jsonl; }
finish() { cat $OUT/watchdog.txt 2>/dev/null; echo "MEMORY lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"; }
