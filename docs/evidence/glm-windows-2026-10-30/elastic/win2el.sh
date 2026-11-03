#!/bin/bash
# Window el (kimi-k3-spark patch 0045, elastic KV): GLM-5.3-Flash on the Spark, full 1M context, expert store.
# A = patched build with elastic KV off (production settings, 58000 MiB cache); B = elastic on, same cache (md5 and
# memory); C = elastic on, 78000 MiB cache (the KV's unused memory lent to the experts), plus a 200k prompt that makes
# the KV grow and the valve hand cache slots back, then the short test again after the KV shrank.
# Every leg restores a copy of the production snapshot. Watchdog kills the server below 3 GiB MemAvailable.
# Production glm53-spark must be at 0.
set -u
cd /srv/glm-5.3-flash
BIN=$HOME/llama-k3-el/build-el/bin/llama-server
export LD_LIBRARY_PATH="$HOME/llama-k3-el/build-el/bin"
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
OUT=legs-el; mkdir -p $OUT
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT-WINDOW a llama-server is running"; exit 1; }
[ -s snapshot/el-seed.k3sn ] || { echo "ABORT-WINDOW no seed snapshot"; exit 1; }
(while sleep 2; do echo "$(date +%s) $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)"; done) > $OUT/mem.txt & MEMP=$!
trap 'kill $MEMP 2>/dev/null; [ -n "${pid:-}" ] && kill -KILL $pid 2>/dev/null' EXIT
STORE_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1"
start() { # tag cache_mib reserve_mib elastic
  echo "=== LEG $1 cache_MiB=$2 reserve_MiB=$3 elastic=$4 $(date +%T)"
  cp snapshot/el-seed.k3sn snapshot/el-$1.k3sn
  local mark; mark=$(wc -l < $OUT/mem.txt); echo "$1 $mark" >> $OUT/marks.txt
  env $STORE_ENV LLAMA_EXPERT_STORE_RESERVE_MIB=$3 LLAMA_ELASTIC_KV=$4 LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/el-$1.k3sn "$BIN" -m $M --alias glm-5.3-flash --jinja -ngl 999 --n-cpu-moe 999 --moe-cache-mib $2 \
    --ctx-size 1048576 --parallel 1 --flash-attn on --batch-size 2048 --ubatch-size 2048 --cache-ram 0 \
    --threads 8 --threads-batch 20 --host 127.0.0.1 --port 18080 --metrics --no-warmup --fit off --load-mode mmap -lv 4 > $OUT/server-$1.log 2>&1 & pid=$!
  ( while kill -0 $pid 2>/dev/null; do a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $1 MemAvailable ${a} MiB" | tee -a $OUT/watchdog.txt; kill -KILL $pid; break; fi; sleep 1; done ) &
  for i in $(seq 1 450); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $1 after $((i*2))s MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB"; grep -a -h -E "KV buffer size|elastic KV|snapshot restored|compute buffer size" $OUT/server-$1.log | head -8 | cut -c1-190; return 0; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $1 server died"; grep -a -v "unused tensor" $OUT/server-$1.log | tail -6 | cut -c1-220; pid=; return 1; }; sleep 2; done
  echo "LEG-FAIL $1 health timeout"; return 1
}
stop() { [ -n "${pid:-}" ] || return 0; kill -INT $pid; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; pid=; }
legmem() { local m; m=$(awk -v t="$1" '$1==t{print $2}' $OUT/marks.txt); echo "MEMORY $1 lowest MemAvailable MiB $(tail -n +$m $OUT/mem.txt | awk '{print $2}' | sort -n | head -1)"; }
short() { # tag
  python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $1 --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
  python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $1-d16k --lens 16384 --n-predict 128 | tee -a $OUT/results.jsonl
  grep -a -E "hit rate" $OUT/server-$2.log | tail -1 | cut -c1-200
}
if start A 58000 4096 0; then short A A; stop; fi; legmem A
if start B 58000 4096 1; then short B B; grep -a -c "elastic KV (" $OUT/server-B.log; grep -a "elastic KV (" $OUT/server-B.log | tail -2 | cut -c1-200; stop; fi; legmem B
if start C 78000 6144 1; then
  short C C
  python3 -I bench.py depth --url http://127.0.0.1:18080 --tag C-200k --lens 200000 --n-predict 64 | tee -a $OUT/results.jsonl
  grep -a -E "memory valve" $OUT/server-C.log | tail -3 | cut -c1-230
  grep -a "elastic KV (" $OUT/server-C.log | tail -2 | cut -c1-200
  sleep 45   # the next request drops the long sequence (shrink), the valve re-arms after 10 s of calm
  short C2 C
  grep -a -E "memory valve|elastic KV \(seq_rm\)" $OUT/server-C.log | tail -4 | cut -c1-230
  sleep 30
  python3 -I bench.py conc --url http://127.0.0.1:18080 --tag C3 --n 1 --rounds 2 --n-predict 256 | tee -a $OUT/results.jsonl
  stop
fi; legmem C
cat $OUT/watchdog.txt 2>/dev/null
echo "MEMORY window lowest MemAvailable MiB $(awk '{print $2}' $OUT/mem.txt | sort -n | head -1)"
echo WIN2EL-DONE
