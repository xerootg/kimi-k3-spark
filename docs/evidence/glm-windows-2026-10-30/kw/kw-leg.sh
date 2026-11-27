#!/bin/bash
# Kernel windows on the Spark (2026-11-27): one llama-server leg with the production env and args (single stream),
# binary $BD/bin/llama-server (default ~/kf/src/build). glm53-spark must be at 0 replicas.
# Usage: kw-leg.sh TAG MODE "EXTRA_ENV"      MODE = bench (1 warm + 3 x 256 tokens, then 16k decode) | nsys (64-token trace)
set -u
cd /srv/glm-5.3-flash; OUT=legs-kw; mkdir -p $OUT nsys
BD=${BD:-$HOME/kf/src/build}; BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@} ${!LLAMA_KV_@} ${!LLAMA_PREFILL_@} ${!LLAMA_BI_@} ${!LLAMA_SLOT_@} ${!LLAMA_CLIENT_@} ${!LLAMA_PRIORITY_@} ${!LLAMA_KPOOL_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
PROD_ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_ELASTIC_KV_REFUSE=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1 GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_EXPERT_STORE_HOOK_STATS=1"
ARGS="-m $M --alias glm-5.3-flash --host 127.0.0.1 --port 18080 --ctx-size 1024000 --parallel 5 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --jinja --metrics --no-warmup --fit off --load-mode mmap -lv 4"
P64='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":64,"cache_prompt":false}'
TAG=$1; MODE=$2; EXTRA=${3:-}
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT $TAG: a llama-server is running"; exit 1; }
cp snapshot/el-seed.k3sn snapshot/kw-$TAG.k3sn
echo "=== LEG $TAG mode=$MODE extra=[$EXTRA] $(date -u +%T)"
if [ $MODE = nsys ]; then
  nsys sessions list 2>/dev/null | grep -q kw$TAG && nsys stop --session=kw$TAG >/dev/null 2>&1
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/kw-$TAG.k3sn $EXTRA nsys launch --session-new=kw$TAG --trace=cuda,nvtx --cuda-graph-trace=node $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & pid=$!
else
  env $PROD_ENV LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/kw-$TAG.k3sn $EXTRA $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & pid=$!
fi
nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active --format=csv,noheader -lms 250 > $OUT/clk-$TAG.txt 2>&1 & CLK=$!
( low=999999; while kill -0 $pid 2>/dev/null; do a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo); [ "$a" -lt "$low" ] && low=$a && echo "$low" > $OUT/lowmem-$TAG.txt
    if [ "$a" -lt 3072 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; pkill -KILL -P $pid; kill -KILL $pid; break; fi; sleep 1; done ) & WD=$!
ok=0; for i in $(seq 1 150); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s"; ok=1; break; }
  kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert|fail" $OUT/server-$TAG.log | tail -5 | cut -c1-300; break; }; sleep 2; done
if [ $ok = 1 ]; then
  grep -a -c "CUDA0" $OUT/server-$TAG.log | sed "s/^/CUDA0 lines: /"
  if [ $MODE = nsys ]; then
    curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d "$P64" > /dev/null
    nsys start --session=kw$TAG -o $PWD/nsys/kw-$TAG --force-overwrite=true > $OUT/$TAG.nsys.out 2>&1; sleep 2
    curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d "$P64" > $OUT/$TAG.r1.json
    nsys stop --session=kw$TAG >> $OUT/$TAG.nsys.out 2>&1
    python3 -c "import json,hashlib;d=json.load(open(\"$OUT/$TAG.r1.json\"));print(\"traced\",round(d[\"timings\"][\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])"
  else
    python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $TAG --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
    python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $TAG-d16k --lens 16384 --n-predict 128 | tee -a $OUT/results.jsonl
  fi
fi
kill -INT $pid 2>/dev/null; pkill -INT -P $pid 2>/dev/null; for i in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null; kill $WD $CLK 2>/dev/null
python3 - "$OUT/clk-$TAG.txt" <<"PY"
import sys,collections
rows=[l.strip().split(", ") for l in open(sys.argv[1]) if "MHz" in l]
sm=collections.Counter(r[0] for r in rows); pw=[float(r[2].split()[0]) for r in rows if r[2][0].isdigit()]
thr=collections.Counter(r[3] for r in rows)
print("clocks: sm", sm.most_common(3), "power W p50/max", sorted(pw)[len(pw)//2] if pw else "-", max(pw) if pw else "-", "throttle", thr.most_common(2))
PY
grep -a "hit rate\|host bank hooks" $OUT/server-$TAG.log | tail -2 | cut -c1-200
if [ $MODE = nsys ]; then for r in cuda_gpu_kern_sum cuda_gpu_trace; do nice -n 19 nsys stats --report $r --format csv -o nsys/kw-$TAG nsys/kw-$TAG.nsys-rep > /dev/null 2>&1; done; fi
rm -f snapshot/kw-$TAG.k3sn snapshot/kw-$TAG.k3sn.tmp
echo "lowest MemAvailable $(cat $OUT/lowmem-$TAG.txt 2>/dev/null) MiB; KW-LEG-DONE $TAG $(date -u +%T)"
