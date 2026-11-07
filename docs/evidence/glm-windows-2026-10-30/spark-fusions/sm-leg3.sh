#!/bin/bash
# sm-leg.sh TAG BUILD MODE [ENV=V ...]   (Spark kernel-count work, 2026-11-07)
# BUILD = el (production ~/llama-k3-el/build-el) or sm (~/llama-k3-sm/build-sm). MODE = bench | dump | nsys | quick
# Production env + args (glm53-spark Deployment). Must run under flock window.lock (the caller wraps it).
set -u
TAG=$1; B=$2; MODE=$3; shift 3
cd /srv/glm-5.3-flash; OUT=legs-sm; mkdir -p $OUT nsys
case $B in el) BD=$HOME/llama-k3-el/build-el;; sm) BD=$HOME/llama-k3-sm/build-sm;; *) BD=$B;; esac
BIN=$BD/bin/llama-server; export LD_LIBRARY_PATH="$BD/bin"; export PATH=/usr/local/cuda/bin:$PATH
for v in ${!LLAMA_EXPERT_STORE_@} ${!GGML_CUDA_@} ${!LLAMA_EXACT_@} ${!LLAMA_ELASTIC_@}; do unset "$v"; done
M=q2/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf
pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT a llama-server is running"; exit 1; }
cp snapshot/sm-seed.k3sn snapshot/sm-$TAG.k3sn
ENV="LLAMA_LOG_COLORS=0 LLAMA_EXPERT_STORE=$PWD/store/glm53flash-q2.k3xs LLAMA_EXPERT_STORE_SNAPSHOT=$PWD/snapshot/sm-$TAG.k3sn LLAMA_EXPERT_STORE_THREADS=16 LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=6144 LLAMA_ELASTIC_KV=1 LLAMA_EXPERT_STORE_BYTE_STRIDE=1 LLAMA_EXPERT_STORE_PREFETCH=0 LLAMA_EXPERT_STORE_PREFETCH_INPUT=3 LLAMA_EXPERT_STORE_SPLIT_LATE=1 LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1 LLAMA_EXPERT_STORE_HOOK_LEAN=1 LLAMA_EXPERT_STORE_POLICY=lfu LLAMA_EXPERT_STORE_LFU_HALF_LIFE=262144 LLAMA_EXPERT_STORE_PREFILL_BANK=1 LLAMA_EXPERT_STORE_THP_BANK=1"
ARGS="-m $M --alias glm-5.3-flash --jinja --ctx-size 1048576 --parallel 1 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 78000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --metrics --no-warmup --fit off --load-mode mmap --host 127.0.0.1 --port 18080"
P64='{"prompt":"Write a Python quicksort function and explain it.","temperature":0,"seed":1,"n_predict":64,"cache_prompt":false}'
echo "=== LEG $TAG $B $MODE $(date +%T) extra: $*"
if [ $MODE = nsys ] || [ $MODE = nsys16k ]; then
  nsys sessions list 2>/dev/null | grep -q sm$TAG && nsys stop --session=sm$TAG >/dev/null 2>&1
  env $ENV "$@" nsys launch --session-new=sm$TAG --trace=cuda,nvtx,osrt --cuda-graph-trace=node $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & pid=$!
else
  env $ENV "$@" $BIN $ARGS > $OUT/server-$TAG.log 2>&1 & pid=$!
fi
( while kill -0 $pid 2>/dev/null; do a=$(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo); if [ "$a" -lt 4096 ]; then echo "WATCHDOG kill $TAG ${a} MiB" | tee -a $OUT/watchdog.txt; pkill -KILL -P $pid; kill -KILL $pid; break; fi; sleep 1; done ) & WD=$!
trap "kill -KILL $pid 2>/dev/null; kill $WD 2>/dev/null" EXIT
ok=0; for i in $(seq 1 300); do curl -fsS -m 2 127.0.0.1:18080/health 2>/dev/null | grep -q ok && { echo "READY $TAG after $((i*2))s"; ok=1; break; }; kill -0 $pid 2>/dev/null || { echo "LEG-FAIL $TAG died"; grep -a -i -E "error|abort|assert" $OUT/server-$TAG.log | tail -5 | cut -c1-300; break; }; sleep 2; done
if [ $ok = 1 ]; then
  case $MODE in
    bench) python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $TAG --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl
           python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $TAG-d16k --lens 16384 --n-predict 128 | tee -a $OUT/results.jsonl ;;
    fnv)   python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $TAG-fnv --n 1 --rounds 0 --n-predict 256 | tee -a $OUT/results.jsonl
           python3 -I bench.py depth --url http://127.0.0.1:18080 --tag $TAG-fnv-d16k --lens 16384 --n-predict 32 | tee -a $OUT/results.jsonl ;;
    nsys16k) curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d @$OUT/p16k-warm.json > /dev/null
           curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d @$OUT/p16k-dec.json > $OUT/$TAG.r0.json
           nsys start --session=sm$TAG -o $PWD/nsys/sm-$TAG --force-overwrite=true > $OUT/$TAG.nsys.out 2>&1; sleep 2
           curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d @$OUT/p16k-dec.json > $OUT/$TAG.r1.json
           nsys stop --session=sm$TAG >> $OUT/$TAG.nsys.out 2>&1
           python3 -c "import json,hashlib;d=json.load(open(\"$OUT/$TAG.r1.json\"));print(\"traced16k\",d[\"timings\"][\"prompt_n\"],round(d[\"timings\"][\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])" ;;
    quick) python3 -I bench.py conc --url http://127.0.0.1:18080 --tag $TAG --n 1 --rounds 3 --n-predict 256 | tee -a $OUT/results.jsonl ;;
    dump)  curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d "$P64" > $OUT/$TAG.r1.json
           python3 -c "import json,hashlib;d=json.load(open(\"$OUT/$TAG.r1.json\"));print(\"dump\",round(d[\"timings\"][\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])" ;;
    nsys)  curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d "$P64" > /dev/null
           nsys start --session=sm$TAG -o $PWD/nsys/sm-$TAG --force-overwrite=true > $OUT/$TAG.nsys.out 2>&1; sleep 2
           curl -s -m 900 127.0.0.1:18080/completion -H "Content-Type: application/json" -d "$P64" > $OUT/$TAG.r1.json
           nsys stop --session=sm$TAG >> $OUT/$TAG.nsys.out 2>&1
           python3 -c "import json,hashlib;d=json.load(open(\"$OUT/$TAG.r1.json\"));print(\"traced\",round(d[\"timings\"][\"predicted_per_second\"],2),\"tok/s md5\",hashlib.md5(d[\"content\"].encode()).hexdigest()[:10])" ;;
  esac
fi
kill -INT $pid 2>/dev/null; for i in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done; pkill -KILL -P $pid 2>/dev/null; kill -KILL $pid 2>/dev/null; wait $pid 2>/dev/null
kill $WD 2>/dev/null
if [ $MODE = nsys ] || [ $MODE = nsys16k ]; then nice -n 19 nsys export --type sqlite -o nsys/sm-$TAG.sqlite --force-overwrite=true nsys/sm-$TAG.nsys-rep > /dev/null 2>&1; ls -la nsys/sm-$TAG*; fi
[ $MODE = fnv ] && { grep -a -o "LOGITS_FNV.*" $OUT/server-$TAG.log > $OUT/$TAG.fnv; echo "fnv lines $(wc -l < $OUT/$TAG.fnv) last $(tail -1 $OUT/$TAG.fnv)"; }
echo "MemAvailable now $(awk "/MemAvailable/{print int(\$2/1024)}" /proc/meminfo) MiB; END $TAG $(date +%T)"
