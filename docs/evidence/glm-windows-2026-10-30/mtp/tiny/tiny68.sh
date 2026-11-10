#!/bin/bash
# tiny68.sh TAG "ENV..." [server args...]: the server logic of patch 0068 (exact speculative decoding) on a small stock
# model on the CPU, with n-gram self-speculation as the draft source (no MTP head needed): one server process, the
# request list, stop by pid. T = work dir (build/bin/llama-server, model/smollm2-135m-q8.gguf), B = dir of specbench.py.
set -u
T=${T:?work dir}; B=${B:?script dir}; TAG=$1; EXTRA=$2; shift 2
O=$T/out; mkdir -p $O; URL=http://127.0.0.1:18190
env LLAMA_LOG_COLORS=0 LLAMA_PREFILL_ISOLATE=1 LLAMA_SPEC_EXACT_TRACE=1 $EXTRA nice -n 10 $T/build/bin/llama-server \
  -m $T/model/smollm2-135m-q8.gguf --host 127.0.0.1 --port 18190 --ctx-size 32768 --parallel 4 --batch-size 2048 \
  --ubatch-size 2048 --threads 6 --threads-batch 6 --no-warmup -lv 1 --n-gpu-layers 0 --cache-ram 0 "$@" > $O/server-$TAG.log 2>&1 &
PID=$!; echo $PID > $O/server.pid
for i in $(seq 1 60); do curl -fsS -m 2 $URL/health 2>/dev/null | grep -q ok && break; kill -0 $PID 2>/dev/null || { echo "DIED $TAG"; tail -5 $O/server-$TAG.log; exit 1; }; sleep 1; done
echo "== $TAG [$EXTRA] $*"
R=$O/results-$TAG.jsonl; : > $R
c() { timeout 900 python3 -I $B/specbench.py "$@" --url $URL --tag $TAG >> $R; kill -0 $PID 2>/dev/null || echo "SERVER DIED during [$*]"; }
c solo p0 p1 p2 p3
c solo p0:700 L2000:300
c solo p0 p1 p2 --temperature 1 --top-p 0.95 --seed 1
c solo p0 p1 --temperature 1 --top-p 0.95 --seed 2
c multi p0 p1 --rounds 1
c multi p0:40 p1:200 p2 --rounds 1
c multi p0 p1 p2 p3 --rounds 1
c multi p4:300 p5:90 --rounds 1 --temperature 1 --top-p 0.95 --seed 3
c follow p0 p3
c stop p1 p2
c solo p6 --n-probs 3
kill -INT $PID; for i in $(seq 1 20); do kill -0 $PID 2>/dev/null || break; sleep 1; done; kill -KILL $PID 2>/dev/null
rm -f $O/server.pid
grep -a "^SPEC_TRACE" $O/server-$TAG.log > $O/trace-$TAG.txt
echo "requests: $(wc -l < $R); trace lines: $(wc -l < $O/trace-$TAG.txt); errors: $(grep -a -c '"error"' $R)"
