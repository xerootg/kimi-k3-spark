#!/bin/bash
# run.sh TAG "ENV..." : one small CPU server (docker, no GPU), the three tests, stop.
set -u
TAG=$1; EXTRA=$2; D=$HOME/nc/tiny
EOPTS=""; for kv in LLAMA_LOG_COLORS=0 LLAMA_PREFILL_ISOLATE=1 $EXTRA; do EOPTS="$EOPTS -e $kv"; done
sudo docker run --rm -d --name nctiny --network host --cpus 16 -v $D:/models:ro -v $HOME/nc/out-a90:/opt/llama:ro $EOPTS \
  ghcr.io/xerootg/llamacpp-spark:547208dca250-g68d4166 -m /models/smollm2-135m-q8.gguf --host 127.0.0.1 --port 18190 --ctx-size 32768 --parallel 4 \
  --batch-size 2048 --ubatch-size 2048 --threads 8 --threads-batch 8 --no-warmup -lv 4 --n-gpu-layers 0 > /dev/null
for i in $(seq 1 60); do curl -fsS -m 2 127.0.0.1:18190/health 2>/dev/null | grep -q ok && break; sleep 1; done
echo "== $TAG [$EXTRA]"
for m in solo share turns; do timeout 600 python3 -I $D/t.py $m; done
sudo docker logs nctiny > $D/server-$TAG.log 2>&1
sudo docker kill nctiny > /dev/null
grep -a -c "piece" $D/server-$TAG.log
