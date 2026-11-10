#!/bin/bash
# run2.sh TAG OUTDIR "ENV..." "SERVER ARGS..." MODE... : one small CPU server (docker, no GPU) on LFM2-350M, the modes of t2.py, stop.
set -u
TAG=$1; OUT=$2; EXTRA=$3; SARGS=$4; shift 4; D=$HOME/nc/tiny
EOPTS=""; for kv in LLAMA_LOG_COLORS=0 LLAMA_PREFILL_ISOLATE=1 $EXTRA; do EOPTS="$EOPTS -e $kv"; done
sudo docker run --rm -d --name nctiny2 --network host --cpus 16 -v $D:/models:ro -v $HOME/nc/$OUT:/opt/llama:ro $EOPTS \
  ghcr.io/xerootg/llamacpp-spark:547208dca250-g68d4166 -m /models/lfm2-350m-q8.gguf --host 127.0.0.1 --port 18190 --ctx-size 163840 --parallel 5 \
  --batch-size 2048 --ubatch-size 2048 --threads 8 --threads-batch 8 --no-warmup -lv 4 --n-gpu-layers 0 --slots $SARGS > /dev/null
for i in $(seq 1 60); do curl -fsS -m 2 127.0.0.1:18190/health 2>/dev/null | grep -q ok && break; sleep 1; done
echo "== $TAG [$OUT] [$EXTRA] [$SARGS]"
for m in "$@"; do timeout 1200 python3 -I $D/t2.py $m ${NB:-20000} ${NA:-1500}; done
sudo docker logs nctiny2 > $D/server2-$TAG.log 2>&1
sudo docker kill nctiny2 > /dev/null
echo "pieces: $(grep -a -c 'pieces' $D/server2-$TAG.log)  refused ranges: $(grep -a -c 'contains cell' $D/server2-$TAG.log)  whole (state not in place): $(grep -a -c 'runs whole' $D/server2-$TAG.log)"
grep -a "RSSUM seq=" $D/server2-$TAG.log | sed -E 's/^.*RSSUM/RSSUM/' | awk '$4 != "n=1"' > $D/rssum-$TAG.txt; wc -l < $D/rssum-$TAG.txt
