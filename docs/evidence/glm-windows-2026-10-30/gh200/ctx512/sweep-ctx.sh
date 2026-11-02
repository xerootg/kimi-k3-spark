#!/bin/bash
# GH200 context-size sweep: for each (ctx, HBM tier) start the production-equivalent server (ship build),
# read a ~100k-token prompt, decode 64 tokens at that depth, run the standard request, record peak GPU memory.
# glm53-gh200 must be scaled to 0.
cd /srv/glm-5.3
export BIN=$HOME/llama-ship/build/bin/llama-server LONG_TOK=100000
for cfg in "262144 40960" "393216 28672" "524288 16384"; do
  set -- $cfg; ctx=$1; hbm=$2; tag=ctx${ctx}-hbm${hbm}
  (while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > prof/gpumem-$tag.txt & m=$!
  echo "=== $tag $(date +%T)"
  CTX=$ctx ./ctx-bench.sh $tag GGML_CUDA_MMQ_ID_TILING=1 LLAMA_EXPERT_STORE_HBM_MIB=$hbm 2>&1 | grep -a -E "BENCH|died|rror"
  kill $m; echo "PEAK $tag $(sort -n prof/gpumem-$tag.txt | tail -1) MiB of 97871"
  grep -a -E "KV buffer size|kv.*MiB" prof/$tag/server.log | head -2 | cut -c1-160
done
echo SWEEP-DONE
