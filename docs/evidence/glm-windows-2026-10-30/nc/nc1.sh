#!/bin/bash
# GH200 window nc1 (2026-11-09): diagnostics for the no-compromise items.
#  item 1: one client, short prompt, warm: stock single slot (ST) vs the new config on one slot (A0), one slot with the KV
#          in Grace (A1G), production five slots (P5), five slots with the KV in HBM (P5H), stock five slots (ST5);
#          op profiles (R-*), nsys kernel traces (NS-*).
#  item 2: do a prompt's bytes depend on its ubatch boundaries? LLAMA_LOGITS_FNV=1, cold vs split prompts (CH), and
#          ubatch 2048 (CH2).
# glm53-gh200 at 0, workbench paused. Usage (detached): setsid nohup ./nc1.sh LEG... > nc1.log 2>&1 < /dev/null &
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p legs keys
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
STOCK=ghcr.io/xerootg/llamacpp-spark:547208dca250-g7f117e1
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
S=("gold.py 2" "ekbench.py solo p0 p0 p0 p0 p0 p0 p0" "ekbench.py solo L16384")
PROF="GGML_CUDA_OP_PROFILE=16 LLAMA_EXPERT_STORE_HOOK_STATS=1"
Q="ekbench.py solo p0 p0 p0 --n-predict 128"
for leg in "$@"; do
  case $leg in
    ST)   IMG=$STOCK bash nc-leg.sh ST 1 "" "${S[@]}" ;;
    A0)   IMG=$NEW bash nc-leg.sh A0 1 "$P" "${S[@]}" ;;
    A1G)  IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A1G 1 "$P $GR" "${S[@]}" ;;
    P5)   IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh P5 5 "$P $GR" "${S[@]}" ;;
    P5H)  IMG=$NEW bash nc-leg.sh P5H 5 "$P" "${S[@]}" ;;
    ST5)  IMG=$STOCK bash nc-leg.sh ST5 5 "LLAMA_PREFILL_ISOLATE=1" "${S[@]}" ;;
    CH)   IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh CH 5 "$P $GR LLAMA_LOGITS_FNV=1" "chunk.py cold L16384 8" "chunk.py cold L16384 8" \
            "chunk.py split L16384 6001 8" "chunk.py cold L65536 8" "chunk.py split L65536 30001 8" ;;
    CH2)  IMG=$NEW CACHE_RAM=65536 XARGS="--batch-size 2048 --ubatch-size 2048" bash nc-leg.sh CH2 5 "$P $GR LLAMA_LOGITS_FNV=1" \
            "chunk.py cold L16384 8" "chunk.py cold L65536 8" ;;
    R-ST) IMG=$STOCK bash nc-leg.sh R-ST 1 "$PROF" "$Q" ;;
    R-A0) IMG=$NEW bash nc-leg.sh R-A0 1 "$P $PROF" "$Q" ;;
    R-P5) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh R-P5 5 "$P $GR $PROF" "$Q" ;;
    NS-ST) IMG=$STOCK NSYS=1 STOPWAIT=600 bash nc-leg.sh NS-ST 1 "" "$Q" ;;
    NS-A0) IMG=$NEW NSYS=1 STOPWAIT=600 bash nc-leg.sh NS-A0 1 "$P" "$Q" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
pgrep -f "[l]lama-server" >/dev/null && echo "WARN a llama-server is still running" || echo "no test server left"
echo "NC1-DONE $(date -u +%T)"
