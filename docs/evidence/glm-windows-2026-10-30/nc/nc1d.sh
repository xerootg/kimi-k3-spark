#!/bin/bash
# GH200 window nc1, part d: decode at depth on a cached prompt (no prefill right before), stock single slot vs 0059 + 0050.
set -u
G=/srv/glm-5.3-flash/nc; cd $G
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
STOCK=ghcr.io/xerootg/llamacpp-spark:547208dca250-g7f117e1
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
S=("ekbench.py solo p0 p0 p0" "ekbench.py solo L16384 L16384 L16384 --cache" "ekbench.py solo L65536 L65536 L65536 --cache")
for leg in "$@"; do
  case $leg in
    DST)  IMG=$STOCK bash nc-leg.sh DST 1 "" "${S[@]}" ;;
    DB5F) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh DB5F 5 "$P $GR $F" "${S[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "NC1D-DONE $(date -u +%T)"
