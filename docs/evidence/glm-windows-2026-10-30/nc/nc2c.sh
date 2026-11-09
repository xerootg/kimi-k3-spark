#!/bin/bash
# GH200 window nc2, part c: patch 0060 after the cut-rule fix (pieces of k layers really cut between layers) and the
# capped decode reserve: prompt logits hash with pieces of 4 / 1 layers, every client's md5.
set -u
G=/srv/glm-5.3-flash/nc; cd $G
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
FNV=("chunk.py cold L16384 8" "chunk.py cold L65536 8" "ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1")
for leg in "$@"; do
  case $leg in
    SL0b)  IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SL0b 5 "$P $GR $F LLAMA_LOGITS_FNV=1 LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_LOG=1" "${FNV[@]}" ;;
    SL1Lb) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SL1Lb 5 "$P $GR $F LLAMA_LOGITS_FNV=1 LLAMA_PREFILL_SLICE_LAYERS=1 LLAMA_PREFILL_SLICE_LOG=1" "chunk.py cold L16384 8" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "NC2C-DONE $(date -u +%T)"
