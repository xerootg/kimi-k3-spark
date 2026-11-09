#!/bin/bash
# GH200 window nc1, part b: patch 0059 (one-sequence prompt reserve under LLAMA_PREFILL_ISOLATE=1) on the production
# config (five slots, KV in Grace, 0054-0058 switches), dev build mounted over the CI image (OPT).
set -u
G=/srv/glm-5.3-flash/nc; cd $G
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
S=("gold.py 2" "ekbench.py solo p0 p0 p0 p0 p0 p0 p0" "ekbench.py solo L16384" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 p1 p3 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1")
for leg in "$@"; do
  case $leg in
    B5) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh B5 5 "$P $GR" "${S[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "NC1B-DONE $(date -u +%T)"
