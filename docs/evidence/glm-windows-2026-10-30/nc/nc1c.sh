#!/bin/bash
# GH200 window nc1, part c: 0059 + the 0050 small-kernel fusions (never measured on the GH200) on the production config.
set -u
G=/srv/glm-5.3-flash/nc; cd $G
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
S=("gold.py 2" "ekbench.py solo p0 p0 p0 p0 p0 p0 p0" "ekbench.py solo L16384" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 p1 p3 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1" "ekbench.py solo L65536")
for leg in "$@"; do
  case $leg in
    B5F) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh B5F 5 "$P $GR $F" "${S[@]}" ;;
    STF) IMG=$NEW OPT=$HOME/nc/out-a90 bash nc-leg.sh STF 1 "$F" "gold.py 2" "ekbench.py solo p0 p0 p0 p0 p0 p0 p0" "ekbench.py solo L16384" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "NC1C-DONE $(date -u +%T)"
