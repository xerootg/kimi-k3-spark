#!/bin/bash
# GH200 window nc2 (2026-11-09): acceptance of patch 0059 + the 0050 fusion flags on the CI image (NEW_TAG), production
# config otherwise (#171: five slots, KV in Grace, 0054-0058 switches; client classes left out, no keys in the test).
# Per-client md5 equal to solo at N=1..5 with 16k and 64k mixes, two rounds, two server processes (A1, A2).
set -u
G=/srv/glm-5.3-flash/nc; cd $G
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?set NEW_TAG}
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64")
for leg in "$@"; do
  case $leg in
    A1) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A1 5 "$P $GR $F" "${FULL[@]}" ;;
    A2) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A2 5 "$P $GR $F" "${FULL[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "NC2-DONE $(date -u +%T)"
