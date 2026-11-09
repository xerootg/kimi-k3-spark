#!/bin/bash
# Spark window bi2-2 (2026-11-09): patch 0058 (one ubatch for slots with holes, LLAMA_KV_STREAM_HOLES=1) and the
# GGML_CUDA_BI_FUSE=1 variants (one-column fusions and small-K kept in batch-invariant mode), on top of 0056 chunk 128.
# Binary ~/bi2/a121b. glm53-spark at 0 replicas. The solo requests first give every prompt a home slot (LCP similarity),
# so the later multi rounds land on non-contiguous slots.
cd /srv/glm-5.3-flash; export BD=$HOME/bi2/a121b
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128"
HOLES=("solo p0 p1 p2 p3" "multi p0 p1 p3 --rounds 2" "multi p0 p2 --rounds 2" "multi p1 p3 L16384 --rounds 1" "multi p0 p1 p2 p3 --rounds 1" "solo L16384")
for leg in "$@"; do
  case $leg in
    H0) bash bi-leg.sh H0 4 4194304 "$P" "${HOLES[@]}" ;;
    H1) bash bi-leg.sh H1 4 4194304 "$P LLAMA_KV_STREAM_HOLES=1" "${HOLES[@]}" ;;
    H2) bash bi-leg.sh H2 4 4194304 "$P LLAMA_KV_STREAM_HOLES=1 GGML_CUDA_BI_FUSE=1" "${HOLES[@]}" ;;
    P1F) bash bi-leg.sh P1F 1 1048576 "$P GGML_CUDA_BI_FUSE=1" "solo p0 p0 L16384" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "BI2-WIN2-DONE $(date -u +%T)"
