#!/bin/bash
# Spark window bi2-1 (2026-11-09): patches 0056 (batch-invariant split decode attention GGML_CUDA_BI_ATTN_CHUNK, exact
# radix-select top-k) and 0057 (GGML_CUDA_KV_STAGE prompt-attention staging of host KV). Binary ~/bi2/a121 (CUDA 13.0.1
# container build of upstream + 0001-0055 + dev patch). glm53-spark at 0 replicas.
cd /srv/glm-5.3-flash; export BD=$HOME/bi2/a121
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
FULL=("solo p0 p1 p2 p3 L16384" "multi p0 p1 --rounds 2" "multi p0 p1 p2 --rounds 2" "multi p0 p1 p2 p3 --rounds 2"
      "multi p0 L16384 --rounds 2" "multi p1 p2 L16384 --rounds 1" "multi p1 p2 p3 L16384 --rounds 2" "solo p0 L16384")
for leg in "$@"; do
  case $leg in
    X1) bash bi-leg.sh X1 4 4194304 "$P GGML_CUDA_BI_TOPK_SORT=1" "solo p0 L16384" ;;
    X2) bash bi-leg.sh X2 4 4194304 "$P" "solo p0 L16384" ;;
    T1|T2) bash bi-leg.sh $leg 4 4194304 "$P GGML_CUDA_BI_ATTN_CHUNK=128" "${FULL[@]}" ;;
    Q0) bash bi-leg.sh Q0 1 1048576 "$P GGML_CUDA_BI_TOPK_SORT=1 GGML_CUDA_OP_PROFILE=16" "solo p0 L16384" ;;
    Q1) bash bi-leg.sh Q1 1 1048576 "$P GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_OP_PROFILE=16" "solo p0 L16384" ;;
    Q2) bash bi-leg.sh Q2 1 1048576 "GGML_CUDA_OP_PROFILE=16" "solo p0 L16384" ;;
    K0) XARGS="" bash bi-leg.sh K0 1 262144 "$P GGML_CUDA_BI_ATTN_CHUNK=128 LLAMA_KV_HOST=1" "solo L16384 L32768" ;;
    K1) XARGS="" bash bi-leg.sh K1 1 262144 "$P GGML_CUDA_BI_ATTN_CHUNK=128 LLAMA_KV_HOST=1 GGML_CUDA_KV_STAGE=1" "solo L16384 L32768" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "BI2-WIN1-DONE $(date -u +%T)"
