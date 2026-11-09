#!/bin/bash
# Window bi1 (2026-11-09): per-op dumps of one client solo vs N=2/3/4 batches, plain and with the exact knobs (EXACT6).
cd /srv/glm-5.3-flash; L=legs-bi; mkdir -p $L
P="LLAMA_PREFILL_ISOLATE=1"
X="GGML_CUDA_EXACT_COLS=1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXACT_ATTN_COLS=1 GGML_CUDA_ARGSORT_EXACT_ROWS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
D="LLAMA_BI_DUMP_N=1234 LLAMA_BI_DUMP_STEPS=4"
RUNS=("solo p0 --n-predict 8" "multi p0 p1 --rounds 0 --n-predict 8" "multi p0 p1 p2 --rounds 0 --n-predict 8" "multi p0 p1 p2 p3 --rounds 0 --n-predict 8")
for leg in "$@"; do
  case $leg in
    D1) rm -f $L/dump-D1.txt; bash bi-leg.sh D1 4 4194304 "$P $D LLAMA_BI_DUMP=$PWD/$L/dump-D1.txt" "${RUNS[@]}" ;;
    D2) rm -f $L/dump-D2.txt; bash bi-leg.sh D2 4 4194304 "$P $X $D LLAMA_BI_DUMP=$PWD/$L/dump-D2.txt" "${RUNS[@]}" ;;
  esac
done
echo "BI-WIN1-DONE $(date -u +%T)"
