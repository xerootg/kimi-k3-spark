#!/bin/bash
# Window bi2 (2026-11-09): patch 0054 v1 (GGML_CUDA_BATCH_INVARIANT=1): md5 per client solo vs N=2/3/4, then per-op dumps.
cd /srv/glm-5.3-flash; L=legs-bi; mkdir -p $L
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
D="LLAMA_BI_DUMP_N=1234 LLAMA_BI_DUMP_STEPS=3"
R64=("solo p0 p1 p2 p3 --n-predict 64" "multi p0 p1 --rounds 0 --n-predict 64" "multi p0 p1 p2 --rounds 0 --n-predict 64" "multi p0 p1 p2 p3 --rounds 0 --n-predict 64")
RUNS=("solo p0 --n-predict 8" "multi p0 p1 --rounds 0 --n-predict 8" "multi p0 p1 p2 --rounds 0 --n-predict 8" "multi p0 p1 p2 p3 --rounds 0 --n-predict 8")
for leg in "$@"; do
  case $leg in
    B1) bash bi-leg.sh B1 4 4194304 "$P" "${R64[@]}" ;;
    B1D) rm -f $L/dump-B1D.txt; bash bi-leg.sh B1D 4 4194304 "$P $D LLAMA_BI_DUMP=$PWD/$L/dump-B1D.txt" "${RUNS[@]}" ;;
  esac
done
echo "BI-WIN2-DONE $(date -u +%T)"
