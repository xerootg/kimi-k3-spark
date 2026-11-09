#!/bin/bash
# Window bi3 (2026-11-09): success test of patch 0054 v1 (GGML_CUDA_BATCH_INVARIANT=1): per-client md5 solo vs N=2/3/4 with
# the 16k request, two full rounds in two server processes, then LLAMA_SLOT_SERIAL=1 and a single-slot server for speed.
cd /srv/glm-5.3-flash; L=legs-bi; mkdir -p $L
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
FULL=("solo p0 p1 p2 p3 L16384" "multi p0 p1 --rounds 2" "multi p0 p1 p2 --rounds 2" "multi p0 p1 p2 p3 --rounds 2"
      "multi p0 L16384 --rounds 2" "multi p1 p2 L16384 --rounds 1" "multi p1 p2 p3 L16384 --rounds 2" "solo p0 L16384")
for leg in "$@"; do
  case $leg in
    T1|T2) bash bi-leg.sh $leg 4 4194304 "$P" "${FULL[@]}" ;;
    S1) bash bi-leg.sh S1 4 4194304 "LLAMA_PREFILL_ISOLATE=1 LLAMA_SLOT_SERIAL=1" "solo p0" "multi p0 p1 p2 p3 --rounds 2" "multi p0 p1 --rounds 1" ;;
    S2) bash bi-leg.sh S2 4 4194304 "$P LLAMA_SLOT_SERIAL=1" "solo p0" "multi p0 p1 p2 p3 --rounds 1" ;;
    P1) bash bi-leg.sh P1 1 1048576 "GGML_CUDA_BATCH_INVARIANT=1" "solo p0 p0 p0 L16384" ;;
  esac
done
echo "BI-WIN3-DONE $(date -u +%T)"
