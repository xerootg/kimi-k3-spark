#!/bin/bash
# Window bi4, part 2 (2026-11-09): warm single-stream decode on the 4-slot server, stock vs 0054, same window.
cd /srv/glm-5.3-flash
for leg in "$@"; do
  case $leg in
    W0) bash bi-leg.sh W0 4 4194304 "LLAMA_PREFILL_ISOLATE=1" "solo p0 p0 p0 p0" "multi p0 p1 p2 p3 --rounds 1" ;;
    W1) bash bi-leg.sh W1 4 4194304 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1" "solo p0 p0 p0 p0" "multi p0 p1 p2 p3 --rounds 1" ;;
    W2) bash bi-leg.sh W2 1 1048576 "" "solo p0 p0 p0 p0" ;;
  esac
done
echo "BI-WIN5-DONE $(date -u +%T)"
