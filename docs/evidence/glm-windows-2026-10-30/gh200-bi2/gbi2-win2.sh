#!/bin/bash
# GH200 window gh200-bi4 (2026-11-09): acceptance of patches 0056-0058 on the CI image. Per-client md5 equal to solo at
# N=1..5 with 16k and 64k mixes, two rounds, two server processes (A1, A2); throughput against the current production
# config (PR: image g64500e2, five slots) and stock single-slot production before #169 (ST: image g7f117e1).
# glm53-gh200 at 0, workbench paused. Usage (detached): setsid nohup ./gbi2-win2.sh LEG... > win2.log 2>&1 < /dev/null &
set -u
G=/srv/glm-5.3-flash/gbi2; cd $G; mkdir -p legs keys
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?set NEW_TAG}
PROD=ghcr.io/xerootg/llamacpp-spark:547208dca250-g64500e2
STOCK=ghcr.io/xerootg/llamacpp-spark:547208dca250-g7f117e1
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1"     # production (#169) minus classes
N="GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 GGML_CUDA_KV_STAGE=1 LLAMA_KV_STREAM_HOLES=1"  # 0056-0058
FULL=("ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536")
for leg in "$@"; do
  case $leg in
    PR) IMG=$PROD CACHE_RAM=65536 bash gbi2-leg.sh PR 5 "$P" "gold.py 2" "${FULL[@]}" "bench.py depth --lens 100000 --n-predict 64" ;;
    A1) IMG=$NEW  CACHE_RAM=65536 bash gbi2-leg.sh A1 5 "$P $N" "gold.py 2" "${FULL[@]}" "bench.py depth --lens 100000 --n-predict 64" ;;
    A2) IMG=$NEW  CACHE_RAM=65536 bash gbi2-leg.sh A2 5 "$P $N" "gold.py 2" "${FULL[@]}" ;;
    ST) IMG=$STOCK bash gbi2-leg.sh ST 1 "" "gold.py 2" "ekbench.py solo p0 p0 p0 L16384 L65536" ;;
    A0) IMG=$NEW  bash gbi2-leg.sh A0 1 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 $N" "gold.py 2" "ekbench.py solo p0 p0 p0 L16384 L65536" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
pgrep -f "[l]lama-server -m " >/dev/null && echo "WARN a llama-server is still running" || echo "no test server left"
echo "GBI2-WIN2-DONE $(date -u +%T)"
