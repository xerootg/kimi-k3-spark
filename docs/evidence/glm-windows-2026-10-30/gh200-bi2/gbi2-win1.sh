#!/bin/bash
# GH200 window gh200-bi3 (2026-11-09): measure patches 0056 (batch-invariant split decode attention, exact top-k) and
# 0057 (prompt-attention staging of KV in Grace memory) against stock and 0054, and find why 0054's N=4 aggregate dips
# (CUDA graph stats). glm53-gh200 at 0, workbench paused. Dev build ~/bi2/out-a90 (CUDA 13.0.1 container, same flags
# as the CI image) mounted over the CI image's /opt/llama.
# Usage (detached): setsid nohup ./gbi2-win1.sh LEG... > win1.log 2>&1 < /dev/null &
set -u
G=/srv/glm-5.3-flash/gbi2; cd $G; mkdir -p legs keys
D=$HOME/bi2/out-a90
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
DEPTH="ekbench.py solo p0 p0 p0 L16384 L65536"
G3=("ekbench.py multi p0 p1 p2 --rounds 1" "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1")
for leg in "$@"; do
  case $leg in
    S0)   bash gbi2-leg.sh S0 1 "" "gold.py 2" "$DEPTH" ;;                                    # stock, production image, 1 slot
    B0)   bash gbi2-leg.sh B0 1 "$P" "gold.py 2" "$DEPTH" ;;                                  # 0054, production image, 1 slot
    Z0)   OPT=$D bash gbi2-leg.sh Z0 1 "$P GGML_CUDA_BI_TOPK_SORT=1" "$DEPTH" ;;              # 0054 paths in the dev build
    Z1)   OPT=$D bash gbi2-leg.sh Z1 1 "$P" "$DEPTH" ;;                                       # + exact top-k (same bytes)
    N64|N128|N256) c=${leg#N}; OPT=$D bash gbi2-leg.sh $leg 1 "$P GGML_CUDA_BI_ATTN_CHUNK=$c" "gold.py 2" "$DEPTH" ;;
    F1)   OPT=$D bash gbi2-leg.sh F1 1 "$P GGML_CUDA_BI_FUSE=1" "gold.py 2" "$DEPTH" ;;              # 0054 + one-column fusions
    F5)   OPT=$D bash gbi2-leg.sh F5 5 "$P GGML_CUDA_BI_FUSE=1" "ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1" ;;
    RS)   bash gbi2-leg.sh RS 1 "GGML_CUDA_OP_PROFILE=16" "ekbench.py solo p0 L16384 L65536" ;;
    RZ)   OPT=$D bash gbi2-leg.sh RZ 1 "$P GGML_CUDA_BI_TOPK_SORT=1 GGML_CUDA_OP_PROFILE=16" "ekbench.py solo p0 L16384 L65536" ;;
    RN)   OPT=$D bash gbi2-leg.sh RN 1 "$P GGML_CUDA_BI_ATTN_CHUNK=${CHUNK:-128} GGML_CUDA_OP_PROFILE=16" "ekbench.py solo p0 L16384 L65536" ;;
    GB)   bash gbi2-leg.sh GB 5 "$P GGML_CUDA_GRAPH_STATS=1" "${G3[@]}" ;;                   # 0054: graph stats at N=3/4/5
    GS)   bash gbi2-leg.sh GS 5 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_GRAPH_STATS=1" "${G3[@]}" ;; # stock: same
    KE)   CACHE_RAM=32768 OPT=$D bash gbi2-leg.sh KE 5 "$P GGML_CUDA_BI_ATTN_CHUNK=${CHUNK:-128}" "bench.py depth --lens 16384,100000 --n-predict 64" ;;
    KH)   CACHE_RAM=32768 OPT=$D bash gbi2-leg.sh KH 5 "$P GGML_CUDA_BI_ATTN_CHUNK=${CHUNK:-128} LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1" \
            "bench.py depth --lens 16384,100000 --n-predict 64" ;;
    KS)   CACHE_RAM=32768 OPT=$D bash gbi2-leg.sh KS 5 "$P GGML_CUDA_BI_ATTN_CHUNK=${CHUNK:-128} LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1" \
            "bench.py depth --lens 16384,100000 --n-predict 64" ;;
    HS)   bash gbi2-leg.sh HS 5 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_GRAPH_STATS=1" "ekbench.py solo p0 p1 p2 p3 p4 L16384" \
            "ekbench.py multi p0 p1 --rounds 1" "ekbench.py multi p0 p1 p2 --rounds 1" "ekbench.py multi p0 p1 p2 p3 --rounds 2" ;;  # stock, T1's order (slot holes)
    RF)   OPT=$D bash gbi2-leg.sh RF 1 "$P GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 GGML_CUDA_OP_PROFILE=16" "ekbench.py solo p0 L16384 L65536" ;;
    K5)   CACHE_RAM=32768 OPT=$D bash gbi2-leg.sh K5 5 "$P GGML_CUDA_BI_ATTN_CHUNK=128 LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1" \
            "bench.py depth --lens 500000 --n-predict 32" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
pgrep -f "[l]lama-server -m " >/dev/null && echo "WARN a llama-server is still running" || echo "no test server left"
echo "GBI2-WIN1-DONE $(date -u +%T)"
