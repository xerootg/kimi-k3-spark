#!/bin/bash
# Spark window nuc5, last leg: the column split only from three columns (patch 0067, LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3).
# Window nuc5 showed the split costs ~3.5 % at two clients and pays from three. Build ~/nt/d121 (stack through 0067
# with the gate; 0068 is in the build and inert). Gw: bytes and tok/s at one to five clients and the 16k mix.
# Waits for window.lock (the running chain holds it), so it starts when the chain ends.
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock 9
export BD=$HOME/nt/d121
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SC="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3"
G=("ekbench.py solo p0" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 1" "ekbench.py multi p0 p1 p2 p3 --rounds 1"
   "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1" "ekbench.py multi L16384 p0 p1 p2 p3 --rounds 0")
for leg in "$@"; do
  case $leg in
    Gw) bash nuc5-leg.sh Gw 5 1024000 "$P $SC" "${G[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "SPARK-NUC5G-DONE $(date -u +%T)"
