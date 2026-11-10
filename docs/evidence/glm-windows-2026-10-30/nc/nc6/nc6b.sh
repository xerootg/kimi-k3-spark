#!/bin/bash
# GH200 window nc6, second runner (2026-11-10): candidate patch 0074 (LLAMA_PREFILL_SLICE_SAME_FREE=<tokens>: a prompt
# with at most that many tokens left is not held to the same-class share). KA1/KA2 of this window against PB (the
# production image, adjacent): equal at two clients, 1.2 % lower at three, 2.6 % at four, 2.3 % at five in ekbench's
# aggregate (tokens over the round's wall time), while every stream's own decode speed is equal or higher. The same
# pattern as on the Spark (window nc-s2): with 0072 the first short prompt starts generating, the other prompts of
# its class are then held to LLAMA_PREFILL_SLICE_SHARE_SAME.
# Dev build DEV (sm_90, the dev tree at 0074) mounted over the CI image FIX_TAG. Legs:
#  KJ0: the dev build with the variable unset, the multi commands only (the control for the build itself)
#  KJ1: the same with LLAMA_PREFILL_SLICE_SAME_FREE=512: gold, solo, N=2..5 two rounds, the 16k mixes, staggered
#       pairs, p1 with 512 tokens alone (the reference for "samelong" of XK/XH), gold
set -u
G=/srv/glm-5.3-flash/nc; cd $G
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
FIXIMG=ghcr.io/xerootg/llamacpp-spark:${FIX_TAG:?}
DEV=${DEV:?}
KO="LLAMA_KPOOL_OWN_SEQS=1"; SF="LLAMA_PREFILL_SHORT_FIRST=1"
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=0.05 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
M=("ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2")
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    KJ0) IMG=$FIXIMG OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh KJ0 5 "$P $GR $F $SL $KO $SF" "ekbench.py solo p0" "${M[@]}" ;;
    KJ1) IMG=$FIXIMG OPT=$DEV CACHE_RAM=65536 bash nc-leg.sh KJ1 5 "$P $GR $F $SL $KO $SF LLAMA_PREFILL_SLICE_SAME_FREE=512" \
           "gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "${M[@]}" "ekbench.py multi p0 L16384 --rounds 1" \
           "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 2" "ekbench.py solo p1 --n-predict 512" "gold.py 1" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $G/nc3.pid
echo "NC6B-DONE $(date -u +%T)"
