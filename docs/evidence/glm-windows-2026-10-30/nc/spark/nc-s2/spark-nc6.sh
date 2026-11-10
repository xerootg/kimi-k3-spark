#!/bin/bash
# Spark window nc-s2, third runner (2026-11-10): the abort of leg SA3 ("expert store: the split's real lookup of layer
# 16 came without its early lookup (-1)"), which hit when p0's 6-token first ubatch ran while slot 1 generated.
# Reading: a prompt ubatch of 6..8 tokens (more tokens than slots, so a prompt ubatch; at most 8, so the column split of
# patch 0067) goes through the pieces path of patch 0061 when another slot generates, and that combination is new.
# stag.py sends p1, then p0 after a set gap (1 s: p0 arrives while p1 generates; 0.01 s: while p1's prompt is read).
#  RA: the config of SA3 (pieces, 0071, 0072)            RB: the same without LLAMA_PREFILL_SHORT_FIRST
#  RC: the config of SA3 without the column split         RD: today's production config (no pieces)
#  RF: the config of SA3, more gaps and both orders (for a build with a fix, BD=...)
#  SA4: the acceptance sequence on the config of SA3
set -u
cd /srv/glm-5.3-flash; export BD=${BD:-$HOME/nc/a121h}
PN="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 $PN"
SL="LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_SHARE=0.05 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
T="LLAMA_ELASTIC_KV_FLOOR_CELLS=204800 LLAMA_PREFILL_SLICE_TG_CELLS=204800 LLAMA_KPOOL_OWN_SEQS=1"
FIN="$P $SL $T LLAMA_PREFILL_SHORT_FIRST=1"
OUT=legs-nc2; mkdir -p $OUT; echo $$ > $OUT/runner.pid
ACC=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py solo L16384"
     "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2"
     "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1"
     "ekbench.py solo p0 --n-predict 128")
ST=("stag.py p1 p0 --delays 1.0 0.01 --reps 3")
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    RA) bash nc2-leg.sh RA 5 5242880 "$FIN LLAMA_PREFILL_SLICE_LOG=1" "${ST[@]}" ;;
    RB) bash nc2-leg.sh RB 5 5242880 "$P $SL $T LLAMA_PREFILL_SLICE_LOG=1" "${ST[@]}" ;;
    RC) bash nc2-leg.sh RC 5 5242880 "$PN $SL $T LLAMA_PREFILL_SHORT_FIRST=1 LLAMA_PREFILL_SLICE_LOG=1" "${ST[@]}" ;;
    RD) bash nc2-leg.sh RD 5 1024000 "$P" "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 3" "stag.py p0 p1 --delays 1.0 0.01 --reps 3" ;;
    RF*) bash nc2-leg.sh $leg 5 5242880 "$FIN LLAMA_PREFILL_SLICE_LOG=1" "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 4" "stag.py p0 p1 --delays 1.0 0.01 --reps 3" \
          "stag.py p2 p0 --delays 2.0 --reps 2 --n-predict 256" ;;
    SA4|SA5) bash nc2-leg.sh $leg 5 5242880 "$FIN" "${ACC[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $OUT/runner.pid
echo "SPARK-NC6-DONE $(date -u +%T)"
