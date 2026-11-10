#!/bin/bash
# Spark window mtp2 (patch 0068 with coupled drafting): is speculation with the MTP head invisible in the output, and
# what does it buy? Legs in the order of their worth, so a short window still answers the first question.
#  S0 : plain decoding (no MTP), production env, traced: the reference bytes and logits rows
#  S3 : exact speculation (4 columns: 3 drafts alone, 1 each at two clients, none at three or more), coupled drafting,
#       p_min 0.7, traced; T03 = tracecmp S0 / S3
#  S5 : plain, no trace: speed            S4 : S3 without the trace: speed
#  S6 : S4 with p_min 0.4                 S7 : S4 with the drafter's own greedy pick (LLAMA_SPEC_EXACT_COUPLED=0)
#  S8 : S4 with 2 columns (one draft token)
#  DIAG (run by "auto" only when T03 differs): S1 = MTP loaded, one column (no drafts); S2 = S1 with
#       LLAMA_RS_KEEP_TAIL=0; T01, T02
# Usage: spark-mtp2.sh auto | LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=${BD:-$HOME/nt/e121}
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
T="LLAMA_SPEC_EXACT_TRACE=1"
X="LLAMA_SPEC_EXACT=1 LLAMA_RS_KEEP_TAIL=0"
mtp() { echo "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min $1"; }
T1="--temperature 1 --top-p 0.95"
SHORT=("specbench.py solo p0 p1 p2 p3 p4" "specbench.py solo p0 --n-predict 128"
       "specbench.py multi p0 p1 --rounds 0" "specbench.py multi p0 p1 p2 p3 p4 --rounds 0")
FULL=("${SHORT[@]}"
      "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py multi p0 p1 --rounds 1" "specbench.py multi p2 p5 --rounds 1 $T1 --seed 7"
      "specbench.py multi p0 p1 p2 --rounds 0" "specbench.py multi p0:60 p1:256 p2:150 --rounds 0" "specbench.py multi p3:40 p4:200 --rounds 0 $T1 --seed 11"
      "specbench.py follow p0 p6" "specbench.py follow p2 $T1 --seed 5" "specbench.py stop p0 p3" "specbench.py stop p2 p5 $T1 --seed 3"
      "specbench.py multi L16384 p0 p1 --rounds 0" "specbench.py solo p1 --n-probs 3 --n-predict 48")
SPEED=("specbench.py solo p0 p0 p1 p2 p3 p4" "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py solo p1 p3 p6 $T1 --seed 9"
       "specbench.py multi p0 p1 --rounds 2" "specbench.py multi p2 p5 --rounds 1 $T1 --seed 7" "specbench.py multi p0 p1 p2 --rounds 1")
QUICK=("specbench.py solo p0 p0 p1 p2" "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py solo p1 p3 p6 $T1 --seed 9")
MS=${MTPSEED:-fl-mtp.k3sn}
leg() {
  case $1 in
    S0) bash mtp-leg.sh S0 5 1024000 "$P $T" "${FULL[@]}" ;;
    S1) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh S1 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1" "${SHORT[@]}" ;;
    S2) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh S2 5 1024000 "$P $T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1 LLAMA_RS_KEEP_TAIL=0" "${SHORT[@]}" ;;
    S3) SEED=$MS SAVE=mtp-seed.k3sn XARGS="$(mtp 0.7)" bash mtp-leg.sh S3 5 1024000 "$P $T $X" "${FULL[@]}" ;;
    S4) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh S4 5 1024000 "$P $X" "${SPEED[@]}" ;;
    S5) bash mtp-leg.sh S5 5 1024000 "$P" "${SPEED[@]}" ;;
    S6) SEED=$MS XARGS="$(mtp 0.4)" bash mtp-leg.sh S6 5 1024000 "$P $X" "${QUICK[@]}" ;;
    S7) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh S7 5 1024000 "$P $X LLAMA_SPEC_EXACT_COUPLED=0" "${QUICK[@]}" ;;
    S8) SEED=$MS XARGS="$(mtp 0.7)" bash mtp-leg.sh S8 5 1024000 "$P $X LLAMA_SPEC_EXACT_COLS=2" "${QUICK[@]}" ;;
    T01) python3 -I tracecmp.py --subset legs-mtp/trace-S0.txt legs-mtp/trace-S1.txt | tail -8 ;;
    T02) python3 -I tracecmp.py --subset legs-mtp/trace-S0.txt legs-mtp/trace-S2.txt | tail -8 ;;
    T03) python3 -I tracecmp.py legs-mtp/trace-S0.txt legs-mtp/trace-S3.txt | tee legs-mtp/t03.txt | tail -12 ;;
    *) echo "unknown leg $1" ;;
  esac
}
if [ "${1:-}" = auto ]; then
  leg S0; leg S3; leg T03
  if grep -a -q "^TRACE-IDENTICAL" legs-mtp/t03.txt; then
    for l in S5 S4 S6 S8 S7; do leg $l; done
  else
    echo "T03 differs: diagnosis legs instead of speed"
    for l in S1 T01 S2 T02; do leg $l; done
  fi
else
  for l in "$@"; do leg $l; done
fi
echo "SPARK-MTP2-DONE $(date -u +%T)"
