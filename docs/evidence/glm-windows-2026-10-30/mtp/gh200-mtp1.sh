#!/bin/bash
# GH200 window mtp-g1 (patch 0068 with coupled drafting): the same two questions as spark-mtp2.sh on the GH200, where
# extra columns in a decode batch are much cheaper than on the Spark.
#  G0 : plain decoding (no MTP), production env, traced: the reference bytes and logits rows
#  G3 : exact speculation, 4 columns (3 drafts alone, 1 each at two clients, none at three or more), coupled drafting,
#       p_min 0.7, traced; the first MTP server on this box starts without a snapshot and saves one (mtp-seed.k3sn).
#       T03 = tracecmp G0 / G3
#  G5 : plain, no trace: speed            G4 : G3 without the trace: speed
#  G9 : G4 with 20 columns (3 drafts per client up to five clients), traced on the FULL list first (G9T, T09)
#  G6 : G4 with p_min 0.4                 G7 : G4 with the drafter's own greedy pick (LLAMA_SPEC_EXACT_COUPLED=0)
#  DIAG (run by "auto" only when T03 differs): G1 = MTP loaded, one column (no drafts); T01
# Usage: gh200-mtp1.sh auto | LEG...
G=/srv/glm-5.3-flash/mtp; cd $G
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
T="LLAMA_SPEC_EXACT_TRACE=1"
X="LLAMA_SPEC_EXACT=1 LLAMA_RS_KEEP_TAIL=0"
mtp() { echo "--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min $1"; }
T1="--temperature 1 --top-p 0.95"
SHORT=("specbench.py solo p0 p1 p2 p3 p4" "specbench.py solo p0 --n-predict 128"
       "specbench.py multi p0 p1 --rounds 0" "specbench.py multi p0 p1 p2 p3 p4 --rounds 0")
FULL=("${SHORT[@]}"
      "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py multi p0 p1 --rounds 1" "specbench.py multi p2 p5 --rounds 1 $T1 --seed 7"
      "specbench.py multi p0 p1 p2 --rounds 0" "specbench.py multi p0:60 p1:256 p2:150 --rounds 0" "specbench.py multi p3:40 p4:200 --rounds 0 $T1 --seed 11"
      "specbench.py multi p0:50 p1:256 p2:120 p3:200 p4:90 --rounds 0 $T1 --seed 13"
      "specbench.py follow p0 p6" "specbench.py follow p2 $T1 --seed 5" "specbench.py stop p0 p3" "specbench.py stop p2 p5 $T1 --seed 3"
      "specbench.py multi L16384 p0 p1 --rounds 0" "specbench.py solo p1 --n-probs 3 --n-predict 48")
SPEED=("specbench.py solo p0 p0 p1 p2 p3 p4" "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py solo p1 p3 p6 $T1 --seed 9"
       "specbench.py multi p0 p1 --rounds 2" "specbench.py multi p2 p5 --rounds 1 $T1 --seed 7" "specbench.py multi p0 p1 p2 --rounds 1"
       "specbench.py multi p0 p1 p2 p3 p4 --rounds 2" "specbench.py multi p0 p1 p2 p3 p4 --rounds 1 $T1 --seed 21")
QUICK=("specbench.py solo p0 p0 p1 p2" "specbench.py solo p0 p2 p5 $T1 --seed 42" "specbench.py solo p1 p3 p6 $T1 --seed 9"
       "specbench.py multi p0 p1 --rounds 1")
MS=${MTPSEED:-mtp-seed.k3sn}
[ -s snap/$MS ] && MS3=$MS || MS3=none
leg() {
  case $1 in
    G0) bash gmtp-leg.sh G0 5 "$T" "${FULL[@]}" ;;
    G1) SEED=$MS3 XARGS="$(mtp 0.7)" bash gmtp-leg.sh G1 5 "$T LLAMA_SPEC_EXACT=1 LLAMA_SPEC_EXACT_COLS=1 LLAMA_RS_KEEP_TAIL=0" "${SHORT[@]}" ;;
    G3) SEED=$MS3 SAVE=mtp-seed.k3sn XARGS="$(mtp 0.7)" bash gmtp-leg.sh G3 5 "$T $X" "${FULL[@]}" ;;
    G4) SEED=$MS XARGS="$(mtp 0.7)" bash gmtp-leg.sh G4 5 "$X" "${SPEED[@]}" ;;
    G5) bash gmtp-leg.sh G5 5 "" "${SPEED[@]}" ;;
    G6) SEED=$MS XARGS="$(mtp 0.4)" bash gmtp-leg.sh G6 5 "$X" "${QUICK[@]}" ;;
    G7) SEED=$MS XARGS="$(mtp 0.7)" bash gmtp-leg.sh G7 5 "$X LLAMA_SPEC_EXACT_COUPLED=0" "${QUICK[@]}" ;;
    G9T) SEED=$MS XARGS="$(mtp 0.7)" bash gmtp-leg.sh G9T 5 "$T $X LLAMA_SPEC_EXACT_COLS=20" "${FULL[@]}" ;;
    G9) SEED=$MS XARGS="$(mtp 0.7)" bash gmtp-leg.sh G9 5 "$X LLAMA_SPEC_EXACT_COLS=20" "${SPEED[@]}" ;;
    T01) python3 -I tracecmp.py --subset legs/trace-G0.txt legs/trace-G1.txt | tail -8 ;;
    T03) python3 -I tracecmp.py legs/trace-G0.txt legs/trace-G3.txt | tee legs/t03.txt | tail -12 ;;
    T09) python3 -I tracecmp.py legs/trace-G0.txt legs/trace-G9T.txt | tee legs/t09.txt | tail -12 ;;
    *) echo "unknown leg $1" ;;
  esac
}
if [ "${1:-}" = auto ]; then
  leg G0; leg G3; leg T03
  if grep -a -q "^TRACE-IDENTICAL" legs/t03.txt; then
    for l in G5 G4 G9T T09 G9 G6; do leg $l; done
  else
    echo "T03 differs: diagnosis leg instead of speed"
    for l in G1 T01; do leg $l; done
  fi
else
  for l in "$@"; do leg $l; done
fi
echo "GH200-MTP1-DONE $(date -u +%T)"
