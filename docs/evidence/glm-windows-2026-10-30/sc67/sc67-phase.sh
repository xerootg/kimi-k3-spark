#!/bin/bash
# sc67-phase.sh TAG URL: the request list of one canary process; results to $D/results.jsonl. Full list for tags
# without a "2", the short list (second process) for tags with one. Samples GPU power while it runs.
D=/srv/glm-5.3-flash/legs-sc67; TAG=$1; URL=$2
L=("gold --n 1" "gold --n 1 --who juno" "auth" "solo p1 p2 p3 p4" "multi p0 p1 --rounds 2" "multi p0 p1 p2 --rounds 2"
   "multi p0 p1 p2 p3 --rounds 2" "five --rounds 2" "multi L16384 p0 p1 p2 --rounds 1" "junofirst")
L2=("gold --n 1" "multi p0 p1 --rounds 1" "multi p0 p1 p2 --rounds 1" "multi p0 p1 p2 p3 --rounds 1" "five --rounds 1"
    "multi L16384 p0 p1 p2 --rounds 1")
case $TAG in *2) Q=("${L2[@]}") ;; *) Q=("${L[@]}") ;; esac
( while [ ! -e $D/phase-$TAG.done ]; do echo "$(date +%s) $(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null)"; sleep 2; done > $D/power-$TAG.txt ) &
echo "=== PHASE $TAG $(date -u +%T)"
for cmd in "${Q[@]}"; do
  echo "--- $cmd $(date -u +%T) $(date +%s)"
  python3 -I $D/acc67.py $cmd --keys $D/keys --url $URL --tag $TAG | tee -a $D/results.jsonl
done
echo "PHASE-DONE $TAG $(date -u +%T)"
touch $D/phase-$TAG.done
