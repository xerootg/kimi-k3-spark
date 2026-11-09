#!/bin/bash
# spark4-phase.sh TAG URL SCENARIO-LINE...   each line: "<scenario> [specs/options]"; results to $D/results.jsonl
D=/srv/glm-5.3-flash/legs-spark4; TAG=$1; URL=$2; shift 2
echo "=== PHASE $TAG $(date -u +%T)"
for cmd in "$@"; do
  echo "--- $cmd $(date -u +%T)"
  python3 -I $D/spark4.py $cmd --keys $D/keys --url $URL --tag $TAG | tee -a $D/results.jsonl
done
echo "PHASE-DONE $TAG $(date -u +%T)"
