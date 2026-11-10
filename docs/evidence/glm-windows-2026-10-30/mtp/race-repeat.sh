#!/bin/bash
# Repeats of the race legs of gh200-mtp1.sh (R0: the server's synchronize and the backend's tripwire both off;
# R1: synchronize off, tripwire on), to see whether R0's damaged draft stream and R1's silence repeat.
# Each run's server log, trace and comparison are kept under legs/ with the run number.
# Usage: DUB=512 [FIRST=2] race-repeat.sh N     (N runs numbered from FIRST; the legs take window.lock themselves)
G=/srv/glm-5.3-flash/mtp; cd $G
for i in $(seq ${FIRST:-2} $((${FIRST:-2} + ${1:-2} - 1))); do
  for t in R0 R1; do
    # the previous leg's monitor may hold the lock for another two seconds (its sleep inherits the descriptor)
    flock /srv/glm-5.3-flash/window.lock true
    bash gh200-mtp1.sh $t | grep -a -v "^[0-9]\.[0-9][0-9]\.\|llama_model_loader" | cut -c1-420
    python3 -I tracecmp.py --subset legs/trace-G0.txt legs/trace-$t.txt | tail -2
    cp legs/server-$t.log legs/server-$t-run$i.log; cp legs/trace-$t.txt legs/trace-$t-run$i.txt
    echo "RUN $i $t: aborts $(grep -a -c 'host bank:\|after the plan' legs/server-$t.log)"
    grep -a "draft acceptance" legs/server-$t.log | cut -c1-160
  done
done
echo "RACE-REPEAT-DONE $(date -u +%T)"
