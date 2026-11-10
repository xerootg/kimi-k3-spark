#!/bin/bash
# Window nc6: once leg XH has ended, stop the first runner before leg XT (dropped for time: the 0074 legs matter more)
# and start the second runner. Kills by the pid group in nc3.pid, as the deadman does.
G=/srv/glm-5.3-flash/nc; cd $G
T0=$SECONDS
until command grep -a -q "^server-XH.log" nc6a.log; do [ $((SECONDS - T0)) -gt ${WAIT:-400} ] && { echo "XH not ended yet"; exit 3; }; sleep 2; done
command grep -a -q "NC6-DONE" nc6a.log && { echo "runner already done"; exit 4; }
p=$(cat nc3.pid); case "$p" in ''|*[!0-9]*) echo "no runner pid"; exit 5;; esac
kill -KILL -- -$p; echo "killed runner group $p $(date -u +%T)"
for c in $(sudo docker ps -q --filter name=nc-XT); do sudo docker kill -s KILL $c > /dev/null; echo "killed container nc-XT"; done
sleep 3; rm -f nc3.pid keys/classes.txt; sudo rm -f snap/snap-XT.k3sn
sudo docker ps --filter name=nc- --format "{{.Names}}" | command grep . || echo "no nc container left"
echo "XT dropped by nc6-switch.sh $(date -u +%T)" >> nc6a.log
( FIX_TAG=547208dca250-g76c2c21 DEV=/home/ubuntu/nc/out-j90 DEADLINE=$1 setsid bash nc6b.sh KJ0 KJ1 > nc6b.log 2>&1 < /dev/null & )
sleep 2; echo "second runner pid $(cat nc3.pid) $(date -u +%T)"
