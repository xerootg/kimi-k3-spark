#!/bin/bash
# deadman-s8.sh DEADLINE_EPOCH MARKER : laptop-side guard for Spark window nc-s8 (from nc/deadman.sh's Spark case; the
# GH200 case is not here, this window never touches that box). Started detached (setsid) before the window. If MARKER
# exists at any check the window ended normally and this exits. At DEADLINE it stops the window's runner, leg and test
# server by the pids in /srv/glm-5.3-flash/legs-nc3/, removes the keys and the per-leg snapshots, puts glm53-spark back
# to 1 replica (sparkprod.sh up: waits Ready, verifies the gold through the Service) and writes a note to the
# coordination file.
DL=$1; MARK=$2
SP=/claude/claude-1000/-home-xero-repos-argo-things/fd7931bb-f519-4321-a3de-81d4f945b7bf/scratchpad
PROD=$SP/nc/k3s/docs/evidence/glm-windows-2026-10-30/nc/spark/sparkprod.sh
LOG=$SP/nc/s8/deadman-s8.log
echo "$(date -u +%FT%T) armed for the Spark, deadline $(date -u -d @$DL +%T) UTC, marker $MARK" >> $LOG
while [ "$(date +%s)" -lt "$DL" ]; do [ -e "$MARK" ] && { echo "$(date -u +%T) marker present, disarmed" >> $LOG; exit 0; }; sleep 20; done
[ -e "$MARK" ] && { echo "$(date -u +%T) marker present, disarmed" >> $LOG; exit 0; }
echo "$(date -u +%T) DEADLINE: stopping the window on the Spark and restoring production" >> $LOG
ssh -o ConnectTimeout=20 xero@192.168.4.209 'cd /srv/glm-5.3-flash/legs-nc3 2>/dev/null || exit 0
  for f in runner.pid leg.pid; do [ -s $f ] && { p=$(cat $f); kill -KILL -- -$p 2>/dev/null; kill -KILL $p 2>/dev/null; echo "killed $f $p"; }; done
  [ -s server.pid ] && { p=$(cat server.pid); kill -KILL $p 2>/dev/null; echo "killed server $p"; }
  sleep 3; rm -f runner.pid leg.pid server.pid /srv/glm-5.3-flash/keys-nc/classes.txt /srv/glm-5.3-flash/keys-nc/key-cc.txt /srv/glm-5.3-flash/snapshot/nc-S*.k3sn
  pgrep -af "[l]lama-server -m " || echo "no llama-server left"' < /dev/null >> $LOG 2>&1
bash $PROD up >> $LOG 2>&1
cd /home/xero/repos/argo-things && python3 $SP/coord-append54.py "NOTE (deadman of the bisection agent, automatic): Spark window nc-s8 passed its hard deadline; the test processes were stopped by pid and glm53-spark was scaled back to 1. Result of the restore: $(tail -4 $LOG | tr '\n' ' ' | cut -c1-400). Treat the nc-s8 Spark CLAIM as RELEASED." >> $LOG 2>&1
echo "$(date -u +%T) deadman done" >> $LOG
