#!/bin/bash
# Holds /srv/glm-5.3-flash/window.lock for the canary part of the window and watches memory: logs every new low of
# MemAvailable, kills the canary's llama-server (the only one on the box; pid by exact name at that moment) below
# 3072 MiB. Stops when $D/wd.stop exists.
D=/srv/glm-5.3-flash/legs-sc67; low=999999
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held" > $D/wd.err; exit 1; }
echo $$ > $D/wd.pid
while [ ! -e $D/wd.stop ]; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt "$low" ]; then low=$a; echo "$(date -u +%T) low $low MiB" >> $D/lowmem.log; fi
  if [ "$a" -lt 3072 ]; then
    for p in $(pgrep -x llama-server); do echo "$(date -u +%T) WATCHDOG KILL pid $p at $a MiB" >> $D/lowmem.log; sudo -n kill -KILL "$p"; done
  fi
  sleep 1
done
rm -f $D/wd.pid
