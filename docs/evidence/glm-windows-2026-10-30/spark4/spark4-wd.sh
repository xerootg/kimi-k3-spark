#!/bin/bash
# Memory watchdog for the four-slot canary (2026-11-09): logs every new low of MemAvailable with a timestamp and a
# sample every 10 s; kills the canary's llama-server (explicit pid in $D/canary.pid) below 3072 MiB. Stops when
# $D/wd.stop exists.
D=/srv/glm-5.3-flash/legs-spark4; low=999999; n=0
while [ ! -e $D/wd.stop ]; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$a" -lt "$low" ]; then low=$a; echo "$(date -u +%T) low $low MiB" >> $D/lowmem.log; fi
  n=$((n+1)); [ $((n % 10)) -eq 0 ] && echo "$(date -u +%T) $a" >> $D/mem.log
  if [ "$a" -lt 3072 ] && [ -s $D/canary.pid ]; then
    p=$(cat $D/canary.pid); echo "$(date -u +%T) WATCHDOG KILL pid $p at $a MiB" >> $D/lowmem.log; sudo -n kill -KILL "$p"; : > $D/canary.pid
  fi
  sleep 1
done
