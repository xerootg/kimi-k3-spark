#!/bin/bash
# start f53s (row order per request by the wire magic, 2026-11-10) on this NUC: nucserve5.sh <name> <port>
# A memory watchdog kills it below 2 GiB MemAvailable. Stop it with: kill $(cat ~/nd/f53s.pid)
set -u
cd ~/nd
N=$1; P=$2
[ -f f53s.pid ] && [ "$(cat /proc/$(cat f53s.pid)/comm 2>/dev/null)" = f53s4 ] && { echo "already running pid $(cat f53s.pid)"; exit 0; }
F53_MUFU=$HOME/glm53-mufu F53_CPUS=0,1,2,5,6,7,8,9,10,11,12,13 F53_SPIN_US=${F53_SPIN_US:-20000} setsid nohup ./f53s4 serve $N.bin $N.idx $P > f53s5-$N.log 2>&1 < /dev/null &
echo $! > f53s.pid
pid=$(cat f53s.pid)
setsid nohup bash -c "while kill -0 $pid 2>/dev/null; do a=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo); if [ \$a -lt 2048 ]; then echo \"WATCHDOG kill f53s \$a MiB \$(date)\" >> f53s-watchdog.log; kill -9 $pid; fi; sleep 1; done" > /dev/null 2>&1 < /dev/null &
echo "started pid $pid"
