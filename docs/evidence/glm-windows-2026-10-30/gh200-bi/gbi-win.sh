#!/bin/bash
# GH200 window gh200-bi (2026-11-09): patches 0054 (batch-invariant decode) + 0055 (client priority) + 0051 (KV in
# Grace memory) on the GH200 with five slots, docker legs of the CI image g64500e2. glm53-gh200 at 0, workbench paused.
# G: LLAMA_KV_HOST_THP=1 because the probe showed 64 KiB pages (0051's default on this kernel) thrash the GPU TLB at depth.
# Usage (detached): setsid nohup ./gbi-win.sh LEG... > win.log 2>&1 < /dev/null &
set -u
G=/srv/glm-5.3-flash/gbi; cd $G; mkdir -p legs keys
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
FULL=("ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536")
K=/keys/classes.txt; Q="prio_gh.py"
newkeys() { # five test keys, 0600, never printed; deleted at the end of the window
  ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$G/keys/classes.txt','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" )
  chmod 0644 $G/keys/classes.txt   # read by the server as root inside the container and by prio_gh.py; the dir is 0700
  chmod 0700 $G/keys
}
for leg in "$@"; do
  case $leg in
    probe) pgrep -f "[l]lama-server -m " >/dev/null && { echo "ABORT probe: a llama-server is running"; continue; }
           echo "=== kvprobe $(date -u +%T)"; $G/kvprobe 24 16384,100000,500000,1000000 dev,reg,regnothp,halloc 2>&1 | tee legs/kvprobe.txt ;;
    P1) bash gbi-leg.sh P1 1 "" "gold.py 2" "ekbench.py solo p0 p0 p0 L16384" ;;
    P0) bash gbi-leg.sh P0 1 "GGML_CUDA_BATCH_INVARIANT=1" "gold.py 2" "ekbench.py solo p0 p0 p0 L16384" ;;
    T1|T2) bash gbi-leg.sh $leg 5 "$P" "${FULL[@]}" ;;
    S1) bash gbi-leg.sh S1 5 "LLAMA_PREFILL_ISOLATE=1 LLAMA_SLOT_SERIAL=1" "ekbench.py solo p0 p0" "ekbench.py multi p0 p1 --rounds 1" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" ;;
    B1) bash gbi-leg.sh B1 5 "LLAMA_PREFILL_ISOLATE=1" "ekbench.py solo p0 p0" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 1" \
          "ekbench.py multi p0 p1 p2 p3 --rounds 1" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" ;;
    Q0|Q1) [ -s $G/keys/classes.txt ] || newkeys
           H=""; [ $leg = Q1 ] && H="LLAMA_PRIORITY_HOLD_PROMPTS=1"
           bash gbi-leg.sh $leg 5 "$P LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1 $H" "$Q warm --classes $G$K" "$Q ttft4 --classes $G$K" "$Q ttft4 --classes $G$K" \
             "$Q prefill --classes $G$K --delay 10" "$Q mixed --classes $G$K --delay 10" "$Q junofirst --classes $G$K" "$Q nokey --classes $G$K" ;;
    H) CACHE_RAM=32768 bash gbi-leg.sh H 5 "$P" "bench.py depth --lens 16384,100000,500000 --n-predict 128" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1" ;;
    G) CACHE_RAM=32768 bash gbi-leg.sh G 5 "$P LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1" "bench.py depth --lens 16384,100000,500000 --n-predict 128" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
if [ -s $G/keys/classes.txt ]; then
  for f in legs/server-*.log; do python3 -c "
keys=[l.split()[0] for l in open('$G/keys/classes.txt') if l.strip()]
t=open('$f',errors='replace').read()
print('$f', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; done
fi
rm -f $G/keys/classes.txt
pgrep -f "[l]lama-server -m " >/dev/null && echo "WARN a llama-server is still running" || echo "no test server left"
echo "GBI-WIN-DONE $(date -u +%T)"
