#!/bin/bash
# Window bi5 (2026-11-09): patch 0055 v2 (lower-class prompts wait while a higher class generates), and LLAMA_SLOT_BATCH_MAX. Test keys are generated here, kept in a 0600
# file and never printed; the file is deleted at the end of the window.
cd /srv/glm-5.3-flash; L=legs-bi; mkdir -p $L
K=$L/classes.txt
umask 077
python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))"
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1"
C="python3 -I prio_bench.py"
for leg in "$@"; do
  case $leg in
    Q2) CLIENT="$C" bash bi-leg.sh Q2 4 4194304 "$P LLAMA_CLIENT_CLASSES=$PWD/$K LLAMA_SLOT_RESERVE=1" \
          "ttft3 --classes $K" "queue4 --classes $K" "prefill --classes $K --delay 20" "prefill --classes $K --delay 20" ;;
    Q3) CLIENT="$C" bash bi-leg.sh Q3 4 4194304 "$P LLAMA_CLIENT_CLASSES=$PWD/$K LLAMA_SLOT_RESERVE=1 LLAMA_SLOT_BATCH_MAX=2" \
          "ttft3 --classes $K" ;;
  esac
done
for f in $L/server-Q2.log $L/server-Q3.log; do
  [ -f $f ] && python3 -c "
import sys
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$f',errors='replace').read()
print('$f', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"
done
rm -f $K
echo "BI-WIN6-DONE $(date -u +%T)"
