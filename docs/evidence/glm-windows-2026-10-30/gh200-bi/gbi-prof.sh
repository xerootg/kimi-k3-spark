#!/bin/bash
# gh200-bi follow-up legs (2026-11-09):
#  R0/R1: per-op decode profile, stock vs 0054, five slots, N=1 / N=4 / N=5 (why 0054's N=4 aggregate dips on the GH200:
#         49-50 tok/s against 63 at N=3 and 65 at N=5; stock 65 at N=4). Op profiler on (CUDA graphs off).
#  C1:    the production candidate as one leg: 0054 + 0055 classes (reserve 1, hold rule on) + KV in Grace memory (THP),
#         cache-ram 65536, test keys (generated here, 0600 dir, never printed, deleted at the end).
set -u
G=/srv/glm-5.3-flash/gbi; cd $G; mkdir -p keys; chmod 0700 keys
C=("ekbench.py solo p0 p0 p0" "ekbench.py multi p0 p1 p2 p3 --rounds 1" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1")
K=/keys/classes.txt; Q="prio_gh.py"
for leg in "$@"; do
  case $leg in
    R0) bash gbi-leg.sh R0 5 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_OP_PROFILE=16" "${C[@]}" ;;
    R1) bash gbi-leg.sh R1 5 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_OP_PROFILE=16" "${C[@]}" ;;
    C1) ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$G/keys/classes.txt','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); chmod 0644 $G/keys/classes.txt
        CACHE_RAM=65536 bash gbi-leg.sh C1 5 "LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1 LLAMA_PRIORITY_HOLD_PROMPTS=1" \
          "$Q warm --classes $G$K" "$Q ttft4 --classes $G$K" "$Q junofirst --classes $G$K" "$Q prefill --classes $G$K --delay 10" "$Q nokey --classes $G$K"
        python3 -c "
keys=[l.split()[0] for l in open('$G/keys/classes.txt') if l.strip()]
t=open('$G/legs/server-C1.log',errors='replace').read()
print('server-C1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"
        rm -f $G/keys/classes.txt ;;
  esac
done
echo "GBI-PROF-DONE $(date -u +%T)"
