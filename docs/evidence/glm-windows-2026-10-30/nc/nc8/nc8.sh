#!/bin/bash
# GH200 window nc8 (2026-11-10/11): config A = today's production flags unchanged (hold rule on, no prompt pieces, no
# floor or burst) plus LLAMA_KPOOL_OWN_SEQS=1 (patch 0071) on the CI image FIX_TAG (stack through 0076; 0076 has no
# switch), against the production image OLD_TAG with production flags in the same window. Legs:
#  CP8 : production image and flags: p0 and 16k fresh, the mix that leaves a 16k and a 64k conversation in two slots,
#        p0/16k/64k alone, a 200k conversation, then 16k and p0 again (SEQ of window nc5).
#  AP8 : config A, the same sequence.
#  PB8 : production image and flags, the throughput part of FULL (one to five clients, the 16k and 64k mixes).
#  AA18, AA28 : config A, acceptance, two server processes (the FULL sequence of nc4-nc7 incl. the 100k requests).
#  WH8 : production image and flags with client classes: Juno alone and the class scenarios (her first token and
#        decode under load).
#  WA8 : config A with client classes, the same scenarios.
# Test keys generated here, never printed, deleted.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
OLD=ghcr.io/xerootg/llamacpp-spark:${OLD_TAG:?}
FIXIMG=ghcr.io/xerootg/llamacpp-spark:${FIX_TAG:?}
KO="LLAMA_KPOOL_OWN_SEQS=1"; H="LLAMA_PRIORITY_HOLD_PROMPTS=1"
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
CL="LLAMA_CLIENT_CLASSES=/keys/classes.txt LLAMA_SLOT_RESERVE=1"
K=$G/keys/classes.txt
mkkeys() { ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); chmod 0644 $K; }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$G/legs/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
SEQ=("ekbench.py solo p0 L16384" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
     "ekbench.py solo L200000" "ekbench.py solo L16384 p0 p0 p0")
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
wleg() {  # TAG IMG ENV
  mkkeys
  IMG=$2 CACHE_RAM=65536 bash nc-leg.sh $1 5 "$P $GR $F $CL $3" \
    "prio_nc.py warm --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py warm --classes $K --jn 1024" \
    "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
    "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py mixed --classes $K" \
    "prio_nc.py mixed --classes $K" "prio_nc.py warm --classes $K"
  leakcheck $1; rm -f $K
}
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    CP8) IMG=$OLD CACHE_RAM=65536 bash nc-leg.sh CP8 5 "$P $GR $F $H" "${SEQ[@]}" ;;
    AP8) IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh AP8 5 "$P $GR $F $H $KO" "${SEQ[@]}" ;;
    PB8) IMG=$OLD CACHE_RAM=65536 bash nc-leg.sh PB8 5 "$P $GR $F $H" "${FULL[@]:0:11}" "gold.py 1" ;;
    AA18|AA28) IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $H $KO" "${FULL[@]}" ;;
    WH8) wleg WH8 $OLD "$H" ;;
    WA8) wleg WA8 $FIXIMG "$H $KO" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC8-DONE $(date -u +%T)"
