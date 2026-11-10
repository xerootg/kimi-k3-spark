#!/bin/bash
# GH200 window nc4, second runner (after nc4.sh): the share knob. V2 measured Juno against solo numbers of another
# process (HB3 of window nc3), so these legs measure Juno alone (256, 512 and 1024 tokens) in the same server process as
# the contended scenarios, at LLAMA_PREFILL_SLICE_SHARE 0.12 (leg W12) and 0.08 (leg W08). CI image NEW_TAG, test keys
# generated here, never printed, deleted.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?}
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
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    W12) SH=0.12 ;;
    W08) SH=0.08 ;;
    *) echo "unknown leg $leg"; continue ;;
  esac
  SL="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=$SH LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
  mkkeys
  IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL" \
    "prio_nc.py warm --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py warm --classes $K --jn 1024" \
    "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
    "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py mixed --classes $K" \
    "prio_nc.py warm --classes $K"
  leakcheck $leg; rm -f $K
done
rm -f $K $G/nc3.pid
echo "NC4B-DONE $(date -u +%T)"
