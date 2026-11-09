#!/bin/bash
# GH200 window nc2, part b: patch 0060 (prompt ubatches in pieces, LLAMA_PREFILL_SLICE_LAYERS) on the dev build
# (~/nc/out-a90 = CI tree + 0059 + 0060), production config + 0050 flags.
#  SL0u / SL0 / SL1L: prompt logits hash unsliced vs pieces of 4 / 1 layers (LLAMA_LOGITS_FNV=1), and every client's md5.
#  HB: production behaviour (hold rule on, no pieces) and SLP: pieces of 4 layers with the hold rule off, Juno's key at 10
#      (test keys generated here, never printed, deleted at the end): Juno alone; Juno answering while a low client's
#      64k prompt arrives; a 64k prompt being read when Juno arrives; Juno first with four low clients at +1 s.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
NEW=ghcr.io/xerootg/llamacpp-spark:547208dca250-ga5fa848
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
CL="LLAMA_CLIENT_CLASSES=/keys/classes.txt LLAMA_SLOT_RESERVE=1"
K=$G/keys/classes.txt
FNV=("chunk.py cold L16384 8" "chunk.py cold L65536 8" "ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1")
PRIO=("prio_nc.py warm --classes $K --jn 512" "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10"
      "prio_nc.py junofirst --classes $K" "prio_nc.py nokey --classes $K")
mkkeys() { ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); chmod 0644 $K; }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$G/legs/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
for leg in "$@"; do
  case $leg in
    SL0u) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SL0u 5 "$P $GR $F LLAMA_LOGITS_FNV=1" "${FNV[@]}" ;;
    SL0)  IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SL0 5 "$P $GR $F LLAMA_LOGITS_FNV=1 LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_LOG=1" "${FNV[@]}" ;;
    SL1L) IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SL1L 5 "$P $GR $F LLAMA_LOGITS_FNV=1 LLAMA_PREFILL_SLICE_LAYERS=1" "chunk.py cold L16384 8" ;;
    HB)   mkkeys; IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh HB 5 "$P $GR $F $CL LLAMA_PRIORITY_HOLD_PROMPTS=1" "${PRIO[@]}"; leakcheck HB; rm -f $K ;;
    SLP)  mkkeys; IMG=$NEW OPT=$HOME/nc/out-a90 CACHE_RAM=65536 bash nc-leg.sh SLP 5 "$P $GR $F $CL LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck SLP; rm -f $K ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K
echo "NC2B-DONE $(date -u +%T)"
