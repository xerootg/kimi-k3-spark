#!/bin/bash
# GH200 window nc4, fourth runner (nc4d.sh): DG and DW of nc4c.sh read a long prompt alone at the speed without pieces
# (1638 and 1481 tok/s), so the 1334 / 1408 of A1t and A2t come from the state the earlier mixes leave (a 16k and a 64k
# conversation in two slots). Legs here rebuild that state (the mix p1 p2 L16384 L65536, then p0, 16k, 64k alone):
# DH with pieces and the pieces log, DJ the same image with the hold rule and no pieces (production flags), DI with
# pieces on but every ubatch whole (LLAMA_PREFILL_SLICE_MIN_TOKENS above any ubatch). Header of nc4c.sh:
# GH200 window nc4, third runner: legs DG and DW look at a long prompt read ALONE with pieces (acceptance A1t/A2t read
# 16k at 1334 and 64k at 1408 tok/s against 1642 and 1488 without pieces): DG logs the pieces, DW is the control with
# the same build and flags but LLAMA_PREFILL_SLICE_MIN_TOKENS above any ubatch (every ubatch runs whole). Then W12 of
# nc4b.sh. Original header of nc4b.sh: V2 measured Juno against solo numbers of another
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
  SL="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=0.12 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
  SEQ=("ekbench.py solo p0 L16384" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536")
  case $leg in
    DH) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh DH 5 "$P $GR $F $SL LLAMA_PREFILL_SLICE_LOG=1" "${SEQ[@]}" ;;
    DJ) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh DJ 5 "$P $GR $F LLAMA_PRIORITY_HOLD_PROMPTS=1" "${SEQ[@]}" ;;
    DI) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh DI 5 "$P $GR $F $SL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_PREFILL_SLICE_MIN_TOKENS=1000000" "${SEQ[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC4D-DONE $(date -u +%T)"
