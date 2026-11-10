#!/bin/bash
# GH200 window nc5 (2026-11-10). Window nc4 saw a long prompt read slower once other slots hold long conversations
# (about 2 ns per prompt token and foreign KV cell) and booked it against image gc93c6d9; the per-ubatch times and the
# code point at the pooled indexer of the base code instead (its pool list covers every sequence). Legs:
#  CP : control on the PRODUCTION image OLD_TAG with production flags and the hold rule: p0 and 16k fresh, the mix that
#       leaves a 16k and a 64k conversation in two slots, p0/16k/64k alone, a 200k conversation, then 16k and p0 again
#       (a third point for the "linear in foreign cells" reading, and decode next to 264k foreign cells).
#  KP : the same sequence with patch 0071 (LLAMA_KPOOL_OWN_SEQS=1: a ubatch's pool list holds its own sequences only)
#       and production flags otherwise. Expect the fresh speeds after the mixes and CP's md5 for every request.
#  KV : the detector gate of nc4's V1 (pieces config, client classes, LLAMA_RS_LOG/LLAMA_RS_SUM) with 0071 on.
#       v1digest.py decides; a failure stops the legs that have 0071 on.
#  KA1, KA2: acceptance, two server processes (the FULL sequence of nc4), pieces config and 0071 on.
#  WK08, WK12, WK05: the share decision for the pieces with 0071 on: Juno alone (256/512/1024 tokens) in the same server
#       process as the contended scenarios, LLAMA_PREFILL_SLICE_SHARE 0.08, 0.12, 0.05 (W12 of nc4: 0.12 without 0071).
#  W08: the same at 0.08 on the CI image NEW_TAG without 0071 (only if 0071 fails).
#  WH : the same scenarios on the PRODUCTION image and config (hold rule on): the "before" of the share legs.
# The 0071 legs run on the CI image FIX_TAG when it is set, else on the dev build DEV mounted over NEW_TAG.
# Test keys generated here, never printed, deleted.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?}
OLD=ghcr.io/xerootg/llamacpp-spark:${OLD_TAG:?}
DEV=${DEV:-/home/ubuntu/nc/out-g90}
if [ -n "${FIX_TAG:-}" ]; then FIXIMG=ghcr.io/xerootg/llamacpp-spark:$FIX_TAG; FIXOPT=""; else FIXIMG=$NEW; FIXOPT=$DEV; fi
KO="LLAMA_KPOOL_OWN_SEQS=1"
E="prio_nc_e.py"
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
sl() { echo "LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=$1 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"; }
SEQ=("ekbench.py solo p0 L16384" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
     "ekbench.py solo L200000" "ekbench.py solo L16384 p0 p0 p0")
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
wleg() {  # TAG IMG OPT ENV
  mkkeys
  IMG=$2 OPT=$3 CACHE_RAM=65536 bash nc-leg.sh $1 5 "$P $GR $F $CL $4" \
    "prio_nc.py warm --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py warm --classes $K --jn 1024" \
    "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
    "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py mixed --classes $K" \
    "prio_nc.py warm --classes $K"
  leakcheck $1; rm -f $K
}
GATE=ok
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in K*|WK*) [ $GATE = ok ] || { echo "GATE failed, leg $leg skipped"; continue; } ;; esac
  case $leg in
    CP) IMG=$OLD CACHE_RAM=65536 bash nc-leg.sh CP 5 "$P $GR $F LLAMA_PRIORITY_HOLD_PROMPTS=1" "${SEQ[@]}" ;;
    KP) IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh KP 5 "$P $GR $F LLAMA_PRIORITY_HOLD_PROMPTS=1 $KO" "${SEQ[@]}" ;;
    KV) D="--dump $G/legs/dump-KV"; mkkeys
        IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh KV 5 "$P $GR $F $CL $(sl 0.12) $KO LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K $D" "$E sololong --classes $K --long L16384 $D" "$E sololong --classes $K $D" \
          "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" \
          "$E mixed --classes $K $D" "$E stagger --classes $K $D" "$E mixed --classes $K $D" \
          "$E junolong --classes $K --jn 512 --delay 1 $D" "$E prefill --classes $K --delay 10 $D" "$E stagger --classes $K --long L16384 $D"
        leakcheck KV; rm -f $K
        python3 -I $G/v1digest.py KV || GATE=fail ;;
    KA1|KA2) IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $(sl 0.12) $KO" "${FULL[@]}" ;;
    WK08) wleg WK08 $FIXIMG "$FIXOPT" "$(sl 0.08) $KO" ;;
    WK12) wleg WK12 $FIXIMG "$FIXOPT" "$(sl 0.12) $KO" ;;
    WK05) wleg WK05 $FIXIMG "$FIXOPT" "$(sl 0.05) $KO" ;;
    W08) wleg W08 $NEW "" "$(sl 0.08)" ;;
    WH) wleg WH $OLD "" "LLAMA_PRIORITY_HOLD_PROMPTS=1" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC5-DONE gate=$GATE $(date -u +%T)"
