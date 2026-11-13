#!/bin/bash
# GH200 window nc10 (2026-11-13): what argo-things #177 still lacks before it can merge. Config B = the flags of #177
# (prompts in pieces at 4 layers and share 0.05, hold rule off, 0071, 0072, 0074) on the CI image FIX_TAG (stack through
# 0076, the image production runs since #179), and for the first time image requests: production serves the poster
# pipeline (--mmproj), and no window with prompts in pieces ever sent an image. Legs:
#  KV : the detector gate (pieces config, client classes, LLAMA_RS_LOG/LLAMA_RS_SUM). v1digest.py decides; a failure
#       stops the legs after it.
#  VB : config B, image requests (vis10.py): each alone twice, then next to generating clients, while Juno starts,
#       while Juno generates, next to a long prompt being read, three images at once; each alone again.
#  VA : config A (today's production: hold rule on, no pieces), the same requests: the control.
#  KB1: config B, acceptance (the FULL sequence of nc4-nc8), one server process.
#  XK : config B, the class scenarios of nc7, plus "mixed" with a 1024-token answer for Juno twice.
#  VL : config B plus LLAMA_PREFILL_SLICE_LOG=1 (the log shows whether an image prompt is read in pieces), a short
#       subset of VB.
#  ST : staggered pairs (stag.py).
# Test keys generated here, never printed, deleted.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
FIXIMG=ghcr.io/xerootg/llamacpp-spark:${FIX_TAG:?}
KO="LLAMA_KPOOL_OWN_SEQS=1"; SF="LLAMA_PREFILL_SHORT_FIRST=1 LLAMA_PREFILL_SLICE_SAME_FREE=512"; H="LLAMA_PRIORITY_HOLD_PROMPTS=1"
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
B="$(sl 0.05) $KO $SF"
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
ORD=("prio_nc.py mixed --classes $K" "prio_nc.py mixed --classes $K" "prio_nc.py shortbehind --classes $K --delay 5"
     "prio_nc.py samelong --classes $K --jn 512 --delay 2" "prio_nc.py twolong --classes $K")
SOLO=$(for s in loc-2652 loc-7d73 loc-eaa1 loc-fb48 read-2652 read-7d73 read-eaa1 read-fb48 full-eaa1; do printf '0:hef:%s ' $s; done)
W="vis10.py"; C="--classes $K"
vleg() {  # TAG ENV: image requests alone, next to other requests, alone again
  mkkeys
  IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh $1 5 "$P $GR $F $CL $2" \
    "$W solo $C --seq --rounds 2 $SOLO" \
    "$W g3-loc $C 0:wb0:p1 0:wb1:p2 0:cc:p3 1:hef:loc-2652" \
    "$W g3-read $C 0:wb0:p1 0:wb1:p2 0:cc:p3 1:hef:read-7d73" \
    "$W v-juno $C 0:hef:full-eaa1 1:juno:p0:512" \
    "$W v-juno2 $C 0:hef:read-fb48 0.3:juno:p0:256" \
    "$W juno-v $C 0:juno:p0:1024 2:hef:loc-7d73 3:cc:read-2652 9:hef:loc-eaa1" \
    "$W long-v $C 0:wb0:L65536 2:hef:loc-fb48 5:cc:read-7d73 8:hef:full-eaa1 12:wb1:p1" \
    "$W v-long $C 0:hef:full-eaa1 0.5:wb0:L16384 1:wb1:p2 1.5:juno:p0:256" \
    "$W mixed $C 0:wb0:p1 0:wb1:p2 0.5:cc:L16384 1:hef:read-2652 3:juno:p0:512" \
    "$W solo2 $C --seq $SOLO" \
    "$W v3 $C 0:hef:full-2652 0:cc:read-eaa1 0.5:wb0:loc-fb48" \
    "$W solo3 $C --seq 0:hef:loc-2652 0:hef:read-7d73 0:hef:full-2652 0:juno:p0:256"
  leakcheck $1; rm -f $K
}
GATE=ok
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in K*|X*|ST|V*) [ $GATE = ok ] || { echo "GATE failed, leg $leg skipped"; continue; } ;; esac
  case $leg in
    KV) D="--dump $G/legs/dump-KV10"; mkkeys
        IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh KV10 5 "$P $GR $F $CL $B LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K $D" "$E sololong --classes $K --long L16384 $D" "$E sololong --classes $K $D" \
          "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" \
          "$E mixed --classes $K $D" "$E stagger --classes $K $D" "$E mixed --classes $K $D" \
          "$E junolong --classes $K --jn 512 --delay 1 $D" "$E stagger --classes $K --long L16384 $D"
        leakcheck KV10; rm -f $K
        python3 -I $G/v1digest.py KV10 || GATE=fail ;;
    VB) vleg VB10 "$B" ;;
    VA) vleg VA10 "$H $KO" ;;
    VL) mkkeys
        IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh VL10 5 "$P $GR $F $CL $B LLAMA_PREFILL_SLICE_LOG=1" \
          "$W solo $C --seq 0:hef:loc-2652 0:hef:read-7d73 0:hef:full-eaa1" \
          "$W g3-read $C 0:wb0:p1 0:wb1:p2 0:cc:p3 1:hef:read-7d73" \
          "$W v-juno $C 0:hef:full-eaa1 1:juno:p0:512" \
          "$W long-v $C 0:wb0:L65536 2:hef:loc-2652 5:cc:read-7d73 8:hef:full-eaa1 12:wb1:p1"
        leakcheck VL10; rm -f $K ;;
    ST) IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh ST10 5 "$P $GR $F $B LLAMA_PREFILL_SLICE_LOG=1" \
          "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 3" "stag.py p0 p1 --delays 1.0 0.01 --reps 3" "stag.py p2 p0 --delays 2.0 --reps 2 --n-predict 256" \
          "stag.py L16384 p0 --delays 0.5 2.0 --reps 2" "stag.py p0 L16384 --delays 0.3 --reps 2 --n-predict 128" "stag.py L16384 p1 --delays 0.2 --reps 2" ;;
    KB1) IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh KB110 5 "$P $GR $F $B" "${FULL[@]}" ;;
    XK) mkkeys
        IMG=$FIXIMG CACHE_RAM=65536 bash nc-leg.sh XK10 5 "$P $GR $F $CL $B" \
          "prio_nc.py warm --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py warm --classes $K --jn 1024" \
          "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
          "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "${ORD[@]}" \
          "prio_nc.py mixed --classes $K --jn 1024" "prio_nc.py mixed --classes $K --jn 1024" "prio_nc.py warm --classes $K"
        leakcheck XK10; rm -f $K ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC10-DONE gate=$GATE $(date -u +%T)"
