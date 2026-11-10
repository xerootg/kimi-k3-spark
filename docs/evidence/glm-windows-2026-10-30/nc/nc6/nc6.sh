#!/bin/bash
# GH200 window nc6 (2026-11-10): the pieces with patches 0071 (LLAMA_KPOOL_OWN_SEQS=1), 0072
# (LLAMA_PREFILL_SHORT_FIRST=1: a prompt whose rest fits in one ubatch goes first) and 0073 (a prompt in pieces stops
# at a scheduler split boundary only where the caller allows it; fix to 0061, found in Spark window nc-s2) on the CI
# image FIX_TAG (the whole stack; 0070 is in it and off unless set). Window nc5 accepted the pieces with 0071 on image
# gadd9f81 and showed the cost of the turn order: in "mixed" the tail ubatch of each short prompt waited behind the
# first ubatch of the 64k prompt (first tokens after 3.1-3.3 s against 0.4-1.0 s on production). 0073 changes where a
# prompt may be interrupted on this box too (short ubatches were cut once, at the graph's split boundary). Legs:
#  KV : the detector gate (pieces config, client classes, LLAMA_RS_LOG/LLAMA_RS_SUM) with 0071, 0072, 0073.
#       v1digest.py decides; a failure stops the legs after it.
#  ST : staggered pairs (stag.py, the repro of the Spark abort): a second prompt arriving while the first is read or
#       generates, short and long; each answer against its solo bytes.
#  KA1, KA2: acceptance, two server processes (the FULL sequence of nc4/nc5), pieces config, 0071 and 0072 on.
#  PB : the production image and config (hold rule on), the throughput part of FULL: the "before" of KA1/KA2, adjacent.
#  XK : the share and order scenarios at share 0.05 with 0071 and 0072 on: Juno alone (256/512/1024 tokens), junolong,
#       prefill, junofirst (256 and 1024), mixed twice, shortbehind, samelong, twolong.
#  XH : the order scenarios on the PRODUCTION image and config (hold rule on): the "before".
#  XT : the order scenarios (mixed twice, shortbehind, samelong, twolong) with 0071 on and 0072 off, same image
#       (the turns of 0065 alone, what nc5 accepted).
# Test keys generated here, never printed, deleted.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?}
OLD=ghcr.io/xerootg/llamacpp-spark:${OLD_TAG:?}
DEV=${DEV:-/home/ubuntu/nc/out-g90}
if [ -n "${FIX_TAG:-}" ]; then FIXIMG=ghcr.io/xerootg/llamacpp-spark:$FIX_TAG; FIXOPT=""; else FIXIMG=$NEW; FIXOPT=$DEV; fi
KO="LLAMA_KPOOL_OWN_SEQS=1"; SF="LLAMA_PREFILL_SHORT_FIRST=1"
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
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
ORD=("prio_nc.py mixed --classes $K" "prio_nc.py mixed --classes $K" "prio_nc.py shortbehind --classes $K --delay 5"
     "prio_nc.py samelong --classes $K --jn 512 --delay 2" "prio_nc.py twolong --classes $K")
GATE=ok
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in K*|X*|ST) [ $GATE = ok ] || { echo "GATE failed, leg $leg skipped"; continue; } ;; esac
  case $leg in
    KV) D="--dump $G/legs/dump-KV6"; mkkeys
        IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh KV6 5 "$P $GR $F $CL $(sl 0.05) $KO $SF LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K $D" "$E sololong --classes $K --long L16384 $D" "$E sololong --classes $K $D" \
          "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" \
          "$E mixed --classes $K $D" "$E stagger --classes $K $D" "$E mixed --classes $K $D" \
          "$E junolong --classes $K --jn 512 --delay 1 $D" "$E stagger --classes $K --long L16384 $D"
        leakcheck KV6; rm -f $K
        python3 -I $G/v1digest.py KV6 || GATE=fail ;;
    ST) IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh ST 5 "$P $GR $F $(sl 0.05) $KO $SF LLAMA_PREFILL_SLICE_LOG=1" \
          "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 3" "stag.py p0 p1 --delays 1.0 0.01 --reps 3" "stag.py p2 p0 --delays 2.0 --reps 2 --n-predict 256" \
          "stag.py L16384 p0 --delays 0.5 2.0 --reps 2" "stag.py p0 L16384 --delays 0.3 --reps 2 --n-predict 128" "stag.py L16384 p1 --delays 0.2 --reps 2" ;;
    PB) IMG=$OLD CACHE_RAM=65536 bash nc-leg.sh PB6 5 "$P $GR $F LLAMA_PRIORITY_HOLD_PROMPTS=1" "${FULL[@]:0:11}" "gold.py 1" ;;
    KA1|KA2) IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh ${leg}6 5 "$P $GR $F $(sl 0.05) $KO $SF" "${FULL[@]}" ;;
    XK) mkkeys
        IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh XK 5 "$P $GR $F $CL $(sl 0.05) $KO $SF" \
          "prio_nc.py warm --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py warm --classes $K --jn 1024" \
          "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
          "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "${ORD[@]}" "prio_nc.py warm --classes $K"
        leakcheck XK; rm -f $K ;;
    XT) mkkeys
        IMG=$FIXIMG OPT=$FIXOPT CACHE_RAM=65536 bash nc-leg.sh XT 5 "$P $GR $F $CL $(sl 0.05) $KO" "prio_nc.py warm --classes $K" "${ORD[@]}"
        leakcheck XT; rm -f $K ;;
    XH) mkkeys
        IMG=$OLD CACHE_RAM=65536 bash nc-leg.sh XH 5 "$P $GR $F $CL LLAMA_PRIORITY_HOLD_PROMPTS=1" "prio_nc.py warm --classes $K" "${ORD[@]}"
        leakcheck XH; rm -f $K ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC6-DONE gate=$GATE $(date -u +%T)"
