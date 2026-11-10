#!/bin/bash
# GH200 window nc4 (2026-11-10): patch 0069 (prompt pieces keep the recurrent cell in place; the lower classes' share
# split between prompts and answers) on the CI image NEW_TAG (kimi-k3-spark c93c6d9, patches through 0069), production
# config of #174 otherwise.
#  V1: pieces config (4 layers, share 0.12, decode share, same-class share 0.5, turns, hold rule off), Juno's key at 10
#      (test keys generated here, never printed, deleted), with the detectors LLAMA_RS_LOG=1 LLAMA_RS_SUM=512: the long
#      prompts alone, then "mixed" with 16k (three times) and 64k (twice), "stagger", junolong, prefill. v1digest.py is
#      the gate: any md5 not gold, any state hash that differs, any range containing the frozen cell stops the window.
#  V2: the same config without the detectors: speeds (D1's scenario list of window nc3, ttft4 first and mixed again last).
#  A1, A2: acceptance, two server processes, no client classes: per-client md5 at N=1..5 with 16k/64k/100k mixes, two
#      rounds; throughput against #174.
set -u
G=/srv/glm-5.3-flash/nc; cd $G; mkdir -p keys; chmod 0700 keys
echo $$ > $G/nc3.pid   # started with setsid: this pid is the process group the deadman stops
NEW=ghcr.io/xerootg/llamacpp-spark:${NEW_TAG:?}
P="LLAMA_PREFILL_ISOLATE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1"
GR="LLAMA_KV_HOST=1 LLAMA_KV_HOST_THP=1 GGML_CUDA_KV_STAGE=1"
F="GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1 GGML_CUDA_FUSE_CONV_STATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE=0.12 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
CL="LLAMA_CLIENT_CLASSES=/keys/classes.txt LLAMA_SLOT_RESERVE=1"
K=$G/keys/classes.txt
E="prio_nc_e.py"
mkkeys() { ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); chmod 0644 $K; }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$G/legs/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
FULL=("gold.py 2" "ekbench.py solo p0 p1 p2 p3 p4 L16384" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
      "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p0 L16384 --rounds 1"
      "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 2" "ekbench.py multi p0 p1 p2 p3 L65536 --rounds 1" "ekbench.py multi p1 p2 L16384 L65536 --rounds 1" "ekbench.py solo p0 L16384 L65536"
      "ekbench.py solo p0 p0 p0 p0" "ekbench.py solo L16384 L16384 L65536 L65536 --cache" "bench.py depth --lens 100000 --n-predict 64"
      "ekbench.py solo L100000" "ekbench.py multi p0 p1 p2 L100000 --rounds 1" "gold.py 1")
GATE=ok
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  [ $GATE = ok ] || { echo "GATE failed, leg $leg skipped"; continue; }
  case $leg in
    V1*) D="--dump $G/legs/dump-$leg"; mkkeys
        IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K $D" "$E sololong --classes $K --long L16384 $D" "$E sololong --classes $K $D" \
          "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" "$E mixed --classes $K --long L16384 $D" \
          "$E mixed --classes $K $D" "$E stagger --classes $K $D" "$E mixed --classes $K $D" \
          "$E junolong --classes $K --jn 512 --delay 1 $D" "$E prefill --classes $K --delay 10 $D" "$E stagger --classes $K --long L16384 $D"
        leakcheck $leg; rm -f $K
        python3 -I $G/v1digest.py $leg || GATE=fail ;;
    V2*) mkkeys
        IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh $leg 5 "$P $GR $F $CL $SL LLAMA_PREFILL_SLICE_LOG=1" \
          "prio_nc.py ttft4 --classes $K" "prio_nc.py warm --classes $K --jn 512" "prio_nc.py junolong --classes $K --jn 512 --delay 1" "prio_nc.py prefill --classes $K --delay 10" \
          "prio_nc.py junofirst --classes $K" "prio_nc.py junofirst --classes $K --jn 1024" "prio_nc.py ttft4 --classes $K" "prio_nc.py mixed --classes $K" \
          "prio_nc.py samelong --classes $K --jn 1024 --delay 2" "prio_nc.py same3long --classes $K --jn 1024 --delay 2" "prio_nc.py shortbehind --classes $K --delay 5" \
          "prio_nc.py twolong --classes $K --delay 5" "prio_nc.py mixed --classes $K" "prio_nc.py nokey --classes $K"
        leakcheck $leg; rm -f $K ;;
    A1) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A1t 5 "$P $GR $F $SL" "${FULL[@]}" ;;
    A2) IMG=$NEW CACHE_RAM=65536 bash nc-leg.sh A2t 5 "$P $GR $F $SL" "${FULL[@]}" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $G/nc3.pid
echo "NC4-DONE gate=$GATE $(date -u +%T)"
