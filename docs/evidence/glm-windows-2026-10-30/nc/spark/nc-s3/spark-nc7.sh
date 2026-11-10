#!/bin/bash
# Spark window nc-s3 (2026-11-10): acceptance of the binaries of the CI image 547208dca250-gb5ee55a (the stack through
# patch 0074; /opt/llama of the image copied to ~/nc/ci-b5ee55a/bin, the server library byte-identical to the image's)
# with the config proposed for the Spark. glm53-spark at 0, bare servers on 127.0.0.1:18080 under window.lock.
# FIN = five slots of 1M (--ctx-size 5242880), a guaranteed floor of 204800 cells per slot with bursts past it, prompts
# in pieces of 2 layers at share 0.05 with the hold rule off, LLAMA_KPOOL_OWN_SEQS=1 (0071), LLAMA_PREFILL_SHORT_FIRST=1
# (0072), LLAMA_PREFILL_SLICE_SAME_FREE=512 (0074), production's column split and batch-invariant flags.
#  SV7: the detector gate (client classes, LLAMA_RS_LOG/LLAMA_RS_SUM); v1digest.py decides, a failure stops the rest
#  SA6, SA7: acceptance, two server processes: per-client md5 equal to solo and gold at N=1..5 with the 16k mixes
#  SP7: today's production config (five 200k slots, no floor, no pieces) on the same binaries: the "before"
#  SQ7: FIN without LLAMA_PREFILL_SLICE_SAME_FREE, the multi commands (what 0074 changes on this box)
#  RF7: staggered pairs (the repro of the abort that patch 0073 fixed)
#  SD7: FIN with client classes: Juno's decode while others' prompts are read in pieces, and the order scenarios
#  SS7: the order scenarios with LLAMA_PREFILL_SLICE_SHARE_SAME=0.25 (the untested knob of window nc-s2)
#  G7:  floor + burst mechanics at one tenth scale (see spark-nc4.sh G1)
#  FB7: full scale: a 500k session read first, four ~190k clients arriving during it, bytes of all five alone and
#       together, a refused growth with four floors resident, release, valve re-arm under load
# Env DEADLINE (epoch seconds): no leg and no client command starts after it. Test keys generated here, never printed,
# deleted.
set -u
cd /srv/glm-5.3-flash; export BD=${BD:-$HOME/nc/ci-b5ee55a}
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_SHARE=${SHARE:-0.05} LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
T="LLAMA_ELASTIC_KV_FLOOR_CELLS=204800 LLAMA_PREFILL_SLICE_TG_CELLS=204800 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1"
FIN="$P $SL $T LLAMA_PREFILL_SLICE_SAME_FREE=512"
OUT=legs-nc2; mkdir -p $OUT; echo $$ > $OUT/runner.pid; SD="--state-dir /srv/glm-5.3-flash/$OUT"
KD=$PWD/keys-nc; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1"
ACC=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py solo L16384"
     "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2"
     "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1"
     "ekbench.py solo p0 --n-predict 128")
PRIO=("prio_spark.py warm --classes $K --jn 512" "prio_spark.py junolong --classes $K --jn 512 --long L16384 --delay 1"
      "prio_spark.py prefill --classes $K --long L16384 --delay 10" "prio_spark.py junofirst --classes $K"
      "prio_spark.py mixed --classes $K --long L16384" "prio_spark.py samelong --classes $K --jn 512 --long L16384 --delay 2"
      "prio_spark.py shortbehind --classes $K --long L16384 --delay 5" "prio_spark.py nokey --classes $K")
PRIOB=("prio_spark.py mixed --classes $K --long L16384" "prio_spark.py samelong --classes $K --jn 512 --long L16384 --delay 2"
       "prio_spark.py shortbehind --classes $K --long L16384 --delay 5")
mkkeys() { mkdir -p $KD; chmod 0700 $KD; ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$OUT/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
GATE=ok
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  [ $GATE = ok ] || { echo "GATE failed, leg $leg skipped"; continue; }
  case $leg in
    SV7) mkkeys; E="prio_spark_e.py"; KA="--classes $K --long L16384"
        bash nc2-leg.sh SV7 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K" "$E sololong $KA" "$E mixed $KA" "$E stagger $KA" "$E mixed $KA" "$E junolong $KA --jn 512 --delay 1"
        leakcheck SV7; rm -f $K
        python3 -I v1digest.py SV7 /srv/glm-5.3-flash/$OUT spark || GATE=fail ;;
    SA6|SA7) bash nc2-leg.sh $leg 5 5242880 "$FIN" "${ACC[@]}" ;;
    SP7) bash nc2-leg.sh SP7 5 1024000 "$P" "ekbench.py solo p0" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2" \
          "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" ;;
    SQ7) bash nc2-leg.sh SQ7 5 5242880 "$P $SL $T" "ekbench.py solo p0" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2" \
          "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" ;;
    RF7) bash nc2-leg.sh RF7 5 5242880 "$FIN LLAMA_PREFILL_SLICE_LOG=1" "stag.py p1 p0 --delays 1.0 0.01 0.3 --reps 4" "stag.py p0 p1 --delays 1.0 0.01 --reps 3" \
          "stag.py p2 p0 --delays 2.0 --reps 2 --n-predict 256" ;;
    SD7) mkkeys; bash nc2-leg.sh SD7 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck SD7; rm -f $K ;;
    SS7) mkkeys; bash nc2-leg.sh SS7 5 5242880 "${FIN/SHARE_SAME=0.5/SHARE_SAME=0.25} $CL" "prio_spark.py warm --classes $K --jn 512" "${PRIOB[@]}"; leakcheck SS7; rm -f $K ;;
    G7) bash nc2-leg.sh G7 5 5242880 "$P $SL LLAMA_ELASTIC_KV_FLOOR_CELLS=20480 LLAMA_PREFILL_SLICE_TG_CELLS=20480 LLAMA_ELASTIC_KV_FLOOR_MIB=30000 LLAMA_SLOT_KEEP_CHILD=1 LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1 LLAMA_PREFILL_SLICE_SAME_FREE=512" \
          "ekbench.py solo p0" "fb2.py bfirst 80000 4 10 30 --ptok 19000 $SD" "fb2.py floor 4 --ptok 19000 $SD" "fb2.py bsolo 80000 $SD" \
          "fb2.py conc 4 80000 --ptok 19000 $SD" "fb2.py grow 1000000 30 --slot-of B+T $SD" "fb2.py bsolo 80000 $SD" \
          "fb2.py grow 120000 20 --slot-of B+T $SD" "fb2.py bsolo 80000 $SD" "fb2.py newcomer 15000 --slot-of P+S2 $SD" \
          "fb2.py release --slot-of B+T $SD" "fb2.py floorc 4 2 --ptok 19000 $SD" "ekbench.py solo p0" ;;
    FB7) bash nc2-leg.sh FB7 5 5242880 "$FIN LLAMA_SLOT_KEEP_CHILD=1" \
          "ekbench.py solo p0" "fb2.py bfirst 500000 4 60 420 $SD" "fb2.py bsolo 500000 $SD" "fb2.py conc 4 500000 $SD" "fb2.py floor 4 $SD" \
          "fb2.py grow ${GROW:-1000000} 60 --slot-of B+T $SD" "fb2.py bsolo 500000 $SD" "fb2.py release --slot-of B+T $SD" "fb2.py floorc 4 2 $SD" "ekbench.py solo p0" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $OUT/runner.pid
echo "SPARK-NC7-DONE gate=$GATE $(date -u +%T)"
