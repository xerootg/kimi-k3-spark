#!/bin/bash
# Spark window nc-s2, second runner (2026-11-10): the legs of spark-nc4.sh that had not started, reordered, on binary
# ~/nc/a121h = a121g with patch 0072 in the server (LLAMA_PREFILL_SHORT_FIRST=1: a prompt whose rest fits in one ubatch
# goes first). Leg SV1 of spark-nc4.sh showed the tail ubatch of a short prompt waiting behind the first ubatch of a
# long prompt that arrived after it (first tokens after 7.6-19.3 s in "mixed"); SV1 and SA1 ran without 0072.
#  M1:  admission only, full scale, fresh server: the largest burst admitted next to four full floors
#  SV2: the detector gate of SV1 on FIN + 0072; v1digest.py decides, a failure stops the legs after it
#  SD2: FIN + 0072 with client classes: Juno's decode while others' prompts are read in pieces, and the order scenarios
#  SB1: the order scenarios on today's production config (hold rule on, 200k slots)
#  SA3: acceptance on FIN + 0072 (per-client md5 equal to solo and gold at N=1..5 with a 16k mix, two rounds; speeds)
#  SP0: today's production config on this build (throughput baseline)
#  G1, FB: as in spark-nc4.sh (window B)
set -u
cd /srv/glm-5.3-flash; export BD=${BD:-$HOME/nc/a121h}
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_SHARE=${SHARE:-0.05} LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
FIN="$P $SL LLAMA_ELASTIC_KV_FLOOR_CELLS=204800 LLAMA_PREFILL_SLICE_TG_CELLS=204800 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1"
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
    G1) bash nc2-leg.sh G1 5 5242880 "$P $SL LLAMA_ELASTIC_KV_FLOOR_CELLS=20480 LLAMA_PREFILL_SLICE_TG_CELLS=20480 LLAMA_ELASTIC_KV_FLOOR_MIB=30000 LLAMA_SLOT_KEEP_CHILD=1 LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1" \
          "ekbench.py solo p0" "fb2.py bfirst 80000 4 10 30 --ptok 19000 $SD" "fb2.py floor 4 --ptok 19000 $SD" "fb2.py bsolo 80000 $SD" \
          "fb2.py conc 4 80000 --ptok 19000 $SD" "fb2.py grow 1000000 30 --slot-of B+T $SD" "fb2.py bsolo 80000 $SD" \
          "fb2.py grow 120000 20 --slot-of B+T $SD" "fb2.py bsolo 80000 $SD" "fb2.py newcomer 15000 --slot-of P+S2 $SD" \
          "fb2.py release --slot-of B+T $SD" "fb2.py floorc 4 2 --ptok 19000 $SD" "ekbench.py solo p0" ;;
    M1) bash nc2-leg.sh M1 5 5242880 "$FIN" "ekbench.py solo p0" "fb2.py probe 30 1000000 900000 800000 700000 600000 $SD" "ekbench.py solo p0" ;;
    M2) bash nc2-leg.sh M2 5 5242880 "$FIN LLAMA_EXPERT_STORE_VALVE_FLOOR=${VF:-70}" "ekbench.py solo p0" "fb2.py probe 30 1000000 900000 800000 700000 $SD" "ekbench.py solo p0" ;;
    SV2) mkkeys; E="prio_spark_e.py"; KA="--classes $K --long L16384"
        bash nc2-leg.sh SV2 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K" "$E sololong $KA" "$E mixed $KA" "$E stagger $KA" "$E mixed $KA" "$E junolong $KA --jn 512 --delay 1"
        leakcheck SV2; rm -f $K
        python3 -I v1digest.py SV2 /srv/glm-5.3-flash/$OUT spark || GATE=fail ;;
    SA3|SA4) bash nc2-leg.sh $leg 5 5242880 "$FIN" "${ACC[@]}" ;;
    SD2) mkkeys; bash nc2-leg.sh SD2 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck SD2; rm -f $K ;;
    SP0) bash nc2-leg.sh SP0 5 1024000 "$P" "ekbench.py solo p0" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2" \
          "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" ;;
    SB1) mkkeys; bash nc2-leg.sh SB1 5 1024000 "$P $CL LLAMA_PRIORITY_HOLD_PROMPTS=1" "${PRIOB[@]}"; leakcheck SB1; rm -f $K ;;
    FB) bash nc2-leg.sh FB 5 5242880 "$FIN LLAMA_SLOT_KEEP_CHILD=1" \
          "ekbench.py solo p0" "fb2.py bfirst 500000 4 60 420 $SD" "fb2.py bsolo 500000 $SD" "fb2.py conc 4 500000 $SD" "fb2.py floor 4 $SD" \
          "fb2.py grow ${GROW:-1000000} 60 --slot-of B+T $SD" "fb2.py bsolo 500000 $SD" "fb2.py release --slot-of B+T $SD" "fb2.py floorc 4 2 $SD" "ekbench.py solo p0" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $OUT/runner.pid
echo "SPARK-NC5-DONE gate=$GATE $(date -u +%T)"
