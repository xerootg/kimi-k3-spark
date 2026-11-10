#!/bin/bash
# Spark windows nc-s5 and nc-s6 (2026-11-10): the binaries of the CI image 547208dca250-gfbb8831 (the stack through
# patch 0076; /opt/llama of the image copied to ~/nc/ci-fbb8831/bin) with the config proposed for the Spark.
# glm53-spark at 0, bare servers on 127.0.0.1:18080 under window.lock. FIN as in spark-nc8.sh.
# Patch 0076 keeps the k-pool marks (4 bytes per cell of the whole cache) in the memory object instead of allocating
# and zeroing them in every decode step; window nc-s4 found five 1M slots 0.3-0.5 % under five 200k slots.
#  SF0, SP0, SO0: one and the same short command sequence on FIN with the new binaries, on today's production config
#       (five 200k slots, no floor, no pieces) with the new binaries, and on FIN with the binaries of g0fa9510
#  SV9: the detector gate; v1digest.py decides, a failure stops the rest
#  SA0, SA1: acceptance, two server processes: per-client md5 equal to solo and gold at N=1..5 with the 16k mixes
#  SP1, SF1, SQ1: the short sequence again in the other order; SQ1 = the production config on the binaries of g0fa9510
#  FB9: full scale: a 500k session read first, four ~190k clients arriving during it, bytes of all five alone and
#       together, a refused growth with four floors resident that leaves the session cached, release
# Env DEADLINE (epoch seconds): no leg and no client command starts after it. Test keys generated here, never printed,
# deleted.
set -u
cd /srv/glm-5.3-flash; export BD=${BD:-$HOME/nc/ci-fbb8831}; OLD=$HOME/nc/ci-0fa9510
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SL="LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_SHARE=0.05 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1 LLAMA_PRIORITY_HOLD_PROMPTS=0"
T="LLAMA_ELASTIC_KV_FLOOR_CELLS=204800 LLAMA_PREFILL_SLICE_TG_CELLS=204800 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1"
FIN="$P $SL $T LLAMA_PREFILL_SLICE_SAME_FREE=512"
OUT=legs-nc2; mkdir -p $OUT; echo $$ > $OUT/runner.pid; SD="--state-dir /srv/glm-5.3-flash/$OUT"
KD=$PWD/keys-nc; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1"
ACC=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py solo L16384"
     "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2"
     "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1"
     "ekbench.py solo p0 --n-predict 128")
SHORT=("ekbench.py solo p0" "ekbench.py multi p0 p1 --rounds 3" "ekbench.py multi p0 p1 p2 --rounds 2")
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
    SF0) bash nc2-leg.sh SF0 5 5242880 "$FIN" "${SHORT[@]}" ;;
    SP0) bash nc2-leg.sh SP0 5 1024000 "$P" "${SHORT[@]}" ;;
    SO0) BD=$OLD bash nc2-leg.sh SO0 5 5242880 "$FIN" "${SHORT[@]}" ;;
    SV9) mkkeys; E="prio_spark_e.py"; KA="--classes $K --long L16384"
        bash nc2-leg.sh SV9 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K" "$E sololong $KA" "$E mixed $KA" "$E stagger $KA" "$E mixed $KA" "$E junolong $KA --jn 512 --delay 1"
        leakcheck SV9; rm -f $K
        python3 -I v1digest.py SV9 /srv/glm-5.3-flash/$OUT spark || GATE=fail ;;
    SA0) bash nc2-leg.sh SA0 5 5242880 "$FIN" "${ACC[@]}" "ekbench.py solo p1 --n-predict 512" ;;
    SA1) bash nc2-leg.sh SA1 5 5242880 "$FIN" "${ACC[@]}" ;;
    SP1) bash nc2-leg.sh SP1 5 1024000 "$P" "${SHORT[@]}" ;;
    SF1) bash nc2-leg.sh SF1 5 5242880 "$FIN" "${SHORT[@]}" ;;
    SQ1) BD=$OLD bash nc2-leg.sh SQ1 5 1024000 "$P" "${SHORT[@]}" ;;
    FB9) BT="--slot-of B+T"
        bash nc2-leg.sh FB9 5 5242880 "$FIN LLAMA_SLOT_KEEP_CHILD=1" \
          "ekbench.py solo p0" "fb3.py bfirst 500000 4 60 420 $SD" "fb3.py bsolo 500000 $SD" "fb3.py conc 4 500000 $SD" "fb3.py floor 4 $SD" \
          "fb3.py grow ${GROW:-1000000} 60 $BT $SD" "fb3.py bsolo 500000 $BT $SD" "fb3.py release $BT $SD" "fb3.py floorc 4 2 $SD" "ekbench.py solo p0" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $OUT/runner.pid
echo "SPARK-NC9-DONE gate=$GATE $(date -u +%T)"
