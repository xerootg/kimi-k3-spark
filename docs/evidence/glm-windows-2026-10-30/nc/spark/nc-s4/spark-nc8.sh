#!/bin/bash
# Spark windows nc-s4 and nc-s5 (2026-11-10): the binaries of the CI image 547208dca250-g0fa9510 (the stack through
# patch 0075; /opt/llama of the image copied to ~/nc/ci-0fa9510/bin, the server library byte-identical to the image's)
# with the config proposed for the Spark. glm53-spark at 0, bare servers on 127.0.0.1:18080 under window.lock.
# FIN = five slots of 1M (--ctx-size 5242880), a guaranteed floor of 204800 cells per slot with bursts past it, prompts
# in pieces of 2 layers at share 0.05 with the hold rule off, LLAMA_KPOOL_OWN_SEQS=1 (0071), LLAMA_PREFILL_SHORT_FIRST=1
# (0072), LLAMA_PREFILL_SLICE_SAME_FREE=512 (0074), production's column split and batch-invariant flags.
#  G8:  floor + burst mechanics at one tenth scale, the sequence of G7 (window nc-s3) with the burst's own requests
#       pinned to its slot: after the refused growth to 1M the conversation must still be cached (patch 0075: cache_n
#       of the next request about 79.9k instead of 0)
#  SV8: the detector gate (client classes, LLAMA_RS_LOG/LLAMA_RS_SUM); v1digest.py decides, a failure stops the rest
#  SA8, SA9: acceptance, two server processes: per-client md5 equal to solo and gold at N=1..5 with the 16k mixes;
#       SA8 ends with p1 at 512 tokens alone (the reference for "samelong")
#  SF8, SP8: FIN and today's production config (five 200k slots, no floor, no pieces) with one and the same short
#       command sequence: is the two-client difference of window nc-s3 the config or the 16k prompt before the rounds?
#  SP9, SF9, SH9, SI9 (added after SF8/SP8 showed FIN 0.5 % under the production config at two and three clients, one
#       process each): the pair again in the other order, then the halves: SH9 = production plus five 1M slots with
#       the floor, SI9 = production's five 200k slots plus the pieces flags
#  SJ9, SK9 (added after SH9 = SF9 and SI9 = SP9): SH9's two changes apart: SJ9 = production with five 1M slots and no
#       floor, SK9 = production's five 200k slots with the floor variable set (the floor is then the whole slot)
#  FB8: full scale: a 500k session read first, four ~190k clients arriving during it, bytes of all five alone and
#       together, a refused growth with four floors resident that leaves the session cached, release
# Env DEADLINE (epoch seconds): no leg and no client command starts after it. Test keys generated here, never printed,
# deleted.
set -u
cd /srv/glm-5.3-flash; export BD=${BD:-$HOME/nc/ci-0fa9510}
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
    G8) BT="--slot-of B+T"
        bash nc2-leg.sh G8 5 5242880 "$P $SL LLAMA_ELASTIC_KV_FLOOR_CELLS=20480 LLAMA_PREFILL_SLICE_TG_CELLS=20480 LLAMA_ELASTIC_KV_FLOOR_MIB=30000 LLAMA_SLOT_KEEP_CHILD=1 LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1 LLAMA_PREFILL_SLICE_SAME_FREE=512" \
          "ekbench.py solo p0" "fb3.py bfirst 80000 4 10 30 --ptok 19000 $SD" "fb3.py floor 4 --ptok 19000 $SD" "fb3.py bsolo 80000 $SD" \
          "fb3.py conc 4 80000 --ptok 19000 $SD" "fb3.py grow 1000000 30 $BT $SD" "fb3.py bsolo 80000 $BT $SD" \
          "fb3.py grow 120000 20 $BT $SD" "fb3.py bsolo 80000 $BT $SD" "fb3.py newcomer 15000 --slot-of P+S2 $SD" \
          "fb3.py release $BT $SD" "fb3.py floorc 4 2 --ptok 19000 $SD" "ekbench.py solo p0" ;;
    SV8) mkkeys; E="prio_spark_e.py"; KA="--classes $K --long L16384"
        bash nc2-leg.sh SV8 5 5242880 "$FIN $CL LLAMA_PREFILL_SLICE_LOG=1 LLAMA_RS_LOG=1 LLAMA_RS_SUM=512" \
          "$E warm --classes $K" "$E sololong $KA" "$E mixed $KA" "$E stagger $KA" "$E mixed $KA" "$E junolong $KA --jn 512 --delay 1"
        leakcheck SV8; rm -f $K
        python3 -I v1digest.py SV8 /srv/glm-5.3-flash/$OUT spark || GATE=fail ;;
    SA8) bash nc2-leg.sh SA8 5 5242880 "$FIN" "${ACC[@]}" "ekbench.py solo p1 --n-predict 512" ;;
    SA9) bash nc2-leg.sh SA9 5 5242880 "$FIN" "${ACC[@]}" ;;
    SF8) bash nc2-leg.sh SF8 5 5242880 "$FIN" "${SHORT[@]}" ;;
    SP8) bash nc2-leg.sh SP8 5 1024000 "$P" "${SHORT[@]}" ;;
    SP9) bash nc2-leg.sh SP9 5 1024000 "$P" "${SHORT[@]}" ;;
    SF9) bash nc2-leg.sh SF9 5 5242880 "$FIN" "${SHORT[@]}" ;;
    SH9) bash nc2-leg.sh SH9 5 5242880 "$P LLAMA_ELASTIC_KV_FLOOR_CELLS=204800" "${SHORT[@]}" ;;
    SI9) bash nc2-leg.sh SI9 5 1024000 "$P $SL LLAMA_PREFILL_SLICE_TG_CELLS=204800 LLAMA_KPOOL_OWN_SEQS=1 LLAMA_PREFILL_SHORT_FIRST=1 LLAMA_PREFILL_SLICE_SAME_FREE=512" "${SHORT[@]}" ;;
    SJ9) bash nc2-leg.sh SJ9 5 5242880 "$P" "${SHORT[@]}" ;;
    SK9) bash nc2-leg.sh SK9 5 1024000 "$P LLAMA_ELASTIC_KV_FLOOR_CELLS=204800" "${SHORT[@]}" ;;
    FB8) BT="--slot-of B+T"
        bash nc2-leg.sh FB8 5 5242880 "$FIN LLAMA_SLOT_KEEP_CHILD=1" \
          "ekbench.py solo p0" "fb3.py bfirst 500000 4 60 420 $SD" "fb3.py bsolo 500000 $SD" "fb3.py conc 4 500000 $SD" "fb3.py floor 4 $SD" \
          "fb3.py grow ${GROW:-1000000} 60 $BT $SD" "fb3.py bsolo 500000 $BT $SD" "fb3.py release $BT $SD" "fb3.py floorc 4 2 $SD" "ekbench.py solo p0" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $OUT/runner.pid
echo "SPARK-NC8-DONE gate=$GATE $(date -u +%T)"
