#!/bin/bash
# Spark window nc-s7 (2026-11-11): config A' = today's production flags (five slots of 200k, --ctx-size 1024000, the
# column split from three clients, client classes with the hold rule) plus LLAMA_KPOOL_OWN_SEQS=1 (patch 0071) on the
# binaries of the CI image 547208dca250-gfbb8831 (~/nc/ci-fbb8831/bin; patch 0076 needs no switch), next to a control
# on the binaries of today's production image 547208dca250-g08d176f (~/nc/ci-08d176f/bin) with today's flags.
# No prompt pieces, no floor, no burst in any leg. glm53-spark at 0, bare servers on 127.0.0.1:18080 under window.lock.
#  TA0, TA1: A', acceptance, two server processes: per-client md5 equal to solo and gold at N=1..5 with the 16k mixes
#  QC0:      the control, the same command sequence, between the two
#  WQ0, WT0: client classes (Juno priority 10, four others priority 1, one slot reserved, hold rule): Juno alone, a
#            16k prompt arriving in her answer, three generating clients + a 16k prompt + Juno, four clients arriving
#            in her answer; the control first, then A'
# Env DEADLINE (epoch seconds): no leg and no client command starts after it. Test keys generated here, never printed,
# deleted.
set -u
cd /srv/glm-5.3-flash; NEW=$HOME/nc/ci-fbb8831; OLD=$HOME/nc/ci-08d176f
P="LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3 LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
H="LLAMA_PRIORITY_HOLD_PROMPTS=1"; KO="LLAMA_KPOOL_OWN_SEQS=1"
OUT=legs-nc2; mkdir -p $OUT; echo $$ > $OUT/runner.pid
KD=$PWD/keys-nc; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1"
ACC=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py solo L16384"
     "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2" "ekbench.py multi p0 p1 p2 p3 --rounds 2"
     "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi p1 p2 p3 p4 L16384 --rounds 1" "ekbench.py multi p0 L16384 --rounds 1"
     "ekbench.py solo p0 --n-predict 128")
mkkeys() { mkdir -p $KD; chmod 0700 $KD; ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('$OUT/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
wleg() { # wleg TAG BD "ENV"
  mkkeys; local E="prio_spark.py" KA="--classes $K --long L16384"
  BD=$2 bash nc2-leg.sh $1 5 1024000 "$3 $CL" "$E warm --classes $K" "$E warm --classes $K" "$E junolong $KA" "$E mixed $KA" "$E junofirst --classes $K" "$E warm --classes $K"
  leakcheck $1; rm -f $K; }
for leg in "$@"; do
  [ -n "${DEADLINE:-}" ] && [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "DEADLINE passed, leg $leg skipped"; continue; }
  case $leg in
    TA0|TA1) BD=$NEW bash nc2-leg.sh $leg 5 1024000 "$P $H $KO" "${ACC[@]}" ;;
    QC0) BD=$OLD bash nc2-leg.sh QC0 5 1024000 "$P $H" "${ACC[@]}" ;;
    WQ0) wleg WQ0 $OLD "$P $H" ;;
    WT0) wleg WT0 $NEW "$P $H $KO" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K $OUT/runner.pid
echo "SPARK-NC10-DONE $(date -u +%T)"
