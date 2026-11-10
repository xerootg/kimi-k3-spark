#!/bin/bash
# Spark window nc-s1 (2026-11-09/10). Binary ~/nc/a121 (dev build: CI tree + 0059 + 0060, sm_121). glm53-spark at 0.
#  S0:   patch 0059 on the five-slot production config (5 x 200k)
#  F2:   floor + burst (0060), refusal path: a floor no burst can meet
#  SSL0u / SSL0: prompt logits hash unsliced vs pieces of 2 layers (LLAMA_LOGITS_FNV=1)
#  SPB / SPS / SPSD: hold rule on (production) vs pieces of 2 layers with the hold rule off, and with the lower classes'
#        decode steps inside the share too; test keys (never printed, deleted)
#  F1:   floor + burst with 1M per slot and a 204800-cell floor: a 500k session next to four ~190k clients
cd /srv/glm-5.3-flash; export BD=$HOME/nc/a121
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
KD=$PWD/keys-nc; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1"
PRIO=("prio_spark.py warm --classes $K --jn 512" "prio_spark.py junolong --classes $K --jn 512 --long L16384 --delay 1"
      "prio_spark.py prefill --classes $K --long L16384 --delay 10" "prio_spark.py junofirst --classes $K" "prio_spark.py nokey --classes $K")
mkkeys() { mkdir -p $KD; chmod 0700 $KD; ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('legs-nc/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
for leg in "$@"; do
  case $leg in
    S0) bash nc-leg.sh S0 5 1024000 "$P" "ekbench.py solo p0 p0 p0 p0 p0" "ekbench.py solo p0 --n-predict 128" "ekbench.py multi p0 p1 p2 p3 --rounds 1" "ekbench.py solo L16384" ;;
    F2) bash nc-leg.sh F2 5 5242880 "$P LLAMA_ELASTIC_KV_FLOOR_CELLS=8192 LLAMA_ELASTIC_KV_FLOOR_MIB=200000" "ekbench.py solo p0" "fb.py refuse 16384" "ekbench.py solo p1" ;;
    SSL0u) bash nc-leg.sh SSL0u 5 1024000 "$P LLAMA_LOGITS_FNV=1" "chunk_spark.py cold L16384 8" "ekbench.py solo p0 p1" "ekbench.py multi p0 p1 p2 p3 --rounds 1" ;;
    SSL0)  bash nc-leg.sh SSL0 5 1024000 "$P LLAMA_LOGITS_FNV=1 LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_LOG=1" "chunk_spark.py cold L16384 8" "ekbench.py solo p0 p1" "ekbench.py multi p0 p1 p2 p3 --rounds 1" ;;
    SPB) mkkeys; bash nc-leg.sh SPB 5 1024000 "$P $CL LLAMA_PRIORITY_HOLD_PROMPTS=1" "${PRIO[@]}"; leakcheck SPB; rm -f $K ;;
    SPS) mkkeys; bash nc-leg.sh SPS 5 1024000 "$P $CL LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck SPS; rm -f $K ;;
    SPSD) mkkeys; bash nc-leg.sh SPSD 5 1024000 "$P $CL LLAMA_PREFILL_SLICE_LAYERS=2 LLAMA_PREFILL_SLICE_DECODE_SHARE=1 LLAMA_PREFILL_SLICE_LOG=1" "${PRIO[@]}"; leakcheck SPSD; rm -f $K ;;
    F1) bash nc-leg.sh F1 5 5242880 "$P LLAMA_ELASTIC_KV_FLOOR_CELLS=204800" "ekbench.py solo p0 p0" "fb.py fill 190000 4" "fb.py floor 4" \
          "fb.py burst 500000" "fb.py bsolo 500000" "fb.py conc 4 500000" "fb.py floor 4" "ekbench.py solo p0" ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K
echo "SPARK-NC1-DONE $(date -u +%T)"
