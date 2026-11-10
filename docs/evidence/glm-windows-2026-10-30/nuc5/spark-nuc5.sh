#!/bin/bash
# Spark window nuc5 (2026-11-10): does the batched NUC expert tier (0049, parked "until multi-slot serving") pay on the
# five-slot batch-invariant production stack? Binary ~/nt/a121 (stack through 0059 + 0067: the tier asks the boxes for
# the one-warp MoE row order). glm53-spark at 0; f53s (magic-aware) running on nuc2 and nuc3, port 19700.
#  A  / A2 : production config (five slots, batch-invariant), tier off
#  B  / B2 : + LLAMA_EXPERT_STORE_SPLIT_COLS=1 (the N-column split the tier needs), tier off
#  C  / C2 : + the batched remote tier
#  PA / PC : client classes + slot reserve + hold rule, tier off / on: Juno's TTFT next to four generating clients
#  V       : tier on, the NUCs' rows dumped with their inputs and replayed through the GPU kernels
# Every leg: every client's bytes at one to five clients and with a 16k prompt in the mix; two rounds; legs X and X2 are
# two processes. Usage: spark-nuc5.sh LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=$HOME/nt/a121
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SC="LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=8,6 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1 LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100"
KD=$PWD/keys-nuc5; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1 LLAMA_PRIORITY_HOLD_PROMPTS=1"
L=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
   "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi L16384 p0 p1 p2 p3 --rounds 1")
PRIO=("prio_spark.py warm --classes $K" "prio_spark.py ttft4 --classes $K" "prio_spark.py ttft4 --classes $K" "prio_spark.py junofirst --classes $K" "prio_spark.py nokey --classes $K")
mkkeys() { mkdir -p $KD; chmod 0700 $KD; ( umask 077; python3 -c "
import secrets
rows=[('juno',10),('workbench',1),('workbench',1),('claude-code',1),('hef-posters',1)]
open('$K','w').write(''.join('%s %d %s\n'%(secrets.token_urlsafe(24),p,n) for n,p in rows))" ); }
leakcheck() { python3 -c "
keys=[l.split()[0] for l in open('$K') if l.strip()]
t=open('legs-nuc5/server-$1.log',errors='replace').read()
print('server-$1.log', 'KEY-LEAK' if any(k in t for k in keys) else 'no key in log')"; }
for leg in "$@"; do
  case $leg in
    A|A2) bash nuc5-leg.sh $leg 5 1024000 "$P" "${L[@]}" ;;
    B|B2) bash nuc5-leg.sh $leg 5 1024000 "$P $SC" "${L[@]}" ;;
    C|C2) bash nuc5-leg.sh $leg 5 1024000 "$P $SC $REM" "${L[@]}" ;;
    C0)   bash nuc5-leg.sh C0 5 1024000 "$P $SC $REM" "ekbench.py solo p0" "ekbench.py multi p0 p1 p2 p3 --rounds 1" ;;
    PA) mkkeys; bash nuc5-leg.sh PA 5 1024000 "$P $CL" "${PRIO[@]}"; leakcheck PA; rm -f $K ;;
    PC) mkkeys; bash nuc5-leg.sh PC 5 1024000 "$P $CL $SC $REM" "${PRIO[@]}"; leakcheck PC; rm -f $K ;;
    V)  rm -f legs-nuc5/dump.bin
        bash nuc5-leg.sh V 5 1024000 "$P $SC $REM LLAMA_EXPERT_STORE_REMOTE_DUMP=$PWD/legs-nuc5/dump.bin LLAMA_EXPERT_STORE_REMOTE_DUMP_N=8192" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1"
        ls -la legs-nuc5/dump.bin
        GGML_CUDA_BATCH_INVARIANT=1 LD_LIBRARY_PATH=$BD/bin $HOME/nd/flashreplay store/glm53flash-q2.k3xs $HOME/nd/bases.txt legs-nuc5/dump.bin 2>&1 | grep -a "replay\|differ" | tail -12
        rm -f legs-nuc5/dump.bin ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K; rmdir $KD 2>/dev/null
echo "SPARK-NUC5-DONE $(date -u +%T)"
