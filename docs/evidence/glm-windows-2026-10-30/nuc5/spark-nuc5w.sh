#!/bin/bash
# Spark window nuc5, second build (~/nt/a121w: 0067 with the weighted fold of the split-column layout in batch-invariant
# mode). Legs D1/D2/V showed the boxes' rows are the GPU's rows and that LLAMA_EXPERT_STORE_SPLIT_COLS=1 alone moved
# the bytes: an N-column split step computed MUL + seven ADDs where every other step computes the fused fold.
#  W1 / W2 : quick byte probes, split columns without / with the boxes
#  Aw      : production config, tier off                    (full client list, two rounds)
#  Bw      : + LLAMA_EXPERT_STORE_SPLIT_COLS=1, no boxes
#  Cw      : + the batched remote tier
#  A2w B2w C2w : second processes, shorter list (one round)
#  PBw / PCw : client classes + slot reserve + hold rule with split columns / with the tier (PA is the baseline)
#  Vw      : tier on, rows dumped and replayed through the GPU kernels
# Usage: spark-nuc5w.sh LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=$HOME/nt/a121w
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SC="LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=8,6 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1 LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100"
KD=$PWD/keys-nuc5; K=$KD/classes.txt
CL="LLAMA_CLIENT_CLASSES=$K LLAMA_SLOT_RESERVE=1 LLAMA_PRIORITY_HOLD_PROMPTS=1"
Q=("ekbench.py solo p0" "ekbench.py multi p0 p1 p2 p3 --rounds 1")
L=("ekbench.py solo p0 p1 p2 p3 p4" "ekbench.py solo p0 --n-predict 128" "ekbench.py multi p0 p1 --rounds 2" "ekbench.py multi p0 p1 p2 --rounds 2"
   "ekbench.py multi p0 p1 p2 p3 --rounds 2" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 2" "ekbench.py multi L16384 p0 p1 p2 p3 --rounds 1")
L2=("ekbench.py solo p0 p4" "ekbench.py multi p0 p1 --rounds 1" "ekbench.py multi p0 p1 p2 --rounds 1"
   "ekbench.py multi p0 p1 p2 p3 --rounds 1" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1" "ekbench.py multi L16384 p0 p1 p2 p3 --rounds 1")
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
    W1)  bash nuc5-leg.sh W1 5 1024000 "$P $SC" "${Q[@]}" ;;
    W2)  bash nuc5-leg.sh W2 5 1024000 "$P $SC $REM" "${Q[@]}" ;;
    Aw)  bash nuc5-leg.sh Aw 5 1024000 "$P" "${L[@]}" ;;
    Bw)  bash nuc5-leg.sh Bw 5 1024000 "$P $SC" "${L[@]}" ;;
    Cw)  bash nuc5-leg.sh Cw 5 1024000 "$P $SC $REM" "${L[@]}" ;;
    A2w) bash nuc5-leg.sh A2w 5 1024000 "$P" "${L2[@]}" ;;
    B2w) bash nuc5-leg.sh B2w 5 1024000 "$P $SC" "${L2[@]}" ;;
    C2w) bash nuc5-leg.sh C2w 5 1024000 "$P $SC $REM" "${L2[@]}" ;;
    PBw) mkkeys; bash nuc5-leg.sh PBw 5 1024000 "$P $CL $SC" "${PRIO[@]}"; leakcheck PBw; rm -f $K ;;
    PCw) mkkeys; bash nuc5-leg.sh PCw 5 1024000 "$P $CL $SC $REM" "${PRIO[@]}"; leakcheck PCw; rm -f $K ;;
    Vw)  rm -f legs-nuc5/dump.bin
         bash nuc5-leg.sh Vw 5 1024000 "$P $SC $REM LLAMA_EXPERT_STORE_REMOTE_DUMP=$PWD/legs-nuc5/dump.bin LLAMA_EXPERT_STORE_REMOTE_DUMP_N=8192" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1"
         GGML_CUDA_BATCH_INVARIANT=1 LD_LIBRARY_PATH=$HOME/nt/a121/bin $HOME/nt/flashreplay store/glm53flash-q2.k3xs $HOME/nd/bases.txt legs-nuc5/dump.bin 2>&1 | grep -a "replay\|differ" | tail -12
         rm -f legs-nuc5/dump.bin ;;
    *) echo "unknown leg $leg" ;;
  esac
done
rm -f $K; rmdir $KD 2>/dev/null
echo "SPARK-NUC5W-DONE $(date -u +%T)"
