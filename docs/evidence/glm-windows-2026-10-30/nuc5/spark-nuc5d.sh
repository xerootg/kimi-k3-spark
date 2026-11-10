#!/bin/bash
# Spark window nuc5, diagnosis legs: leg C0 (tier on) answered with other bytes than production already for one client.
# Which part moves the bytes?
#  D1 : production + LLAMA_EXPERT_STORE_SPLIT_COLS=1, no boxes
#  D2 : tier on, LLAMA_EXPERT_STORE_REMOTE_DISCARD=1 (requests sent and awaited, every pair computed locally)
#  D3 : tier on, only non-resident experts go to the boxes (LLAMA_EXPERT_STORE_REMOTE_OFFLOAD=0)
#  V  : tier on, the boxes' rows dumped with their inputs and replayed through the GPU kernels (new-build flashreplay)
# Usage: spark-nuc5d.sh LEG...
cd /srv/glm-5.3-flash
exec 9>/srv/glm-5.3-flash/window.lock; flock -n 9 || { echo "ABORT: window.lock is held"; exit 1; }
export BD=$HOME/nt/a121
P="LLAMA_ELASTIC_KV_REFUSE=1 GGML_CUDA_BATCH_INVARIANT=1 GGML_CUDA_BI_ATTN_CHUNK=128 GGML_CUDA_BI_FUSE=1 LLAMA_KV_STREAM_HOLES=1 LLAMA_PREFILL_ISOLATE=1"
SC="LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=8,6 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1 LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100"
Q=("ekbench.py solo p0" "ekbench.py multi p0 p1 p2 p3 --rounds 1")
for leg in "$@"; do
  case $leg in
    D1) bash nuc5-leg.sh D1 5 1024000 "$P $SC" "${Q[@]}" ;;
    D2) bash nuc5-leg.sh D2 5 1024000 "$P $SC $REM LLAMA_EXPERT_STORE_REMOTE_DISCARD=1" "${Q[@]}" ;;
    D3) bash nuc5-leg.sh D3 5 1024000 "$P $SC $REM LLAMA_EXPERT_STORE_REMOTE_OFFLOAD=0" "${Q[@]}" ;;
    V)  rm -f legs-nuc5/dump.bin
        bash nuc5-leg.sh V 5 1024000 "$P $SC $REM LLAMA_EXPERT_STORE_REMOTE_DUMP=$PWD/legs-nuc5/dump.bin LLAMA_EXPERT_STORE_REMOTE_DUMP_N=8192" "ekbench.py multi p0 p1 p2 p3 p4 --rounds 1"
        ls -la legs-nuc5/dump.bin
        GGML_CUDA_BATCH_INVARIANT=1 LD_LIBRARY_PATH=$BD/bin $HOME/nt/flashreplay store/glm53flash-q2.k3xs $HOME/nd/bases.txt legs-nuc5/dump.bin 2>&1 | grep -a "replay\|differ" | tail -12
        ;;
    *) echo "unknown leg $leg" ;;
  esac
done
echo "SPARK-NUC5D-DONE $(date -u +%T)"
