#!/bin/bash
# Window nd7 (ctx 262144, prompt isolation): per-N adaptive pairs per box; N=2/4 on a parallel-4 server, N=8 on a
# parallel-8 server; remote off (I0) / on (I1) back to back.
set -u
OUT=/srv/glm-5.3-flash/legs-nd7
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
export CTX=262144
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_PREFILL_ISOLATE=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=${BOXROWS:-6,4} LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1 LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100"
stats() { grep -a "BATCHED\|remote box [01] (192.*posts" $OUT/server-$1.log | tail -7 | cut -c1-600; }
if start J0 4 $EXM; then mb multi --tag J0 --n 4 --rounds 3; mb multi --tag J0 --n 2 --rounds 2; stop; fi
if start J1 4 $EXM $REM; then mb multi --tag J1 --n 4 --rounds 3; mb multi --tag J1 --n 2 --rounds 2; stop; stats J1; fi
if start K0 8 $EXM; then mb multi --tag K0 --n 8 --rounds 2; stop; fi
if start K1 8 $EXM $REM; then mb multi --tag K1 --n 8 --rounds 2; stop; stats K1; fi
finish
echo WIN-ND7-DONE
