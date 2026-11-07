#!/bin/bash
# Window nd8: repeat of nd7 in A/B/A/B order for variance (ctx 262144, prompt isolation, per-N adaptive pairs).
set -u
OUT=/srv/glm-5.3-flash/legs-nd8
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
export CTX=262144
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_PREFILL_ISOLATE=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=8,6 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1 LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100"
stats() { grep -a "BATCHED\|remote box [01] (192.*posts" $OUT/server-$1.log | tail -7 | cut -c1-600; }
if start Y1 8 $EXM $REM; then mb multi --tag Y1 --n 8 --rounds 2; stop; stats Y1; fi
if start Y0 8 $EXM; then mb multi --tag Y0 --n 8 --rounds 2; stop; fi
if start X1 4 $EXM $REM; then mb multi --tag X1 --n 4 --rounds 3; stop; stats X1; fi
if start X0 4 $EXM; then mb multi --tag X0 --n 4 --rounds 3; stop; fi
finish
echo WIN-ND8-DONE
