#!/bin/bash
# Window nd5: the same comparison without memory-valve activity: parallel 4 cannot use elastic KV, so the total context
# is 262144 (65536 per slot; KV ~5 GB instead of 19.7) and the 78000 MiB cache is never released.
# R0 = remote off (MMVQ exact columns + N-column split); A2 = remote on, adaptive pairs per box (start 1, max 4,3).
set -u
OUT=/srv/glm-5.3-flash/legs-nd5
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
export CTX=262144
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000"
stats() { grep -a "BATCHED\|remote box [01] (192.*posts\|valve: host\|hit rate" $OUT/server-$1.log | tail -8 | cut -c1-600; }
legs() { mb solo --tag $1 --n 1; mb multi --tag $1 --n 2 --rounds 3; mb multi --tag $1 --n 4 --rounds 3; }
if start R0 4 $EXM; then legs R0; stop; stats R0; fi
if start A2 4 $EXM $REM LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=4,3 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1; then legs A2; stop; stats A2; fi
finish
echo WIN-ND5-DONE
