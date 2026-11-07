#!/bin/bash
# Window nd2: batched remote tier, first run. R0 = MMVQ exact columns + N-column split, remote off; R1 = the same with
# nuc2 + nuc3 serving (one request per box per MoE layer per decode step). Same requests; md5 per stream compared.
set -u
OUT=/srv/glm-5.3-flash/legs-nd2
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=${BOXROWS:-4,3} LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=${DEADLINE:-3000}"
legs() { mb solo --tag $1 --n 1; mb multi --tag $1 --n 2 --rounds 2; mb multi --tag $1 --n 4 --rounds 3; grep -a "split GEMV\|hit rate\|remote\|columns" $OUT/server-$1.log | tail -8 | cut -c1-420; }
if [ "${SKIP_R0:-0}" != 1 ]; then if start R0 4 $EXM; then legs R0; stop; fi; fi
if start R1 4 $EXM $REM; then legs R1; stop; grep -a "remote" $OUT/server-R1.log | tail -8 | cut -c1-500; fi
finish
echo WIN-ND2-DONE
