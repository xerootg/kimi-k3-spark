#!/bin/bash
# Window nd4: after the fallback-eviction fix. D1 = remote on but every pair computed locally (DISCARD, tests the
# fallback path and the graph change); A1 = remote on, adaptive pairs per box (max 4,3). Reference: nd2 R0.
set -u
OUT=/srv/glm-5.3-flash/legs-nd4
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1"
stats() { grep -a "BATCHED\|remote box [01] (192.*posts\|split columns" $OUT/server-$1.log | tail -6 | cut -c1-600; }
if start D1 4 $EXM $REM LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=2,1 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_DISCARD=1 LLAMA_EXPERT_STORE_REMOTE_TRIP=1000000; then
  mb multi --tag D1 --n 2 --rounds 2; stop; stats D1; fi
if start A1 4 $EXM $REM LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=4,3 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1; then
  mb multi --tag A1 --n 2 --rounds 2; mb multi --tag A1 --n 4 --rounds 2; stop; stats A1; fi
finish
echo WIN-ND4-DONE
