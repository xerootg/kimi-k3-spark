#!/bin/bash
# Window nd6 (ctx 262144, no valve): V = remote rows dumped with their inputs for a GPU replay (row exactness with
# real activations); I0/I1 = prompt isolation (LLAMA_PREFILL_ISOLATE=1) so the four streams' decode steps have a fixed
# composition, remote off / on (adaptive pairs, short wait for resident pairs).
set -u
OUT=/srv/glm-5.3-flash/legs-nd6
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
export CTX=262144
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_SPIN_US=5000 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=4,3 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000 LLAMA_EXPERT_STORE_REMOTE_ADAPT=1"
stats() { grep -a "BATCHED\|remote box [01] (192.*posts" $OUT/server-$1.log | tail -7 | cut -c1-600; }
rm -f $OUT/V.dump
if start V 4 $EXM $REM LLAMA_EXPERT_STORE_REMOTE_DUMP=$OUT/V.dump LLAMA_EXPERT_STORE_REMOTE_DUMP_N=32768; then mb solo --tag V --n 1; mb multi --tag V --n 2 --rounds 1; stop; stats V
  $HOME/nd/flashreplay $PWD/store/glm53flash-q2.k3xs $HOME/nd/bases.txt $OUT/V.dump 2>&1 | tail -12; fi
if start I0 4 $EXM LLAMA_PREFILL_ISOLATE=1; then mb multi --tag I0 --n 4 --rounds 3; mb multi --tag I0 --n 2 --rounds 1; stop; fi
if start I1 4 $EXM LLAMA_PREFILL_ISOLATE=1 $REM LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US=100; then mb multi --tag I1 --n 4 --rounds 3; mb multi --tag I1 --n 2 --rounds 1; stop; stats I1; fi
finish
echo WIN-ND6-DONE
