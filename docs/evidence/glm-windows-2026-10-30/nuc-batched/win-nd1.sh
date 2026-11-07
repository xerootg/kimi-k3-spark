#!/bin/bash
# Window nd1: the batched remote tier's base config without NUCs: expert-only exact columns (MMVQ) + N-column split.
set -u
OUT=/srv/glm-5.3-flash/legs-nd1
BIN=$HOME/llama-k3-el/build-el/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-el/build-el/bin"
source $HOME/nd/ndlib.sh
EXM="GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1"
if start M4 4 $EXM; then mb solo --tag M4 --n 4; mb multi --tag M4 --n 2 --rounds 2; mb multi --tag M4 --n 4 --rounds 3; grep -a "split GEMV\|hit rate\|columns" $OUT/server-M4.log | tail -3 | cut -c1-400; stop; fi
finish
echo WIN-ND1-DONE
