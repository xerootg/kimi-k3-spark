#!/bin/bash
# Window nd3: which knob changes the standard request's bytes at N=1 (build-nd, parallel 4, solo P0), and the 16k gold
# with the batched remote tier on.
set -u
OUT=/srv/glm-5.3-flash/legs-nd3
BIN=$HOME/llama-k3-nd/build-nd/bin/llama-server; export LD_LIBRARY_PATH="$HOME/llama-k3-nd/build-nd/bin"
source $HOME/nd/ndlib.sh
REM="LLAMA_EXPERT_STORE_REMOTE=192.168.4.249:19700,192.168.4.250:19700 LLAMA_EXPERT_STORE_REMOTE_SET=$HOME/nd/nuc2.set,$HOME/nd/nuc3.set LLAMA_EXPERT_STORE_REMOTE_COLS=1 LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS=4,3 LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US=3000"
if start F0 4; then mb solo --tag F0 --n 1; stop; fi
if start F1 4 GGML_CUDA_MMVQ_EXACT_COLS=1; then mb solo --tag F1 --n 1; stop; fi
if start F2 4 LLAMA_EXPERT_STORE_SPLIT_COLS=1; then mb solo --tag F2 --n 1; stop; fi
if start F3 1 GGML_CUDA_MMVQ_EXACT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS=1 $REM; then mb solo --tag F3 --n 1; python3 -I bench.py depth --url http://127.0.0.1:18080 --tag F3-d16k --lens 16384 --n-predict 128 | tee -a $OUT/results.jsonl; stop; fi
finish
echo WIN-ND3-DONE
