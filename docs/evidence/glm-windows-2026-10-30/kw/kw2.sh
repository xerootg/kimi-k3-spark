#!/bin/bash
cd ~/kf/win
DB="LLAMA_EXPERT_STORE_DEVICE_BANK=1 GGML_CUDA_HOST_BANK_LAUNCHER_WAIT=1"
./kw-leg.sh B6 bench "$DB"
./kw-leg.sh A4 bench ""
./kw-leg.sh B7 nsys "$DB"
./kw-leg.sh B8 bench "$DB"
echo KW2-DONE
