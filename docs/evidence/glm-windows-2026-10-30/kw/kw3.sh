#!/bin/bash
cd ~/kf/win
./kw-leg.sh C1 bench ""
./kw-leg.sh C2 nsys ""
./kw-leg.sh B9 bench "LLAMA_EXPERT_STORE_DEVICE_BANK=1 GGML_CUDA_HOST_BANK_LAUNCHER_WAIT=1"
./kw-leg.sh C3 bench ""
echo KW3-DONE
