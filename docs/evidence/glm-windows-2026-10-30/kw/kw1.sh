#!/bin/bash
cd ~/kf/win
./kw-leg.sh A1 bench ""
./kw-leg.sh B1 bench "LLAMA_EXPERT_STORE_DEVICE_BANK=1"
./kw-leg.sh A2 nsys ""
./kw-leg.sh B2 nsys "LLAMA_EXPERT_STORE_DEVICE_BANK=1"
./kw-leg.sh A3 bench ""
./kw-leg.sh B3 bench "LLAMA_EXPERT_STORE_DEVICE_BANK=1"
echo KW1-DONE
