#!/bin/bash
# all.sh gpu|cpu: ggml baseline (glmbench) against g53x (iGPU) or g53c (CPU) on the six extracted slices, k = 1,2,4,8.
# IT iterations (60), T CPU threads (16). The weights are replicated into 64 slots, so each token streams from DRAM.
T0=$(cd "$(dirname "$0")" && pwd); W=${W:-$HOME/glm53-work}
SL=("UD-Q2_K_XL_L3 17 17 18" "UD-Q2_K_XL_L8 18 18 23" "UD-Q2_K_XL_L75 17 17 23" "UD-Q4_K_XL_L3 12 12 13" "UD-Q4_K_XL_L8 13 13 14" "UD-Q4_K_XL_L75 12 12 14")
IT=${IT:-60}; T=${T:-16}
if [ "$1" = gpu ]; then
  cmd=""
  for s in "${SL[@]}"; do set -- $s
    cmd+="build-sycl/glmbench SYCL0 data/$1 $2 $3 $4 1,2,4,8 $IT 2>/dev/null; G53X_PROF=1 ./g53x data/$1 $2 $3 $4 bench 1,2,4,8 $IT 2>/dev/null; "
  done
  $T0/dsycl.sh "$cmd"
else
  cd $W
  for s in "${SL[@]}"; do set -- $s
    build-cpu/glmbench CPU data/$1 $2 $3 $4 1,2,4,8 $IT $T 2>/dev/null
    [ "$4" = 14 ] || ./g53c data/$1 $2 $3 $4 $T bench 1,2,4,8 $IT 2>/dev/null   # no Q6_K on the CPU yet
  done
fi
