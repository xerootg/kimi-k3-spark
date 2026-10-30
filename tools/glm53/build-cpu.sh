#!/bin/bash
# build-cpu.sh: glmbench/glmref with the ggml CPU backend (native), g53c and bwcpu, on the host (gcc). Outputs in $W.
set -e
T=$(cd "$(dirname "$0")" && pwd); W=${W:-$HOME/glm53-work}
cmake -S $T -B $W/build-cpu -DGGML_SRC=$W/llama/ggml -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON >/dev/null
cmake --build $W/build-cpu -j 16 2>&1 | grep -E 'error|Built target'
g++ -O2 -std=c++17 -march=${MARCH:-alderlake} -mavxvnni -pthread -I$W/llama/ggml/src $EXTRA -o $W/g53c $T/g53c.cpp
gcc -O2 -mavx2 -fopenmp -o $W/bwcpu $T/bwcpu.c
