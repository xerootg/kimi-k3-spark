#!/bin/bash
# Native build for a per-op invariance bisect if the GH200 md5s differ: upstream + patches 0001-0055 + bi-dump.patch, sm_90.
set -u
G=/srv/glm-5.3-flash/gbi; SRC=$HOME/llama-gbi; LLAMA_SHA=547208dca250ceb3e4c8615598d043f2b602a527
export PATH=/usr/local/cuda/bin:$PATH
rm -rf "$SRC" && git init -q "$SRC" && cd "$SRC" && git remote add origin https://github.com/ggml-org/llama.cpp \
  && git fetch -q --depth 1 origin $LLAMA_SHA && git checkout -q FETCH_HEAD || { echo BUILD-FAIL fetch; exit 1; }
for p in $G/patches/*.patch $G/dbg/bi-dump.patch; do git apply "$p" || { echo "BUILD-FAIL apply $p"; exit 1; }; done
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CPU_ARM_ARCH=armv9-a+dotprod+fp16+sve+sve2+i8mm+bf16 > $G/dbg/cmake.log 2>&1 \
  && nice -n 19 cmake --build build -j 24 --target llama-server > $G/dbg/build.log 2>&1 || { echo "BUILD-FAIL see $G/dbg/build.log"; exit 1; }
echo "BUILD-OK $SRC/build/bin/llama-server $(date -u +%T)"
