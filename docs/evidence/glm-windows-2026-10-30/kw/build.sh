#!/bin/bash
set -euo pipefail
cd ~/kf
export PATH=/usr/local/cuda/bin:$PATH; export HF_UI_VERSION=b11718
SHA=$(sed -n "s/^LLAMA_SHA=//p" versions.env)
if [ ! -d src/.git ]; then
  mkdir -p src && cd src && git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp
  git fetch -q --depth 1 origin "$SHA" && git checkout -q FETCH_HEAD
  for p in ../patches/*.patch; do echo "applying $p"; git apply "$p"; done
  git add -A && git -c user.name=kf -c user.email=kf@spark commit -qm "stack $(ls ../patches/*.patch | wc -l) patches on $SHA" && cd ..
fi
cd src
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DGGML_RPC=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_CPU_ARM_ARCH=armv9-a+dotprod+fp16+sve+sve2+i8mm+bf16 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=ON -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
cmake --build build -j 20 --target llama-server
echo BUILD-DONE
