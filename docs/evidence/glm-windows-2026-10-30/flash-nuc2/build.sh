#!/bin/bash
# Build moebench (CPU, x86-64, all CPU variants) against llama.cpp 547208d inside ubuntu:24.04.
set -e
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential cmake ca-certificates >/dev/null
git init -q /src && cd /src && git remote add origin https://github.com/ggml-org/llama.cpp && git fetch -q --depth 1 origin 547208dca250ceb3e4c8615598d043f2b602a527 && git checkout -q FETCH_HEAD
cd /w && cmake -S . -B build -DGGML_SRC=/src/ggml -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j 16 >/dev/null
mkdir -p /w/bundle/lib && cp build/moebench /w/bundle/ && find build -name "*.so*" -exec cp -P {} /w/bundle/lib/ \;
ls -la /w/bundle /w/bundle/lib | head -30
