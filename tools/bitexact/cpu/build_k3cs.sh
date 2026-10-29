#!/bin/bash
# Build k3cs (CPU-only remote expert server, static; same flags as build_k3cpu.sh: every FMA explicit, nothing may fuse).
# Needs ../sycl/ggml-common.h (k3x's build.sh copies it from LLAMA_SHA's ggml/src).
set -e
cd "$(dirname "$0")"
${CXX:-g++} -O2 -std=c++17 -ffp-contract=off -fno-fast-math -march=x86-64-v3 -mavxvnni -mtune=${TUNE:-alderlake} -pthread -static \
    $EXTRA -o ${OUT:-k3cs} k3cs.cpp k3cpu.cpp
ls -la ${OUT:-k3cs}
