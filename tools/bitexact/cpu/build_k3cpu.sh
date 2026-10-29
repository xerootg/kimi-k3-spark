#!/bin/bash
# Build k3cpu_test (static, runs on any x86-64 with AVX2 + AVX-VNNI: Alder/Raptor/Meteor/Arrow Lake, Zen 5).
# -ffp-contract=off -fno-fast-math: every FMA of SPEC.md is an explicit _mm256_fmadd_ps / fma, nothing else may fuse.
set -e
cd "$(dirname "$0")"
${CXX:-g++} -O2 -std=c++17 -ffp-contract=off -fno-fast-math -march=x86-64-v3 -mavxvnni -mtune=${TUNE:-alderlake} -pthread -static \
    $EXTRA -o ${OUT:-k3cpu_test} k3cpu_test.cpp k3cpu.cpp
ls -la ${OUT:-k3cpu_test}
