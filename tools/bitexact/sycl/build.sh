#!/bin/bash
# Build k3x (SYCL, Intel Arc iGPU) in intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04:
#   docker run --rm -v <repo>/tools/bitexact:/w -v <llama.cpp>/ggml/src:/ggml intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 /w/sycl/build.sh
# (the older form, -v <repo>/tools/bitexact/sycl:/w ... /w/build.sh, still builds, without the CPU half).
# Precise FP model, no contraction: every FMA in the CUDA SASS is an explicit fma() in the source, nothing else may fuse.
# The runtime (SYCL/UR/Level Zero + compute-runtime 26.18 / IGC 2.34.4 user-mode driver) is the lib/ directory that
# tools/distributed/bundle.sh collects (libmkl*/libggml* not needed); run with LD_LIBRARY_PATH=lib and the render group.
# CPU half (K3X_CPU_CPUS at run time): when ../cpu/k3cpu.cpp is there (NO_CPU=1 skips it), S3's bit-exact AVX2 + AVX-VNNI
# kernels are compiled with g++ exactly as ../cpu/build_k3cpu.sh does (-ffp-contract=off -fno-fast-math) and linked in.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -e
cd "$(dirname "$(readlink -f "$0")")"
G=${GGML_SRC:-/ggml}
cp $G/ggml-common.h .
{ echo '#pragma once'; awk '/GGML_TABLE_BEGIN\(uint32_t, iq1s_grid_gpu/{p=1} p{print} p&&/GGML_TABLE_END/{exit}' ggml-common.h; } > k3x_tables.h
CPUOBJ=; CPUDEF=
if [ -f ../cpu/k3cpu.cpp ] && [ "${NO_CPU:-0}" != 1 ]; then
    g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -march=x86-64-v3 -mavxvnni -mtune=${TUNE:-alderlake} -pthread \
        -c -o k3cpu.o ../cpu/k3cpu.cpp
    CPUOBJ=k3cpu.o; CPUDEF=-DK3X_CPU=1
fi
icpx -fsycl -O2 -ffp-model=precise -ffp-contract=off -fno-fast-math $CPUDEF $EXTRA -o ${OUT:-k3x} k3x.cpp $CPUOBJ -pthread
rm -f k3cpu.o
ls -la ${OUT:-k3x}
