#!/bin/bash
# Build k3x (SYCL, Intel Arc iGPU) in intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04:
#   docker run --rm -v $PWD:/w -v <llama.cpp>/ggml/src:/ggml intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 /w/build.sh
# Precise FP model, no contraction: every FMA in the CUDA SASS is an explicit fma() in the source, nothing else may fuse.
# The runtime (SYCL/UR/Level Zero + compute-runtime 26.18 / IGC 2.34.4 user-mode driver) is the lib/ directory that
# tools/distributed/bundle.sh collects (libmkl*/libggml* not needed); run with LD_LIBRARY_PATH=lib and the render group.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -e
cd /w
G=${GGML_SRC:-/ggml}
cp $G/ggml-common.h .
{ echo '#pragma once'; awk '/GGML_TABLE_BEGIN\(uint32_t, iq1s_grid_gpu/{p=1} p{print} p&&/GGML_TABLE_END/{exit}' ggml-common.h; } > k3x_tables.h
icpx -fsycl -O2 -ffp-model=precise -ffp-contract=off -fno-fast-math $EXTRA -o ${OUT:-k3x} k3x.cpp
ls -la ${OUT:-k3x}
