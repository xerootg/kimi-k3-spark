#!/bin/bash
# Build k3cs (CPU-only remote expert server, static; same flags as build_k3cpu.sh: every FMA explicit, nothing may fuse).
# Needs ../sycl/ggml-common.h (k3x's build.sh copies it from LLAMA_SHA's ggml/src).
set -e
cd "$(dirname "$0")"
# liburing is optional, including its static transitive dependencies. Runtime restrictions
# (e.g. io_uring disabled in a container) fall back to the same fixed pread workers.
uring_flags=""
if pkg-config --exists liburing 2>/dev/null; then
    flags=$(pkg-config --cflags --libs --static liburing)
    if printf '#include <liburing.h>\nint main(){io_uring r; return io_uring_queue_init(1,&r,0);}\n' |
        ${CXX:-g++} -x c++ - -static $flags -o /dev/null 2>/dev/null; then
        uring_flags="-DK3X_HAVE_URING $flags"
    fi
fi
${CXX:-g++} -O2 -std=c++17 -ffp-contract=off -fno-fast-math -march=x86-64-v3 -mavxvnni -mtune=${TUNE:-alderlake} -pthread -static \
    $EXTRA -o ${OUT:-k3cs} k3cs.cpp k3cpu.cpp $uring_flags
ls -la ${OUT:-k3cs}
