#!/bin/bash
# All CUDA stages, both activation-input paths and both down-launch shapes.
# Host entry point: build AND run via the same oneAPI container as build-sycl.sh.
# Optional arguments are container paths: MUFU RECORDS DUMPS.
# Prepare default dumps first with cuda_serve_check.py --no-server.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} != --inside ]]; then
    printf -v cmd '%q ' /w/tools/test-parallel-sycl.sh --inside \
        "${1:-/w/mufu}" "${2:-/w/tools/records}" "${3:-/w/tmp/cuda_serve_check/dumps}"
    exec "$T/dsycl.sh" "$cmd"
fi
shift
OUT=$T/test-parallel
mkdir -p "$OUT"
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
"${SYCL_CXX:-icpx}" -fsycl -O2 -ffp-contract=off -fno-fast-math -std=c++17 \
    -I"$T/ref" -I/bitexact -I"$T/../bitexact" "$T/test-parallel-sycl.cpp" -o "$OUT/test-parallel-sycl"
"$OUT/test-parallel-sycl" "$1" "$2" "$3" 2>&1 | tee "$OUT/sycl-dumps.log"
