#!/bin/bash
# All CUDA stages, both activation-input paths and both down-launch shapes.
# Host entry point: build AND run via the same oneAPI container as build-sycl.sh (same g53x compile flags).
# Optional arguments are container paths: MUFU RECORDS DUMPS. Default dumps: this checkout's evidence/cuda-dumps
# (DUMPS_DIR on the host overrides it), mounted read-only at /dumps.
set -euo pipefail
T=$(cd "$(dirname "$0")" && pwd)
if [[ ${1:-} != --inside ]]; then
    DUMPS_DIR=${DUMPS_DIR:-$T/../../evidence/cuda-dumps}
    [[ -n ${3:-} || -d $DUMPS_DIR ]] || { echo "no CUDA dumps at $DUMPS_DIR" >&2; exit 1; }
    [[ -n ${3:-} ]] || export DSYCL_ARGS="${DSYCL_ARGS:-} -v $(cd "$DUMPS_DIR" && pwd):/dumps:ro"
    printf -v cmd '%q ' /w/tools/test-parallel-sycl.sh --inside \
        "${1:-/w/mufu}" "${2:-/w/tools/records}" "${3:-/dumps}"
    exec "$T/dsycl.sh" "$cmd"
fi
shift
OUT=$T/test-parallel
mkdir -p "$OUT"
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
"${SYCL_CXX:-icpx}" -fsycl -O2 -std=c++17 -Xclang -fexperimental-strict-floating-point \
    -I"$T/ref" -I/bitexact -I"$T/../bitexact" "$T/test-parallel-sycl.cpp" -o "$OUT/test-parallel-sycl" -lze_loader
"$OUT/test-parallel-sycl" "$1" "$2" "$3" 2>&1 | tee "$OUT/sycl-dumps.log"
