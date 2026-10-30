#!/bin/bash
# Default: all CUDA dump cases with the real, read-only MUFU tables.
# Legacy synthetic/slice suite: test-parallel.sh --slices [OUT [DATA]]
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=dumps
if [[ ${1:-} == --slices ]]; then MODE=slices; shift; fi
OUT=${1:-tools/glm53/test-parallel}
mkdir -p "$OUT"
export TMPDIR="$(cd "$OUT" && pwd)/tmp"
mkdir -p "$TMPDIR"
g++ -O2 -std=c++17 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math \
    -Itools/glm53/ref -Itools/bitexact tools/glm53/test-parallel.cpp -o "$OUT/test-parallel"
if [[ $MODE == slices ]]; then
    "$OUT/test-parallel" "${2:-$HOME/glm53-work/data}" | tee "$OUT/test.log"
else
    "$OUT/test-parallel" "${RECORDS_DIR:-tools/glm53/records}" \
        "${DUMPS_DIR:-evidence/cuda-dumps}" "${MUFU_DIR:-$HOME/glm53-work/mufu}" | tee "$OUT/dumps.log"
fi
