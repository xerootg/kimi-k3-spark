#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-tools/glm53/test-exact}
DATA=${2:-$HOME/glm53-work/data}
mkdir -p "$OUT"
g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -Itools/glm53/ref -Itools/bitexact tools/glm53/g53ref_exact.cpp -o "$OUT/g53ref_exact"
"$OUT/g53ref_exact" selftest "$DATA" "$OUT/dumps"
