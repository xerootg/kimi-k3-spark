#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../../.."
B=tools/glm53/cpu/build
mkdir -p "$B"
FLAGS="-O3 -std=c++17 -pthread -march=x86-64-v3 ${G53_VNNI:--mavxvnni} -ffp-contract=off -fno-fast-math -Itools/bitexact -Itools/glm53/ref"
export TMPDIR="$PWD/tools/glm53/cpu/build/tmp"
mkdir -p "$TMPDIR"
${CXX:-g++} $FLAGS tools/glm53/cpu/g53cpu.cpp tools/glm53/cpu/test.cpp -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=posix_memalign,--wrap=aligned_alloc -o "$B/test"
"$B/test" "$@"
