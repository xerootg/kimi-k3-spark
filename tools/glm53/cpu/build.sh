#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../../.."
export TMPDIR="$PWD/tools/glm53/cpu/build/tmp"
mkdir -p "$TMPDIR"
${CXX:-g++} -O3 -std=c++17 -pthread ${G53_LINK:--static} -march=x86-64-v3 ${G53_VNNI:--mavxvnni} -ffp-contract=off -fno-fast-math -Itools/bitexact tools/glm53/cpu/g53cpu.cpp tools/glm53/cpu/g53cs.cpp -o "${OUT:-tools/glm53/cpu/g53cs}"
