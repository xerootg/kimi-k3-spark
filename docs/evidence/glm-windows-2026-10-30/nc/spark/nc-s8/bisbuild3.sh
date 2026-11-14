#!/bin/bash
# bisbuild3.sh: window nc-s8 point p0069d = patches through 0069 plus only the CUDA hunks of 0070 (0070-cuda-only.patch),
# built in the ORIGINAL tree ~/bis/src (reset from its 0076 state to upstream, re-patched; the same build dir and the
# same paths as p0067..p0076, so every object of an untouched file is byte-identical to p0069's). Run after the
# measurement runner has finished (no server running).
set -eu
B=$HOME/bis; export PATH=/usr/local/cuda/bin:$PATH
cd $B/src
git checkout -q -- . && git clean -qfd -e build
for p in $B/patches/*.patch; do n=$(basename $p | cut -c1-4); [ $((10#$n)) -le 69 ] && git apply $p; done
git apply $B/0070-cuda-only.patch
echo "$(date -u +%T) tree = 0001..0069 + cuda-only 0070: $(git diff --stat HEAD | tail -1)"
cmake --build build -j ${JOBS:-20} > $B/make-0069d.log 2>&1 || { echo "BUILD FAILED"; tail -20 $B/make-0069d.log; exit 1; }
D=$B/p0069d/bin; rm -rf $B/p0069d; mkdir -p $D
find build -name '*.so*' -exec cp -a {} $D/ \;
cp build/bin/llama-server $D/
( cd $D && sha256sum llama-server $(ls *.so | sort) > ../SHA256SUMS )
git diff --stat HEAD | tail -1 > $B/p0069d/diffstat.txt
LD_LIBRARY_PATH=$D ldd $D/llama-server | (! grep 'not found')
echo "libs that differ from p0069: $(diff <(cut -d' ' -f1,3 $B/p0069/SHA256SUMS) <(cut -d' ' -f1,3 $B/p0069d/SHA256SUMS) | grep '^>' | awk '{print $3}' | tr '\n' ' ')"
echo "libs that differ from p0070: $(diff <(cut -d' ' -f1,3 $B/p0070/SHA256SUMS) <(cut -d' ' -f1,3 $B/p0069d/SHA256SUMS) | grep '^>' | awk '{print $3}' | tr '\n' ' ')"
echo "BISBUILD3-DONE $(date -u +%FT%T)"
