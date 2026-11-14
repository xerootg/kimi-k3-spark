#!/bin/bash
# bisbuild2.sh: window nc-s8 extra point p0069c = patches through 0069 plus only the CUDA hunks of 0070 (0070-cuda-only.patch:
# ggml/src/ggml-cuda/common.cuh and ggml-cuda.cu), tree ~/bis/src2 (a copy of ~/bis/src reset to upstream and re-patched, so
# every file is recompiled once), same cmake cache and flags as bisbuild.sh. Run after the measurement runner has finished.
set -eu
B=$HOME/bis; export PATH=/usr/local/cuda/bin:$PATH
cd $B/src2
cmake --build build -j ${JOBS:-20} > $B/make-0069c.log 2>&1 || { echo "BUILD FAILED"; tail -20 $B/make-0069c.log; exit 1; }
D=$B/p0069c/bin; rm -rf $B/p0069c; mkdir -p $D
find build -name '*.so*' -exec cp -a {} $D/ \;
cp build/bin/llama-server $D/
( cd $D && sha256sum llama-server $(ls *.so | sort) > ../SHA256SUMS )
git diff --stat HEAD | tail -1 > $B/p0069c/diffstat.txt
LD_LIBRARY_PATH=$D ldd $D/llama-server | (! grep 'not found')
echo "libs that differ from p0069: $(diff <(cut -d' ' -f1,3 $B/p0069/SHA256SUMS) <(cut -d' ' -f1,3 $B/p0069c/SHA256SUMS) | grep '^>' | awk '{print $3}' | tr '\n' ' ')"
echo "libs that differ from p0070: $(diff <(cut -d' ' -f1,3 $B/p0070/SHA256SUMS) <(cut -d' ' -f1,3 $B/p0069c/SHA256SUMS) | grep '^>' | awk '{print $3}' | tr '\n' ' ')"
echo "BISBUILD2-DONE $(date -u +%FT%T)"
