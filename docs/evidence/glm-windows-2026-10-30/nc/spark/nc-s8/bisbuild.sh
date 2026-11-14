#!/bin/bash
# bisbuild.sh: llama-server at the stack points of the kimi-k3-spark patch series (window nc-s8, 2026-11-14), built on
# the Spark itself (the only sm_121 machine) from one source tree, incrementally: upstream LLAMA_SHA, then the patches
# applied in lexical order up to each point (the Dockerfile's loop), a build after each point, the binaries and
# libraries copied to ~/bis/p<point>/bin (the layout of the CI image's /opt/llama, which the nc leg scripts expect).
# A rebuild after a patch recompiles only the translation units the patch touched; the compiler, flags and generator
# are the same for every point, so every object that is not touched is byte-identical across the points.
#
# Differences from docker/llamacpp-spark.Dockerfile, the same for every point: CUDA architecture 121 only (sm_90 is
# not needed here), Unix Makefiles instead of Ninja (not installed on the box), LLAMA_CURL=OFF (no libcurl dev
# package on the box; the server does not download anything in these runs). Everything else is the Dockerfile's.
# Run under a memory-capped user scope with nice while production serves:
#   systemd-run --user --scope -p MemoryMax=16G nice -n 19 bash ~/bis/bisbuild.sh > ~/bis/build.log 2>&1
set -eu
B=$HOME/bis
SHA=547208dca250ceb3e4c8615598d043f2b602a527
POINTS=${POINTS:-"0067 0069 0070 0071 0072 0073 0074 0075 0076"}
JOBS=${JOBS:-6}
export PATH=/usr/local/cuda/bin:$PATH
cd $B
if [ ! -d src/.git ]; then
  mkdir src && cd src && git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin "$SHA" && git checkout -q FETCH_HEAD && cd ..
fi
cd src
touch $B/applied.txt
last=$(tail -1 $B/applied.txt 2>/dev/null || true); last=${last:-0000}
for pt in $POINTS; do
  echo "=== point $pt $(date -u +%FT%T)"
  for p in $B/patches/*.patch; do
    n=$(basename $p | cut -c1-4)
    if [ $((10#$n)) -gt $((10#$last)) ] && [ $((10#$n)) -le $((10#$pt)) ]; then
      echo "applying $(basename $p)"; git apply "$p"; echo "$n" >> $B/applied.txt; last=$n
    fi
  done
  if [ ! -f build/CMakeCache.txt ]; then
    cmake -B build -G "Unix Makefiles" \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DGGML_RPC=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=OFF \
      -DGGML_CPU_ARM_ARCH=armv9-a+dotprod+fp16+sve+sve2+i8mm+bf16 \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined > $B/cmake-$pt.log 2>&1
  fi
  cmake --build build -j $JOBS > $B/make-$pt.log 2>&1 || { echo "BUILD FAILED at $pt, see $B/make-$pt.log"; tail -20 $B/make-$pt.log; exit 1; }
  D=$B/p$pt/bin; rm -rf $B/p$pt; mkdir -p $D
  find build -name '*.so*' -exec cp -a {} $D/ \;
  cp build/bin/llama-server $D/
  ( cd $D && sha256sum llama-server $(ls *.so | sort) > ../SHA256SUMS )
  git diff --stat HEAD | tail -1 > $B/p$pt/diffstat.txt
  LD_LIBRARY_PATH=$D ldd $D/llama-server | (! grep 'not found')
  echo "=== point $pt done $(date -u +%FT%T): $(wc -l < $B/p$pt/SHA256SUMS) files"
done
echo "BISBUILD-DONE $(date -u +%FT%T)"
