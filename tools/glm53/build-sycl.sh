#!/bin/bash
# build-sycl.sh: glmbench with ggml-sycl, and g53x (+ bw, allocgate.so for disk_test.py gate), via dsycl.sh. Outputs in
# $W/build-sycl and $W.
# g53x flags: -DG53X_RG=n / -DG53X_RD=n rows per sub-group (2/2), -DG53X_SIGN_ALU=0 sign masks from SLM.
T=$(cd "$(dirname "$0")" && pwd)
$T/dsycl.sh "cmake -S /w/tools -B /w/build-sycl -DGGML_SRC=/w/llama/ggml -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=OFF \
  -DGGML_CPU=OFF -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON >/dev/null && \
  cmake --build /w/build-sycl -j 12 --target glmbench 2>&1 | grep -E 'error|Built target'; \
  icpx -fsycl -O2 -std=c++17 -I/w/llama/ggml/src -I/bitexact $EXTRA -o /w/g53x /w/tools/g53x.cpp -lze_loader && icpx -fsycl -O2 -o /w/bw /w/tools/bw.cpp && \
  icx -O2 -shared -fPIC -o /w/allocgate.so /w/tools/allocgate.c -ldl && ls -la /w/g53x"
