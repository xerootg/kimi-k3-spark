#!/bin/bash
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
B=/w/moebench/sycl-bundle; rm -rf $B; mkdir -p $B/lib
cp /w/moebench/build-sycl/moebench $B/; cp /w/moebench/build-sycl/bin/libggml-sycl.so $B/lib/; cp -P /w/moebench/build-sycl/ggml/src/libggml*.so* $B/lib/
cd /tmp; wget -q "https://github.com/oneapi-src/level-zero/releases/download/v1.28.2/level-zero_1.28.2%2Bu24.04_amd64.deb" -O lz.deb; dpkg -x lz.deb lz; cp -P lz/usr/lib/x86_64-linux-gnu/libze_loader.so* $B/lib/
mkdir -p neo; cd neo
for u in https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb \
         https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb \
         https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb \
         https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb; do
  wget -q "$u" || echo "FAILED $u"; done
for d in *.deb; do dpkg -x $d x; done
find x -name "*.so*" -exec cp -P {} $B/lib/ \;
CL=/opt/intel/oneapi/compiler/latest/lib
cp -L $CL/libur_adapter_level_zero.so.0 $CL/libur_loader.so.0 $B/lib/ 2>/dev/null
ls $CL | grep -E "libur_adapter" 
for f in $B/moebench $B/lib/libggml-sycl.so $B/lib/libur_adapter_level_zero.so.0 $B/lib/libur_loader.so.0; do
  LD_LIBRARY_PATH=$B/lib:$LD_LIBRARY_PATH ldd $f | awk '/=> \/opt\/intel/ {print $3}'; done | sort -u > /tmp/deps
cat /tmp/deps
while read l; do cp -L $l $B/lib/; done < /tmp/deps
echo "--- missing:"; for f in $B/moebench $B/lib/*.so*; do LD_LIBRARY_PATH=$B/lib ldd $f 2>/dev/null | grep "not found"; done | sort -u
du -sh $B
