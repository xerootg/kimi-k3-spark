# k3x-serve: the bit-exact Kimi K3 routed-expert server for the Intel Arc iGPU
# (tools/bitexact/sycl, `k3x IDX BIN - - serveidx PORT`), with the exact SYCL /
# Level Zero / compute-runtime stack it was validated on (the lib/ bundle of
# tools/distributed/bundle.sh): oneAPI 2026.1.1 SYCL runtime, Level Zero loader
# 1.28.2, compute-runtime 26.18.38308.1, IGC 2.34.4. IGC JIT-compiles the kernels
# at startup, so the IGC version is part of the bit-exactness claim: bump it only
# together with a validate run against the dumps.
#
# Runtime: /opt/k3x/k3x + /opt/k3x/lib (LD_LIBRARY_PATH is set). The weights
# (IDX + BIN) and the MUFU tables (K3X_MUFU) are not in the image; mount them.
# Needs the render node (gpu.intel.com/i915) and runs as root in the container.
ARG UBUNTU_VERSION=24.04

FROM docker.io/intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04 AS build
ARG LLAMA_SHA
# Only the iq1s grid table is taken from ggml-common.h (build.sh extracts it);
# the Spark image applies patches/ on top of LLAMA_SHA but none touch it.
ADD https://raw.githubusercontent.com/ggml-org/llama.cpp/${LLAMA_SHA}/ggml/src/ggml-common.h /ggml/ggml-common.h
COPY tools/bitexact/sycl/build.sh tools/bitexact/sycl/k3x.cpp tools/bitexact/sycl/k3x_act.h \
     tools/bitexact/sycl/k3x_math.h tools/bitexact/sycl/k3x_mufu.h /w/sycl/
# the CPU half (K3X_CPU_CPUS): S3's bit-exact AVX2 + AVX-VNNI kernels, compiled by build.sh with g++
COPY tools/bitexact/cpu/k3cpu.h tools/bitexact/cpu/k3cpu.cpp /w/cpu/
RUN bash /w/sycl/build.sh
# User-mode GPU driver stack, pinned (same URLs as tools/distributed/bundle.sh).
ADD https://github.com/oneapi-src/level-zero/releases/download/v1.28.2/level-zero_1.28.2%2Bu24.04_amd64.deb \
    https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-core-2_2.34.4+21428_amd64.deb \
    https://github.com/intel/intel-graphics-compiler/releases/download/v2.34.4/intel-igc-opencl-2_2.34.4+21428_amd64.deb \
    https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libigdgmm12_22.10.0_amd64.deb \
    https://github.com/intel/compute-runtime/releases/download/26.18.38308.1/libze-intel-gpu1_26.18.38308.1-0_amd64.deb \
    /debs/
RUN set -e; mkdir -p /opt/k3x/lib /x; cp /w/sycl/k3x /opt/k3x/; \
    for d in /debs/*.deb; do dpkg -x "$d" /x; done; \
    find /x -name '*.so*' \( -type f -o -type l \) -exec cp -P {} /opt/k3x/lib/ \; ; \
    CL=/opt/intel/oneapi/compiler/latest/lib; \
    cp -L $CL/libur_loader.so.0 $CL/libur_adapter_level_zero.so.0 $CL/libur_adapter_level_zero_v2.so.0 /opt/k3x/lib/; \
    for i in 1 2 3; do \
      for f in /opt/k3x/k3x /opt/k3x/lib/*.so*; do \
        LD_LIBRARY_PATH=/opt/k3x/lib:$LD_LIBRARY_PATH ldd "$f" 2>/dev/null | awk '/=> \/opt\/intel/ {print $3}'; \
      done | sort -u | while read -r l; do [ -e "/opt/k3x/lib/$(basename "$l")" ] || cp -L "$l" /opt/k3x/lib/; done; \
    done; \
    ls -la /opt/k3x /opt/k3x/lib; md5sum /opt/k3x/k3x

FROM docker.io/ubuntu:${UBUNTU_VERSION}
COPY --from=build /opt/k3x /opt/k3x
ENV LD_LIBRARY_PATH=/opt/k3x/lib
# taskset (util-linux) is in the base image; fail the build on a missing library.
RUN set -e; for f in /opt/k3x/k3x /opt/k3x/lib/*.so*; do ldd "$f" | (! grep 'not found') || { echo "missing deps: $f"; exit 1; }; done; \
    command -v taskset
WORKDIR /opt/k3x
ENTRYPOINT ["/opt/k3x/k3x"]
