# llama.cpp for the NVIDIA DGX Spark (GB10, sm_121, aarch64) with the RPC
# backend enabled, so llama-server on the Spark can push expert tensors to
# ggml-rpc-server workers on other cluster nodes.
#
# Mirrors upstream .devops/cuda.Dockerfile (GGML_BACKEND_DL) with a CUDA 13
# base, sm_121 only, and -DGGML_RPC=ON. The CPU backend is built for exactly
# the GB10's cores (Cortex-X925/A725: Armv9.2-A, SVE2, I8MM, dotprod, FP16, no
# SME) instead of GGML_CPU_ALL_VARIANTS: the armv9.2+sme variant does not
# compile with Ubuntu 24.04's gcc 13 ("invalid feature modifier 'sme'") and
# the GB10 could not use it anyway. The CPU kernels matter here: with
# --n-cpu-moe the routed experts run on these cores.
ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG LLAMA_SHA
ARG CUDA_ARCH=121
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin "${LLAMA_SHA}" && git checkout -q FETCH_HEAD
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DGGML_RPC=ON \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=OFF \
      -DGGML_CPU_ARM_ARCH=armv9.2-a+dotprod+fp16+sve+i8mm \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_CURL=ON \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j "$(nproc)" \
    && mkdir -p /opt/llama \
    && find build -name '*.so*' -exec cp -a {} /opt/llama/ \; \
    && cp build/bin/llama-server build/bin/llama-cli build/bin/llama-bench \
          build/bin/llama-gguf-split build/bin/ggml-rpc-server /opt/llama/ \
    && echo "${LLAMA_SHA}" > /opt/llama/LLAMA_SHA \
    && LD_LIBRARY_PATH=/opt/llama ldd /opt/llama/llama-server | (! grep 'not found') \
    && LD_LIBRARY_PATH=/opt/llama /opt/llama/llama-cli --list-devices 2>&1 | grep 'loaded CPU backend'

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/llama /opt/llama
# ggml finds libggml-<backend>.so next to the executable (GGML_BACKEND_PATH
# would name ONE library file, not a directory), so binaries and libraries
# share /opt/llama.
ENV LD_LIBRARY_PATH=/opt/llama PATH=/opt/llama:$PATH
WORKDIR /opt/llama
ENTRYPOINT ["/opt/llama/llama-server"]
