# llama.cpp for the NVIDIA DGX Spark (GB10, sm_121, aarch64) with the RPC
# backend enabled, so llama-server on the Spark can push expert tensors to
# ggml-rpc-server workers on other cluster nodes.
#
# Mirrors upstream .devops/cuda.Dockerfile (GGML_BACKEND_DL + all CPU variants,
# runtime-selected) with a CUDA 13 base, sm_121 only, and -DGGML_RPC=ON.
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
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_CURL=ON \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build -j "$(nproc)" \
    && mkdir -p /opt/llama/bin /opt/llama/lib \
    && find build -name '*.so' -exec cp {} /opt/llama/lib/ \; \
    && cp build/bin/llama-server build/bin/llama-cli build/bin/llama-bench \
          build/bin/llama-gguf-split build/bin/ggml-rpc-server /opt/llama/bin/ \
    && echo "${LLAMA_SHA}" > /opt/llama/LLAMA_SHA

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/llama /opt/llama
ENV LD_LIBRARY_PATH=/opt/llama/lib PATH=/opt/llama/bin:$PATH \
    GGML_BACKEND_PATH=/opt/llama/lib
WORKDIR /opt/llama
ENTRYPOINT ["/opt/llama/bin/llama-server"]
