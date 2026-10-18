# ggml-rpc-server (CPU backend) for x86-64 cluster nodes. Holds a slice of
# Kimi K3's routed-expert tensors in RAM and runs their FFN locally, so only
# activations cross the (1-2.5 GbE) LAN. Must be built from the SAME llama.cpp
# commit as the llama-server that talks to it: the RPC protocol is not
# versioned across commits.
ARG UBUNTU_VERSION=24.04

FROM docker.io/ubuntu:${UBUNTU_VERSION} AS build
ARG LLAMA_SHA
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential cmake ninja-build libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp \
    && git fetch -q --depth 1 origin "${LLAMA_SHA}" && git checkout -q FETCH_HEAD
RUN cmake -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF \
      -DGGML_RPC=ON \
      -DGGML_BACKEND_DL=ON \
      -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_CURL=ON \
    && cmake --build build -j "$(nproc)" \
    && mkdir -p /opt/llama \
    && find build -name '*.so*' -exec cp -a {} /opt/llama/ \; \
    && cp build/bin/ggml-rpc-server build/bin/llama-bench build/bin/llama-cli /opt/llama/ \
    && echo "${LLAMA_SHA}" > /opt/llama/LLAMA_SHA \
    && LD_LIBRARY_PATH=/opt/llama ldd /opt/llama/ggml-rpc-server | (! grep 'not found') \
    && LD_LIBRARY_PATH=/opt/llama /opt/llama/llama-cli --list-devices 2>&1 | grep 'loaded CPU backend'

FROM docker.io/ubuntu:${UBUNTU_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libcurl4t64 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/llama /opt/llama
# ggml finds libggml-<backend>.so next to the executable (GGML_BACKEND_PATH
# would name ONE library file, not a directory), so binaries and libraries
# share /opt/llama.
ENV LD_LIBRARY_PATH=/opt/llama PATH=/opt/llama:$PATH
WORKDIR /opt/llama
ENTRYPOINT ["/opt/llama/ggml-rpc-server"]
