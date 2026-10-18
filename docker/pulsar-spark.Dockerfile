# Pulsar (https://github.com/giannisanni/pulsar) for the DGX Spark: SSD-streaming
# MoE engine (Rust + CUDA, io_uring O_DIRECT, host LFU expert cache). Kimi K3
# loads since PR #24 (ssm_f_b as f32). PULSAR_CUDA_ARCH=121 is the only GB10
# change needed (classmethod's single-Spark K3 run).
ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build
ARG PULSAR_SHA
ARG CUDA_ARCH=121
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential cmake curl pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
RUN git init -q . && git remote add origin https://github.com/giannisanni/pulsar \
    && git fetch -q --depth 1 origin "${PULSAR_SHA}" && git checkout -q FETCH_HEAD
ENV PULSAR_CUDA_ARCH=${CUDA_ARCH}
RUN cargo build --release -p engine -p serve \
    && cargo build --release -p tokenizer --bin get-chat-template \
    && mkdir -p /opt/pulsar/bin \
    && cp target/release/pulsar-cli target/release/pulsar-serve target/release/get-chat-template /opt/pulsar/bin/ \
    && echo "${PULSAR_SHA}" > /opt/pulsar/PULSAR_SHA

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libssl3t64 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/pulsar /opt/pulsar
ENV PATH=/opt/pulsar/bin:$PATH
WORKDIR /opt/pulsar
ENTRYPOINT ["/opt/pulsar/bin/pulsar-serve"]
