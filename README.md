# kimi-k3-spark

Container builds and notes for running [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
(2.8T-parameter MoE: 93 layers, 896 experts/layer, top-16 + 2 shared, KDA +
gated-MLA attention) on a **single NVIDIA DGX Spark** (GB10, 128 GB unified
memory, one Gen5 NVMe) with help from a heterogeneous Kubernetes cluster.

Deployment manifests live in the `argo-things` GitOps repo (`configs/kimi-k3`).

## Images

| Image | Arch | Contents |
|---|---|---|
| `ghcr.io/xerootg/llamacpp-spark` | arm64 | llama.cpp, CUDA 13 sm_121, `-DGGML_RPC=ON` (llama-server, llama-cli, llama-bench, ggml-rpc-server) |
| `ghcr.io/xerootg/llamacpp-rpc-cpu` | amd64 | ggml-rpc-server, CPU backend, all CPU variants |
| `ghcr.io/xerootg/pulsar-spark` | arm64 | [Pulsar](https://github.com/giannisanni/pulsar) SSD-streaming engine, sm_121 |

Tags are the pinned upstream commit (`versions.env`) plus `-g<short sha>` of
this repo. The Spark `llama-server` and the x86 `ggml-rpc-server` workers must
share a `LLAMA_SHA`. Both llama.cpp images apply [`patches/`](patches/README.md)
on top of that commit; the expert prefetch patch (0001) is what makes an
mmap'd, larger-than-RAM expert set usable at all (unpatched: 147 s per token
on the Spark, patched: 4.45 s). Patch 0002 (zero-copy GPU reads of the
mmap, `GGML_CUDA_HOST_ZERO_COPY=1`) is built in but off: measured at ~19 s
per token on the GB10, whose GPU reads of mapped pages go through UVM
fault handling rather than straight ATS. Patch 0003 (the expert store,
`LLAMA_EXPERT_STORE` + `--moe-cache-mib`) is the production engine:
0.71 tok/s decode and 102 tokens of prefill in 54 s on the Spark, with a
52 GB host LRU hitting 55 % of the lookups. Numbers in the argo-things
README.

## Validating a patch without the Spark

Patch 0003 was checked on a workstation with an Intel iGPU (Vulkan backend)
and a 1.3B MoE (`bartowski/granite-3.0-1b-a400m-instruct-GGUF`, Q8_0): same
topology as production (`-ngl 99 --n-cpu-moe 99`, spine on the GPU, experts
in host memory), `tools/repack-experts.py --verify 50` on the GGUF, then
`llama-completion` at temperature 0 with and without
`LLAMA_EXPERT_STORE=... --moe-cache-mib N`. Generated text is token-identical
for a 128 MiB cache (79 slots, 10 %, hit rate 0.02 %, every expert read from
the store) and a 1024 MiB cache (641 slots, 83.5 %, hit rate 89.5 %); the
59-token prompt exercises the prefill staging path, the 47 decode steps the
graph hook. `-lv 4` shows the store lines (`expert store ...`, `MoE cache
size`, and the hit/miss summary at exit).

## Why these numbers

Unsloth `UD-IQ1_S` is 594 GB. Decode touches 16 experts in each of 92 MoE
layers; at ~6.6 MB per expert that is ~9.8 GB of expert weights per token, so
with everything streamed from NVMe (~7 GB/s at expert-sized reads) the floor is
~1.4 s/token. Pulsar's reported 0.70 tok/s on a Spark matches that exactly:
the engine is NVMe-bound, and the only levers are (a) more experts resident in
RAM anywhere that can compute them, (b) speculative decoding so one expert
fetch serves several tokens, (c) batching clients so one fetch serves several
sequences.

GB10 notes: the GPU can address `mmap`'d file pages (HMM/ATS, unified
LPDDR5x) but GPU-side demand faults are 4 KiB each; prefault on the CPU
(`MADV_POPULATE_READ`, io_uring O_DIRECT) then let the GPU read. cuFile/GDS
adds nothing here because there is no separate VRAM to DMA into.
