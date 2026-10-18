# llama.cpp patches

Applied in order (`git apply`) on top of the `LLAMA_SHA` pinned in
`versions.env`, by both `llamacpp-spark` and `llamacpp-rpc-cpu`, so every
binary that talks RPC runs the same sources. Each patch is a plain
`git diff` against that commit; regenerate it (and re-pin `LLAMA_SHA`) when
bumping llama.cpp.

## 0001-prefetch-mmapped-experts

Kimi K3's routed experts (545 GB of a 594 GB GGUF) cannot fit in the Spark's
128 GB, so llama-server runs with `--load-mode mmap --n-cpu-moe 93`: the
expert tensors stay file-backed and the kernel page cache is the expert cache.
Unpatched, a `MUL_MAT_ID` reads each selected expert through the compute
threads' page faults, one synchronous 4 KiB fault at a time, and the kernel
turns off read-around on the mapping after ~100 misses (`mmap_miss` in
`mm/filemap.c`). Measured on the Spark: 147 s per decoded token, 1350 s to
prefill 102 tokens, with `allocstall`/`pgscan_direct` climbing by millions
(the faulting threads also reclaim pages one at a time). Pulsar, which reads
the same experts with large explicit I/Os, does 0.73 tok/s on the same box.

The patch issues `madvise(MADV_WILLNEED)` for the whole slice of every
selected expert before anything touches it, so the reads become large
asynchronous I/Os that overlap with the compute of the earlier experts:

- `ggml/src/ggml-cpu/ggml-cpu.c`: at the start of `ggml_graph_compute`, for
  every `MUL_MAT_ID` node whose ids are an input of the graph (the scheduler
  copies the GPU-computed expert selection into the CPU split before it runs)
  and whose experts live in a host buffer. This is the decode path with
  `--n-cpu-moe`.
- `src/llama-context.cpp` `sched_copy_experts`: before the selected experts
  are uploaded to the GPU for a large batch (the scheduler offloads the
  expert matmuls of a prefill ubatch). This is the prefill path.
- `src/llama-moe-cache.cpp` `copy`: before the misses of the `--moe-cache-mib`
  LRU are uploaded.

Environment knobs (CPU path only):

- `GGML_CPU_MMID_PREFETCH=0` disables the prefetch.
- `GGML_CPU_MMID_COLD=1` additionally `madvise(MADV_COLD)`s each expert after
  its matmul, so reclaim takes the least recently used experts first and does
  not have to rotate referenced pages. Off by default; measure before
  enabling.

Both calls are no-ops for memory that is not file-backed, so the patch is
harmless for models that fit in RAM.
