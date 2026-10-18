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

## 0002-zero-copy-host-experts

Opt-in (`GGML_CUDA_HOST_ZERO_COPY=1`), CUDA backend only. The GB10 is an
integrated GPU that reads pageable host memory through the host page tables
(`cudaDevAttrPageableMemoryAccess` and `...UsesHostPageTables` are both 1 on
the Spark), so its kernels can read a memory-mapped model file in place. With
the flag set, the CUDA device accepts llama.cpp's `CPU_Mapped` buffers, and
the scheduler keeps every `MUL_MAT_ID` on the GPU with the experts still in
the mapping: no CPU matmul for decode and no 545 GB `cudaMemcpy` of experts
per prefill chunk (which capped prefill at ~1.4 GB/s with patch 0001 alone).

A GPU page fault is serviced one page at a time through the IOMMU, so the
pages must be resident *and mapped* before a kernel touches them. Right before
a zero-copy `MUL_MAT_ID` runs, the backend synchronizes its stream, reads the
selected expert ids back, and on the host issues `MADV_WILLNEED` for every
selected expert slice (all reads in flight at once) followed by
`MADV_POPULATE_READ` (wait and map). CUDA graphs are disabled for such
graphs because the prefetch is host work between kernels.

Only buffers named `CPU_Mapped` are accepted, never the scheduler's own host
buffers: graph inputs keep being copied to the device, so llama.cpp's
`set_inputs` for the next ubatch cannot overwrite memory a kernel is still
reading (llama.cpp only synchronizes before `set_inputs` with pipeline
parallelism).

Knobs: `GGML_CUDA_HOST_ZERO_COPY=1` enables it; `GGML_CUDA_HOST_PREFETCH`
0 = no prefetch (GPU faults everything in, do not do this), 1 = `WILLNEED`
only, 2 = `WILLNEED` + `POPULATE_READ` (default). Patch 0001 stays in: with
the experts on the GPU its CPU-side hook simply finds no `MUL_MAT_ID` nodes.

Measured on the Spark: ~19 s per token. The GB10 services GPU reads of
mapped pages through the UVM fault handler, not straight through ATS, so
this stays opt-in and off.

## 0003-expert-store

The engine for a model whose experts do not fit in RAM: the routed experts
are served from an **expert store** (`tools/repack-experts.py`: one 4 KiB
aligned record per (layer, expert) holding gate | up | down back to back)
with O_DIRECT reads into a host-RAM LRU, and the page cache is out of the
expert path entirely. Enabled with `LLAMA_EXPERT_STORE=<store file>` plus
`--moe-cache-mib <RAM budget>`; the experts still have to be placed in host
memory (`--n-cpu-moe`), the mapping stays for the spine and for RPC-bound
layers.

Built on upstream's `llama_moe_cache` (slot maps, per-group LRU, bank views
read by `MUL_MAT_ID`), with these changes:

- The cache device is the CPU backend. Each group's banks are strided views
  into one `mmap`'d, THP-advised, populated region of records: the expert
  stride of a bank is the record size, so one 6.5 MB read fills gate, up and
  down of an expert at once. The CPU is not a model device, so it takes the
  whole `--moe-cache-mib` budget instead of a `--tensor-split` share
  (upstream's split lookup asserted on the Spark's `--tensor-split 1,0,0,0`).
- Small batches (decode): the slot-map lookup runs in the CPU graph, so no
  scheduler copy happens and no copy callback fires. ggml-cpu gets a graph
  hook (`ggml_cpu_set_graph_hook`, exposed through the backend registry) that
  runs on the thread starting a CPU graph; the cache registers itself, finds
  the `GET_ROWS(slot_map, ids)` nodes of the graph (the ids are an input of
  the split, already copied in), plans the LRU, and reads the misses from the
  store with a pool of `LLAMA_EXPERT_STORE_THREADS` (16) O_DIRECT `pread`s
  before the matmuls run.
- Large batches (prefill): upstream offloads the matmul to the GPU and asks
  the copy callback for the used experts of each tensor. The callback now
  reads the used records once per layer into a pinned staging area of
  `n_expert` records and uploads each tensor's part from there; the mapping
  is never touched. The staging is synchronized against the previous layer's
  uploads before it is overwritten.
- The loader excludes the host expert tensors from the mapping prefetch
  (`lazy_read::exclude_prefetch`): no `MAP_POPULATE` of 545 GB that only
  evicts itself, the spine and RPC-bound experts are still prefetched.
- GPU-side expert matmuls from the host cache ("host bank"): on an
  integrated device whose kernels can use host pointers for registered
  memory (GB10: `cudaDevAttrIntegrated` and
  `cudaDevAttrCanUseHostPointerForRegisteredMem` both 1), the cache regions
  are `cudaHostRegister`ed and wrapped as `CUDA_HostBank` buffers that the
  CUDA backend accepts as weights (`ggml_backend_cuda_host_bank_buffer_from_ptr`,
  via the registry). The scheduler then runs the small-batch `MUL_MAT_ID`
  on the GPU straight from the cache, while `GET_ROWS(slot_map)` stays on
  the CPU so the graph hook still fills the misses first. Unlike patch
  0002's mapped file pages, registered anonymous memory is mapped into the
  GPU up front: no UVM faults. `LLAMA_EXPERT_STORE_GPU=0` keeps the matmuls
  on the CPU, `GGML_CUDA_NO_HOST_BANK=1` disables the buffer type. The
  prefill staging is only pinned when `GGML_CUDA_REGISTER_HOST=1` is set
  (upstream gates `ggml_backend_cuda_register_host_buffer` on it).

Stats are logged every 4096 fills and at exit (hit rate, bytes read for
decode and prefill). Patches 0001 and 0002 are untouched by this; 0001's
prefetch hook sees the bank tensors (anonymous memory) and is a no-op there.
