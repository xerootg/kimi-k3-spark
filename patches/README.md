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
  prefill staging is registered the same way (upstream's
  `ggml_backend_cuda_register_host_buffer` is gated on
  `GGML_CUDA_REGISTER_HOST` and registers read-only, which the GB10
  refused). The CUDA kernels address an expert as `slot * nb[2] / type_size`
  blocks, so the slot stride of a region is the record padded to a multiple
  of every part's block size: no padding for the 80 IQ1_S layers (50-byte
  blocks), ~30 % for the 12 layers with IQ2_XXS parts (66 bytes). Without
  it those layers read misaligned blocks and the output was `????`.
  The bank is enabled where the fabric makes GPU reads of host memory fast:
  an integrated device (GB10) or a discrete one whose pageable access goes
  through the host page tables (GH200 over NVLink-C2C); a PCIe GPU with HMM
  has the pointer but not the bandwidth (`GGML_CUDA_FORCE_HOST_BANK=1` to
  try anyway).
- The CUDA `MUL_MAT_ID` kernels (MMQ, MMVQ) index the expert of a tile as
  `expert * nb[2] / type_size` in a 32-bit int. A normal model has at most
  a few hundred experts; a bank is `n_slots + 1` experts of ~6.6 MB, so on
  the GH200 (49k slots) the index overflowed past slot ~13.7k and the
  kernel read a wild address ("an illegal memory access was encountered"
  on the first prompt that filled that many slots; the Spark's 7.8k slots
  sat just under the limit). The patch applies the sample and channel
  offsets to the `x` pointer in 64 bits (MMQ: strides passed in bytes,
  MMVQ: block stride times the block size) and keeps the 32-bit index for
  the rows within an expert.
- `LLAMA_EXPERT_STORE_CHUNK_KIB` (default 1024, 0 = whole records): the
  pool reads each record as pieces of this size on different threads. One
  6.6 MB O_DIRECT read is latency-bound (2.9 ms on the GH200's virtio NVMe,
  2.3 GB/s), eight pieces in flight finish in 1.2 ms, and a decode fill is
  one or two records, so the fill time is the read latency, not the
  bandwidth. For prefill fills (hundreds of records) it only adds queue
  depth; `LLAMA_EXPERT_STORE_THREADS` should be 32 or so with it.
- With the bank on the GPU the slot maps are registered with it too: the
  `GET_ROWS(slot_map, ids)` lookup runs on the GPU and the CUDA backend
  calls the cache (`ggml_backend_cuda_set_host_bank_hook`) right before
  it, with the selected experts read back from the device, so the misses
  are read into the bank before the lookup and the matmul run. The decode
  graph is then one split instead of three per layer (`-lv 4`: `splits =
  2`). `LLAMA_EXPERT_STORE_GPU_SLOTS=0` keeps the CPU split. Measured on
  the GH200: a small gain (the sync per layer stays), the split overhead
  was not where the time went.
- The MMQ kernel's z dimension walks every expert of `src0`; a bank has
  `n_slots + 1` of them (49k on the GH200) and a batch uses a few hundred
  at most. For `ne02 > 1024` the ids path now compacts the experts with
  columns into a list (`k_mmq_compact_experts`, one small sync to read
  the count back; while a CUDA graph is captured it walks every slot as
  before) and the kernel indexes the list. Prefill of 1024 tokens on the
  GH200: 19.6 → 36.8 tok/s.
- Scan resistance: a fill for a batch of more than 32 tokens (prefill)
  into a full cache does not promote its hits and parks its misses at the
  LRU head, so a long prompt, which touches every expert of a layer, does
  not evict the working set of the generation that follows (it had
  dropped the hit rate from 93 % to 68 %). While the cache is still
  filling every fill is kept, and the victim search steps over the hits
  of the layer being filled (they are not at the tail in this mode; a
  miss evicting one of them left a -1 in the slot map that the GPU then
  read as an expert index: an MMU fault). `LLAMA_EXPERT_STORE_PREFILL_PROMOTE=1`
  restores plain LRU.
- The host-bank hook is asynchronous where the driver API is linked
  (`GGML_CUDA_HOST_BANK_ASYNC`, default on): instead of synchronizing the
  stream before each lookup, a small kernel writes the selected experts to
  a mailbox in mapped host memory and the stream waits on a flag
  (`cuStreamWaitValue32`); a CPU thread spinning on the mailbox runs the
  fill and sets the flag. Nothing blocks the host, so decode graphs are
  captured and replayed as CUDA graphs again (97 % of the graphs on the
  GH200, `GGML_CUDA_GRAPH_STATS=1` prints the counts). Measured: same
  speed as the synchronous hook (11 tok/s warm). The launch overhead was
  not the bottleneck, the GPU waiting on missed experts and the expert
  matmuls over C2C are; it stays on as the base for overlapping reads
  with compute.
- Predictive prefetch (`LLAMA_EXPERT_STORE_PREFETCH=1`, off by default):
  right after layer l's lookup the graph runs layer l+1's router on layer
  l's input (`build_moe_cache_prefetch`, a `moe_prefetch-<il>` lookup that
  the CUDA backend posts to a second mailbox without waiting), and the
  poller reads the predicted misses between real requests. Lossless. On
  the GH200: 83 % of the predicted reads were used and the real misses
  fell from 1.0 to 0.8 per layer, but decode did not get faster (7.53 vs
  7.75 tok/s): the GPU works ~1 ms between a prediction and the next real
  request, a prefetch read takes about as long, and the real request
  queues behind it. Kept for a box where layers take longer than a read.
- Self-prefetch (`LLAMA_EXPERT_STORE_SELF_PREFETCH=1`, off by default, needs
  the slot maps on the GPU): right after the router, the layer's real ids are
  posted as a `moe_prefetch-<il>` lookup (no wait), then the shared expert and
  the latent down projection are queued before the real lookup, so the misses
  are read while the GPU computes them. The real lookup still plans and waits;
  only node order and read timing change. Spark: 1.49 -> 1.60 tok/s,
  bit-identical.
- `GGML_CUDA_OP_PROFILE=N`: CUDA-event timing of every node, printed per
  op kind (host-bank vs device weights) every N graphs. Turns CUDA graphs
  off and inflates tiny kernels; use it for the big ops. GH200 warm decode:
  29.5 ms of expert matmuls from host memory (~340 GB/s over C2C), 32 ms
  of Q6_K dense matmuls in HBM, the rest small ops and miss waits.
- The prefill staging (`n_expert` records, 6.7 GB) is allocated at load
  when prompts will go through it (the default 32-token cache limit), and
  only on first use when `LLAMA_EXPERT_STORE_MAX_BATCH` covers the ubatch
  and it is a fallback (GH200). An image that allocated it lazily
  everywhere pushed the memory-tight Spark into 40 minutes of reclaim
  thrash on the first long prompt. After the load, with the store
  active, the mapping's page cache is released (`madvise(MADV_PAGEOUT)`:
  the spine was copied to the device and the experts come from the
  store), which is ~45 GB that `--moe-cache-mib` can use instead;
  `LLAMA_EXPERT_STORE_KEEP_PAGE_CACHE=1` keeps it.
- The cache regions (and the slot maps) are `mbind`-ed to the NUMA nodes
  that have CPUs. A GH200 exposes the H100's HBM as a CPU-less NUMA node
  that Linux hands out to anyone once the host node runs low, *before*
  reclaiming its page cache: a 390 GiB populate spilled into the HBM and
  the context's compute buffers then failed with `cudaMalloc ... out of
  memory`. Bound, the populate reclaims the host node's page cache
  instead. `LLAMA_EXPERT_STORE_NUMA_BIND=0` disables it.
- `LLAMA_EXPERT_STORE_MAX_BATCH` (default 32): the largest batch served
  from the cache; larger ones take upstream's copy path (every used expert
  read from the store into the pinned staging and uploaded, per ubatch).
  On a box whose cache holds most of the experts (GH200: ~86 %), setting it
  to the ubatch size makes prefill run from the bank with only the misses
  read from the NVMe. It needs the bank on the GPU and is clamped to 32
  otherwise (with the bank on the CPU the scheduler would offload the
  large-batch matmul by copying the strided bank views, which fails in
  the allocator).
- HBM tier (`LLAMA_EXPERT_STORE_HBM_MIB`, off by default): on a GH200 the
  GPU's HBM is a CPU-less NUMA node, and host pages placed there are read
  by the GPU through the same registered mapping at 2-3 TB/s instead of
  ~345 GB/s over C2C (docs/uvm-experiment.md). This many MiB are added to
  the cache and the first slots of each group's region are bound to that
  node (transparent huge pages: 2.6 vs 2.1 TB/s for first reads). Every
  decode fill counts the uses of its experts (halved every
  `LLAMA_EXPERT_STORE_HBM_DECAY` tokens, default 128) and exchanges up to
  `LLAMA_EXPERT_STORE_HBM_PROMOTE` (default 2) of the most used experts in
  host slots with the least used experts in HBM slots, a parallel memory
  swap on the I/O threads (256 KiB pieces, ~0.14 ms per 6.6 MB record)
  while the GPU waits for the fill. Lossless (the same bytes, bit-identical
  output). GH200, 800-token coding generations, warm: 7.77 tok/s without,
  8.39 / 8.87 / 9.31 with 8 / 16 / 24 GiB (21 / 30 / 37 % of the lookups
  served from HBM), 9.49 with THP on the HBM pages and the finer swaps,
  9.58 with decay 512 (deployed; decay 32: 9.19, too many swaps). The size
  is bounded by the HBM left after the spine, KV and compute buffers,
  which are allocated after the cache: ~6.5 GB stay free at 24 GiB.
- Reads through io_uring (default when the build finds liburing;
  `LLAMA_EXPERT_STORE_URING=0` goes back to the threads): each thread that
  fills (the poller, the prefill path) has its own ring and submits the
  whole fill at once, one SQE per record. The size of the pieces does not
  matter (the block layer splits everything at 128 KiB, the PM9E1's MDTS is
  512 KiB); what matters is having the whole fill queued. Microbenchmark of
  the decode pattern (`tools/readbench.cpp`, 7 random records into
  preallocated memory): 4.5 ms with 16 threads preading 1 MiB pieces, 3.9 ms
  with io_uring; single records 1.13 vs 0.80 ms. Registered buffers and
  files bring nothing over plain reads. Spark, end to end: reads 9.6 ->
  10.5-11.1 GB/s, decode 1.37 -> 1.47 tok/s, bit-identical. The threads stay
  for the HBM swaps.
  The io_uring path submits records as `LLAMA_EXPERT_STORE_CHUNK_KIB`
  pieces like the threads do. On the GH200's virtio disk it still loses to
  the 32 threads (8.4 vs 9.6 tok/s: one submitter feeds one blk-mq queue),
  so the GH200 runs with `LLAMA_EXPERT_STORE_URING=0`.
- `LLAMA_EXPERT_STORE_TRACE=<file>` appends every fill as int32
  `[layer, n_ids, ids...]`, for cache studies offline.
- Q6_K matrix-vector product for Hopper (`mmvq.cu`, `GGML_CUDA_MMVQ_Q6K_TMA=0`
  turns it off): a persistent kernel that streams tiles of 2-4 weight rows
  into a 2-stage shared-memory buffer with the bulk-copy engine (TMA,
  L2 evict-first) and the q8_1 activations from shared memory, then runs the
  unchanged `vec_dot_q6_K_q8_1` with the stock thread mapping and reduction
  order, so the output is bitwise identical. Plain GEMVs with >= 2048 rows
  only; the fused gate/up stays on stock (it measured slower in the server,
  `GGML_CUDA_MMVQ_Q6K_TMA_FUSED=1` to try). In isolation 2.45 -> 2.88 TB/s
  on the 7168x12288 attention shape, 3.39 TB/s on the output head
  (`tools/q6k-gemv-bench.cu`); in the K3 decode server the unfused Q6_K time
  goes 20.8 -> 19.2 ms per token.
  `GGML_CUDA_MMVQ_Q6K_TMA=2` also runs it on newer GPUs (the shared memory
  limit comes from the device). On the GB10 it is 5-10 % faster than stock
  but not bit-identical (stock uses the GB10's own warp layout there), so it
  stays opt-in until a variant mirrors that layout.
- Cache snapshot (`LLAMA_EXPERT_STORE_SNAPSHOT=<file>`, off by default): at
  exit and every `LLAMA_EXPERT_STORE_SNAPSHOT_EVERY` seconds (default 300,
  0 = exit only) the cache writes which experts are resident, in recency
  order with their use counts: ~6 bytes per slot, not the bytes themselves,
  which the store still holds. The next start reads the file back, places the
  most used experts in the HBM slots and the rest in recency order, and
  re-reads every record from the store before serving, so the first token
  runs on a warm cache instead of the first minutes of misses. The file is
  tied to the store (size), the model (layers, experts) and the group layout
  (record sizes); a cache size change keeps the most recent entries that
  fit, and anything that does not match starts cold. Spark: 7980 experts,
  52.5 GB re-read in 5.5 s, the first run after a restart at 1.59 instead of
  ~1.3 tok/s, hit rate 85 % from the start instead of 48 %, bit-identical.
  The pod needs a writable mount for the file and a termination grace
  period that covers the exit write (manifests: `/snapshot`, 120 s).
- `LLAMA_EXPERT_STORE_SPIN_US` (default 0): the I/O threads and the thread
  waiting for a fill busy-wait this long before sleeping, to skip the wake-up
  of a parked thread (or vCPU) on the next fill. Measured neutral on the
  GH200 (11.47 vs 11.54 tok/s): the gap between the engine's ~1 ms per missed
  record and the microbenchmark's ~0.4 ms is not wake-up latency. Kept as an
  opt-in for the revisit list.
- Two I/O pool fixes. A worker of the previous run could still be between
  its last `fetch_sub` and the `fetch_add` that ends its loop when the next
  run reset the counters; it then ran a job of the new run twice and
  `remaining` hit zero with a read still in flight. `run()` now waits until
  no worker is inside the previous run, and runs are serialized.
- Transparent huge pages are on for the HBM part only. On the host part
  they are opt-in (`LLAMA_EXPERT_STORE_THP=1`): on the Spark, with them,
  4 of 7 server starts produced NaN from the first prefill (the model
  answers "GGGG..."), and 0 of 4 without. The cause is not pinned down; the
  GPU-registered staging the prefill uploads from is the suspect. Hugepage
  destinations would make O_DIRECT faster (3.9 vs 4.5 ms per fill in the
  microbenchmark, one IOMMU mapping per page), but the load's reclaim split
  most of them anyway (12 of ~50 GiB stayed huge).

Stats are logged every 4096 fills and at exit (hit rate, bytes read for
decode and prefill). Patches 0001 and 0002 are untouched by this; 0001's
prefetch hook sees the bank tensors (anonymous memory) and is a no-op there.
