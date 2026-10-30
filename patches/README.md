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
  with compute. What crosses the mailbox is read by the poller when the
  GPU *reaches* the request, which with eager launches (no CUDA graph:
  batches of more than one token) can be long after the host launched it
  and after the host has built the next ubatch's graph in the same
  scheduler memory. So the mailbox carries the slot map (a tensor the cache
  owns) and the kind of lookup by value, never the lookup node: passing
  the node made the poller read a tensor of the next graph, fill the wrong
  layer or none, and the lookup then read -1 as an expert slot. That was
  the "two sequences decoding together crash" on the GH200 (a second
  client's prompt joining a running sequence makes a multi-ubatch decode;
  the fix is in the hook's signature, `ggml_backend_cuda_host_bank_hook_t`).
  `GGML_CUDA_HOST_BANK_CHECK=1` validates every lookup's slot indices on the
  GPU and reports the first bad one with its node (a diagnostic: it
  synchronizes at the end of every graph).
- Predictive prefetch (`LLAMA_EXPERT_STORE_PREFETCH=1`, off by default, needs
  the slot maps on the GPU and io_uring): before layer l's lookup the graph
  builds layer l+1's router on layer l's input (`build_moe_cache_prefetch`, a
  `moe_prefetch-<il>` lookup that the CUDA backend posts to a second mailbox
  without waiting); `build_moe_cache_slots` expands it right behind the real
  lookup, so the GPU posts it the moment the lookup is served, which is when
  the drive goes idle. Lossless: the real lookup still plans the real ids.
  The first version read the predictions through the I/O pool on the poller
  thread, at full depth: the next real post queued behind the whole
  speculative fill and the speculative reads saturated the drive just when
  the real ones arrived (Spark 1.33 vs 1.41 tok/s, GH200 7.53 vs 7.75).
  The second version (2026-10-21) reads them on a side thread with its own
  ring and a bounded number of pieces in flight
  (`LLAMA_EXPERT_STORE_PREFETCH_DEPTH`, default 8 of the pool's chunk), at
  most `LLAMA_EXPERT_STORE_PREFETCH_MAX` records per post (default 16), best
  router scores first. A real fill preempts it: no new record is started
  until the next prediction, records already started are finished and
  adopted if the real fill selects them (it waits instead of re-reading;
  a half-read record is never resident), records not started are handed
  back and their slots emptied. The LRU is only touched on the poller
  thread; a per-slot state keeps the reader's slots from being chosen as
  victims or HBM exchanges. `LLAMA_EXPERT_STORE_PREFETCH_DEMOTE=1` parks
  predictions the real fill did not select at the eviction end;
  `LLAMA_EXPERT_STORE_PREFETCH_INPUT=1` applies the next router to the
  un-normed FFN input plus the shared expert's output under the next
  layer's FFN norm instead of this layer's router input. Spark, with
  self-prefetch: 1.60 -> 1.80 tok/s with depth 8 and `PREFETCH_INPUT=1`
  (1.755 with the router input, 1.71 at depth 4, demote neutral), the
  self-prefetch fill 6.7 -> 4.1 records, ~4.8 records started per post of
  which 70 % are used, bit-identical. Deployed on the Spark. It also works
  with the slot maps on the CPU (`LLAMA_EXPERT_STORE_GPU_SLOTS=0`, 2026-10-22):
  the prediction is then computed on the device before the layer's real
  router and its lookup follows the real lookup in the same CPU graph (no
  extra split), where the CPU graph hook hands it to the side reader right
  after the real fill; and the side reader uses `depth` pread threads instead
  of a ring when the pool does (`LLAMA_EXPERT_STORE_PREFETCH_URING`, default
  as the pool, which is what the virtio disk of the GH200 needs). Measured on
  the GH200 in that form: 13.84 -> 11.3-11.5 tok/s in every variant (at 99 %
  residency the wrong guesses evict resident experts and the predictor's GPU
  time exceeds what the reads could give back), so it stays off there
  (docs/experiments.md).
  `LLAMA_EXPERT_STORE_PREFETCH_INPUT=2` (Kimi K3 only) feeds the next router
  the stream before this layer's FFN plus the shared expert, mixed the way
  the next layer's FFN mixes its residual checkpoints (`res_mix_pred`: the
  next layer's `ffn_res_score`; on a checkpoint layer the stream stands in
  as the new checkpoint and the unknown attention output is zero), under the
  next FFN norm. Offline on 400 decode tokens: top-16 precision 0.675 vs
  0.614 for input 1, the checkpoint layers (every 12th) 0.48 vs 0.34 and the
  layer after them 0.58 vs 0.46; Spark A/B +2 %, self-prefetch fill 4.1 -> 3.8
  records. `tools/k3-pfdump` (an eval-callback dump of the predictor and the
  real router per token and layer) and `LLAMA_EXPERT_STORE_PREFETCH_DUMP=1`
  (more predictor variants as named tensors, measurement only) are what
  measured it; `tools/pfeval.py` (kimi-k3-spark) replays the pipeline through an LRU.
- Self-prefetch (`LLAMA_EXPERT_STORE_SELF_PREFETCH=1`, off by default, needs
  the slot maps on the GPU): right after the router, the layer's real ids are
  posted as a `moe_prefetch_self-<il>` lookup (no wait), then the shared expert and
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
  goes 20.8 -> 19.2 ms per token. End to end on the GH200 (2026-10-26, A/B
  with `=0` on the same build, md5 ac78af0e5d in every run): steady decode
  12.99 -> 13.25 tok/s (+2 %, ~1.3 ms per token).
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
- Memory lock (`LLAMA_EXPERT_STORE_MLOCK`, default 1): at the first fill,
  when the model, the context and the GPU buffers are in place, the engine
  mlocks every mapping of the process except the model files (`.gguf`, the
  store) and device mappings: the bank and the staging as resident
  (`MLOCK_ONFAULT`), the heap, the small anonymous regions and the libraries
  populated. On the Spark that is ~63 GB in 5-17 s, once. Why: after a load
  the box has ~7 GB of headroom and the server has ~150 MB in swap (an 86 MB
  anonymous region, part of the heap, libcublasLt's code); every first touch
  of such a page is a major fault served behind the expert reads that keep
  the NVMe queue full, ~10 ms each, and a few hundred in a row stall a
  request for seconds. Needs `RLIMIT_MEMLOCK` to cover it or `CAP_IPC_LOCK`
  (the pod adds the capability); otherwise it warns and runs unlocked.
  Measured neutral on tok/s (1.82 vs 1.82), `VmSwap` 0 afterwards.
- Split expert GEMV (`LLAMA_EXPERT_STORE_SPLIT=1`, off by default, needs the
  slot maps on the GPU and no HBM tier; single-token decode only): the
  resident experts' matmuls run while the layer's misses are read. The graph
  looks the selected experts up twice. The early lookup (`moe_early-<il>`,
  on an "early map") is a request the poller answers after the plan alone: it
  publishes the slots of the experts already complete in the bank, points the
  rest at a zero record (one extra, zeroed slot at the end of each group's
  region, `n_slots`), starts the misses' reads on a fill thread and returns.
  The GPU then runs the shared expert, the expert matmuls over those
  n_expert_used + 1 ids (the last one the zero record: a scratch row) and
  the next layer's prefetch predictor, all under the fill. The real lookup
  (`ffn_moe_slots`, now on a position-indexed "late map" read with a constant
  0..n-1 id list) waits for the fill thread and yields the misses' slots
  packed first, the zero record after; a second matmul over those
  `LLAMA_EXPERT_STORE_SPLIT_LATE` (default 8) rows computes the misses, and
  `ggml_set_rows` copies each miss's row into the early output at its
  position (a "merge map", also by position; the zero record's rows land in
  the scratch row). Every expert's row is computed once, from the same slot,
  by the same `mul_mat_vec_q` block as in the single lookup, and the weighted
  sum runs in the same order, so the output is bit-identical (gold md5 on
  every run). A layer with more misses than late rows reads the excess at
  the early lookup (the speculative reader's records are adopted first), so
  the late matmul never overflows. The zero record's rows cost real
  bandwidth (the GPU does not cache the host bank in L2: ~28 us per dummy
  id for the three matmuls), which is why the late matmul is kept short.
  `PREFETCH_INPUT=3` feeds the predictor the early output's weighted sum
  (the resident experts' share of the routed output) through the routed norm
  and up projection, on top of input 2.
  Spark A/B (2026-10-22, docs/experiments.md): 1.85 -> 1.91-1.93 tok/s
  (+3.4 %, +1 % more with `PREFETCH_INPUT=3`), 11.6 of 16 experts complete
  at the early lookup, the GPU's wait at the real lookup 2.8 -> 1.2-1.4 ms,
  gold md5 on all 21 runs. 16 late rows gave 1.91 (the dummies cost more
  than the synchronous excess of 8 rows saves).
- Memory pressure valve (`LLAMA_EXPERT_STORE_VALVE`, default 1): the cache
  sizes itself to what the box has left instead of trusting the hand-picked
  `--moe-cache-mib`. Why: that number is only ever checked with short
  prompts; a long prompt grows the CUDA compute buffers by several GB (a
  30k-token prompt on the GH200 added ~10 GB on the HBM node), the driver
  and the server allocate a little more over time, and when the host node
  reaches its min watermark the kernel's answer is SIGKILL (the bank is
  anonymous, locked, and there is no swap): the GH200 pod was killed that
  way on 2026-10-22 with 441 GB of anonymous memory on a 449 GB node. A
  thread samples the CPU nodes' `MemAvailable` (the system figure minus the
  HBM node's free memory, which host allocations never get) and, with an HBM
  tier, the HBM node's `MemFree`, every `LLAMA_EXPERT_STORE_VALVE_INTERVAL_MS`
  (500) and whenever a PSI trigger fires (`/proc/pressure/memory`,
  `some 150000 2000000`: tasks stalled on reclaim for 150 ms within 2 s,
  the kernel's only early warning; it needs root on these kernels, the
  timer is the fallback). Under the reserve
  (`LLAMA_EXPERT_STORE_RESERVE_MIB` 4096 for the host,
  `LLAMA_EXPERT_STORE_HBM_RESERVE_MIB` 6144 for the HBM node) it releases
  the least recently used slots of that domain in one batch sized to the
  deficit plus a quarter-reserve band: `munlock` + `MADV_DONTNEED` give the
  pages back, the mapping stays, the experts become misses. Only the whole
  pages inside a run of released slots are dropped: on the GH200's 64 KiB
  pages a slot boundary is almost never page aligned, and the first build
  rounded outwards, zeroing the edge of a resident neighbour (wrong md5) or
  failing with EINVAL; a release that gives back less than half of what it
  took now stops the valve instead of emptying the cache. The GPU reads the
  bank through the host page tables on both boxes (registered, but
  coherent), so a dropped page is never read stale: measured with a
  registered 1 GiB buffer, `MADV_DONTNEED`, a CPU rewrite and a GPU sum. Released slots
  are parked (the plan never takes them, they sit at the tail of the LRU,
  and they count as full for the prefill scan resistance: counting them as
  empty switched it off and cost the GH200 20 %)
  until the domain has been above the reserve plus two bands for
  `LLAMA_EXPERT_STORE_VALVE_REGROW_S` (10) seconds, then they are re-armed
  256 at a time (`MLOCK_ONFAULT` again, the pages return on the next fill).
  Never released: the slots of the last layers filled (the GPU may be
  reading them), slots the speculative reader is writing, and the last
  `LLAMA_EXPERT_STORE_VALVE_FLOOR` (50) percent of a domain's slots, and
  the split GEMV's zero record (slot `n_slots`, outside the LRU). The
  fill hooks and the valve share a lock around the LRU, so a release only
  runs between fills. Lossless: a record is dropped whole or not at all.
  The cgroup's `memory.events` are read too and a kill or a `memory.high`
  breach is logged. The periodic stats line reports the lowest
  `MemAvailable` seen, the slots released and re-armed, and the PSI
  wake-ups. The prefill staging is not covered (a large batch may need it
  at any time).
- Prefill lookahead (`LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD=1`, off by
  default, CPU split only): after the real fill of layer il of a batch of at
  least 8 ids per expert (`LLAMA_EXPERT_STORE_PREFILL_LOOKAHEAD_MIN`), every
  expert of layer il+1 is planned (scan resistant; the victim search steps
  over layer il, which the GPU computes meanwhile, `moe_cache_lru::skip_layer`)
  and the misses are read on the fill thread. Any later fill first waits for
  those reads (`la_finish`). A large batch touches nearly every expert of a
  layer, so "all of them" is the prediction. The memory valve treats the
  layer in flight as protected. GH200, 16k-token prompt: 98-106 -> 127-145
  tok/s. A stats line per 4 ubatches (`expert store: prefill: ...`) reports
  the records read in the fill and ahead, and the wait for the lookahead.
- `LLAMA_EXPERT_STORE_SPIN_US` (default 0): the I/O threads and the thread
  waiting for a fill busy-wait this long before sleeping, to skip the wake-up
  of a parked thread (or vCPU) on the next fill. Measured neutral on the
  GH200 (11.47 vs 11.54 tok/s): the gap between the engine's ~1 ms per missed
  record and the microbenchmark's ~0.4 ms is not wake-up latency. Kept as an
  opt-in for the revisit list.
- Second chance (`LLAMA_EXPERT_STORE_SECOND_CHANCE=N`, default 0 = plain
  LRU): one byte per slot. Every hit on a resident expert raises its slot's
  count (at most N); the victim search at the head of the recency list
  spends a count instead of evicting (count--, slot to the tail) and takes
  the first slot with none. New occupants, demoted slots (unused predictions,
  prefill scans, valve releases) and the HBM exchange start at 0, and a
  prefill scan takes the head as LRU does. Frequency on top of recency, the
  classic CLOCK second chance. Chosen on the replay of the real top-16
  selections of the two `k3-pfdump` traces (400 coding and 300 prose decode
  tokens) at the Spark's 7980 slots, scored after 150 tokens: LRU 57.5 / 61.7
  % hits, N=3 59.6 / 64.1, S3-FIFO 59.5 / 64.7, ARC 58.5 / 61.8, LFU with
  decay 58.9 / 64.3, Belady 76.6 / 79.2; with the deployed speculative
  prefetch (window 5) LRU 73.4 / 76.4 and N=3 74.4 / 77.8. About 5 % fewer
  misses per layer for one byte per slot, so the simplest of the winners
  went in. Decode output does not depend on which slot holds an expert
  (checked with granite-3.0-1b-a400m on Vulkan: the text of 0 and 3 at
  128 and 1024 MiB equals the no-store baseline; at 1024 MiB, 83.5 % of the
  experts, misses 1044 -> 909).
- `LLAMA_EXPERT_STORE_CPUS=<cpu list>` (e.g. `0-9`, unset = no pinning): the
  I/O pool workers, the speculative reader (its thread and pread workers), the
  split's fill thread and the CUDA host-bank poller pin themselves to these
  CPUs on entry (`sched_setaffinity`; the poller has its own copy of the
  parser in ggml-cuda.cu, the backend cannot call into llama). For the GB10's
  10 X925 + 10 A725: the poller and the readers are on every layer's critical
  path and the scheduler is free to put them on a little core, which is what
  `--threads 8 > 16` already showed for the matmul threads. Which CPU numbers
  are the X925s is read from `/sys/devices/system/cpu/cpu*/cpu_capacity`
  on the box.
- Remote experts (`LLAMA_EXPERT_STORE_REMOTE=host:port[,host:port...]` plus
  `LLAMA_EXPERT_STORE_REMOTE_SET=<one file of "layer expert" lines per box,
  same order>`, off by default, needs the split GEMV): other boxes hold
  static sets of records (`tools/distributed/remoteset.py --boxes` places
  them, `remoteexport.py` / `remotecompose.py` build the files) and compute
  those experts' rows for a single token with a bit-exact kernel
  (`tools/bitexact/sycl`). At the early lookup, the selected experts of a
  usable box's set that are not in the bank are left out of the plan (no
  slot, no NVMe read, the zero record in the early matmul), as many as the
  late matmul could still take (`LLAMA_EXPERT_STORE_REMOTE_ROWS`, default 8,
  and SPLIT_LATE minus the local misses), grouped by box. The graph copies
  the experts' input (`routed_in`, f32) into a per-layer registered vector
  right before the early lookup, and the early lookup itself hands input +
  global ids (`layer * n_expert + expert`) to each box involved as soon as
  it has chosen the candidates, before it plans the local reads.
  (`LLAMA_EXPERT_STORE_REMOTE_LATE_POST=1` restores the first version: the
  copy after the early lookup and a separate waiting post `moe_rpost-<il>`;
  that sent the request ~130 us later and cost the GPU one more wait on the
  poller per layer.) One client per box: a persistent TCP connection
  (`TCP_NODELAY`), its own network thread, requests pipelined (wire format
  of `tools/distributed/rrclient.c`, header and input in one `sendmsg`).
  The network thread busy-polls while a request is in flight and for
  `LLAMA_EXPERT_STORE_REMOTE_SPIN_US` (300) after its last work, then
  blocks on a futex doorbell that the post rings (it used to nap 100 us at
  a time, which cost ~75 us per request: the requests are one per layer,
  3-8 ms apart). `LLAMA_EXPERT_STORE_REMOTE_CPUS=<cpu>[,<cpu>...]` pins the
  network thread of box i to the i-th CPU (the last one repeats; unset:
  `LLAMA_EXPERT_STORE_CPUS` or no pinning). Each box has
  its own deadline (`LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US`, one value or one
  per box, default 3000, from the post) and stats. The real lookup, after the
  local reads, waits for each box until its deadline and places the rows
  that arrived (a partial reply's complete rows too; rows with a NaN, which a
  server answers for ids it does not hold, are refused) in the layer's remote
  rows; a second `ggml_set_rows` after the late merge puts them in the early
  output. Every row that did not arrive is read from the store into the late
  matmul's free rows instead, so the output never depends on the network,
  only on the servers computing the same rows as the GPU. Reliability: a
  broken link, a send that stalls (200 ms) or a request unanswered for
  `LLAMA_EXPERT_STORE_REMOTE_STALE_MS` (1000) drops the connection and frees
  its requests; reconnects use a non-blocking connect (500 ms) with
  exponential backoff (50 ms .. 2 s), so a box that comes back is used again
  without a restart; `LLAMA_EXPERT_STORE_REMOTE_TRIP` (8) deadline misses in a
  row skip a box for `LLAMA_EXPERT_STORE_REMOTE_TRIP_MS` (2000), then one
  request probes it. A box that is down only costs its experts' reads. The
  speculative reader skips the experts of usable boxes.
  While the remote experts are on, the server holds `/dev/cpu_dma_latency`
  (`LLAMA_EXPERT_STORE_REMOTE_DMA_LATENCY`, default 0 us, -1 = leave it): on
  the GB10 the cores that take the NIC's interrupts otherwise sleep in a
  deep idle state (~430 us exit) between requests. It needs write access to
  the device (root, a udev rule, or in a container the device mapped in);
  without it a warning is logged and the round trips are just slower.
  `LLAMA_EXPERT_STORE_REMOTE_SHADOW=1` sends and times the requests but reads
  every expert locally (gold md5 with any server). The periodic stats print
  one line per box: posts, rows per post, rows placed, short requests and
  rows read locally, partial replies, NaN rows, breaker trips, round-trip
  percentiles, connects and drops.
  `LLAMA_EXPERT_STORE_REMOTE_TRACE=<path>` (diagnostics, off by default)
  records every request's hand-offs (CLOCK_MONOTONIC_RAW ns): the early
  lookup, the `moe_rpost` hook (= the early lookup with the early post),
  the cache lock, the post, the network thread's pick-up and send, the
  reply's first byte, first row and last byte, the real lookup's hook, the
  resolve, the wait's end, the rows placed, plus the network thread's cpu,
  involuntary switches and idle blocks (doorbell waits). The ring (65536 requests per box) is allocated at start and
  written as TSV to `<path>.<box>` on SIGUSR1 (when nothing is in flight)
  and at shutdown. With the trace on, header word 16 carries the request's
  number (k < 16; servers ignore ids past k), so a k3x `K3X_TRACE` dump
  joins request by request (`tools/bitexact/sycl/trace_report.py`).
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

## 0004-byte-stride-slots

The MoE cache's slot stride used to be a multiple of 4096 **and** of every part's ggml block size, because mmvq turned
the bank's nb[2] into a block stride. For GLM-5.3 UD-Q2_K_XL (IQ2_XS 74 B and IQ3_XXS 98 B blocks) that padded
12091392-byte records to 14852096-byte slots: 22.8 % of the cache RAM. This patch passes nb[2]/nb[3] to mmvq as byte
strides (an expert is base + slot * nb02 bytes, then the unchanged block indexing), so only the O_DIRECT 4096 rounding
is left. Only pointer arithmetic changes; for ordinary tensors (nb02 a multiple of the type size) the pointer is the
same. `LLAMA_EXPERT_STORE_BYTE_STRIDE=1` turns it on (default 0); the old padded rule stays for paths that cannot use
byte addressing (the CPU split `GPU_SLOTS=0`, CPU MUL_MAT_ID, types other than IQ2_XS / IQ3_XXS / IQ4_XS).

Measured on the Spark, GLM-5.3, 80000 MiB cache, 64k context, 128-token runs, legs off/on/off/on from the same cache
snapshot: 3.83/3.86, **4.25/4.24**, 3.86/3.86, **4.25/4.25 tok/s (+10.3 %)**, md5 `9449ae5738` in every run, same
MemAvailable. Written by a sol worker (gpt-6.1-sol) from the code; path audit in its RESULT (scratch), summary in
docs/experiments.md (2026-10-30).
