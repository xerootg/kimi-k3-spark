# llama.cpp patches

Applied in order (`git apply`) on top of the `LLAMA_SHA` pinned in
`versions.env`, by both `llamacpp-spark` and `llamacpp-rpc-cpu`, so every
binary that talks RPC runs the same sources. Each patch is a plain
`git diff` against the tree after the preceding patches; regenerate the stack
(and re-pin `LLAMA_SHA`) when
bumping llama.cpp.

Patch order and the check: patches are cumulative and applied in lexical filename
order, exactly as the Dockerfile's `patches/*.patch` glob; each patch must apply
after all earlier patches. `.github/workflows/patch-stack.yml` shallow-fetches
`LLAMA_SHA`, checks then applies each patch (naming any rejection), and syntax-checks
the touched `src/**/*.cpp` and `ggml/src/ggml-cpu/**/*.cpp` files with host `c++`.
The job has a two-minute limit and does not build CUDA or check `.c`, `.cu`, or
`.cuh` translation units. Locally, run `tools/patch-stack-check.sh ~/llama-k3-s2`
with Git, Bash, GNU coreutils/findutils, a C++17 compiler, and `liburing-dev` installed.
The checkout's HEAD must match `LLAMA_SHA` (currently `547208d`); the script uses
a temporary copy of HEAD, ignores local edits, and leaves the checkout untouched.

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

2026-10-30, second revision: the guard audits 19 ggml types (load widths from the CUDA vec-dots: 2-byte for IQ1_S,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q3_K, Q6_K, Q8_0...; 4-byte for IQ1_M, IQ4_XS, Q2_K, Q4_K, Q5_K, Q4_1, Q5_1) and
checks every part offset at runtime, so Kimi K3's groups qualify too: its two mixed groups (IQ2_XXS/IQ1_S) lose 22.8 %
and 29.6 % of padding, +3.2 % slots overall (GH200 65102 -> 67198 at 424000 MiB; Spark K3 7980 -> 8237). Host unit
test in `tools/validation/`; analysis in `docs/experimental/0004b-byte-stride-types.md`. Not yet measured on K3.

Measured on the Spark, GLM-5.3, 80000 MiB cache, 64k context, 128-token runs, legs off/on/off/on from the same cache
snapshot: 3.83/3.86, **4.25/4.24**, 3.86/3.86, **4.25/4.25 tok/s (+10.3 %)**, md5 `9449ae5738` in every run, same
MemAvailable. Written by a sol worker (gpt-6.1-sol) from the code; path audit in its RESULT (scratch), summary in
docs/experiments.md (2026-10-30).

## 0005-glm-predictive-prefetch

The expert store's predictive prefetch (a side reader fetches the next layer's predicted misses during the current
layer's GPU work) was wired only into Kimi K3's graph. This patch wires it into GLM-5.3 (`glm-dsa.cpp`): the predictor
input for layer L+1 is `ffn_norm_{L+1}(attention residual_L + shared expert_L + the resident experts' weighted share of
routed_L)` (`LLAMA_EXPERT_STORE_PREFETCH_INPUT=3`; 2 drops the routed share), run through layer L+1's router (sigmoid,
expert bias, top-8 of 256). Dense layers 0-2, the last MoE layer and the MTP layer post nothing. It adds a top-k
precision stat and fixes a latent planner bug the side reader exposed: with busy speculative slots (or second-chance
rotation) a multi-token plan could evict experts the same plan had selected (`slot -1 after the plan`).

Spark, GLM-5.3 with patch 0004, legs off/on/off/on: 4.256/4.239, **4.899/4.926**, 4.252/4.249, **4.877/4.906 tok/s
(+15.3 %)**, md5 `9449ae5738` in every run; top-8 precision 85.3 %, 67.8 % of the predicted misses read were used.
Written by sol workers (gpt-6.1-sol), built and measured here.

## 0006-expert-store-v2

The reader accepts expert store format v2 (`repack-experts.py --stride-align`, per-entry stride, header align): v1
stores behave byte-identically. Measured on the Spark with GLM-5.3 (2026-10-30, same warm snapshot per leg, legs
v2/v1/v2/v1): **v2 is slower, 4.82 against 4.90 tok/s**, md5 unchanged. The 512 KiB stride that made K3's reads ~8 %
faster buys nothing here and the padded records are 4 % more bytes. Kept for compatibility (the NUCs' stores are v2);
the Spark keeps its v1 store. Written by a sol worker.

## 0009-split-skip-zero

The split expert GEMV (patch 0003) runs the late matmul over 16 row positions so the graph shape never changes, and
fills the positions that have no record with a zero "dummy" record in the host bank. The GPU reads the host bank
uncached, so each dummy row still costs its full record read, about 28 us per id. This patch skips the dummy rows
inside `mul_mat_vec_q` (a per-row early return keyed on the slot map's dummy marker; the accumulation order of the real
rows and the exact zero contribution of the skipped rows are unchanged). `LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO=1` turns
it on (default 0).

Measured on the Spark, GLM-5.3, 80000 MiB cache, 64k context, 128-token runs, legs off/on/off/on from the same cache
snapshot: 4.895/4.901, **5.123/5.136**, 4.856/4.897, **5.133/5.133 tok/s (+4.8 %)**, md5 `9449ae5738` in every run.
Written by a sol worker (gpt-6.1-sol) from the op profile (`docs/glm-floor-profile-2026-10-30.md`); summary in
docs/experiments.md (2026-10-30, candidate windows).

## 0012-hook-lean

The host-bank hooks (early lookup, real lookup, predictor post) are CUDA host nodes served by a poller thread. This
patch trims the wake-up tax around them: 32 request threads instead of 256 for hooks with at most 32 ids, no unused
request-counter atomic at capture, a 20 us instead of 200 us idle poller sleep, a 200 us spin in the fill worker
after a completion before it sleeps, and a stable insertion sort for the predictor's small fills (no temporary
allocation). No id, map value, fill order, fence or GPU wait changes. `LLAMA_EXPERT_STORE_HOOK_LEAN=1` turns it on
(default 0). `LLAMA_EXPERT_STORE_HOOK_STATS=1` prints per-hook counts and service totals at teardown
(`host bank hooks: early=N/Ems real=N/Rms post=N/Pms idle_sleeps=S lean=B`), zero cost when off.

Measured on the Spark, GLM-5.3, same protocol as 0009 (0009 off in all four legs): 4.828/4.864, **5.111/5.122**,
4.837/4.812, **5.137/5.118 tok/s (+5.8 %)**, md5 `9449ae5738` in every run. The stats showed early and post hooks at
1.3-1.7 us each and the real hook at 0.55-0.60 ms: the hook bucket of the profile is the real-lookup wait, not the
handshake. Written by a sol worker (gpt-6.1-sol); summary in docs/experiments.md (2026-10-30, candidate windows).

## 0018-lfu-policy

Opt-in decaying LFU eviction for the expert store. Each group's volatile, model-key-sized frequency table counts
selected experts on non-scan plans; the least frequent eligible resident is evicted, with LRU order breaking ties.
Prefill scans retain the existing scan-resistant policy. Frequencies start fresh after snapshot restore; the
snapshot format and HBM use counts are unchanged. The four-layer GPU window protects the union of selected keys,
while busy/parked slots, the executing layer and the current plan's selections remain ineligible for eviction.

`LLAMA_EXPERT_STORE_POLICY=lfu` enables it (`default` keeps the existing policy).
`LLAMA_EXPERT_STORE_LFU_HALF_LIFE` sets decay in planned unique ids (default 262144, range 1..4294967295).
`LLAMA_EXPERT_STORE_SECOND_CHANCE` continues to control the default policy; LFU chooses by frequency instead.
`LLAMA_EXPERT_STORE_HOOK_STATS=1` also prints slot, selection and protection diagnostics if a plan fails.

Measured on the Spark, GLM-5.3, window l, from the same frozen snapshot per leg: **5.40 -> 5.66-5.74 tok/s
(+6 %)**, bit-identical output, hit rate **92.0 -> 92.7 %**, **234 -> 212 GB** read per leg. Adopted.
V1 crashed because whole-layer protection left no eligible victim on a skewed restored snapshot; v2 protects only
selected keys in the four-layer window, including the union from repeated hooks for one layer.

Both reviewer fixes are folded in: speculative headroom excludes LFU-protected keys, and four fixed-size key
buffers plus active lengths avoid heap growth on the hook path. The measured tree's speculative headroom guard
was absent from the committed stack, so its corrected guard and executing-layer skip are included only for LFU.
The CUDA hook call is re-contexted without the unadopted 0012b timing instrumentation. This patch applies after
0001-0006, 0009 and 0012 on pinned llama.cpp `547208d`; it does not require 0012b or 0012c.

## 0020-prefill-bank (the ledger's "0010": numbered 0020 in this directory because it applies after 0018)

Opt-in direct bank prefill (`LLAMA_EXPERT_STORE_PREFILL_BANK=1`, default off).
Batches over 32 tokens read resident expert records directly on CUDA and read
misses into a shared memfd landing area, avoiding the staging uploads. Preserve
the staged path's logical expert order and MMQ grid. Decode and requests of at
most 32 tokens keep their original views, cache capacity and fill paths.

Window x, GLM-5.3, pb2, ubatch 2048: **2048-token prefill 72.5 -> 106.3 tok/s
(+47 %)** and **8192-token prefill 70.7 -> 102.0 (+44 %)**. Decode after those
prompts was unchanged: 4.03/4.02 and 4.10/4.07 tok/s. Standard decode gold
`9449ae5738` and the long-prompt decode golds matched; all window gates passed
(`evidence/legs-x.txt`). These are pb2 measurements; LFU/c-tree confirmation is x3.

V1 subtracted 10344 MiB from the decode cache budget and expanded the ordinary
expert view even for short requests. It regressed decode and changed short-request
bytes (gold `9449ae5738` became `9a76bcc98b`); the exact arithmetic source of that
hash change was not isolated. V2 restored the original cache capacity, selected a
separate expanded view only above 32 tokens, and shared landing storage across
groups. V3 preserved those bytes and arithmetic while reserving dedicated landing
jobs, staging ID/maps and fixed reader scratch before requests, with bounds guards.

Reviewer fixes are folded in: bounded allocation-free bank I/O scratch and strict
alias registration from v3, plus drain readers before bank free (`filler.wait()`,
`spec.reset()`, `pool.reset()` before unregister/unmap). The CUDA memfd alias
registration logs success and aborts on unavailable/failed registration, including
the CUDA error, rather than silently falling back. The caller still synchronizes
GPU work before teardown.

GLM's physical landing area is **3984 MiB**, shared across groups and replacing
ordinary prefill staging; it is separate from the unchanged resident cache budget.
Budget for it while the model is loaded and during prefill: x3 uses 84000 MiB cache,
prints loaded MemAvailable and each leg's valve lowest line, and requires more than
the 4096 MiB valve reserve throughout the valve's recorded samples.

Rebased onto the committed 0001-0006 + 0009 + 0012 + 0018 stack; the LFU selected-key
protection call is retained before bank dispatch. No 0008 dependency: its opt-in
`GGML_CUDA_MMVQ_EXACT_COLS` dispatch affects only 2-4-column MMVQ, while this bank
path starts above 32 tokens and uses MMQ. Short-request views and MMVQ are unchanged;
the window x script does not set that knob. No portion of 0008 is added. The unchanged MMQ
base was recovered from the local 0006 index because the supplied post-0018 fixture
contains only ggml-cuda.cu; 0009/0012/0018 do not modify MMQ.

`0010-prefill-bank.stack.patch` is byte-identical to this production patch for
`~/win2`. Apply `0018-lfu-policy.ctree.patch` first in the copied c tree. The actual
c-tree files and c-tree LFU patch are absent here, so c-tree compatibility and a
need for a c-tree variant cannot be certified offline; x3 checks both applications
and stops on conflict. Host syntax checks do not build CUDA or repeat GPU/model gates.

## 0026-thp-bank

Opt-in `LLAMA_EXPERT_STORE_THP_BANK=1` (unset/0 keeps the existing path), Linux
with 2 MiB PMDs. Applied after the committed 547208d stack (0001-0006, 0009,
0012, 0018) and `evidence/0010-prefill-bank.stack.patch`. This rebases the v2
window patch; it changes only `src/llama-moe-cache.cpp`.

The private resident banks and their slot strides are 2 MiB-aligned. Before
CUDA registration, the allocator applies `MADV_HUGEPAGE`, populates writable
pages (`MADV_POPULATE_WRITE`, with a page-touch fallback), then requests
`MADV_COLLAPSE`. It logs actual `AnonHugePages` backing and all three return
codes, and refuses use below 90% huge backing or if HUGEPAGE fails. The dummy
slot follows the same geometry. Staging keeps its existing policy; with 0010
prefill bank enabled, the shared memfd landing tail also keeps its existing
policy. Alignment happens before mapping that tail, and THP population stops
at the private resident-bank boundary.

The memory valve releases whole huge pages belonging to evicted, parked slots,
with assertions protecting live, in-flight and dummy slots. Regrowth populates
and collapses quiescent ranges before publishing slots as usable. Victim lists
are reserved at initialization. Under `LLAMA_EXPERT_STORE_HOOK_STATS=1`, a
buffered store fd checks the first and last 64 bytes of every 1000th completed
record against its file offset; failures abort with layer/expert/slot/offset/len.
This covers the pool's thread and io_uring paths (including 0010's bank reader)
and speculative completions before publication. These are sampled boundary
checks, not exhaustive record verification. The older prototype's staging-hit
copy checks are absent because the committed stack's staged path reads all used
records; pool completion checks cover them directly.

For the measured GLM record size, stride grows from 12222464 to 12582912 bytes
(+360448 bytes per slot). Capacity below is `floor(budget / stride) - 1`,
including one dummy slot; it illustrates a single bank's capacity, before
multi-group budget distribution and 0010's separate shared landing allocation:

| Cache budget | 4 KiB stride slots | 2 MiB stride slots | Capacity loss |
| --- | ---: | ---: | ---: |
| 80000 MiB | 6862 | 6665 | 197 slots, 2.87% |
| 84000 MiB | 7205 | 6999 | 206 slots, 2.86% |

Window `thp`, 80000 MiB, fixed checker (`evidence/legs-thp.txt`), tok/s:

| Leg | THP_BANK | r0 | r1 | r2 | Mean r1/r2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TH0 | 0 | 5.363 | 5.325 | 5.386 | 5.3555 |
| TH1 | 1 | 5.479 | 5.428 | 5.431 | 5.4295 |
| TH0b | 0 | 5.428 | 5.387 | 5.376 | 5.3815 |
| TH1b | 1 | 5.486 | 5.457 | 5.451 | 5.4540 |

Both ON legs confirm a small gain: +1.38% and +1.35% for r1/r2 means against
their preceding OFF legs (roughly +1.5%). All 12 outputs have gold md5
`9449ae5738`. Both ON legs report 100% huge backing for all three banks:
78555119616 + 1493172224 + 3831496704 bytes (78.6 + 1.5 + 3.8 GB), with
HUGEPAGE/populate/COLLAPSE all returning 0. Record checks were 59/61/59/61
per leg with no mismatch, and the window ended `failed=none check_fired=none`.
Loaded MemAvailable was 12277/11659/11650/11618 MiB in leg order (about
11.6 GiB for the last three legs); no valve releases or re-arms occurred.

The same window's read benchmark gave G (mmap+THP+register) 160.346 GB/s
versus G0 (mmap+register, 4 KiB) 160.818 GB/s, both matching the device md5.
There is no GPU read-bandwidth gain here; the throughput improvement is
consistent with a fill-side gain, though this comparison does not isolate its
cause. These measurements used the earlier window stack, not this offline
rebase with 0010; combined runtime behavior remains to be measured.

The old broad `LLAMA_EXPERT_STORE_THP=1` experiment produced NaNs from the
first prefill in 4/7 starts versus 0/4 without THP; registered staging was a
suspect and the cause remains unresolved. Earlier THP-window checker failures
also included false positives from normal startup `logit bias = -inf` lines.
The corrected window here showed 100% bank backing, gold outputs and no sampled
mismatch. That evidence does not resolve the older NaN cause or exercise the
THP valve under pressure. At 84000 MiB, budget growth is 4000 MiB = 3.906 GiB
= 4.194 GB: starting from 11618 MiB = 12.182 GB leaves about 7.987 GB
(7.438 GiB), above the 4.295 GB reserve by about 3.69 GB. In the requested
rounded shorthand, 11.6 - 4.0 = 7.6 GB > 4.3 GB. THP release/regrow under
production pressure at 84000 MiB is still the main risk; keep this opt-in.

## Exact batching set (0030-0038), measured 2026-11-01 (windows fdr, fdq)

Apply the following eight patches lexically after production 0001..0026
(eleven production patches, nineteen total). Rebuild the image with this set.
The debug-only 0034 tensor-digest observer is excluded.

| Stack patch | Source / behavior |
| --- | --- |
| 0030-nsplit | 0030 v3hot, N<=8; 0032 union/recall folded; retained sync jobs and cached hook stats |
| 0031-exact-mmvq-cols | 0008.hot, singleton quantized column dispatch |
| 0033-exact-cols | 0035 v8, v7hot allocation/dispatch changes and padded split-bank MoE weighted reduction |
| 0034-exact-attn | 0036 v5hot, physical stream extents, contiguous Q packing, inline extent/top-k storage |
| 0035-prefill-isolate | 0037.hot, isolated prompts and retained scheduler scratch / templated callbacks |
| 0036-exact-rows | 0038, singleton SUM_ROWS geometry / wide unmasked softmax |
| 0037-exact-guards | 0039, N>=2 attention branch and rejection of unified parallel DSA KV |
| 0038-exact-graph-budget | 0040.hot, exact graph budget with cached flag |

0039's graph hunk is rebased onto v5hot's cached boolean; its guards are unchanged.
0030 retains v3's eight-column bounds and reservations. The c-tree graph-side
PREFETCH_WIDE flag is cached; production has no wide-prediction feature and
retains its fixed eight candidates. Production LFU-specific diagnostics and
recall accounting are preserved. Older additive joint-MoE fixes remain archived
outside the apply glob; do not combine them with v8.

Exact rows are retained for parity with the reported measured fd9 ancestry:
the user reports fd9 found 0038 already applied from fd6, and supplied
RESULT-REVIEW17 lists it in the prior c-tree sequence. The raw fd9 apply/window
log and actual runtime tree are absent here; legs-fd6.txt starts at build output
and cannot independently certify that ancestry. FDQ explicitly applies 0038.

User-reported WINDOW fdr PASS (driver 580.178.04, kernel 7.0): N1 gold at
5.03 aggregate tok/s; N2 both clients gold in three rounds at 7.07 / 7.16 /
7.14 aggregate tok/s (about +40%); short+long both exact (long md5 prefix
e38742e4); parallel=4 with two clients gold at 7.08 aggregate tok/s.
These measurements precede the complete hot replacement set. FDQ timing and
full N4/N8 client gates are pending. The proposed manifest v3 uses parallel=4,
ctx=65536 and all six knobs; parallel=8 requires the Q8 result.

Run `bash stack-exact/verify.sh` from the sandbox root: real production patch
sequence plus exact stack application and host syntax checks, including a
separate server-context check. This offline fixture cannot certify the pinned
commit identity or compile CUDA. Postimage differences retained under
verification/stack2 reflect production LFU/hook instrumentation and omitted
c-tree-only PREFETCH_WIDE/SHORT_CTX_SKIP experiments. Shared CUDA and KV leaves
are compared; host PASS is not a GPU gold result. See ../docs/exact-batching.md.


Measured on the Spark (driver 580.178.04, kernel 7.0), window fdq: with the six knobs set (GGML_CUDA_EXACT_COLS, GGML_CUDA_MMVQ_EXACT_COLS, LLAMA_EXACT_ATTN_COLS, GGML_CUDA_ARGSORT_EXACT_ROWS, LLAMA_EXPERT_STORE_SPLIT_COLS, LLAMA_PREFILL_ISOLATE all =1) every client at --parallel 2/4/8 is byte-identical to the single-sequence production output (cold, warm, mixed short and 2048-token prompts, staggered arrivals); aggregate 5.55 (N=1) / 7.14 (N=2) / 8.1 (N=4) / 8.72 (N=8) tok/s. With the knobs unset the set is inert (N=1 gold, same speed). The 0034 digest observer used to find the divergences stays outside this directory (debug only).

## 0042-ordered-expert-list

Patch 0003 compacts the experts that have columns in a MUL_MAT_ID into a list (`k_mmq_compact_experts`) so the MMQ
kernel's z dimension does not walk tens of thousands of empty slots. The compaction used `atomicAdd`, so the list
order changed from run to run. The stream-k decomposition walks the list in order, which moves an expert's K split
points, and therefore its float summation order. On the GH200 (prefill bank off, so prefill MoE goes through this
path) the tensor digest showed a layer's `ffn_moe_up` output differing between two identical requests in one process
while its activations, slot ids and the slot bytes (hashed before and after the matmul) were identical. The kernel
is now one 1024-thread block doing a stable ballot/prefix compaction in ascending slot order. With it, repeated
requests in one process are byte-identical (5/5, md5 `b533bf356d`). The list is tiny and n_experts is ~20k, so the
cost is a few microseconds per prefill MUL_MAT_ID.

## 0043-mmq-id-tiling

With 0042 the list order is stable inside a process, but the slot an expert occupies still differs between
processes (the expert store's mapping): the cross-process digest first differs at `ffn_moe_slots-3` (same experts,
different slots). The list is in slot order, so an expert's position, and with it its stream-k split, still varied
between processes. `GGML_CUDA_MMQ_ID_TILING=1` (exact value `1`, read once) launches the MUL_MAT_ID MMQ kernel with
one block per tile, the existing tiling mode of the stream-k kernel: every tile is summed over its full K range in
one order, independent of its position, and no fixup pass runs. Inert when unset. Measured on the GH200
(2026-11-02, full residency, production env): three processes x two standard requests all md5 `c592f33e7c` at
22.9 tok/s; a 24k-token prompt prefills at 584.8 tok/s with and without the switch. Without it, every process
gave a different md5 (`b533bf356d`, `b5933beb88`, ...). Spark: its prefill uses the prefill bank's map, not the
compacted list, and its decode uses MMVQ, so neither patch is expected to touch the Spark gold; confirm the gold
before the Spark adopts an image with them.

## 0045-elastic-kv

`LLAMA_ELASTIC_KV=1` (exact value, inert when unset) makes the KV cache elastic. Every K/V buffer still reserves the
whole context as GPU address space (CUDA VMM, `cuMemAddressReserve`), but physical memory is mapped one VMM chunk
(2 MiB on the GB10) at a time and only for cells `[0, padded use + margin)` of each tensor. Addresses never move, so
views and captured CUDA graphs stay valid. Newly mapped chunks are zeroed, as the whole buffer was before.

- Grows at the end of `apply_ubatch`, before the graph that writes the new cells runs, in steps of
  `LLAMA_ELASTIC_KV_MARGIN` cells (default 4096).
- Shrinks in `seq_rm`/`clear` once `LLAMA_ELASTIC_KV_SHRINK` cells (default 65536) are spare, after a device sync.
- A K-shift maps the whole context first, because its graph views every cell.
- Before growing, the KV cache calls `llama_moe_cache_make_room(bytes)`. The expert store's memory valve then
  releases least recently used slots so the host keeps its reserve. The valve re-arms them by itself after 10 s of
  calm once the KV has shrunk.
- Only for one KV stream with no transposed V (flash attention) and unshared cells. Other configurations log a
  warning and allocate as before.
- CUDA side: `ggml_backend_cuda_elastic_buffer_alloc` / `..._set_ranges` via `get_proc_address`. These return
  buffers of the device's ordinary buffer type, so the scheduler and the graph checks see a plain CUDA buffer.
  `clear` touches only mapped chunks.

**Window el (Spark, GLM-5.3-Flash, 1M context, 2026-11-03):**

| Setup | Decode, short (tok/s) | Decode after 16k (tok/s) | Prompt reading, 16k (tok/s) |
|---|---|---|---|
| Elastic on, cache 58000 MiB | 13.5 | 10.3 | ~300 |
| Elastic on, cache 78000 MiB | 15.4 | 13.8–14.4 | 345–354 |

- **Exactness:** with the cache unchanged (58000 MiB), output md5 `53816a69bc` / `145c14aa8b`, the same as elastic
  off, at the same speed. The KV starts with 110 MiB mapped instead of 19.7 GB.
- **Bigger cache:** with the KV's unused memory lent to the expert cache (78000 MiB), the md5s are still the same.
  Hit rate rose from 90.7 % to 97.9 %.
- **Long prompt:** a 200k-token prompt read at 272 tok/s and decoded at 12.5 tok/s. The KV grew to 3.85 GB, the
  valve released 1.71 GB of cache, and the lowest MemAvailable was 7.6 GB against a 6.4 GB reserve.
- **Shrink:** the next request dropped the long sequence, so the KV shrank back to 110 MiB. After one refill turn
  (6.9 tok/s) decode was back at 15.4 tok/s.

## 0047-hbm-spare-slots

`LLAMA_EXPERT_STORE_HBM_SPARE=1` (exact value, inert when unset) lets the expert cache have more host slots than
experts, as many extra as the HBM tier has slots.

**The problem it fixes:** when the cache budget holds every expert, slot sizing stops at one slot per expert. The HBM
tier's slots are part of that count, so the cache has no spare slot. Each HBM slot the memory valve releases
therefore drops its expert out of the cache entirely, and that expert is read from the store on every use until
something else is evicted.

On the GH200 (512k context, 4096-token ubatch, 8 GiB tier) the valve releases 352 HBM slots (4.30 GB) right after
start. That is about 1.8 % of the experts, read from NVMe on demand.

**With the switch:** 18,841 slots for 18,176 experts in the IQ3_XXS group (103.7 %), so released HBM slots fall back to
host slots.

**Measured (native, production env, 2026-11-05):**

| Measurement | Off | On |
|---|---|---|
| 24k prompt | 646 tok/s | 657 tok/s (+1.7 %) |
| 100k prompt | 596 tok/s | 600 tok/s |
| Decode | 26.4–27.1 tok/s | 26.8–27.1 tok/s |

Standard md5 `c592f33e7c` in both.

## 0049-remote-batched

Window nd (2026-11-07), env-gated, default off. The remote expert tier batched across sequences. The single-token tier
(0003) sends one request per MoE layer per token, each on the critical path (43 a token for GLM-5.3-Flash), and only
for experts the Spark does not hold. `LLAMA_EXPERT_STORE_REMOTE_COLS=1` instead hooks the N-column split GEMV (0030,
`LLAMA_EXPERT_STORE_SPLIT_COLS=1`, N = 2..8 sequences decoding together): at the early lookup of each MoE layer the
poller picks, per box, up to `LLAMA_EXPERT_STORE_REMOTE_BOX_ROWS` (one value or one per box) (expert, token) pairs
from the box's `LLAMA_EXPERT_STORE_REMOTE_SET` (experts used by one token first; resident experts too, so the Spark
does not read them; `LLAMA_EXPERT_STORE_REMOTE_OFFLOAD=0` restricts it to non-resident ones) and sends ONE request per
box with the inputs of the tokens involved and the pairs. The graph copies the N inputs into the layer's remote input
before the early lookup; the remote experts point at the zero record in the early and late matmuls; at the real
lookup the rows that arrived by `LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US` (from the post) are placed in the remote rows
and merged by `set_rows` after the late merge, and every pair that did not arrive is computed by the late matmul
(its expert read first if it is not resident; that read plans the layer's whole union, so no slot the late matmul
still reads is evicted). The output never depends on the network. Single-token decode stays local.

- `LLAMA_EXPERT_STORE_REMOTE_ADAPT=1`: each box's pairs per layer adapt per column count between 1 and its maximum
  (start 1; one more after 8 replies in a row complete when the real lookup came; one fewer after a wait longer than
  `LLAMA_EXPERT_STORE_REMOTE_ADAPT_WAIT_US`, default 100, or a miss).
- `LLAMA_EXPERT_STORE_REMOTE_MAX_WAIT_US`: how long the real lookup waits for a box whose pairs are all resident on
  the Spark (a late pair then costs one late-matmul row instead of a wait); pairs the Spark would have to read still
  wait until the deadline.
- Test knobs: `LLAMA_EXPERT_STORE_REMOTE_DISCARD=1` (send and wait, compute every pair locally),
  `LLAMA_EXPERT_STORE_REMOTE_DUMP=<path>` with `_DUMP_N` (placed rows with their inputs, written at exit; replayed on
  the GPU by `flashreplay.cpp` in docs/evidence/glm-windows-2026-10-30/nuc-batched/).

Wire: int32 hdr[68] { 0x4E440000|k, ntok, seq, layer, (gid, token) x 32 } + ntok x n_in f32; reply k x n_out f32 in
request order. The NUC server is tools/glm53/flash/f53s.cpp. Requires `GGML_CUDA_MMVQ_EXACT_COLS=1`: the stock
multi-token MoE kernel has another reduction order, so without it the boxes' rows are not the Spark's (a warning is
logged). Reproducible multi-stream bytes additionally need `LLAMA_PREFILL_ISOLATE=1` (0035), with or without the tier.
Measured (docs/experiments.md, windows nd0-nd8): byte-identical tier on/off at N=2/4/8, +1.3-2.5 % at N=4, +4.0-4.2 %
at N=8, neutral at N=2, nothing at N=1. The diff is against the production tree (llama-k3-el); it applies on the
committed stack (`git apply --check`) and the touched files pass a host syntax check.

## 0050-small-kernel-fusions

Five opt-in switches (each `=1`, all inert when unset) that cut the number of small CUDA kernels per decode token of
GLM-5.3-Flash without changing any arithmetic. Every fused kernel performs the same floating-point operations in the
same order with the same roundings as the kernels it replaces (`__fmul_rn`/`__fadd_rn` where two former kernels must
not contract into an FMA, the same device functions for sigmoid and silu, the SCALE kernel's FFMA, the ssm_conv loop
copied as is). Single-token, single-sequence graphs only; prefill graphs run unchanged.

- `GGML_CUDA_Q8_REUSE`: a quantized MMVQ activation (`quantize_q8_1`) is kept in one of eight retained 1 MiB slots for
  the rest of the graph evaluation, keyed by the source pointer, shape and strides; a later MMVQ with the same source
  skips its quantize kernel. A slot is dropped when any node (or fused group) writes over its source range, and all
  slots are dropped at the start and end of each evaluation. Stream 0 only, not with concurrent streams. Decode
  graph: 356 of 812 quantize launches removed (routed gate/up, shared gate/up, KDA q/k/v/f_a/beta/g_a).
- `GGML_CUDA_FUSE_HC`: the hyper-connection weight chain `MUL(mix, scale) -> ADD(base) -> SIGMOID -> SCALE` is
  computed inside the `DSV4_HC_PRE` that consumes it, and inside `DSV4_HC_POST` (after a normal `DSV4_HC_COMB`) for the
  post weights: 6 kernels become 1 and 6 become 2 per hyper-connection pair, 720 launches per token.
- `GGML_CUDA_FUSE_GDN_STATE`: the KDA `GATED_DELTA_NET` reads its input state straight from the recurrent cache row
  that the `GET_ROWS` would have gathered (row `ids[0]`, read through the ids on the device), and that `GET_ROWS` (a
  4 MiB copy per KDA layer) is skipped. Planned per evaluation: only when the GET_ROWS result has no other real
  consumer, nothing between the gather and the GDN writes the cache, and the gdn_cache fusion writes the new state.
  The GDN kernel reads every state element before writing it, from the same thread, so the in-place row is safe.
- `GGML_CUDA_FUSE_KDA`: `SSM_CONV -> RESHAPE -> SILU` runs through the existing ssm_conv+silu kernel (the upstream
  pattern only matched adjacent nodes), and the KDA gate chain `MUL -> SCALE -> SIGMOID -> SCALE` runs as one kernel.
- `GGML_CUDA_FUSE_CONV_STATE`: the one-token short-conv state path (`GET_ROWS` of the conv cache row, per projection
  `CONCAT(old taps, new column)`, `SSM_CONV`, `SILU`, and a `CPY` of the shifted taps back into the cache) runs as one
  kernel per projection, at the CONCAT's position (the new column is overwritten by the next projection's matmul
  before the SSM_CONV would run), reading the old taps from the cache and writing the shifted ones. Planned only when
  nothing between the CONCAT and the SILU touches the SILU's memory.

Measurement knobs (inert unless set): `GGML_CUDA_DUP_SMALL=1` runs every small out-of-place node twice and
`GGML_CUDA_DUP_QUANT=1` every quantize twice (same bytes, to price one more kernel); `GGML_CUDA_DUMP_GRAPH=<file>`
appends the node list and the fused/skipped steps of the first four large graphs; `LLAMA_LOGITS_FNV=1` prints an
FNV-1a hash of every decode's last logits row to stderr (bit-for-bit comparison of two builds or flag sets);
`GGML_CUDA_SCHED=yield|blocking` overrides the GB10's spin scheduling (both slower, see the ledger).

Measured (native, production env and args, GLM-5.3-Flash on the Spark, 2026-11-07; docs/experiments.md has the
interleaved legs):

| Build / flags | Decode, short (tok/s, 3 rounds) | Decode after 16k (tok/s) | Prompt reading, 16k |
|---|---|---|---|
| Production tree (llama-k3-el), no flags | 16.46 / 16.45 / 16.40 | 14.07 | 330 tok/s |
| This patch, no flags | 16.24 / 16.23 / 16.26 (other legs 16.32-16.35) | | |
| This patch, all five flags | 16.89 / 16.90 / 16.87 | 14.51 | 342 tok/s |

Output: md5 `53816a69bc` (256 tokens) and `145c14aa8b` (after the 16,414-token prompt) with every flag set; with
`LLAMA_LOGITS_FNV=1` the logits of the prompt and all 256 decode tokens hash identically with the flags off, each
of the first four flags alone, the four together and all five together. Kernel launches per decode token: graph steps 3,773 -> 2,506, quantize launches
812 -> 456. The diff is against the production tree (llama-k3-el); it applies on the committed stack including 0049
(`git apply --check`).

Production setting: `GGML_CUDA_Q8_REUSE=1 GGML_CUDA_FUSE_HC=1 GGML_CUDA_FUSE_GDN_STATE=1 GGML_CUDA_FUSE_KDA=1
GGML_CUDA_FUSE_CONV_STATE=1`.

## 0051-kv-host-memory

`LLAMA_KV_HOST=1` (exact value, inert when unset) puts the KV buffers of GPU layers in host memory that the GPU reads
and writes directly. For GLM-5.3-Flash that is the attention KV (`cache_k_l*`, 11.0 KiB per token) and the DSA
indexer cache (`cache_idx_k_l*`, 8.2 KiB per token); the KDA recurrent state stays in device memory. Built to test
the GH200 idea of keeping the KV in Grace memory and reading it over NVLink-C2C (`docs/ideas.md`).

- The buffer keeps the GPU's ordinary buffer type, so the scheduler still runs attention and the indexer on the GPU
  and CUDA graphs, views and copies treat it as any CUDA buffer. `--no-kv-offload` is different: it moves those ops
  to the CPU.
- Memory: anonymous `mmap`, `MADV_HUGEPAGE`, `mbind` to the NUMA nodes that have CPUs (only when there is more than
  one node; on the GH200 this keeps the pages off the HBM node), populated on the CPU, then `cudaHostRegister`. This
  is the expert host bank's path (345 GB/s over C2C in `docs/uvm-experiment.md`). `cudaMallocManaged` was not used
  because the driver can migrate pages to HBM, which hides the thing being measured. `cudaHostAlloc` was not used
  because it gives no NUMA control and, on the GB10, had the slowest scattered reads (below). With unified addressing
  the device address equals the host address; the startup log line says so.
- `LLAMA_KV_HOST_THP=0/1` overrides the huge-page default: on with 4 KiB base pages, off with 64 KiB pages (the
  GH200 kernel, where a THP is 512 MiB). On the GB10, without huge pages the GPU's scattered reads are 2-6x slower.
- Replaces elastic KV (0045) for these caches: the whole context is allocated in host memory at start, and a warning
  is logged when `LLAMA_ELASTIC_KV=1` is also set.
- Copies into or out of such a buffer use `cudaMemcpyDefault` (the direction is inferred); other buffers keep their
  explicit kinds. On the GB10, explicit kinds also worked on registered memory (kvprobe checks).
- The op profiler (`GGML_CUDA_OP_PROFILE`) appends `kv=cache_k`, `kv=cache_idx_k` or `kv=cache_r` to the key of
  any node that reads or writes a cache, in either placement, so attention and the indexer's gathers can be compared.
- CUDA side: `ggml_backend_cuda_host_kv_buffer_alloc(device, size, thp)` via `get_proc_address`.

**Spark (GB10, GLM-5.3-Flash Q2, 2026-11-08):** same binary, production env and args except elastic KV off and
`--ctx-size 262144` in both legs. Output byte-identical at every depth and on a 32k needle, decode speed unchanged
within noise. The GB10's host and device memory are the same LPDDR5X, so this measures correctness and the cost of
the coherent host-access path, not the GH200's C2C link. Numbers are in `docs/experiments.md`. GH200 window prepared
in `docs/evidence/glm-windows-2026-10-30/kv-placement/gh200-kvhost.sh`.

## 0052-elastic-kv-per-stream

Elastic KV (0045) for several KV streams, so `--parallel N` (non-unified KV: one stream per slot, each slot's cells
contiguous from cell 0) reserves no memory up front. With `LLAMA_ELASTIC_KV=1` and `n_stream > 1` every stream of the
attention KV and of the DSA indexer cache is mapped on its own: cells `[0, padded use + margin)` per stream, grown at
the end of `apply_ubatch`, shrunk in `seq_rm`/`clear` (same `LLAMA_ELASTIC_KV_MARGIN` / `_SHRINK` knobs). llama.cpp
divides `--ctx-size` by `--parallel`, so `--parallel 4 --ctx-size 4194304` gives each slot 1M of address space
(44 + 33 GiB of VA on the GB10, reserved in ~1.2 s) and maps only what the conversations use. `n_stream == 1` takes
the unchanged 0045 path.

- **Backed buffers.** A multi-stream batch views all of its streams at the longest stream's extent (and the indexer
  reads a sentinel row one past it), so a short stream next to a long one is read past its own mapping. The
  multi-stream buffers are therefore "backed": one shared, zeroed 2 MiB chunk is mapped at every address without
  memory of its own (`cuMemMap` of one handle ~0.6 us per chunk, access set once for the whole range). Those rows
  are always masked out; growing a stream unmaps the backing chunk (after a device sync) and maps real memory.
  CUDA side: `ggml_backend_cuda_elastic_buffer_alloc_backed`; `..._set_ranges` swaps chunks between backing and real.
- **Stream copies** (`seq_cp` across streams) copy only the source's mapped cells, through the host, so the backing
  chunk is never written.
- **Memory policy, off by default.** Growth asks the expert cache for room (`llama_moe_cache_make_room`) as in 0045.
  `LLAMA_ELASTIC_KV_REFUSE=1`: a slot may grow up to its address space; when the host would keep less than
  `LLAMA_ELASTIC_KV_FLOOR_MIB` (default 2048) after the expert cache gave back what it could, the growth is refused.
  `LLAMA_ELASTIC_KV_STREAM_MAX=<cells>`: a fixed share per slot. With either set, the server calls the new
  `llama_elastic_kv_reserve(ctx, seq, n_cells)` for every active slot before its tokens enter a batch (the whole
  prompt at once, so a request that cannot fit fails before its prefill); a refused slot gets HTTP 503 ("out of KV
  cache memory"), gives its memory back (`seq_rm`), and the other slots continue. Without a policy a failed mapping
  still aborts, as in 0045.
- **`LLAMA_SLOT_SERIAL=1`** (server): one request at a time, the oldest (lowest task id), prompt and generation;
  the other slots wait with their KV kept. Batched decode changes the bytes (below), serial decode does not, and
  several conversations stay resident without re-reading their prompts.
- **0050 guard.** `GGML_CUDA_FUSE_GDN_STATE` faulted (illegal address) on the first one-sequence decode after another
  slot finished, in any multi-slot server (also with elastic KV off, window ek). Not root-caused; the fusion now
  applies only when the recurrent cache has one row (`--parallel 1`, production unchanged).

**Window ek (Spark, GLM-5.3-Flash Q2, 2026-11-08):** production env and args (cache 78000 MiB, 0050 flags) plus
`LLAMA_PREFILL_ISOLATE=1`, `--parallel 4 --ctx-size 4194304` (1M per slot); distinct prompts p0-p3 (256 tokens) and
the 16k request; aggregate = generated tokens / wall time; details in `docs/experiments.md`.

| Setup | N=1 (tok/s) | N=2 aggregate | N=4 aggregate | bytes per client vs the same request alone |
|---|---|---|---|---|
| `--parallel 1` (0052 binary, n_stream 1) | 16.93-16.98, 16k 344 tok/s | - | - | production gold `53816a69bc` / `145c14aa8b` |
| `--parallel 4`, batched decode | 16.2 | 20.8-21.3 (+30 %) | 24.1 (+49 %) | **differ** (reproducible run to run, but depend on the batch) |
| `--parallel 4`, exact-batching knobs (0030-0038) | 16.4 | 15.9-16.0 | 17.6 | equal at N=2, differ at N=3 and N=4 |
| `--parallel 4`, `LLAMA_SLOT_SERIAL=1` | 15.3-16.2 | - | 14.2-14.7 | **equal to production gold**, every client |

- Memory: four idle slots map 4 x 4352 cells (264 MiB attention + 176 MiB indexer); a 16k conversation grows its
  stream in 6144-cell steps (1.0-1.1 s when the valve must release cache first, 2-6 ms otherwise).
- Refusal: with `LLAMA_ELASTIC_KV_STREAM_MAX=8192`, or with a floor no growth can meet, the 16k request got HTTP 503
  at once and the concurrent short request finished with the gold md5; the server kept serving.
- `--parallel 8 --ctx-size 8388608` does not fit next to a useful cache: startup took MemAvailable to 2.4 GB with
  the 78000 MiB cache (watchdog), and with 70000 MiB the cache thrashed (single stream 8-9 tok/s, N=8 12.7 aggregate).

## 0053-ctx-past-train

`LLAMA_CTX_PAST_TRAIN=1` (server, exact value, inert when unset): a slot may use its whole context even past the
model's training length. Upstream caps every slot at `n_ctx_train` (1,048,576 for GLM-5.3-Flash) and rejects
longer prompts; the startup warning still prints. For measuring retrieval and coherence past 1M (`docs/ideas.md`,
"Past the 1M-token ceiling"); results in `docs/experiments.md`.

## 0054-batch-invariant-decode

`GGML_CUDA_BATCH_INVARIANT=1` (exact value, read once, inert when unset): every op on the decode path computes a
token's bytes the same way whatever else shares the batch, so with `--parallel N` each client gets exactly the bytes it
gets alone, at any batch composition, and still shares the weight reads of batched decode. The kernels keep a fixed
reduction order; only kernel choices and work splits that followed the batch are pinned:

- **Quantized matmuls (MMVQ).** Dense widths 1..4 use the 4-warp K partition (width 1 drops the GB10's small-K and
  halve-iters variants, which partition K differently); wider batches run in chunks of at most four columns. Every
  MUL_MAT_ID width, 1 included, runs the one-warp-per-token MoE kernel (width 1 used the 4-warp kernel before). The
  dense matvec fusions (gate/up/GLU, bias) exist only for one column and are off; MUL_MAT_ID+GLU fusions stay (the
  MoE kernel fuses at every width); the routed+shared expert fusion is off. The Hopper Q6_K streaming kernel is off.
- **Float matmuls (F32 router `ffn_gate_inp`, `indexer.proj`).** The float vector kernel (its reduction depends on K
  only) for every width up to 8; stock switches to MMF (TF32 tensor cores) at 4 columns, which is what made N=4
  differ from N=2/3 in window ek.
- **Attention.** Decode (one query per stream) always takes the sparse gather path, which visits only the stream's
  selected cells in cell order (stock used it only when the batch's longest stream had >= 4096 cells, so a short
  stream next to a long one changed kernels), and each output tile gets one block over its whole KV range (no
  stream-K, whose seams move with the number of streams and the KV length).
- **Top-k / argsort.** A total order: value, then lower index, +0 equal to -0, padding last. Rows that fit in shared
  memory use the bitonic network with that tie-break, longer rows a stable segmented radix sort of canonical keys
  (same permutation, capture-safe). Stock picked CUB DeviceTopK, bitonic or an atomic radix select by the number of
  rows, with batch-dependent tie handling at the k-th score.
- **Row reductions.** `SUM_ROWS` and `MEAN` take their block size from the row width only (stock used 512 threads
  when there were fewer than two rows per SM, i.e. depending on the batch), and `MEAN` never takes the one-row CUB
  path.

Everything else on the GLM-5.3-Flash decode path was already batch-invariant (row-local norms, softmax, the
lightning indexer, the gated delta net per stream, the routing top-k, the MoE weighted sum, the 0050 fusions, the
expert-store split GEMV, whose rows come from the same kernel). Prefill is unchanged (prompts run alone with
`LLAMA_PREFILL_ISOLATE=1`, 0035). Found by per-op dumps of one client alone and in batches (debug observer
`docs/evidence/glm-windows-2026-10-30/bi/bi-dump.patch`, comparator `bicmp.py`).

**Spark, GLM-5.3-Flash Q2, windows bi1-bi3 (2026-11-09):** production env and args plus `LLAMA_PREFILL_ISOLATE=1`,
`--parallel 4 --ctx-size 4194304`; prompts p0-p3 (256 tokens) and the 16k request (128 tokens); details in
`docs/experiments.md`.

| Setup | N=1 (tok/s) | N=2 aggregate | N=3 aggregate | N=4 aggregate | bytes per client vs alone |
|---|---|---|---|---|---|
| `--parallel 4`, 0054 on | 16.2-16.6 warm (stock 16.3-16.7 in the same window) | 20.3-20.9 | 23.2-23.3 | 23.8 | **equal**, every client, with and without the 16k request, two server processes |
| `--parallel 4`, `LLAMA_SLOT_SERIAL=1` (0054 off) | 15.4 | 15.2 | - | 14.8 | equal (production gold) |
| `--parallel 1`, 0054 on | 16.8 (warm) | - | - | - | same bytes as the 4-slot 0054 server |
| `--parallel 4`, stock batched (window ek) | 16.2-16.5 | 20.8-21.3 | - | 24.1 | differ |

New gold with the switch: p0 `3a7986622d`, p1 `bb5110ee45`, p2 `2866d894b9`, p3 `1aab74269d`, 16k request
`145c14aa8b` (the same as production for that request). Production (switch unset) is unchanged.

## 0055-client-priority

Per-client priority in llama-server, for several clients on one batch-invariant server (0054); all off unless set.

- `LLAMA_CLIENT_CLASSES=<file>`: one `<api-key> <priority> <name>` per line (`#` comments). The keys are accepted like
  `--api-key` values (added to the list before the HTTP server starts; the log prints only the count), and a
  request's key (`Authorization: Bearer` or `X-Api-Key`) gives its task that priority; unknown keys get 0.
- Deferred requests (no free slot) leave the queue highest priority first (ties: oldest).
- `LLAMA_SLOT_RESERVE=<n>`: the last n free slots are kept for the top priority, so its request gets a slot (and,
  with `LLAMA_PREFILL_ISOLATE=1`, its prompt runs at the next step) instead of waiting behind the others. Lower
  classes also avoid taking a free slot that still holds a top-class conversation while another free slot exists.
- With classes and `LLAMA_PREFILL_ISOLATE=1`, the highest class's pending prompt is processed first, also between
  the ubatches of another prompt that is being read (each ubatch is computed the same way whenever it runs, so bytes
  do not change).
- `LLAMA_PRIORITY_HOLD_PROMPTS=1`: lower classes' prompts also wait while a higher class is generating, so its decode
  is never paused by their prefill; their new requests wait for it instead (measured: they can wait for its whole
  answer).
- `LLAMA_SLOT_BATCH_MAX=<k>`: at most k sequences decode per step, highest class first, then the slots that got the
  fewest steps (fair share of the remaining capacity).

**Spark, windows bi4-bi5 (2026-11-09):** 0054 on, `LLAMA_PREFILL_ISOLATE=1`, `--parallel 4`, five test keys generated
on the Spark and deleted after the windows (Juno priority 10; workbench x2, Claude Code, hef-posters 1),
`LLAMA_SLOT_RESERVE=1`; `prio_bench.py` in `docs/evidence/glm-windows-2026-10-30/bi/`.

| Scenario | Juno's time to first token, with classes | without classes | bytes |
|---|---|---|---|
| three others generating, Juno arrives | 0.40-0.44 s (alone: 0.38-0.45 s) | 0.48 s | every client equal to alone |
| three others generating and a fourth waiting for a slot | 0.53-0.56 s | 95.1 s (waited for a slot, then behind the fourth's 16k prompt) | equal |
| another client's 16k prompt being read | 2.2-2.7 s (the current ubatch finishes first) | 27.5 s | equal (16k request `145c14aa8b`) |

While the 16k prompt resumes, Juno's decode pauses for its remaining ubatches (4.8 tok/s over her answer); with
`LLAMA_PRIORITY_HOLD_PROMPTS=1` she decodes at 14.1-14.4 tok/s and the 16k client's first token moves from 47 s to
65 s. `LLAMA_SLOT_BATCH_MAX=2` with Juno and three others: Juno 7.1 tok/s, the three share the second position at
3.6 tok/s each. A request without a key gets 401.

## 0056-bi-split-attention

Removes the depth cost and most of the single-stream cost of batch-invariant decode (0054) without giving up its rule:
each client's bytes equal the same request alone, at any batch composition. Three parts, all inert unless set:

- **`GGML_CUDA_BI_ATTN_CHUNK=<cells>`** (production: 128): with `GGML_CUDA_BATCH_INVARIANT=1`, decode attention on the
  sparse gather path splits each output tile's index list into fixed chunks of that many cells (rounded down to the
  kernel's KV tile, 64 cells for GLM-5.3-Flash's 512-wide MLA), one block per chunk, and combines a tile's partial
  results in a fixed order (the uniform stream-K fixup, last chunk first). A chunk covers the same cells of the
  stream's own list whatever else is in the batch; chunks past the stream's live count are empty and combine as exact
  no-ops, so the number of chunks (it follows the batch's longest list) does not change the bytes. At least two chunks
  per tile always, so every tile takes the same combine path. 0054 gave each tile one block over its whole list: 8
  blocks for a single stream on a 132-SM GPU. The chunk size changes the bytes (another split), so a server keeps one.
- **Exact top-k.** 0054 made the indexer's top-k batch invariant by sorting the whole row (total order: value, then
  lower index, +0 equal to -0) in one block. The k first entries of that order are unique, so any exact algorithm
  gives the same bytes: an 8-bit radix select for the k-th key (four histogram passes), one ordered pass that keeps the
  winners (ties by lower index), a shared-memory bitonic sort of the k winners. One block per row. Always on in BI
  mode; `GGML_CUDA_BI_TOPK_SORT=1` keeps the 0054 sort for A/B timing (same bytes, checked on the Spark and the
  GH200). Standalone check against a CPU stable sort: 828 cases (ties, +-0, -inf, rows 1..262144, k 1..n) all equal.
- **`GGML_CUDA_BI_FUSE=1`** (production: 1): keeps the one-column variants that compute the same bits as the plain
  kernels, which 0054 turned off: the dense matvec fusions (gate/up/GLU, bias), the routed+shared expert fusion, and
  the small-K row packing of MMVQ (it only packs more rows into a block; each row keeps the same warps and K
  partition). GB10's halve-iters doubles a row's warps and stays off. Measured: the same bytes with and without it at
  N=1..5 on both boxes.

**GH200 (window gh200-bi3, one slot, Hopper sm_90, dev build = CI flags):** node time per token, CUDA graphs off.

| Op (ms per token) | stock | 0054 | 0056 (chunk 128) | 0056 + `BI_FUSE` (fusions only, before small-K) |
|---|---|---|---|---|
| attention, 16k deep | 0.43 | 2.29 | 0.54 | 0.53 |
| top-k, 16k deep | 0.47 | 1.72 | 0.24 | 0.23 |
| attention, 64k deep | 0.62 | 2.50 | 0.74 | - |
| top-k, 64k deep | 0.47 | 1.60 | 0.38 | - |
| dense q8_0 matvecs (+ fused), short | 7.54 | 8.54 | 8.56 | 8.36 |
| all nodes, short / 16k | 33.8 / 34.5 | 37.1 / 40.2 | 37.1 / 37.1 | 35.8 / 35.9 |

Bytes: the chunked attention changes some answers once, then every client keeps its bytes alone and batched. GH200
(acceptance window gh200-bi4, CI image `ga5fa848`, five slots, N=1..5 with 16k and 64k mixes, two server processes):
Service check `bc32ecab82`, p0 `8e312d7bdd`, p4 `e7851c4b56`, 16k `145c14aa8b`, 64k `8acec1b966` unchanged; new gold
p1 `74fb633e8e`, p2 `f8531a5f07`, p3 `4d6338450b`. `BI_FUSE` and the exact top-k change nothing. Spark (Q2 weights,
windows bi2-1/bi2-2): p0 `ce2fc95752`, p1 `bb64203d71`, p2 `2d592efcb6` new; p3 and 16k unchanged.

**GH200 acceptance (window gh200-bi4, tok/s):** production #169 config vs the same plus 0056-0058 (`BI_ATTN_CHUNK=128`,
`BI_FUSE=1`, `KV_STAGE=1`, `KV_STREAM_HOLES=1`): one client short 38-40 -> 42-43 (stock single-slot 50, the new config
on one slot 49), after 16k 33-36 -> 39-43 (stock 41.6), after 64k 36 -> 41-42 (stock 43.8); N=4 on slots {0,1,2,4}
50 -> 69; N=5 66 -> 66.7; prefill at 100k 647 -> 1422.

## 0057-kv-stage

`GGML_CUDA_KV_STAGE=1` (exact value, inert when unset): prompt attention over a KV cache in host memory (0051,
`LLAMA_KV_HOST=1`) first copies the viewed KV of the layer into a device scratch buffer (one `cudaMemcpy2DAsync` per
stream, from the device pool) and runs the same kernel on the copy. A prompt ubatch's sparse gather re-reads the
selected cells many times, which over NVLink-C2C cost about half of the prefill speed; one streaming copy of the
stream's KV costs ~1 ms per layer at 100k cells. Same kernel and values, so the same bytes. Decode (fewer than
`GGML_CUDA_KV_STAGE_MIN_TOKENS` queries, default 32) still reads host memory directly. A copy larger than
`GGML_CUDA_KV_STAGE_MAX_MIB` (default 1024: a 1M-cell stream of 1 KiB cells) is not made; the pool keeps its high-water
mark. The first staged call logs one line (`prompt attention over host KV reads a device copy`).

**GH200 (window gh200-bi3, five slots, 0054 + 0056, `--cache-ram 32768`):**

| Prompt | KV in HBM (elastic) | KV in Grace (0051) | KV in Grace + 0057 |
|---|---|---|---|
| 16k: prefill / decode | 1538 / 37.9 tok/s | 989 / 36.7 | **1550** / 36.7 |
| 100k: prefill / decode | 1360 / 32.4 | 649 / 37.9 | **1418** / 37.9 |
| 500k: prefill / decode | 806 (window gh200-bi) | 476 (window gh200-bi) | **838** / 25.6 |
| bytes (16k, 100k) | `9c80718286` | same | same |

GPU memory peak unchanged (91.0 GB with the 500k request). On the Spark (same LPDDR5X both ways) the staged and direct
legs give the same bytes and speed (window bi2-1).

## 0058-kv-stream-holes

`LLAMA_KV_STREAM_HOLES=1` (exact value, inert when unset): one ubatch may hold sequences whose KV streams are not
contiguous. With `--parallel N` (non-unified KV, one stream per slot) llama.cpp split a batch into one ubatch per run
of consecutive streams (`split_equal(sequential)`), because the attention views the streams `[s0, s1]` as one tensor.
So clients on slots 0, 1, 2 and 4 decoded in two forward passes per step, and the alternating graph shapes stopped
CUDA graphs: on the GH200 four clients made 49.6 tok/s on slots {0,1,2,4} against 67-69 on {0,1,2,3}, with stock
batching as with 0054 (window gh200-bi3, legs HS and GS/GB). That was the "dip at exactly four streams" of window
gh200-bi: the leg's earlier solo requests had given each prompt a home slot, so the four landed on {0,1,2,4}. Busy
slots are rarely contiguous in production (a client in the middle finishes, the others continue).

- The hybrid-idx memory (glm5-next, qwen4exp) splits non-sequentially when the variable is set.
- glm5-next's DSA layer, when the viewed streams have holes, gives the attention one query row per viewed stream (a
  hole repeats the first sequence's rows: finite inputs, outputs never read), pads the selection mask the same way
  (`get_rows`, exact), and keeps only the real rows of the output. Every real stream's attention reads the same query,
  mask and cells as without holes, so with 0054 the bytes do not change. Row maps are a graph input (I32, rebuilt when
  the streams change); KV writes already used global cell indices. qwen4exp has no padding: do not set the variable
  for it.

**Spark (window bi2-2, four slots, 0054 + 0056 chunk 128):** p0 + p2 on slots {0,2}: 12.7-13.0 tok/s aggregate
without, **18.1-18.8** with; p0 + p1 + p3 and the 16k mixes unchanged; every client its solo md5 in every round, also
with `GGML_CUDA_BI_FUSE=1`.

## 0059-pp-reserve-one-seq

With `LLAMA_PREFILL_ISOLATE=1` (0035) and `--parallel N` (non-unified KV), the prompt graph is reserved for one sequence
over one KV stream, the shape every isolated prompt ubatch has. Upstream's worst case spreads the ubatch over all N slots:
each slot views its whole stream, and the DSA indexer's pooled-key state is sized for N streams. Batched decode (N
sequences of one token each) is reserved as before, and the reserve after a memory update (K-shift, stream copy) uses the
same one-sequence prompt graph. No switch of its own: active whenever `LLAMA_PREFILL_ISOLATE=1` and N > 1. Kernels and
shapes of every graph that runs are unchanged, so the bytes are too.

**GH200, five 1M slots (window nc1, 2026-11-09, production config of #171 with a dev build):** CUDA compute buffer
40,235 -> 28,008 MiB (the single-slot size), CUDA_Host 18,794 -> 10,540 MiB; the memory valve no longer releases 4.7 GB of
the HBM expert tier at start (decode hit rate 99.9 % with 3.7 GB read from NVMe -> 100 %, tier share 47.5 -> 50.4 %).
One client, seventh warm request: 50.5 -> 51.7 tok/s; five clients 66.7 -> 68.5-69.8. Every client its gold md5 at N=1, 2, 3,
5 and next to the 16k request. On the Spark the compute buffer comes out of the same memory as the expert cache.

## 0060-elastic-kv-floor-burst

Guaranteed floor + burst for per-slot elastic KV (0052), so a slot can take a conversation longer than its fixed share
without any other slot ever being refused below its share. All off unless `LLAMA_ELASTIC_KV_FLOOR_CELLS` is set with
`LLAMA_ELASTIC_KV_REFUSE=1`.

- **`LLAMA_ELASTIC_KV_FLOOR_CELLS=<cells>`**: every slot may always grow to the floor; a growth to the floor is never
  refused for lack of memory (only if the device itself cannot map it, logged as an error). Growth past the floor (a
  *burst*) is allowed while MemAvailable plus what the expert cache's valve can still release (down to its floor,
  `llama_moe_cache_releasable()`) covers the growth, plus what every *other* slot still needs to reach its floor, plus
  the valve's reserve and `LLAMA_ELASTIC_KV_FLOOR_MIB`. Otherwise that one request gets HTTP 503 before its prompt runs;
  the other slots are unaffected. The accounting runs over all caches of the model (attention KV and DSA indexer).
  A burst asks the valve to make room for its growth only, so the expert cache shrinks only while a long conversation is
  resident.
- **Answer reserved with the prompt** (server, `LLAMA_ELASTIC_KV_GEN_RESERVE`, default 8192 cells): a prompt past the
  floor reserves its whole answer too (`n_predict`, or the default when unset), so a burst that cannot be met is a clean
  503 at the start and not an error mid-answer; a prompt under the floor reserves up to the floor at most.
- **No shrink under an active reservation**: a long prompt's per-ubatch `seq_rm` used to shrink the stream and the
  next ubatch's reserve mapped it again (~1.7 s each; window spark4); streams now keep their last reservation mapped.

Measured on the Spark in window nc-s1 (2026-11-10, dev build): the refusal path returns a clean 503 with every gold
md5 before and after (leg F2); a 500k-token conversation was admitted next to floor-sized ones with the valve releasing
15.5 GB of expert cache and MemAvailable never under 5.2 GB (leg F1). Not in production until the acceptance of 0062.

## 0061-prompt-pieces

A lower class's prompt ubatch computed in pieces while a higher class generates, so the higher class keeps most of its
decode speed and the other prompt still moves, without changing anyone's bytes. Off unless
`LLAMA_PREFILL_SLICE_LAYERS=<k>` is set (needs `LLAMA_PREFILL_ISOLATE=1` and `--parallel` > 1).

- **Why pieces and not smaller ubatches:** a prompt's bytes depend on its ubatch boundaries (window nc1: logits hash of
  the 16k request differs when the same prompt runs in ubatches that start at another offset, or with
  `--ubatch-size 2048`). The ubatches stay as they are; each is computed as views of its own graph that end between
  two layers (every k layers), with the same kernels and fusions as the whole graph (`ggml_backend_sched_graph_compute_part`).
- **libllama:** a prompt ubatch (one sequence, more tokens than slots) goes to a scheduler and compute buffer of its own;
  `llama_decode` returns 3 after the first piece and `llama_decode_resume` computes the next one (3 while pieces
  remain, 0 with the outputs in place as `llama_decode` would have left them). Decode batches of other slots run on the
  main scheduler in between; the decode graph's own buffer is reserved for 64k-cell views
  (`LLAMA_PREFILL_SLICE_TG_CELLS`) and grows on demand. Ubatches under `LLAMA_PREFILL_SLICE_MIN_TOKENS` (512) run whole.
- **Server:** while a higher class is generating, a lower class's prompt pieces get `LLAMA_PREFILL_SLICE_SHARE`
  (default 0.12) of the time; a higher class's own prompt never waits for the share (it waits at most for the rest of
  the pending ubatch). With `LLAMA_PREFILL_SLICE_DECODE_SHARE=1` the lower classes' decode steps join the higher
  class's steps only within the same share (their prompts still progress and their first token comes when the prompt
  is done). Use with `LLAMA_PRIORITY_HOLD_PROMPTS=0`: the hold rule is what this replaces.

Measured (dev builds; hold rule on -> pieces with the hold rule off): GH200, pieces of 4 layers, Juno answering while a
64k prompt arrives: Juno 54.2 -> 46.6 tok/s (88 % of alone) and the 64k prompt's first token at 52.8 s instead of 53.5 s
(window nc2). Spark, pieces of 2 layers with the decode share: Juno 16.6 tok/s alone, 14.5 (87 %) next to a 16k prompt
and 13.9 (86 %) with four short clients arriving a second after her, whose first tokens come after 4-18 s instead of
16-18 s (window nc-s1). Prompt logits and every answer are bit-identical with and without pieces on both boxes.

## 0062-floor-compute-reserve

The compute buffers of a floor + burst server (0060) sized for the floor, so 1M slots cost what 200k slots cost until a
conversation passes the floor. Active with `LLAMA_ELASTIC_KV_FLOOR_CELLS` + `LLAMA_ELASTIC_KV_REFUSE=1` and
`LLAMA_PREFILL_ISOLATE=1` (0059).

- **Prompt graph reserved at the floor** (`LLAMA_ELASTIC_KV_PP_CELLS` overrides the size): a full-depth prompt graph of
  a 1M slot costs ~14 GB of compute buffer on the Spark (about 19 KiB per cell past 200k), which a five-slot server would
  otherwise hold at all times. What it would need is measured at start (sizes only), the buffer grows when a
  conversation past the floor first runs a prompt ubatch (counted in that conversation's burst, and the valve makes
  room for it), and goes back to the reserve once no conversation is past the floor.
- **Decode graph reserved at the floor:** reserved at full depth, the decode graph of five 1M streams took 10.3 GB on
  the Spark where five 200k streams take 3.3 GB (window nc-s1, leg F1). It is reserved for streams viewed at the floor
  (with prompts in pieces, 0061: at `LLAMA_PREFILL_SLICE_TG_CELLS`) and grows on demand by ~2 KiB per cell and stream,
  which the prompt graph's burst accounting covers. A memory update re-reserves with the same caps.
- **`LLAMA_SLOT_KEEP_CHILD=1`** (server, tests only): the children of an `n_cmpl` request keep their context, so one
  request leaves several slots holding the same long conversation without reading it once per slot.

## 0063-valve-rearm-unlocked

The memory valve's re-arm (0039/0044: released expert-cache slots get their pages back once the host has been calm for
10 s) held the cache's LRU lock while it populated a chunk of 256 slots, and with THP banks (0026) every slot's populate
read `/proc/self/smaps_rollup` twice for its log line. In window nc-s1 (Spark, after a 500k-token burst had made the
valve release 15.5 GB) each 2.15 GB chunk took about 55 s under the lock: a decode ran at 0.28 tok/s for 226 s and a
prompt lost 65 s. Now the slots leave the released list under the lock and stay parked (no key, never a victim) while
their pages are populated outside it; they rejoin the LRU under the lock. A re-arm populates without the smaps reads
and logs one line per chunk with its duration. Byte-neutral: only when slots become usable again changes.


## 0064-prompt-share-same-class

With prompts read alone (0035) a prompt always ran to its end before the generating slots took another step, so one
client's long prompt stopped every other client's answer for as long as the prompt took: 49 s for a 64k prompt on the
GH200, 12 minutes for a 190k prompt on the Spark, and about an hour for a 500k burst (0060). 0061 only covered a
*higher* class's answer. `LLAMA_PREFILL_SLICE_SHARE_SAME=<share>` (default 0 = off, the old order) gives a prompt that
share of the time while a client of its own class is generating; the generating slots take decode steps in the rest.
With 0.5 the prompt and the answers each get half. Needs `LLAMA_PREFILL_SLICE_LAYERS` (the pieces are where the
prompt can pause). Scheduling only: the pieces and the decode steps are computed as before, so nobody's bytes change.
A higher class's prompt still runs first, and a lower class's prompt still gets `LLAMA_PREFILL_SLICE_SHARE` while a
higher class generates.

Not in production until its acceptance windows are done (see docs/experiments.md).

## 0065-prompt-turns

With prompts read alone (0035) the prompts of one class were read one after another: the prompt in flight first, then
the oldest task. A short prompt that arrived behind a long one waited for the whole long prompt, which on the Spark
is 12 minutes behind a 190k prompt and about an hour behind a 500k burst (0060). `LLAMA_PREFILL_TURNS=1` (default 0 =
the old order) makes the prompts of one class take turns, one ubatch each: the prompt that had its last ubatch longest
ago goes next, and a prompt that has not started yet goes before all of them (ties by task id). A higher class's
prompt still goes first. Scheduling only: every ubatch still holds one prompt alone and is computed as before, so
nobody's bytes change.

Smoke test for 0064 and 0065 on a small CPU model (SmolLM2-135M Q8, three slots, no GPU; the bytes of every client
were the same in the solo run, the old order and the new order):

| | old order | `LLAMA_PREFILL_SLICE_LAYERS=4 LLAMA_PREFILL_SLICE_SHARE_SAME=0.5 LLAMA_PREFILL_TURNS=1` |
|---|---|---|
| A is answering, B's long prompt arrives: A's longest gap between tokens | 46.5 s | 2.9 s |
| the same: A's answer done after | 73.8 s | 55.0 s |
| the same: B's first token | 46.5 s | 72.9 s |
| B's long prompt is being read, C's short prompt arrives: C's first token | 45.9 s | 4.9 s |
| the same: B's first token | 46.5 s | 46.4 s |

Not in production until its acceptance windows are done (see docs/experiments.md).

## 0066-burst-owed-compute

A correction to the burst admission of 0060/0062. A burst past the floor needs two kinds of memory: its KV cells,
which are mapped at admission, and the growth of the prompt graph's compute buffer (about 9.5 KiB per cell past the
floor on the Spark), which only happens ubatch by ubatch as the long prompt is read. The admission of a second burst
looked at free memory (MemAvailable plus what the expert cache can release) and did not see the part of the compute
buffer already promised to the first burst but not allocated yet, so two bursts admitted shortly after each other
could together promise up to about 8 GB more than was free, against a reserve of about 10 GB.

The context now reports the compute buffer's real size before every decode (`llama_elastic_kv_pp_buffer_now`), and
the admission adds "granted growth minus growth already allocated" to what a new burst has to leave free. Both
admission log lines print the number ("compute buffer owed to earlier bursts"). Admission only: no ubatch is computed
differently, and with no floor configured (the GH200) nothing changes.

Not in production until its acceptance windows are done (see docs/experiments.md).
