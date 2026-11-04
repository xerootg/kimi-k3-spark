# RESULT: GH200 prefill levers 1 and 2 (2026-11-04)

Files on the box: `/srv/glm-5.3/pfub/` (leg.sh, driver.sh, server-*.log, gpumem-*.txt, leg-*.out, bw.cu, bw, bw.out). Copies of the scripts, bw.cu and this file are in the local dir.

## 1. Lever 1: ubatch (BASE env as prod, `--batch-size`=`--ubatch-size`=U, `LLAMA_EXPERT_STORE_MAX_BATCH`=U, ctx 524288)

| leg | U | HBM tier | load s | 24k tok/s | step-300 s | 100k tok/s | std md5 | peak HBM (MiB, nvidia-smi) |
|---|---|---|---|---|---|---|---|---|
| u2048-h16384 | 2048 | 16 GiB | 53 | 571.1 | 1.34 | 538.9 | c592f33e7c | 89212 |
| u4096-h16384 | 4096 | 16 GiB | - | FAIL (OOM at graph reserve) | - | - | - | 78796 |
| u4096-h8192 | 4096 | 8 GiB | 52 | 642.3 | 1.40 | 594.1 | c592f33e7c | 91336 |
| u8192-h16384 / h8192 / h0 | 8192 | all three failed | - | FAIL | - | - | - | 78646 / 70469 / 62282 |

- 4096 vs 2048: +12.5% on 24k (642 vs 571), +10.2% on 100k (594 vs 539). The tier differs between these two legs (8 GiB vs 16 GiB), so the gain is U plus tier. Decode in the 8 GiB leg was 22.8-24.1 tok/s vs 23.7-24.1 at 16 GiB in these requests (decode: 22.77 / 21.66 on the 24k/100k requests vs 23.65 / 22.19).
- 4096 needs `HBM_MIB=8192`, because 16384 fails. The memory valve also released HBM cache slots at runtime in the 2048 leg (2.93 GB) and the 4096 leg (4.30 GB), because HBM free fell under the 8.59 GB reserve. The peak HBM in the 4096 leg, 91.3 GB (nvidia-smi MiB), is the highest seen.
- 8192 does not fit even with the tier at 0: the compute buffer asks for 41408 MiB (U=4096 asks for 20704 MiB, U=2048 fits). It is linear in U, so ~10.1 MiB/token. The 8192 buffer plus the ~60 GB of weights/KV does not fit in 96 GB. A prefill compute buffer of 41 GB would be needed, so 8192 would need a smaller ctx or less resident state; not tried.
- md5: the 24k output md5 is c2c04a7f36 and the 100k output md5 is 9a3f47ed20 at both U=2048 and U=4096, unchanged. The std request is c592f33e7c in both completed legs.
- The `compute buffer size` log lines are not printed at this log verbosity. The sizes above come from the failed-cudaMalloc lines (`allocating 20704.17 MiB ... out of memory` for U=4096, `41408.28 MiB` for U=8192). The 2048 and 4096-with-8 GiB sizes were not captured (not logged).

## 2. Lever 2: host to HBM bandwidth (bw.cu, 8 GiB buffers, 4 GiB per test, sm_90, 132 SMs)

a: cudaMemcpyAsync H2D, GB/s (best of 3):

| source | chunk | 1 stream | 4 streams |
|---|---|---|---|
| cudaHostRegister'd mmap (64 KiB pages) | 2 MB | 241.6 | 218.2 |
| | 4 MB | 283.3 | 268.0 |
| | 8 MB | 312.5 | 302.1 |
| | 16 MB | 330.3 | 324.7 |
| plain malloc (unregistered) | 2 MB | 149.3 | 144.7 |
| | 4 MB | 207.2 | 203.1 |
| | 8 MB | 261.0 | 258.7 |
| | 16 MB | 300.4 | 298.3 |
| cudaMallocHost | 2 MB | 259.3 | 232.5 |
| | 4 MB | 301.3 | 283.9 |
| | 8 MB | 327.6 | 316.4 |
| | 16 MB | 342.1 | 336.5 |

b: read kernel (16-byte loads, grid-stride, all SMs), GB/s:

| source | 264 blocks x1024 thr | 528-1056 blocks | notes |
|---|---|---|---|
| registered malloc | 354-356 | 355-357 | independent of unroll U=1/4/8 |
| cudaMallocHost | 353-356 | 353-356 | same |
| HBM cudaMalloc | 3800 | 3815-3849 | |
| plain malloc | 23 / 17 (first runs) | 3620-3840 | the pages were migrated into HBM (ATS/access-counter migration), so this is an HBM number, not host bandwidth |
| registered, 16 blocks / 64 blocks | 181 / 357 | | one block per few SMs is latency-bound |

Best achievable host to HBM: ~356 GB/s by direct kernel reads with at least 64 blocks and >=2 KB in flight per SM, ~330-342 GB/s by cudaMemcpyAsync with 16 MB chunks (1 stream is enough). Chunks of <=4 MB lose 15-30%, and unregistered memory loses a further ~10-40% at small chunks.

### Diagnosis
- The link is not the limit: a plain streaming kernel over the same kind of registered Grace memory reads at ~356 GB/s, 2.6x the 137 GB/s the in-place MMQ gets. The cache region is mmap'd with 64 KiB pages and `cudaHostRegister(Portable)`'d (llama-moe-cache.cpp `store_region_alloc`; ggml-cuda.cu `ggml_backend_cuda_host_bank_buffer_impl`), exactly the registered case here, so the memory type is not the cause.
- Likely cause of 137 GB/s (inferred from the bench, not profiled in the MMQ kernel): MMQ is not a pure stream. Each tile loads quantized blocks with small and irregular accesses, with the loads interleaved with dequant/dp4a compute, so the bytes in flight per SM are low and the kernel is host-latency-bound. The 16-block test shows that low concurrency halves the rate (181 GB/s), and the MMQ in-place path also re-reads an expert once per token tile (ID tiling), so the effective rate falls further. It is also not a bank-conflict issue; HBM-resident experts hide this latency.
- Likely cause of 0044's ~10 GB/s (also not directly profiled; the bench shows plain copies run at 280-340 GB/s with 4-16 MB chunks, so the link and the copy engine are fine): `PREFILL_FROM_CACHE` issues one `ggml_backend_tensor_set_async` per used expert (one per expert slice of each of gate/up/down, so small copies), interleaved with a 512-byte copy of a pageable `std::vector` zero buffer after each gap, and a stream synchronize per layer and per part. A pageable-source cudaMemcpyAsync is staged and effectively synchronous with the host, which serializes the whole sequence into small, latency-bound copies with no overlap with compute. My own 2 MB-chunk copies get 240 GB/s, so only this serialization can explain 10 GB/s.

### Prototype
Not built. Decision: ~20 min remained after the benchmark (16 of 38 min elapsed when it finished). That was judged too little for the source changes, a full rebuild and a ~6 min client run with a safe margin before the deadline, so the design is given instead. Design:
- Per MoE layer, allocate two HBM staging buffers (2 x n_used x expert-bytes; ~3.1 GB at U=2048 per layer, <= the full layer for worst-case, so size for the layer). This is an extra 6+ GB of HBM, which conflicts with the U=4096 headroom; drop the HBM tier to 0 or 8 GiB to pay for it.
- On a second non-blocking stream, copy each used expert run to the staging buffer for layer L+1 while layer L computes. Merge runs of adjacent used experts into one cudaMemcpyAsync (>=8-16 MB each), and copy the experts of the part's gate/up/down in a few large copies (the cache slot layout has `slot_stride`, so copy whole slots with strided cudaMemcpy2DAsync if the slot is contiguous). No pageable sources: preallocate the zero pad in pinned memory or memset on device.
- Synchronize with events (record on the copy stream, `cudaStreamWaitEvent` on the compute stream), never `ggml_backend_synchronize`. The ids of layer L+1 are not known before layer L finishes (the router runs in the layer), so either copy all experts of the layer (3.1 GB at ~330 GB/s = ~9.4 ms) or copy after the router with ~2 layers of pipelining. At 330 GB/s a layer is ~9 ms vs ~22 ms now at 137 GB/s: the expert-read half of the prefill time would shrink ~2.4x if it overlaps with compute, i.e. prefill could reach roughly 700+ tok/s at U=2048 (estimate, not measured).
- Alternative with no extra HBM: make the MMQ in-place read latency-tolerant (more bytes in flight per SM, larger tile loads via cp.async/TMA of whole expert rows) instead of copying.

## 3. Failures (shortest decisive lines)
- U=4096, tier 16384: `allocating 20704.17 MiB on device 0: cudaMalloc failed: out of memory` then `failed to allocate compute buffers`.
- U=8192, tiers 16384, 8192, 0: `allocating 41408.28 MiB on device 0: cudaMalloc failed: out of memory`.
- u4096-h8192 exit: `~llama_context: CUDA_Host compute buffer size of 1920.2031 MiB, does not match expectation of 8576.2032 MiB` (a warning at shutdown; the leg completed normally).
- `compute buffer size` lines are not in the logs at this verbosity (see section 1).

## 4. Final state
`pgrep -af "[l]lama-server"` is empty and `nvidia-smi` shows ~3 MiB used. Snapshot copies (`snap-*.k3sn`) were deleted after each leg. Nothing was touched under snapshot/, store/ or q2/, and no Kubernetes or git operation was made.
