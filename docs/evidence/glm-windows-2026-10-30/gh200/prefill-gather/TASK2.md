# Task 2: GH200 prefill, stage each layer's experts into HBM ahead of the matmul (prototype + measure)

Continuation of TASK.md / RESULT.md in this directory (read both first: the machine, the hard rules 1-6, the leg
harness `leg.sh`, the client and the bandwidth results all still apply). Same box, same rules: never touch Kubernetes,
never modify `/srv/glm-5.3/{snapshot,store,q2}`, snapshot copies per leg, explicit-pid kills only, one server at a time,
nothing running at the end, no git push. The production server is scaled to 0 by the supervisor.

**Deadline: write RESULT2.md in this directory within 75 minutes of starting**, whatever state the work is in.

## Goal
Prefill on the GH200 spends half its time in the routed-expert MUL_MAT_ID (MMQ) reading expert weights in place from
registered Grace memory at ~137 GB/s, while a streaming kernel reads the same memory at ~356 GB/s (RESULT.md). With a
2048-4096-token ubatch nearly every expert of a layer is used, so "the experts the layer needs" is effectively the
whole layer (~3.1 GB) and can be fetched before the router has even run. Make the needed experts be in HBM, ready,
when the MMQ starts, and measure the prefill gain. Outputs must stay byte-identical (md5 gates below).

## Where the code is (`~/llama-pf`, production patch stack; work in a copy)
- `src/llama-moe-cache.cpp`: `fill_prefill(...)` (~line 5645) is the staged path. It is used for batches larger than
  `LLAMA_EXPERT_STORE_MAX_BATCH`: the scheduler's copy callback (`src/llama-context.cpp` `sched_copy_experts`, ~2786)
  asks it to fill `dst`, a GPU copy of the layer's expert weight tensor, with the used experts. By default it reads them
  from the store file. `LLAMA_EXPERT_STORE_PREFILL_FROM_CACHE=1` (patch 0044, see `/claude/claude-1000/-home-xero-repos-argo-things/fd7931bb-f519-4321-a3de-81d4f945b7bf/scratchpad/sol-gh200glm/apply-0044.py`
  on the laptop = the block after "Patch 0044:" in fill_prefill) uploads them from the host cache slots instead
  (`g.region + slot*g.slot_stride`, slot map `g.lru.slot_map[il][expert]`), but ran at ~10 GB/s: one
  `ggml_backend_tensor_set_async` per expert, a pageable 512-byte zero copy, and `ggml_backend_synchronize` per layer.
- Production keeps `LLAMA_EXPERT_STORE_MAX_BATCH` = ubatch, so prefill uses the in-place host-bank MMQ instead. Set it
  to 32 to route prefill through `fill_prefill`.
- The cache region is mmap'd (64 KiB pages) and `cudaHostRegister`'d, so device kernels can read it directly
  (`cudaHostGetDevicePointer`, or the same address with unified addressing; check what the code uses).

## Steps
1. `cp -a ~/llama-pf ~/llama-pfub`, then configure a fresh build dir there with the same CMake flags as
   `~/llama-pf/build/CMakeCache.txt` (`GGML_CUDA=ON`, arch 90, Release, VMM on) and build `llama-server` with `-j 60`.
2. Baseline legs with the unmodified copy (port 18104, `pf_client.py`), U=2048, HBM tier 0 (frees HBM for staging):
   (a) in-place (`MAX_BATCH=2048`), (b) 0044 as is (`MAX_BATCH=32 PREFILL_FROM_CACHE=1`). Record what each takes.
3. Fast path, version 1 (no overlap): replace 0044's per-expert copies with ONE gather kernel per (layer, part):
   `dst[x] = src(slot[x])` for every used expert x, 16-byte coalesced loads, grid over all SMs, reading the registered
   host slots directly; a device-side zero fill for the record past the last used expert where the old code wrote the
   512-byte zero pad; the slot list built on the host into a pinned buffer once per layer. No `ggml_backend_synchronize`.
   It must run on the backend's stream so the MMQ that follows sees the data (stream order, no extra sync).
   Measure the gather's GB/s with `LLAMA_EXPERT_STORE_PREFILL_TIMING=1` (or cudaEvents) and the client numbers.
4. Version 2, only if version 1 works and time allows: overlap. Gather layer L+1's experts (all of them, ids unknown
   yet) on a second stream into a second staging buffer while layer L computes; event-sync before L+1's MMQ.
5. md5 gates (each leg): standard request `c592f33e7c`, 24k `c2c04a7f36`, 100k `9a3f47ed20` (the client prints them).
   A change in any of them is a failure of that version, report it, do not ship it.
6. If memory allows, also run the best version at U=4096 (the new production ubatch, tier 8 GiB) and say whether the
   staging buffers fit; production now needs to keep decode's HBM tier, so report the HBM cost of the staging.

## RESULT2.md format
1. Table: leg, version, U, tier, 24k tok/s, 100k tok/s, step-300 s, decode tok/s, md5s ok?, peak HBM, gather GB/s.
2. The diff of your change (`git diff` in `~/llama-pfub` against `~/llama-pf`, copied here as `0046-staged-gather.patch`).
3. What failed, shortest decisive log lines.
4. Final state: no llama-server running, GPU idle.
