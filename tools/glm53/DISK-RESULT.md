# g53x serveidx disk tier: result

## What changed
- `tools/glm53/g53x.cpp` serveidx: `K3X_DISK_TIER=1` (the k3cs env names: `K3X_DISK_STORE`, `_LRU_GB` (1),
  `_DEADLINE_MS` (20), `_THREADS` (8), `_PIECE_KB` (512)).
  - The LRU arena is one `sycl::malloc_host` block, made resident and mlocked.
  - The field-run repack runs in place on the reader threads, row by row through a stack buffer.
  - Miss reads start when the header arrives; resident rows compute first; each disk row launches when its record
    completes. A row past the deadline is answered NaN (only that row). Stats are on the `disk:` line.
  - Off: same code path and bytes as before (serve_test's old runs pass unchanged).
- Request path without allocations or faults:
  - pre-recorded Level Zero lists on an in-order command queue (`G53X_SUBMIT`);
  - NEO keys `DisableIndirectAccess`, `EnableGemCloseWorker=0`, `DispatchCmdlistCmdBufferPrimary=0` (`G53X_NEO_TUNE`);
  - `zeContextMakeMemoryResident` on every USM buffer;
  - `mlockall` (`G53X_MLOCKALL`);
  - a GPU-hang timeout (`G53X_GPU_TIMEOUT_MS`); ULLS is opt-in (`G53X_ULLS`).
- `tools/bitexact/expert_disk.h`:
  - `Cache` takes an external arena, a per-piece in-place `transform`, non-blocking `settled()`, cross-record pieces;
  - p50/p99, busy time and GB/s in the stats.
  - k3cs is unchanged in behaviour. `test_disk_tier.py` got only an offline `K3X_GGML_COMMON` option.
- New `tools/glm53/allocgate.c`: an LD_PRELOAD counter of malloc, new, mmap and the ze alloc/event/list calls,
  windowed by SIGUSR2, plus a perf fault sampler.
- New `tools/glm53/disk_test.py`: gate, bench, clean. `serve_test.py`: disk runs (cold, LRU hit, deadline 0) for both quants and v1/v2.
- `build-sycl.sh` and `dsycl.sh`: allocgate build, env pass-through, memlock.
- Docs: `docs/glm53-kernels.md` section 10 (design, the allocation findings, bench tables, defaults, NUC how-to).

## Allocation and fault sources found (all fixed except the last)
1. SYCL submit (handler and UR event per launch).
2. The first execution of each list.
3. NEO's per-execute indirect-allocation vector. Pack mode avoided it but gave all-zero rows under memory pressure.
4. The i915 GEM close-worker deque.
5. The immediate command list's command-buffer vector.
6. Page faults on library and anonymous pages, and per-execute jump-back patching of batch buffers.
7. Not fixed: 1-3 minor faults per 1200 requests on i915 GEM mmaps (the queue's 192 KiB ring buffer). mlock cannot
   pin these PFN maps, and the kernel shrinker reclaims them; swap was 100 % full on this laptop.

## Tests (last lines)
- `python3 tools/glm53/serve_test.py` ->
  `PASS: serveidx rows bit-identical to g53x check (UD-Q2_K_XL, UD-Q4_K_XL; store v1, v2; streamed and not), NaN replies as k3x; disk tier: cold reads and LRU hits bit-identical, per-row NaN, deadline)`
- `./tools/glm53/check.sh` vs `~/glm53-work/chk-baseline.txt` -> IDENTICAL.
- `TMPDIR=$HOME/glm53-work/tmp K3X_GGML_COMMON=$HOME/glm53-work/llama/ggml/src/ggml-common.h python3 tools/bitexact/cpu/test_disk_tier.py`
  -> `PASS ALL CPU disk-tier regressions (pread)`. This was run after the last expert_disk.h change.
- `python3 tools/glm53/disk_test.py gate` (1200-request window) ->
  `ALLOCGATE window malloc=0 calloc=0 realloc=0 memalign=0 new=0 free=0 mmap=0 munmap=0 ze_host=0 ze_device=0 ze_shared=0 ze_free=0 ze_event=0 ze_eventpool=0 ze_cmdlist=0`
  `page faults minor/major: 1 0` ... `fault ... anon_inode:i915.gem` -> `GATE FAILED: page faults 1/0`
  (a rerun gave 3/0). All rows bit-exact and the stats matched. Earlier low-pressure runs gave 0/0.
- `python3 tools/glm53/disk_test.py bench` -> `72 cells, 0 NaN rows (deadline 1000.0 ms), 0 wrong rows` (plus 81
  cells in pass 1).
  - Defaults 512 KiB x 8 readers: k=8 100 % cold 15.6 ms median, 6.5 GB/s wall; k=1 100 % cold 3.8 ms.

## Unsure
- **The zero-fault criterion is not met on this laptop.** The residual i915 GEM faults look like memory pressure. They
  need confirming on an idle NUC (xe KMD, which may behave differently).
- The NEO debug keys need `NEOReadDebugKeys` support in the NUC's driver. If they are ignored, allocations come back
  (heap `new` per execute) but rows stay correct.
- The cause of the ULLS GPU stall is unknown, so ULLS is off by default.
- The host was loaded (load average 7-9), so latencies vary by up to 1.5x between runs. Re-measure on the NUC.
  A cold 12 MB expert at ~5.7 GB/s is about 2.1 ms, so cold rows fit the ~3 ms budget only partly.
- Stats p50/p99 are power-of-two buckets. The deadline is measured from the request header.
