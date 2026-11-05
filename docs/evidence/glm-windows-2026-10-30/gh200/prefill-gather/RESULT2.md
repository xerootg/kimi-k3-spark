# RESULT2: staged expert gather for GH200 prefill (2026-11-04)

**Outcome: version 1 is built and the gather kernel runs at ~355 GB/s, but it is NOT shippable. The 100k md5 does not match `9a3f47ed20`, and there is no prefill gain (557-562 tok/s vs 580 in-place). Version 2 and the U=4096 leg were not run.**

Box files: `/srv/glm-5.3/pfub/` (leg2.sh, leg-*.out, server-*.log, gpumem-*.txt). Source: `~/llama-pfub` (build dir `build`, fresh cmake, same flags as llama-pf). `leg2.sh` is `leg.sh` plus args: max_batch, binary tree, extra env.

## 1. Table (all legs: U=2048, HBM tier 0, port 18104, `pf_client.py`)

| leg | version | U | tier | 24k tok/s | 100k tok/s | step-300 s | decode tok/s (24k / 100k) | md5s ok? | peak HBM (MiB) | gather GB/s |
|---|---|---|---|---|---|---|---|---|---|---|
| base-inplace (llama-pf binary) | in-place, MAX_BATCH=2048 | 2048 | 0 | 579.9 | 542.5 | 1.36 | 21.1 / 19.6 | yes: 24k c2c04a7f36, 100k 9a3f47ed20, std c592f33e7c | 72924 | n/a |
| base-0044 (llama-pf binary) | MAX_BATCH=32, FROM_CACHE=1 | 2048 | 0 | 92.7 | not run (~25 min projected; killed after step-60) | 17.0 | 20.0 / - | 24k yes; 100k and std not run | 72924 | ~10 (observed 92 tok/s) |
| gather-v1 (timing on) | one gather kernel per (layer, part) + miss reads | 2048 | 0 | 556.8 | 524.6 | 1.46 | 20.8 / 19.5 | **100k NO (6f4aed9a4c)**; 24k yes, std yes | 72924 | 350-358 |
| gather-v1b (timing off) | same | 2048 | 0 | 561.8 | 526.2 | 1.46 | 20.8 / 19.5 | **100k NO (2d2fd9789c)**; 24k yes, std yes | 72924 | not logged (same kernel) |
| gather-ver (verify + timing, fully synchronous) | same, every gathered expert compared to its source after each launch | 2048 | 0 | 99.4 | 102.8 | 15.7 | 20.7 / 19.4 | **100k NO (908d05e631)**; 24k yes, std yes | 72924 | 350-358 |

- "md5s ok" is judged on the gates: std `c592f33e7c`, 24k `c2c04a7f36`, 100k `9a3f47ed20`. 24k and std pass in every leg. 100k fails in all 3 gather runs, with 3 different values.
- The step-300 / step-60 md5s (not gates) vary: in-place gives c2c04a7f36 / 83c6e7468a; 0044 gives c2c04a7f36 / c2c04a7f36; gather-v1 gives 0cea9ee842 / 83c6e7468a; v1b gives c2c04a7f36 / 83c6e7468a; ver gives c2c04a7f36 / c2c04a7f36.
- Peak HBM is the same in all legs (72.9 GB) because the gather writes into the scheduler's existing per-split copy buffer. **Staging HBM cost: ~0 extra**, only 4 x 16 KiB job lists on device and pinned.
- The 0044 100k was not run, so I do not know whether the old path gives 9a3f47ed20 at 100k.

## 2. Diff

`0046-staged-gather.patch` (in this dir; made with `diff -u` of `~/llama-pf` vs `~/llama-pfub`, because `llama-pf` has uncommitted changes of its own). It touches:
- new `ggml/src/ggml-cuda/gather-experts.cuh`: the kernel (uint4 loads, 4-way unroll, grid nsm*4 x 512 threads, one flat loop over all used experts' 16-byte units, plus a zero-fill of the 512 bytes at each unused expert that follows a used one), a ring of 4 pinned job lists with device copies, optional cudaEvent timing (`PREFILL_TIMING=1`) and an optional verifier (`LLAMA_EXPERT_STORE_PREFILL_GATHER_VERIFY`).
- `ggml-cuda.cu`: includes the header and exports `ggml_backend_cuda_gather_begin` / `_launch` via `get_proc_address`.
- `llama-moe-cache.cpp` (`fill_prefill`): with `LLAMA_EXPERT_STORE_PREFILL_FROM_CACHE=1 LLAMA_EXPERT_STORE_PREFILL_GATHER=1`, the slot list is built once per layer into the pinned job list; one launch per part on the backend stream; no `ggml_backend_synchronize`.
- **Added beyond the brief:** experts missing from the host cache are read from the store into the pinned staging (pool->run) and uploaded with `tensor_set_async`. Without this, nearly every layer fell back to store reads (see section 3).

## 3. What failed / findings (shortest decisive lines)

- First v1 run: 122 tok/s, gdb on the main thread showed `__libc_pread64 <- io_pool::work <- fill_prefill`. Cause: `n_slots 18176 filled 17947` of 19200 expert slots, so each layer has 0-8 experts not in the cache (avg 2.3 of ~200 used), and 0044's `all_resident` test sent the whole layer to the store-read path. I added the per-expert miss path (miss read cost 0.74 ms/layer on average). This is a prerequisite for any from-cache scheme: the cache is not a complete copy of the model.
- Gather kernel: `prefill layer 6 part 2 gather 3.106 ms 355.1 GB/s (229 experts, 1.103 GB)`. Range 350-358 GB/s, the same as the streaming-read ceiling in RESULT.md (356). Per layer it moves 1.7-3 GB (~150-230 experts of 256, not 3.1 GB) in 5-8 ms. The 0044 cost of ~230 ms/layer was entirely the store-read fallback, not the copies.
- No gain: 24k 557-562 vs 580 tok/s in-place. I did not profile why the gather + HBM MMQ does not beat the in-place MMQ; the stalls I know of are the per-layer ids sync plus CPU planning/miss reads, which leave the GPU idle. The timing-off run (v1b) did not change this. Estimated but unmeasured: ~25-30 ms/layer of non-MMQ work stays serialized.
- md5: the 100k output differs from 9a3f47ed20 in all 3 gather runs and the 3 values differ from each other (6f4aed9a4c, 2d2fd9789c, 908d05e631), even in the fully synchronous verify run. The verifier found **0 mismatches** between gathered bytes and the host slots (`grep -c GATHER-VERIFY` = 0), so the kernel and its placement are correct. The cause is not found. Candidates I could not test in the time: the missed-expert staging path, cache slot contents (the cache has 3 residency states?) differing from the store, and 100k being sensitive to run-to-run state. The 24k and std md5s always match. Treat v1 as failed.
- Server exit after the gather legs: `Segmentation fault` at SIGINT shutdown in gather-v1 and gather-ver (and in the killed dbg leg); not seen in the baseline legs. Not investigated. It occurs after all requests complete.
- The 0044 baseline 100k was skipped by design (~25 min); the client was killed after step-60 (`CLIENT-FAIL` in its out file is that kill).
- Not done: version 2 (overlap on a 2nd stream), and U=4096. Gather staging adds no HBM, so U=4096 would need the same HBM as the production tier-8 GiB config. Not measured.

## 4. Final state

No llama-server running (`pgrep -af "[l]lama-server"` empty), `nvidia-smi` shows 4 MiB used, snapshot copies `snap-*.k3sn` deleted. Nothing under snapshot/, store/, q2/ was touched; no Kubernetes or git push. `~/llama-pf` is unmodified; `~/llama-pfub` holds the changes.
