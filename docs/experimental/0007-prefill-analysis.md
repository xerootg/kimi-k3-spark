Changed: added this analysis and `tests/analyze-win2a.py`; no production code/manifest changed. No network, commits, or writes outside this directory.
Evidence references below use `evidence/win2a/` as their prefix; GB is decimal, MiB binary. These are single sequential runs, not repeated matched-state trials.

Recommendation: target `--batch-size 2048 --ubatch-size 2048 --ctx-size 65536 --moe-cache-mib 80000`, production build WITHOUT 0007, retain MAX_BATCH=32 and existing decode settings.
That build/ubatch combination is unmeasured: gate rollout on baseline2048 timing + standard MD5. The fastest already validated configuration is p2048 WITH 0007; keep that until the replacement passes. Do not revert to ubatch512.
0007 saved disk bytes but lost wall time at matched ubatch512; no demonstrated decode benefit justifies its default warming. WARM=0 disables warming only, not the hit-copy regression.

| Build/ubatch | 2k: seconds / tok/s / prefill GB | 8k: seconds / tok/s / prefill GB | 30k |
|---|---|---|---|
| base512 | 85.174 / 24.04 / 772.6 | 342.395 / 23.93 / 3049.5 | unmeasured |
| p512 | 97.746 / 20.95 / 496.7 | 379.792 / 21.57 / 1817.5 | unmeasured |
| p1024 | 54.725 / 37.42 / 267.0 | 216.962 / 37.76 / 1003.8 | unmeasured |
| p2048 | 31.070 / 65.92 / 139.5 | 127.365 / 64.32 / 535.4 | 493.769 s / 60.76 tok/s / 1993.3 GB |
Timing sources: base512.log:538-539,598-599; p512.log:538-539,598-599; p1024.log:536-537,588-589; p2048.log:536-537,583-584,664-665; JSON timings at each *-2048/8192/30000.json:94-105 agree.
Bytes are cumulative-counter DIFFERENCES: base512.log:519,579 (772.6 -> 3822.1); p512.log:519,579 (496.7 -> 2314.2); p1024.log:517,569 (267.0 -> 1270.8); p2048.log:517,564,645 (139.5 -> 674.9 -> 2668.2).
Thus 0007 cuts logical prefill reads 35.7%/40.4%, but tok/s falls 12.9%/9.8% (time rises 12.572/37.397 s). p2048 vs base512 is 2.74x/2.69x; patched2048 vs patched512 is 3.15x/2.98x.

Path: llama/src/llama-moe-cache.cpp:4886-4962 synchronizes prior work, copies resident records SERIALly into pinned staging, runs miss I/O, optionally copies misses back into the bank, then uploads every used gate/up/down part. Copies precede reads; warming precedes upload. Hit reuse does not remove upload.
Only `jobs.size()*rec` counts as prefill bytes (:4928); this is logical submitted store I/O, not measured NVMe traffic. Hit copies, warming, and upload bytes/time are absent from the log. `per layer fill` and split/predictor timings are DECODE counters (:4821-4848), not staging profiling; their 0.00 ms cannot establish free prefill I/O.
At B512 the baseline reads 193.15/190.59 GB per pass (4/16 passes). The logged full bank is 223800 MiB = 234.67 GB (p2048.log:475-478): about 82.3%/81.2% byte-weighted coverage, equivalent to ~211/208 of 256 at uniform record size. No per-layer union histogram exists; “all 256 everywhere” is unsupported.
Assuming comparable selected unions, patched512 replaces ~275.9/1232.0 GB of disk reads with hit memcpy, leaving 496.7/1817.5 GB of misses. Hits are ~36-40% of selected bytes, NOT dominant by byte count. Different warming/routing can perturb this estimate.
Warm execution is EXPECTED, not directly logged: default-on branch at :4931-4943; snapshot already fills 6859 experts (p2048.log:472); no valve releases (:647,655,725). Main 6496-slot and 273-slot groups can warm <=256 selections; the 90-slot group skips if >90 selected (:476-478). Busy-slot checks can also skip. No captured environment/warm counters prove each invocation.
Successful warming copies planned misses staging->bank without rereading disk; full-cache scan resistance puts them at eviction head (cache.cpp:258-345), churning scan slots rather than preserving every decode resident. Warm bytes approximate eligible miss bytes, not zero, but are unreported.
Therefore “RAM memcpy slower than NVMe” is NOT established: the regression combines serialized hit copies, extra warm copies, planning, synchronization and possible paging. p512.log:489 reports mlock failure. Snapshot restore 13.4 GB/s (:472) is a separate workload, not a prefill disk/copy microbenchmark.
Keep 0007 as an experiment only. Warm-only may help an empty cache, but these restored-cache runs show virtually identical base/p512 decode timings and ~86% final hits (base512.log:609; p512.log:609). A hit-rate threshold alone is insufficient: saved disk time must exceed hit-copy + warm-copy + planning time; measure WARM=0/1 and warm-only separately before selecting any threshold.

B2048 reduces full layer/weight traversals: 2k = 1 vs 4 passes; 8k = 4 vs 16; 30k = 15. Patched miss GB/pass are 139.50,133.85,132.89; B1024 = 133.50/125.47; B512 = 124.17/113.59. Larger unions partly offset fewer passes, but weight movement/setup amortization clearly matters.
Do not model MoE compute as 256 experts times every token: routing selects 8 experts/token, average 16/32/64 assignments per expert at B512/1024/2048 (mmq.cu:211-217,297-319). Loading a broad union does not imply dense all-expert computation. MMQ quantizes/maps routed activations and reads weights for their assigned rows.
The logs cannot distinguish GPU matmul, upload, serialized I/O/copies or attention as the dominant B2048 component. Its miss bytes / prompt time are 4.49/4.20/4.04 GB/s, below snapshot bandwidth, but that does not exclude I/O latency/contention. At an ASSUMED 13.4 GB/s, misses take 10.4/40.0/148.8 s; those are estimates, not measured component times.
A two-term fit of patched 8k B1024/2048 time gives ~22.4 s fixed cost per pass + ~37.8 s token/context cost. It predicts ~82.6 s at B4096 (~99 tok/s, 1.54x), NOT another guaranteed 2x; different unions/kernels/attention invalidate extrapolation. B4096 needs BOTH batch-size and ubatch-size >=4096.

Measured memory at ctx65536: CUDA model 14102.60 MiB (p2048.log:432), main KV5616 + indexer KV336 = 5952 (:455,460), host cache79982.34 (:475), pinned staging3984 (:471), output0.59 (:453).
| Ubatch | CUDA compute MiB | Host compute MiB | Sum of above allocations MiB |
|---|---:|---:|---:|
| 512 | 2248.08 | 176.08 | 106445.69 |
| 1024 | 2864.09 | 352.10 | 107237.72 |
| 2048 | 4096.12 | 704.13 | 108821.78 |
Compute sources: base512.log:484-485, p1024.log:484-485, p2048.log:484-485. This replaces RESULT-impl.md's linear-from-zero compute/KV estimates; staging is fixed at max_record*256, independent of ubatch (cache.cpp:4860-4880).
B4096 extrapolation: CUDA6560.16 + host1408.19, total111989.88 MiB, +3168.10 over B2048. Nominal124609 MiB device memory leaves12619 MiB before other costs; ACTUAL B2048 MemAvailable minimum10.25 GB (~9775 MiB), leaving ~6607 MiB if incremental costs stay linear, versus4096 MiB reserve. Plausible, not a peak-memory guarantee; graph growth/pools/OS can trigger the valve. Do not add the mapped 224423 MiB host-model accounting as resident RAM (p2048.log:718-720).
Direct registered-bank MUL_MAT_ID is supported by the existing bank/slot path (cache.cpp:3032-3046,3487-3503,4970-4979; p2048.log:469,474). Raise MAX_BATCH to test it, not merely ubatch: it removes staging hit copies and per-part uploads, reads misses into slots and may overlap lookahead. It still reads misses and computes MoE; shared RAM traffic remains.
Capacity caveat: get_slot_map requires min(B*8,256) slots per group. The 90-slot group cannot serve B2048 directly and would retain staging unless rebalanced to >=256 slots (~2583 MiB extra for166 slots at16318464 bytes). The 273/6496 groups suffice. A bank experiment can alter dispatch/rounding and decode LRU behavior; repeat MD5 and memory checks. Pipelining layer reads/uploads or reducing repeated weight passes is promising; no evidence here guarantees doubling.

Decode: p2048 is 4.03/4.07/4.16 tok/s at2k/8k/30k vs short4.77 (p2048.log:537,584,665,711): 248/246/240 vs210 ms/token. It improves slightly with length, so this is not evidence of a monotonic long-context attention penalty.
Base512 and p512 match to ~0.01 tok/s at2k/8k. Cache stays populated; no valve shrink; standard request immediately after30k resets context (:681-682) and recovers4.77. Thus no demonstrated persistent cache damage from0007.
Attention is MLA with DSA indexer/top-k2048 (p2048.log:56-58,454-463; models/glm-dsa.cpp:254-350,433-435). Index scoring/top-k still depends on context, and sparse attention has a floor once2048 keys are selected; cannot attribute the whole gap to attention. The MTP comment at :553 is stale for the trunk; completion speculative.types is none (*-30000.json:83).
Workload routing is a strong alternative: long-prompt predictor match stays ~70%, real split wait~0.9 ms/layer; during standard request cumulative match rises to73.7%, wait falls to0.77 (p2048.log:648-658,687-706,726-730). These are cumulative, not isolated causal timings. 75 active MoE layers times~0.9 ms is already~67.5 ms/token of potential exposed waiting; cannot sum overlapped read timers as independent costs.
Conclusion on decode: some context work is inherent, this observed ~15% gap is not shown inherent. Compare identical continuation/routing across context lengths, per-request hit/miss/predictor deltas and GPU attention timers; predictor-off and resident-expert controls separate I/O from attention. Current different-content prompts confound these causes.

Validation command: `python3 tests/analyze-win2a.py` (offline, exit0). Checks all9 JSON timings against token counts, extracts cumulative read deltas, verifies standard MD5, computes memory estimates. No Spark/model/CUDA execution here.
Last output lines: `B=2048: accounted allocations=108821.78 MiB; nominal headroom=15787.22 MiB`; `B=4096: accounted allocations=111989.88 MiB; nominal headroom=12619.12 MiB (extrapolated)`; `PASS: offline evidence analysis; no Spark execution or component profiling`.
MD5 independently verified: `9449ae5738f2e56f99ab7ab6e9626eb5`. Remaining uncertainties: component timings/physical NVMe bytes, exact unions/warm eligibility/environment, B2048 without0007, B4096 peak memory/performance, and causal decode attribution; none can be recovered from these logs alone.

Next Spark measurements: matched restored snapshot, baseline2048 vs patched2048 WARM=0/1, repeat2k/8k and standard128; then B4096 with batch4096. Instrument synchronization, hit-copy bytes/time, pool reads bytes/time, warm planned/skipped bytes/time, CUDA upload and matmul/attention events; collect physical disk bytes and peak MemAvailable. This is required to identify the actual bound, not an additional result claimed here.
