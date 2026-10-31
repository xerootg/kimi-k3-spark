# THP bank investigation and proposed window
Scope: offline source review; no engine patch or GB10 result. Keep 0024 hostalloc separate; use register banks.
Anchors below: P = patches/0003-expert-store.patch (patch-file lines); S = src/llama-moe-cache.cpp
in the read-only ../sol-hooks/checks/0020c-pre tree; later stack changes require context-based rebasing.
Evidence: docs/experiments.md:95-99 and evidence/ledger-thp-lines.txt record 3.9 vs 4.5 ms/fill,
4/7 THP starts NaN on first prefill (temperature-0 GGGG), versus 0/4 without; ~75% huge pages split during load.
The old comparison was a 7-record fill (:238), not a measured 13% reduction for window-w exposed singles.
S:1631/P:3855 mmap anonymous private -> HBM mbind+populate -> host mbind -> optional HUGEPAGE -> populate.
Host advice is opt-in LLAMA_EXPERT_STORE_THP; errors from HUGEPAGE are unchecked; population falls back to touches.
S:3115/P:5237 allocates bank, then backend from_ptr calls cudaHostRegister(Portable) at P:590.
mlock is later, process-wide at first use (S:2196, :5075), not before registration; partial locking can fail.
S:5051/P:7134 uses that SAME THP allocator for registered prefill staging; this was not a bank-only experiment.
## Mechanism assessment (no demonstrated root cause)
CUDA Runtime API cudaHostRegister semantics, recalled offline: with PageableMemoryAccessUsesHostPageTables=1,
registration does not page-lock the range, but populates unpopulated pages. PageableMemoryAccess=1 ALONE
is insufficient to infer this or a no-op; population still matters. Query both attributes and driver/kernel versions.
CUDA for Tegra's coherent/pageable-memory model permits host-page-table access; do not infer ATS from integrated=1.
No local NVIDIA API manual was found; these semantics need version-matched confirmation on the window host.
On a real pinned path, pins normally prevent migration/collapse of the physical pages; on host page tables,
MMU invalidation must track collapse. THP is not documented here as forbidden for cudaHostRegister.
(a) Stale GPU PTEs after collapse would require a driver/kernel invalidation or pinning defect, not normal THP behavior.
(b) Arithmetic DOES show sharing: stride=12,222,464 B (already 4096-aligned), remainder=1,736,704 B modulo 2 MiB.
But it does NOT establish the proposed NaN mechanism: S:2501/P:4640 rounds the drop range INWARD to base pages.
Linux anonymous THP partial MADV_DONTNEED splits/zaps the requested base pages; it does not round out to 2 MiB
and discard a live neighbour. Whole-page collateral loss needs an extra bug/explicit outward rounding, absent here.
Nor does O_DIRECT require 2 MiB alignment just because its destination has THP; current 4 KiB offsets/lengths remain valid.
Dropping live quantized bytes could corrupt output, but zero bytes do not necessarily produce NaNs.
(c) mbind precedes host population; failed host bind warns and population has fallback. NUMA/THP allocation failures
normally affect placement/latency, not contents. No post-fill migration or destructive mbind is visible here.
Ranking: first isolate shared registered staging and I/O/publication races (S:516 documents a fixed pool race);
then (a) only as a driver defect; (b) as stated contradicted by drop semantics; (c) lowest absent failed-call evidence.
None is proven; aligning slots is useful isolation/THP retention engineering, not a demonstrated NaN fix.

## Exact proposed hunk sketches (opt-in THP-bank mode, not the global staging switch)
S:2914/P:5036: slot_stride=GGML_PAD(record_size, thp_bank ? 2*MiB : 4096); retain type-divisibility checks.
S:2930/P:5237 allocator call: pass explicit bank policy; reserve size+2MiB, align base, unmap prefix/suffix;
round region/dummy and HBM boundary to huge-page size; retain munmap ownership/length, overflow checks and budget limits.
S:1631/P:3855: bind -> MADV_HUGEPAGE -> populate -> MADV_COLLAPSE before backend registration.
Check/log every result; gate unsupported COLLAPSE or incomplete huge backing rather than silently calling it THP.
Use 2 MiB only on the Spark after checking kernel PMD size; GH200 64 KiB kernel used 512 MiB THP, not this geometry.
Lock bank before memory-intensive model load/registration when supported; measure AnonHugePages before/after load.
mlock alone does not guarantee huge backing; later regrow must repopulate/collapse only quiescent, wholly owned pages.
S:2501/P:4640: in THP mode round inward to huge-page boundaries; drop only pages whose slots are all evicted,
with existing LRU/in-flight/GPU quiescence guards; unlock/drop/relock only those pages; log actual returned bytes.
S:3826/:4032/:4781/:4959/:5114 (P:5891/:6092/:7031/:7185): check all landing paths, including staging.
Under HOOK_STATS, once per 1000 fills compare first 64 bytes of EACH landed record with store e.offset(expert),
after all pieces complete and before ready publication; use a separate buffered store fd (64 B is not O_DIRECT-safe).
Abort with layer/expert/slot/offset on mismatch; no existing bank checksum found (S:393/:2920 are header/layout only).
Prefix equality cannot detect tail damage or GPU mapping errors: add boundary/tail canaries and finite GPU-output checks.

## Capacity and window gates
80,000 MiB aggregate: 6,863 old slots -> 6,666 padded slots, 197 fewer (2.8705%); group/dummy rounding adds loss.
Pad=360,448 B/slot; 6,666 slots spend 2,291.4375 MiB padding; lost payload=2,296.28125 MiB.
Potential exposed saving=37*(1-3.9/4.5)=4.93 ms/token; worthwhile only if extra misses cost less than that.
Legs: A current no-THP; B aligned bank-only THP/collapse, valve off with headroom; C B+huge-aware valve;
D staging-only THP diagnostic; A2 baseline repeat. Freeze window-w snapshot/prompt/seed/predictor and 80,000 MiB budget.
Gates: injected prefix corruption must abort; injected nonfinite GPU output must fail; zero real byte/finite/MD5 failures.
Repeat startups/prefill beyond 7 trials; force valve pressure/regrow with live-neighbour checks, no in-flight drops.
Accept only retained huge backing after load, no swap/OOM, and lower exposed read/token time after capacity-miss cost;
report fill p50/p90/p99, miss counts, THP coverage/splits, valve bytes, output hash, tok/s; D must pass before any combined mode.
