Review: 0011 v2 FIX FIRST; 0010 v2 FIX FIRST; 0014 pb FIX FIRST; 0012b SHIP TO THE SPARK; 0012c FIX FIRST.
References use work2/<stack>/ before follow-ups; P1 correctness, P2 hot-path cost, P3 validation gap. Findings ranked within each patch. CUDA hunks reviewed previously; no nvcc/GPU/model here, so no production byte/timing certification.
Base B = canonical 0001-0006 + 0008, reconstructed in work2/base; base-normalization.patch removes supplied llama/'s extra staged-prefill warming block. apply.log records all applications.
0011 v2 stack: B + 0009 + 0012 + 0011 v2 (work2/dsa).
- P2 dsa/src/llama-graph.cpp:647: per-decode identity vector allocates; fix: own/populate identity storage during graph construction and upload after each allocation.
- Original unused-mask crash is guarded; reuse checks widths/token counts, threshold crossing rebuilds identity input; prefill unchanged; same positional mask bits and attention inputs in source. GPU lifecycle/bytes remain untested.
0010 v2 stack: B + 0010 v2 (work2/bank).
- P2 bank/src/llama-moe-cache.cpp:3799, :675: fill jobs/pool/uring scratch can allocate on request; fix: reserve full expert union and persistent reader scratch at setup.
- P3 bank/src/llama-moe-cache.cpp:3073; bank/ggml/src/ggml-cuda/ggml-cuda.cu:1983: repeated memfd alias registration/coherence unverified; fix: standalone Spark register/write/read/unregister probe, reject fallback.
- Page alignment/overread guard and separate >32-token views preserve decode capacity and exclude landing records from decode LRU; no definite remaining short-request byte change found. Same-stream arena handoff is ordered; concurrent contexts unverified.
0014 pb stacks: B + 0010 old + 0014 pb (exact-old), or B + 0010 v2 + 0014 pb (exact-v2); references below use exact-v2.
- P1 exact-v2/src/llama-moe-cache.cpp:4698: up to 32 prediction IDs reserve slots before trimming to 16, risking live-row eviction/demand assertion; fix: 0014-reservation-headroom.patch.
- P2 exact-v2/src/llama-moe-cache.cpp:4682, :4714: union growth/sort can allocate; fix: preallocate 4*n_expert_used per-layer/planner/reader scratch and use allocation-free stable ordering.
- P2 exact-v2/ggml/src/ggml-backend.cpp:55; ggml/src/ggml-cuda/common.cuh:53; ggml/src/ggml-cuda/top-k.cu:296: unset TRACE still calls/loads/classifies; fix: gate before diagnostic calls/atomic marker/name classification.
- prediction_batch precedes fill_decode; both rebases compile. TRACE alone preserves top-k dispatch; documented graph bypass/synchronization applies when tracing. Exact CUDA pool growth on first verify is unproven; reserve before serving.
0012b stack: B + 0009 + 0012 + 0012b (stats). No blocking finding: added clocks/scans gated by STATS, two preallocated histograms, locked map reads, teardown after poller join; no new wait.
0012c stack: B + 0009 + 0012 + 0012b + 0012c (wide).
- P1 wide/src/llama-moe-cache.cpp:4711: 12 reservations can evict executing rows and starve real demand; fix: 0012c-reservation-headroom.patch. PREFETCH_MAX=16 admits 12; setting 8 trims too late.
- P2 wide/src/llama-moe-cache.cpp:4697, :4739: vectors can grow and >8 fills use allocating stable_sort; fix: reserve all bounded per-layer/planner/reader storage at setup and extend allocation-free stable ordering.
Fix applicability and A/B semantics (two alternative follow-ups; do not stack them together):
- review2/fixes/0012c-reservation-headroom.patch applies on B + 0009 + 0012 + 0012b + 0012c, verified against wide.
- review2/fixes/0014-reservation-headroom.patch applies on B + 0010 old + 0014 pb OR B + 0010 v2 + 0014 pb, verified against both exact trees.
- BOTH fixes are UNCONDITIONAL within enabled speculative prefetch, NOT gated by LLAMA_EXPERT_STORE_PREFETCH_WIDE=1 or LLAMA_SPEC_EXACT=1. With prefetch disabled the added code is not reached.
- Flag off: ordinary eight-row predictions also scan slots, protect the executing layer and may reject speculation; eviction choices, hit rate, I/O and measured baseline timing can change. No new allocation in the follow-up scan.
- Acceptable for an incremental feature A/B ONLY with the same fix in both legs and a fresh baseline; not an unchanged historical baseline. Scan cost is O(slots*n_ids) and unmeasured.
- Output bytes MUST NOT change: valid eviction choices change hit rate, never expert bytes, router IDs/order or arithmetic. Fixes prevent invalid live-row overwrite; prior corrupted/crashing runs are not a byte reference. Model hash equality still requires Spark validation.
Re-run commands/output (all new temporaries/logs under work2/tmp/, retained; no network):
`python3 work2/tmp/planner-repro.py > work2/tmp/planner-repro.log 2>&1` (exit 0; expected assertion child):
  planner: /claude/claude-1000/-home-xero-repos-argo-things/fd7931bb-f519-4321-a3de-81d4f945b7bf/scratchpad/sol-review/work2/tmp/planner.cpp:207: bool moe_cache_lru::plan(int32_t, const int32_t*, size_t, std::vector<fill>&, size_t&, bool): Assertion `s >= 0' failed.
  PASS: widened prediction evicts 4 still-readable current-layer rows at 16 slots
  PASS: 12 in-flight predictions starve a disjoint real lookup (assertion reproduced)
`python3 work2/check.py > work2/tmp/check.log 2>&1`: C++17 -fsyntax-only -pthread, each stack's -I{include,src,ggml/include,ggml/src}; output:
  dsa/src/llama-graph.cpp: PASS
  dsa/src/models/glm-dsa.cpp: PASS
  bank/src/llama-graph.cpp: PASS
  bank/src/llama-moe-cache.cpp: PASS
  exact-old/ggml/src/ggml-backend.cpp: PASS
  exact-old/src/llama-context.cpp: PASS
  exact-old/src/llama-graph.cpp: PASS
  exact-old/src/llama-moe-cache.cpp: PASS
  exact-old/src/models/glm-dsa.cpp: PASS
  exact-v2/ggml/src/ggml-backend.cpp: PASS
  exact-v2/src/llama-context.cpp: PASS
  exact-v2/src/llama-graph.cpp: PASS
  exact-v2/src/llama-moe-cache.cpp: PASS
  exact-v2/src/models/glm-dsa.cpp: PASS
  stats/src/llama-moe-cache.cpp: PASS
  wide/src/llama-moe-cache.cpp: PASS
  wide/src/llama-graph.cpp: PASS
  dsa git diff --check: PASS
  bank git diff --check: PASS
  exact-old git diff --check: PASS
  exact-v2 git diff --check: PASS
  stats git diff --check: PASS
  wide git diff --check: PASS
`python3 work2/check-fixes.py > work2/tmp/fix-check.log 2>&1` (application, fixed full-TU syntax, diff --check):
  wide: follow-up applies; full TU syntax and diff --check PASS
  exact-old: follow-up applies; full TU syntax and diff --check PASS
  exact-v2: follow-up applies; full TU syntax and diff --check PASS
`ASAN_OPTIONS=detect_leaks=0 python3 work2/tmp/test-followup.py > work2/tmp/followup.log 2>&1`: PASS: follow-up rejects 16-slot speculation, preserves current rows, allows 32-slot speculation and disjoint demand
