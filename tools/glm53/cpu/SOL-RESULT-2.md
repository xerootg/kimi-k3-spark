Optimized exact GLM CPU kernels; no float dependency/reduction order changed.
Baseline: fdbe758a0ae0; host: Meteor Lake Core Ultra 7 165H (not the 225H NUC).

Profile first: perf stat + perf record -e cycles:u -g on the original benchmark.
Initial row kernels consumed 83–84% of cycles; caller spin 8–11%.
IQ2 gate/up alone: 34–40%; its annotated scalar scaling/truncation/loop: ~34% locally.
Grid decode ~17%, dot/correction/horizontal sum ~21% of IQ2 local samples.
Final pinned profile: E cores ~99% rows; P cores ~74% rows, ~16% caller spin.
Matched perf stat: retired instructions 144.57B -> 114.35B (~21% fewer),
cycles 44.41B -> 40.15B; counters include fixture setup and have multiplexing.

Changes: cache aligned q8 bytes, exact integer half-vector bias corrections,
and converted half scales once per request/phase instead of once per row.
SIMD integer scaling/truncation and unrolling for IQ2/IQ4; compact scalar IQ3.
SIMD final float butterfly keeps each output's original addition subtree/order.
Isolate queue/barrier atomics by cache line; prefault scratch/worker stacks,
warm every row specialization on every worker, and wait for worker startup.
Kept compact grids, dynamic 32-row P/E work claiming, and original spin completion.
Added paired benchmark, read-only mapped fixtures, strict allocation/fault checks.
THP is disabled only in test/benchmark processes; no host settings changed.

Median ms/expert; each cell is k=1 / 4 / 8; 51 alternating baseline/current pairs.
SYNTHETIC: random packed grids, finite scales, nonzero [-1,1] x, zero gate;
closed-form MUFU subset. These are not extracted-weight/nonzero-gate timings.
Workers: P=0,1,3,6; append E=12..15 (8 threads), E=12..19 (12); caller=8.

| Format | Threads | Before | After |
|---|---:|---|---|
| L3 | 4 | 3.189 / 3.132 / 2.733 | 2.636 / 2.460 / 2.219 |
| L8 | 4 | 3.162 / 3.228 / 4.271 | 2.859 / 3.012 / 3.933 |
| L75 | 4 | 5.691 / 4.861 / 4.259 | 4.170 / 3.640 / 3.252 |
| L3 | 8 | 2.815 / 2.689 / 2.918 | 2.354 / 2.227 / 2.434 |
| L8 | 8 | 2.352 / 2.210 / 2.726 | 2.169 / 2.044 / 2.518 |
| L75 | 8 | 2.352 / 2.267 / 2.577 | 1.956 / 1.808 / 2.068 |
| L3 | 12 | 2.743 / 2.080 / 2.325 | 2.290 / 1.802 / 1.961 |
| L8 | 12 | 1.941 / 1.725 / 2.013 | 1.824 / 1.597 / 1.847 |
| L75 | 12 | 2.030 / 1.792 / 2.187 | 1.729 / 1.522 / 1.805 |

Example: `G53_BENCH_CALLER=8 G53_BENCH_CPUS=0,1,3,6 G53_BASELINE_REF=fdbe758a0ae0 tools/glm53/cpu/bench.sh 4 51`
Raw timings/profiles in tools/glm53/cpu/build/: paired-final-*.log, stat-final.txt, stat-original-final.txt, profile-final.txt.

Tests (checkout root; no network):
`tools/glm53/cpu/test.sh`  
`G53_VNNI=' ' tools/glm53/cpu/test.sh`  
Each test binary passed four consecutive runs; last output lines for both:
`PASS closed pool IQ2/3/4 up words; k=1/4/8 threads=4/8/12 allocations=0 faults=0/0`  
`SKIP extracted slices and table-dependent end-to-end: pass DATA MUFU`  
`PASS ALL AVAILABLE GLM CPU exact tests`  
600 arbitrary-q8/exceptional-half rows also matched the unchanged scalar oracle.
`G53_TEST_CLOSED=1 G53_TEST_PIPE=1 python3 tools/glm53/cpu/test_server.py`  
Last: PASS ALL GLM CPU server/disk regressions
`tools/glm53/cpu/build.sh`  
Exit 0; production static build produced no output. git diff --check passed.

Uncertain: real weights/MUFU files are absent inside this checkout; those tests
and actual 225H execution remain unrun. Shared-host clocks/migration vary;
anonymous-fixture/futex-wait experiments showed intermittent minor faults.
Final strict gates observed 0/0; unlocked observations cannot guarantee residency.
The ~0.2 ms bandwidth floor remains unmet. No network, commits or pushes.
