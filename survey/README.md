Latency survey kit (offline preparation; owner executes on real hosts)

From the laptop checkout, with passwordless SSH as xero by IP:

```bash
./survey/run.sh all
./survey/run.sh net
./survey/run.sh nvme
./survey/run.sh igpu
./survey/run.sh e2e
python3 survey/analyze.py survey/results/TIMESTAMP --log /path/to/glm-server.log
python3 survey/analyze.py survey/results/TIMESTAMP --trace /path/to/existing-remote-trace.tsv --budget-us 3000
```

Prerequisites: laptop bash, GNU timeout, Python 3, scp/ssh, g++, existing oneAPI
container used by dsycl.sh, `W=~/glm53-work` with llama headers, and static C++/libc development libraries for CPU binaries. Verified MUFU
tables must already exist at `~/glm53-mufu/{rcp,ex2p,ex2n}.bin` on both NUCs.
The NUCs need Python 3, timeout,
passwordless sudo, render permissions, and the runtime at ~/k3-bitexact/sycl/lib.
The laptop builds SYCL plus fully static CPU/NVMe executables. A remote g++/liburing
installation is optional; otherwise NVMe uses the shipped static pread binary.
`--skip-build` reuses **survey-instrumented** `$W/survey-build/g53x` and g53c.
Do not supply an uninstrumented production binary: raw bench samples would be absent.
No extract/download script runs. L3 records come from each NUC's actual store.

Each invocation creates a unique remote directory `/home/xero/glm53-survey/TIMESTAMP`
and local `survey/results/TIMESTAMP` (override with `--results DIR`). Repeat runs
leave old results intact. A rerun never adopts or kills an existing server. Port
29653 (29654 for reverse TB) must be free: a conflict fails instead of killing its
owner. Server sessions close through their own stdin; server children have owned
process groups and terminate/kill escalation. Remote GNU timeout bounds orphaned
SSH sessions as well. Remote files intentionally remain for inspection/removal.
The store is opened read-only; no cache flush, drop_caches, sysctl, forcewake,
frequency/EPP, cpuidle, NIC or pod settings are changed. No CUDA library, engine
endpoint, or port 8080 is used. Spark runs only Python network probes and metadata.

Measurements and interpretation:

- net: TCP_NODELAY, persistent connection, 10 warmups, target 1,000 observations
  of send-to-last-byte latency at k=1/2/4/8. Request is exactly 68+24,576 bytes;
  reply is k*24,576 bytes. Empty-operation ping-pong uses a one-byte framing marker
  each way (TCP cannot transmit a zero-byte message). One 128 MiB client-to-server
  bulk transfer follows each path. Measures Spark->each NUC and both NUC Ethernet
  directions. TB /30 peer addresses must match nuc3's reported TB interfaces;
  forward and reverse routes are checked. All matching TB links run both directions.
  Bulk is one stream, including final acknowledgement; not a latency distribution.
- metadata: uname, IP addresses/routes, disk model, ethtool driver/coalescing/EEE,
  cpuidle files, EPP, and sudo fuser holders of /dev/cpu_dma_latency. Command errors
  are retained, never interpreted as disabled settings. No privileged writes.
- nvme: new GLM parser uses validated K3XS v2 offsets/stride, selects ONLY real
  12,091,392-byte records. QD1 whole-record and 512 KiB/1 MiB/4 MiB pieces at
  QD1/4/8/16, for one and eight distinct records from the same randomly chosen
  layer. Last piece is short and 4 KiB aligned. Persistent workers, one ring per
  worker when liburing builds and initializes; otherwise pread. Backend counts
  are saved. Queue depth is aggregate across all pieces/records. Reported unit
  is completion of the entire record or eight-record group, including submission
  and worker wake-up. Five warmups; O_DIRECT bypasses page cache, not SSD firmware
  caches. Not sustained thermal endurance or SSD power-loss testing.
- igpu: same eight L3 experts, fast/exact raw g53x bench samples, g53c eight-thread
  CPU bench, and same-host serveidx client latency at k=1/2/4/8. Native bench modes
  replicate eight experts into 64 weight slots; serveidx holds eight. Both are
  reported distinctly. taskset 4-11; g53x bench and serve use sudo -u xero -g render.
  g53c has no server mode; k3cs implements K3 math, not a GLM CPU comparison server.
- e2e: Spark CPU client, serialized NUCs, fast/exact, k=1..8, resident, LRU hit,
  cold. Nonzero deterministic Gaussian activation, ten warmups, reply finite check
  outside timing. TCP has no correctness identity claim beyond this missing-row
  gate; existing EXACT-RESULT.md supplies the separate arithmetic evidence.
  No artificial interrequest gap by default. To examine idle/wake-up overhead,
  run probe.py client manually with --gap .008 (serve/client test tooling only).
- disk prototype: this checkout has NO g53x disk tier or disk_test.py. prepare.py
  generates an isolated survey variant: eight repacked host-USM slots with LRU
  ages, O_DIRECT full-record pread on misses, synchronous repack before original
  g53x compute. It is deliberately labelled prototype, not a shipped tier.
  Hits reuse repacked slots; misses rotate over all 256 L3 experts. Counters in
  server logs prove LRU requests had no reads and cold traffic did read. Request
  ids are pinned against eviction; clients finish replies before the next request.
  No deadline truncation, io_uring pipeline, or production disk-tier performance
  is implied. Production offload needs a separately reviewed implementation.

Bounds and statistical limits:

`run.sh` enforces an overall 1,180 s limit plus 15 s termination grace. Sections
have watchdogs: net 240, NVMe 180, iGPU 480, e2e 480 s. Startup/read/connect/build
operations also have timeouts. Typical target is under 20 minutes, including a
cached container build; slow compilation/hosts may leave partial results.
Default per-case caps: NVMe 4 s, e2e 4 s, loopback 8 s, native bench 20 s.
The exact path documented in EXACT-RESULT.md costs ~12 ms/expert on the NUCs in this run: 1,000 samples
for every exact case alone cannot fit this runtime. Partial raw samples survive
bench timeouts; analyzer labels <1,000 samples weak. A p99 of 1,000 samples still
has only ~10 upper-tail observations. Counts and max always accompany percentiles.
Caps include warmups, which can consume much of an exact k=8 case.

For a deliberate longer tail run (changes the runtime target), for example:

```bash
SURVEY_TOTAL_SECONDS=5000 SURVEY_SECTION_SECONDS=4800 \
SURVEY_CASE_SECONDS=240 ./survey/run.sh e2e --skip-build
SURVEY_TOTAL_SECONDS=2200 SURVEY_SECTION_SECONDS=2000 \
SURVEY_BENCH_SECONDS=240 SURVEY_SERVE_SECONDS=240 ./survey/run.sh igpu --skip-build
SURVEY_TOTAL_SECONDS=1500 SURVEY_SECTION_SECONDS=1400 \
SURVEY_NVME_SECONDS=40 ./survey/run.sh nvme
```

SURVEY_SAMPLES controls target count (default 1,000); other cap variables are
SURVEY_CASE_SECONDS, SURVEY_SERVE_SECONDS, SURVEY_BENCH_SECONDS,
SURVEY_NVME_SECONDS, SURVEY_SECTION_SECONDS and SURVEY_TOTAL_SECONDS.
Keep counts/caps unchanged when comparing hosts. Timing runs can contend with
other NUC workloads; this kit never stops those workloads. Benchmark CPU load,
SSD reads and container compilation can temporarily affect shared resources.

Analyzer uses nearest-rank quantiles of raw observations, grouped by hop/k/tier/
math mode; summary means from logs are kept in separate rows. It prints one
latency table, bulk rates separately, missing-data/error notes, and p99 deadline
checks for Spark e2e only. Fast math is not certified offload even when it fits.
The 3,000 us default is K3's assumed deadline; 235 ms/75 is only an average layer
spacing, not proof of the post-to-real-lookup window. Decode tok/s in supplied
logs estimates mean layer spacing; existing `wait at the real lookup` and
`per layer fill` summary formats supply aggregate means. `--trace` accepts the
existing tab-separated remote trace (CLOCK_MONOTONIC_RAW ns): t_post to
 t_late_hook is the available request window; t_resolve to t_wait_end is wait.
No trace is enabled and no signal is sent to the live Spark engine. Use a measured
lower-tail window with --budget-us; analyzer never substitutes a median window.
Successful microbench replies do not establish real decode overlap or throughput.

Offline validation:

```bash
bash -n survey/run.sh
python3 -m py_compile survey/*.py
python3 survey/prepare.py survey/.validation-build
g++ -std=c++17 -pthread -Itools/bitexact -fsyntax-only survey/glm_readbench.cpp
g++ -std=c++17 -march=alderlake -mavxvnni -pthread -Itools/glm53/ref -fsyntax-only survey/.validation-build/g53c.cpp
python3 survey/validate.py
python3 survey/validate_second_pass.py
```

No SYCL compiler, liburing headers, real hosts or hardware are available in this
checkout environment. Thus generated g53x SYCL compilation, remote lifecycle,
GPU timing, direct I/O, and runtime-bundle compatibility remain owner checks.

Second-pass evidence and remaining gaps: [FINDINGS.md](FINDINGS.md); full error
reconciliation: [DIAGNOSIS.md](DIAGNOSIS.md). Rebuild after updating this kit;
`--skip-build` can retain the old dynamically linked, incompatible CPU binary.
