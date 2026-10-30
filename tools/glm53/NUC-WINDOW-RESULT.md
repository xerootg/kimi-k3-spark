Kubernetes follow-up: see ../../RESULT.md for current usage and validation.
The native-mode notes below describe the original implementation.

Implemented tools/glm53/nuc-window.sh, its remote stdlib driver nuc-window.py,
and test_nuc_window.py. No network, commits, pushes or hardware runs performed.

Owner usage from laptop (Bash, Python 3, SSH/SCP, Docker and kubectl required):
  tools/glm53/nuc-window.sh stage
  tools/glm53/nuc-window.sh start
  tools/glm53/nuc-window.sh probe
  REMOTE_PATCH=/path/to/0013.patch tools/glm53/nuc-window.sh ab
  ROWS=2 tools/glm53/nuc-window.sh ab
  tools/glm53/nuc-window.sh stop

stage defaults to ~/glm53-work/g53x and the existing build container image;
it never pulls an image. Inspects g53x's ELF dependencies plus dlopened SYCL,
UR and Level Zero libraries under /opt/intel/oneapi/compiler/latest/lib.
Copies only missing bundle filenames through the laptop; checks ldd on NUC
for the binary and adapters. Existing bundle precedes added libs in search path.
If an existing library has incompatible symbols/versions, repair the bundle.
No actual runtime inventory could be collected without laptop/container access.

Placement defaults to /srv/glm-5.3/placement/nuc2-ram.set on Spark, then NUC.
Spark placement transfers directly over LAN. PLACEMENT overrides the path.
Without placement, synthetic hotness ranks expert IDs ascending, then layers
3..77 ascending: a stated low-ID proxy, NOT measured hotness. Selects the first
N fitting 44 decimal GB using actual store g+u+d bytes, matching g53x's cap.
start requires every selected resident to appear in K3X_SETOUT, prints its
served count, and exports only resident owners. Disk-wide served IDs are never
used as resident ownership. It waits for 'serving on PORT (busy-poll)'.

NUC launch uses sudo/root prlimit then setpriv xero:render for unlimited memlock,
nohup, CPUs 4-11, choom 1000, G53X_EXACT=2, required MUFU directory,
K3X_MAX_GB=44, disk tier=1, LRU=1 GB, disk deadline=20 ms, HIDDEN=6144.
Requires sudo permission for prlimit/setpriv; uses the supplied runtime bundle,
store and tables in place. nuc3 is deliberately unused in this first test.

probe runs on Spark: persistent TCP, 17 little-endian int32 header, 6144 f32
input, distinct resident IDs in one layer, k=1/2, 10 warmups + 200 timed each,
p50/p99 wall RTT in ms, finite-response gate. It does not prove bit equality.

ab applies supplied 0013 only when source markers are absent, then rebuilds.
0013 itself is absent here; markers are a prerequisite, not a patch audit.
Uses documented scale-to-zero command, waits for ALL llama-server processes,
never kills unrelated servers, and restores one replica/rollout via EXIT trap.
Requires initial replicas=1. SSH loss/laptop crash can require manual restore.
Four unique artifact legs: off/on/off/on, same frozen snapshot, 80000 MiB,
65536 context, v2 store, documented prompt/flags, warm + two timed 128-token
requests. Every timed response requires md5 prefix 9449ae5738 and byte equality;
on legs also require nonzero remote used rows and print remote layer stats.
Default remote deadline=2500 us, ROWS=1; repeat ROWS=2. PID identity uses Linux
start time and executable; stop only signals this harness's recorded process.
Unverified: host paths, capacity, sudo/SSH between hosts, patch application,
CUDA/SYCL builds, actual readiness/latencies, MUFU/store numerical equivalence,
fixture digest with v2 store, and hardware throughput. Protocol has no handshake;
owner must establish matching weights/tables. No claim of speedup is made.
Offline test command:
  python3 tools/glm53/test_nuc_window.py
Last output lines: Ran 4 tests in 6.023s; OK.
Also passed: bash -n tools/glm53/nuc-window.sh; Python compile; git diff --check.
Tests mock networking; cover budget/placement, wire/fragmentation, stale PID
safety and unresolved-library rejection. Full SSH/pod/engine lifecycle untested.
