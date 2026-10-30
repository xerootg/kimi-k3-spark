Implemented GLM exact CPU kernels and g53cs in tools/glm53/cpu/.
- AVX2/VNNI IQ2_XS/IQ3_XXS/IQ4_XS integer dots; exact AVX2 fallback.
- CUDA virtual-thread FMA/reduction, half q8 scales/float sums, MUFU SwiGLU,
  FTZ and canonical NaN match unchanged g53ref_exact.
- Persistent workers, preallocated scratch, raw read-only/prefaulted MUFU tables.
- K3 wire protocol, K3XS v1/v2, expert_disk.h RAM/disk tier, deadlines/pins.
- Valid zero rows preserved; unsupported/missing rows fail closed as NaNs.
- Startup requires mlockall; G53_ALLOW_UNLOCKED=1 explicitly permits test mode.
- Added allocation/page-fault instrumentation, offline tests, CI workflow,
  tools/glm53/cpu/README.md and docs/glm53-kernels.md section 10.

Tests (from checkout root, no downloads):
  tools/glm53/cpu/test.sh ~/glm53-work/data ~/glm53-work/mufu
Last lines:
  BENCH L75 threads=8 k=4 fast_us/expert=3868.32 scalar_us/expert=23721 ratio=6.13211
  BENCH L75 threads=8 k=8 fast_us/expert=3384.93 scalar_us/expert=18838.6 ratio=5.56543
  PASS ALL GLM CPU exact tests
All g/u/h/down words passed: three layers, eight experts, four cases, 4/8 workers.
600 synthetic rows include all signed q8 bytes and exceptional half scales.
All six final layer/thread gates: allocations=0, warm minor/major faults=0/0.

  python3 tools/glm53/cpu/test_server.py
Last lines:
  PASS g53cs: reference wire rows k=1/4/8/16, valid zeros, unsupported/missing ids, RAM/disk, duplicates, partial misses, deadline, zero request allocations
  PASS ALL GLM CPU server/disk regressions
Both store versions and heterogeneous gate/up types matched scalar reference.
Shared disk unit suite also passed pins/LRU/read failures/concurrency/deadlines.
  G53_TEST_CLOSED=1 python3 tools/glm53/cpu/test_server.py
Last line: PASS ALL GLM CPU server/disk regressions
  G53_VNNI=' ' tools/glm53/cpu/test.sh
Last lines: PASS 600 AVX rows vs g53ref_exact (all q8 bytes, exceptional half scales, CUDA reduction)
  SKIP extracted slices and table-dependent end-to-end: pass DATA MUFU
Production static build: tools/glm53/cpu/build.sh (PASS).

Meteor Lake Core Ultra 7 165H: median us/expert, k=1 / 4 / 8:
Layer threads   fast us/expert             scalar/fast ratio
L3    4         4849 / 5076 / 4698          5.83 / 4.72 / 5.78
L3    8         4080 / 3817 / 3585          6.26 / 5.47 / 6.72
L8    4         7774 / 6459 / 5730          4.09 / 5.01 / 5.41
L8    8         4458 / 3793 / 3748          4.54 / 5.75 / 5.12
L75   4         5644 / 5455 / 5794          4.79 / 4.55 / 5.10
L75   8         4226 / 3868 / 3385          4.89 / 6.13 / 5.57
Fast median of 7; scalar median of 3; same workers; finite uniform [-1,1] x.
Shared host, unpinned workers; no socket/disk time. Full raw output saved in
tools/glm53/results/cpu-exact-mtl-165h.txt.
NUC commands and full scalar timings are in docs/glm53-kernels.md section 10.

Limitations/uncertainty:
- 4.09–6.72x faster, but k=8 still ~27–30 ms/layer at eight threads:
  the earlier ~3 ms/layer budget is NOT met; further optimization is needed.
- Submit is synchronous: compute and socket transmission do not overlap.
- Local sockets are prohibited here: wire integration used the same request
  loop over test-only pipes; CI selects TCP. TCP/NUC execution remains unrun.
- Memlock is denied here: tests used explicit unlocked mode. Observed warm
  zero faults do not guarantee residency; locked NUC execution remains unrun.
- No new CUDA run; exactness is against the previously CUDA-verified oracle.
No network, commits or pushes. All task artifacts are inside this checkout.
