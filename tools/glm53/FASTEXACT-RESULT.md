## RESULT-1.md
Implemented G53X_EXACT=2, a parallel CUDA-exact IQ candidate in g53x.
G53X_EXACT=1 remains the scalar exact default until iGPU verification.

Changes:
- SIMD16 subgroup per row: two CUDA lane positions per Intel lane, all four
  virtual warps; preserve per-thread FMA order, ordered warp adds, XOR tree.
- Field-run weights, integer DP4A IQ2/3/4 dots, ALU sign masks, SLM grid tables.
- Quantize once per work-group with explicit CUDA-order subgroup shuffles;
  preserve half(float block sum), half(scale), and broadcast one MUFU RCP/block.
- Preconvert activation scales in SLM; use unchanged MUFU SwiGLU.
- IQ row intermediates are multiples of 2^-48 (two half scales times integers):
  no FP32 subnormals, so omit redundant FTZ tests there; canonicalize every NaN.
  Quantization/SwiGLU retain full FTZ helpers. Separate scale multiply retained.
- cuda_serve_check.py now respects G53X_EXACT instead of forcing mode 1.
- Shared SIMD arithmetic + CPU emulation in g53parallel.h/test-parallel.cpp;
  independent scalar g53ref_exact row/dot oracle, read-only extracted slices.

Local test command (from checkout root):
  ./tools/glm53/test-parallel.sh
Last output:
  quantizer: 4096 SIMD16 float-sum/half-scale/q8 blocks bit-identical PASS (power-of-two RCP subset)
  synthetic: 768 rows; IQ2/3/4 DP4A + SIMD16 reduction bit-identical PASS
  L3: 81920 extracted gate/up/down rows, all 8 experts bit-identical PASS
  L8: 81920 extracted gate/up/down rows, all 8 experts bit-identical PASS
  L75: 81920 extracted gate/up/down rows, all 8 experts bit-identical PASS
  PASS: scheduling/integer emulation; GPU compilation, MUFU and performance require owner checks
Synthetic cases include signed truncation, cancellation, signed zero, half
subnormals, infinity and NaN. Extracted rows use seeded nonzero q8 activations.
Host C++ compilation/syntax passed -Wall -Wextra -Werror; shell/Python syntax
and git diff --check passed. No SYCL compiler/runtime or GPU measurements here.

Owner iGPU commands, from checkout root (verified GB10 tables in $W/mufu):
```bash
export W=$HOME/glm53-work G53X_MUFU=/w/mufu MUFU_DIR=$HOME/glm53-work/mufu
./tools/glm53/build-sycl.sh
G53X_EXACT=2 ./tools/glm53/check.sh
G53X_EXACT=2 ./tools/glm53/check-exact.sh
G53X_EXACT=2 G53X_FUSEQ=0 ./tools/glm53/check.sh
G53X_EXACT=2 python3 tools/glm53/cuda_serve_check.py
for ex in 0 1 2; do
  G53X_EXACT=$ex G53X_PROF=1 ./tools/glm53/dsycl.sh './g53x data/UD-Q2_K_XL_L8 18 18 23 bench 1,4,8 50'
done
```
check.sh writes fresh GPU stages and invokes check-exact.sh; the latter alone
compares existing chk stages. Repeat bench for L3 (17 17 18), L75 (17 17 23).
CUDA checker expects tools/glm53/records and evidence/cuda-dumps; override with
--records PATH --dumps PATH if needed. Existing zero-row NaN wire sentinels still
cause checker failures: inspect each difference; computed nonzero rows must match.

Unverified: SYCL compilation, IGC DP4A/shuffle lowering, GPU byte identity,
non-power-of-two MUFU quantization on GPU, and actual speed/register pressure.
CPU emulation proves scheduling/integer equivalence, not GPU code generation.
Promote mode 2 only after g/u/h/dn and CUDA checks pass and timing improves.
No network, commits or pushes; all writes stayed in this checkout.

## RESULT-2.md
Full CPU reproduction PASSES; the iGPU fix remains unverified.

Findings:
- Original g53parallel SIMD16 arithmetic, original subgroup quantization, real
  MUFU tables, recorded x/ids: all 96 cases match all CUDA g/u/h/dn words.
- This excludes an arithmetic/scheduling error for these dumps. It does NOT
  establish the precise SYCL/IGC failure mechanism. No GPU is exposed here.
- check-exact.sh only compares existing chk files; setting G53X_EXACT=2 on
  that command alone does not run mode 2 or validate serveidx's fused path.

Changes:
- Mode 2 fused quantization now gives one whole block to one work-item and
  calls g53exact::quant, as the verified scalar/separate paths do. Removed
  distributed Q8 writes and the divergent lane-zero RCP lookup/broadcast.
  This is a candidate mitigation, not a demonstrated GPU root-cause fix.
- SIMD16 row arithmetic, DP4A, reduction order and per-work-group SLM stay.
- serveidx now honors G53X_FUSEQ=0, including streamed per-expert offsets.
- test-parallel.sh defaults to all recorded cases, not extracted slices.
  It checks original subgroup quantization against scalar quantization too.
- Added test-parallel-sycl.{cpp,sh}: production kernels, every CUDA stage,
  fused/separate quantization, batch/stream down, all layers in one process.
  It prints the first mismatching stage/case/index; no wire sentinel masking.

CPU test command (from this checkout; tables are read-only):
  MUFU_DIR=$HOME/glm53-work/mufu ./tools/glm53/test-parallel.sh
Last output:
  L75: 32 cases, 256 expert rows; differing words g/u/h/dn=0/0/0/0
  quantizer: 67584 real-MUFU blocks, scalar/subgroup bit-identical PASS
  PASS: 96 CUDA dump cases, all recorded ids, all g/u/h/dn words bit-identical with real MUFU tables
L3 and L8 also report 0/0/0/0; 9,437,184 stage words compared in total.
Legacy test also passed:
  ./tools/glm53/test-parallel.sh --slices
  PASS: scheduling/integer emulation; GPU compilation, MUFU and performance require owner checks
SYCL production binary and stage-test binary compile with local icpx 2026.0.
GPU execution: "No device of requested type 'info::device_type::gpu' available."
Shell syntax and git diff --check pass.

Owner verification (uses the existing verified $W/mufu as previous runs did):
  export W=$HOME/glm53-work-exact G53X_EXACT=2 G53X_MUFU=/w/mufu
  ./tools/glm53/build-sycl.sh
  python3 tools/glm53/cuda_serve_check.py --no-server
  ./tools/glm53/dsycl.sh '/w/tools/test-parallel-sycl.sh /w/mufu /w/tools/records /w/tmp/cuda_serve_check/dumps'
  G53X_FUSEQ=1 python3 tools/glm53/cuda_serve_check.py
  G53X_FUSEQ=0 python3 tools/glm53/cuda_serve_check.py
  G53X_PROF=1 ./tools/glm53/dsycl.sh './g53x data/UD-Q2_K_XL_L8 18 18 23 bench 1,4,8 50'
Direct stage test must print PASS: 384 SYCL case/configurations, differing
stage words=0. Wire checks may still flag the documented all-zero sentinels;
computed rows and real CUDA NaN rows must match. No new speed measurements.
If the stage test fails with fused=0, investigate row/DP4A/shuffle lowering;
if only fused=1 fails, the failure is in fused block staging/quantization.
Do not promote mode 2 until those owner checks pass. Exact GPU root cause
and the candidate mitigation's effectiveness/performance remain uncertain.
No network, commits or pushes; all requested source/result edits are here.

## RESULT-3.md
Mode 2 now quantizes each distinct activation block once, outside row kernels.
The previous fused path repeated scalar quantization in every 16-row group:
L8 gate/up repeated x 128*k times, down repeated each h vector 384 times.
At k=8 this drops quantizer calls from 393216 to 704 blocks per request.
This is a source-level bottleneck finding, not a measured GPU speedup.

Changes:
- g53x_parallel.inc: block-parallel preparation into reusable device Q8/scales;
  gate/up shares one prepared x across all experts; down prepares each distinct
  h block once (streamed calls process disjoint experts). Separate-input mode
  packs existing q8/scales once. Cooperative byte copies populate row-group SLM.
- Scratch grows only for a new largest shape, after draining the in-order queue;
  it is reused across requests and streamed calls, with no steady-state allocs.
- Preserved scalar quantizer, half scales/sums, DP4A and ordered row arithmetic.
  No MUFU reads in staging or dot loops; SwiGLU still reads MUFU once per row.
  G53X_FUSEQ=1 uses a preparation launch in mode 2, rather than fused quantization.
- test-parallel-sycl.sh now invokes dsycl.sh from the host, compiling and running
  inside oneAPI. Its optional arguments are container paths; defaults use dumps
  prepared by cuda_serve_check.py --no-server. Both input paths remain tested.
  Stage test forces mode 2 even if the caller exported another exact mode.

Local gates (from checkout root):
  MUFU_DIR=/home/xero/glm53-work/mufu ./tools/glm53/test-parallel.sh
  L75: 32 cases, 256 expert rows; differing words g/u/h/dn=0/0/0/0
  quantizer: 67584 real-MUFU blocks, scalar/subgroup bit-identical PASS
  PASS: 96 CUDA dump cases, all recorded ids, all g/u/h/dn words bit-identical with real MUFU tables
  TMPDIR="$PWD/tools/glm53/test-parallel/tmp" /opt/intel/oneapi/compiler/latest/bin/icpx -fsycl -O2 -ffp-contract=off -fno-fast-math -std=c++17 -Itools/glm53/ref -Itools/bitexact tools/glm53/test-parallel-sycl.cpp -o tools/glm53/test-parallel/test-parallel-sycl
  Exit 0; no compiler diagnostics (includes the production kernels).
  ./tools/glm53/test-parallel-sycl.sh
  permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
  tools/glm53/test-parallel/test-parallel-sycl /home/xero/glm53-work/mufu tools/glm53/records evidence/cuda-dumps
  error: No device of requested type 'info::device_type::gpu' available.
  bash -n tools/glm53/test-parallel-sycl.sh; git diff --check
  Exit 0; no output.
Logs: tools/glm53/test-parallel/{cpu-current,container-test,sycl-current-build,
sycl-current-runtime}.log

Owner commands, from checkout root (existing verified tables at $W/mufu):
```bash
export W=$HOME/glm53-work-exact G53X_MUFU=/w/mufu G53X_FUSEQ=1
./tools/glm53/build-sycl.sh
MUFU_DIR="$W/mufu" ./tools/glm53/test-parallel.sh
python3 tools/glm53/cuda_serve_check.py --no-server
./tools/glm53/test-parallel-sycl.sh
G53X_EXACT=2 python3 tools/glm53/cuda_serve_check.py
for ex in 0 2; do
  G53X_EXACT=$ex G53X_PROF=1 ./tools/glm53/dsycl.sh './g53x data/UD-Q2_K_XL_L8 18 18 23 bench 1,4,8 50'
done
```
Stage gate must print: PASS: 384 SYCL case/configurations, differing stage words=0.
Wire gate must preserve changed=0 for computed rows (documented zero sentinels).
Compare total wall p50: gate/up and down profiling events exclude preparation.
Repeat benches for L3 (17 17 18) and L75 (17 17 23) with the matching data prefix.

Uncertain: GPU exactness after moving preparation, IGC resource use, and whether
mode 2 reaches <=2x mode 0. No iGPU or accessible Docker here, so those gates and
benchmarks remain owner checks. Scratch assumes the existing single in-order
queue/request model. No network, commits or pushes; edits and logs are here.

## RESULT.md
Candidate regression fix; owner iGPU gate is still required (no GPU here).

Changes:
- Kept once-per-distinct-block preparation, shared gate/up inputs, reusable
  scratch, cooperative SLM copies and SIMD16 row kernels (v3 speed structure).
- Preparation now explicitly uses WG256/SIMD16, matching passing v2's shape.
- Made quantizer binary16 rounding integer-defined, ties-to-even; preparation
  expands the stored half scale through integer bits. Preserves half subnormals
  and signed zero without native FP16 conversion/folding assumptions.
- Operation audit: still calls g53exact::quant; absolute max, FTZ inputs/results,
  16/8/4/2/1 butterfly adds, float d, real-table RCP(float d) BEFORE half rounding,
  roundaway bytes and half(float sum) are unchanged. No replacement sum(q).
  v2 used local Q8/half expansion; v3 added a global prepass and immediate
  half->float scale expansion, with an unconstrained subgroup size.
- test-parallel-sycl.sh includes /bitexact and captures stderr. Its gate checks
  synthetic edge preparation, actual x/h Q8 bytes, half(d), half(sum), expanded
  scales, then CUDA g/u/h/dn for both input paths and down-launch shapes.
  Preparation failures name block/field/byte; stage failures name case/index.
- cuda_serve_check.py --no-server also copies supplied g/u/h into container
  dumps; previously only x/ids/dn/kind were copied, leaving stage inputs absent.

Local tests, from checkout root:
  MUFU_DIR=/home/xero/glm53-work/mufu ./tools/glm53/test-parallel.sh
Last lines (tools/glm53/test-parallel/cpu-fix.log):
  L75: 32 cases, 256 expert rows; differing words g/u/h/dn=0/0/0/0
  quantizer: 67584 real-MUFU blocks, scalar/subgroup bit-identical PASS
  PASS: 96 CUDA dump cases, all recorded ids, all g/u/h/dn words bit-identical with real MUFU tables
Binary16 check passed all expansions, signed rounding boundaries and 1000000
random FP32 inputs against native CPU half conversion (including NaN payloads).

  SYCL_CXX=/opt/intel/oneapi/compiler/latest/bin/icpx ./tools/glm53/test-parallel-sycl.sh --inside /home/xero/glm53-work/mufu tools/glm53/records evidence/cuda-dumps
Build succeeds; runtime exits 1. Last output (sycl-fix.log):
  error: No device of requested type 'info::device_type::gpu' available.
  ./tools/glm53/test-parallel-sycl.sh
Exit 1 (container-fix.log):
  permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
Production SYCL binary also compiles; dump-copy smoke test passes.
Shell syntax, Python AST and git diff --check pass.

Exact owner commands from checkout root (existing verified tables at $W/mufu):
```bash
set -e
export W=$HOME/glm53-work-exact G53X_MUFU=/w/mufu G53X_EXACT=2 G53X_FUSEQ=1
python3 tools/glm53/cuda_serve_check.py --no-server
./tools/glm53/test-parallel-sycl.sh
./tools/glm53/build-sycl.sh
G53X_FUSEQ=1 python3 tools/glm53/cuda_serve_check.py
G53X_FUSEQ=0 python3 tools/glm53/cuda_serve_check.py
for ex in 0 2; do
  G53X_EXACT=$ex G53X_PROF=1 ./tools/glm53/dsycl.sh './g53x data/UD-Q2_K_XL_L8 18 18 23 bench 1,4,8 50'
done
```
Gate must end: PASS: 384 SYCL case/configurations, differing stage words=0.
Wire gate must restore changed=0. Compare wall p50 to 1923/5062/9239 us;
repeat L3 (17 17 18) and L75 (17 17 23) with matching data prefix.

Uncertain: precise IGC failure mechanism, whether this candidate fixes owner
bytes, and resulting iGPU timing. CPU success cannot establish GPU exactness.
No network, commits or pushes; all writes stayed in this checkout.

