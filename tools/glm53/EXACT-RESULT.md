Added tools/glm53/cuda_serve_check.py (Python standard library only).
- Builds K3XS v1 store + JSON sidecar + set covering layers 3/8/75, experts 0..7.
- Copies the supplied records without repacking; verifies every copied slot and padding.
- Validates all dump lengths, case labels, and expert ids; preserves raw float bytes.
- Runs /w/g53x serveidx through dsycl.sh with G53X_EXACT=1, G53X_MUFU=/w/mufu.
- Sends all cases with their original x and ordered ids using the 17-int k3x header.
- Compares every returned byte, including NaN payloads; exits 1 on any difference.
- Reports per-layer cases/rows/differing bytes and every differing case's kind label.
- Diagnoses changed NaN rows and zero rows replaced by NaN sentinels.
- Reports real all-NaN rows that the engine will treat as missing and compute locally.
- Verifies K3X_SETOUT covers all 24 experts; closes client and cleans up server.
- Saves inputs, store, wire replies, server.log, run.log, report.json under $W/tmp/cuda_serve_check.

Owner command, from checkout root (existing g53x and verified tables under $W):
```
W="$HOME/glm53-work-exact" python3 tools/glm53/cuda_serve_check.py
```
Optional preparation-only command:
```
W="$HOME/glm53-work-exact" python3 tools/glm53/cuda_serve_check.py --no-server
```
Local command (W inside checkout to obey write restriction):
```
W="$PWD" python3 tools/glm53/cuda_serve_check.py --no-server
```
Last output lines:
```
store v1: 3 layers, 24 experts, 338960384 bytes; copied records verified
L3: parsed cases=32 rows=256 CUDA NaN rows=7
L8: parsed cases=32 rows=256 CUDA NaN rows=8
L75: parsed cases=32 rows=256 CUDA NaN rows=7
DRY RUN PASS: store records verified and all dumps parsed; serveidx comparison NOT RUN
```
Additional local checks: Python syntax PASS; integer NaN classification (both payloads,
signed NaNs, infinity, zero) and payload distinction PASS.
Inspected dumps: each layer has 24 all-zero rows; L8 and L75 each have one all-NaN row.
Current g53x.cpp replaces all-zero rows with NaN sentinels. Therefore strict CUDA wire
identity is expected to FAIL those rows; this checker deliberately counts them as differences.
No engine changes were requested or made. An all-NaN row matching CUDA still triggers
engine fallback; the checker reports that ambiguity without calling matching bytes a failure.
Unverified: actual container/SYCL execution and wire results (unavailable here).
No network, commits, or pushes; all writes stayed inside this checkout.
Implemented a source-derived exact candidate, NOT yet CUDA-certified.
- SPEC: tools/glm53/SPEC-glm.md (source citations, launch shapes, integer truncation, SASS checklist).
- glmdump.cpp: production split-SwiGLU graph, eight ids, edge/random cases, four output graphs.
- g53ref_exact.cpp + g53exact.h: scalar CPU and shared FTZ/half/MUFU/IQ arithmetic.
- g53x: exact candidate defaults on; scalar per row, CUDA virtual-thread reduction, half q8 scales/sums.
  G53X_EXACT=0 selects the retained non-exact path. Exact requires verified RCP/EX2 tables.
- check.sh now checks Q2 g/u/h/dn bytes against CPU; Q4 regressions explicitly use non-exact mode.

Local test: ./tools/glm53/test-exact.sh
Last output (tools/glm53/test-exact/test.log):
```
synthetic: q8 half scale/sum, rounding, FTZ, FMA, SwiGLU zero, IQ signed truncation, raw/repacked K=2048/6144 PASS
L3: zero + signed denormal g/u/h/dn PASS; nonzero gate/up 32768/32768 finite nonzero
L8: zero + signed denormal g/u/h/dn PASS; nonzero gate/up 32768/32768 finite nonzero
L75: zero + signed denormal g/u/h/dn PASS; nonzero gate/up 32768/32768 finite nonzero
PASS (table-free subset only; random end-to-end requires GB10 MUFU tables and CUDA dumps)
```
CPU and glmdump pass -Wall -Wextra -Werror syntax checks; shell/Python checks pass.
ref/ lacks public ggml headers: glmdump syntax used ~/glm53-work/llama/ggml/include; CPU uses ref/ggml-common.h.
No network, CUDA/iGPU execution, performance measurements, commits or pushes.

Owner: prepare records on laptop, then place tools/glm53/records/ in the Spark checkout:
```
python3 tools/glm53/exact_test.py records ~/glm53-work/data tools/glm53/records
```
On Spark, from checkout root (tables already at ~/k3bx/mufu):
```
T=$HOME/llama-k3-sc
B=tools/glm53
c++ -O2 -std=c++17 -I"$T/ggml/include" "$B/glmdump.cpp" -o "$B/glmdump" -L"$T/build/bin" -Wl,-rpath,"$T/build/bin" -lggml -lggml-base
c++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math -Iref -Itools/bitexact "$B/g53ref_exact.cpp" -o "$B/g53ref_exact"
for spec in '3 17 17 18 12091392' '8 18 18 23 16318464' '75 17 17 23 13959168'; do
  read -r L TG TU TD STRIDE <<< "$spec"
  "$B/glmdump" "$T/build/bin/libggml-cuda.so" "$B/records/L$L.records" 8 "$TG" "$TU" "$TD" "$STRIDE" 32 7 "$B/cuda-dumps/L$L"
  "$B/g53ref_exact" records ~/k3bx/mufu "$B/records/L$L.records" 8 "$TG" "$TU" "$TD" "$B/cuda-dumps/L$L" "$B/cpu-dumps/L$L"
  python3 "$B/exact_test.py" compare "$B/cuda-dumps/L$L" "$B/cpu-dumps/L$L" || break
done
```
Also capture nsys and cuobjdump -sass of that library; verify SPEC-glm.md §7, especially IQ FFMA and SiLU lowering.
On laptop, install the verified rcp.bin/ex2p.bin/ex2n.bin under ~/glm53-work/mufu, then:
```
export W=$HOME/glm53-work G53X_EXACT=1 G53X_MUFU=/w/mufu
export MUFU_DIR=$W/mufu
./tools/glm53/build-sycl.sh
./tools/glm53/check.sh
python3 tools/glm53/serve_test.py
```
Unverified: random nonzero end-to-end CPU output (MUFU tables unavailable here), GLM CUDA SASS/dumps,
and SYCL compilation/execution (no compiler/container/iGPU access). Zero tests do not certify these.
MTP layer 78 Q2_K/Q3_K is documented but unsupported; serveidx rejects it for local fallback.
No exact Q4_K/Q5_K/Q6_K support was added. Scalar SYCL may be very slow; no speed claim.
Do not enable production GLM offload until all CUDA stage comparisons and laptop checks pass.

## Verification by the owner's session (2026-10-29)

- Spark, `glmdump` (production libggml-cuda.so, ~/llama-k3-sc) vs `g53ref_exact`, L3/L8/L75 x 32 cases: x, ids, g, u, h
  bit-identical from the first run; dn differed only in case 7 (sigma 1000) and only in NaN payload (CUDA 0x7fffffff,
  CPU 0x7fc00000, same NaN/inf positions). `ftz()` in `g53exact.h` now canonicalises NaN to 0x7fffffff; after that all
  four stages are bit-identical on all 96 cases (9.4 M values).
- Laptop iGPU (Meteor Lake Arc), `check-exact.sh`: g53x exact = CPU exact, g/u/h/dn, all three layers.
- Laptop iGPU, `cuda_serve_check.py` (g53x serveidx, the production path, fed the 96 CUDA dump inputs): no computed row
  differs from CUDA ("changed=0"). The only byte differences are cases 0, 2 and 3 (zeros, 1e-30, denormals), whose
  correct CUDA rows are all zero; serveidx sends an all-zero row as the all-NaN "missing" sentinel by design, so the
  engine recomputes those rows locally and the output is unchanged. Real decode inputs never produce them.
- `ref/ggml-common.h` (upstream 547208d) is vendored as `tools/glm53/ref/ggml-common.h`; the other `ref/` sources Sol
  read are not committed.
- Speed (laptop iGPU, `g53x data/UD-Q2_K_XL_L8 18 18 23 bench 1,4,8 50`): fast path (`G53X_EXACT=0`) p50 842 / 1694 /
  3615 us for k = 1 / 4 / 8; exact path 22067 / 96182 / 192670 us (~24 ms per expert, 0.7 GB/s): 26-53x slower, far
  outside the ~3 ms per-layer budget. The exact path is the reference that proves equivalence; serving needs a fast
  exact kernel (vectorised, CUDA's virtual-thread reduction order kept), which is the next task.
