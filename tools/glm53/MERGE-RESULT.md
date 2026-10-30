# RESULT: merge of origin/main (CUDA-exact) into copilot/g53x-ssd (disk tier)

## What changed (all conflicts resolved and staged; merge NOT committed, see below)
- `tools/glm53/g53x_exact.inc`: main's exact gate-up/down SYCL lambdas became functors `ExactGateUpK` /
  `ExactDownK` (nd_range<2>, WG 64, the MUFU tables and IQ grids passed as pointer args). Per-item math is unchanged.
  This lets serveidx pre-record exact launches in Level Zero lists, so the request path stays allocation-free in
  both modes. `exact_quantize()` was moved here. The file is wrapped in precise FP with contract off.
- `tools/glm53/g53exact.h`: the same precise-FP pragma scope (`#ifdef __clang__`).
- `tools/glm53/g53x.cpp`:
  - Both include sets are kept. The pickers return `{run, ze_kernel, rows-per-group, exact}`. In exact mode, types
    without an exact kernel fail closed.
  - The serveidx record loop builds args per kind (exact kinds get the table args).
  - The MUFU tables and grids are added to the resident USM list. `exact_init` runs at startup only.
- `build-sycl.sh`:
  - Dropped main's TU-wide `-ffp-contract=off -fno-fast-math`, which changed the fast-path output against the
    baseline.
  - Added `-Xclang -fexperimental-strict-floating-point` instead, so the scoped `#pragma float_control` takes
    effect on spir64.
  - Kept `-lze_loader`, bw and allocgate.so.
- `dsycl.sh`: both env pass-through sets, plus `G53X_MUFU` defaulting to `/w/mufu`.
- `serve_test.py`: the disk-tier cases plus main's `G53X_EXACT=0` for Q4; a docstring note.
- `docs/glm53-kernels.md` §10: a note on exact mode in serveidx. Added `tools/glm53/MERGE-RESULT.md` (gates).

## Tests (W=~/glm53-work; full logs in ~/glm53-work/tmp/*.txt)
- `./tools/glm53/build-sycl.sh` and `./tools/glm53/build-cpu.sh`: `[100%] Built target glmbench`, OK.
- `G53X_EXACT=0 ./tools/glm53/check.sh`: diff against `chk-baseline.txt` is IDENTICAL.
- `./tools/glm53/check.sh` (exact): rc 0; check-exact `g/u/h/dn: bit-identical` for every layer.
- `python3 tools/glm53/serve_test.py`, exact and `G53X_EXACT=0`: both end with
  `PASS: serveidx rows bit-identical to g53x check (...); disk tier: cold reads and LRU hits bit-identical, per-row NaN, deadline)`.
- `python3 tools/glm53/disk_test.py gate`, both modes: all ALLOCGATE window counters are 0.
  - exact: `page faults minor/major: 5 0`, 213 s window, then `GATE FAILED: page faults 5/0`.
  - fast: `1 0`, 16.9 s window, then `GATE FAILED: page faults 1/0`.
  - Every fault is on an `anon_inode:i915.gem` mapping, the known residual from DISK-RESULT.md.
- `TMPDIR=$HOME/glm53-work/tmp python3 tools/bitexact/cpu/test_disk_tier.py`: `PASS ALL CPU disk-tier regressions (pread)`.
- `python3 tools/glm53/cuda_serve_check.py`: NOT RUN. The run printed `copied records verified`, then
  `ERROR: No such file ... cuda-dumps/L3/x.bin`.

## Unsure / not done
- CUDA dumps are missing: there is no `sol-exact/evidence/` directory and `~/glm53-work-exact` is unreadable. The
  records were copied to `~/glm53-work/tmp/records`. Building proxy dumps through `g53ref_exact` was denied. Exactness
  vs CUDA is therefore only indirect: g53x exact = CPU exact (check-exact), and serveidx = g53x (serve_test).
- The disk gate "fails" on minor faults in i915 GEM rings only, never on allocations. Exact mode shows more faults
  because its window is 12x longer.
- Exact FP semantics rely on the scoped pragmas plus the strict-FP flag. I could not inspect device IR (no llvm-dis).
  It is verified empirically only.
- Commit: the task asks for a merge commit, but the closing rule says "Do not commit". I staged everything and left
  the merge open. This is a worktree whose git dir is outside the checkout, so the prepared message is in
  `MERGE_COMMIT_MSG.txt` (untracked). To finish: `git commit -F MERGE_COMMIT_MSG.txt && rm MERGE_COMMIT_MSG.txt`.
- `.copilot-agent/` (untracked, from the harness) was left alone. `~/glm53-work/tmp/cuda_serve_check` holds a
  339 MB partial store from the failed run and can be deleted.

## Owner check after the merge

`G53X_EXACT=1 python3 tools/glm53/cuda_serve_check.py` with the CUDA dumps: changed=0 on L3, L8 and L75 (only the documented all-zero-row sentinels differ).
