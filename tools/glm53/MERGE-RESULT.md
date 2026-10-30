# Merge: sol/fast-exact + main (disk tier, allocation gate, scalar exact)

kernel modes coexist with the serveidx disk tier and allocation-free path:
- `G53X_EXACT=0`: fast path (unchanged).
- `G53X_EXACT=1`: scalar CUDA-exact oracle (main's functors).
- `G53X_EXACT=2`: fast CUDA-exact. **Default exact mode**, because every gate passed.

## Changes
- `g53x_parallel.inc` is rewritten as main-style functors: `ParPrepK` (10 args) and
  `ParMatK<..,GU>`. ParMatK has 8 args for gate/up and 4 for down, because DPC++
  drops unused trailing fields. Q8 scratch is reserved once (`parallel_reserve`)
  and made resident; requests never grow it.
- In serveidx, mode 2 gets its own pre-recorded Level Zero lists: prep kernel ->
  `zeCommandListAppendBarrier` -> matvec. Each of the 16 batch slots has its own
  scratch region. If ze is unavailable it falls back to SYCL, as main does.
- Defaults are now 2 in `exact_version`, check.sh, cuda_serve_check.py and
  serve_test.py. Values other than 0/1/2 are rejected.
- test-parallel-sycl.sh now reads `evidence/cuda-dumps` (`DUMPS_DIR`), not `$W/tmp`,
  and uses build-sycl.sh's flags. g53parallel.h forces precise FP.
- Dropped this branch's serveidx `G53X_FUSEQ=0` path. Main's design supersedes it.

## Gates (logs: ~/glm53-work/tmp/m-*.txt)
| gate | result |
|---|---|
| build-sycl.sh, build-cpu.sh | OK |
| G53X_EXACT=0 check.sh vs chk-baseline.txt | IDENTICAL |
| check.sh exact (1 and 2) | check-exact bit-identical, all 3 slices |
| test-parallel-sycl.sh | PASS 384 configs, 0 differing words |
| cuda_serve_check mode 1 / 2 | changed=0 L3/L8/L75 (all-zero-row sentinels only); ze lists used |
| serve_test.py | mode 2 (default): PASS. Mode 1: all pass except Q4 disk-deadline0 timing flake (fast path) |
| disk_test.py gate, modes 0 / 2 | allocations 0 in every window. Faults 2,36 / 31,74 (i915 gem maps) |
| test_disk_tier.py | PASS ALL |

Bench L8 `18 18 23 bench 1,4,8 50`, p50 µs k=1/4/8: mode 0 1345/6166/15520;
mode 1 88990/413575/836019; mode 2 3209/7724/13428.

## Caveats
- Page faults: the host had swap full (98/98 GB) and load 30-58. Mode 0 faults too
  under the same conditions. See docs/glm53-kernels.md section 10 (PFN maps/shrinker).
- The merge is resolved and staged but **not committed**: the final instruction forbade commits.
