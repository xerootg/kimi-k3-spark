Patch 0004b is folded into bstride.patch, a full replacement for 0004 against llama/ HEAD (base).
The same changes remain applied as uncommitted edits in llama/; no commits or network used.

Changed:
- Guard now audits 19 types: IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL,
  Q2_K/Q3_K/Q4_K/Q5_K/Q6_K, Q4_0/1, Q5_0/1, Q8_0.
- Shared inline helpers in src/llama-moe-cache.h let the host test exercise the actual guard.
- Require part_off, nb[1], and block size divisible by the type's 2- or 4-byte load alignment.
  A 4096-byte slot stride preserves it. Explicit block-size checks reject odd blocks;
  even 50/66/74/98-byte blocks support 2-byte loads, but not fixed-field 4-byte loads.
- IQ1_M/IQ4_XS/Q2_K/Q4_K/Q5_K/Q4_1/Q5_1 need 4 bytes; the other audited types need 2.
- Per-group log lists layer IDs, all part types, stride, byte/padded mode, and reason.
  Rejected alignment logs include part, type, required alignment, block size, offset and row stride.
- CUDA GPU-slot/host-bank gating, opt-in environment variable, fallback padding and aborts remain.
- Original MMVQ byte pointer changes are included in the replacement; arithmetic is unchanged.
- validation/alignment-audit.md records field offsets/load widths, wider-load audit and fallback paths.
- validation/store-capacity.py now handles all 19 audited types and an optional MiB budget.

K3 capacity (gate/up/down; one zero slot/group, proportional byte budgets, no HBM addition):
| Group | Layers | Old stride B | New stride B | Padding removed B/slot | 424000 MiB slots | 52000 MiB slots |
|---|---:|---:|---:|---:|---:|---:|
| IQ1_S/IQ1_S/IQ1_S | 80 | 6451200 | 6451200 | 0 | 58436 -> 58436 | 7165 -> 7165 |
| IQ2_XXS/IQ2_XXS/IQ1_S | 10 | 10137600 | 7827456 | 2310144 (22.788%) | 5639 -> 7303 | 690 -> 894 |
| IQ1_S/IQ1_S/IQ2_XXS | 2 | 10137600 | 7139328 | 2998272 (29.576%) | 1027 -> 1459 | 125 -> 178 |
Totals: GH200 65102 -> 67198 (+2096, 3.220%); Spark 7980 -> 8237 (+257, 3.221%).
All three groups pass the new alignment guard with CUDA GPU slots enabled.
Reproduce: python3 validation/k3-capacity.py (also regenerates validation/store-parts.tsv).

Test command: sh validation/check-byte-stride.sh
Last output lines:
PASS: 19 CUDA type layouts, exhaustive field alignment, odd-block rejection, 81 store parts
PASS: block layouts, load alignment, ordinary pointer equivalence, >32-bit offsets, zero-slot capacity (6495..6496)
PASS: cache C++ syntax, Python syntax, diff whitespace
Replacement check: git -C llama apply --reverse --check ../bstride.patch exited 0.
Forward patch dry-run against the three pristine base files also passed.

Limits/uncertainty:
- No nvcc available; CUDA compilation, decode hashes and GPU performance were not tested here.
- K3 capture truncates inside layer 27: tests use 26 complete records and their true rows/offsets.
  Capacity uses the plan's 80/10/2 counts, reconciled exactly to index total_bytes=545349709824.
  Later layer IDs remain unverified; counts assume all 92 layers cached, no valve/remote reductions.
- No full GLM index supplied. Tests use prior RESULT-0004.md offsets 0/3637248/7274496,
  representative single-block rows, and all audited types crossed with both stores' known offsets.
  Actual GLM rows, IQ4 group offsets and Q5 store layouts are checked at runtime, not assumed here.
