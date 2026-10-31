# Audit items 1, 2, 7, 8 on the K3 engine (agent Q, 2026-10-29): what was built and measured, and why it is parked here

`q-audit-items-on-c4efdca.patch` is a diff **on top of patch 0003 as of c4efdca** (three files: `ggml-cuda/common.cuh`,
`ggml-cuda/mmvq.cu`, `src/llama-moe-cache.cpp`). It was built and run as the native tree `~/llama-k3-q` on the Spark. It is
not in the patch stack: the stack moved on the same evening (0004-0018, GLM-5.3 replacing K3 on the Spark) and item 1 was
adopted independently as patch 0009. Anything taken from here must be rebased onto the current stack and re-measured.

## Item 1: skip the zero record's ids inside `mul_mat_vec_q` (superseded by patch 0009)

Mechanism in this patch: the cache marks the zero record's channel on the split bank views (`op_params[8]` of the tensor =
channel + 1, `GGML_CUDA_MMID_ZERO_CHANNEL_PARAM`), `ggml_cuda_mul_mat_vec_q` reads it into the fusion args, and
`mul_mat_vec_q` / `mul_mat_vec_q_moe` write +0.0f for a block whose id selects it and return before any weight load
(the rows were +0.0f before too: a zero record's dot products are `fma(0, inner, +0)`). `LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO`,
default 1 here (patch 0009 keys the skip on the slot map's dummy marker and defaults to 0).

Measured on the Spark with K3, 2026-10-29 21:41Z-22:20Z, interleaved native legs, production env (tier on against the
production nuc2:19600, snapshot restored per leg), warm-up + 3 x 200-token coding runs, gold md5 (`b2c79009` / `e801484d`)
on every run:

| leg | engine | decode tok/s |
|---|---|---|
| A1 | c4efdca (`~/llama-k3-s2`) | 2.051 / 2.055 / 2.053 |
| B1 | this patch + `LLAMA_EXPERT_STORE_SPLIT_LATE=16` | **2.093 / 2.096 / 2.092** |
| A2 | c4efdca | 2.044 / 2.053 / 2.054 |
| B2 | this patch + `SPLIT_LATE=16` | lost: a second (root) llama-server was started on the box during the warm-up, `MemAvailable` went to 0 and the server died; the Spark was rebooted at 22:46Z for the GLM-5.3 server |

+2.0 % (2.094 against 2.052), against the audit's +5-6 % estimate. Stats line: lookups with more misses than late rows
6190 of 56794 -> 0, records read synchronously at the early lookup 0.16 -> 0.00 per layer, the early lookup's plan
0.165 -> 0.006 ms, wait at the real lookup 0.80 -> 0.97 ms (the excess reads moved from the early lookup to the fill),
11.3 -> 11.0 of 16 experts complete at the early lookup. With the skip, 16 late rows cost nothing, so `SPLIT_LATE=16` is
the setting that goes with it. nsys per-layer kernel times were not taken (no window left).

## Item 2: the speculative reader's bytes in flight (built, not measured)

`LLAMA_EXPERT_STORE_PREFETCH_CHUNK_KIB` (default: the pool's chunk) gives the side reader its own piece size, so
`depth x piece` bounds what is queued at the drive when a real fill lands. `LLAMA_EXPERT_STORE_PREFETCH_PAUSE=1` makes
a real fill's preempt also stop the pieces of the record being read (the pieces in flight drain; the rest waits for the
next post or for a fill that adopts the record: `spec_adopt` calls `spec->resume()` before its wait, so the deadlock the
audit warns about cannot happen; `preempt` became an atomic read by the submit loop). Planned env A/B: 256 KiB x depth
12-16, then PAUSE on top; watch `self-prefetch fill ... ms per post` and ms per record.

## Item 7: io_uring setup flags (built, not measured)

`LLAMA_EXPERT_STORE_URING_DEFER=1` sets up every ring of the store (the pool's thread-local rings, the speculative
reader's) with `IORING_SETUP_SINGLE_ISSUER | IORING_SETUP_DEFER_TASKRUN | IORING_SETUP_COOP_TASKRUN`; each ring is
created, submitted to and reaped by one thread, which the flags require. A setup that fails with the flags falls back to
none, logged once. The Spark's kernel is 6.17 and its liburing 2.5 has the flags.

## Item 8: predictions do not promote hits or earn second chances (replayed, built, not measured)

`LLAMA_EXPERT_STORE_PREFETCH_PROMOTE=0`: `moe_cache_lru::plan` gets `touch_hits`; a prediction's hits keep their place
and their second-chance counts, only its misses take slots, and the victim search steps over the prediction's hits as it
does in a prefill scan (otherwise a miss of the same plan could evict one of them).

Replay first (`tools/cachepolicy.py`, which now runs both ways; both k3-pfdump traces, 7980 slots, warm 150), hit rate on
the real lookups with the deployed second-chance-3 policy:

| | window 5 | window 8 |
|---|---|---|
| code, promoted (engine before) / left in place | 74.25 / 74.43 % | 77.39 / 77.64 % |
| prose, promoted / left in place | 77.61 / 77.78 % | 80.22 / 80.49 % |

About 1 % fewer misses: real, consistent on both traces, and below what a 3 x 200-token A/B resolves (leg noise ~0.3 %).
