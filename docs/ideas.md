# Owner's ideas and their sanity checks

Haley asked (2026-11-08) for every unconventional idea she raises to be written down here and checked for sanity
against measured numbers, not only intuition. Each entry gives the idea in her terms, a verdict, the reason with
numbers, and what happens next. Re-check an entry when the facts under it change (new links, new model, new
patch); the ledger (`docs/experiments.md`) holds the experiments themselves, this file holds the ideas.

Verdicts: **sane** (numbers support it, worth doing), **promising** (numbers point the right way, needs a
measurement), **partly** (right direction, smaller payoff than hoped), **no** (numbers rule it out),
**premise wrong** (rests on something that is not true), **rejected** (owner's call).

| date | idea | verdict | next |
|---|---|---|---|
| 2026-11-06 | NUCs make too many network round trips; we are using them wrong | sane | built (0049), off in production until multi-slot serving |
| 2026-11-06 | Distribute inference to the NUCs once the faster links exist | partly | re-check when the links land |
| 2026-11-08 | KV cache that resizes per conversation | sane, built (0052) | open decision: up to 1M each (refuse when full) or fixed shares; both are switches |
| 2026-11-08 | Keep each conversation's KV contiguous | sane | part of the per-conversation design |
| 2026-11-08 | SIMD can't help because the KV is not contiguous | premise wrong | none |
| 2026-11-08 | Put the KV on another machine on the network | no (active), sane (idle parking) | parking only |
| 2026-11-08 | Compress the KV (8-bit) | rejected | none |
| 2026-11-08 | Park idle conversations' KV on NVMe, reload on demand | sane | next step after 0052: resident slots now refuse when memory is full instead of parking |
| 2026-11-08 | Several 1M conversations on the GH200; leave KV in HBM and serve as many as fit | sane | follows the per-conversation build |
| 2026-11-08 | Keep the KV in Grace memory and let the GPU read it over the fast link | promising (built, correct on the Spark) | GH200 window, script ready (~1 h) |
| 2026-11-08 | The Spark as the general local failover for the GH200 | sane | Juno done in argo-things #159; public endpoint and workbench next |
| 2026-11-08 | Crack the 1M-token context ceiling | promising | test past 1M on the Spark after the KV-placement prototype |

## NUC round trips (2026-11-06)

Idea: the NUC tier makes too many round trips, and cutting them saves time.

Verdict: sane. Patch 0049 batches every stream's NUC rows into one request per layer, so a token costs 43/N round trips
instead of 43 (about 11 at four streams, about 5 at eight). Output is byte-exact with the tier on or off. The gain is
small (+2 % at four streams, +4 % at eight, nothing at one) because the Spark's own dense work, not the network,
limits the step. It needs multi-slot serving to engage at all.

## NUCs over faster links (2026-11-06)

Idea: once the links are faster, distribute more inference to the NUCs.

Verdict: partly. Faster links shrink the round-trip time, but each NUC reads memory at about a quarter of the Spark's
rate, so even a zero-latency link caps the single-stream gain at about +9 %. Multi-stream gains are larger because one
round trip then serves several streams. Re-measure when the links are in.

## Per-conversation KV resizing (2026-11-08)

Idea: let each conversation's KV grow and shrink on its own so several conversations can share the memory.

Verdict: sane. Patch 0045 (elastic KV) already does this for one conversation but switches off when `n_stream > 1`;
with it off, `--parallel 4` reserves the full cache up front. Extending it per stream is the prerequisite for
multi-slot serving (about +46 % aggregate at four streams on the Spark, plus the NUC tier). Open decision: can each
conversation reach 1M (shared pool, refuse when full) or does each get a fixed share?

Built 2026-11-08 as patch 0052 (window ek, `docs/experiments.md`): four slots of 1M each map ~440 MiB when idle and
grow per conversation; `LLAMA_ELASTIC_KV_REFUSE=1` (shared pool, a request that does not fit gets HTTP 503 and the
others continue) and `LLAMA_ELASTIC_KV_STREAM_MAX` (fixed share) are both switches, off by default. Catch found on
the way: batched decode across slots (+30 % at two, +49 % at four) changes each client's bytes depending on what else
is in the batch, so it breaks the reproducibility rule; the exact-batching knobs fix two streams but not four and eat
the gain. What holds the rule is serving the resident slots one request at a time (`LLAMA_SLOT_SERIAL=1`): same bytes
as production, and conversations stop evicting each other's KV.

## Contiguous KV per conversation (2026-11-08)

Verdict: sane, and the design already does it. Each conversation gets its own stream starting at cell 0. The virtual
addresses stay contiguous and never move; only the physical 2 MiB chunks behind them come and go. The alternative,
llama.cpp's unified cache, interleaves conversations, leaves holes when one ends, and places cells by arrival time,
which threatens byte reproducibility.

## SIMD with non-contiguous KV (2026-11-08)

Verdict: premise wrong. The KV is contiguous in the address space that kernels see; only the physical backing is in
2 MiB pieces, and aligned vector loads never straddle a piece. The GPU's form of SIMD (warps reading 16-byte
vectors side by side) is already fully used.

## KV on another machine (2026-11-08)

Idea: keep the KV on a different machine on the network.

Verdict: no for conversations that are generating, sane for parking idle ones. Each token's attention needs that
conversation's selected KV rows (about 23 MB at any depth, as 22,000 scattered 1 KiB reads) plus the indexer's scan
(up to about 0.7 GB at 1M). At 1-10 Gb/s, and with a round trip per scattered read, that is far slower than the
token itself. Moving the attention computation to the KV's machine instead adds a round trip per attention layer,
and the NUCs read memory at a quarter of the Spark's rate.

## 8-bit KV (2026-11-08)

Verdict: rejected by the owner. Each entry is rounded once, so the error does not grow inside the cache, but every
token's attention sees slightly different numbers. Sooner or later a near-tie flips, the conversation diverges, and
over a long agentic run that can change tool calls. It also breaks byte-identical comparison with the reference.

## Parking idle KV on NVMe (2026-11-08)

Idea: RAM is finite and M.2 is fast, so move the KV of conversations that are not generating to local storage and
pay the reload when they come back.

Verdict: sane. A 1M conversation on Flash is about 19 GB (11.0 KiB per token of KV plus 8.2 KiB per token of indexer
cache, from the Spark's startup log). Reloading it from NVMe takes seconds; re-reading it from scratch takes the
Spark about an hour at ~320 tok/s or less. A raw copy restores identical bytes, so reproducibility holds. Granularity is a
whole conversation. The catch: the Spark's NVMe also serves expert-cache misses, so reloads should be throttled or
scheduled between tokens. The GH200 already parks in host memory (`--cache-ram 98304`, about five 1M conversations).

## Several 1M conversations on the GH200 (2026-11-08)

Idea: leave the KV in HBM and serve as many conversations as HBM holds.

Verdict: sane; it is the per-conversation build applied to the GH200 (stashed item "0045 make_room for the HBM
domain"). Each live conversation takes only what it uses; a full 1M one costs about 19 GB, taken from the HBM expert
tier, whose size mattered little in the 2026-11-02 context sweep (40 vs 16 GiB, same speed). Each conversation gets
somewhat slower when several generate at once; aggregate throughput rises. Not yet measured with Flash.

## KV in Grace memory over NVLink-C2C (2026-11-08)

Idea: the GH200's GPU can read Grace memory directly, so keep the KV there instead of in HBM.

Verdict: promising. Checked on paper 2026-11-08 (ledger entry of that date):

- Flash attention is already sparse. Of 46 blocks, 11 use attention (one 512-wide latent per token, 1 KiB); the rest
  are linear KDA layers with a fixed-size state. Each attention layer's indexer picks 2048 tokens (512 pools of 4,
  plus the 3-token tail), and the CUDA flash-attention kernel compacts the mask and gathers only those rows once the
  context passes about 4,100 tokens. So the KV read per token is about 23 MB at any depth, not the whole cache.
- The indexer scan does grow with depth: one 128-wide pooled key per 4 tokens per layer, about 0.7 GB per token at
  1M (the code copies the pooled keys before scoring, which may add to that; measure).
- NVLink-C2C is 450 GB/s each way by specification (not measured here). At 1M that is roughly 2-5 ms per token over
  the link against a ~20 ms token, and much less at shorter depth. The scattered 1 KiB gathers may be latency-bound
  rather than bandwidth-bound; that is the main unknown.
- Payoff: about 19 GB of HBM freed per 1M conversation, and room for tens of 1M conversations in Grace memory with no
  reload step.

Built and checked on the Spark 2026-11-08 (patch 0051, `LLAMA_KV_HOST=1`; ledger entry "KV in host memory, patch
0051"). Both caches sit in registered, huge-page host memory, and attention and the indexer still run on the GPU.
Output is byte-identical to device placement at 16k, 100k and 200k and on a needle test, and decode speed is the
same. The Spark's host and device memory are the same chips, so this proves correctness and prices the host-access
path, not the link:

- Scattered attention gathers cost the same through the host path.
- The indexer's streaming scan reads ~1.4x slower.
- Huge pages are essential: without them the GPU's scattered reads are 2-6x slower.

On the Spark the KV ops are 2-5 % of a token up to 100k. The indexer's part grows with depth: ~2.4 ms at 100k,
~20 ms at 1M by extrapolation.

GH200 estimate: at 500k the indexer reads ~350 MB per token. That is about 1.2 ms over C2C against 0.2 ms from HBM,
plus whatever latency the scattered reads add. So expect a few percent slower decode at 500k-1M, for 9.6-19 GB of
HBM freed per conversation.

Next: the GH200 window (`docs/evidence/glm-windows-2026-10-30/kv-placement/gh200-kvhost.sh`). Build before the
downtime, then about an hour down: microbenchmark, HBM vs Grace decode at 16k / 100k / 500k with md5s, and per-op
profiles at 100k / 500k.

## The Spark as the GH200's failover (2026-11-08)

Idea: when the GH200 is down, its clients should land on the Spark, which serves the same model, before going to a
hosted model.

Verdict: sane. Both servers run GLM-5.3-Flash (Q4 on the GH200, Q2 on the Spark) with the same API key and a 1M
context. The Spark is slower (decode ~17 against ~50 tok/s, prompts ~320 against ~1,400 tok/s), so a cold 120k-token
session takes minutes to load there, but conversations stay on owned hardware. Output differs between the two
quantizations; each server stays reproducible on its own. Juno's chain (gh200, then spark, then Sonnet) is argo-things
#159; her utility calls cannot fail over until OpenClaw 2026.11.1. Still to do: the public endpoint
(kimi-gh200-api) and the RE workbench's LLM profile, and the hef-posters vision calls, which have no fallback.

## Past the 1M-token ceiling (2026-11-08)

Idea: serve conversations longer than the model's 1M-token training context.

Verdict: promising, untested. Nothing in Flash's architecture fixes a maximum length: the 11 attention layers use no
positional encoding at all (`rope.dimension_count = 0`, "nope MLA"), the 35 KDA layers carry a fixed-size state, and
the indexer's only position term (`indexer_kpool_ape`) is a position inside a pool of 4 tokens, not in the sequence.
llama.cpp accepts a larger `--ctx-size` with a warning (`n_ctx_seq > n_ctx_train -- possible training context
overflow`). The DSA top-2048 selection also caps how many tokens each attention step mixes, which may help it hold up
past the trained length. Risks, in order:

- Quality: the model never saw a sequence this long, so retrieval and coherence past 1M must be measured, not
  assumed (needle-in-a-haystack at 1.2M, 1.5M and 2M, plus a long coding transcript).
- Integer overflow: a 2M-cell K tensor per layer is 2 GiB, past signed 32-bit byte offsets; kernels that index with
  `int` would break past about 2.1M cells (this bit the expert store once, fixed in the 32-bit offset entry).
- Memory and time: a 2M conversation is about 38 GB of KV plus indexer cache. Loading 2M tokens takes the Spark about
  two hours at ~320 tok/s or less; on the GH200 it does not fit in HBM next to the experts, so it depends on the
  Grace-memory placement idea above.

Next: after the KV-placement prototype releases the Spark, run 1.2M / 1.5M / 2M contexts there with retrieval checks
and an offset audit of the attention, indexer and KV-copy kernels.

