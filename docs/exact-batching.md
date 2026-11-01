# Exact batching on GLM-5.3 / Spark

Status: **fd9 pending; NOT FOR ROLLOUT**. Install the seven lexical exact
patches after production 0001..0026, rebuilding the image from this stack.
0030 is N<=8 v3 with 0032 folded; 0033 is 0035 v5 with the joint fix folded;
0034 is 0036 v4; retain 0031/0035; append 0036 rows and 0037 guards.
The additive joint fix is archived and must not be applied.

Use --parallel 2, total --ctx-size 65536 (32768 per slot), production fusion,
existing LLAMA_EXPERT_STORE_SPLIT=1 and these six deployment knobs:

```sh
GGML_CUDA_EXACT_COLS=1
GGML_CUDA_MMVQ_EXACT_COLS=1
LLAMA_EXACT_ATTN_COLS=1
GGML_CUDA_ARGSORT_EXACT_ROWS=1
LLAMA_EXPERT_STORE_SPLIT_COLS=1
LLAMA_PREFILL_ISOLATE=1
```

The ARGSORT knob is retained for manifest compatibility but v4 has no handler
for it; production argsort stays in place. No old global prefill sort patch.
32768 prompt tokens do not fit a 32768 slot: reserve generation headroom
(e.g. <=32640 prompt tokens for 128 output tokens) and enforce client limits.

Proven: offline lexical application and host syntax are reproducible from the
reconstructed base; commit identity remains unverified. Raw c-tree/production
postimage differences are in verification/stack2/postimages.diff and its index.
The supplied fag GPU probe reports 34 configurations, 12 differing outputs,
identical valid membership and identical repeats: effective KV geometry changes
attention output. Changing padding TOP_K IDs alone is not a membership change.
This probe does not establish full model gold or v4 throughput.

V4 views each physical stream at its own padded extent for score/mask/TOP_K/FA;
K=min(extent,2048), not valid-key count. Equality to standalone depends on
matching occupancy, padding, capacity, holes, shifts and cache history. Reuse
checks the whole extent vector. 0039 restricts this branch to N>=2 and rejects
unified-parallel exact KV in the constructor. N1 attention branch preservation
does not imply whole-stack flagged N1 inertness: row fixes and generic column
splits can change prefill arithmetic. No-flags source reasoning is not GPU gold.

Review 16/17 gates, all required before rollout:
- Fresh native SM121/CUDA build; host syntax excludes CUDA and normally server.
- Full fixed external standard gold at N1/no flags and N1/all flags; independent
  no-flags long reference. Warm self-reference is not external gold.
- Per-client complete output bytes and prefill/decode checks against matched
  standalone cold/warm histories, short+short, short+long, long+long, N4,
  simultaneous/staggered arrivals, ties, multi-ubatch prompts and resumed decode.
- Log actual N-column overlap; concurrent submissions alone do not prove it.
  Require N3/5..8 evidence before promoting those widths.
- fd9 runner must preflight its comparator, install/verify 0030v3 and guards,
  fail-stop on any leg/comparison/timing failure, and enforce completion.
  WIN2FD9-DONE alone is not a PASS. Review17's 0040/0041 runner fixes are
  referenced evidence, not supplied/executed here.
- Repeat/interleave long-prefill controls and enforce <=1.05x time; every
  client's mismatch fails regardless of aggregate speed. Final output equality
  alone does not prove intermediate prefill tensors equal.
- Aggregate wall throughput >= matched standalone workload, accepted per-client
  latency/stalls, memory/valve/read health, cancellation and bounded arrivals.
  Measure allocator growth at warmup/replay/shape transitions before claiming
  no per-token allocations. Numerical/cache failures fail the round.

Known costs: flagged prefill is about 2x until fixed, so the 1.05x gate remains
open. Row fixes affect N1 prefill. Singleton launches, packed copies, conversion
scratch, score/FA launches and concats reduce batching gains. Prompt isolation
serializes prompts and pauses decoders; unbounded arrivals can starve them.
V4 padded-extent boundary changes rebuild/reallocate graphs and may recapture
or update CUDA graphs. Ordinary append changes each extent once per padding
interval; staggered streams can cause up to N events per interval. Measure
rebuild/capture counts, wall throughput and per-client stalls. Host-bank graph
compatibility may independently disable capture. No fd9 GPU gold, timing or
throughput has been run here. The manifest v2 is review material only.
