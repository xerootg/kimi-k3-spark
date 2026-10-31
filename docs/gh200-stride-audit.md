# K3 byte-stride audit, 2026-10-30

Offline only. Source references below are relative to the read-only engine
`../sol-hooks/checks/0020c-pre`, HEAD `57800ff8992b0787bd5f3872bb518f9c196c44a2`.
They deliberately refer to that supplied source, not line numbers in a candidate.
The source includes more experiments than this repository's lexical patch stack.
`configs/` and `docs/gh200-dev.md` are absent; copied Argo evidence is authoritative
for the reviewed deployment. Evidence files were not modified.

The memo/plan and supplied source describe an older 0004: IQ2_XS/IQ3_XXS/IQ4_XS
only. This repository's 0004 already audits 19 types, including IQ1_S/IQ2_XXS
with 2-byte alignment. Adding them again would be redundant. Both versions
reject CPU slot maps. `0004b-byte-stride-k3-types.patch` therefore enables the
CPU-map/GPU-bank combination, preserving opt-in, type/row/offset alignment,
CUDA registry checks and fatal bank-registration failure. CPU expert matmul
(`LLAMA_EXPERT_STORE_GPU=0`) remains ineligible. Apply 0004b after 0004, before
0005; default byte-stride=0 behavior stays padded.

Let R be record bytes, S the RAM slot stride, P the part offset, B the registered
region base, s the slot ID, b the quant block size. CUDA reads
`B + P + int64(s)*S + row*nb[1] + block*b`; never `floor(S/b)*b`.
All three K3 R values are multiples of 4096. IQ1_S b=50 and IQ2_XXS b=66
are even; gate/up rows are 700/924 B, down rows 600/792 B. Part offsets:
all-IQ1_S 0/2150400/4300800; mixed gate/up 0/2838528/5677056;
mixed down 0/2150400/4300800. Every offset/row/block preserves 2-byte loads.

| Site in supplied source | Arithmetic / finding |
|---|---|
| src/llama-moe-cache.cpp:2899,2907,3488 | Older type/map guards prevent byte-stride K3; current 0004 extends types, 0004b removes map restriction. |
| src/llama-moe-cache.cpp:2914,2930 | S=ceil(R/4096)*4096, fallback increments 4096 until S%b=0 for all parts; allocation S*(n_slots+1). |
| src/llama-moe-cache.cpp:3045,3056 | Proportional host+HBM byte budget; n_hbm=min(n_slots,floor(hbm_budget/S)). |
| src/llama-moe-cache.cpp:3115; :1639 | Tier prefix n_hbm*S; allocator rounds prefix up to host pages. At most one boundary page extends into a host slot, affecting placement, not addressing. |
| src/llama-moe-cache.cpp:3143,3144,3146 | nb[2]=S; nb[3]=S*(n_slots+1); bank base B+P. |
| src/llama-moe-cache.cpp:3167,3174 | Cached and split views retain bank nb[1]/S; no quant-block conversion. |
| src/llama-moe-cache.cpp:3335 | Dummy slot B+n_slots*S, zero R bytes, within extra slot. |
| src/llama-moe-cache.cpp:3745 | CPU graph GET_ROWS consumes I32 selection/map IDs, fills bank before downstream CUDA matmul; recursive LRU lock and protect_layer retain valve ordering. No CPU read of quant weights introduced. |
| src/llama-moe-cache.cpp:3826 | Prefill lookahead fills B+size_t(slot)*S, R bytes. |
| src/llama-moe-cache.cpp:4032 | Demand decode/prefill bank fill uses same expression; store offset is independent file record stride. |
| src/llama-moe-cache.cpp:3967; :609 | HBM swap exchanges R bytes at B+s*S and B+h*S; pool chunks swaps in bytes, then lru.exchange publishes keys. |
| src/llama-moe-cache.cpp:4781 | Speculative read target B+slot*S, R bytes; CPU production has predictive prefetch off. |
| src/llama-moe-cache.cpp:4851,4959 | Snapshot verifies store identity/layout, reassigns keys to available slots, rereads R bytes at B+s*S. Snapshot contains keys/use counts, not old byte offsets or padded payload. Capacity change may truncate entries; require restored log. |
| src/llama-moe-cache.cpp:5105,5125 | Staging hits copy R from B+slot*S; staging-to-cache warm copies R back. Staging addresses expert*R, not slot*S. |
| src/llama-moe-cache.cpp:5226 | Generic cache-copy view starts slots[e]*bank.nb[2], keeps byte stride. |
| src/llama-moe-cache.cpp:2503,2504 | Valve run [first,last]: lo=B+first*S, hi=B+(last+1)*S; a=ceil(lo/page)*page, b=floor(hi/page)*page. Drops/repopulates only [a,b), never neighbour edges on 64 KiB pages. |
| src/llama-moe-cache.cpp:2585,2652,2660 | Valve budget/re-arm accounting in S-byte slots, not blocks; released slots count parked/full for scan resistance. |
| ggml/src/ggml-cuda/mmvq.cu:1979,1981 | s02=nb[2], s03=nb[3], both int64 byte strides. |
| ggml/src/ggml-cuda/mmvq.cu:770,1209 | Ordinary and MoE MMVQ: char pointer += sample*s03+channel*s02 (MoE: channel only); fused gate receives identical byte offset; rows still whole blocks. |
| ggml/src/ggml-cuda/mmq.cu:193,195; mmq.cuh:1057,1153,1239 | MMQ passes byte strides; all tile paths add sample*S3+int64(expert)*S2 to char pointer. Compacted IDs are slot IDs. |
| ggml/src/ggml-cuda/ggml-cuda.cu:2797 | Sync fallback selects a single expert at char(src0.data)+i02*nb02 before generic matmul; no truncation. mmf.cu:136 rejects quantized types, so float element-stride path is not a K3 reader. |
| ggml/src/ggml-cuda/vecdotq.cuh:18,1111,1316 | get_int_b2 assembles two uint16 loads; IQ2_XXS qs and IQ1_S qs/qh need only 2-byte alignment, despite alternating block bases mod4. |
| ggml/src/ggml-cuda/mmq-load-tiles.cuh:1085,1146; dequantize.cuh:274,366 | MMQ uses get_int_b2/uint16 fields; dequantization uses uint16/byte fields. Wider grid/table and activation reads are independent of the expert-bank base. |

Conclusion: these address expressions are valid for all three K3 sizes with CPU
slot maps and registered CUDA bank readers; the guard change is necessary to
exercise them. Static audit is not evidence of sm_90 compilation, visibility,
restore equivalence, or pressure/concurrency correctness on GH200. Those remain
window gates. No bandwidth increase or measured speedup is claimed.

Capacity follows docs/gh200-glm-plan.md: three groups, 896 experts/layer,
80/10/2 layers. For H=424000*1048576=444596224000 B, weight W=608649216,
group budget H*(layers*R)/W, usable slots=floor(budget/S)-1 (one zero slot).

| Layer mix | R/new S | old S | host slots off/on |
|---|---:|---:|---:|
| 80 IQ1_S/IQ1_S/IQ1_S | 6451200 | 6451200 | 58436/58436 |
| 10 IQ2_XXS/IQ2_XXS/IQ1_S | 7827456 | 10137600 | 5639/7303 |
| 2 IQ1_S/IQ1_S/IQ2_XXS | 7139328 | 10137600 | 1027/1459 |

65102 -> 67198, +2096 slots (+3.220%). Total experts=82432: nominal host
residency 78.977% -> 81.519%; uniform routing would have those hit fractions,
not the actual ~99% warm hits. No exact current working-set cardinality is
documented. The old trace's +3%/+4% capacity gave 1.35%/0.68% misses against
1.79%, but cannot predict today's hit ceiling. Absolute hit ceiling is 100%,
only ~1 percentage point above measured warm traffic. HBM 24 GiB is additional;
the actual engine adds each group's HBM budget before rounding slots. Valve
releases and 16 GiB prompt cache alter effective residency. Savings are RAM
capacity, not disk repacking, NVMe bandwidth, or C2C bandwidth.

The runner is local to gh200-dev and has a 75-minute experiment alarm plus
~12 minutes rollback budget. Source/load/snapshot evidence: experiments.md:590
historical ~/llama-k3-la; Argo hostPaths give /srv/kimi-k3/{models,store,snapshot};
README evidence load ~1 min and experiments.md:449 restore 47.5 s. Twelve
800-token runs take ~12 min at 13.65 tok/s; five loads including rollback
~5 min plus four restores ~3.2 min conservatively. Gates add ~12–20 min:
plan ~32–40 min plus native build (20-minute timeout). No measured build time
is documented, so ~52–60 min is a planning budget, not a promised duration.
The previous simple four-leg Q6_K window took 25 min (evidence README, ninth
pass). Two 66k prompts already exceeded 55 min; this runner uses two synthetic
~16k prompts, 480-second per-request timeouts, not that unbounded stress test.
Production restoration can exceed the deadline if Kubernetes cannot recover.

Exact GH200 gold JSON is missing (only descriptions and hash); the Spark
128-token quicksort JSON is a different fixture. NATIVE_BASE and K3_REQUEST_JSON
must be supplied, and all preflight checks pass before scale-down. The live
Deployment must match captured image/args/env, replica=1 and node selector.
No image/manifest or production snapshot changes are made. A private frozen
snapshot is copied once after scale-down and then independently for every leg.
The current lexical patch stack is built from the pinned SHA with no fetching;
its later Spark changes are held fixed across all four legs. ST0 must reproduce
gold before any ST1 result can be used. Per-leg logs retain hit/valve data;
review releases, low points and re-arm before considering adoption. None of
these hardware gates has been executed here.
