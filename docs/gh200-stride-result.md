# GH200 K3 stride preparation (offline; no outage taken)
Audit source: ../sol-hooks/checks/0020c-pre (read-only, HEAD 57800ff).
Full audit/arithmetic: docs/gh200-stride-audit.md; M=src/llama-moe-cache.cpp below.
| Site (reference source) | Arithmetic / verdict |
|---|---|
| M:2899,2907,3488 | Reference excludes K3 types and CPU maps; repo 0004 already has types, 0004b enables CPU maps with CUDA banks. |
| M:2914,2930,3045,3056 | S=ceil(R/4096)*4096; bytes=S*(slots+1); tier slots=floor(HBM_share/S). |
| M:3115,3143,3144,3146,3167,3174 | Bank=B+part; nb2=S, nb3=S*(slots+1); views retain S; HBM prefix=n_hbm*S. |
| M:3335,3826,4032,4781 | Dummy/CPU-demand/lookahead/speculative targets B+slot*S, R bytes. |
| M:3967; M:609 | HBM swap exchanges R bytes at B+s*S / B+h*S before key exchange. |
| M:3745 | CPU GET_ROWS reads I32 IDs; hook fills bank, CUDA reads weights; no block division. |
| M:4851,4959 | Snapshot keys remapped; store rereads R bytes into B+s*S, independent of old padding. |
| M:5105,5125,5226 | Staging/cache copies R bytes at B+s*S; generic copy view uses slot*nb2. |
| M:2503,2504,2585,2652,2660 | Valve bounds ceil((B+first*S)/page)..floor((B+(last+1)*S)/page); budgets count S. |
| ggml/src/ggml-cuda/mmvq.cu:770,1209,1979,1981 | int64 sample*nb3+slot*nb2 byte offsets, including fused gate. |
| ggml/src/ggml-cuda/mmq.cu:193,195; mmq.cuh:1057,1153,1239 | All tile paths use int64 byte strides. |
| ggml/src/ggml-cuda/ggml-cuda.cu:2797 | Fallback slices expert at data+i02*nb02; float MMF rejects quantized types. |
| ggml/src/ggml-cuda/vecdotq.cuh:18,1111,1316; mmq-load-tiles.cuh:1085,1146; dequantize.cuh:274,366 | 2-byte loads/assembly; even blocks/parts/rows suffice. |
Allowlist: IQ1_S/IQ2_XXS already in repo 0004 (older reference/memo differ).
0004b-byte-stride-k3-types.patch instead removes CPU-map guards; keeps registration/alignment failures fatal.
All records are 4096 multiples; 50/66-byte blocks need only 2-byte alignment.
B=424000*1048576=444596224000; W=80*6451200+10*7827456+2*7139328=608649216.
Group budget=B*layers*R/W; slots=floor(budget/S)-1 (zero slot), 896 experts/layer.
| Layers | Record/new stride B | Old stride B | Host slots old -> new |
|---|---:|---:|---:|
| 80 IQ1_S x3 | 6451200 | 6451200 | 58436 -> 58436 |
| 10 IQ2_XXS gate/up | 7827456 | 10137600 | 5639 -> 7303 |
| 2 IQ2_XXS down | 7139328 | 10137600 | 1027 -> 1459 |
Total 65102 -> 67198 (+2096, +3.220%): capacity, not bandwidth or store repack.
82432 experts; uniform host residency 78.977% -> 81.519%, not measured hit rates.
Actual warm hits ~99%; absolute ceiling 100%. Exact current working-set count absent.
Old trace +3/+4% capacity misses 1.35/0.68% vs 1.79%; not today's prediction.
24 GiB HBM is additional; valve releases and 16 GiB prompt cache reduce effective residency.
Runner: four frozen-copy ST0/ST1/ST0b/ST1b legs; gold, restore, stride and hit-rate gates; pressure/concurrent probes; unchanged-image rollback/Ready.
Duration estimate ~32–40 min + unknown native build; build capped 20 min, experiment 75 min, rollback ~12 min (target <90).
Unknown: exact ac78af0e5d JSON absent; native current absolute path unverified (historical ~/llama-k3-la). Script refuses missing inputs before scale-down.
Host paths documented: /srv/kimi-k3/{models,store,snapshot}; configs/ and docs/gh200-dev.md absent; use copied Argo evidence.
Tests: bash tools/gh200/check.sh; host syntax/math/mock rollback only. No nvcc/hardware/full pinned-stack validation available locally.
Ready to run when authorized AND placeholders resolved: `NATIVE_BASE=/VERIFIED/llama-k3-la K3_REQUEST_JSON=/VERIFIED/k3-gold.json bash tools/gh200/win-stride.sh --authorized`
