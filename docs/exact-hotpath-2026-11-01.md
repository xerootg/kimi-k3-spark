# Exact batching hot-path inventory, 2026-11-01

Scope: full replacements 0008.hot, 0030.v2hot, 0035.v7hot, 0036.v5hot, 0037.hot, 0040.hot.
0030 v3 and 0039 are absent from the supplied sandbox/evidence; v2 is audited without claiming it is v3.
Locations below are predecessor `hotpath-src/` file:line; C = `ggml/src/ggml-cuda/`, S = `src/`, T = `tools/server/`.

| Check / allocation, predecessor location | Where it runs | Before | After |
| --- | --- | --- | --- |
| MMVQ_EXACT_COLS, C/mmvq.cu:1394 (0008) | per host dispatch/type | cached bool load+branch | unchanged; full replacement |
| EXACT_COLS, C/ggml-cuda.cu:284 | per host dispatch/fusion matching | cached bool load+branch | retained |
| EXACT_STATS + EXACT_COLS, C/exact-stats.cuh:10 | per host dispatch | cached bool; argument evaluation still occurred | complete ID record guarded by cached bool; no disabled arguments |
| MMVQ_EXACT_COLS stats argument, C/ggml-cuda.cu:2853 | per host dispatch | getenv scan even with stats off | function-local static bool, accessed only inside enabled guard |
| DISABLE_FUSION_CLASSES, C/ggml-cuda.cu:291,295 | per host dispatch and graph optimization | cached string; per-call string construction, possible heap, substring scan | allocation-free initial CSV parse; enum mask load and bit test |
| Fusion operation classification, C/ggml-cuda.cu:4287 | per host dispatch/matcher | bounded operation-list scan + strings | same bounded operation selector; enum values, no strings |
| LLAMA_EXACT_ATTN_COLS, S/llama-graph.cpp:3915 | per graph build | getenv scan + strcmp | static const bool load+branch |
| LLAMA_DSA_SHORT_CTX_SKIP, S/models/glm-dsa.cpp:235 | per graph build/layer | getenv scan + strcmp | cached bool; diagnostic env pointer cached too |
| LLAMA_EXACT_ATTN_COLS, S/llama-context.cpp:2563 (0040) | graph budget/reservation | getenv scan + strcmp | static const bool load+branch |
| LLAMA_PREFILL_ISOLATE, T/server-context.cpp:3444 | per server scheduling iteration | static bool, first-use std::string | static bool, initial strcmp; load+branch thereafter |
| SPLIT_COLS, S/llama-moe-cache.cpp:2184; C/mmvq.cu:1989 | context creation; per host dispatch | context member; function-local static bool | retained; bool load+branch |
| PREFETCH_WIDE, S/llama-graph.cpp:2808 | per graph build | getenv scan + strcmp | static const bool load+branch; cache hook already cached |
| HOOK_STATS, S/llama-moe-cache.cpp:4375 | failed host fill plan only | getenv scan + strcmp | static const bool load+branch |
| CUBLAS_COMPUTE_TYPE, C/ggml-cuda.cu:2377 | per singleton cuBLAS host call | getenv, std::string, lowercase loop, comparisons | static enum override; load+branch; parse/string only at first use |
| Packed destinations, C/ggml-cuda.cu:2690,2720,4606,4645 | per host dispatch, enclosing column loop | four pool scratch sites; recursive axes may overlap lifetimes | two context-owned slots per stream; capacity check, pointer arithmetic; no warmed allocations |
| MMVQ q8 activation, C/mmvq.cu:1959 | per singleton host launch | pool allocation/free request each launch | retained context/stream buffer with any exact flag; no warmed pool requests |
| cuBLAS conversions/pointers, C/ggml-cuda.cu:2173,2174,2229,2311,2312 | per singleton host launch | five pool buffer handles/requests | five retained independent slots per used stream under EXACT_COLS; no warmed pool requests |
| Extents/top-k, S/llama-graph.h:448,449 | per graph build/layer | vector construction/growth | eight-element inline storage; bounded writes, no heap |
| KV slot-info copy, S/llama-kv-cache.cpp:2894 | per graph build/stream/layer | deep copy of strm/idxs vectors | reference source; empty local slot-info with only scalar bounds set |
| Split sync jobs, S/llama-moe-cache.cpp:4672 | per host fill fallback | temporary vector allocation | context member, reserved at creation; assign within capacity |
| Split union/predictions, S/llama-moe-cache.cpp:4463,5124 | per host cache hook | fixed arrays, reserved member vectors; insertion sort for <=48 fills | retained; no new per-call map/container construction |
| Generating/drafting, T/server-context.cpp:3464,3465 | per server scheduling iteration | local vectors grow each iteration | context vectors reserved for slots at initialization, then clear/reuse |
| Slot callbacks, T/server-context.cpp:3194,3206 | per server scheduling iteration | std::function conversion may heap-allocate large closures | templated synchronous callbacks, no type-erasure allocation |
| 0008 batchdump.cpp strings/vectors | standalone diagnostic executable | heap containers for files/tensors/results | retained; outside engine dispatch/token path |
| GGML constructors, graph-input make_unique, res->add_fused_node (S/llama-graph.cpp:1516) | per graph build | arena nodes, graph-input ownership, persistent fused-node vector growth | graph structure/ownership retained; fused vector clear/reuse keeps warmed capacity; extra vectors/copy above removed |
| Flag checks in GPU kernels | per kernel | none introduced by this stack | none; singleton column-loop bounds are host-side; existing kernels unchanged |

Pool read: supplied C/ggml-cuda.cu:597-711 (leg) searches 256 cached buffers, reuses fitting warmed buffers, calls cudaMalloc on a miss and cudaFree when full; OOM flush can discard warmed buffers.
C/ggml-cuda.cu:714-852 (VMM) reuses aligned high-water storage; growth uses cuMemCreate/map, with temporary access-descriptor vectors. Neither pool guarantees allocation-free behavior under changing capacity/cache pressure.
Packed scratch and q8 are preallocated during graph optimization when EXACT_COLS is enabled. cuBLAS slots warm on the first actual selected conversion; direct backend callers also warm lazily. No allocation at unchanged high-water capacity.
Separate buffers cover concurrent streams and two nested packed axes; q8 and cuBLAS slots cannot alias packed outputs. Old arenas remain until context destruction so captured pointers survive growth; capacities grow geometrically to bound retired storage. Q8 reservation excludes wide prefill matmuls.
This is a steady-state guarantee for audited exact paths, not a claim that initial warmup, graph creation, larger shapes, unrelated ID fallback pools or the whole server never allocate.
CUDA graph replay skips operator host dispatch entirely; capture/recapture and non-graph execution run it. User-reported fdg3 evidence was ~98% graph replay, so these dispatch checks are off the token path except recapture/direct execution. Graph reuse/launch bookkeeping still runs.
Read 1: all predecessor hunks, flag evaluation, heap containers, pool implementations and nested scratch. Read 2: replacement hunks, CSV exact matching, enum coverage, recursion/stream isolation, buffer lifetimes, flag defaults and stack composition.
Validation: `python3 tests/check-hotpath.py` and `python3 tests/check-hotpath-dispatch.py`; apply/reverse/reapply, seven full host translation units with -fsyntax-only, actual CUDA host fragments with spies, enabled/off flags, scheduler boundaries and 10000-call warmed allocators.
No CUDA SDK/GPU/model is available: full CUDA compilation, capture/replay execution, end-to-end identity, real memory usage and throughput remain unverified.
