Patch 0007: miss-only prefill staging (uncommitted)

Changes: llama/src/llama-moe-cache.cpp:4910 copies each used resident record from its host-bank slot; only missing records enter the existing I/O pool. :4928 counts actual submitted store bytes, so the existing fill_late stats line now reports reduced prefill bytes without changing decode counters.
:4931 warms the cache from staging after successful reads, using the existing LRU plan. No second disk read. Hits are copied before any eviction; the existing backend synchronization and LRU lock protect staging reuse and slot lifetime. prediction_discard/la_finish retain their existing speculative-I/O barriers.
Warming defaults on; LLAMA_EXPERT_STORE_PREFILL_WARM=0 disables it. Too few available slots (including parked/busy slots) skips warming, but still serves the prompt. No extra record allocation.
Default PREFILL_PROMOTE=0 uses existing scan resistance: empty caches fill; once full, hits retain recency and new records go to the eviction head. A saturated scan churns at most one layer's selected records per group (<=256 slots/group), NOT zero decode evictions. Cold prefill leaves a populated, scan-resistant cache, not necessarily the final prompt's optimal decode set.
PREFILL_PROMOTE=1 explicitly opts into full LRU promotion and existing SECOND_CHANCE=N; it can replace the entire decode working set. Default full-cache scans intentionally bypass second chances, as existing bank prefill does. Disable warming to preserve cache contents exactly.

Arithmetic: decode functions, graph construction, routing, kernels, sums and upload code are untouched. For fixed batching, each selected expert has the same record bytes at staging[x*record_size]; gate/up/down uploads use the original offsets, lengths and expert order, including zero padding. Cache slot changes only relocate identical weight bytes. No arithmetic changes are intended; GPU/end-to-end identity remains unmeasured here.
The 76% saving requires 76% byte-weighted hits among PREFILL's selected records. Decode hit rate alone does not imply that: near-all-expert prefill sees cache coverage, roughly 80000 MiB / 238 GB (~35%, before layout costs), not decode's frequency-weighted rate. Report actual bytes, do not promise 76%.

Ubatch analysis (existing limits, no defaults changed):
llama/src/llama-moe-cache.cpp:3486 allocates max_record*256 ONCE, shared across layers and tensor parts, independent of token count. Evidence reports 3984 MiB (max_record=16318464 B, including unused MTP); 512/1024/2048 all require the same staging. Warming copies directly into the existing 80000 MiB cache.
:3487 clamps MAX_BATCH>32 only when bank_on_gpu is false; :4977 also requires enough slots for min(tokens*experts_used,256). This controls bank vs staging, NOT --ubatch-size. Keep MAX_BATCH=32 for initial measurements; 1024/2048 still use staging.
"Host bank" is RAM holding packed records. "GPU bank" here is that SAME host allocation wrapped/registered as CUDA-accessible memory (:3038), not an 80000 MiB cudaMalloc. GB10 can read it directly. CUDA supports the buffer; a CPU-only bank would be copied/offloaded as large strided views, hence the clamp.
CUDA paths: llama/ggml/src/ggml-cuda/ggml-cuda.cu:6644 accepts registered banks; :2661 dispatches MUL_MAT_ID; llama/ggml/src/ggml-cuda/mmq.cu:193 passes expert stride in bytes. Registered banks CAN serve batched MUL_MAT_ID in place with GPU slots enabled, sufficient slots and MAX_BATCH raised; benchmark separately because this changes graph/kernel shapes and avoids staging uploads.
The practical >512 limits are shared RAM headroom, scheduler compute/attention buffers, MMQ scratch and throughput, not staging. MMQ scratch (:320) is B*8*pad(D)*q8_bytes_per_value plus O(B*8) maps; F32 routed activations cost 4*B*8*F each. Fallback sorted GEMM (ggml-cuda.cu:2708) adds 4*B*8*(D+F) bytes; actual D/F and dispatch must come from model/logs.
Planning estimate ONLY: evidence's 2184 MiB compute at B=512 -> 4368 at1024 / 8736 at2048 if linear; extra +2184/+6552 MiB. Attention can scale differently. At ctx32k: cache80000 + staging3984 + KV2976 + spine~14438 + compute4368/8736 + auxiliary112 = ~105878/~110246 MiB, excluding OS, allocator pools and other allocations. ctx64k adds ~2976 MiB KV; these are not measured peak-RSS bounds.
Propose --batch-size 2048 --ubatch-size 1024 first, ctx65536 for 45k work; try2048 only after checking MemAvailable, compute-buffer logs, peak RSS and valve events. For 30k, ubatches fall from ~59 to30/15; never infer a speedup solely from this ratio. Different ubatching or bank/MMQ dispatch may change floating-point rounding and generated output; compare hashes against each configuration's baseline.

Local validation (no Spark/model/CUDA available; no network used):
```sh
c++ -std=c++17 -fsyntax-only -pthread -Illama/include -Illama/src -Illama/ggml/include -Illama/ggml/src llama/src/llama-moe-cache.cpp
python3 tests/test-prefill-staging.py
python3 -m py_compile tests/measure-prefill.py
git -C llama diff --check
```
Syntax check: exit 0, no diagnostics. Diff check: exit 0, no diagnostics. Python compile: exit 0.
Runtime test compiles the production LRU and fill_prefill body with mock I/O/backend under ASan+UBSan (leak detection disabled because sandbox ptrace prevents LSan). It does not emulate GPU execution or speculative concurrency.
Last output: `PASS: miss-only bytes, three parts, padded slots, warm on/off, repeat hits, bounded scan eviction, parked slots`

Owner-run Spark commands, from this delivered directory (commands only; not run here):
```sh
TASK="$PWD"
cd ~/llama-k3-bs
git apply "$TASK/0007-prefill.patch"
CUDACXX=/usr/local/cuda/bin/nvcc cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 12 --target llama-server
```
Apply the delivered diff to that native tree before building; retain its existing CUDA architecture/build options. No container build. Stop other model instances. Use the owner's current v2 store if its path differs from the evidence's v1 path below; use the SAME file for baseline/patched comparisons.
Terminal A (repeat for UB=512,1024,2048; keep this log path consistent with Terminal B):
```sh
UB=1024
M=/srv/glm-5.3/models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf
E=/srv/glm-5.3/store/glm53-q2-v1.k3xs
/usr/bin/time -v env LLAMA_EXPERT_STORE="$E" LLAMA_EXPERT_STORE_SELF_PREFETCH=1 LLAMA_EXPERT_STORE_SPLIT=1 LLAMA_EXPERT_STORE_SECOND_CHANCE=3 LLAMA_EXPERT_STORE_RESERVE_MIB=4096 LLAMA_EXPERT_STORE_MAX_BATCH=32 LLAMA_EXPERT_STORE_PREFILL_PROMOTE=0 LLAMA_EXPERT_STORE_PREFILL_WARM=1 ~/llama-k3-bs/build/bin/llama-server -m "$M" --host 127.0.0.1 --port 18080 --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size "$UB" --n-gpu-layers 999 --n-cpu-moe 999 --moe-cache-mib 80000 --cache-ram 0 --threads 8 --threads-batch 20 --flash-attn on --no-warmup --fit off --load-mode mmap -lv 4 > "$TASK/spark-$UB.log" 2>&1
```
Terminal B, after server readiness (TASK must point to the delivered directory):
```sh
UB=1024
for N in 2048 8192 30000; do python3 "$TASK/tests/measure-prefill.py" --tokens "$N" --server-log "$TASK/spark-$UB.log" --output "$TASK/run-$UB-$N.json"; done
python3 "$TASK/tests/measure-prefill.py" --standard-request "$TASK/standard-128.json" --server-log "$TASK/spark-$UB.log" --output "$TASK/standard-$UB.json"
rg 'fill_late:.*GB for prefill|compute buffer|memory valve|Maximum resident' "$TASK/spark-$UB.log" | tail -12
```
Generator produces deterministic code-like text, tokenizes locally and slices to exactly 2048/8192/30000 token IDs; each completion generates128 tokens (ignore_eos), reports prompt timing/tok/s, full content MD5 and rounded prefill GB delta from fill_late. Counters are logical store bytes, not physical NVMe counters; rounding resolution is 0.1 GB. Keep raw JSON/logs.
For matched comparisons, restart per prompt size with identical snapshot/cache state and repeat each request twice; the loop above intentionally also measures a continuous warming workbench session. Run baseline/patched at512 first; then1024/2048. Compare WARM=0/1 to isolate warming, and record memory-valve releases and decode tok/s/hit rates after prefill to detect working-set loss.
Unverified: no standard prompt/request fixture exists here. Supply the ORIGINAL standard128 request as standard-128.json, including its original sampler settings; helper asserts MD5 prefix9449ae5738. Generated long prompts have independent baselines, not that hash. No Spark throughput, NVMe volume, peak RAM or end-to-end decode MD5 was measured in this sandbox.
