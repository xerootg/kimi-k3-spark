Verdict: no P1 defect established; conditional source approval, with material backend/performance qualifications. No P1 fix patch warranted.
Scope: read-only c-tree and inputs untouched; cp -a to review13/candidate; applied/check-applied 0013, then 0013b, then 0013c. No network/commit/push/rm.
C below = review13/candidate/src/llama-moe-cache.cpp; detailed paths/table/estimates: review13/evidence.md.
(1) C:2831 initializes pool backend from STORE_URING (default true with liburing); no-liburing defaults false.
0013: predictor spec_uring=pool->use_uring; b: pool forced false before that assignment; c: C:3458 saves old pool value before :3459 false, :3630 uses saved value.
All stacks: explicit PREFETCH_URING atoi()!=0 overrides predictor default at C:3631-3633; without compiled liburing :3635 forces false.
Own fills are NOT restored to pre-b backend: pre-b eligible non-swap reads use uring when available/healthy; b AND c use pread. C:499 also excludes completion-tagged jobs from uring.
Thus predictor restoration is correct for active hedge; complete recovery of the i2 regression remains unproven. P2 HEDGE=1 without active remote init selects unsaved false predictor default.
(2) C:4296 enqueues before :4299 plan, but C:4541 already SKIPS ready resident slots. Only absent/spec-busy selected IDs are posted.
Predictions completing during plan and synchronous excess reads can make posted rows early/resident. No row cancellation; all count toward request posts and posted rows C:4552-4554.
C:4641 abandons whole request after pending rows settle; unsent requests may be skipped C:1106-1124. Sent rows all consume NUC compute and replies drain/drop.
Waste fraction B/P (unbound posted rows / posted rows) cannot be inferred from supplied request-post/win/hit counters; cache hit 62.2% is NOT wasted-post fraction.
Loose bounds from 7214 posts/384 tokens: <=300.583 posted rows/token at wire cap16; >=5.784 wins/token. Measure posted/bound/sent rows to quantify waste.
Eight client slots x16 rows =128-row maximum client pipeline/box; ordered replies and send-first pipelining allow wasted work to delay useful rows and refuse new posts. NUC internal queue depth unprovided.
(3) Byte guard C:4495-4511: pending entries come ONLY from split_late_used/merge_map; match posted IDs there. Early/resident rows have no pending entry.
Only winning pending rows copy bytes/rmerge position and dummy their own late-map row C:4609-4616; enqueue initializes all other remote merges to scratch C:4529.
(4) No-post layers bypass collector C:4409. No eligible outstanding bound row => pending_remote=false => exactly zero exposed wait C:4601-4605/:4631.
Counter includes only pauses, not scan/copy/cleanup; last 64-pause batch can straddle completion/deadline. Concurrent exposure is not proof of incremental remote delay.
Deadline starts at post; network publishes only on-budget prefixes C:1163-1169; choose requires finite, <=deadline, strictly earlier than local (ties local).
Finite reply after settled local is ignored/drained; late reply after 2500us cannot win even if local is slower: collector waits local and checks failure. Deadline never cancels local.
RTT p99 3.34ms exceeds budget; tail work is wasted, though earlier rows of those requests can meet deadline. No post-return map mutation occurs.
(5) Startup reserves/pending/rings persist; enqueue/bind/collect add no per-token allocations. Inherited predictor growth and periodic allocating/sorting remote_log remain exceptions.
Own completion release/acquire chain, speculative landed publication, serialized hook, protected speculative slots, next-plan drain and copy-before-abandon remain as round11; host piece test passes.
No new race established; timestamps still measure publication, not physical arrival. Graph source absent: exact-byte graph lineage and NUC output identity remain external prerequisites.
Exit ratios C:4885-4892 use max layer hedge-hook count as token denominator; posts=requests, wins=rows. Not independently verified generated-token count.
(6) win2i4 bash -n PASS; actual embedded parser tested: K1 AND K1b reject zero posts/wins, accept positive activity; K0 permits zero. Latest cumulative layer values avoid double count.
Window retains exact attestation, RAM/served ownership, health, gold MD5, four isolated legs, native rebuild. Missing token-count/finite-timing/speedup/waste/clean-exit gates; shutdown discards status.
Checks: python3 review13/check.py > review13/logs/check.log 2>&1 -> PASS offline checks complete; no GPU/model/NUC execution.
Host C++17 syntax uses local review9 headers, no liburing; extracted actual pool: 200 generations x17 pieces acquire-visible bytes; tie/deadline/NaN choices PASS.
No CUDA/native liburing/window run; source approval is not speedup or arbitrary row-byte identity certification. review13/fixes/ empty (no P1).
