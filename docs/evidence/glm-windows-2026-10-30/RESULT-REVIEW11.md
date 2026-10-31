Verdict: no P1 defect established in 0013 + 0013b; conditional source-level approval, not native bit-identity/performance certification. No P1 fix patch warranted.
Scope: copied read-only review11/c-tree with cp -a to review11/candidate; applied supplied 0013 then 0013b. Inputs untouched.
References below: C=review11/candidate/src/llama-moe-cache.cpp; U=candidate/ggml/src/ggml-cuda/ggml-cuda.cu; M=mmvq.cu there.
(1) Placement: C:4494 saves pos=merge_map[j], id=ids[pos], late=j. Winner C:4585-4591 copies unweighted bytes to remote_rows[out], rmerge[out]=pos, late_map[j]=g.n_slots.
It never zeros the physical slot, slot_map, or another layer's map; the zero record is the dedicated extra bank record. Other late-map/merge positions stay unchanged.
Inherited graph corroboration: review9/candidate/src/llama-graph.cpp:2543-2555 first set_rows installs local late rows; second overwrites pos with remote bytes; router multiplication follows, then ascending expert adds :2571-2588.
C-tree contains no llama-graph.cpp: this corroborates the available inherited implementation, not an independently supplied Spark graph/build. Owner must establish that lineage.
M:681-692/:1196-1205 skip only the selected channel, with finite activation scales and dispatch restrictions :1915-1921; otherwise zero-bank arithmetic runs and that position is still overwritten.
Ordering: U:1709-1712 publishes hook writes before done; :1813-1818 stream wait precedes late get_rows/GEMV and both merges. A host-enqueued matmul is not an executing matmul.
Replies arriving while the real hook waits can win. After hook return/abandon they cannot alter maps/remote_rows; no late asynchronous overwrite races a launched GEMV.
Identity therefore additionally requires the NUC's entire unweighted expert result to match the local kernel; attestation and fixture text MD5 cannot prove arbitrary row-byte identity.
(2) Own pieces C:529 and :644-645: acq_rel decrements chain all piece writes into last completion's release timestamp; collector :4477 acquire sees all bytes (host test passed).
Spec reader :1578 releases landed after all pieces; :1320 clears landed before queue publication. In-flight spec_state prevents reassignment; no stale timestamp is used for a new queued generation.
Prepare :4492 clears own completion and all pending fields; pending capacity n_expert_used at init :3462; next-layer/token reuse occurs after filler drain. No generation number needed under serialized hooks.
C:3813 serializes hook state with lru_mutex; breaker next_free/collect counters belong to this hook thread, not socket/read workers. Network connectivity/drop counters are atomic.
P2 precision: timestamps describe completion publication work, not physical NVMe/socket arrival. A preemption between timestamp creation and release publication can hide an earlier timestamp from a local-first settle; acquire/release alone is not a simultaneous snapshot.
(3) Drain code C:4451-4458 calls filler.wait(), checks failure, then demotes; fill_early :4193 and fill_plan :4112 call it BEFORE the next real plan.
Predictor can precede this drain, but C:5049-5070 skips executing_il=split_il/last_fill_il and speculative busy slots; valve :2637-2642 skips busy/protected layers under the same lock.
Losing speculative reads need not all drain: spec_state keeps their slots protected until completion. Destructor :3741-3747 drains own fills and joins speculative reader before bank destruction.
Remote ring: rows_done release/acquire publishes immutable completed prefixes; memcpy to layer-owned remote_rows occurs BEFORE abandon/free :4606-4612. Only hook posts reuse; abandoned unfinished replies drain on net thread before FREE.
(4) Pending, slot timestamps, socket x/rows rings and hedge job capacity are allocated at startup :3447-3462/:948-951; collect uses fixed stack arrays, memcpy and scalar checks, no string formatting with HOOK_STATS off.
Not globally allocation-free: inherited speculative deque/vector growth remains possible; every 4096 fills :4441 calls timing logging, whose remote_log :4876 allocates/sorts a vector even with HOOK_STATS off. Init to_string is not per token.
(5) C:4697 rejects every NaN/Inf float by exponent bits. Invalid rows keep local fallback and do not themselves increment the breaker; other rows in the same request can still produce a genuine failure.
choose() uses remote only finite/on-time/strictly earlier; ties local. Deadline disables remote eligibility, never cancels unfinished local; local-before-deadline settlement avoids a spurious timeout.
Collect counts posted transport/deadline failures once per box/request, local wins and NaN neutral; healthy remote wins reset streak unless another row failed. Trips suppress posting, never local reads.
REMOTE_EXACT must literally equal 1 and early post be enabled :3443-3445; SHADOW refused :3480. These checks apply when remote initialization is active, not to an off/invalid configuration.
(6) Window requires NUC_EXACT_VERIFIED=1; its invalid-gid probe requires exactly 6144 NaNs and establishes protocol health only. No exactness certificate is exchanged.
RAM set is validated nuc2-ram intersect served, after disjoint/exhaustive 19200 owner pairs and range/duplicate checks; SSD and other owners cannot enter it.
Window rebuilds copied tree; isolated snapshots; H0,H1,H0b,H1b,H1c; each response requires gold MD5. Every H1 leg also requires aggregate posts>0 AND remote_wins>0 after shutdown.
Thus silently zero-win H1 cannot pass; wins need not occur in every request. Breaker/NaN/local-only behavior can intentionally fail this activity gate while output remains correct.
P2 window gaps: no predicted_n=128, finite-positive timing, per-response wins, speedup/repeatability, lower exposed wait, drain-stall, NaN/deadline/breaker-threshold or clean-exit gate; stop_server discards status.
Checks: python3 review11/check.py > review11/logs/check.log 2>&1 -> PASS; base apply-check, hedge reverse-check, bash -n, full C++17 host -fsyntax-only using local review9 headers (liburing disabled).
Actual extracted io_pool test: 200 generations x17 pieces; acquire-visible complete bytes before join; header tie/deadline/NaN choices PASS. No CUDA compiler/device/model/NUC/window run; no network/commit/push/rm.
