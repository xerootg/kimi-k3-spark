# Decode remote hedge: i4 verdict, 2026-10-31
Evidence: [i4 legs](../evidence/legs-i4.txt), full [K0](../evidence/K0.log), [K1](../evidence/K1.log), [K0b](../evidence/K0b.log), [K1b](../evidence/K1b.log); reproduce with `python3 tests/analyze-remote6.py`.
K0/K1 mean completion latency: 185.598/187.244 ms/token (−0.88% throughput); repeats 185.038/187.257 (−1.18%). All supplied completion hashes are gold.
Final K1: 6963 requests, 8513 posted rows, 2131 wins, 235 short requests, zero NaN; K1b: 6942/8499/2196/246/0.
The quoted 16.0 posts, 4.33 wins, 2.28 ms wait/token are the FIRST 54-token dump. Final K1 is 18.086/5.535/2.731741; repeat 18.031/5.704/2.721753, denominator 385 hook tokens.
Buckets below use 384 generated tokens (cumulative scopes also include the prompt); `a` overlaps `b+c`, so never add it again.

| ms/generated token | K0 | K1 | delta | K0b | K1b | delta |
|---|---:|---:|---:|---:|---:|---:|
| a_ready | 51.731 | 52.578 | +0.846 | 50.955 | 52.761 | +1.806 |
| b_early | 37.097 | 24.118 | −12.979 | 36.655 | 24.293 | −12.362 |
| b_predict | 5.555 | 5.439 | −0.116 | 5.261 | 5.373 | +0.112 |
| c_real | 9.061 | 9.390 | +0.329 | 9.020 | 9.387 | +0.368 |
| e_other | 2.590 | 15.946 | +13.356 | 2.438 | 15.167 | +12.729 |

The b sum falls 13.096/12.250 ms/token; this is NOT measured savings from wins. `fill_late` replaces filler.wait/adopt with hedge_collect; b counts only pause batches, leaving scans/copies/other collector time in e_other. `a` actually rises.
Nonoverlapping b+c+d+e grows 0.589/0.847 ms/token; 1.057/1.372 of the wall regression is outside those bucket deltas (different scopes, drain, enqueue, scheduling). Drain itself is only 0.098/0.119 ms/token.
`exposed_remote_wait` counts pauses while a remote candidate is outstanding, often simultaneously waiting for its local read. It is occupancy, NOT incremental remote-induced delay; subtracting it from the wall loss to infer win savings is invalid.
For one row let L=local remaining at real hook, R=post-to-reply RTT, H=early-to-real lead, D=deadline from post, A=max(0,R−H).
Ideal read saving S=(L−A)+ times 1[R≤D]; waiting min(L,A) replaces local wait L. It is not an additional charge. Layer completion is the maximum across required rows; overlapping row savings cannot simply be summed. Remote wins also skip local GEMV.
With supplied rl median L≈1.5 ms, H≈0.5 and R≈1.79, A≈1.29 and a representative winning read saves ≈0.21 ms. Medians do not determine E[S] or the winning-tail mean.
Observed row win probability is 2131/8513=0.2503 (repeat 0.2584). Assuming mean saving/win 0.2–1 ms gives gross 0.050–0.250 ms/posted row (repeat 0.052–0.258); this is a sensitivity range, not a fitted expectation.
Measured NET workload gain/post is −0.0908/−0.1227 ms per request, or −0.0743/−0.1002 ms per posted row. Those estimates include every host/cache/compute effect and run variation; requests average 1.22 rows.
An exact counterfactual expected row gain is unidentifiable: i4 lacks joint issue/post/real/local-completion/remote-completion samples and local critical-path completion for substituted rows. rl marginal percentiles and sampled RTT percentiles cannot supply that joint distribution. No precise positive expectation is supported.
(a) Young/unstarted selection is a plausible, unproven way to improve win quality: target E[remaining | read age, size, queue] > expected remote arrival + overhead. Age alone is a coarse proxy; old reads can still be stragglers.
Real-demand c calls are only 222/28875 hooks ≈0.77% in K1 (all owners, not remotely eligible rows). The 9.39 ms/token c bucket includes rare expensive cases; it is not an available remote saving budget.
0013d records first speculative record dispatch before in-flight publication, resets on enqueue/reuse, filters before remote post, and leaves every local read/fallback intact. Dispatch age includes local worker/submission queuing; it is not per-chunk kernel block-I/O start time.
`LLAMA_EXPERT_STORE_REMOTE_HEDGE_MAX_AGE_US=300`: allow unstarted or age strictly <300 us; `=0`: unstarted only; unset: v2. Completed residents remain excluded. Queued-to-issued transitions can race the eligibility snapshot; this is a policy filter, not cancellation.
Test i5: L0/L1(300)/L0b/L1b(300)/L1c(0), 2500-us deadline, same frozen placement/exact/hash gates; L1c may legitimately have no posts/wins. No claimed hardware improvement before this run.
(b) Wire serialization: (24576+24576)*8/2.5e9=0.1573 ms, before headers/interrupts/stack. Survey Spark→NUC k=1 echo p50=0.297 ms, empty=0.104 ([findings](../survey/FINDINGS.md)); serialization is not the whole RTT.
Fast-exact evidence is heterogeneous: [FASTEXACT](../tools/glm53/FASTEXACT-RESULT.md) records a 1923-us k=1 comparison baseline; [MERGE](../tools/glm53/MERGE-RESULT.md) measures 3209 us on heavily loaded L8. Survey owner report is 1.2–1.9 ms/expert, not isolated i4 GPU timing. Do not substitute the ≈0.6-ms non-exact loopback number.
Final i4 sampled RTT p50/p90/p99=1.79/2.45/3.42 ms (K1b 1.79/2.45/3.29); 3.34 belongs to earlier evidence, not K1's final summary. These are reply samples, not paired per-row critical-path timings.
Optimistic current-kernel floor ≈1.4–1.6 ms if compute reaches the reported 1.2-ms lower end plus ≈0.2–0.3 ms transport; slower exact kernels raise it above 2 ms. This is an engineering estimate, not a measured lower bound.
The remaining RTT includes GPU preparation/submission/clock ramp, TCP handling, NUC busy-poll service and Spark net-thread wakeup/publication. Busy-poll removes a server sleep, not compute or Spark scheduling. i4 has no component trace to partition these costs; subtracting unrelated medians would invent precision.
Pinning/wakeup and keeping GPU clocks warm might remove hundreds of us, not the whole 1.79 ms. Even a 0.4-ms reduction gives ≤5.535*0.4≈2.21 ms/token at unchanged wins, an optimistic additive sensitivity (~+0.065 tok/s) before critical-path overlap/overhead; new wins could change it.
(c) 2500 us covers roughly p90, not p99. Cutting to 1790 us rejects many replies (roughly half the sampled RTT population versus ≈10% beyond 2500); the lost WIN fraction is unknown and not equal to reply fraction.
For fixed local/remote timings a shorter deadline cannot improve min(local, eligible remote) completion: local already ends the race; expiration only removes options. It lowers the exposed-wait COUNTER without necessarily lowering wall time, because the collector still waits/scans for local completion. Breaker effects could make it worse; test only as a secondary experiment, not a promised fix.
Owner verdict: today's broad decode hedge loses ≈1%; keep it off for throughput. A selective hedge can theoretically pay on this LAN, most plausibly (a) plus lower RTT (b); (c) alone supplies no timing reason for a gain.
Quantified selective scenario, NOT a forecast: retain 2 critical eligible rows/token averaging net +0.2–0.5 ms/row after overhead → save 0.4–1.0 ms/token; pooled baseline 5.396 tok/s becomes 5.408–5.425 (+0.012–0.029). Eligibility frequency/remaining tails are unknown; zero gain or regression remains plausible.
If i5 cannot repeat a positive gold result, stop decode offload work with these kernels. [Prefill study](nuc-prefill-offload-2026-10-31.md) also rejects today's kernels. Use NUCs for placement/traces, serving/control/monitoring, expert-store preparation and exactness testing; asynchronous independent work tolerates this LAN latency. None is a demonstrated GLM throughput gain.
