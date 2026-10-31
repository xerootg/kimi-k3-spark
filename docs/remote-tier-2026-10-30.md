# Window i: remote decode audit and 0013b design (2026-10-30)
Evidence: `evidence/legs-i.txt`, `R0.log`, `R1.log`, `R1b.log`, `R1d.log`, `win2i.sh`, `0013-remote-tier-glm.patch`; source references below are checkout-relative.
All 15 outputs match gold 9449ae5738; this proves fixture equality, not every remote row's arithmetic.
R0 mean 5.394 tok/s; R0b 5.394; R1 4.999 (−7.41%); R1b 5.042 (−6.53%); R1d 5.173 with zero offload.
R0/R1 reciprocal means imply 185.36/200.03 ms/token: about +14.67 ms/token.

## Where time goes
The premise “post at real lookup” needs correction: 0013 changes no graph timing; inherited 0003 already posts at SAME-layer early lookup with REAL ids/input, before local planning.
See `patches/0003-expert-store.patch:1862` (input copy before early), `:6231` (remote_post); REMOTE_LATE_POST is unset by win2i.sh.
That overlaps remaining same-layer GPU work, unlike the local predictor's one-layer-ahead weight reads. The real lookup resolves/waits and issues fallback.
R1: 13669 candidate layers / (3×128 reported tokens) = 35.60/token; ×0.792 ms = 28.19 ms/token exposed remote resolution.
Counters span decode plus short single-token prompt tails; 28579 early lookups / 75 layers = 381.05 token-equivalents, giving 35.87 candidates and 28.41 ms/token instead. Treat both as approximate decode normalizations.
Crucially, 0.792 includes fallback I/O, not just network wait: 0003:6457 starts the timer, :6579 ends it AFTER fill_plan/pool->run/fill_adopt. Do not add fallback time again.
R1 RTT p50/p90/p99 = 1.57/2.03/2.26 ms; some latency already overlaps before resolve.
Window w's supplied exposed local wait is 36.9 ms/token at ~1.35 records/layer (~101 records/token); this earlier-window number is not independently present in the supplied files.
Window i R0 instead logs 0.57 ms/early lookup ×~74.4 = 42.4 ms/token, 1.36 records; R1 0.41×~74.4 = 30.5 ms/token, 1.00 record.
Thus ~11.9 ms/token less local split wait offsets ~28.2 ms remote resolution: net ~16.3 ms, close to observed +14.7; rounded cumulative counters prevent exact attribution.
A correctly predicted weight already ready locally has ~zero marginal exposed read time: replacing it with a remote wait loses. Not all predictions are ready; R0 logs 6834 in-flight waits averaging 0.304 ms.
R0→R1 logged decode reads INCREASE 236.16→273.20 GB (+15.7%); predictor-used fraction rises 66.8→77.1% but these logical counters do not measure physical NVMe traffic. Offload hits do not imply aggregate I/O savings.

## NaN refusals and late local fallback
R1: 9642 used, 3979 non-finite refusals, 48 late = 13669; fallback 29.461%, NaN alone 29.110%; R1b 3156 NaN +22 late /10418 =30.505% fallback.
Remote input set was (nuc2 RAM ∪ SSD owners) ∩ served =7228; only 3588 records were reported loaded in RAM. SETOUT advertises supported disk rows too (`tools/glm53/g53x.cpp:696`).
NUC refusal paths (`g53x.cpp:964–1035`): unknown/unavailable kind, failed disk acquire/read or expired disk deadline, failed launch, all-zero GPU result guard; genuine computed non-finites are also rejected by Spark.
`evidence/NUC-DEPLOY.md:111` explains 20 ms disk expiry, but cannot identify these 3979 rows: 0013 publishes rows only by 2.5 ms, so a fresh 20 ms timeout cannot explain on-time NaN counts.
Nonresident alone is not refusal with a working disk tier: supported disk rows are read/computed. With disk absent or mismatched resident set, unknown ids refuse immediately.
Layer 78 is unsupported MTP, but win2i validates sets as 3≤layer<78 and R1 per-layer counters end at 77: it cannot explain these refusals.
The zero-row guard is a documented alternative (`tools/glm53/EXACT-RESULT.md:36–39`); the supplied Spark logs carry neither rejected expert IDs nor NUC refusal reasons. No defensible per-cause split exists here.
Missing NUC runtime env, disk stats and zero-row counters must be captured before claiming “29% SSD timeout” or “29% nonresident”.
Fallback costs remote resolution followed by local fill/adopt and late GEMV; when no speculative read exists this is a fully exposed local read. Existing predictor reads can still be adopted, so “always a fresh read” is too strong.

## R1d: breaker, not broken connection
`evidence/R1d.log:1298` says “0 unsent, 39 breaker trips”; 1 connects, 0 drops (0 stale), 46 posts, 0 used, 18 NaN +28 late; resolve averages 1.403 ms.
Inherited breaker is `patches/0003-expert-store.patch:3130–3157`: 8 consecutive short requests → skip 2000 ms, then one half-open retry; another miss re-trips. 8 initial +38 retries =46 posts for 39 trips.
0013:182–187 still increments misses_in_row for ANY short request, including NaN. Log warnings continue every ~2 s: the tier did not permanently stop; it mostly bypassed candidates while open.
1500 us is below R1's 1.57 ms median RTT; no accepted successes reset the streak. Residual 4% slowdown cannot be assigned to 46 resolves (~64.5 ms total) alone; cache/I/O variability remains.

## 0013b: exact hunk sketches and feasibility
References `0013:NN` below mean line NN of `evidence/0013-remote-tier-glm.patch`; target paths are engine paths, not files edited here. These are design sketches, not an applicable implementation patch.
1. `src/llama-moe-cache.cpp`, layer fields at 0013:65–68: `+ PendingRemote pending; // token, layer, gid[], slots, expiry; no global cross-layer scratch`.
2. Predictor callback (inherited 0003 prefetch path), paired graph change in `src/llama-graph.cpp` at 0003:1845: `+ prepare_remote(next_layer, predicted_ids, token);` before local predictor prefetch; intersect RAM-only owners, skip Spark-ready rows, cap globally at one row initially.
3. Do NOT send next-layer FFN work there with current-layer x. k3x requires 17 int32 words (k, up to16 gids) PLUS the destination's exact 6144-float activation; that input depends on unfinished current-layer output and next-layer work.
4. Current protocol cannot return exact next-layer rows one layer ahead from predicted IDs alone. A new PREPARE(ids)/EXECUTE(exact_x) protocol could prepare weights then compute at earliest exact-x availability; for RAM rows PREPARE offers little beyond bookkeeping.
5. Earliest exact-input hook (`src/llama-graph.cpp`, inherited 0003:1874): `+ copy_exact_x(dst); post_prepared(dst, token);` retain stream dependency. Existing same-layer early post already does this with real IDs; moving it earlier requires proving exact input readiness, not renaming a hook.
6. `src/llama-moe-cache.cpp` remote_resolve at 0013:136–149: `- spin until deadline; + collect_ready(token, layer, gid);` real hook only collects; map surviving predicted IDs to ACTUAL positions, discard predictor misses/stale generations, fall back immediately for missing rows.
7. Keep local prediction reads as a hedge initially; suppress only when a valid remote row is ready. Otherwise speculation failures turn into late NVMe stalls. This preserves latency but initially saves no speculative NVMe bandwidth.
8. At 0013:174–195: separate `nan_refusals`, `deadline_misses`, `prediction_unused`; breaker counts transport/deadline failures, not speculative unused IDs. Preserve bitwise finite test (:126–133), deadline publication (:58–60), per-row holes (:158–168).
9. At 0013:199–208 add split timers for network collection versus fallback/adopt, token-normalized used/predicted-wrong/ready-local counts; NUC log gid plus refusal reason. Assert wire framing, token ownership, stale-slot drain, predictor holes and exact-input identity before hardware A/B.

## Expected value and next measurement
RAM-only remote.set = nuc2-ram.set ∩ fresh nuc2.served.set (same for nuc3); 3588 is a resident count, not a promise all generated RAM rows loaded. Preserve host ordering/unique owners; keep SSD ownership for future experiments only.
Ideal hiding could recover the ~14.7 ms regression toward 5.4 tok/s; it is not evidence of speedup beyond baseline. Even eliminating ALL window-w exposed I/O gives only 1000/(185.36−36.9)=6.74 tok/s, an unattainable whole-system upper bound for this subset tier.
Current used rows ~25.1/token could replace ~0.30 GB/token of 12091392-byte records if all reads were avoided; at 5.4 tok/s ~1.64 GB/s logical bandwidth ceiling. Existing speculation/hedging reduces this substantially; physical NVMe savings are unmeasured.
Predictor misses cannot benefit from a request restricted to predicted IDs unless separately posted once real IDs arrive. Decode gains need avoided exposed I/O or bandwidth relief to exceed NUC wait/coordination; neither is established at 5.4 tok/s.
Prefill/batch>1 may amortize network and use NUC compute, but inherited remote path is single-token split GEMV; batching needs new engine/protocol work and measured throughput. Its value there is a hypothesis, not a demonstrated win.
Next hardware gate: frozen-snapshot off / RAM-only same-layer early / prepared collect-only, matched gold AND nonzero used rows; separate read/resolve timers, predictor coverage, NUC refusal reasons and physical disk bytes. No hardware or network used for this audit.
