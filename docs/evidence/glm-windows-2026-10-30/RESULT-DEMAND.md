1. HW0: 20525.258/384 = 53.45 ms/token of real-hook service, not pure NVMe time.
a=19.891 s is entry-to-readiness/adoption when not ready at entry; it includes b+c, planning and other work. It does NOT mean only already-in-flight reads.
b_early=14.165 s times filler.wait(), b_predict=1.995 s times per-slot adoption spins, c=3.724 s times nonempty synchronous fill_real pool runs. Counts are timed hook totals, not record counts.
b+c=19.884 s; adding a again double-counts. d=.000909 s is lock acquisition; e=.638458 s is remaining hook service after b+c+d. abc histogram deliberately double-counts readiness/waits.
53.45 ms/token includes non-split/prompt hooks; it is not an isolated steady-state decode stall measurement.
evidence/legs-w.out:71 reports 235.76 decimal GB for HW0 decode accounting: /384=.614 GB/token, not 7.3 GB/384.
At the supplied 6-7 GB/s ceiling, that alone needs 33.68-39.29 s/leg =87.71-102.33 ms/token of overlappable device service.
At 5.405 tok/s, wall time is about185.0 ms/token, implied counted traffic3.32 GB/s; this does not prove drive saturation or that all53 ms is removable.
The counter sums selected fill records; it excludes side-reader speculative traffic. n_prefetch_reads counts QUEUED records, including reclaimed unread ones; 42770 cannot simply be multiplied into physical bytes.
docs/experiments.md is absent here. Evidence's floor report identifies7.333 GB as routed GPU payload PER TOKEN, not NVMe decode reads. Need completed physical bytes/device busy time to decide saturation.
Only quantiles were printed: exact hooks >1 ms cannot be recovered. HW0 p50=.010, p90=1.950 bounds roughly2888..14437 hooks, or7.52..37.60/token (75.20 total hooks/token).
A .010-ms bin includes tiny overhead, not proof of zero wait. Entry-ready=14359+74=14433/28875, just under half; b counts also include tiny residual waits.
Early statistics average1.35 incomplete records/layer; no evidence maps each tail hook to exactly one or two records. Need threshold counters and batch sizes.
2. Default liburing: fill caller has its own TLS ring(depth64); early fill thread and real-hook poller have separate rings. Spec coordinator has another ring(default8 pieces).
Without liburing, fill run_mutex serializes the shared16-worker pool (caller also helps); speculative reader has separate pread workers. They share the block device; hardware queue mapping is not established by this log.
fill_plan already preempts NEW speculative records; started records finish, including remaining pieces. Existing NVMe submissions cannot be preempted by a dedicated thread or RT priority.
c:3724.066/222=16.775 ms per BATCH, not per record. fill_late never issues c reads: split-decode predictor misses are issued EARLY and charged to b.
28875-28579=296 non-split hooks;222=3x74 strongly suggests the three prompt passes. The logs lack IDs/batch sizes to prove attribution; demand-queue congestion is unproven.
Conditional single-record scenario:222x4=888 ms; saving2836.066 ms=7.39 ms/token,13.8% of real service, about4.2% throughput at185 ms/token.
Removing c entirely saves9.70 ms/token,18.1% of real service, about5.5% throughput. Neither is a justified decode forecast if c belongs to prompts.
3. b_early:14164.544/10337=1.370 ms residual wait/counted hook,36.89 ms/token; it excludes time overlapped before the real hook.
At layer time2.7 ms,12 MB single-stream at3 GB/s takes4 ms; at6-7 GB/s ideal service is1.71-2.00 ms/record, or2.31-2.70 ms for1.35 records/layer.
The early-to-real overlap is only the resident matmul, not necessarily the entire2.7 ms layer. Issue timing, device contention and incomplete overlap can explain b; logs do not measure full record latency/concurrency.
0003 already splits records into1 MiB pieces (~12 per record); depth64 fill ring or16 workers, plus default8 speculative pieces. Splitting into4 pieces is not new parallelism and may reduce it.
More depth cannot beat bytes/bandwidth; test lead time and actual completion latency before increasing depth. A dedicated N=2 pread path could lose throughput on large prompt batches.
0025: LLAMA_EXPERT_STORE_DEMAND_THREADS=0/unset preserves today;1..64 creates reserved pread readers. Only real CUDA-hook fill_decode synchronous reads use them, independent of HOOK_STATS.
Queue storage is a fixed borrowed-batch descriptor/cursor, allocated with the reader at init; caller blocks until every worker leaves. Existing1 MiB chunking, offsets, bytes, LRU and math stay unchanged.
Each reader requests IOPRIO_CLASS_RT/0 once using ioprio_set on its own TID. Failure warns and continues; device scheduler support/privileges determine effect. No extra speculative pause was added.
Tests: clean-copy apply/check, full-source syntax, extracted actual queue host test and ASan/UBSan; scripts parse. Local RT request returns EPERM. No Spark/model run here.
win2dp.sh copies c -> ~/llama-k3-dp; DP0/DP1=2/DP0b/DP1b=2, SZ=1 HL=1 stats=1 wide=0, frozen snapshot and all-request md5 gate9449ae5738. Compare buckets and prompt vs decode timings before adoption.
