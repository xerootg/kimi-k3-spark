# GEMV6
BG0 5.357/5.368 → BG1 2.137/2.136 tok/s (−60.2%); gold 9449ae5738; BG1 explicitly logs bank on.
0028's three-group 16 B bank schedule is rejected; keep GGML_CUDA_GB10_BANK_GEMV=0.
All eight standalone shapes pass correctness but regress 2.87–12.87×; bank useful bandwidth 10.51–19.20 GB/s.
Standalone stock is 42.80–163.92 GB/s; historical engine aggregate ≈132 GB/s is a different measurement.
Issued vector pointers ARE 16 B aligned: bank_bytes rounds down, optionally reads the next vector, then extracts bytes.
Vector-field lane-request inflation: IQ2 2.75×, IQ3 3.333×, IQ4 1.5×; coalescing means these are not fabric traffic ratios.
Per-lane reconstruction/staging adds instructions and long-lived temporaries; IQ4 stages 1632 B/thread of source bx storage.
Spills/register pressure and system-path queue pressure remain hypotheses without SASS/ptxas/counters; no precise hardware attribution is justified.
Late one-live shapes also collapse; IQ3 down beats gate/up at identical stride/live; IQ4 has least inflation and greatest slowdown.
Thus live-count overload, rounding alone and page-size alone do not explain the results; supplied THP observation also weakens the latter.
Preserve stock compact warp-coalesced field/line access and reuse; no promised gain from more per-lane groups.
A's source/full output are absent: cannot establish its exact access pattern; uvmbw.cu is a different aligned chunk scan, not verified A.
132 vs ~160 GB/s suggests limited headroom (~9.7 ms/token optimistic), not a proven hard ceiling or proof stock cannot improve.
Updated docs/experimental/host-bank-gemv-plan.md: former rank 1 dead; DMA ring for ~1 missing record/layer and sparse late launches remain unvalidated.
GM1.log has zero GB10 GEMV lines, not even disabled; early device-discovery logging is absent more generally.
No evidence of a bad 1210/48-SM guard: BG1 passes equivalent checks; missing init logging does not prove GM1 used stock.
0027 now logs policy/guard at both MMVQ entrypoints and actual custom launches per shape; window scripts unchanged.
Tests: python3 tools/check-gemv27.py; python3 work/check-gemv6.py (patch chain, source checks, bandwidth audit, bash -n): PASS.
No nvcc/GPU available; rerun gm on Spark and require GB10 GEMV dispatch evidence before interpreting its ~−0.8% delta.
