H1−H0: +5.038 ms/token (190.418 vs 185.380); H1b−H0b: +6.488 ms/token; 384 generated tokens/leg.
H1−H0 real buckets, ms/token: a_ready +4.122; b_early +9.360; b_predict −5.396;
c_real +0.094646; d_lock +0.000052; e_other +0.098021. a overlaps b+c: NEVER add a again.
Combined early+predictor wait grew 3.964 ms/token; non-overlapping b+c+d+e grew 4.157.
Thus ~83% of the first slowdown appears in real hooks; ~0.881 ms/token remains outside these buckets.
Repeat H1b−H0b: a +6.057; early+predictor +5.542; c +0.445; lock −0.000029; other +0.651.
Repeat non-overlapping growth is 6.639 ms/token (timing scopes differ from completion throughput).
Startup logs reveal H0 predictor io_uring versus H1 predictor pread threads: 0013b overwrote its default!
Own completion-tagged fills also use pread; logs cannot partition backend cost from polling/bookkeeping.
0.622×28579/384 ≈46.29 ms/token includes LOCAL-only waits and collection overhead, not extra remote delay.
H1 labels all 28579 split hooks candidates although only 7214 posts occurred (18.786/token).
2221 remote rows won (5.784/token); 267 short requests; zero NaN; supplied gold hashes match.
Final full-log drain is 46.529 ms/28578 calls =0.121 ms/token; 3947/5.505 was an intermediate dump.
Final H1 RTT p50/p90/p99=1.78/2.45/3.34 ms: 2500 us does NOT cover the measured p99.
Hit rates 62.20/62.21 do not explain the loss; drain is small; late replies waste work.
Source proves post() ALREADY queues into eight preallocated slots; only remote_client::main calls send_all2.
No synchronous hook-thread TCP send exists to move; the hook retains its necessary exact-activation copy.
0013c preserves predictor backend before the tagged-fill override; explicit PREFETCH_URING still takes precedence.
Enqueue now precedes local plan/issue, using the exact same-layer activation; all selected local reads remain.
After planning, bind posted expert IDs only to actual late rows; newly ready/synchronous rows stay local.
No-post layers use normal wait/adopt. Actual-post layers poll in 64-pause batches; NaN rows are scanned once.
Remote wait counts only blocking pauses with unresolved, eligible hedged rows, excluding scan/copy/cleanup.
Local-only/expired/refused rows add zero remote wait; own/predictor waits remain in their respective buckets.
Exit summary prints posts/token, wins/token, exposed_remote_wait_ms/token and its explicit token denominator.
Queue/enqueue/collector add no per-token allocations; byte substitution, deadlines and local fallback remain.
win2i4.sh: K0/K1/K0b/K1b, 2500 us, same placement/exact/md5/wins gates, applies 0013+0013b+0013c.
Expected: remove much of the 4–6 ms/token host regression, bringing H1 toward ~5.39 tok/s; speedup unproven.
Recovering 4–6 ms from first H1 gives ~5.36–5.42 tok/s; coverage and RTT tails still limit wins.
Local checks: bash tests/run-remote-hedge2.sh; python3 tests/analyze-remote5.py. All checks pass.
No network/hardware run; full engine compilation needs absent llama headers/CUDA. i4 must establish gold and speed.
