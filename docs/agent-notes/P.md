# P notes (phase pass: head-to-head A/B/C on the Spark, Intel-side tuning, nuc3 as a second box, GitOps deploy)

- 20:00Z 2026-10-29 P START. Holding NOTHING. Work dirs: nuc2 ~/k3-bitexact/p, nuc3 ~/k3-bitexact/p, Spark ~/p-runs.
  Test servers on port 19651 (nuc2) / 19652 (nuc3) only. Every window is logged here as WINDOW START/END.
- killed two stale wait loops left by earlier agents on nuc2 (pids 3252225 "until ... b_fc.log ... kill k3x-v10 19611",
  3354121 "until ! pgrep /opt/k3x/k3x ... MemAvailable"); both were only sleeping.
- 20:02Z **P WINDOW 1 START (Spark + nuc2)**: kimi-k3-llama and kimi-k3-k3x-nuc2 scaled to 0 for phase 1 (6 interleaved legs A B C A B C, ~100 min). Test server nuc2:19651 only; EPP switched per leg (logged by psrv.sh), restored to balance_performance at the end.
- 20:14Z sampler.py (GPU runtime-PM + vmstat deltas, cpu 13, 20 ms) running for the whole window 1 -> sampler-w1.log
- 20:33Z coordinator added 3 Spark tests after phase 1: GB10 SM clock lock (-lgc 3003) A/B, --threads 4/6/8/10 sweep, 1/2/3 concurrent clients throughput with the pool on.
- 20:50Z coordinator: agent Q takes ~45 min Spark windows only when P.md shows no open window (read Q.md before opening one). Added: audit #4 (REMOTE_ROWS 3/4 with K3X_CPU_SPLIT=1,2,3,4, DEADLINE ~1700) and audit #10 (k3cpu re-execution of stalled chunks + SO_BUSY_POLL; GGUF arena if the CPU half wins).
- 21:21Z **CPU-stall cause found (nuc3, same platform):** k3cpu_test bench (v7, k=2, 5 ms gaps, workers 1-11, caller cpu 0) + bpftrace
  (off-CPU/runqueue/irq/softirq/fault/reclaim probes): 246 of 252 tokens over 3 ms coincide with a worker or the CALLER thread being
  PREEMPTED (state R) by node threads: longhorn(-manager), kubelet, containerd(-shim), runc, kworkers, for 1-24 ms. Not SMIs (turbostat
  SMI 0 on nuc2), not page faults (k3x minflt flat during decode), not reclaim. Why K3X_CPU_RT alone (I's c4) did not fix it: the request
  thread itself stayed SCHED_OTHER. Bench: base >3ms 318/10000 (max 27 ms); caller RT only 114; caller + workers RT 1 (p50 0.805 vs 0.898 ms).
- 21:21Z k3x change (worktree ~/repos/k3s-p, branch p/k3cpu-help): K3X_RT=<prio> = request thread SCHED_FIFO|RESET_ON_FORK,
  blocks in poll() after K3X_RX_SPIN_US (default 1000 with RT) so it never spins unbounded at RT; memwarm thread forced SCHED_OTHER.
  vall L1/L11/L83 bit-identical (GPU and CPU paths) on nuc3. nuc3 local rrclient (cpu 13, 5 ms gaps, 8000 req, CPU half split 1,2):
  no RT: server p50 848 p99 2694 p99.9 10614 max 28330 us, >3ms 67 | K3X_RT=50 + K3X_CPU_RT=50: p50 698 p99 867 p99.9 1038 max 1792 us, >2ms 0.
- 21:35Z **P WINDOW 1 RESULT (phase 1, all 24 runs gold).** Native S2 engine (~/llama-k3-s2), prod env, nuc2:19651, 3x200 tok/leg:
  | leg | server | tok/s runs | eng RTT p50/p99 | >3 ms / short | nuc2 pkg W* | cabinet W (runs) |
  | A1 | g580ea49 fw+memwarm, EPP bal | 2.112 2.111 2.106 | 1.30/1.58 ms | 32 / 29 | 21.1 | 550 |
  | B1 | g486d2b9 CPU half 1,2, EPP perf | 2.110 2.109 2.098 | 0.67/1.29 | 133 / 126 | 25.3 | 550 |
  | C1 | g486d2b9 split 1,1 + fw, EPP perf | 2.106 2.109 2.100 | 0.86/1.42 | 115 / 107 | 22.9 | 551 |
  | A2 | = A1 | 2.104 2.108 2.102 | 1.30/1.57 | 11 / 11 | 21.8 | 583 |
  | B2 | = B1 | 2.105 2.102 2.106 | 0.67/1.24 | 87 / 79 | 25.1 | 557 |
  | C2 | = C1 | 2.102 2.109 2.110 | 0.86/1.42 | 90 / 83 | 23.3 | 565 |
  (*RAPL package incl. the Spark's load phase.) Means A 2.107, B 2.105, C 2.106: a tie at REMOTE_ROWS=2 (the reply is in before the
  real lookup in all three: wait 1-3 us/layer). Spark GPU power 24-27 W in all legs. Cabinet noise (+-20 W) hides the nuc2 delta.
  Q's build (21:02-21:08Z) overlapped B2's load phase only. Test servers served 7025 of 7031 records (MAX_GB 49.5), like I's legs.
- 21:35Z **P WINDOW 1 END pending**: both prod pods scaled back to 1 at 21:35Z, gold check via API next.
- 21:40Z **P WINDOW 1 END (Spark + nuc2)**: prod back, nuc2 pod FULLCHECK 6967/6967 serving 19600, API gold b2c79009 / e801484d (2.046 tok/s). Holding NOTHING on nuc2/Spark. nuc3: my test servers on 19652 come and go (k3cs work), nothing on nuc2. Q may take its window now.
- 22:01Z P: pushed kimi-k3-spark 09d9da7 (k3x K3X_RT + k3cs CPU-only GGUF server, both in image k3x-serve:g09d9da7, CI building). k3cs on nuc3: FULLCHECK 636/636 + remotecanary K=1/2/3/4/16 L1/L11/L83 PASS (container-built binary md5 25f43065). Capacity: 21 GB of records locked on nuc3 with 7 GB MemAvailable left. Placement replays (Spark ~/p-tools): NVMe misses/layer rows2/3/4: nuc2 only 6.13/5.98/5.93; nuc2+nuc3(20 GB, by layer) 5.93/5.68/5.56.
