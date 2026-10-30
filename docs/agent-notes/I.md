# I notes (integrate S1 zero-copy k3x + S3 k3cpu: split experts iGPU/CPU on one arena; rrclient + Spark A/B)

- 05:11Z I START. Holding NOTHING. Dev builds on the laptop (docker), validation on nuc3 first (~/k3-bitexact/i/ there),
  then a nuc2 window (pod to 0) logged here as "I WINDOW START/END". Test servers on port 19631 only.
- 05:31Z **I WINDOW 1 START (nuc2)**: scaling kimi-k3-k3x-nuc2 to 0 for ~70 min (until ~06:45Z at the latest). My test server on port 19631 only (root, cpus 0-3 + CPU workers on 4-11). Back to 1 + Ready before WINDOW END. nuc3: my validation runs (cpus 0-11) done for now.
- 05:44Z I: nuc2 host change for a test: EPP balance_performance -> performance on all cpus (will revert to balance_performance)
- 05:45Z I: nuc2 host change for a test: uncore min_freq_khz 400000 -> 3800000 (will revert to 400000)
- 05:46Z I: uncore min_freq_khz reverted to 400000
- 05:59Z I: EPP reverted to balance_performance on all cpus (A/B continues)
- 06:00Z I: EPP -> performance again for the final-config sweep (will revert)
- 06:04Z **I WINDOW 2 START (Spark)**: kimi-k3-llama scaled to 0 for ~95 min (7 native legs: c1 d1 b1 a1 b2 d2 c2; until ~07:45Z at the latest). nuc2 window 1 is EXTENDED through it: nuc2 swaps between my server (19631, legs c) and the production pod (legs a/b) during the tier-off legs d. Restore of both + gold md5 before WINDOW END.
- 06:16Z I: swap during leg d1: my server stopped, EPP back to balance_performance, scaling the pod to 1 for legs b1 a1 b2
- 07:00Z I: swap during leg d2: pod to 0, vall + my server (k3x-g486d2b9 = CI image binary) on 19631 for leg c2, EPP performance
- 07:24Z I: all 7 legs done (all gold). My server stopped, EPP balance_performance restored, uncore min 400000; restoring kimi-k3-k3x-nuc2 (1) and kimi-k3-llama (1).
- 07:31Z **I WINDOWS END (nuc2 + Spark)**: kimi-k3-k3x-nuc2 back at 1 (production image g4de37ce, FULLCHECK 6967/6967, Ready); kimi-k3-llama back at 1, API gold b2c79009 / e801484d (2.043 tok/s). nuc2 host settings as before (EPP balance_performance, uncore min 400000). Holding NOTHING on nuc2/nuc3/Spark.
- 07:32Z I RESULT: CPU half committed (kimi-k3-spark 49c4c92, 486d2b9, d7b63ed; image k3x-serve:g486d2b9). Engine A/B all gold: c (S2 engine + CPU half) 2.102/2.101/2.088 and 2.111/2.111/2.106 tok/s vs b (S2 + prod server) 2.052/2.057/2.054 and 2.040/2.052/2.048, a 2.047/2.045/2.040, d (tier off) ~2.04. Engine RTT p50 0.65-0.67 ms (was 1.93). Needs EPP performance on nuc2 + K3X_CPUS=0, K3X_CPU_CPUS=1-11, K3X_CPU_SPLIT=1,2, K3X_MEMWARM=4000, K3X_MEMWARM_CPU=12. Details in tools/bitexact/sycl/README.md.
