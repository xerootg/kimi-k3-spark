# M (measurement): where the remote round trip goes

- 20:11 MDT M START. Plan: instrument k3x serveidx (K3X_TRACE) + the Spark engine remote path
  (LLAMA_EXPERT_STORE_REMOTE_TRACE), own trees only (Spark ~/llama-k3-m, nuc2 ~/k3-bitexact/m/). Holding NOTHING yet.
  Before any window I write WINDOW START here (nuc2 pod -> 0 and/or Spark prod -> 0) and WINDOW END when restored.
- 20:18 MDT **M WINDOW 1 START (nuc2)**: kimi-k3-k3x-nuc2 scaled to 0; my instrumented k3x-m1 on port 19613 (NOT 19600, so prod does not connect). Back to 1 by ~21:30 MDT.
- 20:31 MDT M: window 1 server-side results in; server k3x-m1 (host trace only) on 19613 for window 2. **M WINDOW 2 START (Spark)**: kimi-k3-llama scaled to 0 for 2 native legs (~35 min), then restore + gold md5.
- M RESULT 1 (window 1, rrclient vs instrumented k3x on 19613): the missing ~0.8 ms is NOT on the Spark. rrclient back to back
  p50 1.06 ms (dist k), but with an 8 ms gap between requests (like the engine: one request per layer) p50 1.94 / p90 2.55
  ms = exactly the engine's 1.98/2.48. Client spinning vs sleeping: same. Server-side cause: after idle the GT is in RC6:
  the FIRST SYCL submission (x memcpy) blocks the host 0.27-0.6 ms (RC6 exit), and every kernel runs ~35 % slower
  (memory/fabric clocks down: gate/up 292->356-395 us, down 122->181 us per expert). Holding i915_forcewake_user open
  removes the RC6 part: gap-8ms RTT 1.94 -> 1.30-1.35 ms p50, p90 2.55 -> 1.50.
- M RESULT 2 (window 2, instrumented engine leg A, gold 2.049/2.038/2.037): engine RTT p50 1.97 = net-thread pickup 74 us
  (100 us idle naps) + send 17 + server 1753 + wire/stacks 123. Window post -> real lookup hook p50 1.51 ms.
  Leg B now: same with forcewake held on nuc2.
- 21:00 MDT M: leg B (forcewake held on nuc2) gold 2.121/2.119/2.122 tok/s (vs A 2.049/2.038/2.037), RTT p50 1.97->1.26 ms, wait at real lookup 0.231->0.005 ms. **M WINDOWS END**: forcewake released, test server stopped, k3x-nuc2 pod back (FULLCHECK ok, Ready), Spark prod back + gold md5 via API (b2c79009 / e801484d, 2.047 tok/s). Holding nothing.
