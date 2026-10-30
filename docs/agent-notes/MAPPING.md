# Notes to ledger mapping

Ledger titles are shortened: "10-28 section" = "Spark: bit-exact remote experts on the Intel NUCs, the CPU latency hold, the disk (2026-10-28)";
"10-29 section" = "Spark + nuc2: where the remote round trip went, and four fixes (2026-10-29, workflow k3-remote-marshaling)".

- A.md: SiTU instead of SwiGLU, spec read from the SASS: already in ledger (10-28 section)
- A.md: MUFU RCP/TANH/EX2 tables verified over all 2^32 inputs: already in ledger (10-28 section)
- A.md: scalar and AVX2 CPU reference, bit-exact on all stages and three layer kinds: already in ledger (10-28 section)
- A.md: production image SASS identical to the native build the dumps came from: already in ledger (10-28 section, "read from the sm_121 SASS"; the image comparison itself is not stated there)
- B.md: first SYCL kernel, bit-exact against the dumps: already in ledger (10-28 section)
- B.md: kernel speed (native dp4a, repacked layout, SLM, spin waits): already in ledger (10-28 section, "Kernel speed"); the full per-k table is added in "Both ends: the Arc serve path on nuc2 and nuc3, wrong rows, and what the memory limit does"
- B.md: round trip from the Spark by build, GPU minimum frequency pin, streamed rows against one write: added
- B.md: zero rows above ~13-35 GB, host USM, caps, EFAULT investigation: added (detail; the headline is in the 10-28 section "iGPU memory")
- B.md: full 46 GB set, 6 cold starts with FULLCHECK on k3x-v11: already in ledger (10-28 section, "iGPU memory"); the KEEPCACHE start contradicts its page-cache statement, named in the added entry
- B.md: serveidx wrong rows on nuc3 and the remotecanary tool (with C.md): added
- C.md: remote set choice (skip table) and placement replays: added ("Spark: which experts go to the remote boxes, and the engine's first runs against a stand-in server")
- C.md: window 1 with the stand-in (remote off, shadow, plumbing-only): added (same entry)
- C.md: window 2 interleaved A/B, tier on/off, nuc3 killed mid-run: already in ledger (10-28 section, "Native A/B, interleaved")
- C.md: multi-box engine fault tests (kill, SIGSTOP, breaker): added ("Both ends: the Arc serve path ...")
- C.md: nuc3 wrong rows on first runs, self-check limits on nuc3: added (same entry as B's)
- D.md: ASPM L1, deep idle / cpu_dma_latency, rx-usecs, NIC queue on an LP-E core: already in ledger (10-28 section, "Network latency, Spark to nuc2"); the D notes do not contain the ledger's ~100 us and ~70 us figures
- D.md: gro_flush / napi_defer and per-cpu pm_qos tried and reverted: already in ledger only as "QoS: no effect" at best; the notes record no result, so no entry was added
- E.md: nuc2 expert server as a k8s pod, memory limit breaks the self-check, RTT after cut-over: added ("Both ends: the Arc serve path ...")
- E.md: NIC settings as the kimi-k3-nic-lowlat DaemonSet: already in ledger (10-28 section, "Deployed")
- I.md: CPU half in the engine, legs a/b/c/d: already in ledger (10-29 section, "I, integration")
- I.md: uncore minimum frequency and EPP changes: added ("nuc2 CPU frequency knobs for the CPU half"), result not recorded
- M.md: RC6 and forcewake on nuc2, engine round trip breakdown, legs A and B: already in ledger (10-29 section)
- P.md: head-to-head A/B/C of forcewake server, CPU half and split, 6 legs: added ("the forcewake server against the CPU half"), conflicts with the 10-29 section
- P.md: CPU-stall cause (preemption by node threads), K3X_RT: added (same entry), conflicts with "cause unknown"
- P.md: k3cs CPU-only server and placement replays: added (same entry)
- P.md: queued tests (SM clock lock, --threads sweep, concurrent clients, audit 4 and 10): not run in the notes; mentioned in the P entry as "not measured", no separate entry
- Q.md: item 8 replay, promoted against left in place: added ("Spark: zero-record skip, speculative reader knobs and plan mode")
- Q.md: engine A/B with the zero-record skip, window 1: added (same entry), interrupted before a result
- S1.md: zero-copy host-USM serveidx and its round trips: already in ledger (10-29 section, "S1"); the gapped / graph / memwarm numbers are added in the S1 entry
- S1.md: host-USM coherence probes, spin kernel hang, command-list modes: added ("nuc2 iGPU zero copy: what host USM allows")
- S2.md: early post, futex doorbell, one sendmsg: already in ledger (10-29 section, "S2")
- S3.md: AVX-VNNI bit-exact CPU kernel: already in ledger (10-29 section, "S3")
- V.md: independent bit-exactness verification and negative control: already in ledger (10-29 section, "Verification (V)")
- V.md: flag that M's forcewake leg beat the CPU half: already in ledger (10-29 section, "Caveat"); resolved by P's legs, see the P entry
