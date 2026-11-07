# Window nd: NUC distribution batched across streams (2026-11-07)

Ledger: docs/experiments.md, entries "NUC distribution, batched across streams" (two). Patch 0049, NUC server
tools/glm53/flash/f53s.cpp.

- `exactness.txt`: NUC rows against the GB10 (flashrow cases, over the wire, and 32,768 serving rows replayed).
- `flashrow.cpp`: GPU reference rows from the production ggml CUDA kernels (8 records, random inputs, T tokens).
- `flashreplay.cpp`: replays a `LLAMA_EXPERT_STORE_REMOTE_DUMP` file through the GPU kernels, row by row.
- `mbench.py`: multi-stream client with distinct prompts per stream (solo / multi modes).
- `ndlib.sh`, `win-nd0.sh` .. `win-nd9.sh`: the window scripts (all legs under `flock window.lock`, 4 GiB watchdog).
- `nucserve.sh`: starts f53s on a NUC with a 2 GiB MemAvailable watchdog; `mkown.py`: ownership sets, f53s index and
  the O_DIRECT dd lines that stream each box's records from the Spark's store.
- `ndclient.py`: wire-level check of a running f53s; `ndcmp.py`: md5/throughput comparison of results.jsonl legs.
- `windows-nd0-nd6.txt`, `windows-nd7-nd8.txt`, `windows-nd9.txt`: runner output; `results-*.jsonl`: every measurement;
  `server-remote-stats-*.txt`: the engine's remote counters per leg; `f53s-logs-*.txt`: the NUC servers' logs.

Windows: nd0 baselines (production binary, 1M context); nd1 killed by the watchdog (a build ran beside it);
nd2 first remote run (fixed caps, bytes differ, -37 % at N=2); nd3 which knob changes the short request, 16k gold with
the tier on; nd4 discard test and adaptive caps (valve active); nd5 ctx 262144 (valve idle), off/on without prompt
isolation; nd6 the dump replay and prompt isolation (off/on identical at N=2/4); nd7 per-N adaptive caps, N=8;
nd8 the same in reversed order for variance; nd9 the faster NUC row kernel.
