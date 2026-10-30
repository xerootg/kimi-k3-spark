Implemented GLM placement and replay in `tools/glm53/place.py` (stdlib).
Usage and the exact collection/deployment environment: `tools/glm53/PLACEMENT.md`.
One-command Spark workflow: `tools/glm53/placement.sh` (see its required inputs).
No network, commits, pushes, or hardware runs were performed.

Outputs in `placement/` are a synthetic example, NOT a measured GLM placement:
- owners.csv: all 19,200 pairs, frequency, bytes, unique preferred owner.
- spark.set, nuc2/3-ram.set, nuc2/3-ssd.set: plain layer/expert lines.
- engine.env / engine.yaml: ordered hosts/actual served-set paths, deadline, rows.
- deploy.sh: Spark-to-NUC LAN copy, start g53x, fetch K3X_SETOUT, check readiness.
- report.json: baseline, resident replay, hypothetical SSD-prefetch replay.
With --store, place.py also emits g53cs IDX files with actual v1/v2 offsets.
No store index is available here; no CPU offsets were invented.
Use --trace decode.bin --eval heldout.bin --store STORE for real placement;
use --owners owners.csv to replay an existing or edited owner table.

Assumptions: 400 synthetic tokens, Zipf exponent 1, seed 53, chronological
200-token train/eval halves; missing predictor scores assume 85.3% precision.
This is not calibrated to GLM's 91.9% prefetch-inclusive hit statistic.
Spark: 6859 desired slots. NUC record budgets: 44/18 decimal GB, leaving
6 GB each from the stated 50/24 GB pools. Most-layer 12091392 B assumed.
Counts: Spark 6859; nuc2 RAM 3638; nuc3 RAM 1488; SSD 4932/2283.
Global REMOTE_ROWS=2 also enforces each NUC <=2; overflow falls back on Spark.
Deadline 2200 us = resident k=1 p99 2023 +177 us, below 2721 us mean spacing.
The actual post-to-needed window still needs measurement.

Synthetic resident replay: ~141.3 Spark reads/token (20.4 demand, 120.8
prefetch), ~61.2 remote rows/token; each NUC rows p50=0, p99=2, max=2.
No mean-service deadline fallback in this scenario. Throughput sensitivity:
4.90 tok/s with no credit for saved overlapping I/O; serial-I/O credit is an
optimistic 8.91 tok/s ceiling, not a predicted measured speedup. Baseline
reads are 240.0/token; the SSD ideal serial-I/O ceiling is 13.83 tok/s.
SSD-prefetch hypothetical: ~98.3 Spark reads/token, ~29.3 fallback rows/token.

Integration limits: the engine cannot pin spark.set; replay warms an LRU from
that desired set, then allows eviction. Cache groups/second chance may differ.
The deployed split fill does NOT execute LLAMA_EXPERT_STORE_TRACE's hook;
collect graph pfdump kind-3 IDs / kind-1 predictor scores instead. At 2048
tokens expect ~7 minutes plus load and callback overhead, roughly 1 GB.
The wrapper preserves production model/cache CLI args supplied by the owner;
glm53-spark.yaml itself is absent here, so its cache flags cannot be verified.
Current wire protocol has no predictor weight-prefetch message. SSD ownership
is planned, but live deployment uses RAM sets only. Disk SETOUT otherwise
advertises all supported SSD records on both hosts, losing unique ownership.
Fast mode 0 timings/deploy config do not establish bit exactness; measured
exact mode 1 (~11.4 ms/expert) does not fit; NUC mode 2 remains unmeasured.
K3 binary traces are absent: validation is synthetic, not a skip-table replay.

Tests: `python3 tools/glm53/test_place.py`
Last output lines: `Ran 4 tests in 0.814s` then `OK`.
Also ran: `python3 tools/glm53/place.py --out placement`;
`--tokens 8 --owners placement/owners.csv` replay; Python compile and shell syntax checks.
The generated scripts were syntax checked, never executed over SSH.
