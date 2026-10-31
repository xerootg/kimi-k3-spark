# GLM owner planning

Run entirely on the Spark. `place.py` uses Python's standard library and does no
network I/O. `placement.sh` is an explicit LAN deployment procedure; it was not
executed in this checkout. NUC stores, store sidecars, MUFU files and built
`/srv/glm-5.3/bin/g53x` must already exist. Keep the production engine stopped
while collecting a separate model run, so the 80 GB cache fits.

```bash
# Offline assumption, no hardware or trace needed:
python3 tools/glm53/place.py --out placement
# Real trace; metadata reads only the header and index, never weight payload:
python3 tools/glm53/place.py --trace decode.bin --eval heldout.bin \
  --store /srv/glm-5.3/store/glm53-q2-v2.k3xs --out /srv/glm-5.3/placement
# Replay a manually edited owner table:
python3 tools/glm53/place.py --trace decode.bin --owners placement/owners.csv --out replay
# Full collection -> planning -> LAN copy -> start -> fetch actual served sets:
PFDUMP=$HOME/llama-k3-bs/build/bin/llama-k3-pfdump PROMPT_FILE=$HOME/glm-prompts.txt \
  bash tools/glm53/placement.sh -m /path/to/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf \
  [the production model/cache CLI arguments from glm53-spark.yaml]
```

The bracketed arguments are a placeholder: that YAML is absent from this checkout.
Use its 80000 MiB cache, 65536 context, GPU placement and thread arguments.
Build/update the collector using [../k3-pfdump/README.md](../k3-pfdump/README.md)
against the engine with patches 0001–0006 and the GLM predictor. Patch 0003
installs the source; the Dockerfile applies patches rather than copying tools.
The CMake target/binary is `llama-k3-pfdump` (a local `k3-pfdump` may be an alias).
Sanity-check without loading a model:

```bash
~/llama-k3-bs/build/bin/llama-k3-pfdump --check decode.bin
```

Expect 75 real layers (3..77), with predictor pairs normally only for 4..77;
layer 78 is MTP. Counts should match the requested decode length. Precision is
measured only on same-step, same-destination-layer pairs.
The script sets the exact store, split, byte-stride and predictor environment;
remote tier is unset during capture. Do not set PREFETCH_DUMP: alternative K3
predictors are unnecessary. At 4.9 tok/s 2048 decode tokens take about 7 minutes
plus model load/prefill (callback overhead can make it longer). Expect roughly
0.5 MB/token, 1 GB for 2048: 75 layers of predictor and routing float vectors,
top-8 integers and headers. `--ignore-eos` forces all 2048 decode steps; otherwise EOS may shorten it; use multiple representative
prompts and an independent `--eval` run for a useful out-of-domain estimate.

Trace formats are explicitly selected: default pfdump is little-endian
`step,layer,kind,n` int32, then n words; kind 3 is real top-8, kind 1 is predictor
float32 scores tagged with the destination layer. The parser drops incomplete
75-layer tokens, rejects malformed framing/IDs and reports discarded tokens.
`--format fill` reads `LLAMA_EXPERT_STORE_TRACE` int32 `[layer,n,ids...]` instead,
infers tokens from layer wrap and assumes predictor precision. This trace hook
is in unsplit `fill()`, not deployed split `fill_early()`: enabling that env on
the current server is insufficient. Do not disable split just to infer latency.
`REMOTE_TRACE` is a different request-timing diagnostic, not routing hotness.

Outputs: `owners.csv` uniquely covers 75x256 experts, `spark.set` is the desired
hottest 6859, `nuc{2,3}-ram.set` is the next hottest fitting 44/18 decimal GB
(after 6 GB headroom per host), and `*-ssd.set` partitions the remainder by
frequency-weighted bytes / 5.7 or 4.6 GB/s. There is no duplicated remote owner.
Spark NVMe retains the whole store as fallback. All `.set` files contain plain
`layer expert` lines, without headers. Given `--store`, `.idx` files contain
`gid tg td offset` for g53cs; v1/v2 offsets use the index's actual stride, never
12091392 as a v2 stride. Use metadata from each NUC's identical v2 store, not
the Spark v1 file. Without metadata all layers assume 12091392 B; mixed Q4
layers make this an estimate. Header/index copies work for offline planning.

For CPU service the equivalent is `K3X_CPU_CPUS=<available cores> K3X_MAX_GB=44
K3X_SETOUT=served.set g53cs nuc2-ram.idx STORE 19653`; use 18 on nuc3. CPU
latencies are unmeasured here; the simulator's GPU timing model does not apply.
Both servers may reject unsupported records. Deploy fetches `K3X_SETOUT` after
load: remote.set = that host's `*-ram.set` intersected with its fresh served.set.
ONLY these RAM-only intersections go in REMOTE_SET, in host-list order; never
use RAM+SSD ownership or the full disk-enabled SETOUT advertisement. Confirm
the selected RAM records actually loaded (window i: 3588 resident records).
SETOUT precedes socket listen; verify server.log/listener before starting engine.
The script refuses an existing listener or served file; stop old servers and
archive their served files before a deliberate redeployment. SSH/scp are
Spark-to-NUC, never laptop relays. Merge `engine.yaml` into glm53-spark.yaml and
mount the output directory at `/srv/glm-5.3/placement`, or source `engine.env`
for a native run. If PLACEMENT_OUT differs, adjust the generated mount paths.

Limits that affect the estimate:

* The engine cannot pin `spark.set`. Replay warms an LRU from it, then allows
  eviction and local fallback to change residency. Real cache groups and
  second-chance eviction can differ; owner means intended preferred tier.
* Global REMOTE_ROWS=2 caps both NUCs together, hence each NUC <=2. Static
  placement cannot guarantee <=2 selected experts for arbitrary future tokens;
  overflow always reads on Spark. No remote server receives overflow work.
* One-layer predictor precision defaults to 85.3%, but missing scores use
  independent Bernoulli overlap, not a measured GLM routing distribution.
  Default Zipf exponent 1 is not calibrated to the 91.9% lookup hit log; those
  hits include prefetch. Window=5 predicted Spark misses are counted including
  wasted reads; demand reads and all reads are reported separately. Default
  training/evaluation split is chronological halves; explicit --eval is better.
* Resident p50 service interpolates Spark->nuc2 .917 ms at k=1 to 3.493 at k=8;
  p99 scenarios use 2.023/1.752 ms at k=1/2. nuc3 uses the same conservative
  surrogate. These full round trips already include the .104 ms empty RTT.
  Deadline 2200 us = k=1 p99 +177 us, below 2721 us mean layer spacing.
  Mean spacing is not a measured post-to-needed window; measure lower tails.
* Fast mode 0 is explicit in deploy.sh. These latencies do not certify exact
  arithmetic. Mode 1 measured ~11.4 ms/expert and cannot fit; mode 2 is not
  measured on NUCs here. Confirm the chosen kernels before production use.
* `ssd_ideal` assumes a NEW predictor weight-prefetch channel and bounded
  per-layer SSD bandwidth, plus cold-read fallback. The current wire protocol
  sends only actual inputs; current disk cold prototype measured ~7-8 ms k=1.
  Merely enabling K3X_DISK_TIER does not implement that channel. Deployment
  therefore uses resident sets only. Its SETOUT with disk enabled lists all
  supported SSD records on both servers, defeating exclusive SSD ownership;
  use explicit owner filters when implementing disk prefetch integration.
* Throughput fields are sensitivities: no I/O credit vs serial Spark reads at
  assumed 13 GB/s saved from a 4.9 tok/s baseline. Much I/O already overlaps;
  neither is a measured speedup or confidence interval. Actual compute,
  cache pollution, stalls and concurrent clients require engine A/B.

K3 reference: docs/experiments.md skip 6000, 6967 records/46 GB gives 6.80 ->
5.00 misses/layer before row caps, but code2 was in-sample. The held-out pool
improved 6.80 ->6.44 with 22+6.5 GB. No binary K3 traces exist in this checkout;
validation here is synthetic, not a reproduction of those numbers.
