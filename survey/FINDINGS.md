# Findings from the owner run

Evidence: `results/20261030T074347386706Z/`, checkout `survey-run.log`.
All numbers below are milliseconds, nearest-rank percentiles of individual
observations. k is the number of experts returned per request. Full k sweeps,
counts, maxima, and diagnostics are in [MEASUREMENTS.md](MEASUREMENTS.md).
Nothing was rerun on hardware during this second pass.

At the owner's planning budget of about **3.1 ms/layer**, network latency is
small enough to consider resident offload, but k and arithmetic mode dominate.
4.9 tok/s across 75 MoE layers gives **2.72 ms mean spacing** (1000/4.9/75),
not 3.1 ms; neither average establishes the actual post-to-needed deadline.
The original run used an assumed 3.0 ms budget. The tables preserve its raw
measurements; validate overlap against a lower-tail real lookup window before
making a throughput claim. Finite replies alone do not prove bit exactness.

| Hop / mode | k | p50 ms | p99 ms | n | Interpretation |
|---|---:|---:|---:|---:|---|
| nuc2->nuc3/ethernet | 0 | 0.183 | 0.340 | 1000 | Empty ping-pong; no expert compute |
| nuc2->nuc3/ethernet | 1 | 0.261 | 0.569 | 1000 | Wire-size echo; no expert compute |
| nuc2->nuc3/ethernet | 8 | 0.636 | 1.039 | 1000 | Wire-size echo; no expert compute |
| nuc2->nuc3/thunderbolt0 | 0 | 0.184 | 0.355 | 1000 | Empty ping-pong; no expert compute |
| nuc2->nuc3/thunderbolt0 | 1 | 0.270 | 0.565 | 1000 | Wire-size echo; no expert compute |
| nuc2->nuc3/thunderbolt0 | 8 | 0.625 | 1.344 | 1000 | Wire-size echo; no expert compute |
| nuc2/exact=0/resident/loopback | 1 | 0.595 | 1.117 | 1000 | Local TCP + resident GPU; excludes LAN |
| nuc2/exact=0/resident/loopback | 8 | 3.288 | 4.455 | 1000 | Local TCP + resident GPU; excludes LAN |
| nuc2/exact=1/resident/loopback | 1 | 11.887 | 13.327 | 627 | Local TCP + resident GPU; excludes LAN |
| nuc2/exact=1/resident/loopback | 8 | 74.850 | 81.947 | 92 | Local TCP + resident GPU; excludes LAN |
| nuc3->nuc2/ethernet | 0 | 0.292 | 0.567 | 1000 | Empty ping-pong; no expert compute |
| nuc3->nuc2/ethernet | 1 | 0.273 | 0.543 | 1000 | Wire-size echo; no expert compute |
| nuc3->nuc2/ethernet | 8 | 0.556 | 1.040 | 1000 | Wire-size echo; no expert compute |
| nuc3->nuc2/thunderbolt0 | 0 | 0.195 | 0.406 | 1000 | Empty ping-pong; no expert compute |
| nuc3->nuc2/thunderbolt0 | 1 | 0.273 | 0.533 | 1000 | Wire-size echo; no expert compute |
| nuc3->nuc2/thunderbolt0 | 8 | 0.552 | 1.043 | 1000 | Wire-size echo; no expert compute |
| nuc3/exact=0/resident/loopback | 1 | 0.557 | 1.358 | 1000 | Local TCP + resident GPU; excludes LAN |
| nuc3/exact=0/resident/loopback | 8 | 4.030 | 6.161 | 1000 | Local TCP + resident GPU; excludes LAN |
| nuc3/exact=1/resident/loopback | 1 | 11.450 | 14.026 | 648 | Local TCP + resident GPU; excludes LAN |
| nuc3/exact=1/resident/loopback | 8 | 82.085 | 93.248 | 83 | Local TCP + resident GPU; excludes LAN |
| spark->nuc2/ethernet | 0 | 0.104 | 0.206 | 1000 | Empty ping-pong; no expert compute |
| spark->nuc2/ethernet | 1 | 0.297 | 0.394 | 1000 | Wire-size echo; no expert compute |
| spark->nuc2/ethernet | 8 | 0.903 | 1.137 | 1000 | Wire-size echo; no expert compute |
| spark->nuc2/exact=0/cold/prototype | 1 | 7.913 | 9.671 | 483 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc2/exact=0/cold/prototype | 8 | 56.300 | 71.528 | 58 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc2/exact=0/lru/prototype | 1 | 0.898 | 1.230 | 1000 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc2/exact=0/lru/prototype | 8 | 3.530 | 5.239 | 606 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc2/exact=0/resident | 1 | 0.917 | 2.023 | 1000 | Spark CPU client + LAN + resident GPU |
| spark->nuc2/exact=0/resident | 8 | 3.493 | 4.613 | 600 | Spark CPU client + LAN + resident GPU |
| spark->nuc2/exact=1/cold/prototype | 1 | 18.559 | 19.977 | 206 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc2/exact=1/cold/prototype | 8 | 123.749 | 135.501 | 22 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc2/exact=1/lru/prototype | 1 | 11.015 | 11.971 | 340 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc2/exact=1/lru/prototype | 8 | 74.614 | 81.894 | 42 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc2/exact=1/resident | 1 | 11.448 | 12.720 | 332 | Spark CPU client + LAN + resident GPU |
| spark->nuc2/exact=1/resident | 8 | 72.078 | 77.490 | 44 | Spark CPU client + LAN + resident GPU |
| spark->nuc3/ethernet | 0 | 0.104 | 0.275 | 1000 | Empty ping-pong; no expert compute |
| spark->nuc3/ethernet | 1 | 0.274 | 0.669 | 1000 | Wire-size echo; no expert compute |
| spark->nuc3/ethernet | 8 | 0.956 | 1.115 | 1000 | Wire-size echo; no expert compute |
| spark->nuc3/exact=0/cold/prototype | 1 | 7.205 | 8.647 | 521 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc3/exact=0/cold/prototype | 8 | 48.711 | 53.690 | 70 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc3/exact=0/lru/prototype | 1 | 0.748 | 1.094 | 1000 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc3/exact=0/lru/prototype | 8 | 4.058 | 5.025 | 672 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc3/exact=0/resident | 1 | 0.747 | 1.021 | 1000 | Spark CPU client + LAN + resident GPU |
| spark->nuc3/exact=0/resident | 8 | 4.109 | 6.076 | 580 | Spark CPU client + LAN + resident GPU |
| spark->nuc3/exact=1/cold/prototype | 1 | 17.795 | 19.950 | 214 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc3/exact=1/cold/prototype | 8 | 128.570 | 133.138 | 21 | O_DIRECT read + synchronous repack + compute + LAN; prototype |
| spark->nuc3/exact=1/lru/prototype | 1 | 11.818 | 13.040 | 322 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc3/exact=1/lru/prototype | 8 | 84.480 | 91.104 | 36 | Eight-slot prototype, all hits; compute + LAN |
| spark->nuc3/exact=1/resident | 1 | 11.610 | 12.809 | 326 | Spark CPU client + LAN + resident GPU |
| spark->nuc3/exact=1/resident | 8 | 84.089 | 89.341 | 37 | Spark CPU client + LAN + resident GPU |

Spark->nuc2 empty RTT p50 is **104 us**, while the full wire echo is 297 us
at k=1 and 903 us at k=8. These are round trips, not one-way delay. The
laptop is roughly 1,800 miles away (Seattle vs Fort Collins); these clients ran
on cluster hosts, so the measurements do not include the laptop WAN path.
No bulk data should be staged from the laptop. MUFU tables already reside at
`~/glm53-mufu` on both NUCs; only small executables and scripts are staged.

NUC2 fast resident loopback is **0.594 ms at k=1 / 3.288 ms at k=8**.
Spark resident fast p99 is 2.022/1.752/2.449 ms at k=1/2/3; k=4 is
3.155 ms and k=8 4.613 ms. Small-k fast results suggest useful headroom;
fast arithmetic is not certified exact offload. Do not add separate p99s or
subtract medians from independent runs to claim isolated stage costs.

Exact mode 1 costs about **12 ms for one expert** on these NUCs; nuc2
loopback p50 reaches 74.850 ms at k=8. It is unusable at this budget even
before disk misses. Fast exact mode 2, owner-reported at **1.2–1.9 ms/expert
on the laptop iGPU**, is in progress and was not measured by this run. Its
NUC performance, k scaling, correctness and tails remain open. This kit still
sweeps modes 0 and 1; no mode-2 capability is inferred from mode-1 numbers.

Cold prototype fast p50 is 7.913/56.300 ms on nuc2 and 7.205/48.711 ms
on nuc3 at k=1/8. This synchronous implementation cannot fit even k=1.
The data combines disk, CPU repack, GPU and LAN, so it does not identify an
NVMe-only bottleneck. LRU-hit counters are zero reads on both hosts; cold
counters are positive for every session. These are prototype results, not a
production disk tier or a realistic cache hit-rate trace.

Bulk (one 128 MiB transfer each, client->server): Spark->NUCs 2.301/2.305
Gbit/s; NUC Ethernet 10.115/10.130 Gbit/s; TB forward/reverse 12.132/13.273
Gbit/s. TB improved bulk bandwidth here, with no consistent small-message
latency advantage. These are single transfers, not sustained distributions.

## Missing measurements and limits

- NVMe-only latency: all 26 cases on each host are absent, including whole
  record, piece sizes/QDs, and eight-record batches. Failed builds are not
  evidence of slow disks. Rerun `./survey/run.sh nvme` after this fix.
- Native g53x and CPU g53c benches: all 24 cases failed before producing
  timings. Rerun `./survey/run.sh igpu` with a fresh build, without --skip-build.
- Spark end-to-end is **present for every host/mode/tier/k combination**,
  not missing. Many cases hit the 4 s cap: counts range down to 22 and
  p99 is weak. Exact loopback also has fewer than 1,000 samples. Use longer
  documented caps for tails; no incomplete data has been imputed.
- Disk tiers have only Spark-client measurements; local disk-stage breakdown,
  isolated repack timing, realistic routing/cache hit rate, asynchronous
  prefetch and production integration remain unmeasured.
- No GLM CPU server exists here, so no CPU serveidx/e2e comparison is available.
- No Spark server log or remote trace was supplied: actual request deadline,
  overlap, lookup waits, live decode throughput and scheduling remain unknown.
- Mode 2 on cluster hardware, idle-gap effects, concurrent multi-host offload,
  multi-layer experts and interference need separate measurements.

See [DIAGNOSIS.md](DIAGNOSIS.md) for the complete error reconciliation and fixes.
