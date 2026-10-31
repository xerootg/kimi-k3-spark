Spark capacity: use CAP80/CAP84; 88000 and 96000 MiB lack a safe margin. GB10 host/CUDA share RAM; budget units MiB.

| Budget item | MiB | Evidence |
|---|---:|---|
| Kernel MemTotal | 124609.89 | meminfo-now.txt:1; not 128 GiB usable |
| Expert host bank (80000 requested) | 79982.34 | glm-prod-run.log:475 |
| CUDA model spine | 14102.60 | glm-prod-run.log:432 |
| Main + DSA indexer KV, ctx65536, one slot | 5616 + 336 | glm-prod-run.log:455–461 |
| CUDA + host compute workspace | 2248.08 + 176.08 | glm-prod-run.log:484–485 |
| CUDA host output | 0.59 | glm-prod-run.log:453 |
| Pinned prefill staging | 3984.00 | glm-prod-run.log:471 |
| OS/runtime/other unavailable (inferred residual) | ~5766.43 | total minus above minus loaded available |
| Loaded decode MemAvailable minimum | ~12397.77 | glm-prod-run.log:520,635: 13.00 decimal GB |

At 84000/88000/96000, linear projected available is ~8398/~4398/~−3602 MiB, before valve action; rounding is small.
88000 leaves only ~302 MiB above reserve: marginal short-decode feasibility, insufficient safety for load/prefill/workload variation; 96000 cannot stay fully resident. 84000 leaves ~4302 MiB above reserve and is the conservative measurement candidate, not an unconditional OOM guarantee.
Valve source: evidence/stack-base/src/llama-moe-cache.cpp:2181–2187,2403–2411,2513,2569,2597. Host reserve=4096; sample=500 ms; release below4096, or PSI below5120, toward5120; rearm above6144 for10 s, ≤256 slots/sample; retain50% per group.
Valve starts after bank/staging/snapshot allocation (3541–3547); cannot guarantee load survival, instant reclaim, or cgroup headroom; failed reclaim disables releases.
meminfo-now.txt shows118302.18 MiB available and108558.01 MiB free: inconsistent with a resident80k bank; treat as unloaded/not representative, despite supplied description. legs-l.out:32,65,96,129 reports121140–121194 MiB AFTER stop_server (win2l.sh); not loaded headroom. No legs-hb evidence supplied.
load-log.txt contains only CPU_Mapped mapping lines; these are address-space mappings, not an additional resident235GB. Full sizes above come from glm-prod-run.log.
prod-pod-top.txt reports46740Mi; pod working set is neither whole-machine free memory nor separate GPU RAM; do not add/subtract it as another allocation.

| Cache MiB | LFU ready-hit %; bytes read/token | LRU ready-hit %; bytes read/token |
|---|---:|---:|
| 80000 | 91.467%; 2580575136 | 92.947%; 2361883744 |
| 84000 | 91.914%; 2437008928 | 93.333%; 2238525184 |
| 88000 | 92.344%; 2296904160 | 93.717%; 2115381472 |
| 96000 | 93.180%; 2032390464 | 94.381%; 1896964608 |

LFU vs80k: 84000: +0.45pp / 5.6% fewer bytes; 88000: +0.88pp / 11.0% fewer bytes; 96000: +1.71pp / 21.2% fewer bytes. Approximate pooled mean-size slots; 1024 warm/1024 eval, real predictions, ideal completion; includes speculative reads, no valve/timing; evidence/capacity-replay.txt.
