Measured floor: io_uring pieces=1/rings=2/bg=0: p50/p90/p99=1.111/1.301/1.312 ms, 11.003 GB/s (12,222,464-byte records).
The pread floor is pieces=4/rings=4: 1.298/1.380/1.422 ms, 9.415 GB/s. Both exceed the old 6-7 GB/s assumption.
With bg=2 those rows become 2.706/2.901/3.002 ms (4.517 GB/s foreground) and 2.889/3.078/3.213 (4.230).
io_uring bg=2 foreground sustains 4.218-4.517 GB/s; balanced pread rows sustain 4.230-4.507; serialized 12-piece pread falls to 0.711.
The bench does NOT count background bytes. Three equal streams imply an estimated aggregate ceiling 12.654-13.551 GB/s, not a measured ceiling.
RL0: 48,833 completions/381 steps, 128.7 records/token, 1.573 GB/token * 5.4 = 8.49 GB/s scoped physical traffic.
Thus traffic is 77% of the isolated 11.003 GB/s floor rate, or 63% of the inferred 13.551 GB/s shared ceiling; additional untracked demand can increase load.
Record p50/p90/p99=1.5/3.5/5 ms (0.5 ms bin upper bounds); 23,638/48,833=48.4% incomplete, remaining p50/p90/p99=1.5/2.5/5.
Histogram midpoint mean=1.890 ms; Little's law gives ~1.31 outstanding records (128.7*5.4*0.001890), NOT 128.7 concurrently in flight.
The excess above the floor is consistent with queueing, including engine scheduling/bursts; these measurements do not isolate queueing from service variation.
0.614/1.573=39.0%; 61.0% is an apparent overhead fraction, NOT proven unused predictions: scoped physical counts exclude real-issued fills and logical reads have different accounting.
Model (checks/prefetch-cut/model.py): R=128.7*trace_reads/196.38; rho=R*0.012222464*(1000/T)/C; q=(rho/(1-rho))/(rho0/(1-rho0)).
L50=1.111+0.389*q; L90=1.111+2.389*q ms; rho0 uses R=128.7,T0=185.185 ms. This empirical queue factor fits both measured quantiles exactly.
Assume speculative lead h=1.111 ms; exposure W=51.546*[b*max(L50-h,0)+c*L90]/[1.456*0.389+0.554*3.5].
51.546=19638.902/381 from a_ready alone; overlapping b/c bucket times are NOT added. Solve T=133.640+W self-consistently.
Use C=13.551 (equal-share bg estimate), with C=11.003 sensitivity. b,c are trace records/layer; TOP7 linearly interpolates TOP8/TOP6.
|Candidate|R/token|L50 ms|L90 ms|Exposed W ms/token|Expected tok/s|
|---|---:|---:|---:|---:|---:|
|TOP8 / sticky-union / two-ahead-spare|128.70|1.500|3.500|51.55|5.400|
|TOP7 (interpolated)|119.72|1.409-1.429|2.943-3.066|50.57-52.89|5.361-5.429|
|TOP6 (trace top8-cut)|110.74|1.344-1.374|2.541-2.728|50.07-54.07|5.327-5.443|
|sticky-rerank|117.84|1.384-1.408|2.788-2.938|52.33-55.38|5.290-5.377|
|prior-rerank|126.66|1.482-1.486|3.388-3.415|50.65-51.11|5.413-5.426|
|top10-cut|153.23|1.694-1.772|4.690-5.169|59.49-66.17|5.005-5.178|
|none|96.52|1.232-1.266|1.857-2.064|75.02-83.39|4.608-4.792|
|prior-only|98.53|1.237-1.271|1.884-2.092|75.74-84.10|4.593-4.776|
Recommendation: run PC8/PC6/PC7/PC8b with stats and gold gating; retain TOP8 unless cuts win repeatably. Predicted effects are small and capacity-sensitive.
Prior-rerank is a separate future experiment; it is not implemented by TOPK. The model includes latency feedback absent from the instant-completion trace simulator.
Uncertainty: service capacity, speculative lead, trace-to-engine scaling, correlated layer stalls and useful-byte attribution are unmeasured; forecasts are hypotheses, not engine results.
