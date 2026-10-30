# Error reconciliation

Source run: `results/20261030T074347386706Z/`; `survey-run.log` is analyzer
stdout, not an independent host execution transcript. Original raw files
are preserved. `MEASUREMENTS.md` is the updated analyzer output.

| Files / cases | Count in analyzer's 76 | Evidence and cause | Fix |
|---|---:|---|---|
| nuc2-nvme.jsonl, records 1/8, whole + 3 piece sizes × 4 QDs | 26 | rc 127; glm_readbench does not exist | Ship static pread binary; gate sweep on executable preflight |
| nuc3-nvme.jsonl, identical cases | 26 | Same rc 127 | Same |
| nuc2-raw/g53x-{0,1}-{1,2,4,8}.log | 8 | SIGABRT (-6), SYCL reports no GPU | Use sudo render group and explicit runtime/MUFU/math environment, matching successful serveidx |
| nuc3-raw/g53x-{0,1}-{1,2,4,8}.log | 8 | Identical no-GPU exception in all logs | Same |
| nuc2-raw/g53c-0-{1,2,4,8}.log | 4 | rc 1, libm GLIBC_2.43 symbol version missing | Build fully static CPU executable; rebuild old artifacts |
| nuc3-raw/g53c-0-{1,2,4,8}.log | 4 | Identical loader failure | Same |
| Total | 76 | 52 + 16 + 8; no timing samples in these failed cases | Offline regression checks the census |

Both `*-nvme-build.log` files contain two `g++: command not found`
diagnostics (liburing attempt and pread fallback). Each also contains a
structured remote-failed row. Those **two additional build failures are not
included in the analyzer's 76**, because it scanned JSONL only. The repeated
26 missing-executable errors per host are consequences of these build failures,
not 52 separate disk faults. Compiler availability is now optional on the NUC:
a native liburing build may replace the shipped static pread executable only
when compilation succeeds. No software installation is required.

Every `*session.jsonl` has READY and final counters: 12 successful sessions.
Resident and LRU sessions have zero reads; cold sessions have 4,292/1,955 reads
on nuc2 (mode 0/1) and 4,795/1,947 on nuc3. All 96 Spark e2e distributions
exist (2 hosts × 2 modes × 3 tiers × 8 k). The old analyzer did not flag
these as missing; its aggregate presence check was too coarse to detect a
missing host/mode/k on future partial runs. It now checks each combination.

Time-cap status rows are expected censoring, not failures. Four seconds
includes ten warmup requests and cannot collect 1,000 exact/cold samples.
The original 65 s native-bench cap also allowed 12 × 65 s per NUC before
loopback, exceeding the 480 s section watchdog once bench failures are fixed.
The default is now 20 s/case; larger caps require larger section/total bounds.
Timeout stdout and stderr now survive in bench log files.

Metadata has 27/39/37 nonzero commands on Spark/nuc2/nuc3, separate from the
76 timing errors. All ethtool failures explicitly say Operation not supported:
coalescing/EEE on virtual, loopback or TB devices; loopback driver lookup;
Spark also lacks coalescing support on enP7s7 and wlP9s9. These cannot establish
NIC settings. NUC fuser returns 1 with empty output: no holder was reported
for /dev/cpu_dma_latency at that moment. This does not imply a service failure.
Metadata diagnostics remain unmodified and are not converted into timings.

The recurring layer-78 skip warning in GPU server logs is type 10/10 outside
the supported engine types. Requests use only supported L3 IDs (17/17/18),
so this warning did not invalidate these replies. Unsupported production
layers still need arithmetic support or local fallback.

MUFU tables are verified on all three filenames on both remote hosts and
linked per run. The obsolete laptop MUFU prerequisite was removed; large
tables never traverse the laptop WAN. No permission/host tuning was changed.
GPU bench success still requires hardware verification: render permissions
are strongly implicated by the same binary's successful render-group sessions,
but no new remote run was authorized here.
