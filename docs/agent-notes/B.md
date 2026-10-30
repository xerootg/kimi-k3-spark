# B notes (SYCL kernel on nuc2)
- 11:30 PDT start. Toolchain: laptop docker intel/deep-learning-essentials:2026.1.1 + leftover sycl-bundle (runtime libs) from 10-26 study. Reading CUDA mmvq/vecdotq/quantize sources.
- 11:35 read A.md: activation is SITU (separate ggml ops), not SwiGLU. My kernel keeps the elementwise stage pluggable; will implement A's SPEC when it lands.
- CUDA facts I derived (GB10, ncols_dst=1, has_ids): IQ1_S and IQ2_XXS both take the small_k path: nwarps=4, rows_per_block=4, 128 threads/row,
  thread tid -> kbx=tid/8 (one 256-block), kqs=tid%8 (IQ1_S iqs=kqs; IQ2_XXS iqs=2*kqs). gate/up: 14 blocks -> tid<112 active; down: 12 blocks -> tid<96.
  Per row: lane x of warp 0: s=((((0+v[x])+(0+v[x+32]))+(0+v[x+64]))+(0+v[x+96])) (idle lanes 0.0f), then butterfly xor 16,8,4,2,1 (fp32 adds).
  Open FMA question (needs SASS, A?): IQ1_S `d1q*(ds.x*sumi + ds.y*delta)`: which product is fused into FFMA. Also q8_1 `amax/127.0f` and `xi/d` under fast math.

## For agent C: serve-mode record format proposal (B)
- Weights: one flat file of records, each record = gate | up | down exactly as the expert store / moebench (gate: F=3072 rows of row_size(tg,3584); up same; down: 3584 rows of row_size(td,3072)).
  IQ1_S/IQ1_S record = 6,451,200 B. Records of different (tg,td) types may be mixed; then an index file is needed.
- Index (text, one line per record, in file order): `<global_id> <tg> <td> <byte_offset>` (ggml type numbers: IQ1_S=19, IQ2_XXS=16). global_id = whatever C uses (e.g. layer*E+expert).
- Wire protocol = rrclient/moebench serve unchanged: int32 hdr[17] = {k, id1..id16} + 3584 f32 activation; reply k x 3584 f32 unweighted rows (fp32 bits exactly as the CUDA down MUL_MAT_ID output).
  ids are C's global_ids; server maps them via the index. Unknown id -> I will reply NaN rows (C must treat as miss). Please confirm or edit here.
- 11:45 first SYCL kernel (sycl/k3x, source k3x.cpp/k3x_math.h/k3x_act.h on laptop scratch, will commit to tools/bitexact/sycl/).
  Device == my host reference bit for bit (35840/35840 rows) on synthetic IQ1_S/IQ1_S, IQ2_XXS/IQ2_XXS, IQ2_XXS/IQ1_S, with all three IQ1_S FMA variants
  (no-FMA build matches host too -> IGC does NOT contract with -ffp-model=precise -ffp-contract=off). Activation still a placeholder (IEEE situ).
  First speed (synthetic, naive): k=1 0.69 ms, k=8 3.6 ms (14 GB/s) -- compute-bound on emulated dp4a; optimizing.
  Run: cd ~/k3-bitexact/sycl && sudo -n -u xero -g render env LD_LIBRARY_PATH=lib ./k3x RECORDS N TG TD check|bench|serve
- 11:50 SPEC adopted (FTZ everywhere, q8 fmul 1/127, fma(dW,inner,+0), SiTU chain). validate mode vs A dumps:
  gate and up rows (g,u) bit-identical on ALL tested values (L1 first 40 cases: 1966080/1966080 each), MUFU still IEEE placeholders,
  so h/dn differ as expected until ref/mufu tables land. Run: ./k3x ../ref/rec_L1.bin 32 19 19 validate ../ref/dumps/L1
- 11:58 B -> C: please append to C.md the per-layer remote-row histogram P(k=0..16) of your skip-6000 set (remotesim "hist" format),
  so I can run rrclient dist with it. Serving test now on nuc2:19602 (rec_L1 records); index mode next on 19600:
  ./k3x ~/k3-bitexact/remote/experts.idx ~/k3-bitexact/remote/experts.bin - - serveidx 19600 (loads 46 GB into iGPU USM).
- 11:57 WITH ref/mufu/rcp.bin (nv_rcp: table[mantissa] + exact exponent, FTZ): validate vs A dumps
  * g, u: L1 12582912/12582912, L11 7864320/7864320 bit-identical (all cases incl. edge cases 0-7).
  * down stage fed with A dumped h (K3X_HFROM_DUMP=1): dn L1 14680064/14680064, L11 9175040/9175040 bit-identical, every expert row exact.
  * Only missing: MUFU.TANH and MUFU.EX2 in h (still IEEE placeholders). A: what is the layout of ex2p.bin and the tanh table?
- RTT (Spark rrclient -> nuc2 SYCL serve, busy-poll, 300 iters): k=1 p50 1.30/p90 1.53/p99 1.74 ms; k=2 1.67/1.87/2.00; k=4 2.37/2.57/2.71; k=8 3.62/3.86/4.12
- 12:00 **SYCL kernel BIT-EXACT end to end** vs A production dumps, all four stages, every case incl. edge cases 0-7:
  L1 (IQ1_S x3): g/u/h 12582912/12582912 each, dn 14680064/14680064, expert rows 4096/4096.
  L11 (IQ2_XXS gate/up, IQ1_S down): g/u/h 7864320/7864320, dn 9175040/9175040, rows 2560/2560.
  MUFU tables: rcp.bin [mantissa]; tanh.bin base bits 0x39800000 (|x|<2^-12 -> x, >=8 -> +-1); ex2p.bin base 0x33800000 (+x),
  ex2n.bin base 0x33000000 (-x) -- bases found by probing, A please confirm. Not covered by dumps: the IQ1_S/IQ1_S/IQ2_XXS layer kind (down IQ2_XXS).
- 12:28 FOR C: bit-exact server on nuc2:19600 = ./k3x ../remote/experts.idx ../remote/experts.bin - - serveidx 19600 (K3X_HOSTW=1), loading (~85 s), log ~/k3-bitexact/sycl/serve19600.log.
  Unknown gid -> NaN rows. All gids of one request must be one record kind (one layer) -- true for the engine.
  BUG FOUND+FIXED: device USM weights above ~13-23 GB silently read as zeros on this iGPU -> weights now in host USM (same speed, A/B).
  Also fixed: kernels reading/writing host USM x/rows was not coherent (zero / stale rows); now explicit memcpy. Earlier RTT numbers superseded:
  RTT Spark->nuc2 (500-2000 iters, real rows): k=1 p50 1.44/p90 1.62/p99 1.90; k=2 1.82/1.97/2.10; k=3 2.26/2.43/2.64; k=4 2.68/2.84/3.04; k=8 4.32/4.59/4.99 ms
  k ~ skip-6000 histogram (P(k)=.0663,.1641,.2193,.2124,.1610,.0975,.0477,.0205,.0114(>=8), mean 3.1 nonzero): p50 2.27-2.34 / p90 3.39-3.59 / p99 4.46-4.74 ms
- 12:29 ALL THREE LAYER KINDS BIT-EXACT on the Arc: L83 (IQ1_S gate/up, IQ2_XXS down) g/u/h 4718592/4718592, dn 5505024/5505024, rows 1536/1536.
- 12:29 **C: the full 46 GB set does NOT work** -- the Arc iGPU returns all-zero rows when >~25-35 GB of USM is mapped
  (23 GB ok; 33 GB and 46 GB zero rows, device and host USM alike, no error). Do not use a full-set server.
  NOW RUNNING on nuc2:19600: first 3500 records of experts.idx (layers 1-38, 22.9 GB), verified real rows for first/last gid.
  Matching set file for LLAMA_EXPERT_STORE_REMOTE_SET: ~/k3-bitexact/sycl/set-3500.txt ("layer expert"). Any other gid -> NaN rows.
  Committed k3x now caps loading at K3X_MAX_GB=24 (NaN beyond). For a ranked ~22 GB set, rerun remoteset.py with --budget-gb 22.
  Server runs with oom_score_adj 1000. INCIDENT 12:18: I OOMed nuc2 (a leftover 23 GB test server + the 46 GB one);
  kubelet restarted nuc2 daemonset containers (longhorn-manager/csi, kube-router, traefik, metallb, node-exporter, nfd...), all Running again.

## Pass 2 (12:57 PDT start, deadline 16:00)
- GPU freq ORIGINAL values (restore with sudo tee): /sys/class/drm/card0/gt_min_freq_mhz=100, /sys/class/drm/card0/gt/gt0/rps_min_freq_mhz=800,
  /sys/class/drm/card0/gt/gt0/slpc_ignore_eff_freq=0, gt_max/boost/RP0=2200. Not persistent across reboot.

## C -> B (13:45 PDT, appended by agent C; details in notes/C.md)
- Please have the nuc2 server on :19600 serve `~/k3-bitexact/remote/experts-nuc2.idx` (3410 records, 22.0 GB, a subset index
  into the same experts.bin; placement for nuc2+nuc3) with your best build by 15:40, and do not restart it 15:45-16:50
  (my Spark A/B windows + the kill-a-box test, which kills the nuc3 server, not yours). If you cannot, tell me in B.md and
  I will start it myself at 15:45 with the newest committed k3x.
- nuc3 runs k3x (eb42818a) on :19600 with ~/k3-bitexact/remote/experts-nuc3.{idx,bin} (904 records, 6.5 GB), ~8 GB RSS, oom 1000.
- WARNING seen on nuc3: first validate runs after a fresh copy were WRONG (not NaN) for most cases, later runs exact.
  See C.md "Pass 2". A startup self-test in serveidx would make this safe.
- Engine side: streams are fine: rows that arrived by the deadline are used, the rest are read locally.

## C -> B URGENT (14:05 PDT): serveidx returns WRONG rows (not NaN) on nuc3; serve (slot ids) is exact
- New tool: tools/distributed/remotecanary.c (laptop repo, uncommitted; binary on the Spark ~/k3-remote-patches/remotecanary,
  dumps copied to ~/k3-remote-patches/dumps/{L1,L11,L83}). It sends As dump x with the dumps experts as global ids and
  compares the rows with dn.bin bit for bit, through the real serving path:
    remotecanary HOST PORT DUMPDIR NCASES LAYER FIRST STRIDE [K]   (L1: 1 3 28, L11: 11 5 55, L83: 83 7 55)
- nuc3, k3x md5 eb42818a (committed 0db0246), serveidx over experts-nuc3.bin with rec_L1/L11/L83 appended + index lines
  (~/k3-bitexact/remote/experts-nuc3-canary.idx): L1 48/4096 rows exact, L11 48/2560, L83 48/1536 -> only cases 0-2
  (zeros / tiny inputs) match: the rows look like zero or stale x / zero weights. Same with K3X_HOSTW=1 and without,
  and with the canary records loaded first. Same 172033-value signature as my first validate runs on nuc3.
- Same box, same binary, `k3x ../ref/rec_L1.bin 32 19 19 serve 19650` + remotecanary layer 0 first 0 stride 1:
  4096/4096 rows exact, twice. So the bug is in the serveidx path (ext_recs / x handling / repacked chunks?), not the device.
- nuc2:19600 was down when I tried, so I could not run the canary there; please run it against your nuc2 server
  before anything else: the engine would place these rows (it refuses NaN rows only) and the md5 would break.
- I see you are adding a startup probe in serveidx (working tree). Please make it compare against a dump case.
- 14:25 PASS 2 STATE (B). Current build: ~/k3-bitexact/sycl/k3x-v7 = kimi-k3-spark tools/bitexact/sycl (commit after 09dc54b, "v7").
  Bit-exact after every change: L1 4096/4096, L11 2560/2560, L83 1536/1536 rows.
  GPU min freq PINNED for these tests: /sys/class/drm/card0/gt/gt0/rps_min_freq_mhz=2200 (was 800; gt_min_freq_mhz stays 100, write refused).
    Restore: echo 800 | sudo tee /sys/class/drm/card0/gt/gt0/rps_min_freq_mhz . Effect is small (k=1 -0.07 ms wall).
  RUNNING for C on nuc2:19600 (oom_score_adj 1000): k3x-v7 serveidx, K3X_MAX_GB=34 -> first 4167 of 6967 records (layers 1..47) of experts.idx,
    matching set file ~/k3-bitexact/sycl/set-served.txt ("layer expert"). Rows are STREAMED one expert at a time in request order
    (K3X_NOSTREAM=1 = one write). Unknown gid -> NaN rows. Any all-zero row (dropped GPU submission, see below) -> NaN row.
  Start (relocatable dir = k3x-v7 + lib/ [361 MB] + tables dir [2.5 GB, K3X_MUFU=<dir>, default ../ref/mufu]):
    cd DIR && sudo -n -u xero -g render env K3X_MAX_GB=<GB> K3X_MUFU=<tables> K3X_SETOUT=set-served.txt LD_LIBRARY_PATH=lib \
      choom -n 1000 -- ./k3x-v7 <experts.idx|subset.idx> <experts.bin> - - serveidx 19600
    A subset idx (any lines of experts.idx) serves exactly those records (e.g. a ranked set). Wait for "self-check ok" + "serving on".
  Memory footprint: 2.5 GB MUFU tables on the GPU (streamed in, no host copy) + K3X_MAX_GB of records (v7 layout = +19% for IQ1_S,
    +3% IQ2_XXS vs the file) + ~0.3 GB. nuc2 now: ~37 GB used by the 19600 server. nuc3 (30 GB, ~11 GB free): use K3X_MAX_GB<=5.
  Silent zero rows: i915 EXECBUFFER2 fails with EFAULT (seen with NEOReadDebugKeys=1 PrintDebugMessages=1) when the residency set
    is very large (flaky above ~36-40 GB on the GPU; reliable failures right after killing another big server, before i915 has
    released its pages). The runtime does not report it to SYCL. Mitigations in the server: startup self-check + NaN for zero rows.
- C 13:50: experts.bin now has rec_L1/L11/L83 appended (offsets unchanged for everything before; size 46444253184) and
  experts-nuc2.idx ends with their 64 index lines (canary.idx alone too), so remotecanary works against the nuc2 server:
    ~/k3-remote-patches/remotecanary 192.168.4.249 19600 dumps/L1 256 1 3 28   (run on the Spark; L11: 11 5 55, L83: 83 7 55)
- C 14:10: thanks for v7. nuc3 (Arrow Lake-P 8086:7dd1, 31 GB, single channel, Jellyfin QSV on the same GPU) with k3x-v7:
  self-check FAILS at K3X_MAX_GB 5, 4 (passed once, then failed), 3.5; PASSES at 1, 2.5. i915 gem "system" total 33 GB,
  no leftover objects. Running now: nuc3:19600 k3x-v7, K3X_MAX_GB=2.5 (231 records + 64 canary), canary-first index
  ~/k3-bitexact/remote/experts-nuc3v7-c.idx. remotecanary through serveidx: L1 4096/4096, L11 2560/2560, L83 1536/1536 rows ok
  (48 per layer = the all-zero dump rows of cases 0-2, answered NaN by your zero-row guard: fine, the engine reads those locally).
- C plan for nuc2: at ~15:30 I will restart your nuc2:19600 server myself with the SAME command but the canary-first index
  ~/k3-bitexact/remote/experts-nuc2v7-c.idx (64 canary lines + your 4167 served records, same K3X_MAX_GB=34, K3X_SETOUT),
  wait for "self-check ok", run remotecanary, and keep it up until ~16:50. Tell me here if you object or are using it then.
- 14:50 PASS 2 FINAL (B). CURRENT BUILD = ~/k3-bitexact/sycl/k3x-v8 (kimi-k3-spark f3bcb80; md5 prefix above). Bit-exact all three kinds.
  Kernel per token (nuc2, spin wait, p50): k=1 0.72, k=2 0.78, k=3 0.99, k=4 1.18, k=8 1.93 ms
    (pass 1 / start of pass 2: k=1 0.88, k=2 1.26-1.40, k=3 1.67-1.88, k=4 2.06-2.34, k=8 3.62-4.17).
    Wins: IGC native DP4A via __builtin_IB_dp4a_ss (dot_acc was emulated with 32-bit mul+mach), 4 lanes per row on a repacked
    layout, grid tables + activation in SLM, 512-item work-groups, spin waits. GPU min freq pin: small effect.
  RTT Spark rrclient-b -> nuc2 (500/3000 iters): k=1 p50 1.02-1.22, k=2 1.30-1.64, k=3 1.47-1.87, k=4 1.71-2.10, k=8 2.8-3.3 ms;
    k ~ skip-6000 histogram (mean 3.15): p50 1.62-1.77 / p90 2.36-2.53 / p99 2.96-3.31 ms (pass 1: 2.27 / 3.4 / 4.5).
    Stream vs one-write: same p50 within noise; streaming gets the first rows out earlier for partial replies.
  RUNNING for C on nuc2:19600 (oom_score_adj 1000): k3x-v8 serveidx, K3X_MAX_GB=36 -> first 4401 of 6967 records (layers 1..50),
    set file ~/k3-bitexact/sycl/set-served.txt (4401 "layer expert" lines). ~38.5 GB resident; MemAvailable ~22 GB.
  Max servable: ~36 GB on the GPU reliably (2/2 + deploy); 40-42 GB flaky (all kernels silently return zeros: PROBEALL showed
    4872/4872 records zero in a failing run, while a 40 GB run minutes later worked). Not a kernel offset bug (every record fails
    together, and the same records work in smaller sets), not RLIMIT_MEMLOCK (the inner sudo kept the 8 GB limit in the passing
    40 GB run), not upload corruption (pinned staging + read-back checks pass). Most likely the i915/NEO residency of the whole
    allocation set per submission. Full 46 GB set = 47.8-55 GB on the GPU + 2.5 GB tables: over the 50 GB nuc2 budget anyway.
  For nuc3 (C): copy k3x-v8 + lib/ (361 MB) + the 4 MUFU tables (2.5 GB; K3X_MUFU=<dir>); footprint = 2.5 GB tables + K3X_MAX_GB
    + ~0.4 GB; nuc3 has ~11 GB free -> K3X_MAX_GB<=6. Start as in the 14:25 note (+ choom -n 1000). Wait for "self-check ok".

## Pass 3 (16:06 PDT start, report by 18:00)
- 16:11 told C (C.md): their 23:05Z restart of nuc2:19600 failed (setpriv dropped the render group). nuc2 currently has NO k3x server.
- Phase 1 findings so far: kernel 6.8.0-138 (Ubuntu 24.04), i915 in use (xe.ko present, ARL-H 7d51 needs force_probe on 6.8, untested),
  NEO 26.18.38308 + IGC 2.34.4 user-mode (bundled). nuc3 data point from C: self-check fails at >=3.5 GB (+2.5 tables) with ~11 GB free,
  nuc2 at ~34-42 GB (+2.5) with ~61 GB free -> the failure point tracks ~55-65% of MemAvailable at start, not a fixed size.
- 16:20 B WINDOW REQUEST (phase 2): ~16:50-17:40 PDT I need nuc2 with NO other k3x server, to load 40-48 GB in one or two
  processes (split-process test, then 5 cold starts of the full set). Before that I only run validate/bench (~3 GB, minutes).
  C/E: write "B window OK" (or a later slot) in your notes. I will not start phase 2 until then. If C needs 19600 meanwhile,
  start it (fix: setpriv --regid=render) and I will wait for your window to end.
- Plan to fit 46 GB under the 50 GB budget: new record layout without the v7 kb padding (+4% instead of +19%: 47.8 GB) and
  compressed MUFU tables (2.5 -> ~1.3 GB, block base + int16 residuals, exact, verified at load). Then the split test.
- 16:40 B: new build k3x-v10 (kimi-k3-spark 514a284): unpadded layout (+4% on the GPU instead of +19%) + packed exact MUFU
  tables (0.75 GB instead of 2.5 GB). Validated (vall 3 kinds). Full 46 GB set = 47.8 GB records + 0.75 GB tables = 48.6 GB.
  E: for the k8s image, k3x-v10 also cuts the current 36 GB set (4390 records) from 38.5 to ~33 GB resident.
- 16:40 B PHASE-2 WINDOW PROPOSAL: 17:05-17:50 PDT on nuc2. At 17:05 I scale deploy kimi-k3/kimi-k3-k3x-nuc2 to 0 (Es procedure
- 16:40 B: new build k3x-v10 (kimi-k3-spark 514a284): unpadded layout (+4% on the GPU instead of +19%) + packed exact MUFU
  tables (0.75 GB instead of 2.5 GB). Validated (vall 3 kinds). Full 46 GB set = 47.8 GB records + 0.75 GB tables = 48.6 GB.
  E: for the k8s image, k3x-v10 also cuts the current 36 GB set (4390 records) from 38.5 to ~33 GB resident.
- 16:40 B PHASE-2 WINDOW PROPOSAL: 17:05-17:50 PDT on nuc2. At 17:05 I scale deploy kimi-k3/kimi-k3-k3x-nuc2 to 0 (E's
  procedure; or stop the plain process if E has not cut over), then: reference rows in 3 subsets (<=18 GB each), the full set
  (48.6 GB) in one process, 5 cold starts with a full-set row check, and if needed a 2-process split. By 17:50 at the latest
  I scale back to 1 (or restart C's command). C/E: write "B window OK" or "B window NO/later" in your notes by 17:00;
  if neither answers, I wait.
- [E 16:45] B window OK (17:05-17:50), see E.md: wait for "E DONE" in E.md before scaling the pod to 0.
- 16:46 B: thanks E. I wait for "E DONE" in E.md (which also means C's A/B is over), then scale kimi-k3-k3x-nuc2 to 0, run
  my tests (fullset_check.sh), and scale back to 1 by 17:50.
- [E 17:18] E DONE (see E.md): B window may start now.
- 17:19 B WINDOW START: scaling kimi-k3-k3x-nuc2 to 0; will scale back to 1 by 17:50.
- 17:39 B WINDOW END: 6 full-set cold starts done, test servers stopped, tracepoints off, scaled kimi-k3-k3x-nuc2 back to 1.
- 17:40 PASS 3 RESULT (B): the FULL remote set (6967 records, 46.0 GB file) serves bit-exactly from ONE process with k3x-v11
  (kimi-k3-spark 4de37ce): 47.8 GB records + 0.75 GB MUFU tables on the GPU, MemAvailable ~12 GB left while serving.
  6 cold starts in a row (17:24-17:38), each with K3X_FULLCHECK: 6967/6967 records bit-identical to the host reference,
  self-check ok, no zero rows; load 83-98 s + check 34-44 s. vall (3 kinds) exact on v10/v11.
  The 6th start kept the page cache (K3X_KEEPCACHE=1) and also passed, so page-cache pressure alone does not explain the old
  failures. All old failures had the raw 2.5 GB MUFU tables (four 0.03-1.1 GB allocations) and, before 16:30, pageable upload
  staging; v11 has packed tables (0.75 GB) + pinned staging + read-back checks. i915 tracepoints in the passing run:
  3310 i915_gem_evict_node + 20922 i915_vma_unbind during load (GPU address reuse), no i915_gem_shrink.
  Start the full set: K3X_MAX_GB=49 ./k3x-v11 ../remote/experts.idx ../remote/experts.bin - - serveidx 19600 (set = experts.idx).
  For E's pod: image with k3x >= 4de37ce, memcg limit >= ~52 GB (GPU set + host runtime; page cache is dropped while loading).
