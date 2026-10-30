# GLM-5.3 pooled inference on an A10 fleet

Proposal: request a staged fleet allocation, ending at **32 VMs for 7 days** (estimate),
to test whether CUDA expert parallelism makes a larger quant practical without NVLink.
This is research capacity, not a production throughput promise. No hardware was run for this proposal.
GB means decimal bytes; timings and capacities labelled E are estimates or derived planning values.

Sources used below (all local):
- [A: earlier A10 sketch](../evidence/roadmap-a10-section.md): K3 sizing only; do not reuse its model size for GLM.
- [P: Spark profile](../evidence/spark-profile.md): measured decode budget and traffic.
- [L: experiment ledger](experiments.md), [S: current GLM status](GLM-STATUS.md): implemented engine and measured gates.
- [F: LAN findings](../evidence/FINDINGS.md): TCP echo and Intel compute latencies; no datacenter measurement.
- [H: GH200 plan](../evidence/gh200-glm-plan.md): capacity calculations and explicitly estimated GLM throughput.
- [I2](../tools/glm53/results/inventory-UD-Q2_K_XL.txt),
  [I4](../tools/glm53/results/inventory-UD-Q4_K_XL.txt),
  [I5](../tools/glm53/results/inventory-UD-Q5_K_XL.txt): quant bytes and per-token traffic.
- [R: placement limits](../tools/glm53/PLACEMENT.md),
  [M: CUDA math specification](../tools/glm53/SPEC-glm.md): ownership, row cap, exact arithmetic.
- U: owner's supplied hardware assumptions: one A10, 24 GB VRAM, approximately 250 GB host RAM,
  no NVLink, sm_86, PCIe 4 x16 approximately 25 GB/s host-to-GPU; NIC/storage unknown.

## Tier 1: one VM, a necessary control

Use UD-Q2_K_XL, the deployed quality floor (S), with the active spine on GPU and a partial
host expert cache; keep a complete expert store on the VM's local disk. No smaller quant
is qualified by this evidence. Q4/Q5 increase both storage and expert-read cost (I2/I4/I5).
The full Q2 GGUF is 253.869 GB (I2); the v1 store is about 238.17 GB including MTP (L).
Active routed experts are 234.671 GB; total minus active experts is 19.198 GB including
MTP, or about 15.44 GB after removing the unused 3.76 GB MTP inventory (E, I2).
Thus “238 GB experts fit in 250 GB RAM” ignores spine copies, OS and execution buffers.
Even if spine page cache is dropped after upload, available RAM must be measured, not inferred.

Initial host budget E: 210 GB expert slots + 20 GB spine/load copies + 20 GB OS, pinned staging
and reserve = 250 GB. Active-only store leaves 24.671 GB outside cache (10.5%); retaining
the complete v1/MTP store leaves approximately 28.17 GB outside (E, I2/L).
That is disk-backed capacity, not bytes read per token. Under uniform active routing,
expected demand is 7.333 × (24.671/234.671) = 0.771 GB/token (E); hot routing may reduce it,
prediction waste may increase disk traffic. Measure a real GLM hit curve; residency is not hit rate.
Use v1 and byte-stride slots initially: Spark v2 adds padding and lost about 1.7% (L/S).

GPU budget E: approximately 15.44 GB active spine + 4 GB short-context KV/workspace + 4.56 GB
free/staging = 24 GB. Start with one short-context decode slot; loader accounting must gate
context growth. Do not promise Spark's 64k context on this card (S); MLA/indexer buffers matter (H).

If the GPU computes host-resident experts, all their quantized weights cross PCIe.
Q2 real routed traffic is 7.333 GB/token (I2/P): /25 GB/s = **293 ms/token minimum**, hence
**3.41 tok/s expert-read-only ceiling** (E, U). Spark measured 72.45 ms routed intervals,
about 101 GB/s useful reads, within a 204 ms/token, 4.90 tok/s run (P).
A10's supplied PCIe ceiling is roughly four times slower for those reads; cache residency
and predictive prefetch remove disk waits but cannot remove this PCIe limit.
Mapped host reads may attain less than DMA's ceiling; dummy/scratch positions can add traffic
(P's split executes 17 positions for 8 real experts). Benchmark staging versus direct mapping.

Illustrative E: keep Spark's approximately 132 ms non-routed remainder and replace 72 ms
with 293 ms: about 425 ms/token, **2.35 tok/s**, before changed disk exposure or GPU speed.
Planning range E: **1.5–2.5 tok/s** at short context, not a measured A10 result.
At an assumed disk rate of 3 GB/s, uniform misses above cost 257 ms/token if fully exposed,
giving about 1.47 tok/s with that illustrative floor; overlap and skew determine the outcome.
GH200 calibration is not an A10 forecast: H uses approximately 340 GB/s host expert reads over
C2C, and estimates Q5 GLM at 4–9 tok/s warm, about 2 downside; GH200 still runs K3 (S/H).
An A10 lacks C2C. Record local disk bandwidth before interpreting this baseline.

## Tier 2: owner-compute expert parallelism

Each VM owns disjoint (layer, expert) records in its VRAM plus RAM; send the real FFN input
and selected IDs to owners, compute there, return unweighted rows, then merge in original
expert order on the spine owner. Send activations, not weight records, across the network.
GLM has 75 active MoE layers, 256 experts/layer and top-8 routing (I2/M).
With balanced ownership and a single spine coordinator, expected remote rows/layer are
8 × (1−1/N): 4 at N=2, 6 at N=4, 7.67 at N=24 (E); maximum is 8.
Fan out to all selected owners concurrently and batch IDs per destination. One exchange
means one parallel fanout/join per layer, not eight sequential RPCs; the slowest owner wins.
RAM-only sharding solves capacity but retains PCIe reads: at most eight owners can help
one top-8 layer. At balanced uniform routing Q2 weights cost 293/N ms/token until that
fanout limit, plus spine and network (E, optimistic); placement skew worsens this.

Current engine already has RAM cache, disk streaming, split execution, prediction, TCP remote
rows and per-layer deadlines with local fallback (L/S). Its cap is global across hosts (R).
Start with 1 then 2 remote rows and a 2.5 ms deadline (S's audit proposal); generalize to
all 8 selected rows for fleet tests. The 2.72 ms mean spacing = 1000/4.90/75 (P/F)
is not the router-to-merge window. Trace the actual lower-tail available window first.
For faster all-resident runs target total exchange including service at p99 ≤0.5 ms (E),
then tune deadlines from observed slack. Late replies cannot simply stall every layer.
Retain byte-exact RAM/disk fallback; measure deadline misses, fallback cost and stale-reply rejection.

Wire budget E from M's D=6144 and f32 rows: input 24,576 B per destination, output
24,576 B per expert. Worst fanout duplicates input to 8 owners: 393,216 B/layer,
29.49 MB/token across 75 layers, excluding headers. At 10 tok/s that is 295 MB/s,
about 2.36 Gb/s aggregate at the coordinator; small-message latency is still the hard limit.
At 25 Gb/s ideal serialization of that aggregate is 0.126 ms/layer; at 100 Gb/s, 0.031 ms
(E, full aggregate serialized conservatively). Actual duplex parallelism helps; copies and incast hurt.
Request at least 25 GbE, preferably 100 GbE, with topology/oversubscription disclosed (E).
Measure application RTT with real payloads under load, not just ICMP.

F measured empty TCP medians 0.104–0.292 ms here, with p99 up to 0.567 ms;
8-row wire echoes reached about 0.55–0.96 ms median on the local low-bandwidth LAN.
These are not A10 service times. Use **0.05–0.15 ms effective exchange** as a datacenter
screening assumption E (roughly improved from the 0.1–0.3 ms LAN scale), and retain
0.1–0.3 ms as the conservative sensitivity. Validate both payload and fanout tails.
Persistent TCP sockets, pinned buffers, batched messages and asynchronous CUDA copies are
a sufficient first experiment. RDMA/NCCL is not a correctness prerequisite; introduce it if
TCP p99 exchange exceeds the budget or CPU/copy overhead dominates. RDMA removes neither
GPU GEMV time nor PCIe host-weight traffic; NCCL requires a deliberate dispatch/return design.

### All weights in fleet VRAM: the decisive capacity experiment

Use approximately 20 GB usable weight capacity/card (E, U leaves approximately 4 GB runtime).
Q4 full GGUF 467.280 GB requires ceil(467.280/20) = **24 A10s** (E, I4).
Q5 full GGUF 562.457 GB requires ceil(562.457/20) = **29 A10s** (E, I5).
These include the spine; they are pooled lower bounds, not a layout that replicates it.
Q4/Q5 full non-routed allocations are about 26.88/28.11 GB (E, inventories/H), exceeding
one card's weight budget; even Q5 without MTP is about 20.73 GB (H).
Partition the spine by contiguous layer ranges across fleet members, colocating expert shards
where possible; transfer hidden state at range boundaries. Never budget a full spine per worker.
A dedicated single-card spine needs selective host offload or lower quant and is a different experiment.
Full-model placement must pack layer/expert granularity and leave KV/workspace on spine owners;
24/29 are conditional minima. Request 32 to absorb packing/runtime peaks and compare both quants (E).

Critical path model E: Ttoken ≈ Tfixed + 75 × (tlocal + texchange), where tlocal includes
attention/shared work and the slowest selected expert GEMV. Add spine-boundary transfers,
queueing and fallback separately; no assumed 1/N token-latency scaling.
A's nominal approximately 600 GB/s/card gives Q4/Q5 routed read service of 0.306/0.371 ms
per layer if serialized on one card (E: I4/H traffic 13.763/16.698 GB divided by 75×600).
Expert fanout can lower that term, but small GEMV, spine reads and dispatch remain.
Screen with tlocal=0.6–1.2 ms and Tfixed=10–30 ms (E, unmeasured A10 efficiencies).
At texchange=0.05–0.15 ms: 59–131 ms/token, **7.6–17.0 tok/s** (E).
At LAN-like 0.1–0.3 ms: 63–143 ms/token, **7.0–16.0 tok/s** (E).
Network alone costs 7.5–22.5 ms/token at the latter RTTs; every extra 1 ms/layer adds 75 ms
(E). These are screening scenarios, not predictions from Intel exact-mode performance.

## Tier 3: hypothesis, allocation and implementation gates

Hypothesis: VRAM-resident owner-compute with concurrent row exchange can beat the
single-A10 PCIe ceiling and run Q4/Q5 at ≥8 tok/s short-context decode (E success target),
while preserving the same-quant single-engine arithmetic. RAM-backed pooling is a useful
capacity control but will be less efficient in latency and energy than VRAM residency.

Allocation request E, staged and released between phases:
- 1 VM for 2 days: build, exactness fixtures, disk/PCIe calibration, Q2 RAM-cache baseline.
- 4 VMs total for 3 days: compare N=1/2/4 at the same Q2 quant, cache budget and prompt;
  test RAM versus VRAM hot sets, fanout, deadlines, dropped replies and concurrent clients.
- 32 VMs total for 7 days: Q4 at 24 and 32, Q5 at 29 and 32, then repeated mixed-load runs.
  Maximum request is 32 simultaneous VMs; staged total is 238 VM-days (5712 GPU-hours).
  Proceed to the large phase only after CUDA remote rows pass identity and measured tails fit.

Measure warm/cold decode tok/s, prefill separately, p50/p95/p99 token and layer latency,
bytes across PCIe/NIC/disk, owner imbalance, remote rows accepted versus fallen back, and
peak RAM/VRAM. Report tok/s per allocated GPU and total fleet wall watts: tok/s/W;
also report GPU telemetry separately if VM wall power is unavailable, never call it whole-system power.
At 8 tok/s, 24/29 GPUs yield 0.333/0.276 tok/s/GPU (E); capacity success can still be poor efficiency.
Use fixed prompts/seeds, immutable cache snapshots, short and long context and held-out routing traces.
Compare stage dumps and output bytes against same-quant CUDA reference; prove rows were actually
remote. Cross-quant hashes are not meaningful; sm_86 versus GB10/Hopper identity is an explicit test.

Code work in kimi-k3-spark: build existing CUDA quant kernels for sm_86; no new ISA port is
required, but remove Hopper-only assumptions and validate dispatch/reduction/FTZ/SwiGLU (M/H).
Implement a CUDA engine row-service mode reusing its actual expert GEMV kernels and quantized
records, not the Intel SYCL/scalar servers (S: GLM exact scalar path is too slow).
Add multi-owner placement/actual served-set reporting, concurrent fanout, an 8-row cap,
request IDs, cancellation, bounded queues and deadline-safe fallback; extend tests for late/partial replies.
Add device-resident expert banks and layer-range spine placement with explicit activation handoff;
a RAM cache and current remote tier alone do not implement the all-VRAM design.
Audit Q4/Q5 byte strides and batch dispatch, preserve per-expert arithmetic and merge ordering;
CUDA on both ends does not automatically imply bit identity (M/S).

Risks: no NVLink makes every cross-VM dependency visible; host experts remain PCIe-bound;
network tails and incast multiply across sequential layers; VM CPU scheduling/noisy neighbours
can exhaust deadlines (L reports preemption-driven tails). Quantify NIC/GPU sharing, pin threads
where permitted and repeat at different load periods. Memory peaks, routing skew and long-prefill
numerics can invalidate packing or identity; keep full fallback stores and reject unsupported layouts.
