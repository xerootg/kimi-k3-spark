# GLM-5.3 on the NUC image

`ghcr.io/xerootg/k3x-serve:g<commit>` contains `/opt/k3x/g53x` (SYCL)
and `/opt/k3x/g53cs` (static AVX2 + AVX-VNNI CPU), alongside k3x/k3cs.
Override the image's k3x entrypoint. Use an immutable tested tag/digest.
The SYCL compiler/runtime and user-mode driver stack are the Dockerfile's
pinned oneAPI 2026.1.1 / Level Zero 1.28.2 / compute-runtime 26.18.38308.1 /
IGC 2.34.4. g53x uses build-sycl.sh's standalone compiler flags, without
building glmbench. The host supplies the kernel GPU driver.

## Files and budgets

On each NUC prepare:

- `/srv/glm-5.3/store/glm53-q2-v2.k3xs` and its `.k3xs.json` sidecar,
  produced by repack-experts.py (v1/v2 supported).
- `/srv/glm-5.3/mufu/{rcp,ex2p,ex2n}.bin`: verified GB10 tables,
  mounted read-only at `/mufu`. These total 2,181,038,080 bytes (~2.2 GB);
  no tables, model records or stores are baked into the image. No tanh table
  is needed by GLM. g53x uses `G53X_MUFU=/mufu`; g53cs uses `K3X_MUFU=/mufu`.
- `/srv/glm-5.3/placement`: input placement files and writable served-set output.

From this checkout, generate placement against the actual store and preferably
real training/held-out decode traces (without traces place.py uses synthetic
traffic, which does not establish a performance claim):

```sh
python3 tools/glm53/place.py \
  --store /srv/glm-5.3/store/glm53-q2-v2.k3xs \
  --nuc2-gb 44 --nuc3-gb 18 --out /srv/glm-5.3/placement
```

NUC2 has a 50 GB serving budget; NUC3 has 24 GB. Use `K3X_MAX_GB=44` and
`18`, respectively, reserving 6 GB within those budgets for ~2.2 GB tables,
a 1 GB disk LRU, scratch, pinned USM chunks, threads and runtime overhead.
K3X_MAX_GB counts resident records, not total process memory. Do not set it
to the entire 50/24 GB allowance. Check actual RSS/locked memory and reduce
it further if needed, especially if another server shares the NUC.

place.py writes `nuc2-ram.set`, `nuc3-ram.set` (layer/expert pairs for g53x),
`nuc2-ram.idx`, `nuc3-ram.idx` (gid/type/offset records for g53cs, only with
--store), `nuc2-ssd.set`, `nuc3-ssd.set`, and `spark.set`. Copy the node's files
onto that node under placement, preserving names. Do not feed a two-column
.set to g53cs. Regenerate .idx files whenever the store changes.
The SSD sets describe placement ownership; these servers' disk tiers expose
all supported store records, not only those SSD sets.

Do not run the generated deploy.sh for this deployment: it uses bare binaries,
non-exact mode 0 and port 19653. Adapt engine.env/engine.yaml to port 19600
below and to the actual served sets copied back from the NUCs.

## Docker

On nuc2, run one of these alternatives. On nuc3 set `NODE=nuc3 BUDGET=18`.
Choose the actual render GID and CPU IDs from the host; CPU workers should
start with one worker per physical core. Both examples use host networking
(Docker's equivalent of Kubernetes hostNetwork), so no -p mapping is needed.
Stop any existing K3/GLM listener on port 19600 first.

```sh
NODE=nuc2
BUDGET=44
IMAGE='ghcr.io/xerootg/k3x-serve:g<commit>'
RENDER_GID=$(getent group render | cut -d: -f3)
docker run --rm --name g53x --network host \
  --device /dev/dri --group-add "$RENDER_GID" \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /srv/glm-5.3/store:/store:ro \
  -v /srv/glm-5.3/mufu:/mufu:ro \
  -v /srv/glm-5.3/placement:/placement \
  -e G53X_EXACT=2 -e G53X_MUFU=/mufu \
  -e G53X_MLOCKALL=1 -e G53X_SUBMIT=1 -e G53X_NEO_TUNE=1 \
  -e G53X_ULLS=0 -e K3X_MAX_GB="$BUDGET" \
  -e K3X_DISK_TIER=1 -e K3X_DISK_STORE=/store/glm53-q2-v2.k3xs \
  -e K3X_DISK_LRU_GB=1 -e K3X_DISK_DEADLINE_MS=20 \
  -e K3X_DISK_THREADS=8 -e K3X_DISK_PIECE_KB=512 \
  -e K3X_SETOUT=/placement/served.set \
  --entrypoint /opt/k3x/g53x "$IMAGE" \
  /placement/$NODE-ram.set /store/glm53-q2-v2.k3xs - - serveidx 19600
```

```sh
docker run --rm --name g53cs --network host \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v /srv/glm-5.3/store:/store:ro \
  -v /srv/glm-5.3/mufu:/mufu:ro \
  -v /srv/glm-5.3/placement:/placement \
  -e K3X_MUFU=/mufu -e K3X_CPU_CPUS=0-7 -e K3X_MAX_GB="$BUDGET" \
  -e K3X_DISK_TIER=1 -e K3X_DISK_STORE=/store/glm53-q2-v2.k3xs \
  -e K3X_DISK_LRU_GB=1 -e K3X_DISK_THREADS=8 \
  -e K3X_DISK_DEADLINE_MS=20 -e K3X_SETOUT=/placement/served.set \
  --entrypoint /opt/k3x/g53cs "$IMAGE" \
  /placement/$NODE-ram.idx /store/glm53-q2-v2.k3xs 19600
```

CPU serving does not need /dev/dri or a render group. g53cs is always exact;
G53X_EXACT applies only to g53x. This checkout's exact_mode() treats both
1 and 2 as true and uses the same scalar exact kernels. FASTEXACT-RESULT.md
is absent here; EXACT-RESULT.md and MERGE-RESULT.md are the available evidence.
Setting 2 does not add a fast exact implementation. Confirm the intended mode
2 implementation/latency before production; do not assume mode 0 timings.

K3X_SETOUT includes supported disk records as well as resident records. After
startup succeeds and the listener is live, copy each node's served.set back to
Spark as `nuc2.served.set` / `nuc3.served.set`. Configure the engine's
LLAMA_EXPERT_STORE_REMOTE as `nuc2:19600,nuc3:19600` and REMOTE_SET as those
two absolute paths, in the same order. Both disk servers may advertise the
same supported records; check the engine's owner selection and desired split.
The generated 2200 us remote deadline comes from fast-mode assumptions;
measure and explicitly choose the engine deadline for exact serving. The
20 ms disk deadline can yield per-row NaN fallback to local Spark compute.

## Argo manifest

The external ledger/argo-things checkout was not available in this directory.
Use `configs/kimi-k3/k3x-nuc2.yaml` there as the owner-reviewed template, with
its described hostNetwork port 19600, resource requests without limits, and
privileged /dev/dri access. Create a separate GLM workload, or replace the K3
listener; they cannot share that host port. The pod spec should contain:

```yaml
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
nodeSelector:
  kubernetes.io/hostname: nuc2
containers:
  - name: g53x
    image: ghcr.io/xerootg/k3x-serve:g<commit>
    securityContext:
      privileged: true
      runAsUser: 0
    # Kubernetes has no PodSpec ulimit field. Require the runtime to permit
    # raising the hard memlock limit; fail startup if this cannot be done.
    command: ["/bin/bash", "-ec"]
    args:
      - >-
        ulimit -l unlimited;
        exec /opt/k3x/g53x /placement/nuc2-ram.set
        /store/glm53-q2-v2.k3xs - - serveidx 19600
    ports:
      - {name: expert, containerPort: 19600, protocol: TCP}
    resources:
      requests: {cpu: "1", memory: "50G"}
    env:
      - {name: G53X_EXACT, value: "2"}
      - {name: G53X_MUFU, value: /mufu}
      - {name: G53X_MLOCKALL, value: "1"}
      - {name: G53X_SUBMIT, value: "1"}
      - {name: G53X_NEO_TUNE, value: "1"}
      - {name: G53X_ULLS, value: "0"}
      - {name: K3X_MAX_GB, value: "44"}
      - {name: K3X_DISK_TIER, value: "1"}
      - {name: K3X_DISK_STORE, value: /store/glm53-q2-v2.k3xs}
      - {name: K3X_DISK_LRU_GB, value: "1"}
      - {name: K3X_DISK_DEADLINE_MS, value: "20"}
      - {name: K3X_DISK_THREADS, value: "8"}
      - {name: K3X_DISK_PIECE_KB, value: "512"}
      - {name: K3X_SETOUT, value: /placement/served.set}
    volumeMounts:
      - {name: dri, mountPath: /dev/dri}
      - {name: store, mountPath: /store, readOnly: true}
      - {name: mufu, mountPath: /mufu, readOnly: true}
      - {name: placement, mountPath: /placement}
volumes:
  - name: dri
    hostPath: {path: /dev/dri, type: Directory}
  - name: store
    hostPath: {path: /srv/glm-5.3/store, type: Directory}
  - name: mufu
    hostPath: {path: /srv/glm-5.3/mufu, type: Directory}
  - name: placement
    hostPath: {path: /srv/glm-5.3/placement, type: Directory}
```

Wrap in the template's Deployment metadata/spec, replicas 1, and use Recreate
so an update does not compete for the host port. Add a generous startup probe
and TCP readiness probe on 19600; model loading/JIT can take minutes. For
nuc3 change node selector, .set name, request to 24G, and MAX_GB to 18.
If using a non-root GPU pod, add the host render GID to supplementalGroups.
For g53cs change the exec command to the CPU command above (.idx, no serveidx
arguments), replace G53X_* env with K3X_MUFU and K3X_CPU_CPUS, and remove the
DRI volume. Keep unlimited memlock and the lock capability; do not enable
G53_ALLOW_UNLOCKED in production. Requests are scheduling reservations, not
memory ceilings; size them against actual node allocatable resources.

## Owner verification required

1. Build the linux/amd64 image with versions.env's LLAMA_SHA. Confirm icpx
   strict-FP compilation and -lze_loader linking, static g53cs, both help smoke
   commands, and no missing dynamic dependencies. Check image size/history:
   only binaries/runtime libraries, no tables/stores/compiler in final stage.
2. On both NUCs verify AVX-VNNI support, host GPU driver/render permissions,
   the pinned user-mode stack's compatibility, unlimited /proc/PID/limits,
   startup self-checks, mlock success and RSS/VmLck with the chosen budgets.
3. With verified tables/data run check-exact.sh, serve_test.py and
   cuda_serve_check.py in exact modes 1 and 2 against this built binary/runtime;
   require zero changed computed rows (documented zero-row NaN sentinels
   separately). Run cpu/test.sh and cpu/test_server.py with real tables/data.
   The existing laptop test harness builds/runs its own binaries: adapt it to
   invoke this image so it actually validates the shipped artifact.
4. Run disk_test.py gate and bench on each idle NUC, against the shipped
   binary, including cold/LRU/deadline cases. Verify NEO keys, no request
   allocations, faults, latency and throughput. Known residual i915 GEM minor
   faults and ULLS stalls remain unresolved in DISK-RESULT.md; ULLS stays off.
5. Verify actual mode 2 availability/performance, disk advertised sets and
   engine routing/deadlines, port conflicts, fallback behavior and end-to-end
   decode correctness before enabling offload. Compare the manifest with the
   actual K3 ledger template and check Argo scheduling/readiness/memlock.

CI's --help smoke validates executable startup/linking only; it does not JIT
kernels, load tables, open sockets or establish numerical correctness.
