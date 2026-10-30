# NUC GLM image result

Changed:
- docker/k3x-serve.Dockerfile builds g53x directly with build-sycl.sh's icpx
  flags, including scoped strict FP and -lze_loader; builds static g53cs via
  cpu/build.sh. Both reuse the existing pinned oneAPI/driver runtime bundle.
- Explicit source/header COPY list excludes MUFU tables, stores and datasets.
  Final image adds only the two binaries and their runtime dependencies.
  g53x participates in dependency collection and final missing-library checks.
- Both servers now support --help before device/table/server initialization.
- k3x-serve.yml watches GLM sources, shared MUFU header and versions.env.
  It loads the built image, runs both --help commands without network/GPU,
  and only then pushes the tested tags.
- tools/glm53/NUC-DEPLOY.md documents /mufu mounts, stores/sidecars, placement
  sets versus CPU IDX, Docker and Argo, port 19600, memlock, GPU access,
  budgets, engine setup and a numbered owner verification checklist.

Test command (from checkout root):
```sh
OUT="$PWD/tools/glm53/cpu/build/g53cs-image-smoke" bash tools/glm53/cpu/build.sh
tools/glm53/cpu/build/g53cs-image-smoke --help
file tools/glm53/cpu/build/g53cs-image-smoke
git diff --check
```
Last output lines (exit 0):
```text
usage: g53cs IDX BIN PORT
IDX: gid gate_type down_type offset; BIN: K3XS store
K3X_CPU_CPUS and K3X_MUFU=/mufu are required for serving
K3X_DISK_TIER=1 enables disk reads; K3X_MAX_GB caps resident records
[CPU binary] ELF 64-bit LSB executable, x86-64, statically linked
```
Also parsed workflow/deployment YAML with PyYAML, checked embedded shell
with bash -n, checked Docker RUN shell syntax and existence of COPY sources:
`PASS: workflow YAML/shell, Docker COPY sources/RUN shell, deployment YAML/shell`
Docker instructions/stage layout reviewed manually; no image build attempted.

Uncertainties / owner action:
- No SYCL compiler/image/GPU execution here. Owner must build, run image smoke,
  inspect dependencies/image size and validate both NUCs with actual tables,
  stores, locked memory, TCP, exact comparisons and disk allocation/fault gates.
- This checkout has no FASTEXACT-RESULT.md; read EXACT-RESULT.md instead.
  G53X_EXACT=2 selects the same scalar kernel as 1, not a separate fast kernel.
  Confirm intended mode 2 implementation and measured latency before rollout.
- For 50/24 GB total serving budgets, guide uses MAX_GB=44/18 (6 GB headroom).
  Verify real RSS/VmLck and reduce if needed; MAX_GB is records only.
- External Argo ledger/manifest unavailable within checkout. Example follows
  the user's stated K3 conventions; owner must compare against actual template.
- Existing residual GPU minor faults, NEO debug-key support, and exact-mode
  engine deadlines/routing remain owner checks; detailed list is in the guide.

No network, commits or pushes; all writes stayed in this checkout.
