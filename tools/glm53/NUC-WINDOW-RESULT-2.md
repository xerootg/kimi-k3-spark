Added Kubernetes mode to tools/glm53/nuc-window.sh; native lifecycle retained.
--k8s stage requires replicas=0, copies placement Spark -> NUC over LAN,
selects the resident set, and installs it at the Deployment's host placement path.
It skips g53x, Docker, library copying and runtime ldd checks.
--k8s start scales to 1, waits for rollout/Ready (default 1800s), reads the
resident served count from kubectl logs, validates K3X_SETOUT covers residents,
and writes resident-only remote.set. Disk-only IDs never become remote owners.
--k8s stop scales to 0. probe/ab retain their Spark execution and NUC endpoint.
status shows pods, latest server load/stats lines, and optional Spark leg stats.
Driver/test changes cover runtime skipping, budgets, ownership and mock lifecycle.
No network, live cluster actions, commits or pushes performed.

Laptop usage (run ak first if cluster auth needs it):
  tools/glm53/nuc-window.sh --k8s stage
  tools/glm53/nuc-window.sh --k8s start
  tools/glm53/nuc-window.sh --k8s probe
  REMOTE_PATCH=/path/0013.patch tools/glm53/nuc-window.sh --k8s ab
  ROWS=2 tools/glm53/nuc-window.sh --k8s ab
  tools/glm53/nuc-window.sh --k8s status /srv/glm-5.3/nuc-window/ab-RUN/on1.log
  tools/glm53/nuc-window.sh --k8s stop
Omit --k8s for native mode; status also accepts LEG_LOG (path on Spark).
ab scales glm53-spark in kimi-k3 to zero and restores one replica as before.

Defaults: DEPLOY=glm-nuc2, NAMESPACE=kimi-k3, NUC_NODE=nuc2, port 19600.
PLACEMENT defaults to /srv/glm-5.3/placement/nuc2-ram.set on Spark.
K8S_PLACEMENT defaults to the same path on NUC;
K8S_SERVED=/srv/glm-5.3/served/nuc2.served.set.
READY_TIMEOUT, MAX_GB, NUC, NUC_IP, WORK and SPARK are configurable.
For nuc3 set NUC_NODE=nuc3, NUC/NUC_IP and a separate WORK;
node-specific placement/served paths, glm-nuc3 and 18 GB are then selected.
NUC2 uses 44 GB. MAX_GB accepts 18 or 44; match the Deployment's cap.

Unverified: actual Deployment names, readiness/logs, SSH, host path permissions,
store/MUFU compatibility and hardware behavior. Evidence YAML names glm53-nuc2/3;
defaults follow the requested glm-nuc2/3. Use DEPLOY override if necessary.
The existing Deployment must already use the intended image; this script does
not apply manifests or change images. Evidence YAML still has GLM_TAG placeholders.
Provided image (g53x exact=2 default + disk tier, g53cs):
  ghcr.io/xerootg/k3x-serve:g0b50a14@sha256:95476408866268120b547d9e6ceba6ee0e0776b9a2fd3e561694bc621e949891
Server stats may appear only after a client disconnects; absent stats are reported.

Validation commands:
  bash -n tools/glm53/nuc-window.sh
  python3 tools/glm53/test_nuc_window.py
  git diff --check
Last test output lines:
  ----------------------------------------------------------------------
  Ran 8 tests in 6.308s

  OK
bash -n and git diff --check exited 0 without output.
