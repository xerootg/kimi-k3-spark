#!/usr/bin/env bash
# Laptop orchestrator. Model/store/snapshot traffic stays on the cluster LAN.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPARK=${SPARK:-xero@192.168.4.209}
NUC=${NUC:-xero@192.168.4.249}
NUC_IP=${NUC_IP:-192.168.4.249}
PORT=${PORT:-19600}
WORK=${WORK:-/home/xero/glm53-nuc-window}
SPWORK=${SPWORK:-/srv/glm-5.3/nuc-window}
IMAGE=${IMAGE:-intel/deep-learning-essentials:2026.1.1-devel-ubuntu24.04}
G53X=${G53X:-$HOME/glm53-work/g53x}
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30)
# Quote every remote argument; ssh otherwise reparses argument boundaries.
remote() { local host=$1; shift; local cmd; printf -v cmd '%q ' "$@"; "${SSH[@]}" "$host" "$cmd"; }
install_helper() {
  remote "$1" mkdir -p "$2"
  scp -q -o BatchMode=yes "$HERE/nuc-window.py" "$1:$2/driver.py"
}
nuc() { remote "$NUC" python3 "$WORK/driver.py" "$@" --work "$WORK" --port "$PORT"; }
K8S=0
if [[ ${1:-} == --k8s ]]; then K8S=1; shift; fi
COMMAND=${1:-help}
shift "$(( $# > 0 ? 1 : 0 ))"
LEG_LOG=${1:-${LEG_LOG:-}}
[[ $# -le 1 && ( $# == 0 || $COMMAND == status ) ]] || { echo "Unexpected arguments" >&2; exit 2; }
NUC_NODE=${NUC_NODE:-nuc2}
DEPLOY=${DEPLOY:-glm-$NUC_NODE}
NAMESPACE=${NAMESPACE:-kimi-k3}
K8S_PLACEMENT=${K8S_PLACEMENT:-/srv/glm-5.3/placement/$NUC_NODE-ram.set}
K8S_SERVED=${K8S_SERVED:-/srv/glm-5.3/served/$NUC_NODE.served.set}
READY_TIMEOUT=${READY_TIMEOUT:-1800s}
kube() { kubectl -n "$NAMESPACE" "$@"; }
server_logs() { kube logs "deploy/$DEPLOY" --all-containers=true --tail=-1; }
if [[ $K8S == 1 ]]; then
  case $COMMAND in
  stage)
    install_helper "$NUC" "$WORK"
    # The Deployment must be parked while its mounted placement is replaced.
    [[ $(kube get deploy "$DEPLOY" -o jsonpath='{.spec.replicas}') == 0 ]] || {
      echo "Scale $DEPLOY to zero before staging" >&2; exit 1;
    }
    kube rollout status "deploy/$DEPLOY" --timeout="$READY_TIMEOUT"
    placement=${PLACEMENT:-/srv/glm-5.3/placement/$NUC_NODE-ram.set}
    remote "$SPARK" scp -q -o BatchMode=yes "$placement" "$NUC:$WORK/placement.set"
    budget=44
    [[ $NUC_NODE != nuc3 ]] || budget=18
    nuc stage --placement "$WORK/placement.set" --skip-runtime --max-gb "${MAX_GB:-$budget}"
    remote "$NUC" cp "$WORK/resident.set" "$K8S_PLACEMENT"
    exit ;;
  start)
    kube scale deploy "$DEPLOY" --replicas=1
    kube rollout status "deploy/$DEPLOY" --timeout="$READY_TIMEOUT"
    log=$(server_logs)
    # g53x/g53cs report the resident served count in their startup load line.
    loaded=$(printf '%s\n' "$log" | sed -nE 's/.*loaded ([0-9]+) of ([0-9]+) records.*/served resident records=\1 (requested \2)/p' | tail -n 1)
    [[ -n $loaded ]] || { echo "Ready pod has no served count in log" >&2; exit 1; }
    printf '%s\n' "$loaded"
    nuc ready-set --served "$K8S_SERVED"
    exit ;;
  stop) kube scale deploy "$DEPLOY" --replicas=0; exit ;;
  esac
fi
case $COMMAND in
stage)
  [[ -f $G53X ]] || { echo "Build first: tools/glm53/build-sycl.sh (missing $G53X)" >&2; exit 1; }
  install_helper "$NUC" "$WORK"
  nuc stopped
  remote "$NUC" mkdir -p "$WORK/lib"
  scp -q -o BatchMode=yes "$G53X" "$NUC:$WORK/g53x.new"
  remote "$NUC" mv "$WORK/g53x.new" "$WORK/g53x"
  # No image pull. Inspect ELF closure inside the exact compiler image, plus
  # dlopened UR adapters/Level Zero libraries (not visible in g53x's DT_NEEDED).
  docker image inspect "$IMAGE" >/dev/null
  tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
  docker run --pull=never --rm --network none -v "$G53X:/tmp/g53x:ro" "$IMAGE" bash -ec '
    export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib
    ldd /tmp/g53x >&2
    files=(/tmp/g53x /opt/intel/oneapi/compiler/latest/lib/libsycl.so* /opt/intel/oneapi/compiler/latest/lib/libur_*.so* /opt/intel/oneapi/compiler/latest/lib/libze_*.so*)
    for f in "${files[@]}"; do
      [[ -e $f ]] || continue
      [[ $f == /tmp/g53x ]] || echo "$f"
      ldd "$f" | awk '\''/=> \/opt\/intel\// {print $3}'\''
    done' > "$tmp/paths"
  sort -u "$tmp/paths" -o "$tmp/paths"
  while IFS= read -r lib; do
    name=${lib##*/}
    if ! remote "$NUC" test -e "/home/xero/k3-bitexact/sycl/lib/$name" && ! remote "$NUC" test -e "$WORK/lib/$name"; then
      # docker cp follows library symlinks; only runtime files cross the WAN.
      cid=$(docker create --network none "$IMAGE")
      docker cp -L "$cid:$lib" "$tmp/$name"; docker rm "$cid" >/dev/null
      scp -q -o BatchMode=yes "$tmp/$name" "$NUC:$WORK/lib/$name"
    fi
  done < "$tmp/paths"
  placement=${PLACEMENT:-/srv/glm-5.3/placement/nuc2-ram.set}
  # Placement normally lives on Spark. Transfer it directly over the LAN.
  if remote "$SPARK" test -f "$placement"; then
    remote "$SPARK" scp -q -o BatchMode=yes "$placement" "$NUC:$WORK/placement.set"
    placement=$WORK/placement.set
  fi
  nuc stage --placement "$placement"
  ;;
status)
  kube get pods -l "app.kubernetes.io/name=glm53-remote,kimi-k3/node=$NUC_NODE" -o wide
  if [[ $K8S == 1 ]]; then
    log=$(server_logs) || log='No current pod log available'
  else
    log=$(remote "$NUC" cat "$WORK/server.log") || log='No native server log available'
  fi
  printf '%s\n' "$log" | awk '/loaded [0-9]+ of [0-9]+ records/ {load=$0} /disk:|stats/ {stats=$0} END {if(load) print load; if(stats) print stats; else print "No server stats line yet"}'
  if [[ -n $LEG_LOG ]]; then
    remote "$SPARK" python3 -c 'import pathlib,sys; print("\n".join(s for s in pathlib.Path(sys.argv[1]).read_text(errors="replace").splitlines() if "remote layer" in s))' "$LEG_LOG"
  fi
  ;;
start) nuc start ;;
stop) nuc stop ;;
probe|ab)
  install_helper "$SPARK" "$SPWORK"
  # Fetch only the supported resident ownership set over the LAN, from Spark.
  remote "$SPARK" scp -q -o BatchMode=yes "$NUC:$WORK/remote.set" "$SPWORK/remote.set"
  if [[ $COMMAND == probe ]]; then
    remote "$SPARK" python3 "$SPWORK/driver.py" probe --work "$SPWORK" --ip "$NUC_IP" --port "$PORT"
    exit
  fi
  if [[ -n ${REMOTE_PATCH:-} ]]; then
    scp -q -o BatchMode=yes "$REMOTE_PATCH" "$SPARK:$SPWORK/0013.patch"
  fi
  remote "$SPARK" python3 "$SPWORK/driver.py" build --work "$SPWORK"
  # These are the only production-pod mutations. Always restore on failure/signal.
  restore() { kubectl -n kimi-k3 scale deploy glm53-spark --replicas=1 && kubectl -n kimi-k3 rollout status deploy/glm53-spark --timeout=900s; }
  replicas=$(kubectl -n kimi-k3 get deploy glm53-spark -o jsonpath='{.spec.replicas}')
  [[ $replicas == 1 ]] || { echo "Expected one production replica; aborting without changing it" >&2; exit 1; }
  trap restore EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  kubectl -n kimi-k3 scale deploy glm53-spark --replicas=0
  remote "$SPARK" python3 "$SPWORK/driver.py" ab --work "$SPWORK" --ip "$NUC_IP" --port "$PORT" --rows "${ROWS:-1}"
  ;;
*) cat <<'USAGE'
Usage: tools/glm53/nuc-window.sh [--k8s] stage|start|probe|ab|stop|status [Spark-leg-log]
Run on laptop: stage; start; probe; REMOTE_PATCH=/path/0013.patch ... ab; stop
Then ROWS=2 ... ab. Patch optional only if source is already patched.
Overrides: G53X IMAGE SPARK NUC NUC_IP PORT WORK SPWORK PLACEMENT ROWS.
Kubernetes: --k8s stage; --k8s start; --k8s probe; --k8s ab; --k8s stop.
Run ak first if kubectl needs cluster authentication. Image/deploy must exist.
K8s overrides: DEPLOY (glm-nuc2), NAMESPACE (kimi-k3), NUC_NODE (nuc2),
K8S_PLACEMENT K8S_SERVED READY_TIMEOUT MAX_GB (44 on nuc2, 18 on nuc3).
status [Spark-leg-log] (or LEG_LOG) shows pods, server stats, remote layer lines.
For nuc3 also override NUC, NUC_IP, WORK and DEPLOY to match that node.
USAGE
;;
esac
