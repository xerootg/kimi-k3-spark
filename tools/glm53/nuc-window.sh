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
case ${1:-help} in
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
start) nuc start ;;
stop) nuc stop ;;
probe|ab)
  install_helper "$SPARK" "$SPWORK"
  # Fetch only the supported resident ownership set over the LAN, from Spark.
  remote "$SPARK" scp -q -o BatchMode=yes "$NUC:$WORK/remote.set" "$SPWORK/remote.set"
  if [[ $1 == probe ]]; then
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
Usage: tools/glm53/nuc-window.sh stage|start|probe|ab|stop
Run on laptop: stage; start; probe; REMOTE_PATCH=/path/0013.patch ... ab; stop
Then ROWS=2 ... ab. Patch optional only if source is already patched.
Overrides: G53X IMAGE SPARK NUC NUC_IP PORT WORK SPWORK PLACEMENT ROWS.
Defaults use nuc2 only (44 GB); nuc3 is intentionally not started.
USAGE
;;
esac
