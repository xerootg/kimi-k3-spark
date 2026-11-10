#!/bin/bash
# sc67-win.sh LEG...  (run on the laptop, in the Spark window, glm53-spark at 0, sc67-wd.sh holding window.lock)
# Acceptance of patch 0067 on the CI image as Kubernetes canaries (as window spark4): for each leg the production
# manifest with the new image (A, A2) or with LLAMA_EXPERT_STORE_SPLIT_COLS=1 LLAMA_EXPERT_STORE_SPLIT_COLS_MIN=3
# added (B, B2) runs as glm53-spark-canary (no Service traffic), from the same copy of production's cache snapshot;
# the request list runs on the Spark against the pod IP (sc67-phase.sh), then the canary is deleted.
# Needs: $W/canary-A.yaml $W/canary-B.yaml (mkcanary67.py), the keys file on the Spark (0600, deleted after).
W=${W:?work dir}; SP=xero@192.168.4.209; D=/srv/glm-5.3-flash/legs-sc67
K="kubectl --server=https://192.168.4.2:6443 --tls-server-name=kubernetes -n kimi-k3"
for leg in "$@"; do
  case $leg in A*) Y=$W/canary-A.yaml ;; B*) Y=$W/canary-B.yaml ;; *) echo "unknown leg $leg"; continue ;; esac
  echo "=== LEG $leg $(date -u +%T)"
  [ -n "$($K get pods -l kimi-k3/spark-server=true -o name)" ] && { echo "ABORT: a production pod exists"; exit 1; }
  ssh $SP "cp /srv/glm-5.3-flash/snapshot-canary/seed67.k3sn /srv/glm-5.3-flash/snapshot-canary/glm53flash-q2.k3sn && rm -f $D/phase-$leg.done" || exit 1
  $K apply -f $Y >/dev/null || exit 1
  $K rollout status deployment/glm53-spark-canary --timeout=900s | tail -1 || { echo "LEG-FAIL $leg not ready"; $K delete deployment glm53-spark-canary --wait=true; exit 1; }
  IP=$($K get pods -l app.kubernetes.io/name=glm53-spark-canary -o jsonpath='{.items[0].status.podIP}')
  echo "READY $leg pod IP $IP $(date -u +%T)"
  ssh $SP "setsid bash $D/sc67-phase.sh $leg http://$IP:8080 > $D/phase-$leg.log 2>&1 < /dev/null &"
  n=0; until ssh -o ConnectTimeout=10 $SP "test -e $D/phase-$leg.done"; do
    n=$((n+1)); [ $n -gt 120 ] && { echo "LEG-TIMEOUT $leg"; break; }
    [ -z "$($K get pods -l app.kubernetes.io/name=glm53-spark-canary --field-selector=status.phase=Running -o name)" ] && { echo "CANARY DIED $leg"; break; }
    sleep 15
  done
  $K logs deployment/glm53-spark-canary --tail=-1 > $W/server-$leg.log 2>&1
  $K delete deployment glm53-spark-canary --wait=true >/dev/null
  until [ -z "$($K get pods -l app.kubernetes.io/name=glm53-spark-canary -o name)" ]; do sleep 3; done
  echo "END $leg $(date -u +%T)"
done
echo "SC67-WIN-DONE $(date -u +%T)"
