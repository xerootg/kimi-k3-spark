#!/bin/bash
# gh200prod.sh down|up : glm53-gh200 to 0 (wait until the pod is gone) and suspend the RE workbench CronJob, or back to 1
# (wait Ready), unsuspend the CronJob and verify the gold md5 through the Service (from the Spark, key via stdin).
K="kubectl --server=https://192.168.4.2:6443 --tls-server-name=kubernetes"
D=$(dirname "$0")
case $1 in
down)
  $K -n re-lab patch cronjob workbench-queue -p '{"spec":{"suspend":true}}' >/dev/null && echo "workbench-queue suspend=$($K -n re-lab get cronjob workbench-queue -o jsonpath='{.spec.suspend}')"
  $K -n kimi-k3 scale deploy glm53-gh200 --replicas=0 >/dev/null
  for i in $(seq 1 90); do n=$($K -n kimi-k3 get pods -l app.kubernetes.io/name=glm53-gh200 --no-headers 2>/dev/null | wc -l); [ "$n" = 0 ] && break; sleep 5; done
  ssh ubuntu@192.222.50.246 'pgrep -af "[l]lama-server" || echo "no llama-server on the GH200"; nvidia-smi --query-gpu=memory.used --format=csv,noheader'
  echo "replicas $($K -n kimi-k3 get deploy glm53-gh200 -o jsonpath='{.spec.replicas}') pods $n";;
up)
  $K -n kimi-k3 scale deploy glm53-gh200 --replicas=1 >/dev/null
  for i in $(seq 1 120); do r=$($K -n kimi-k3 get deploy glm53-gh200 -o jsonpath='{.status.readyReplicas}'); [ "$r" = 1 ] && break; sleep 10; done
  echo "ready=$r image $($K -n kimi-k3 get deploy glm53-gh200 -o jsonpath='{.spec.template.spec.containers[0].image}')"
  KEY=$($K -n kimi-k3 get secret kimi-api-key -o jsonpath='{.data.LLAMA_API_KEY}' | base64 -d | cut -d, -f1)
  { printf '%s\n' "$KEY"; cat $D/gh200verify.py; } | ssh xero@192.168.4.209 'read -r K; cat > /tmp/gv.py; printf "%s\n" "$K" | python3 /tmp/gv.py 3; rm -f /tmp/gv.py'
  unset KEY
  $K -n re-lab patch cronjob workbench-queue -p '{"spec":{"suspend":false}}' >/dev/null && echo "workbench-queue suspend=$($K -n re-lab get cronjob workbench-queue -o jsonpath='{.spec.suspend}')";;
esac
