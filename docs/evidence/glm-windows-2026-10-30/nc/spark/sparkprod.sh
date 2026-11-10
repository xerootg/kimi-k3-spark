#!/bin/bash
# sparkprod.sh down|up : glm53-spark to 0 (wait until the pod is gone), or back to 1 (wait Ready) and verify the gold through
# the kimi-k3 Service from the Spark (key via stdin, never printed).
K="kubectl --server=https://192.168.4.2:6443 --tls-server-name=kubernetes"
D=$(dirname "$0")
case $1 in
down)
  $K -n kimi-k3 scale deploy glm53-spark --replicas=0 >/dev/null
  for i in $(seq 1 90); do n=$($K -n kimi-k3 get pods -l app.kubernetes.io/name=glm53-spark --no-headers 2>/dev/null | wc -l); [ "$n" = 0 ] && break; sleep 5; done
  ssh xero@192.168.4.209 'pgrep -af "[l]lama-server" || echo "no llama-server on the Spark"; free -g | head -2' < /dev/null
  echo "replicas $($K -n kimi-k3 get deploy glm53-spark -o jsonpath='{.spec.replicas}') pods $n";;
up)
  $K -n kimi-k3 scale deploy glm53-spark --replicas=1 >/dev/null
  for i in $(seq 1 120); do r=$($K -n kimi-k3 get deploy glm53-spark -o jsonpath='{.status.readyReplicas}'); [ "$r" = 1 ] && break; sleep 10; done
  echo "ready=$r image $($K -n kimi-k3 get deploy glm53-spark -o jsonpath='{.spec.template.spec.containers[0].image}')"
  KEY=$($K -n kimi-k3 get secret kimi-api-key -o jsonpath='{.data.LLAMA_API_KEY}' | base64 -d | cut -d, -f1)
  { printf '%s\n' "$KEY"; cat $D/sparkverify.py; } | ssh xero@192.168.4.209 'read -r K; cat > /tmp/sv.py; printf "%s\n" "$K" | python3 /tmp/sv.py 3; rm -f /tmp/sv.py'
  unset KEY;;
esac
