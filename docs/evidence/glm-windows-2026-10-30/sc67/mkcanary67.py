#!/usr/bin/env python3
"""mkcanary67.py SRC IMAGE OUT [NAME=VALUE ...]: the glm53-spark Deployment of SRC (argo-things
configs/kimi-k3/glm53-spark.yaml) as glm53-spark-canary: own label and no kimi-k3/spark-server label (so no Service
traffic), the given image, the snapshot hostPath on a copy, and extra env appended (or replaced by name)."""
import sys, yaml
src, image, out = sys.argv[1:4]
d = [x for x in yaml.safe_load_all(open(src)) if x and x.get("kind") == "Deployment"][0]
L = {"app.kubernetes.io/name": "glm53-spark-canary"}
d["metadata"]["name"] = "glm53-spark-canary"
d["metadata"]["labels"] = dict(L)
d["metadata"].pop("annotations", None)
d["spec"]["selector"]["matchLabels"] = dict(L)
d["spec"]["template"]["metadata"]["labels"] = dict(L)
c = d["spec"]["template"]["spec"]["containers"][0]
c["image"] = image
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1)
    c["env"] = [e for e in c["env"] if e["name"] != k] + [{"name": k, "value": v}]
for v in d["spec"]["template"]["spec"]["volumes"]:
    if v["name"] == "snapshot":
        v["hostPath"]["path"] = "/srv/glm-5.3-flash/snapshot-canary"
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
