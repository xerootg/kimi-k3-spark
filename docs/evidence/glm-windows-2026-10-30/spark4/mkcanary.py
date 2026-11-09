import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
hold = sys.argv[2]  # "1" or "0"
d["metadata"]["name"] = "glm53-spark-canary"
d["metadata"]["labels"] = {"app.kubernetes.io/name": "glm53-spark-canary"}
d["metadata"].pop("annotations", None)
d["spec"]["selector"]["matchLabels"] = {"app.kubernetes.io/name": "glm53-spark-canary"}
d["spec"]["template"]["metadata"]["labels"] = {"app.kubernetes.io/name": "glm53-spark-canary"}  # no kimi-k3/spark-server: no Service traffic
c = d["spec"]["template"]["spec"]["containers"][0]
for e in c["env"]:
    if e["name"] == "LLAMA_PRIORITY_HOLD_PROMPTS":
        e["value"] = hold
for v in d["spec"]["template"]["spec"]["volumes"]:
    if v["name"] == "snapshot":
        v["hostPath"]["path"] = "/srv/glm-5.3-flash/snapshot-canary"
yaml.safe_dump(d, open(sys.argv[3], "w"), sort_keys=False)
