#!/usr/bin/env python3
"""Needle check for KV placement: a fact near the start of a ~N-token prompt, the question at the end. The answer
needs attention (and the indexer's selection) to reach rows written ~N tokens earlier. Prints one JSON line.
Usage: needle.py URL TAG TOKENS"""
import hashlib, json, random, sys, time, urllib.request

url, tag, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm "
     "chart argocd sync health probe container image registry runner pipeline postgres backup barman wal archive").split()
r = random.Random(11)
filler = " ".join(r.choice(W) for _ in range(int(tokens / 1.29)))
prompt = ("Remember this: the access code for vault seven is 4729-ALPHA-PLUM.\n\nLog follows:\n" + filler +
          "\n\nEnd of log. Question: what is the access code for vault seven? Answer with the code only.\nAnswer:")
body = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": 24, "cache_prompt": False}
t0 = time.time()
req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=7200).read())
t = d["timings"]
print(json.dumps({"kind": "needle", "tag": tag, "prompt_n": t["prompt_n"], "decode_tps": round(t["predicted_per_second"], 2),
                  "found": "4729-ALPHA-PLUM" in d["content"].upper().replace(" ", ""), "content": d["content"][:80],
                  "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "t0": t0, "t1": time.time()}), flush=True)
