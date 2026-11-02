#!/usr/bin/env python3
"""512k validation: A = ~450k-token prompt (cache on), B = a different ~100k prompt (evicts A from the slot into the
host prompt cache), A2 = A plus ~300 new tokens (should restore A from the cache). One JSON line per request."""
import json, random, sys, time, urllib.request
URL = "http://127.0.0.1:18101/completion"
W = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm chart "
     "argocd sync health probe container image registry runner pipeline postgres backup barman wal archive snapshot etcd "
     "raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration affinity scheduler").split()
def text(n, seed):
    r = random.Random(seed); return " ".join(r.choice(W) for _ in range(int(n / 1.29)))
def go(tag, prompt):
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": 32, "cache_prompt": True, "ignore_eos": True}
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=7200).read()); t = d["timings"]
    print(json.dumps({"tag": tag, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"), "prefill_tps": round(t["prompt_per_second"], 1),
                      "prefill_s": round(t["prompt_ms"] / 1e3, 1), "decode_tps": round(t["predicted_per_second"], 2), "wall_s": round(time.time() - t0, 1)}), flush=True)
A = "Summarize this log:\n" + text(int(sys.argv[1]), 7)
go("A-first", A)
go("B-other", "Summarize this other log:\n" + text(100000, 99))
go("A-return", A + "\nFollow-up question: " + text(300, 11))
