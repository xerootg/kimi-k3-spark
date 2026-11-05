#!/usr/bin/env python3
"""Flash on the GH200: 24k cold, +300 cached step, 100k cold, 200k cold, standard request (md5), 3 x 256-token decodes.
One JSON line each. usage: flash_client.py tag [port]"""
import hashlib, json, random, sys, time, urllib.request
tag = sys.argv[1]; port = sys.argv[2] if len(sys.argv) > 2 else "18104"
URL = "http://127.0.0.1:%s/completion" % port
W = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap helm chart "
     "argocd sync health probe container image registry runner pipeline postgres backup barman wal archive snapshot etcd "
     "raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration affinity scheduler").split()
def text(n, seed): r = random.Random(seed); return " ".join(r.choice(W) for _ in range(int(n / 1.29)))
def go(label, req):
    t0 = time.time(); d = json.loads(urllib.request.urlopen(urllib.request.Request(URL, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=7200).read()); t = d["timings"]
    print(json.dumps({"tag": tag, "req": label, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"), "prefill_tps": round(t["prompt_per_second"], 1),
                      "prefill_s": round(t["prompt_ms"] / 1e3, 2), "decode_tps": round(t["predicted_per_second"], 2), "wall_s": round(time.time() - t0, 1),
                      "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10]}), flush=True)
A = "Summarize this log:\n" + text(24000, 7)
go("24k-cold", {"prompt": A, "temperature": 0, "n_predict": 16, "cache_prompt": True, "ignore_eos": True})
go("step-300", {"prompt": A + "\nTool output: " + text(300, 11), "temperature": 0, "n_predict": 16, "cache_prompt": True, "ignore_eos": True})
go("100k-cold", {"prompt": "Summarize this other log:\n" + text(100000, 99), "temperature": 0, "n_predict": 16, "cache_prompt": False, "ignore_eos": True})
go("200k-cold", {"prompt": "Summarize this third log:\n" + text(200000, 123), "temperature": 0, "n_predict": 32, "cache_prompt": False, "ignore_eos": True})
go("std", {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128, "cache_prompt": False})
for r in range(3):
    go("dec256-%d" % r, {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 256, "cache_prompt": False, "ignore_eos": True})
