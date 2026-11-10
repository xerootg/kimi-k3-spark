#!/usr/bin/env python3
"""Scheduling smoke test for patches 0064/0065 on a small stock model (CPU): token timelines of streamed requests.
t.py share : A answers (300 tokens); B's long prompt arrives at +1 s
t.py turns : B's long prompt starts; C's short prompt arrives at +0.5 s
"""
import hashlib, json, random, sys, threading, time, urllib.request
URL = "http://127.0.0.1:18190"
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


def long_prompt(n, seed):
    r = random.Random(seed)
    return "Summarize this log:\n" + " ".join(r.choice(W) for _ in range(n))


def run(name, prompt, n, out, T, delay=0.0):
    time.sleep(delay)
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": False, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(URL + "/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ts = []; text = []; tm = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            d = json.loads(line[5:])
            if d.get("content"):
                ts.append(time.time()); text.append(d["content"])
            if d.get("stop"):
                tm = d.get("timings", {})
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    out[name] = {"name": name, "t_send": round(t0 - T, 2), "ttft_s": round((ts[0] if ts else time.time()) - t0, 2), "t_done": round(time.time() - T, 2),
                 "n": len(ts), "max_gap_s": round(max(gaps), 2) if gaps else None, "prompt_n": tm.get("prompt_n"),
                 "decode_tps": round(tm.get("predicted_per_second", 0), 1), "md5": hashlib.md5("".join(text).encode()).hexdigest()[:10]}


mode = sys.argv[1]; out = {}; T = time.time(); th = []
def go(*a, **k):
    t = threading.Thread(target=run, args=a, kwargs=k); t.start(); th.append(t)
if mode == "share":
    go("A", "Write a long story about a fox.", 6000, out, T)
    go("B", long_prompt(5000, 3), 16, out, T, delay=1.0)
elif mode == "turns":
    go("B", long_prompt(5000, 3), 16, out, T)
    go("C", "Name three colours.", 16, out, T, delay=0.5)
elif mode == "solo":
    for n, p, k in (("A", "Write a long story about a fox.", 6000), ("B", long_prompt(5000, 3), 16), ("C", "Name three colours.", 16)):
        run(n, p, k, out, T)
for t in th:
    t.join()
for k in sorted(out):
    out[k]["mode"] = mode
    print(json.dumps(out[k]), flush=True)
