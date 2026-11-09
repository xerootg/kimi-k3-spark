#!/usr/bin/env python3
"""Patch 0055 (per-client priority) test client. Keys are read from the class file and never printed.
Scenarios (each prints one JSON line per request plus a summary):
  ttft3   : three low-priority streams generating, then Juno's request; Juno's time to first token
  queue4  : four low-priority requests (one must wait for a slot with LLAMA_SLOT_RESERVE=1), then Juno
  prefill : a low-priority 16k prompt in its prefill, then Juno
  nokey   : a request without a key must get 401
Every request: temperature 0, seed 1, ignore_eos, cache_prompt false (same bodies as ekbench.py)."""
import argparse, hashlib, json, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '.')
import ekbench

def load_keys(path):
    keys = {}
    for line in open(path):
        line = line.split('#')[0].split()
        if len(line) >= 3:
            keys.setdefault(line[2], []).append(line[0])
    return keys

def run(url, spec, key, n_predict, out, tag, t_start):
    b = ekbench.body(spec, n_predict)
    b["stream"] = True
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/completion", data=json.dumps(b).encode(), headers=hdr)
    t0 = time.time(); first = None; text = []; timings = {}
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                d = json.loads(line[5:])
                if d.get("content"):
                    if first is None:
                        first = time.time()
                    text.append(d["content"])
                if d.get("stop"):
                    timings = d.get("timings", {})
    except urllib.error.HTTPError as e:
        out.append({"tag": tag, "spec": spec, "error": e.code}); return
    t1 = time.time()
    out.append({"tag": tag, "spec": spec, "t_send": round(t0 - t_start, 2), "ttft_s": round((first or t1) - t0, 3),
                "wall_s": round(t1 - t0, 2), "decode_tps": round(timings.get("predicted_per_second", 0), 2),
                "prompt_n": timings.get("prompt_n"), "md5": hashlib.md5("".join(text).encode()).hexdigest()[:10]})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario"); ap.add_argument("--classes", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18080"); ap.add_argument("--tag", default="")
    ap.add_argument("--delay", type=float, default=4.0)
    a = ap.parse_args()
    k = load_keys(a.classes)
    low = k["workbench"] + k["claude-code"] + k["hef-posters"]
    juno = k["juno"][0]
    out = []; th = []; T = time.time()
    def go(spec, key, n, tag, delay=0.0):
        def f():
            time.sleep(delay); run(a.url, spec, key, n, out, tag, T)
        t = threading.Thread(target=f); t.start(); th.append(t)
    if a.scenario == "nokey":
        run(a.url, "p0", None, 8, out, "nokey", T)
    elif a.scenario == "solo":
        for s in ("p0", "p1", "p2", "p3"):
            run(a.url, s, juno if s == "p0" else low[0], 256, out, "solo", T)
        run(a.url, "L16384", low[0], 128, out, "solo", T)
    elif a.scenario == "warm":
        for _ in range(3):
            run(a.url, "p0", juno, 256, out, "warm", T)
    elif a.scenario == "ttft3":
        for i, s in enumerate(("p1", "p2", "p3")):
            go(s, low[i], 256, "low")
        go("p0", juno, 256, "juno", a.delay)
    elif a.scenario == "queue4":
        for i, s in enumerate(("p1", "p2", "p3", "L16384")):
            go(s, low[i % len(low)], 256 if s[0] == "p" else 128, "low", 0.3 * i)
        go("p0", juno, 256, "juno", a.delay)
    elif a.scenario == "prefill":
        go("L16384", low[0], 128, "low")
        go("p0", juno, 256, "juno", a.delay)
    for t in th:
        t.join()
    for r in sorted(out, key=lambda r: r.get("t_send", 0)):
        r["scenario"] = a.scenario; r["run"] = a.tag
        print(json.dumps(r), flush=True)

if __name__ == "__main__":
    main()
