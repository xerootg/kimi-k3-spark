#!/usr/bin/env python3
"""(nc windows, patch 0060: prompt ubatches in pieces) Patch 0055 (per-client priority) test client for the GH200's five clients (gh200-bi window, 2026-11-09).
Derived from bi/prio_bench.py. Keys come from the class file and are never printed.
Classes: juno (priority 10); workbench x2, claude-code, hef-posters (priority 1) = the four "low" clients.
Scenarios (one JSON line per request):
  warm      : Juno p0 three times, alone
  ttft4     : four low streams generating (p1-p4), Juno's p0 at +delay
  prefill   : a low client's long prompt (L65536) being read, Juno at +delay
  mixed     : three low streams generating + a low long prompt (L65536) + Juno at +delay
  junofirst : Juno starts, four low requests arrive at +1 s (what the hold rule costs the others)
  junolong  : Juno starts (--jn tokens), a low client's long prompt (--long) arrives at +delay (item 2 of the nc task)
  nokey     : a request without a key must get 401
Every request: temperature 0, seed 1, ignore_eos, cache_prompt false (bodies of ekbench.py), streamed."""
import argparse, hashlib, json, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash')
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
        out.append({"tag": tag, "spec": spec, "error": e.code, "t_send": round(t0 - t_start, 2)}); return
    t1 = time.time()
    out.append({"tag": tag, "spec": spec, "t_send": round(t0 - t_start, 2), "ttft_s": round((first or t1) - t0, 3),
                "wall_s": round(t1 - t0, 2), "decode_tps": round(timings.get("predicted_per_second", 0), 2),
                "prompt_n": timings.get("prompt_n"), "prefill_tps": round(timings.get("prompt_per_second", 0), 1),
                "md5": hashlib.md5("".join(text).encode()).hexdigest()[:10]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario"); ap.add_argument("--classes", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18080"); ap.add_argument("--tag", default="")
    ap.add_argument("--delay", type=float, default=4.0)
    ap.add_argument("--long", default="L65536")
    ap.add_argument("--jn", type=int, default=256)
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
    elif a.scenario == "warm":
        for _ in range(3):
            run(a.url, "p0", juno, a.jn, out, "warm", T)
    elif a.scenario == "ttft4":
        for i, s in enumerate(("p1", "p2", "p3", "p4")):
            go(s, low[i], 256, "low")
        go("p0", juno, 256, "juno", a.delay)
    elif a.scenario == "prefill":
        go(a.long, low[0], 128, "low")
        go("p0", juno, a.jn, "juno", a.delay)
    elif a.scenario == "junolong":
        go("p0", juno, a.jn, "juno")
        go(a.long, low[0], 128, "low", a.delay)
    elif a.scenario == "mixed":
        for i, s in enumerate(("p1", "p2", "p3")):
            go(s, low[i], 256, "low")
        go(a.long, low[3], 128, "low", 0.5)
        go("p0", juno, 256, "juno", a.delay)
    elif a.scenario == "junofirst":
        go("p0", juno, a.jn, "juno")
        for i, s in enumerate(("p1", "p2", "p3", "p4")):
            go(s, low[i], 256, "low", 1.0)
    for t in th:
        t.join()
    for r in sorted(out, key=lambda r: r.get("t_send", 0)):
        r["scenario"] = a.scenario; r["kind"] = "prio"; r["leg"] = a.tag
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
