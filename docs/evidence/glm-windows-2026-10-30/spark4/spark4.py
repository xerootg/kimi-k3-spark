#!/usr/bin/env python3
"""Canary client for the four-slot glm53-spark config (argo-things feat/glm53-spark-four-slots, 2026-11-09).
Keys come from a 0600 file with lines "<name> <key>" (names: shared, juno) and are never printed.
Every request streams /completion (temperature 0, seed 1, ignore_eos, cache_prompt false unless --cache) with the
bodies of ekbench.py (pN = mbench prompt N, 256 tokens; LK = ~K-token log prompt, 128 tokens); qs128 = the Service
check (quicksort, 128 tokens, no ignore_eos). One JSON line per request: time to first token, decode tok/s, md5.
Scenarios:
  gold [--n N] [--who shared|juno]   qs128 and p0, N times each
  solo SPEC...                        each alone (shared key)
  multi SPEC... [--rounds R]          all together, barrier start (first spec on Juno's key if --juno-first-spec)
  ttft3                               p1 p2 p3 generating (shared), Juno p0 at +--delay s
  prefill                             L16384 being read (shared), Juno p0 at +--delay s
  junofirst                           Juno p0, then p1 p2 p3 at +1 s (shared)
  auth                                no key and a wrong key must get 401
  mem SPEC...                         all together with cache_prompt true (slots stay grown); LK:tag, @juno
"""
import argparse, hashlib, json, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash')
import ekbench

QS = {"prompt": "Write a Python quicksort function and explain it.", "temperature": 0, "seed": 1, "n_predict": 128,
      "cache_prompt": False}


def body(spec, cache):
    if spec == "qs128":
        return dict(QS)
    if ":" in spec:  # LK:tag = a long prompt with its own first line, so slots share no prefix
        k, tag = spec.split(":", 1)
        b = ekbench.body(k, 0); b["prompt"] = "Conversation %s. " % tag + b["prompt"]; b["cache_prompt"] = cache
        return b
    ekbench.CACHE = cache
    return ekbench.body(spec, 0)


def run(url, spec, key, out, tag, t_start, cache=False):
    b = body(spec, cache); b["stream"] = True
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/completion", data=json.dumps(b).encode(), headers=hdr)
    t0 = time.time(); first = None; text = []; tm = {}
    try:
        with urllib.request.urlopen(req, timeout=10800) as r:
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
                    tm = d.get("timings", {})
    except urllib.error.HTTPError as e:
        out.append({"tag": tag, "spec": spec, "t_send": round(t0 - t_start, 2), "error": e.code,
                    "body": e.read().decode(errors="replace")[:160]})
        return
    except Exception as e:  # connection reset etc.
        out.append({"tag": tag, "spec": spec, "t_send": round(t0 - t_start, 2), "error": repr(e)[:160]})
        return
    t1 = time.time()
    out.append({"tag": tag, "spec": spec, "t_send": round(t0 - t_start, 2), "ttft_s": round((first or t1) - t0, 3),
                "wall_s": round(t1 - t0, 2), "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
                "prefill_tps": round(tm.get("prompt_per_second", 0), 1), "decode_n": tm.get("predicted_n"),
                "decode_tps": round(tm.get("predicted_per_second", 0), 2),
                "md5": hashlib.md5("".join(text).encode()).hexdigest()[:10]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario"); ap.add_argument("specs", nargs="*")
    ap.add_argument("--keys", required=True); ap.add_argument("--url", required=True)
    ap.add_argument("--tag", default=""); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--who", default="shared"); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--delay", type=float, default=4.0)
    a = ap.parse_args()
    keys = dict(l.split()[:2] for l in open(a.keys) if l.strip())
    sh, juno = keys["shared"], keys["juno"]
    T = time.time()
    res = []

    def emit(rows, extra=None):
        for r in sorted(rows, key=lambda r: r.get("t_send", 0)):
            r["scenario"] = a.scenario; r["run"] = a.tag
            if extra:
                r.update(extra)
            print(json.dumps(r), flush=True)

    def together(jobs, extra=None):
        out = []; th = []
        for spec, key, tag, delay, cache in jobs:
            def f(spec=spec, key=key, tag=tag, delay=delay, cache=cache):
                time.sleep(delay); run(a.url, spec, key, out, tag, T, cache)
            t = threading.Thread(target=f); t.start(); th.append(t)
        [t.join() for t in th]
        emit(out, extra)
        return out

    if a.scenario == "gold":
        k = juno if a.who == "juno" else sh
        for spec in ("qs128", "p0"):
            for _ in range(a.n):
                out = []; run(a.url, spec, k, out, a.who, T); emit(out)
    elif a.scenario == "solo":
        for spec in a.specs:
            out = []; run(a.url, spec, sh, out, "solo", T); emit(out)
    elif a.scenario == "multi":
        for rnd in range(a.rounds):
            rows = together([(s, sh, "multi", 0.0, False) for s in a.specs], {"round": rnd + 1, "n": len(a.specs)})
            ok = [r for r in rows if "md5" in r]
            if ok:
                st = min(r["t_send"] for r in ok); en = max(r["t_send"] + r["wall_s"] for r in ok)
                print(json.dumps({"scenario": "multi-agg", "run": a.tag, "round": rnd + 1, "n": len(a.specs),
                                  "agg_tps": round(sum(r["decode_n"] or 0 for r in ok) / (en - st), 2),
                                  "md5": [r["md5"] for r in sorted(ok, key=lambda r: a.specs.index(r["spec"]))]}), flush=True)
    elif a.scenario == "ttft3":
        together([("p1", sh, "low", 0, False), ("p2", sh, "low", 0, False), ("p3", sh, "low", 0, False),
                  ("p0", juno, "juno", a.delay, False)])
    elif a.scenario == "prefill":
        together([("L16384", sh, "low", 0, False), ("p0", juno, "juno", a.delay, False)])
    elif a.scenario == "junofirst":
        together([("p0", juno, "juno", 0, False), ("p1", sh, "low", 1, False), ("p2", sh, "low", 1, False),
                  ("p3", sh, "low", 1, False)])
    elif a.scenario == "auth":
        out = []
        run(a.url, "p0", None, out, "nokey", T)
        run(a.url, "p0", "wrong-" + "x" * 20, out, "wrongkey", T)
        emit(out)
    elif a.scenario == "mem":
        # a spec ending in @juno uses Juno's key (the reserved slot), so four slots grow at once
        together([(s.split("@")[0], juno if s.endswith("@juno") else sh, "mem", 0.5 * i, True)
                  for i, s in enumerate(a.specs)])


if __name__ == "__main__":
    main()
