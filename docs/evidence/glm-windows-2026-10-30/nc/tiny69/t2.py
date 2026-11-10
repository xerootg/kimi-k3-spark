#!/usr/bin/env python3
"""Recurrent-cell test for patch 0069 on a small hybrid model (LFM2-350M, CPU): requests pinned to slots (id_slot), streamed.
Every request: temperature 0, seed 1, ignore_eos, cache_prompt false.
  solo     : each request of the scenarios below alone, one after another (the reference bytes and state hashes)
  straddle : A answers on slot 0; B's long prompt on slot 2 at +1 s; C answers on slot 1 from +2 s. C's cell lands
             above B's, so a decode step of A and C spans B's cell while B's ubatch is between pieces.
  mixed    : A0, A1, A2 answer on slots 0-2; B's long prompt on slot 3 at +1 s; C answers on slot 4 from +2 s.
  late     : B's long prompt on slot 0 first; A answers on slot 1 from +1 s and C on slot 2 from +2 s (new sequences
             get their cells while B is pending).
"""
import hashlib, json, random, sys, threading, time, urllib.request
URL = "http://127.0.0.1:18190"
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
NA = int(sys.argv[3]) if len(sys.argv) > 3 else 1500


def long_prompt(n, seed):
    r = random.Random(seed)
    return "Summarize this log:\n" + " ".join(r.choice(W) for _ in range(n))


REQ = {
    "A":  ("Write a long story about a fox.", NA),
    "A0": ("Write a long story about a fox.", NA),
    "A1": ("Write a long poem about the sea.", NA),
    "A2": ("Explain how a steam engine works, in detail.", NA),
    "C":  ("List one hundred animals with a sentence about each.", NA),
    "B":  (long_prompt(NB, 3), 64),
}


def run(name, slot, out, T, delay=0.0):
    time.sleep(delay)
    prompt, n = REQ[name]
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": False, "ignore_eos": True, "stream": True, "id_slot": slot}
    req = urllib.request.Request(URL + "/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); ts = []; text = []; tm = {}
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                d = json.loads(line[5:])
                if d.get("content"):
                    ts.append(time.time()); text.append(d["content"])
                if d.get("stop"):
                    tm = d.get("timings", {})
    except Exception as e:
        out[name] = {"name": name, "slot": slot, "error": "%s: %s" % (type(e).__name__, e)}
        return
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    out[name] = {"name": name, "slot": slot, "t_send": round(t0 - T, 2), "ttft_s": round((ts[0] if ts else time.time()) - t0, 2),
                 "t_done": round(time.time() - T, 2), "n": len(ts), "max_gap_s": round(max(gaps), 2) if gaps else None,
                 "prompt_n": tm.get("prompt_n"), "decode_tps": round(tm.get("predicted_per_second", 0), 1),
                 "md5": hashlib.md5("".join(text).encode()).hexdigest()[:10]}


SC = {
    "straddle": [("A", 0, 0.0), ("B", 2, 1.0), ("C", 1, 2.0)],
    "mixed":    [("A0", 0, 0.0), ("A1", 1, 0.0), ("A2", 2, 0.0), ("B", 3, 1.0), ("C", 4, 2.0)],
    "late":     [("B", 0, 0.0), ("A", 1, 1.0), ("C", 2, 2.0)],
}
mode = sys.argv[1]; out = {}; T = time.time(); th = []
if mode == "solo":
    for name, slot in (("A", 0), ("A1", 1), ("A2", 2), ("C", 1), ("B", 2)):
        run(name, slot, out, T)
else:
    for name, slot, delay in SC[mode]:
        t = threading.Thread(target=run, args=(name, slot, out, T, delay)); t.start(); th.append(t)
for t in th:
    t.join()
for k in sorted(out):
    out[k]["mode"] = mode
    print(json.dumps(out[k]), flush=True)
