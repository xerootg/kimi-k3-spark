#!/usr/bin/env python3
"""Multi-stream bench for the NUC distribution work (window nd).

Distinct prompts per stream, so the streams route to different experts (bench.py conc sends one prompt N times).
  solo  --n N    : each of the first N prompts alone, one after another (N=1 occupancy): the standalone references.
  multi --n N    : the first N prompts together (barrier start), --rounds measured rounds after one warm round.
Every request: temperature 0, seed 1, n_predict tokens, ignore_eos, cache_prompt false. One JSON line per measurement.
"""
import argparse, hashlib, json, sys, threading, time, urllib.request

PROMPTS = [
    "Write a Python quicksort function and explain it.",
    "Explain how a Kubernetes operator reconciles desired and actual state, with a short Go example.",
    "Write a short story about a lighthouse keeper who finds a stranded robot on the rocks.",
    "Summarize the main causes of the French Revolution in five bullet points, then discuss which mattered most.",
    "Describe the differences between TCP and UDP and when to use each, with examples from games and video calls.",
    "Write a haiku sequence about autumn in Seattle, then explain the imagery.",
    "Explain the math behind RSA encryption step by step with a toy example.",
    "Give a recipe for sourdough bread and explain the role of each ingredient.",
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def body(i, n_predict):
    return {"prompt": PROMPTS[i], "temperature": 0, "seed": 1, "n_predict": n_predict, "cache_prompt": False,
            "ignore_eos": True}


def solo(a):
    for i in range(a.n):
        t0 = time.time()
        d = post(a.url + "/completion", body(i, a.n_predict))
        t1 = time.time()
        t = d["timings"]
        print(json.dumps({"kind": "solo", "tag": a.tag, "p": i, "decode_tps": round(t["predicted_per_second"], 2),
                          "gen": t["predicted_n"], "md5": md5(d["content"]), "t0": t0, "t1": t1}), flush=True)


def multi(a):
    for rnd in range(a.rounds + 1):
        res = [None] * a.n
        bar = threading.Barrier(a.n)

        def w(i):
            bar.wait()
            s = time.time()
            d = post(a.url + "/completion", body(i, a.n_predict))
            res[i] = (s, time.time(), d)

        th = [threading.Thread(target=w, args=(i,)) for i in range(a.n)]
        [x.start() for x in th]
        [x.join() for x in th]
        st = min(r[0] for r in res)
        en = max(r[1] for r in res)
        gen = sum(r[2]["timings"]["predicted_n"] for r in res)
        # decode-phase aggregate: generated tokens over the span in which all streams decode (prompt ends to end)
        print(json.dumps({"kind": "multi", "tag": a.tag, "n": a.n, "round": "warm" if rnd == 0 else rnd,
                          "agg_tps": round(gen / (en - st), 2), "wall_s": round(en - st, 2), "gen": gen,
                          "per_stream_tps": [round(r[2]["timings"]["predicted_per_second"], 2) for r in res],
                          "md5": [md5(r[2]["content"]) for r in res], "t0": st, "t1": en}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["solo", "multi"])
    p.add_argument("--url", default="http://127.0.0.1:18080")
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--n-predict", type=int, default=256)
    a = p.parse_args()
    solo(a) if a.mode == "solo" else multi(a)


if __name__ == "__main__":
    sys.exit(main())
