#!/usr/bin/env python3
"""Spark no-compromise item 3 (patch 0060, guaranteed floor + burst). Long prompts from ekbench.long_prompt by token ids
(the server tokenizes once), temperature 0, seed 1, ignore_eos, cache_prompt true. One JSON line per request.
  fill P_TOK N          : prompt P (seed 11, P_TOK tokens) with n_cmpl N and 1 token: N slots end up holding P
  floor N               : P + suffix i for i < N, one after another (alone), 64 tokens each
  burst B_TOK           : prompt B (seed 13, B_TOK tokens), 1 token (prefill only, leaves B cached)
  bsolo B_TOK           : B + suffix, 64 tokens, alone
  conc N B_TOK          : P + suffix i (i < N) and B + suffix, all at once (barrier), 64 tokens each
  refuse TOK            : a fresh prompt of TOK tokens (seed 17), 16 tokens; expected 503 when the burst cannot be met
"""
import argparse, hashlib, json, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash')
import ekbench

URL = "http://127.0.0.1:18080"
SUF = ["\n\nQuestion %d: which three words appear most often in this log, and roughly how often?" % i for i in range(8)]
SUF_B = "\n\nQuestion: summarize the first and the last thousand words of this log in two sentences each."


def post(path, b, timeout=14400):
    req = urllib.request.Request(URL + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:400]}


def toks(text, special=True):
    return post("/tokenize", {"content": text, "add_special": special})["tokens"]


def long_toks(n, seed):
    import random
    rnd = random.Random(seed)
    text = "Summarize this log:\n" + " ".join(rnd.choice(ekbench.WORDS) for _ in range(int(n / 1.29)))
    return toks(text)


def comp(tag, kind, prompt, n, extra=None, label=""):
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": True, "ignore_eos": True}
    if extra:
        b.update(extra)
    t0 = time.time()
    d = post("/completion", b)
    t1 = time.time()
    if isinstance(d, list):
        d = d[0]
    if "error" in d:
        r = {"kind": kind, "tag": tag, "label": label, "n_tok": len(prompt), "error": d["error"], "body": d.get("body", ""), "t0": t0, "t1": t1}
    else:
        t = d["timings"]
        r = {"kind": kind, "tag": tag, "label": label, "n_tok": len(prompt), "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
             "prefill_s": round(t["prompt_ms"] / 1e3, 1), "prefill_tps": round(t["prompt_per_second"], 1),
             "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 2),
             "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "t0": t0, "t1": t1}
    print(json.dumps(r), flush=True)
    return r


ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("args", nargs="*", type=int)
ap.add_argument("--url", default=URL); ap.add_argument("--tag", default="")
ap.add_argument("--ptok", type=int, default=190000)
a = ap.parse_args()
URL = a.url
if a.mode == "fill":
    P = long_toks(a.args[0], 11)
    comp(a.tag, "fill", P, 1, {"n_cmpl": a.args[1]}, "P x%d" % a.args[1])
elif a.mode == "floor":
    P = long_toks(a.ptok, 11)
    for i in range(a.args[0]):
        comp(a.tag, "floor-solo", P + toks(SUF[i], False), 64, label="P+S%d" % i)
elif a.mode == "burst":
    comp(a.tag, "burst-fill", long_toks(a.args[0], 13), 1, label="B")
elif a.mode == "bsolo":
    comp(a.tag, "burst-solo", long_toks(a.args[0], 13) + toks(SUF_B, False), 64, label="B+T")
elif a.mode == "conc":
    n, btok = a.args
    P = long_toks(a.ptok, 11)
    B = long_toks(btok, 13) + toks(SUF_B, False)
    jobs = [("P+S%d" % i, P + toks(SUF[i], False)) for i in range(n)] + [("B+T", B)]
    bar = threading.Barrier(len(jobs)); res = [None] * len(jobs)
    def w(i):
        bar.wait(); res[i] = comp(a.tag, "conc", jobs[i][1], 64, label=jobs[i][0])
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(jobs))]
    [x.start() for x in th]; [x.join() for x in th]
elif a.mode == "refuse":
    comp(a.tag, "refuse", long_toks(a.args[0], 17), 16, label="R%d" % a.args[0])
else:
    sys.exit("unknown mode")
