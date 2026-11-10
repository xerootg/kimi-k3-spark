#!/usr/bin/env python3
"""Spark no-compromise item 3 (patches 0060/0062/0063: guaranteed floor + burst). Long prompts by token ids (the server
tokenizes once), temperature 0, seed 1, ignore_eos, cache_prompt true. One JSON line per request; the slot each label
ran on is kept in legs-nc/fb-state-<tag>.json so later steps can pin a request to it (id_slot).
  fill P_TOK N            : prompt P (seed 11) with n_cmpl N and 1 token: N slots end up holding P (LLAMA_SLOT_KEEP_CHILD=1)
  floor N                 : P + suffix i for i < N, one after another (alone), 64 tokens each
  floorc N [ROUNDS]       : P + suffix i for i < N, all at once (barrier), 64 tokens each
  burst B_TOK             : prompt B (seed 13), 1 token (prefill only, leaves B cached)
  burstfloor B_TOK N EVERY: burst B while rounds of N floor clients (all at once) run every EVERY seconds until B is read
  bsolo B_TOK             : B + suffix, 64 tokens, alone (on the slot of --slot-of when given)
  conc N B_TOK            : P + suffix i (i < N) and B + suffix, all at once (barrier), 64 tokens each
  newcomer TOK            : a fresh prompt (seed 17) on the slot of --slot-of (a floor client's slot), 64 tokens: under
                            the floor, so it must never be refused
  burst2 TOK CANCEL_S     : a second burst (seed 19) on the slot of --slot-of; admitted or 503; dropped after CANCEL_S
  refuse TOK              : a fresh prompt of TOK tokens (seed 17), 16 tokens
  bfirst B_TOK N DELAY EVERY : burst B first; after DELAY s the fill of P (n_cmpl N) arrives while B is being read (the
                            floor clients must map their floors while the burst holds its memory); once P is there,
                            rounds of N floor clients (all at once) every EVERY s until B is read
  release                 : a short prompt on the slot of --slot-of (frees a burst's memory)
  probe CANCEL_S TOK...   : burst admission only: each size in turn (seed 19, 1 token) on the slot of --slot-of or any;
                            a refused one answers 503 at once; the first one admitted is dropped after CANCEL_S and ends
                            the probe
  grow TOK CANCEL_S       : the burst's own conversation grown to TOK tokens (seed 13: B is its prefix) on the slot of
                            --slot-of; admitted (dropped after CANCEL_S) or a 503 that leaves B as it was
"""
import argparse, hashlib, http.client, json, os, sys, threading, time, urllib.request, urllib.error
sys.path.insert(0, '/srv/glm-5.3-flash')
sys.path.insert(1, '/srv/glm-5.3-flash/nc')
import ekbench

URL = "http://127.0.0.1:18080"
SUF = ["\n\nQuestion %d: which three words appear most often in this log, and roughly how often?" % i for i in range(8)]
SUF_B = "\n\nQuestion: summarize the first and the last thousand words of this log in two sentences each."
LOCK = threading.Lock()
STATE = None


def state_load():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def state_put(label, slot):
    with LOCK:
        s = state_load(); s[label] = slot
        json.dump(s, open(STATE + ".tmp", "w")); os.replace(STATE + ".tmp", STATE)


def post(path, b, timeout=14400):
    req = urllib.request.Request(URL + path, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode(errors="replace")[:400]}
    except Exception as e:  # timeout (the request is dropped: the server cancels the task)
        return {"error": "client", "body": "%s: %s" % (type(e).__name__, e)}


def toks(text, special=True):
    return post("/tokenize", {"content": text, "add_special": special})["tokens"]


def long_toks(n, seed):
    import random
    rnd = random.Random(seed)
    text = "Summarize this log:\n" + " ".join(rnd.choice(ekbench.WORDS) for _ in range(int(n / 1.29)))
    return toks(text)


def comp(tag, kind, prompt, n, extra=None, label="", timeout=14400, rnd=None):
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": True, "ignore_eos": True}
    if extra:
        b.update(extra)
    t0 = time.time()
    d = post("/completion", b, timeout)
    t1 = time.time()
    if isinstance(d, list):
        d = d[0]
    r = {"kind": kind, "tag": tag, "label": label, "n_tok": len(prompt)}
    if rnd is not None:
        r["round"] = rnd
    if "error" in d:
        r.update({"error": d["error"], "body": d.get("body", ""), "wall_s": round(t1 - t0, 1), "t0": t0, "t1": t1})
    else:
        t = d["timings"]
        r.update({"id_slot": d.get("id_slot"), "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"),
                  "prefill_s": round(t["prompt_ms"] / 1e3, 1), "prefill_tps": round(t["prompt_per_second"], 1),
                  "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 2),
                  "wall_s": round(t1 - t0, 1), "md5": hashlib.md5(d["content"].encode()).hexdigest()[:10], "t0": t0, "t1": t1})
        if d.get("id_slot") is not None:
            state_put(label, d["id_slot"])
    with LOCK:
        print(json.dumps(r), flush=True)
    return r


def together(tag, kind, jobs, n=64, rnd=None):
    bar = threading.Barrier(len(jobs)); res = [None] * len(jobs)
    def w(i):
        bar.wait(); res[i] = comp(tag, kind, jobs[i][1], n, label=jobs[i][0], rnd=rnd)
    th = [threading.Thread(target=w, args=(i,)) for i in range(len(jobs))]
    [x.start() for x in th]; [x.join() for x in th]
    return res


ap = argparse.ArgumentParser()
ap.add_argument("mode"); ap.add_argument("args", nargs="*", type=int)
ap.add_argument("--url", default=URL); ap.add_argument("--tag", default="")
ap.add_argument("--ptok", type=int, default=190000)
ap.add_argument("--slot-of", default="")
ap.add_argument("--state-dir", default="/srv/glm-5.3-flash/legs-nc")
a = ap.parse_args()
URL = a.url
STATE = "%s/fb-state-%s.json" % (a.state_dir, a.tag or "x")


def pin():
    if not a.slot_of:
        return None
    s = state_load().get(a.slot_of)
    return {"id_slot": s} if s is not None else None


def floor_jobs(n):
    P = long_toks(a.ptok, 11)
    return [("P+S%d" % i, P + toks(SUF[i], False)) for i in range(n)]


if a.mode == "fill":
    P = long_toks(a.args[0], 11)
    comp(a.tag, "fill", P, 1, {"n_cmpl": a.args[1]}, "P x%d" % a.args[1])
elif a.mode == "floor":
    for lab, p in floor_jobs(a.args[0]):
        comp(a.tag, "floor-solo", p, 64, label=lab)
elif a.mode == "floorc":
    jobs = floor_jobs(a.args[0])
    for r in range(a.args[1] if len(a.args) > 1 else 1):
        together(a.tag, "floor-conc", jobs, rnd=r)
elif a.mode == "burst":
    comp(a.tag, "burst-fill", long_toks(a.args[0], 13), 1, label="B")
elif a.mode == "burstfloor":
    btok, n, every = a.args
    jobs = floor_jobs(n)
    B = long_toks(btok, 13)
    done = threading.Event()
    def burst():
        comp(a.tag, "burst-fill", B, 1, label="B"); done.set()
    tb = threading.Thread(target=burst); tb.start()
    r = 0
    done.wait(20)
    while not done.is_set():
        together(a.tag, "floor-during-burst", jobs, rnd=r); r += 1
        done.wait(every)
    tb.join()
elif a.mode == "bfirst":
    btok, n, delay, every = a.args
    B = long_toks(btok, 13)
    P = long_toks(a.ptok, 11)
    jobs = [("P+S%d" % i, P + toks(SUF[i], False)) for i in range(n)]
    done = threading.Event()
    def burst():
        comp(a.tag, "burst-fill", B, 1, label="B"); done.set()
    tb = threading.Thread(target=burst); tb.start()
    done.wait(delay)
    comp(a.tag, "fill-during-burst", P, 1, {"n_cmpl": n}, "P x%d" % n)
    r = 0
    while not done.is_set():
        together(a.tag, "floor-during-burst", jobs, rnd=r); r += 1
        done.wait(every)
    tb.join()
elif a.mode == "release":
    comp(a.tag, "release", toks("Say hi."), 8, pin(), "REL")
elif a.mode == "probe":
    for t in a.args[1:]:
        r = comp(a.tag, "probe", long_toks(t, 19), 1, pin(), "PR%d" % t, timeout=a.args[0])
        if r.get("error") == "client":
            break  # admitted: the request was dropped after CANCEL_S
elif a.mode == "grow":
    comp(a.tag, "grow", long_toks(a.args[0], 13), 1, pin(), "G%d" % a.args[0], timeout=a.args[1])
elif a.mode == "bsolo":
    comp(a.tag, "burst-solo", long_toks(a.args[0], 13) + toks(SUF_B, False), 64, pin(), label="B+T")
elif a.mode == "conc":
    n, btok = a.args
    together(a.tag, "conc", floor_jobs(n) + [("B+T", long_toks(btok, 13) + toks(SUF_B, False))])
elif a.mode == "newcomer":
    comp(a.tag, "newcomer", long_toks(a.args[0], 17), 64, pin(), "N%d" % a.args[0])
elif a.mode == "burst2":
    comp(a.tag, "burst2", long_toks(a.args[0], 19), 1, pin(), "B2-%d" % a.args[0], timeout=a.args[1])
elif a.mode == "refuse":
    comp(a.tag, "refuse", long_toks(a.args[0], 17), 16, label="R%d" % a.args[0])
else:
    sys.exit("unknown mode")
