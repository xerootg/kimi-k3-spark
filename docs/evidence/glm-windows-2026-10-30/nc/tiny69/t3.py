#!/usr/bin/env python3
"""Rule A of patch 0069 on a small hybrid model: a prompt ubatch whose recurrent state is not in place (the slot got its
state by a sequence copy: n_cmpl with LLAMA_SLOT_KEEP_CHILD=1) must run whole; the ubatches after it may be cut again.
  ref : fill P with n_cmpl 3 (1 token), then P + long suffix on a child's slot, alone
  mix : the same, but two other slots answer while the child's request is read
"""
import hashlib, json, random, sys, threading, time, urllib.request
URL = "http://127.0.0.1:18190"
W = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()


def words(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(W) for _ in range(n))


P = "Summarize this log:\n" + words(2500, 5)
PS = P + "\nSecond part:\n" + words(4000, 7)


def post(b):
    req = urllib.request.Request(URL + "/completion", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def comp(name, prompt, n, out, extra=None, delay=0.0):
    time.sleep(delay)
    b = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": n, "cache_prompt": True, "ignore_eos": True}
    b.update(extra or {})
    t0 = time.time()
    try:
        d = post(b)
    except Exception as e:
        out[name] = {"name": name, "error": "%s: %s" % (type(e).__name__, e)}; return
    L = d if isinstance(d, list) else [d]
    t = L[-1]["timings"]
    out[name] = {"name": name, "slots": [x.get("id_slot") for x in L], "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
                 "wall_s": round(time.time() - t0, 2), "decode_tps": round(t.get("predicted_per_second", 0), 1),
                 "md5": hashlib.md5(L[-1]["content"].encode()).hexdigest()[:10]}


mode = sys.argv[1]; out = {}
comp("fill", P, 1, out, {"n_cmpl": 3})
print(json.dumps(out["fill"]), flush=True)
slots = out["fill"]["slots"]
child = slots[-1]
free = [s for s in range(5) if s not in slots]
th = []
if mode == "mix":
    for i, s in enumerate(free[:2]):
        t = threading.Thread(target=comp, args=("gen%d" % i, "Write a long story about a fox." if i == 0 else "List one hundred animals.", 4000, out, {"id_slot": s, "cache_prompt": False})); t.start(); th.append(t)
    time.sleep(1.0)
comp("child", PS, 64, out, {"id_slot": child})
for t in th:
    t.join()
for k in sorted(out):
    if k != "fill":
        out[k]["mode"] = mode; print(json.dumps(out[k]), flush=True)
