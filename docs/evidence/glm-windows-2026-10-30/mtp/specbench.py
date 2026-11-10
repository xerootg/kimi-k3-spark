#!/usr/bin/env python3
"""Patch 0068 (exact speculative decoding) client. From ekbench.py, with sampling options and draft counters.
Request specs: pN = prompt N (256 tokens), LK = long prompt of ~K tokens (seed 7, 128 tokens); pN:T or LK:T
generates T tokens (clients that finish at different times change the number of generating slots mid-run).
  solo   SPECS...            : each spec alone, one after another
  multi  SPECS... --rounds R : all specs together (barrier start), one warm round + R measured rounds
  follow SPEC                : turn 1 (64 tokens, cache_prompt true), then turn 1's prompt + its answer + a question
                               (cache_prompt true): the second turn reuses the slot's context after speculation
  stop   SPEC                : the request with a stop string and without ignore_eos (ends inside a verified batch)
One JSON line per measurement. md5 is over the generated text.
"""
import argparse, hashlib, json, random, sys, threading, time, urllib.request, urllib.error

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
WORDS = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap "
         "helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal "
         "archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration "
         "affinity scheduler controller operator webhook certificate issuer acme dns").split()
A = None


def long_prompt(tokens, seed=7):
    rnd = random.Random(seed)
    return "Summarize this log:\n" + " ".join(rnd.choice(WORDS) for _ in range(int(tokens / 1.29)))


def prompt(spec):
    spec = spec.split(":")[0]
    return PROMPTS[int(spec[1:])] if spec[0] == "p" else long_prompt(int(spec[1:]))


def body(spec, n_predict=0, text=None, cache=False, ignore_eos=True):
    b = {"prompt": text if text is not None else prompt(spec), "temperature": A.temperature, "seed": A.seed,
         "n_predict": n_predict or (int(spec.split(":")[1]) if ":" in spec else 0) or A.n_predict or (256 if spec[0] == "p" else 128),
         "cache_prompt": cache,
         "ignore_eos": ignore_eos}
    if A.top_p is not None:
        b["top_p"] = A.top_p
    if A.n_probs:
        b["n_probs"] = A.n_probs
    return b


def post(url, b, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": e.code, "body": e.read().decode()[:300]}


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def one(spec, b=None):
    s = time.time()
    d = post(A.url + "/completion", b or body(spec))
    e = time.time()
    if "error" in d:
        return {"spec": spec, "error": d["error"], "body": d["body"], "t0": s, "t1": e}
    t = d["timings"]
    return {"spec": spec, "prompt_n": t["prompt_n"], "cache_n": t.get("cache_n"), "prefill_s": round(t["prompt_ms"] / 1e3, 2),
            "prefill_tps": round(t["prompt_per_second"], 2), "decode_n": t["predicted_n"],
            "decode_tps": round(t["predicted_per_second"], 2), "md5": md5(d["content"]),
            "draft_n": t.get("draft_n"), "draft_ok": t.get("draft_n_accepted"),
            "stop": d.get("stop_type"), "t0": s, "t1": e, "_content": d["content"]}


def out(r, **kw):
    r = {k: v for k, v in r.items() if k != "_content"}
    r.update(kw)
    r.update({"tag": A.tag, "temp": A.temperature, "seed": A.seed})
    print(json.dumps(r), flush=True)


def main():
    global A
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["solo", "multi", "follow", "stop"])
    p.add_argument("specs", nargs="+")
    p.add_argument("--url", default="http://127.0.0.1:18080")
    p.add_argument("--tag", default="")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-probs", type=int, default=0)
    A = p.parse_args()
    if A.mode == "solo":
        for spec in A.specs:
            out(one(spec), kind="solo")
        return
    if A.mode == "follow":
        for spec in A.specs:
            r1 = one(spec, body(spec, 64, cache=True))
            out(r1, kind="follow1")
            if "error" in r1:
                continue
            text = prompt(spec) + r1["_content"] + "\n\nNow list three follow-up questions."
            out(one(spec, body(spec, 96, text=text, cache=True)), kind="follow2")
        return
    if A.mode == "stop":
        for spec in A.specs:
            b = body(spec, 400, ignore_eos=False)
            b["stop"] = ["\n\n\n", "pivot", "In summary"]
            out(one(spec, b), kind="stop")
            b = body(spec, 37, ignore_eos=False)
            out(one(spec, b), kind="limit37")
        return
    for rnd in range(A.rounds + 1):
        res = [None] * len(A.specs)
        bar = threading.Barrier(len(A.specs))

        def w(i):
            bar.wait()
            res[i] = one(A.specs[i])

        th = [threading.Thread(target=w, args=(i,)) for i in range(len(A.specs))]
        [x.start() for x in th]
        [x.join() for x in th]
        st = min(r["t0"] for r in res)
        en = max(r["t1"] for r in res)
        gen = sum(r.get("decode_n", 0) for r in res)
        print(json.dumps({"kind": "multi", "tag": A.tag, "temp": A.temperature, "seed": A.seed, "n": len(A.specs),
                          "round": "warm" if rnd == 0 else rnd, "agg_tps": round(gen / (en - st), 2),
                          "wall_s": round(en - st, 2), "gen": gen, "per_stream_tps": [r.get("decode_tps") for r in res],
                          "md5": [r.get("md5", "ERR%s" % r.get("error")) for r in res], "specs": A.specs,
                          "draft_n": [r.get("draft_n") for r in res], "draft_ok": [r.get("draft_ok") for r in res],
                          "prefill_tps": [r.get("prefill_tps") for r in res], "t0": st, "t1": en}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
