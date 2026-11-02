#!/usr/bin/env python3
"""GH200 GLM-5.3 throughput client. Talks to a local llama-server; prints one JSON line per measurement.

Modes:
  depth  --lens 0,4096,...  : one request per length (prompt of ~L tokens, cache_prompt false), n_predict tokens.
                              Records prefill tok/s, decode tok/s, prompt_n, md5.
  conc   --n N --rounds R   : N concurrent standard requests (10-token prompt), n_predict tokens each, one warm
                              round then R measured rounds. Aggregate tok/s = generated tokens / round wall time
                              (first start to last end). Records per-client decode tok/s and md5.
Power comes from a separate sampler; this script prints wall-clock start/end (unix seconds) per measurement so
the runner can join power samples to measurement windows.
"""
import argparse
import hashlib
import json
import random
import sys
import threading
import time
import urllib.request

STD = "Write a Python quicksort function and explain it."
WORDS = ("kubernetes pod node volume longhorn garage traefik ingress service deployment replica secret configmap "
         "helm chart argocd sync health probe container image registry runner pipeline postgres backup barman wal "
         "archive snapshot etcd raft leader follower kubelet cgroup memory cpu throttle oom eviction taint toleration "
         "affinity scheduler controller operator webhook certificate issuer acme dns").split()


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def long_prompt(tokens, seed=7):
    rnd = random.Random(seed)
    n_words = int(tokens / 1.29)
    return "Summarize this log:\n" + " ".join(rnd.choice(WORDS) for _ in range(n_words))


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()[:10]


def depth(args):
    for L in [int(x) for x in args.lens.split(",")]:
        prompt = STD if L == 0 else long_prompt(L)
        body = {"prompt": prompt, "temperature": 0, "seed": 1, "n_predict": args.n_predict, "cache_prompt": False,
                "ignore_eos": True}
        t0 = time.time()
        d = post(args.url + "/completion", body)
        t1 = time.time()
        t = d["timings"]
        print(json.dumps({"kind": "depth", "tag": args.tag, "target": L, "prompt_n": t["prompt_n"],
                          "prefill_tps": round(t["prompt_per_second"], 2), "prefill_s": round(t["prompt_ms"] / 1e3, 2),
                          "decode_n": t["predicted_n"], "decode_tps": round(t["predicted_per_second"], 2),
                          "md5": md5(d["content"]), "t0": t0, "t1": t1}), flush=True)


def conc(args):
    body = {"prompt": STD, "temperature": 0, "seed": 1, "n_predict": args.n_predict, "cache_prompt": False,
            "ignore_eos": True}
    for rnd in range(args.rounds + 1):
        res = [None] * args.n
        barrier = threading.Barrier(args.n)

        def worker(i):
            barrier.wait()
            s = time.time()
            d = post(args.url + "/completion", body)
            res[i] = (s, time.time(), d)

        th = [threading.Thread(target=worker, args=(i,)) for i in range(args.n)]
        for x in th:
            x.start()
        for x in th:
            x.join()
        start = min(r[0] for r in res)
        end = max(r[1] for r in res)
        gen = sum(r[2]["timings"]["predicted_n"] for r in res)
        per = [round(r[2]["timings"]["predicted_per_second"], 2) for r in res]
        md5s = [md5(r[2]["content"]) for r in res]
        print(json.dumps({"kind": "conc", "tag": args.tag, "n": args.n, "round": "warm" if rnd == 0 else rnd,
                          "agg_tps": round(gen / (end - start), 2), "wall_s": round(end - start, 2), "gen": gen,
                          "per_stream_tps": per, "md5": md5s, "t0": start, "t1": end}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["depth", "conc"])
    p.add_argument("--url", default="http://127.0.0.1:18100")
    p.add_argument("--tag", default="")
    p.add_argument("--lens", default="0")
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--n-predict", type=int, default=256)
    args = p.parse_args()
    depth(args) if args.mode == "depth" else conc(args)


if __name__ == "__main__":
    sys.exit(main())
