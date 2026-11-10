#!/usr/bin/env python3
"""Two requests with a set gap (Spark window nc-s2, the abort of leg SA3): each spec alone first (the reference md5 for
this answer length), then A is sent and B follows DELAY seconds later, REPS times per delay. A short DELAY makes B
arrive while A's prompt is read, a long one while A generates. Stops at the first request that fails (server gone).
Usage: stag.py A B --delays 1.0 0.01 --reps 3 [--n-predict 48] [--tag T]"""
import argparse, json, sys, threading, time
sys.path.insert(0, '/srv/glm-5.3-flash/nc')
import ekbench


def one(url, spec, n):
    try:
        return ekbench.one(url, spec, n)
    except Exception as e:  # connection refused or reset: the server is gone
        return {"spec": spec, "error": "exc", "body": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--delays", type=float, nargs="+", default=[1.0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--url", default="http://127.0.0.1:18104")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ref = {}
    for s in dict.fromkeys([a.a, a.b]):
        r = one(a.url, s, a.n_predict)
        print(json.dumps({"kind": "stag-solo", "tag": a.tag, "spec": s, "md5": r.get("md5"), "decode_tps": r.get("decode_tps"),
                          "error": r.get("error"), "body": r.get("body")}), flush=True)
        if "error" in r:
            return 1
        ref[s] = r["md5"]
    bad = 0
    for d in a.delays:
        for rep in range(a.reps):
            res = [None, None]

            def w(i, spec, wait):
                time.sleep(wait)
                res[i] = one(a.url, spec, a.n_predict)

            th = [threading.Thread(target=w, args=(0, a.a, 0.0)), threading.Thread(target=w, args=(1, a.b, d))]
            [x.start() for x in th]
            [x.join() for x in th]
            ok = all("error" not in r and r["md5"] == ref[r["spec"]] for r in res)
            bad += 0 if ok else 1
            print(json.dumps({"kind": "stag", "tag": a.tag, "a": a.a, "b": a.b, "delay": d, "rep": rep, "ok": ok,
                              "md5": [r.get("md5", "ERR%s" % r.get("error")) for r in res], "ref": [ref[a.a], ref[a.b]],
                              "decode_tps": [r.get("decode_tps") for r in res], "body": [r.get("body") for r in res if "error" in r]}), flush=True)
            if any("error" in r for r in res):
                return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
