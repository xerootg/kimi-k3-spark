#!/usr/bin/env python3
"""sc67sum.py results.jsonl [power dir]: per run (canary process) the md5 verdict against the Spark gold and tok/s."""
import json, sys, collections, os
GOLD = {"qs128": "ee079496a4", "p0": "ce2fc95752", "p1": "bb64203d71", "p2": "2d592efcb6", "p3": "1aab74269d",
        "p4": "0c3f28294b", "L16384": "145c14aa8b"}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
runs = collections.OrderedDict()
for r in rows:
    runs.setdefault(r.get("run"), []).append(r)
for run, rs in runs.items():
    bad = [(r["scenario"], r["spec"], r.get("md5"), r.get("error")) for r in rs
           if "spec" in r and r["scenario"] != "auth" and r.get("md5") != GOLD.get(r["spec"])]
    n = sum(1 for r in rs if "spec" in r and r["scenario"] != "auth")
    auth = [r.get("error") for r in rs if r.get("scenario") == "auth"]
    print("== %s: %d requests, %s; auth %s" % (run, n, "ALL GOLD" if not bad else "NOT GOLD %s" % bad, auth))
    for r in rs:
        if r["scenario"] in ("gold", "solo"):
            print("   %-5s %-7s %-6s ttft %.2f s  %.2f tok/s" % (r["scenario"], r["spec"], r.get("tag"), r.get("ttft_s", 0), r.get("decode_tps", 0)))
        elif r["scenario"] == "multi-agg":
            print("   multi n=%d round %d: aggregate %.2f tok/s" % (r["n"], r["round"], r["agg_tps"]))
    for sc in ("five", "junofirst"):
        by = collections.defaultdict(list)
        for r in rs:
            if r["scenario"] == sc and "md5" in r:
                by[(r.get("round", 1), r["tag"])].append(r)
        for (rnd, tag), xs in sorted(by.items()):
            print("   %s round %d %-4s: ttft %.2f s, %.2f tok/s per stream (%d)" % (sc, rnd, tag,
                  sum(x["ttft_s"] for x in xs) / len(xs), sum(x["decode_tps"] for x in xs) / len(xs), len(xs)))
