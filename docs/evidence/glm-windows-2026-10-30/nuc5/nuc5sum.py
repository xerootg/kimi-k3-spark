#!/usr/bin/env python3
"""Summarise legs-nuc5/results.jsonl: every client's md5 per leg against the gold table, and tok/s per leg and N."""
import collections, json, sys
GOLD = {"p0": "ce2fc95752", "p1": "bb64203d71", "p2": "2d592efcb6", "p3": "1aab74269d", "L16384": "145c14aa8b", "p0@128": "ee079496a4"}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
md = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))   # tag -> (kind, spec) -> md5 counts
sp = collections.defaultdict(lambda: collections.defaultdict(list))                   # tag -> key -> values
for r in rows:
    t = r.get("tag") or r.get("leg")
    if r.get("kind") == "solo":
        spec = r["spec"] + ("@128" if r["spec"] == "p0" and r["decode_n"] == 128 else "")
        md[t][("solo", spec)][r.get("md5")] += 1
        sp[t]["solo " + spec].append(r["decode_tps"])
    elif r.get("kind") == "multi":
        mix = "L16384" in r["specs"]
        for s, m in zip(r["specs"], r["md5"]):
            md[t][("mix" if mix else "N=%d" % r["n"], s)][m] += 1
        if r["round"] != "warm" or mix:
            sp[t][("mix " if mix else "") + "N=%d agg" % r["n"]].append(r["agg_tps"])
            sp[t][("mix " if mix else "") + "N=%d per stream" % r["n"]].append(sum(x for x in r["per_stream_tps"] if x) / r["n"])
    elif r.get("kind") == "prio":
        md[t][(r["scenario"], r["tag"] + " " + r["spec"])][r.get("md5", "ERR%s" % r.get("error"))] += 1
        if "ttft_s" in r:
            sp[t]["%s %s ttft_s" % (r["scenario"], r["tag"])].append(r["ttft_s"])
            sp[t]["%s %s tps" % (r["scenario"], r["tag"])].append(r["decode_tps"])
tags = sorted(md)
print("== bytes (md5 per client; '=' gold, otherwise the value) ==")
keys = sorted({k for t in tags for k in md[t]})
ref = {}
for k in keys:
    cells = []
    for t in tags:
        c = md[t].get(k)
        if not c:
            cells.append("-"); continue
        g = GOLD.get(k[1].split()[-1])
        cells.append(" ".join(("=" if m == g else m) + "x%d" % n for m, n in c.items()))
        for m in c:
            ref.setdefault(k[1].split()[-1], set()).add(m)
    print("%-10s %-14s " % k + " | ".join("%s: %s" % (t, c) for t, c in zip(tags, cells)))
print("distinct md5 per spec over all legs:", {s: sorted(v) for s, v in sorted(ref.items())})
print("== tok/s (mean of measured rounds) ==")
for k in sorted({k for t in tags for k in sp[t]}):
    print("%-26s " % k + " | ".join("%s: %s" % (t, ("%.2f" % (sum(sp[t][k]) / len(sp[t][k])) + " (%d)" % len(sp[t][k])) if sp[t].get(k) else "-") for t in tags))
