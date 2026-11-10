#!/usr/bin/env python3
"""Digest of the share legs (prio_nc.py records, "leg" = W12/WK08/...): Juno alone per answer length (the three "warm"
blocks, in order 256/512/1024 tokens, and the closing one), then every contended scenario with Juno's decode as a
share of her solo speed for the same md5 (= the same answer length).
Usage: wdigest.py results.jsonl LEG [LEG ...]"""
import json, sys
GOLD = {"p0": {"8e312d7bdd", "22da3a7294", "168f9c9c04"}, "p1": {"74fb633e8e"}, "p2": {"f8531a5f07"}, "p3": {"4d6338450b"},
        "p4": {"e7851c4b56"}, "L16384": {"145c14aa8b"}, "L65536": {"8acec1b966"}}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
for leg in sys.argv[2:]:
    rs = [r for r in rows if r.get("leg") == leg and r.get("kind") == "prio"]
    solo = {}
    for r in rs:
        if r["scenario"] == "warm":
            solo.setdefault(r["md5"], []).append(r["decode_tps"])
    print("== %s: %d records" % (leg, len(rs)))
    for m, v in solo.items():
        print("  juno alone md5 %s: %s  mean %.2f" % (m, " ".join("%.2f" % x for x in v), sum(v) / len(v)))
    bad = 0
    cur = None
    for r in rs:
        ok = r.get("md5") in GOLD.get(r.get("spec"), set())
        bad += 0 if ok or "error" in r else 1
        if r["scenario"] == "warm":
            continue
        if r["scenario"] != cur or r["t_send"] == 0.0 and r["tag"] == "juno":
            cur = r["scenario"]
        ratio = ""
        if r["tag"] == "juno" and r.get("md5") in solo:
            s = solo[r["md5"]]
            ratio = " = %.0f %% of alone (%.0f-%.0f %%)" % (100 * r["decode_tps"] / (sum(s) / len(s)), 100 * r["decode_tps"] / max(s), 100 * r["decode_tps"] / min(s))
        if "error" in r:
            print("  %-10s %-5s %-7s ERROR %s" % (r["scenario"], r["tag"], r.get("spec"), r["error"]))
            continue
        print("  %-10s %-5s %-7s send %6.2f ttft %7.3f wall %6.2f gap %5.2f decode %6.2f prefill %7.1f %s%s%s" % (
            r["scenario"], r["tag"], r["spec"], r["t_send"], r["ttft_s"], r["wall_s"], r["max_gap_s"], r["decode_tps"],
            r["prefill_tps"], r["md5"], "" if ok else " NOT-GOLD", ratio))
    print("  md5 not gold: %d" % bad)
