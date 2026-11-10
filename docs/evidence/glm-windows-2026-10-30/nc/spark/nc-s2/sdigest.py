#!/usr/bin/env python3
"""Digest of the priority legs on the Spark (prio_spark.py / prio_spark_e.py records): Juno alone (the "warm" blocks),
then every scenario in the order it ran, Juno's decode as a share of her speed alone for the same md5.
Usage: sdigest.py results.jsonl LEG [LEG ...]"""
import json, sys
GOLD = {"p0": {"ce2fc95752", "cfa6c68a74", "ee079496a4"}, "p1": {"bb64203d71", "4d4058c69f"}, "p2": {"2d592efcb6"},
        "p3": {"1aab74269d"}, "p4": {"0c3f28294b"}, "L16384": {"145c14aa8b"}}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
for leg in sys.argv[2:]:
    rs = [r for r in rows if r.get("leg") == leg and r.get("kind") == "prio"]
    solo = {}
    for r in rs:
        if r["scenario"] == "warm" and "error" not in r:
            solo.setdefault(r["md5"], []).append(r["decode_tps"])
    print("== %s: %d records" % (leg, len(rs)))
    for m, v in solo.items():
        print("  juno alone md5 %s: %s  mean %.2f" % (m, " ".join("%.2f" % x for x in v), sum(v) / len(v)))
    bad = 0
    for r in rs:
        if "error" in r:
            print("  %-11s %-5s %-7s ERROR %s" % (r["scenario"], r["tag"], r.get("spec"), str(r["error"])[:80]))
            continue
        ok = r.get("md5") in GOLD.get(r.get("spec"), set())
        bad += 0 if ok else 1
        if r["scenario"] == "warm":
            continue
        ratio = ""
        if r["tag"] == "juno" and r.get("md5") in solo:
            s = solo[r["md5"]]
            ratio = " = %.0f %% of alone" % (100 * r["decode_tps"] / (sum(s) / len(s)))
        print("  %-11s %-5s %-7s send %6.2f ttft %7.3f wall %6.2f gap %5.2f decode %6.2f prefill %7.1f %s%s%s" % (
            r["scenario"], r["tag"], r["spec"], r["t_send"], r["ttft_s"], r["wall_s"], r["max_gap_s"], r["decode_tps"],
            r["prefill_tps"], r["md5"], "" if ok else " NOT-GOLD", ratio))
    print("  md5 not gold: %d" % bad)
