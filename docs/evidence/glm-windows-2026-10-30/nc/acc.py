#!/usr/bin/env python3
"""Acceptance digest for nc windows: per tag, the solo md5 of every spec, whether every multi round gave each client its
solo md5, and the throughput lines. Usage: acc.py results.jsonl TAG [TAG...]"""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
GOLD = {"p0": "8e312d7bdd", "p1": "74fb633e8e", "p2": "f8531a5f07", "p3": "4d6338450b", "p4": "e7851c4b56",
        "L16384": "145c14aa8b", "L65536": "8acec1b966"}
for tag in sys.argv[2:]:
    R = [r for r in rows if r.get("tag") == tag]
    solo = {}
    bad = []
    print(f"== {tag}")
    for r in R:
        if r.get("kind") == "gold":
            print(f"  gold {r['md5']} {r['decode_tps']}")
        if r.get("kind") == "solo" and "md5" in r:
            solo.setdefault(r["spec"], set()).add(r["md5"])
            print(f"  solo {r['spec']:7s} {r['md5']} dec {r['decode_tps']:6.2f} pre {r['prefill_tps']:8.1f} cache_n {r.get('cache_n')}")
    for r in R:
        if r.get("kind") == "multi":
            ok = all(m == GOLD.get(s, next(iter(solo.get(s, {'?'})))) for s, m in zip(r["specs"], r["md5"]))
            if not ok:
                bad.append(r)
            print(f"  multi n={r['n']} round={r['round']} agg {r['agg_tps']:6.2f} per {r['per_stream_tps']} {'OK' if ok else 'DIFF'} {r['specs']} {r['md5']}")
        if r.get("kind") == "depth" or "depth" in str(r.get("kind")):
            print(f"  depth {r}")
    for s, ms in solo.items():
        if s in GOLD and ms != {GOLD[s]}:
            print(f"  SOLO-DIFF {s} {ms} gold {GOLD[s]}")
    print(f"  multi rounds differing from solo/gold: {len(bad)}")
