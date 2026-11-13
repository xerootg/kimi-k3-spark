#!/usr/bin/env python3
"""GH200 window nc10: digest of the vision legs (records of vis10.py). Per leg: every image request's md5 alone (the
scenario "solo", every round) and whether every other occurrence (next to other requests) gave the same md5; every text
request against its gold. Then every image request's md5 across the legs. Exit 1 if anything differs.
Usage: visdigest.py results.jsonl LEG [LEG ...]"""
import collections, json, sys
GOLD = {("p0", 256): "8e312d7bdd", ("p0", 512): "22da3a7294", ("p0", 1024): "168f9c9c04", ("p1", 256): "74fb633e8e",
        ("p2", 256): "f8531a5f07", ("p3", 256): "4d6338450b", ("p4", 256): "e7851c4b56", ("L16384", 128): "145c14aa8b",
        ("L65536", 128): "8acec1b966"}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
bad = 0; ref = {}
for leg in sys.argv[2:]:
    rs = [r for r in rows if r.get("leg") == leg and r.get("kind") == "vis"]
    print("== %s: %d records" % (leg, len(rs)))
    solo = collections.defaultdict(list); other = collections.defaultdict(list); nt = nt_ok = 0
    for r in rs:
        if "error" in r:
            print("  ERROR %s %s %s %s %s" % (r["scenario"], r["tag"], r["spec"], r["error"], r.get("body", "")[:120])); bad += 1; continue
        if "-" not in r["spec"]:
            nt += 1; g = GOLD.get((r["spec"], r.get("n")))
            if r["md5"] == g:
                nt_ok += 1
            else:
                print("  TEXT-DIFF %s %s %s n=%s got %s gold %s" % (r["scenario"], r["tag"], r["spec"], r.get("n"), r["md5"], g)); bad += 1
            continue
        (solo if r["scenario"].startswith("solo") else other)[r["spec"]].append(r)
    for spec in sorted(set(solo) | set(other)):
        s = solo.get(spec, []); o = other.get(spec, [])
        sm = collections.Counter(r["md5"] for r in s); om = collections.Counter(r["md5"] for r in o)
        allm = set(sm) | set(om)
        state = "same" if len(allm) == 1 else ("ALONE-DIFFERS-RUN-TO-RUN" if len(sm) > 1 else "DIFFERS-NEXT-TO-OTHERS")
        if len(allm) != 1:
            bad += 1
        r0 = (s or o)[0]
        ref.setdefault(spec, {})[leg] = allm
        print("  %-9s prompt %5s tokens, answer %4s tokens (%s), alone %dx %s wall %s s; next to others %dx %s wall %s s: %s" % (
            spec, r0.get("prompt_n"), r0.get("pred_n"), r0.get("finish"), len(s), dict(sm), "/".join("%.1f" % r["wall_s"] for r in s),
            len(o), dict(om), "/".join("%.1f" % r["wall_s"] for r in o), state))
    print("  text requests with their gold md5: %d of %d" % (nt_ok, nt))
    if not rs or not solo:
        print("  NOTHING-MEASURED"); bad += 1
if len(sys.argv) > 3:
    print("== across legs %s" % " ".join(sys.argv[2:]))
    for spec in sorted(ref):
        u = set().union(*ref[spec].values()); miss = [l for l in sys.argv[2:] if l not in ref[spec]]
        ok = len(u) == 1 and not miss
        bad += 0 if ok else 1
        print("  %-9s %s%s %s" % (spec, " ".join("%s=%s" % (l, ",".join(sorted(ref[spec].get(l, {"-"})))) for l in sys.argv[2:]),
                                 " MISSING in " + ",".join(miss) if miss else "", "same" if ok else "DIFFERENT"))
print("VIS-DIGEST %s" % ("PASS" if bad == 0 else "FAIL (%d)" % bad))
sys.exit(1 if bad else 0)
