#!/usr/bin/env python3
"""GH200 window nc4, leg with the recurrent-state detectors of patch 0069 (LLAMA_RS_LOG=1 LLAMA_RS_SUM=512).
Usage: v1digest.py LEG [DIR [spark]]   (reads DIR/results.jsonl and DIR/server-LEG.log, DIR = the GH200's legs
directory by default; "spark" = the Spark's golds; exit 1 if anything is wrong)
Checks: every request's md5 is the gold of its request; every long prompt ubatch (same end position and length) has
one and the same state hash wherever it ran (alone, between other clients' steps, whole or in pieces); no ubatch's
cell range contains the frozen cell; the guard in find_slot never fired."""
import collections, json, re, sys
G = sys.argv[2] if len(sys.argv) > 2 else "/srv/glm-5.3-flash/nc/legs"
leg = sys.argv[1]
GOLD = {"p0": "8e312d7bdd", "p1": "74fb633e8e", "p2": "f8531a5f07", "p3": "4d6338450b", "p4": "e7851c4b56",
        "L16384": "145c14aa8b", "L65536": "8acec1b966"}
JN = {512: "22da3a7294", 1024: "168f9c9c04"}
if len(sys.argv) > 3 and sys.argv[3] == "spark":
    GOLD = {"p0": "ce2fc95752", "p1": "bb64203d71", "p2": "2d592efcb6", "p3": "1aab74269d", "p4": "0c3f28294b", "L16384": "145c14aa8b"}
    JN = {512: "cfa6c68a74"}
bad = 0
rows = [json.loads(l) for l in open(G + "/results.jsonl") if l.strip().startswith("{")]
rows = [r for r in rows if r.get("leg") == leg and r.get("kind") == "prio"]
n_ok = 0
for r in rows:
    if "md5" not in r:
        if r.get("scenario") != "nokey":
            print("  NO-MD5", r); bad += 1
        continue
    gold = GOLD.get(r["spec"])
    if r["scenario"] == "junolong" and r["tag"] == "juno":
        gold = JN[512]
    if r["md5"] == gold:
        n_ok += 1
    else:
        print("  MD5-DIFF %s %s %s got %s gold %s" % (r["scenario"], r["tag"], r["spec"], r["md5"], gold)); bad += 1
print("requests with gold md5: %d of %d" % (n_ok, len(rows)))
sums = collections.defaultdict(collections.Counter)
n_fz = n_in = n_guard = n_whole = n_pieces = 0
shape = collections.Counter()
for line in open(G + "/server-%s.log" % leg, errors="replace"):
    m = re.search(r"RSSUM seq=(-?\d+) pos=(-?\d+) n=(\d+) sum=([0-9a-f]+)", line)
    if m:
        sums[(int(m.group(2)), int(m.group(3)))][m.group(4)] += 1
        continue
    m = re.search(r"RSLOG nt=(\d+) head=(\d+) n=(\d+) z=(-?\d+) fz=(-?\d+) seqs=(\S*)", line)
    if m:
        head, n, fz = int(m.group(2)), int(m.group(3)), int(m.group(5))
        if fz >= 0:
            n_fz += 1
            side = "below" if head + n - 1 < fz else "above" if head > fz else "CONTAINS"
            shape[(fz, side, len([x for x in m.group(6).split(",") if x]))] += 1
            if side == "CONTAINS":
                n_in += 1
        continue
    if "contains cell" in line:
        n_guard += 1
    elif "runs whole" in line:
        n_whole += 1
    elif "done in" in line and "pieces" in line:
        n_pieces += 1
multi = {k: v for k, v in sums.items() if len(v) > 1}
once = sum(1 for v in sums.values() if sum(v.values()) == 1)
twice = len(sums) - once
print("state hashes: %d distinct long ubatches (end position, length), %d seen once only, %d with MORE THAN ONE hash" % (len(sums), once, len(multi)))
for k in sorted(multi):
    print("  HASH-DIFF pos=%d n=%d %s" % (k[0], k[1], dict(multi[k])))
print("prompt ubatches done in pieces: %d, run whole (state not in place): %d" % (n_pieces, n_whole))
print("ubatches while a cell was frozen: %d; range contains the frozen cell: %d; guard refusals: %d" % (n_fz, n_in, n_guard))
for k in sorted(shape):
    print("  frozen cell %d: %6d ubatches %s it with %d sequence(s)" % (k[0], shape[k], k[1], k[2]))
bad += len(multi) + n_in + n_guard
if not rows or twice == 0 or n_pieces == 0 or n_fz == 0:
    print("  NOTHING-MEASURED (no request, no state hash seen twice, no pieces or no frozen cell seen)"); bad += 1
print("V1-DIGEST %s %s" % (leg, "PASS" if bad == 0 else "FAIL (%d)" % bad))
sys.exit(1 if bad else 0)
