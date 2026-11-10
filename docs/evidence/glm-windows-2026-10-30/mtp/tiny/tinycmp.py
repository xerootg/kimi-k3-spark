#!/usr/bin/env python3
"""tinycmp.py A.jsonl B.jsonl: the same request list against two servers; md5, token counts and stop types must agree."""
import json, sys
def load(p):
    rows = []
    for l in open(p):
        r = json.loads(l)
        rows.append((r.get("kind"), r.get("round"), json.dumps(r.get("spec", r.get("specs"))), r.get("temp"), r.get("seed"),
                     json.dumps(r.get("md5")), json.dumps(r.get("decode_n", r.get("gen"))), json.dumps(r.get("stop")),
                     r.get("draft_n"), r.get("draft_ok")))
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
bad = 0
for x, y in zip(a, b):
    same = x[:8] == y[:8]
    bad += not same
    print("%s %-8s %-28s temp %s seed %s md5 %s n %s stop %s | drafts %s/%s vs %s/%s%s" % ("ok  " if same else "DIFF", x[0], x[2][:28], x[3], x[4], x[5][:40], x[6], x[7],
          x[9], x[8], y[9], y[8], "" if same else "   B: md5 %s n %s stop %s" % (y[5][:40], y[6], y[7])))
print("RESULT: %d requests, %d differ%s" % (len(a), bad, "" if len(a) == len(b) else " (lengths %d vs %d)" % (len(a), len(b))))
