#!/usr/bin/env python3
"""Compare the SPEC_TRACE lines (LLAMA_SPEC_EXACT_TRACE=1: slot, index, token, FNV-1a of the logits row) of two server
logs. A request is a run of lines of one slot whose index restarts at 0. The two logs must hold the same requests
(which slot served one is free): every sampled token and every logits row bit-identical.
With --subset, B may hold fewer requests than A.
With --tail N, a request of B may go on for up to N lines after the end of the same request of A: the exact
speculation samples the accepted draft tokens of a step before it emits them one by one, so when an emitted token
ends the request (a stop word) the tokens after it were sampled and traced but never emitted. Such requests are
counted and listed; the client-side check (rescmp.py: md5 and token count of every answer) covers what was emitted.
Usage: tracecmp.py [--subset] [--tail N] A.log B.log"""
import sys
from collections import Counter, defaultdict


def requests(path):
    cur, done = defaultdict(list), []
    for line in open(path, errors="replace"):
        if not line.startswith("SPEC_TRACE "):
            continue
        f = line.split()
        if len(f) != 5:
            continue
        slot, i, tok, h = int(f[1]), int(f[2]), int(f[3]), f[4]
        if i == 0 and cur[slot]:
            done.append(tuple(cur[slot]))
            cur[slot] = []
        if i != len(cur[slot]):
            print("%s: slot %d index %d after %d lines (gap)" % (path, slot, i, len(cur[slot])))
        cur[slot].append((tok, h))
    done += [tuple(v) for v in cur.values() if v]
    return done


def common(x, y):
    n = 0
    while n < len(x) and n < len(y) and x[n] == y[n]:
        n += 1
    return n


subset = "--subset" in sys.argv
args = [x for x in sys.argv[1:] if x != "--subset"]
tail = 0
if "--tail" in args:
    i = args.index("--tail")
    tail = int(args[i + 1])
    del args[i:i + 2]
a, b = requests(args[0]), requests(args[1])
if not a or not b:
    # a leg that died at load leaves an empty trace: with --subset that would otherwise read as identical
    print("requests %d / %d" % (len(a), len(b)))
    print("TRACE-EMPTY")
    sys.exit(2)
ca, cb = Counter(a), Counter(b)
only_a, only_b = list((ca - cb).elements()), list((cb - ca).elements())
print("requests %d / %d, tokens %d / %d, identical requests %d" % (len(a), len(b), sum(map(len, a)), sum(map(len, b)), sum((ca & cb).values())))
n_tail = 0
for y in list(only_b):
    # B's request is A's plus at most `tail` sampled tokens after A's end
    for x in only_a:
        if len(x) < len(y) <= len(x) + tail and common(x, y) == len(x):
            print("  TAIL: B samples %d tokens after the end of A's request of %d tokens (the first %d lines identical)" % (len(y) - len(x), len(x), len(x)))
            only_a.remove(x)
            only_b.remove(y)
            n_tail += 1
            break
for y in only_b:
    best = max(only_a, key=lambda x: common(x, y), default=None)
    if best is None:
        print("  only in B: %d tokens, first token %d" % (len(y), y[0][0]))
        continue
    n = common(best, y)
    why = "one ends" if n >= min(len(y), len(best)) else ("token %d vs %d" % (best[n][0], y[n][0]) if best[n][0] != y[n][0] else "same token, logits differ")
    print("  DIFF at index %d of %d / %d: %s" % (n, len(best), len(y), why))
if not subset:
    for x in only_a:
        if not only_b:
            print("  only in A: %d tokens, first token %d" % (len(x), x[0][0]))
ok = not only_b and (subset or not only_a)
print(("TRACE-IDENTICAL" + (" (%d requests with a sampled tail after a stop)" % n_tail if n_tail else "")) if ok else "TRACE-DIFFERENT")
sys.exit(0 if ok else 1)
