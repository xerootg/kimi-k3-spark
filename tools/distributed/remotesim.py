#!/usr/bin/env python3
"""Replay real K3 routing (k3-pfdump kind 3 = real top-16 ids) through the Spark's LRU bank plus a static
remote expert set held by other boxes.

usage: remotesim.py --train a.bin[,b.bin] --eval c.bin [--cap 7980] [--warm 150] [--sizes 0,600,3700,7000]

Per (token, MoE layer): local LRU hit, remote (in the static set, never cached locally), NVMe miss.
The static set = the most frequent (layer, expert) pairs in the training dumps, optionally after skipping
the `skip` hottest (which the LRU would hold anyway).
"""
import sys, collections, struct, os

def load_ids(path):
    out = collections.OrderedDict()
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        pos = 0
        while pos + 16 <= size:
            f.seek(pos)
            step, il, kind, n = struct.unpack("<4i", f.read(16))
            pos += 16
            if kind == 3:
                out.setdefault(step, {})[il] = list(struct.unpack(f"<{n}i", f.read(4*n)))
            pos += 4*n
    return out

def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    cap = int(opts.get("cap", 7980)); warm = int(opts.get("warm", 150))
    sizes = [int(x) for x in opts.get("sizes", "0,600,3700,7000,10000").split(",")]
    skips = [int(x) for x in opts.get("skips", "0,4000,7980").split(",")]
    train = [load_ids(p) for p in opts["train"].split(",")]
    ev = load_ids(opts["eval"])
    freq = collections.Counter()
    for t in train:
        for step, layers in t.items():
            for il, ids in layers.items():
                for e in ids: freq[(il, int(e))] += 1
    ranked = [k for k, _ in freq.most_common()]
    layers = sorted({il for s in ev.values() for il in s})
    steps = list(ev.keys())
    print(f"train: {sum(len(t) for t in train)} steps, {len(freq)} distinct experts seen; eval {len(steps)} steps x {len(layers)} layers")
    for skip in skips:
        for R in sizes:
            remote = set(ranked[skip:skip+R])
            lru = collections.OrderedDict()
            hit = rem = miss = n = 0
            per_layer_rem = []
            for si, st in enumerate(steps):
                count = si >= warm
                for il in layers:
                    ids = ev[st].get(il)
                    if ids is None: continue
                    r_here = 0
                    for e in ids:
                        k = (il, int(e))
                        if k in remote:
                            r_here += 1
                            if count: rem += 1
                            continue
                        if k in lru:
                            lru.move_to_end(k)
                            if count: hit += 1
                        else:
                            lru[k] = 1
                            if len(lru) > cap: lru.popitem(last=False)
                            if count: miss += 1
                    if count:
                        n += 1; per_layer_rem.append(r_here)
            pl = sorted(per_layer_rem)
            if R: print("   hist " + ",".join(f"{sum(1 for x in pl if x==i)/len(pl):.4f}" for i in range(17)))
            print(f"skip={skip:5d} remote={R:6d} ({R*6.6/1000:5.1f} GB): per layer local hits {hit/n:5.2f}  remote {rem/n:4.2f}  "
                  f"NVMe misses {miss/n:4.2f}  | layers with >=1 remote {sum(1 for x in pl if x>0)/len(pl)*100:4.1f}%  remote p90 {pl[int(.9*(len(pl)-1))]} max {pl[-1]}")

main()
