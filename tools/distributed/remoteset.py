#!/usr/bin/env python3
"""Choose and place the static remote expert sets for LLAMA_EXPERT_STORE_REMOTE / _REMOTE_SET.

Single set (one box):
  remoteset.py --store STORE --train a.bin[,b.bin] [--eval c.bin] --budget-gb 46 [--skip 4000]
               [--skips 0,2000,4000,7980] [--out set.txt]

Placement over several boxes:
  remoteset.py --store STORE --train ... --eval ... --boxes nuc2:22:1.0,nuc3:6.5:0.5 [--skip 6000]
               [--configs "nuc2:22:1.0;nuc2:22:1.0,nuc3:6.5:0.5"] [--out-prefix set-]

STORE may be the store itself or a copy of its first 12 KiB (header + index): only record sizes are read.
Dumps: k3-pfdump files (kind 3 = the real top-16 ids per token and layer).

Selection: rank (layer, expert) pairs by how often the training dumps select them, drop the `skip` hottest (the
Spark's LRU keeps those anyway) and take the next ones until the budget is full (record sizes from the store).

Placement (--boxes name:GB:speed, speed = relative compute speed, e.g. 1.0 for nuc2's dual-channel Arc, 0.5 for a
single-channel or busy box): the pool is the selection for the sum of the budgets. A layer's request goes to every
box that holds one of its selected remote experts, so the pool is placed by layer, not by expert: each layer's pool
goes to one box whenever it fits (fan-out 1), and only the layers at a box's capacity edge are split. Layers are
ordered by how often their pool fires per byte; the fastest box takes the densest layers, the slowest box the
sparsest (it gets the fewest requests per GB). A split layer keeps the experts that fire together with the box's
existing ones first (co-occurrence in the training tokens), then by frequency.

Evaluation: the eval dump is replayed through the Spark's LRU (--cap, 7980 slots, no speculation modelled) with
the engine's rule that remote candidates are capped to the late matmul's free rows (--late 8 minus the local
misses, --rows 8): per layer, NVMe misses, remote rows, boxes touched (fan-out) and per-box request sizes.
"""
import collections, os, struct, sys

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
                out.setdefault(step, {})[il] = list(struct.unpack(f"<{n}i", f.read(4 * n)))
            pos += 4 * n
    return out

def store_records(path):
    with open(path, "rb") as f:
        h = f.read(64)
        magic, ver, n_rec, n_exp, align, esz, ioff, roff = struct.unpack("<4sIIIIIQQ", h[:40])
        assert magic == b"K3XS", magic
        f.seek(ioff)
        rs = {}
        for _ in range(n_rec):
            e = struct.unpack("<IIIIQQQQQ", f.read(esz)[:56])
            rs[e[0]] = e[4] + e[5] + e[6]  # layer -> bytes a remote box holds (gate + up + down, no padding)
    return rs, n_exp

def replay(ev, box_of, cap, warm, late=8, rows=8, nbox=1):
    """box_of: dict (layer, expert) -> box index. Returns per-layer means and distributions."""
    lru = collections.OrderedDict()
    hit = miss = rem = n = 0
    fan = collections.Counter()
    kdist = [collections.Counter() for _ in range(nbox)]
    for si, st in enumerate(ev):
        count = si >= warm
        for il, ids in sorted(ev[st].items()):
            keys = [(il, int(e)) for e in ids]
            cand = [k for k in keys if k in box_of and k not in lru]
            local_miss = sum(1 for k in keys if k not in box_of and k not in lru)
            room = min(rows, max(0, late - local_miss))
            remote = cand[:room]
            rset = set(remote)
            per = collections.Counter(box_of[k] for k in remote)
            for k in keys:
                if k in rset:
                    continue
                if k in lru:
                    lru.move_to_end(k)
                    hit += count
                else:
                    lru[k] = 1
                    if len(lru) > cap:
                        lru.popitem(last=False)
                    miss += count
            if count:
                n += 1
                rem += len(remote)
                fan[len(per)] += 1
                for b, c in per.items():
                    kdist[b][c] += 1
    return hit / n, rem / n, miss / n, fan, kdist, n

def main():
    o = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    rs, n_exp = store_records(o["store"])
    cap, warm = int(o.get("cap", 7980)), int(o.get("warm", 150))
    late, rows = int(o.get("late", 8)), int(o.get("rows", 8))
    freq = collections.Counter()
    co = collections.defaultdict(collections.Counter)  # (layer, e) -> Counter of (layer, e') selected with it
    for p in o["train"].split(","):
        for st, layers in load_ids(p).items():
            for il, ids in layers.items():
                ks = [(il, int(e)) for e in ids]
                for k in ks:
                    freq[k] += 1
                if "boxes" in o or "configs" in o:
                    for k in ks:
                        c = co[k]
                        for k2 in ks:
                            if k2 != k:
                                c[k2] += 1
    ranked = [k for k, _ in freq.most_common()]
    print(f"train: {len(freq)} distinct (layer, expert) pairs seen, n_expert {n_exp}")
    ev = load_ids(o["eval"]) if "eval" in o else None

    def take(skip, budget):
        s, tot = [], 0
        for k in ranked[skip:]:
            if tot + rs[k[0]] > budget:
                break
            s.append(k)
            tot += rs[k[0]]
        return s, tot

    if "boxes" in o or "configs" in o:
        skip = int(o.get("skip", 6000))
        configs = o["configs"].split(";") if "configs" in o else [o["boxes"]]
        if ev is not None:
            h, r, m, fan, kd, n = replay(ev, {}, cap, warm, late, rows)
            print(f"no remote: local hits {h:5.2f}  NVMe misses {m:4.2f} per layer")
        for cfg in configs:
            boxes = []
            for b in cfg.split(","):
                name, gb, speed = b.split(":")
                boxes.append((name, float(gb) * 1e9, float(speed)))
            budget = sum(b[1] for b in boxes)
            pool, tot = take(skip, budget)
            # by layer: bytes and fires
            by_layer = collections.defaultdict(list)
            for k in pool:
                by_layer[k[0]].append(k)
            dens = sorted(by_layer, key=lambda il: -sum(freq[k] for k in by_layer[il]) / sum(rs[il] for _ in by_layer[il]))
            order = sorted(range(len(boxes)), key=lambda i: -boxes[i][2])  # fastest first
            assign = {}
            free = {i: boxes[i][1] for i in range(len(boxes))}
            bi = 0
            for il in dens:
                rest = sorted(by_layer[il], key=lambda k: -freq[k])
                while rest and bi < len(order):
                    b = order[bi]
                    if free[b] < rs[il]:
                        bi += 1
                        continue
                    # take as many as fit, co-firing with what the box already holds of this layer first
                    held = [k for k in assign if assign[k] == b and k[0] == il]
                    if held and len(rest) * rs[il] > free[b]:
                        rest.sort(key=lambda k: (-sum(co[k][h] for h in held), -freq[k]))
                    while rest and free[b] >= rs[il]:
                        k = rest.pop(0)
                        assign[k] = b
                        free[b] -= rs[il]
                    if rest:
                        bi += 1
            print(f"\nconfig {cfg}: skip {skip}, pool {len(pool)} records {tot/1e9:.1f} GB, placed {len(assign)}")
            for i, (name, gb, speed) in enumerate(boxes):
                ks = [k for k, b in assign.items() if b == i]
                print(f"  box {i} {name}: {len(ks)} records {sum(rs[k[0]] for k in ks)/1e9:5.1f} GB, layers {len({k[0] for k in ks})}, "
                      f"train selections {sum(freq[k] for k in ks)}")
                if "out-prefix" in o:
                    with open(f"{o['out-prefix']}{name}.txt", "w") as f:
                        for k in sorted(ks):
                            f.write(f"{k[0]} {k[1]}\n")
            if ev is not None:
                h, r, m, fan, kd, n = replay(ev, assign, cap, warm, late, rows, len(boxes))
                print(f"  eval: local hits {h:5.2f}  remote rows {r:4.2f}  NVMe misses {m:4.2f} per layer")
                print("  fan-out (boxes per layer): " + ", ".join(f"{f}: {fan[f]/n*100:.1f}%" for f in sorted(fan)))
                for i, (name, _, _) in enumerate(boxes):
                    tk = sum(kd[i].values())
                    if tk:
                        mk = sum(c * v for c, v in kd[i].items()) / tk
                        p90 = sorted(c for c, v in kd[i].items() for _ in range(v))[int(.9 * (tk - 1))]
                        print(f"  box {name}: requests in {tk/n*100:.1f}% of layers, k mean {mk:.2f} p90 {p90} max {max(kd[i])}")
        return

    budget = float(o.get("budget-gb", 46)) * 1e9
    if ev is not None:
        h, r, m, fan, kd, n = replay(ev, {}, cap, warm, late, rows)
        print(f"no remote set: local hits {h:5.2f}  NVMe misses {m:4.2f} per layer")
        for skip in [int(x) for x in o.get("skips", "0,2000,4000,7980").split(",")]:
            s, tot = take(skip, budget)
            h, r, m, fan, kd, n = replay(ev, {k: 0 for k in s}, cap, warm, late, rows)
            p90 = sorted(c for c, v in kd[0].items() for _ in range(v))
            print(f"skip {skip:5d}: {len(s)} records {tot/1e9:5.1f} GB  local hits {h:5.2f}  remote {r:4.2f}  NVMe misses {m:4.2f} per layer")
    if "out" in o:
        s, tot = take(int(o.get("skip", 4000)), budget)
        with open(o["out"], "w") as f:
            for il, e in s:
                f.write(f"{il} {e}\n")
        print(f"wrote {len(s)} pairs, {tot/1e9:.1f} GB, to {o['out']}")

main()
