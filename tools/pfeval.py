#!/usr/bin/env python3
"""Offline evaluation of the Kimi K3 speculative-prefetch predictor variants from llama-k3-pfdump dumps.

kinds: 1 deployed predictor (input 1), 2 real selection probs, 3 real top-k ids, 4 real probs,
       6 pfv2 (stream + shared expert, through the target layer's residual mix), 7 pfv3 (stream + whole FFN output),
       8 pfv4 (pfv2 two layers ahead), 9 pfv5 (pfv3 two layers ahead)
"""
import sys, collections, numpy as np

K = 16
NAMES = {1: "deployed (input 1)", 6: "pfv2 mix+sh", 7: "pfv3 mix+ffn", 8: "pfv4 mix+sh, 2 ahead", 9: "pfv5 mix+ffn, 2 ahead"}

def load(paths):
    steps = {}
    order = []
    for fi, path in enumerate(paths):
        buf = np.fromfile(path, dtype=np.int32)
        pos = 0
        while pos + 4 <= len(buf):
            step, il, kind, n = buf[pos:pos+4]
            pos += 4
            data = buf[pos:pos+n]
            pos += n
            key = (fi, int(step))
            if key not in steps:
                steps[key] = {}
                order.append(key)
            if kind == 3:
                steps[key].setdefault(int(il), {})[3] = data.copy()
            elif kind in (1, 6, 7, 8, 9):
                steps[key].setdefault(int(il), {})[int(kind)] = np.argsort(-data.view(np.float32))[:2*K].copy()
    return steps, order

def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    cap = int(opts.get("cap", 7980))
    warm = int(opts.get("warm", 150))
    steps, order = load(paths)
    layers = sorted({il for key in order for il in steps[key] if 3 in steps[key][il]})
    print(f"{len(order)} steps, {len(layers)} MoE layers {layers[0]}..{layers[-1]}")
    kinds = sorted({k for key in order for il in steps[key] for k in steps[key][il] if k != 3})

    print("\n== top-16 precision by predictor (and by rank 1/4/8/12/16), per layer class")
    ckpt = [il for il in layers if il % 12 == 0]
    for k in kinds:
        hits = np.zeros(2*K); n = 0
        per_layer = collections.defaultdict(list)
        for key in order:
            for il in layers:
                d = steps[key].get(il)
                if d is None or k not in d or 3 not in d: continue
                T = set(int(x) for x in d[3])
                h = np.array([int(t) in T for t in d[k]])
                hits += h; n += 1
                per_layer[il].append(h[:K].mean())
            # noqa
        if n == 0: continue
        pl = {il: np.mean(v) for il, v in per_layer.items()}
        ck = [pl[il] for il in ckpt if il in pl]
        nck = [pl[il] for il in layers if il in pl and il % 12 != 0]
        after = [pl[il] for il in layers if il in pl and il % 12 == 1]
        print(f"  kind {k} {NAMES[k]:24s} n={n:6d} top16 {hits[:K].sum()/(n*K):.3f} top32-recall {hits.sum()/(n*K):.3f} | "
              f"ranks {hits[0]/n:.2f} {hits[3]/n:.2f} {hits[7]/n:.2f} {hits[11]/n:.2f} {hits[15]/n:.2f} | "
              f"checkpoint layers {np.mean(ck):.2f}, layers after them {np.mean(after):.2f}, others {np.mean(nck):.2f}")
        worst = sorted(pl.items(), key=lambda x: x[1])[:8]
        print("     worst layers: " + " ".join(f"{il}:{p:.2f}" for il, p in worst))

    # --- pipeline simulation ---
    def simulate(name, one_kind, two_kind, W1, W2=None, per_layer_mask=None):
        lru = collections.OrderedDict()
        def touch(k):
            if k in lru: lru.move_to_end(k)
            else:
                lru[k] = 1
                if len(lru) > cap: lru.popitem(last=False)
        real = 0; nl = 0; started = 0; used = 0; started2 = 0; used2 = 0; m0 = 0
        for si, key in enumerate(order):
            count = si >= warm
            for li, il in enumerate(layers):
                d = steps[key].get(il)
                if d is None: continue
                T = [int(x) for x in d[3]]
                miss = [e for e in T if (il, e) not in lru]
                for e in T: touch((il, e))
                if count: real += len(miss); nl += 1; m0 += len(miss)
                # speculation in this layer's window
                budget = W1
                if li + 1 < len(layers):
                    iln = layers[li+1]; dn = steps[key].get(iln)
                    if dn is not None and one_kind in dn and (per_layer_mask is None or per_layer_mask(iln)):
                        Tn = set(int(x) for x in dn[3])
                        for e in dn[one_kind][:K]:
                            e = int(e)
                            if budget <= 0: break
                            if (iln, e) in lru: continue
                            touch((iln, e)); budget -= 1
                            if count:
                                started += 1; used += e in Tn
                if two_kind is not None and li + 2 < len(layers) and budget > 0:
                    il2 = layers[li+2]; d2 = steps[key].get(il2)
                    if d2 is not None and two_kind in d2:
                        T2 = set(int(x) for x in d2[3])
                        b2 = budget if W2 is None else min(budget, W2)
                        for e in d2[two_kind][:K]:
                            e = int(e)
                            if b2 <= 0: break
                            if (il2, e) in lru: continue
                            touch((il2, e)); b2 -= 1; budget -= 1
                            if count:
                                started2 += 1; used2 += e in T2
        print(f"  {name:44s} real fill {real/nl:5.2f}  started {started/nl:4.2f} (prec {used/max(started,1):.2f})"
              + (f"  two-ahead started {started2/nl:4.2f} (prec {used2/max(started2,1):.2f})" if two_kind else ""))

    print(f"\n== pipeline simulation: LRU {cap} records, counting from step {warm}; no speculation = {6.7:.1f}-ish misses/layer")
    simulate("none", 99, None, 0)
    for W in (5,):
        simulate(f"deployed (kind 1), window {W}", 1, None, W)
        if 6 in kinds: simulate(f"pfv2 mix+sh, window {W}", 6, None, W)
        if 7 in kinds:
            simulate(f"pfv3 mix+ffn, window {W} (needs the misses: optimistic)", 7, None, W)
            simulate(f"pfv3 mix+ffn, window 3.5 (after the GEMV)", 7, None, 3.5)
        if 6 in kinds and 8 in kinds: simulate(f"pfv2 + pfv4 two-ahead in the spare window, {W}", 6, 8, W)
        if 7 in kinds and 9 in kinds: simulate(f"pfv3 + pfv5 two-ahead, window 3.5", 7, 9, 3.5)
        if 6 in kinds: simulate(f"pfv2, checkpoint layers not predicted, {W}", 6, None, W, per_layer_mask=lambda il: il % 12 != 0)
        if 6 in kinds and 8 in kinds: simulate(f"pfv2 + pfv4, checkpoint layers skipped, {W}", 6, 8, W, per_layer_mask=lambda il: il % 12 != 0)
    # oracle
    lru = collections.OrderedDict()
    def touch(k):
        if k in lru: lru.move_to_end(k)
        else:
            lru[k] = 1
            if len(lru) > cap: lru.popitem(last=False)
    real = 0; nl = 0
    for si, key in enumerate(order):
        for li, il in enumerate(layers):
            d = steps[key].get(il)
            if d is None: continue
            T = [int(x) for x in d[3]]
            miss = [e for e in T if (il, e) not in lru]
            for e in T: touch((il, e))
            if si >= warm: real += len(miss); nl += 1
            if li + 1 < len(layers):
                iln = layers[li+1]; dn = steps[key].get(iln)
                if dn is not None:
                    b = 5
                    for e in [int(x) for x in dn[3]]:
                        if b <= 0: break
                        if (iln, e) in lru: continue
                        touch((iln, e)); b -= 1
    print(f"  {'oracle, window 5':44s} real fill {real/nl:5.2f}")

if __name__ == "__main__":
    main()
