#!/usr/bin/env python3
"""Replay the real top-16 expert selections of a k3-pfdump (kind 3 records) through cache eviction policies at a
given capacity, with and without the deployed speculative prefetch (kind 1, the next layer's predicted ids, window 5).
One global cache as in pfeval.py. Used for the Spark eviction-policy entry of docs/experiments.md (2026-10-26).

  cachepolicy.py <pfdump.bin> [warm_steps=150]
"""
import sys, heapq, collections, numpy as np

K = 16
def load(path):
    buf = np.fromfile(path, dtype=np.int32); pos = 0; steps = {}
    while pos + 4 <= len(buf):
        step, il, kind, n = buf[pos:pos+4]; pos += 4
        if kind == 3: steps.setdefault(int(step), {})[int(il)] = [int(x) for x in buf[pos:pos+n]]
        elif kind == 1: steps.setdefault(int(step), {}).setdefault(('p', int(il)), np.argsort(-buf[pos:pos+n].view(np.float32))[:K].tolist())
        pos += n
    return steps

class LRU:
    def __init__(s, cap): s.cap, s.d = cap, collections.OrderedDict()
    def __contains__(s, k): return k in s.d
    def access(s, k):
        if k in s.d: s.d.move_to_end(k); return True
        if len(s.d) >= s.cap: s.d.popitem(last=False)
        s.d[k] = 1; return False

class LFUDecay:
    def __init__(s, cap, decay): s.cap, s.decay, s.cnt, s.res, s.t, s.last, s.h = cap, decay, {}, set(), 0, {}, []
    def __contains__(s, k): return k in s.res
    def access(s, k):
        s.t += 1
        if s.t % s.decay == 0:
            for x in s.cnt: s.cnt[x] >>= 1
            s.h = [(s.cnt.get(x, 0), s.last.get(x, 0), x) for x in s.res]; heapq.heapify(s.h)
        s.cnt[k] = s.cnt.get(k, 0) + 1; s.last[k] = s.t
        hit = k in s.res
        if not hit:
            while len(s.res) >= s.cap:
                c, l, x = heapq.heappop(s.h)
                if x in s.res and c == s.cnt.get(x, 0) and l == s.last.get(x, 0): s.res.discard(x)
            s.res.add(k)
        heapq.heappush(s.h, (s.cnt[k], s.last[k], k)); return hit

class ARC:
    def __init__(s, c):
        s.c, s.p = c, 0; s.t1, s.t2, s.b1, s.b2 = (collections.OrderedDict() for _ in range(4))
    def __contains__(s, k): return k in s.t1 or k in s.t2
    def _replace(s, k):
        if s.t1 and (len(s.t1) > s.p or (k in s.b2 and len(s.t1) == s.p)):
            x, _ = s.t1.popitem(last=False); s.b1[x] = 1
        else:
            x, _ = s.t2.popitem(last=False); s.b2[x] = 1
    def access(s, k):
        if k in s.t1: del s.t1[k]; s.t2[k] = 1; return True
        if k in s.t2: s.t2.move_to_end(k); return True
        if k in s.b1:
            s.p = min(s.c, s.p + max(len(s.b2) // max(len(s.b1), 1), 1)); s._replace(k); del s.b1[k]; s.t2[k] = 1; return False
        if k in s.b2:
            s.p = max(0, s.p - max(len(s.b1) // max(len(s.b2), 1), 1)); s._replace(k); del s.b2[k]; s.t2[k] = 1; return False
        if len(s.t1) + len(s.b1) == s.c:
            if len(s.t1) < s.c: s.b1.popitem(last=False); s._replace(k)
            else: s.t1.popitem(last=False)
        elif len(s.t1) + len(s.b1) < s.c and len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) >= s.c:
            if len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) == 2 * s.c: s.b2.popitem(last=False)
            s._replace(k)
        s.t1[k] = 1; return False

class S3FIFO:
    # small FIFO (10 %) + main FIFO with one-bit frequency + ghost; scan resistant, frequency aware
    def __init__(s, cap):
        s.cap = cap; s.small_cap = max(1, cap // 10); s.small = collections.deque(); s.main = collections.deque()
        s.ghost = collections.OrderedDict(); s.freq = {}; s.where = {}
    def __contains__(s, k): return k in s.where
    def _evict_main(s):
        while s.main:
            x = s.main.popleft()
            if s.freq.get(x, 0) > 0: s.freq[x] -= 1; s.main.append(x)
            else: del s.where[x]; s.freq.pop(x, None); return
    def _evict_small(s):
        x = s.small.popleft()
        if s.freq.get(x, 0) > 0:
            s.freq[x] = 0; s.main.append(x); s.where[x] = 'm'
            if len(s.main) > s.cap - s.small_cap: s._evict_main()
        else:
            del s.where[x]; s.freq.pop(x, None); s.ghost[x] = 1
            if len(s.ghost) > s.cap: s.ghost.popitem(last=False)
    def access(s, k):
        if k in s.where: s.freq[k] = min(3, s.freq.get(k, 0) + 1); return True
        if k in s.ghost:
            del s.ghost[k]; s.main.append(k); s.where[k] = 'm'; s.freq[k] = 0
            if len(s.main) > s.cap - s.small_cap: s._evict_main()
        else:
            s.small.append(k); s.where[k] = 's'; s.freq[k] = 0
            if len(s.small) > s.small_cap: s._evict_small()
        return False


class SecondChance:
    # the engine's LLAMA_EXPERT_STORE_SECOND_CHANCE=N: LRU list plus a per-slot count (<= N) raised on hits; the victim
    # search at the head spends a count (count--, slot to the tail) and takes the first slot with none
    def __init__(s, cap, n=3): s.cap, s.n, s.d, s.cnt = cap, n, collections.OrderedDict(), {}
    def __contains__(s, k): return k in s.d
    def access(s, k):
        if k in s.d:
            s.cnt[k] = min(s.n, s.cnt[k] + 1); s.d.move_to_end(k); return True
        while len(s.d) >= s.cap:
            v, _ = s.d.popitem(last=False)
            if s.cnt[v] > 0: s.cnt[v] -= 1; s.d[v] = 1
            else: del s.cnt[v]; break
        s.d[k] = 1; s.cnt[k] = 0; return False

def belady_hits(seq, cap, score):
    nxt, last = [0]*len(seq), {}
    for i in range(len(seq)-1, -1, -1): nxt[i] = last.get(seq[i], 1 << 60); last[seq[i]] = i
    res, h, hits = {}, [], 0
    for i, k in enumerate(seq):
        if k in res: hits += score[i]
        else:
            while len(res) >= cap:
                nu, x = heapq.heappop(h); nu = -nu
                if res.get(x) == nu: del res[x]
        res[k] = nxt[i]; heapq.heappush(h, (-nxt[i], k))
    return hits

def run(steps, cap, warm, spec_window=0):
    order = sorted(steps); layers = sorted(il for il in steps[order[0]] if isinstance(il, int) and il >= 0)
    seq, score, spec = [], [], []
    for si, st in enumerate(order):
        d = steps[st]
        for li, il in enumerate(layers):
            if il not in d: continue
            for e in d[il]: seq.append((il, e)); score.append(si >= warm); spec.append(None)
            if spec_window and li + 1 < len(layers):
                iln = layers[li+1]; p = d.get(('p', iln))
                if p: spec[-1] = [(iln, e) for e in p]
    n = sum(score)
    pols = {'LRU': lambda: LRU(cap), 'ARC': lambda: ARC(cap), 'S3-FIFO': lambda: S3FIFO(cap), 'second chance 3': lambda: SecondChance(cap, 3),
            'LFU/4k': lambda: LFUDecay(cap, 4096), 'LFU/32k': lambda: LFUDecay(cap, 32768), 'LFU/256k': lambda: LFUDecay(cap, 262144)}
    out = {}
    for name, mk in pols.items():
        p = mk(); h = 0; spec_started = 0; spec_used = 0
        for i, k in enumerate(seq):
            h += p.access(k) and score[i]
            if spec[i] is not None:
                b = spec_window
                for kk in spec[i]:
                    if b <= 0: break
                    if kk in p: continue
                    p.access(kk); b -= 1; spec_started += 1
        out[name] = h / n
    if not spec_window: out['OPT'] = belady_hits(seq, cap, score) / n
    return out, n

if __name__ == '__main__':
    path = sys.argv[1]; warm = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    steps = load(path)
    print(path.split('/')[-1], len(steps), 'steps, warm', warm)
    for cap in (6000, 7980, 9600, 12000, 16000):
        out, n = run(steps, cap, warm)
        print(f"cap {cap:6d} ({cap/82432*100:4.1f} % of experts): " + "  ".join(f"{k} {v*100:5.2f}" for k, v in out.items()))
    out, n = run(steps, 7980, warm, spec_window=5)
    print("cap 7980 with deployed speculation (kind 1, window 5), hit rate on real lookups: " + "  ".join(f"{k} {v*100:5.2f}" for k, v in out.items()))
