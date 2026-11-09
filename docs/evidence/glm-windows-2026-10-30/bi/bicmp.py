#!/usr/bin/env python3
"""Compare LLAMA_BI_DUMP records: each column of an N>1 decode step against the N=1 record at the same (pos, tok).
Nodes are aligned by (op, name, src names) with unnamed node_<i> names blanked (difflib). A DIFF node whose dumped
sources all match (or are not dumped: weights, inputs, views) is reported as a CULPRIT: it changes bytes by itself.
Usage: bicmp.py DUMP [--ref DUMP] [--n N] [--max K] [--steps S]"""
import sys, re, collections, argparse, difflib
ap = argparse.ArgumentParser(); ap.add_argument('dump'); ap.add_argument('--max', type=int, default=12); ap.add_argument('--n', type=int, default=0)
ap.add_argument('--ref', default=''); ap.add_argument('--steps', type=int, default=1, help='batched steps per N to show')
ap.add_argument('--all', action='store_true', help='list every DIFF, not only culprits')
ap.add_argument('--refn', type=int, default=1, help='reference records have this N')
a = ap.parse_args()
R = re.compile(r'^S (\d+) seq=(-?\d+) pos=(-?\d+) tok=(-?\d+) N=(\d+) o=(\d+) (\S+) (\S*) src0=(\S*) src1=(\S*) ne=(\S+) ([0-9a-f]+)$')
SCAFF = {('GET_ROWS', 'inp_mixed_tokens'), ('DUP', 'CUDA0#inp_mixed_embd#0')}
def load(fn):
    steps = collections.OrderedDict()
    for line in open(fn, errors='replace'):
        m = R.match(line.rstrip('\n'))
        if not m: continue
        st, seq, pos, tok, N, o, op, name, s0, s1, ne, h = m.groups()
        steps.setdefault((int(st), int(seq)), {'pos': int(pos), 'tok': int(tok), 'N': int(N), 'nodes': []})['nodes'].append((op, name, s0, s1, ne, h))
    return steps
cl = lambda n: '' if re.fullmatch(r'node_\d+', n) else n
def tokens(nodes): return ['%s|%s|%s|%s' % (op, cl(nm), cl(s0), cl(s1)) for op, nm, s0, s1, ne, h in nodes]
D = load(a.dump); REF = load(a.ref) if a.ref else D
ref = {}
for (st, seq), v in REF.items():
    if v['N'] == a.refn: ref.setdefault((v['pos'], v['tok']), v)
print(f"N={a.refn} reference columns: {sorted(ref)}")
shown = collections.Counter()
for (st, seq), v in D.items():
    if v['N'] == a.refn or (a.n and v['N'] != a.n): continue
    r = ref.get((v['pos'], v['tok']))
    if r is None or shown[v['N']] >= a.steps: continue
    shown[v['N']] += 1
    rn, bn = r['nodes'], v['nodes']
    sm = difflib.SequenceMatcher(None, tokens(rn), tokens(bn), autojunk=False)
    pairs = [(i, j) for blk in sm.get_matching_blocks() for i, j in zip(range(blk.a, blk.a + blk.size), range(blk.b, blk.b + blk.size))]
    rname = {nd[1]: i for i, nd in enumerate(rn)}; bname = {nd[1]: j for j, nd in enumerate(bn)}
    pmap = dict(pairs)
    same = diff = shape = 0; culprits = []; diffs = []
    status = {}
    for i, j in pairs:
        op, nm, s0, s1, ne, h = rn[i]; bne, bh = bn[j][4], bn[j][5]
        if (op, s1) in SCAFF or (op, s0) in SCAFF: status[i] = 'same'; continue
        if ne != bne: shape += 1; status[i] = 'shape'; continue
        if h == bh: same += 1; status[i] = 'same'; continue
        diff += 1; status[i] = 'diff'
        srcok = []
        for s in (s0, s1):
            if s in ('-', '') or s not in rname: srcok.append('nd'); continue
            si = rname[s]
            st_ = status.get(si)
            srcok.append('ok' if st_ == 'same' else ('?' if st_ is None else st_))
        rec = (op, nm, s0, s1, ne, srcok)
        diffs.append(rec)
        if all(x in ('ok', 'nd') for x in srcok): culprits.append(rec)
    print(f"\n== step {st} seq {seq} N={v['N']} pos={v['pos']} tok={v['tok']}: aligned {len(pairs)}/{len(rn)} ref, {len(bn)} batch; same {same} diff {diff} shape {shape}; culprits {len(culprits)}")
    for op, nm, s0, s1, ne, so in (diffs if a.all else culprits)[:a.max]:
        print(f"   {'CULPRIT' if all(x in ('ok','nd') for x in so) else 'diff   '} {op} {nm} ({s0}:{so[0]}, {s1}:{so[1]}) ne={ne}")
    # first unaligned ref nodes (graph structure differences)
    un = [i for i in range(len(rn)) if i not in pmap]
    if un: print("   unaligned ref nodes (first 6): " + "; ".join('%s %s' % (rn[i][0], rn[i][1]) for i in un[:6]))
