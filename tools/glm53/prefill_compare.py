#!/usr/bin/env python3
"""Compare prefill outputs and resynchronised, named decode compute tensors."""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
import re
import sys
from tensor_firstdiff import read_nodes, aligned_nodes

LAYOUT = {'VIEW', 'RESHAPE', 'PERMUTE', 'TRANSPOSE', 'CONT', 'CPY', 'DUP', 'SET_ROWS', 'REPEAT'}
SPLIT = re.compile(r'ffn_moe_(gate|up|swiglu|down)-(\d+)$')


def shape(node):
    # Singleton dimensions can move across the removed batch axis.
    return tuple(d for d in node.logical_shape if d != 1)


def equal(x, y):
    return x.dtype == y.dtype and shape(x) == shape(y) and x.digest == y.digest


def compute(n):
    # Anonymous graph IDs change with graph construction. Their results are
    # checked at the next named boundary; slot maps describe bank placement.
    return (n.op not in LAYOUT and bool(n.name.strip())
            and not n.name.startswith(('node_', 'cache_', 'moe_'))
            and not n.name.startswith('ffn_moe_slots-'))


def summary(gold, batched, col, emit=print):
    a, b = read_nodes(gold, 0), read_nodes(batched, col)
    if not any(n.column == col for n in b):
        raise ValueError(f'{batched}: no column {col}')
    outputs = []
    for nodes in (a, b):
        samples = [n for n in nodes if n.name == '@sampled_token']
        if not samples or len({n.step for n in samples}) != len(samples):
            raise ValueError('missing/duplicate sampled outputs')
        pairs = []
        for sample in samples:
            logits = [n for n in nodes if n.step == sample.step and n.name == '@logits']
            if len(logits) != 1:
                raise ValueError('missing/duplicate logits at sampled step')
            pairs.append((logits[0], sample))
        outputs.append(pairs)
    oa, ob = outputs
    prefill = sum(equal(x, y) for x, y in zip(oa[0], ob[0]))
    emit(f'PREFILL column={col}: {prefill}/2 output records match (logits + sampled token)')
    failed = prefill != 2 or len(oa) != len(ob)
    if len(oa) != len(ob):
        emit('OUTPUT alignment: unequal sample counts')
    for ordinal, (x, y) in enumerate(zip(oa[1:], ob[1:]), 1):
        ok = all(equal(u, v) for u, v in zip(x, y))
        emit(f'DECODE column={col} sample={ordinal} steps={x[0].step}/{y[0].step}: logits {x[0].digest}/{y[0].digest} {"equal" if equal(x[0], y[0]) else "DIFF"}; sampled token {"equal" if equal(x[1], y[1]) else "DIFF"}')
        failed |= not ok
    # Do not silently accept a suffix stopped by unequal preceding tokens.
    a, b = aligned_nodes(a, b, emit)
    stats = defaultdict(lambda: defaultdict(int))
    for step in sorted({n.key[0] for n in a + b}):
        left = [n for n in a if n.key[0] == step and compute(n)]
        right = [n for n in b if n.key[0] == step and compute(n)]
        # Only recognise split-bank scratch outputs when both graphs actually
        # contain the 9-row early branch AND its late merge map for this layer.
        def split_layers(nodes):
            merges = {n.name.removeprefix('moe_nowait_merge-') for n in nodes if n.name.startswith('moe_nowait_merge-')}
            return {m[2] for n in nodes if (m := SPLIT.fullmatch(n.name))
                    and n.op in ('MUL_MAT_ID', 'GLU') and len(shape(n)) == 2
                    and shape(n)[1] == 9 and m[2] in merges}
        scratch = split_layers([n for n in a if n.key[0] == step]) & split_layers([n for n in b if n.key[0] == step])
        matcher = SequenceMatcher(None, [(n.name, n.op) for n in left],
                                  [(n.name, n.op) for n in right], autojunk=False)
        def count(n, kind):
            m = re.search(r'-(\d+)(?: |$)', n.name)
            stats[(step, int(m[1]) if m else -1)][kind] += 1
        for tag, i, j, k, l in matcher.get_opcodes():
            if tag != 'equal':
                failed = True
                for n in left[i:j] + right[k:l]:
                    count(n, 'unaligned')
                emit(f'ALIGNMENT step={step}: N1 {[n.name+":"+n.op for n in left[i:j]]} N2 {[n.name+":"+n.op for n in right[k:l]]}')
                continue
            for x, y in zip(left[i:j], right[k:l]):
                if x.value.startswith('skipped-large ') or y.value.startswith('skipped-large '):
                    count(x, 'unobserved')
                elif equal(x, y):
                    count(x, 'match')
                elif (m := SPLIT.fullmatch(x.name)) and m[2] in scratch and x.op in ('MUL_MAT_ID', 'GLU') and x.dtype == y.dtype and shape(x) == shape(y) and len(shape(x)) == 2 and shape(x)[1] in (8, 9):
                    count(x, 'expected')
                else:
                    count(x, 'diff')
                    failed = True
                    emit('DIFF N1 ' + x.describe())
                    emit('     N2 ' + y.describe())
    for (step, layer), s in sorted(stats.items()):
        observed = s['match'] + s['diff'] + s['expected']
        emit(f'LAYER column={col} step={step} layer={layer}: {s["match"]}/{observed} compute matches ({100*s["match"]/observed if observed else 0:.2f}%); {s["expected"]} expected split-bank dummy/late-row differences; {s["unaligned"]} unaligned; {s["unobserved"]} unobserved')
    emit(f'SUMMARY column={col}: {"FAIL" if failed else "PASS"}; layout/copy/cache/anonymous/slot-map bookkeeping excluded; expected scratch differences are not exact matches')
    return int(failed)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', action='store_true')
    p.add_argument('gold'); p.add_argument('batched'); p.add_argument('column', type=int, choices=(0, 1))
    args = p.parse_args()
    try:
        sys.exit(summary(args.gold, args.batched, args.column))
    except (ValueError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)
