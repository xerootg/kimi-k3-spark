#!/usr/bin/env python3
"""Compare stride-aware per-column FNV-1a tensor TSVs (exit 0 equal, 1 diff, 2 invalid)."""
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass
class Node:
    step: int
    name: str
    op: str
    dtype: str
    shape: tuple
    axis: int
    column: int
    digest: str
    value: str

    @property
    def key(self):
        return getattr(self, "aligned_step", self.step), self.name, self.op

    @property
    def logical_shape(self):
        shape = list(self.shape)
        if self.axis >= 0:
            del shape[self.axis]
        while shape and shape[-1] == 1:
            shape.pop()
        return tuple(shape)

    def describe(self):
        return (f'step={self.step} {self.name} op={self.op} type={self.dtype} '
                f'shape={self.shape} axis={self.axis} column={self.column} '
                f'digest={self.digest}' + (f' value={self.value}' if self.value else ''))


def read_nodes(path, column):
    result = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, 1):
            if line.startswith('#') or not line.strip():
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 10:
                raise ValueError(f'{path}:{line_no}: expected 10 TSV fields')
            step, order, name, op, dtype, shape, axis, col, digest, value = fields
            int(order)
            node = Node(int(step), name, op, dtype, tuple(map(int, shape.split(','))),
                        int(axis), int(col), digest, value)
            if len(node.shape) != 4 or node.axis not in (-1, 0, 1, 2, 3):
                raise ValueError(f'{path}:{line_no}: invalid shape/axis')
            if len(digest) != 16 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'{path}:{line_no}: invalid FNV-1a digest')
            if node.column in (-1, column):
                result.append(node)
    if not result:
        raise ValueError(f'{path}: no records for column {column}')
    return result


def aligned_nodes(a, b, emit):
    """Sample ordinal identifies prefill/decode phase even in a mixed graph.

    The current output token can differ: require the preceding input token to
    match before attributing a decode tensor difference to kernel arithmetic.
    """
    sa = [n for n in a if n.name == '@sampled_token']
    sb = [n for n in b if n.name == '@sampled_token']
    if not sa or not sb or not all(n.value.isdigit() for n in sa + sb):
        return a, b  # legacy fixtures without sampled values
    if len({n.step for n in sa}) != len(sa) or len({n.step for n in sb}) != len(sb):
        raise ValueError('duplicate sampled token in a column/step')
    pairs = []
    for ordinal, (x, y) in enumerate(zip(sa, sb)):
        emit(f'ALIGN sample={ordinal} N1 step={x.step} token={x.value} N2 step={y.step} token={y.value}')
        if ordinal == 0:
            continue  # prefill intermediate nodes are deliberately not captured in N1
        if sa[ordinal-1].value != sb[ordinal-1].value:
            emit('STOP: preceding sampled tokens differ; later decode inputs are unequal')
            break
        pairs.append((x.step, y.step))
    if not pairs:
        raise ValueError('no decode steps with matching preceding sampled tokens')
    aa, bb = [], []
    for x, y in pairs:
        left = [n for n in a if n.step == x and not n.name.startswith('@')]
        right = [n for n in b if n.step == y and not n.name.startswith('@')]
        # Scope bookkeeping to the exact alternate input-embedding branch.
        for nodes in (left, right):
            prefix = [(n.name, n.op) for n in nodes[:4]]
            if prefix == [('embd', 'GET_ROWS'), ('node_1', 'GET_ROWS'),
                          ('node_2', 'DUP'), (' (view)', 'SET_ROWS')]:
                emit(f'BOOKKEEPING step={nodes[0].step}: alternate mixed-input embedding branch (3 nodes)')
                del nodes[1:4]
        aa.extend(left)
        # Normalize only the alignment key; retain physical steps in reports.
        for n in right:
            n.aligned_step = x
        bb.extend(right)
    return aa, bb


def compare(gold, batched, column, emit=print):
    a, b = read_nodes(gold, 0), read_nodes(batched, column)
    if column and not any(n.column == column for n in b):
        raise ValueError(f'{batched}: no column {column}; concurrent ubatch was not captured')
    ambiguous = sum(n.value == 'ambiguous-axis' for n in b)
    if ambiguous:
        emit(f'CAUTION: {ambiguous} records have multiple dimensions equal N; column axis uses the highest matching dimension.')
    a, b = aligned_nodes(a, b, emit)
    previous = deque(maxlen=3)
    skipped = 0
    for i in range(max(len(a), len(b))):
        left = a[i] if i < len(a) else None
        right = b[i] if i < len(b) else None
        reason = None
        if left is None or right is None or left.key != right.key:
            reason = 'ALIGNMENT: missing/reordered node (name + occurrence order within step)'
        elif left.dtype != right.dtype or left.logical_shape != right.logical_shape:
            reason = 'SHAPE/TYPE: tensors are not comparable after removing the column axis'
        elif left.value.startswith('skipped-large ') or right.value.startswith('skipped-large '):
            skipped += 1
            emit('UNOBSERVED: skipped-large N1 ' + left.describe())
            emit('                        N2 ' + right.describe())
            continue
        elif left.digest != right.digest:
            reason = 'FIRST-DIFF COMPUTE'
        if reason:
            for x, y in previous:
                emit('MATCH N1 ' + x.describe())
                emit('      N2 ' + y.describe())
            emit(reason)
            emit('N1 ' + (left.describe() if left else '<end>'))
            emit('N2 ' + (right.describe() if right else '<end>'))
            if right and right.value == 'ambiguous-axis':
                emit('CAUTION: multiple dimensions equal N; inspect the logged axis before attributing this difference.')
            return 1
        previous.append((left, right))
    emit(f'MATCH: {len(a)-skipped} observed records, N2 column {column}; no differing observed tensor in captured steps; {skipped} skipped-large pairs unobserved')
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('n1')
    p.add_argument('n2')
    p.add_argument('column', type=int, choices=(0, 1))
    args = p.parse_args()
    try:
        return compare(args.n1, args.n2, args.column)
    except (ValueError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
