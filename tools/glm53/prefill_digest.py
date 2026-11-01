#!/usr/bin/env python3
"""Label the two isolated prefill output rows for the existing column comparator.

0034 logs output-row indices, not sequence IDs: both isolated prefills use row 0
and decode step 0. Preserve the raw TSV and relabel only the second output pair.
Requires two fresh simultaneous requests, selected in slot order, no cached prompt.
"""
from pathlib import Path
import sys


def normalize(text):
    lines = text.splitlines(keepends=True)
    outputs = []
    for i, line in enumerate(lines):
        if line.startswith('#') or not line.strip():
            continue
        f = line.rstrip('\n').split('\t')
        if len(f) != 10:
            raise ValueError('expected 10 TSV fields')
        if f[0] == '0' and f[2] in ('@logits', '@sampled_token'):
            if f[7] != '0':
                raise ValueError('isolated prefill must use output row 0')
            outputs.append((i, f))
    if [f[2] for _, f in outputs] != ['@logits', '@sampled_token'] * 2:
        raise ValueError('expected two isolated prefill output pairs at decode step 0')
    for i, f in outputs[2:]:
        f[7] = '1'
        lines[i] = '\t'.join(f) + '\n'
    return ''.join(lines)


if __name__ == '__main__':
    source, raw = map(Path, sys.argv[1:])
    text = source.read_text()
    result = normalize(text)
    with raw.open('x') as f:
        f.write(text)
    source.write_text(result)
    print('PREFILL: preserved raw TSV; labeled isolated output pairs as columns 0 and 1')
