#!/usr/bin/env python3
"""Prepare store records, compare complete dumps, or compare g53x check to CPU.
No network; output paths must be outside the read-only slice directory.
"""
import argparse
import json
from pathlib import Path
import struct

p = argparse.ArgumentParser()
s = p.add_subparsers(dest='mode', required=True)
a = s.add_parser('records'); a.add_argument('data', type=Path); a.add_argument('out', type=Path)
a = s.add_parser('compare'); a.add_argument('cuda', type=Path); a.add_argument('cpu', type=Path)
a = s.add_parser('sycl'); a.add_argument('check_prefix', type=Path); a.add_argument('cpu', type=Path)
args = p.parse_args()
if args.mode == 'records':
    data = args.data.resolve(); out = args.out.resolve()
    if out == data or data in out.parents:
        p.error('output must be outside the read-only data directory')
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data / 'UD-Q2_K_XL.json').read_text())
    for layer in sorted({r['layer'] for r in rows}):
        parts = [next(r for r in rows if r['layer'] == layer and r['proj'] == pr) for pr in ('gate','up','down')]
        assert all(r['n_expert'] == 8 for r in parts)
        path = out / f'L{layer}.records'
        with path.open('wb') as f:
            handles = [(data / r['file']).open('rb') for r in parts]
            try:
                for e in range(8):
                    for r, h in zip(parts, handles):
                        b = h.read(r['bytes_per_expert']); assert len(b) == r['bytes_per_expert']; f.write(b)
                assert all(not h.read(1) for h in handles)
            finally:
                for h in handles: h.close()
        print(f'{path}: 8 records, stride {sum(r["bytes_per_expert"] for r in parts)}')
elif args.mode == 'compare':
    bad = False
    for name in ('x','ids','g','u','h','dn'):
        a = (args.cuda / (name+'.bin')).read_bytes(); b = (args.cpu / (name+'.bin')).read_bytes()
        if len(a) != len(b):
            print(f'{name}: SIZE DIFFER {len(a)} vs {len(b)}'); bad = True; continue
        n = sum(a[i:i+4] != b[i:i+4] for i in range(0,len(a),4))
        print(f'{name}: {len(a)//4-n}/{len(a)//4} bit-identical ({n} differ)'); bad |= bool(n)
    raise SystemExit(2 if bad else 0)
else:
    pre, out = args.check_prefix, args.cpu
    out.mkdir(parents=True, exist_ok=True)
    # Prepare one-case input for CPU, or compare after the CPU has run.
    x = Path(str(pre)+'.x').read_bytes()
    ids = struct.pack('<8i',*range(8))
    if (out/'g.bin').exists():
        assert (out/'x.bin').read_bytes() == x, 'CPU/SYCL input differs'
        assert (out/'ids.bin').read_bytes() == ids, 'CPU/SYCL ids differ'
    else:
        (out/'x.bin').write_bytes(x)
        (out/'ids.bin').write_bytes(ids)
    bad = False
    for name in ('g','u','h','dn'):
        ref = out / (name+'.bin')
        if not ref.exists():
            print('prepared x.bin/ids.bin; run CPU slices into a different directory, then sycl with that directory'); break
        a = Path(str(pre)+'.'+name).read_bytes(); b = ref.read_bytes()
        same = a == b; print(f'{name}: {"bit-identical" if same else "DIFFER"}'); bad |= not same
    raise SystemExit(2 if bad else 0)
