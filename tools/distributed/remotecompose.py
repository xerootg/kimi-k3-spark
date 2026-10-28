#!/usr/bin/env python3
"""Build a remote box's expert file from an existing export (remoteexport.py output) instead of the Spark's store.

usage: remotecompose.py SRC_IDX SRC_BIN SET.txt OUT_IDX [--subset]  > records.bin

SET.txt: "layer expert" lines (remoteset.py --out-prefix). Every expert must be in SRC_IDX. Records go to stdout in
SRC order with a new index (same format: "<global_id> <gate_type> <down_type> <byte_offset>"). --subset writes only
an index into SRC_BIN itself (no copy, nothing on stdout): for a server on the same box as SRC_BIN.
Record sizes follow from the types (Kimi K3: 3584 x 3072 gate/up, 3072 x 3584 down; IQ1_S 50 B / IQ2_XXS 66 B per
256 values).
"""
import sys

N_EXPERT, D, F = 896, 3584, 3072
BLOCK = {19: 50, 16: 66}  # bytes per 256 values

def rec_size(tg, td):
    return 2 * F * D // 256 * BLOCK[tg] + D * F // 256 * BLOCK[td]

def main():
    src_idx, src_bin, setf, out_idx = sys.argv[1:5]
    subset = "--subset" in sys.argv
    want = {int(a) * N_EXPERT + int(b) for a, b in (l.split() for l in open(setf) if l.strip())}
    rows = []
    for l in open(src_idx):
        g, tg, td, off = map(int, l.split())
        if g in want:
            rows.append((g, tg, td, off))
    missing = want - {r[0] for r in rows}
    if missing:
        sys.exit(f"{len(missing)} experts of the set are not in {src_idx}, e.g. gid {sorted(missing)[0]}")
    out = sys.stdout.buffer
    pos = 0
    with open(src_bin, "rb") as f, open(out_idx, "w") as o:
        for g, tg, td, off in rows:
            n = rec_size(tg, td)
            if subset:
                o.write(f"{g} {tg} {td} {off}\n")
                continue
            f.seek(off)
            b = f.read(n)
            assert len(b) == n, (g, len(b))
            out.write(b)
            o.write(f"{g} {tg} {td} {pos}\n")
            pos += n
    print(f"{len(rows)} records, {pos if not subset else sum(rec_size(r[1], r[2]) for r in rows)} bytes", file=sys.stderr)

main()
