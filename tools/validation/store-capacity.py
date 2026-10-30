#!/usr/bin/env python3
"""Calculate exact cache allocations from a v1 store index; all indexed layers assumed cached."""
import collections
import math
import struct
import sys

budget = (int(sys.argv[2]) if len(sys.argv) > 2 else 80000) * 1024**2
# type -> (block bytes, CUDA load alignment), matching llama-moe-cache.h.
layouts = {2: (18, 2), 3: (20, 4), 6: (22, 2), 7: (24, 4), 8: (34, 2),
           10: (84, 4), 11: (110, 2), 12: (144, 4), 13: (176, 4), 14: (210, 2),
           16: (66, 2), 17: (74, 2), 18: (98, 2), 19: (50, 2), 20: (18, 2),
           21: (110, 2), 22: (82, 2), 23: (136, 4), 29: (56, 4)}
blocks = {t: layout[0] for t, layout in layouts.items()}
groups = collections.Counter()
with open(sys.argv[1], 'rb') as store:
    header = store.read(64)
    assert header[:4] == b'K3XS'
    version, count, experts, alignment, entry_size = struct.unpack_from('<5I', header, 4)
    assert version == 1 and entry_size == 64
    store.seek(struct.unpack_from('<Q', header, 24)[0])
    for _ in range(count):
        entry = store.read(entry_size)
        types = struct.unpack_from('<3I', entry, 4)
        sizes = struct.unpack_from('<3Q', entry, 16)
        record = struct.unpack_from('<Q', entry, 40)[0]
        groups[types, sizes, record] += 1
weight = sum(n * sum(sizes) for (_, sizes, _), n in groups.items())
for (types, sizes, record), n in groups.items():
    part = 0
    for t, size in zip(types, sizes):
        assert t in layouts and part % layouts[t][1] == 0 and blocks[t] % layouts[t][1] == 0
        part += size
    share = int(float(budget) * (n * sum(sizes)) / weight)
    old_alignment = math.lcm(4096, *(blocks[t] for t in types))
    old = (record + old_alignment - 1) // old_alignment * old_alignment
    new = (record + 4095) // 4096 * 4096
    counts = [min(n * experts, max(0, share // stride - 1)) for stride in (old, new)]
    print(f'{n} layers types={types} sizes={sizes} record={record} budget={share} stride={old}->{new} slots={counts[0]}->{counts[1]} gain={counts[1]-counts[0]} (one zero slot/group)')
