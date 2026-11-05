#!/usr/bin/env python3
"""Expand an engine-created v1 seed snapshot using the matching v1 store index.
Run offline on GH200 after an isolated 2k prefill and graceful SIGINT seed run.
Never modifies seed/store; output must be new. Each seed group needs a valid key.
"""
import argparse
from pathlib import Path
import struct


def expand(store, seed, output):
    with store.open('rb') as f:
        h = f.read(64)
        magic, version, count, experts, align, entry_size, offset = struct.unpack_from('<4s5IQ', h)
        if magic != b'K3XS' or version != 1 or entry_size != 64 or not align:
            raise ValueError('need verified v1 K3XS store')
        f.seek(offset)
        entries = {}
        for _ in range(count):
            raw = f.read(64)
            layer, *layout = struct.unpack_from('<i3I4Q', raw)
            if layer in entries:
                raise ValueError('duplicate store layer')
            # types[3], sizes[3], record_size: sufficient for this GLM fixed shape.
            entries[layer] = tuple(layout)
    data = seed.read_bytes()
    magic, version, size, nex, layers, groups = struct.unpack_from('<4sIQIII', data)
    if (magic, version, size, nex) != (b'K3SN', 1, store.stat().st_size, experts):
        raise ValueError('seed/store identity differs')
    pos, claimed, total = 28, set(), 0
    seen_all = set()
    p2 = 28
    for _ in range(groups):
        _nl, _rec, _ne = struct.unpack_from("<IQI", data, p2); p2 += 16
        seen_all.update(struct.unpack_from("<iH", data, p2 + i * 6)[0] // experts for i in range(_ne)); p2 += _ne * 6
    result = bytearray(data[:28])
    for _ in range(groups):
        nlayer, record, nentry = struct.unpack_from('<IQI', data, pos)
        pos += 16
        keys = [struct.unpack_from('<iH', data, pos + i * 6)[0] for i in range(nentry)]
        pos += nentry * 6
        layouts = {entries[k // experts] for k in keys if k // experts in entries}
        if len(layouts) != 1:
            raise ValueError('seed every group first; ambiguous or empty group')
        layout = layouts.pop()
        seen = {k // experts for k in keys}; group_layers = sorted(k for k, v in entries.items() if v == layout and k in seen_all)
        if len(group_layers) != nlayer or layout[-1] != record or claimed.intersection(group_layers):
            raise ValueError('group layout mismatch; do not guess snapshot geometry')
        claimed.update(group_layers)
        full = [l * experts + e for l in group_layers for e in range(experts)]
        if any(k >= layers * experts for k in full):
            raise ValueError('store layer outside model')
        result.extend(struct.pack('<IQI', nlayer, record, len(full)))
        result.extend(b''.join(struct.pack('<iH', k, 1) for k in full))
        total += len(full)
    if claimed != set(entries):
        print("store layers not in the engine (not seeded, left out):", sorted(set(entries) - claimed))
    if pos != len(data) or not claimed <= set(entries):
        raise ValueError('snapshot/store group coverage differs')
    with output.open('xb') as f:
        f.write(result)
    payload = sum(v[-1] * experts for v in entries.values())
    print(f'full snapshot: {total} experts, {payload} record bytes; startup must restore exactly this count')
    print('HBM placement starts uniform; run untimed decode to train hot-tier promotions, then freeze snapshot')
    return total


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('store', type=Path)
    p.add_argument('seed', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    expand(a.store, a.seed, a.output)
