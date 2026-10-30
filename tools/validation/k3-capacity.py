#!/usr/bin/env python3
"""Reproduce K3 capacity; group counts from gh200-plan.md, index capture is truncated."""
import json
import math
from pathlib import Path

text = Path('evidence/gh200-store-and-hw.txt').read_text()
decoder = json.JSONDecoder()
pos = text.index('[', text.index('"layers"')) + 1
layers = []
while True:
    pos = text.index('{', pos)
    try:
        layer, end = decoder.raw_decode(text, pos)
    except json.JSONDecodeError:
        break
    layers.append(layer)
    pos = end
assert len(layers) == 26
sizes = {19: 50, 16: 66}
fixtures = []
for layer in layers:
    off = 0
    for part in ('gate', 'up', 'down'):
        t = layer[part]
        fixtures.append(f"{t['type']} {off} {t['dims'][0]//256*sizes[t['type']]}")
        off += layer[part + '_size']
    assert off == layer['record_size']
# GLM Q2 offsets documented in RESULT-0004.md; single-block test rows (actual rows absent).
fixtures += ['17 0 74', '17 3637248 74', '18 7274496 98']
Path('validation/store-parts.tsv').write_text('\n'.join(fixtures) + '\n')
groups = [(80,(19,19,19)), (10,(16,16,19)), (2,(19,19,16))]
records = [sum(sizes[t]*43008 for t in types) for _,types in groups]
assert sum(n*r*896 for (n,_),r in zip(groups,records)) + 12288 == 545349709824
weight = sum(n*r for (n,_),r in zip(groups,records))
print('K3: 26 complete captured layers; 80/10/2 total group counts from plan, reconciled to total_bytes.')
print('Assume all 92 layers cached, no HBM budget/valve/remote slots; reserve one zero slot/group.')
print()
print('| Types gate/up/down | Layers | Record B | Old stride B | New stride B | Pad removed B/slot | Removed % old | 424000 MiB slots old -> new | 52000 MiB slots old -> new |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
totals = [[0,0],[0,0]]
for (n,types),record in zip(groups,records):
    alignment = math.lcm(4096, *(sizes[t] for t in types))
    old = (record+alignment-1)//alignment*alignment
    new = (record+4095)//4096*4096
    slots = []
    for i,mib in enumerate((424000,52000)):
        share = int(float(mib*1024**2)*(n*record)/weight)
        counts = [min(n*896,max(0,share//stride-1)) for stride in (old,new)]
        for j,count in enumerate(counts):
            totals[i][j] += count
        slots.append(f'{counts[0]} -> {counts[1]} (+{counts[1]-counts[0]})')
    names = '/'.join('IQ1_S' if t == 19 else 'IQ2_XXS' for t in types)
    print(f'| {names} | {n} | {record} | {old} | {new} | {old-new} | {(old-new)/old*100:.3f}% | {slots[0]} | {slots[1]} |')
assert totals == [[65102,67198],[7980,8237]]
for mib,(old,new) in zip((424000,52000),totals):
    print(f'{mib} MiB total usable slots: {old} -> {new} (+{new-old}, {(new/old-1)*100:.3f}%)')
print('PASS: K3 index sizes/offsets and total-byte reconciliation; wrote 81 store-part fixtures')
