#!/usr/bin/env python3
"""Offline GLM owner planning and bounded remote replay. Standard library only.

No network operations. Generated deploy.sh must be run on the Spark explicitly.
See RESULT.md and --help for assumptions and limits.
"""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import random
import shlex
import struct

LAYERS = range(3, 78)
KEYS = [(l, e) for l in LAYERS for e in range(256)]
RECORD = 12091392


def load_trace(path, fmt='pfdump'):
    """Strict framing; fill traces have no token or predictor fields."""
    steps = collections.defaultdict(dict)
    preds = {}
    step, previous = -1, 999
    with open(path, 'rb') as f:
        while True:
            h = f.read(16 if fmt == 'pfdump' else 8)
            if not h:
                break
            if len(h) != (16 if fmt == 'pfdump' else 8):
                raise ValueError('truncated trace header')
            if fmt == 'pfdump':
                st, layer, kind, n = struct.unpack('<4i', h)
            else:
                layer, n = struct.unpack('<2i', h)
                if layer <= previous:
                    step += 1
                previous = layer
                st, kind = step, 3
            if not 0 <= n <= 1048576:
                raise ValueError('invalid trace length')
            data = f.read(n * 4)
            if len(data) != n * 4:
                raise ValueError('truncated trace payload')
            if layer not in LAYERS:
                continue
            if kind == 3:
                ids = list(struct.unpack(f'<{n}i', data))
                if n != 8:  # prefill is never evidence of decode hotness
                    continue
                if len(set(ids)) != 8 or any(e < 0 or e >= 256 for e in ids):
                    raise ValueError('invalid GLM top-8')
                if layer in steps[st]:
                    raise ValueError('duplicate token/layer')
                steps[st][layer] = ids
            elif kind == 1:
                if n != 256:
                    raise ValueError('expected 256 predictor scores')
                scores = struct.unpack('<256f', data)
                if not all(math.isfinite(x) for x in scores):
                    raise ValueError('nonfinite predictor scores')
                preds[st, layer] = sorted(range(256), key=lambda e: -scores[e])[:8]
    complete = [(st, steps[st]) for st in sorted(steps) if len(steps[st]) == 75]
    if not complete:
        raise ValueError('no complete 75-layer decode tokens')
    return complete, preds, len(steps) - len(complete)


def synthetic(tokens, alpha, seed):
    rng = random.Random(seed)
    # Rotate ranks per layer so uniform tie-breaking does not fill only early layers.
    weights = [1 / (i + 1) ** alpha for i in range(256)]
    steps = []
    for st in range(tokens):
        layers = {}
        for l in LAYERS:
            ids = []
            while len(ids) < 8:
                e = (rng.choices(range(256), weights)[0] + l * 37) % 256
                if e not in ids:
                    ids.append(e)
            layers[l] = ids
        steps.append((st, layers))
    return steps


def metadata(path):
    """Read only K3XS header/index; offsets must come from the target v2 store."""
    result = {}
    with open(path, 'rb') as f:
        h = f.read(64)
        if len(h) != 64:
            raise ValueError('short store header')
        magic, ver, nl, nx, align, esz, off, data = struct.unpack('<4sIIIIIQQ', h[:40])
        if magic != b'K3XS' or ver not in (1, 2) or nx != 256 or esz != 64:
            raise ValueError('invalid GLM K3XS header')
        if align < 4096 or align & (align - 1) or off < 64 or data < off + nl * 64:
            raise ValueError('invalid store alignment/index')
        f.seek(off)
        for _ in range(nl):
            b = f.read(64)
            if len(b) != 64:
                raise ValueError('short store index')
            l, tg, tu, td, g, u, d, size, base, stride = struct.unpack('<4I6Q', b)
            stride = stride if ver == 2 else size
            if stride < size or stride % align or base % align:
                raise ValueError('invalid store stride')
            for e in range(nx):
                result[l, e] = (size, g + u + d, tg, td, base + e * stride)
    if any(k not in result for k in KEYS):
        raise ValueError('store missing GLM MoE layers')
    return result


def place(freq, meta, slots, budgets):
    ranked = sorted(KEYS, key=lambda k: (-freq[k], (k[1] - k[0]*37) % 256, k[0]))
    owner = {k: 'spark' for k in ranked[:slots]}
    free = dict(zip(('nuc2', 'nuc3'), budgets))
    for k in ranked[slots:]:
        fits = [b for b in free if free[b] >= meta[k][0]]
        if fits:
            # Fill the larger remaining arena; replay enforces the global row cap.
            b = max(fits, key=lambda b: free[b])
            owner[k] = b + '-ram'
            free[b] -= meta[k][0]
    load = {'nuc2': 0., 'nuc3': 0.}
    for k in ranked:
        if k not in owner:
            b = min(load, key=load.get)
            owner[k] = b + '-ssd'
            load[b] += max(freq[k], 1e-9) * meta[k][0] / ({'nuc2': 5.7e9, 'nuc3': 4.6e9}[b])
    return owner


def percentile(xs, p):
    return sorted(xs)[max(0, math.ceil(len(xs) * p) - 1)] if xs else 0


def replay(steps, preds, owner, meta, rows=2, precision=.853, seed=53,
           deadline=2200, ssd=False, slots=6859, warm_keys=None):
    """LRU bank, prediction reads (window 5), resident remotes, sequential disk service.

    SSD ideal mode assumes a NEW weight-prefetch channel; never emitted as live config.
    Mean and tail are separate scenarios, not a fitted latency distribution.
    """
    rng = random.Random(seed)
    cache = collections.OrderedDict()
    def touch(k):
        cache[k] = None
        cache.move_to_end(k)
        if len(cache) > slots:
            cache.popitem(last=False)
    # Optimistic warm bank from the desired Spark set; real engine cannot pin this set.
    for k in (warm_keys if warm_keys is not None else
              [k for k in owner if owner[k] == 'spark']):
        touch(k)
    totals = collections.Counter()
    dist = {b: [] for b in ('nuc2', 'nuc3')}
    for st, layers in steps:
        for l, ids in sorted(layers.items()):
            ks = [(l, e) for e in ids]
            predicted = preds.get((st, l))
            if predicted is None:
                predicted = [e for e in ids if rng.random() < precision]
                wrong = [e for e in range(256) if e not in ids]
                predicted += rng.sample(wrong, 8-len(predicted))
                rng.shuffle(predicted)
            # Predictions are issued at l-1, not before first MoE layer.
            if l == 3:
                predicted = []
            pre = [(l, e) for e in predicted]
            reads = 0
            for k in pre:
                if reads == 5:
                    break
                if k not in cache and (owner[k] == 'spark' or not owner[k].endswith('-ram')):
                    if not (ssd and owner[k].endswith('-ssd')):
                        touch(k)
                        totals['spark_prefetch_reads'] += 1
                    reads += 1
            candidates = [k for k in ks if k not in cache and
                          (owner[k].endswith('-ram') or ssd and owner[k].endswith('-ssd'))]
            local = sum(k not in cache and k not in candidates for k in ks)
            selected = candidates[:min(rows, max(0, 8-local))]
            per = collections.Counter(owner[k].split('-')[0] for k in selected)
            waits = []
            tails = []
            for b in dist:
                n = per[b]
                dist[b].append(n)
                if not n:
                    continue
                # Full Spark-client measurements, includes LAN: do not add 0.104 ms again.
                median = .917 + (n-1) * (3.493-.917)/7
                tail = 2.023 if n == 1 else 1.752
                disk = [k for k in selected if owner[k] == b+'-ssd']
                cold = sum(meta[k][0] for k in disk if k not in pre) / (5.7e6 if b == 'nuc2' else 4.6e6)
                # Ideal prefetch must finish its reads within one layer too.
                pref_bytes = sum(meta[k][0] for k in pre if ssd and owner[k] == b+'-ssd')
                spill = max(0, pref_bytes/(5.7e6 if b == 'nuc2' else 4.6e6)-1000/4.9/75)
                median += cold + spill
                tail += cold + spill
                waits.append(max(0, min(median, deadline/1000)-1000/4.9/75))
                tails.append(max(0, min(tail, deadline/1000)-1000/4.9/75))
                if median*1000 > deadline:
                    totals['deadline_fallbacks'] += n
                    for k in selected:
                        if owner[k].startswith(b):
                            touch(k)
                            totals['spark_demand_reads'] += 1
                else:
                    totals['remote_rows'] += n
            for k in ks:
                if k not in selected:
                    if k not in cache:
                        totals['spark_demand_reads'] += 1
                    touch(k)
            totals['layers'] += 1
            totals['wait_ms'] += max(waits, default=0)
            totals['tail_wait_ms'] += max(tails, default=0)
    nt = len(steps)
    return dict(tokens=nt, spark_reads_per_token=(totals['spark_prefetch_reads']+totals['spark_demand_reads'])/nt,
                demand_reads_per_token=totals['spark_demand_reads']/nt,
                prefetch_reads_per_token=totals['spark_prefetch_reads']/nt,
                remote_rows_per_token=totals['remote_rows']/nt,
                fallback_rows_per_token=totals['deadline_fallbacks']/nt,
                rows={b: dict(p50=percentile(v, .5), p99=percentile(v, .99), maximum=max(v)) for b,v in dist.items()},
                wait_ms_per_token=totals['wait_ms']/nt,
                p99_service_scenario_wait_ms_per_token=totals['tail_wait_ms']/nt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', help='training decode trace; no trace = synthetic assumption')
    p.add_argument('--eval', help='held-out trace; defaults to second half of training')
    p.add_argument('--format', choices=['pfdump', 'fill'], default='pfdump')
    p.add_argument('--store', help='target NUC v2 K3XS store or header/index copy; needed for CPU IDX')
    p.add_argument('--owners', type=Path, help='replay an existing owners.csv instead of placing')
    p.add_argument('--out', type=Path, default=Path('placement'))
    p.add_argument('--tokens', type=int, default=400)
    p.add_argument('--zipf', type=float, default=1., help='0 = uniform, default Zipf exponent 1')
    p.add_argument('--seed', type=int, default=53)
    p.add_argument('--spark-slots', type=int, default=6859)
    p.add_argument('--nuc2-gb', type=float, default=44., help='net record budget, after 6 GB headroom')
    p.add_argument('--nuc3-gb', type=float, default=18., help='net record budget, after 6 GB headroom')
    p.add_argument('--rows', type=int, choices=[1, 2], default=2)
    p.add_argument('--precision', type=float, default=.853)
    p.add_argument('--deadline-us', type=int, default=2200,
                   help='k=1 p99 2023 us + 177 us margin, below 2721 us mean spacing')
    a = p.parse_args()
    if not 1 <= a.spark_slots <= len(KEYS) or a.tokens < 4 or min(a.nuc2_gb,a.nuc3_gb) < 0 or not 0 <= a.precision <= 1 or a.deadline_us <= 0 or a.zipf < 0:
        p.error('invalid capacity, token count, precision, exponent or deadline')
    steps, preds, dropped = load_trace(a.trace, a.format) if a.trace else (synthetic(a.tokens, a.zipf, a.seed), {}, 0)
    if len(steps) < 4:
        p.error('need at least four complete tokens')
    train = steps[:len(steps)//2] if not a.eval else steps
    ev, ep, _ = load_trace(a.eval, a.format) if a.eval else (steps[len(steps)//2:], preds, 0)
    freq = collections.Counter((l,e) for _, ls in train for l, ids in ls.items() for e in ids)
    meta = metadata(a.store) if a.store else {k:(RECORD,RECORD,None,None,None) for k in KEYS}
    owner = place(freq, meta, a.spark_slots, [a.nuc2_gb*1e9,a.nuc3_gb*1e9])
    if a.owners:
        with a.owners.open() as f:
            records = list(csv.DictReader(f))
        owner = {(int(r['layer']), int(r['expert'])): r['owner'] for r in records}
        if len(records) != len(KEYS) or set(owner) != set(KEYS) or any(v not in
                {'spark','nuc2-ram','nuc3-ram','nuc2-ssd','nuc3-ssd'} for v in owner.values()):
            p.error('owner CSV must uniquely cover all 19200 pairs with known owners')
        if sum(v == 'spark' for v in owner.values()) > a.spark_slots:
            p.error('owner CSV exceeds Spark slots')
        for b, gb in [('nuc2', a.nuc2_gb), ('nuc3', a.nuc3_gb)]:
            if sum(meta[k][0] for k in owner if owner[k] == b+'-ram') > gb*1e9:
                p.error('owner CSV exceeds RAM budget')
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'owners.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['layer','expert','owner','train_count','record_bytes'])
        w.writerows((*k,owner[k],freq[k],meta[k][0]) for k in KEYS)
    for name in ['spark','nuc2-ram','nuc3-ram','nuc2-ssd','nuc3-ssd']:
        keys = [k for k in KEYS if owner[k] == name]
        (a.out/(name+'.set')).write_text(''.join(f'{l} {e}\n' for l,e in keys))
        if a.store and name.endswith('-ram'):
            (a.out/(name+'.idx')).write_text(''.join(f'{l*256+e} {meta[l,e][2]} {meta[l,e][3]} {meta[l,e][4]}\n' for l,e in keys))
    report = {'source': 'trace' if a.trace else f'synthetic Zipf alpha={a.zipf}',
              'discarded_incomplete_tokens': dropped, 'counts': dict(collections.Counter(owner.values())),
              'assumptions': ['Spark desired set is not pinned; replay uses LRU warmed from it',
                              'missing predictor scores use Bernoulli top-8 overlap, window 5',
                              'fast mode 0 timings; exact mode 1 does not fit',
                              'SSD ideal mode requires an unimplemented weight-prefetch channel'],
              'resident': replay(ev, ep, owner, meta, a.rows, a.precision, a.seed,a.deadline_us,False,a.spark_slots),
              'ssd_ideal': replay(ev, ep, owner, meta, a.rows, a.precision, a.seed,a.deadline_us,True,a.spark_slots)}
    baseline = replay(ev, ep, {k:'spark' for k in KEYS},meta,a.rows,a.precision,a.seed,a.deadline_us,False,a.spark_slots,
                      [k for k in owner if owner[k] == 'spark'])
    report['baseline'] = baseline
    for mode in ('resident','ssd_ideal'):
        r = report[mode]
        # Serial Spark read bandwidth 13 GB/s; only a sensitivity, much I/O overlaps compute.
        saved = (baseline['spark_reads_per_token']-r['spark_reads_per_token'])*RECORD/13e6
        r['tok_s_no_io_credit'] = 1000/(1000/4.9+r['wait_ms_per_token'])
        r['tok_s_serial_io_credit'] = 1000/max(1,1000/4.9-saved+r['wait_ms_per_token'])
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    env = {'LLAMA_EXPERT_STORE_REMOTE':'nuc2:19653,nuc3:19653',
           'LLAMA_EXPERT_STORE_REMOTE_SET':'/srv/glm-5.3/placement/nuc2.served.set,/srv/glm-5.3/placement/nuc3.served.set',
           'LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US':str(a.deadline_us),
           'LLAMA_EXPERT_STORE_REMOTE_ROWS':str(a.rows), 'LLAMA_EXPERT_STORE_SPLIT':'1',
           'LLAMA_EXPERT_STORE_BYTE_STRIDE':'1',
           'LLAMA_EXPERT_STORE_PREFETCH':'1', 'LLAMA_EXPERT_STORE_PREFETCH_INPUT':'3',
           'LLAMA_EXPERT_STORE_PREFETCH_MAX':'5'}
    (a.out/'engine.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items()))
    (a.out/'engine.yaml').write_text('env:\n'+''.join(f'  - name: {k}\n    value: "{v}"\n' for k,v in env.items()))
    script = '''#!/usr/bin/env bash
# Run ON SPARK. Requires prebuilt /srv/glm-5.3/bin/g53x on NUCs.
set -euo pipefail
cd "$(dirname "$0")"
for host in nuc2 nuc3; do
  ssh "$host" 'mkdir -p /srv/glm-5.3/placement'
  scp "$host-ram.set" "$host:/srv/glm-5.3/placement/resident.set"
  case "$host" in nuc2) budget=BUDGET2;; nuc3) budget=BUDGET3;; esac
  ssh "$host" "cd /srv/glm-5.3/placement; nohup env G53X_EXACT=0 K3X_MAX_GB=$budget K3X_SETOUT=served.set /srv/glm-5.3/bin/g53x resident.set /srv/glm-5.3/store/glm53-q2-v2.k3xs - - serveidx 19653 >server.log 2>&1 < /dev/null &"
  # SETOUT appears after load; process failure/timeout stops deployment.
  ssh "$host" 'cd /srv/glm-5.3/placement; for i in $(seq 1 600); do if test -s served.set && test -n "$(ss -ltnH sport = :19653)"; then exit 0; fi; sleep 1; done; tail -20 server.log; exit 1'
  scp "$host:/srv/glm-5.3/placement/served.set" "$host.served.set"
done
printf '%s\\n' 'Mount this directory at /srv/glm-5.3/placement on Spark engine; merge engine.yaml or source engine.env.'
'''.replace('BUDGET2',str(a.nuc2_gb)).replace('BUDGET3',str(a.nuc3_gb))
    # Refuse stale files / existing listeners rather than silently adopting an old server.
    script = script.replace("'mkdir -p /srv/glm-5.3/placement'", """'mkdir -p /srv/glm-5.3/placement; test ! -e /srv/glm-5.3/placement/served.set; test -z "$(ss -ltnH sport = :19653)"'""")
    (a.out/'deploy.sh').write_text(script)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
