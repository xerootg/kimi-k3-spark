#!/usr/bin/env python3
"""Summarize nsys CSVs using only the Python standard library.

Use directories containing report-name CSVs, or --prefix glm-short/glm-long
for win2m.sh's shared export directory. No pre/post-capture idle is counted.
"""

import argparse
import csv
import re
from collections import defaultdict
from functools import lru_cache


BANK = 'host-bank expert GEMV'
DENSE = 'dense-spine GEMV (device)'
FAMILIES = (BANK, DENSE, 'shared expert', 'attention / flash-attn',
            'DSA indexer', 'norms', 'rope', 'cpy / get_rows / set_rows', 'other')
BUCKETS = ('<50 us', '50-500 us', '0.5-5 ms', '>5 ms')
REPORTS = ('cuda_gpu_kern_sum', 'cuda_gpu_trace', 'cuda_api_sum')


@lru_cache(maxsize=256)
def normalized(value):
    return ' '.join(value.strip().lower().split())


def column(row, *names):
    for name in names:
        if name in row:
            return row[name]
        for key, value in row.items():
            if normalized(key) == normalized(name):
                return value
    return None


def time_ns(row, *stems):
    for stem in stems:
        for unit, scale in (('ns', 1), ('us', 1000), ('µs', 1000),
                            ('ms', 1000000), ('s', 1000000000)):
            value = column(row, stem + ' (' + unit + ')')
            if value is not None and value.strip():
                return float(value.replace(',', '')) * scale
    return None


def read_csv(path):
    """Find the header after arbitrary nsys banners/comments/blank lines."""
    with open(path, newline='', encoding='utf-8-sig') as source:
        reader = csv.reader(source)
        for fields in reader:
            keys = [normalized(field) for field in fields]
            if ('name' in keys or 'kernel name' in keys) and any(
                    key.startswith(('total time (', 'start (', 'start time ('))
                    for key in keys):
                header = fields
                break
        else:
            raise ValueError(path + ': no recognizable CSV header')
        result = []
        for fields in reader:
            if not fields or not any(field.strip() for field in fields):
                continue
            if fields[0].lstrip().startswith('#') or fields == header:
                continue
            if len(fields) != len(header):
                raise ValueError(path + ': malformed CSV row')
            result.append(dict(zip(header, fields)))
        return result


def load_reports(directory, prefix=None):
    # Known output names avoid filesystem dependencies; custom nsys -o names
    # are supported explicitly via --prefix. Both common report suffixes work.
    prefixes = [prefix] if prefix is not None else ['', 'glm-short', 'glm-long']
    found = {}
    for candidate in prefixes:
        reports = {}
        for report in REPORTS:
            for separator in (['_', '-'] if candidate else ['']):
                path = directory.rstrip('/') + '/' + candidate + separator + report + '.csv'
                try:
                    reports[report] = read_csv(path)
                    break
                except FileNotFoundError:
                    pass
        if reports:
            found[candidate] = reports
    if len(found) > 1:
        raise ValueError('multiple traces in ' + directory + '; select --prefix (and --prefix2)')
    return next(iter(found.values()), {})


def grid(row, axis):
    value = column(row, 'Grd' + axis, 'Grid ' + axis, 'Grid' + axis,
                   'Grid ' + axis + ' (blocks)')
    if value and value.strip():
        return int(value.replace(',', ''))
    name = column(row, 'Name', 'Kernel Name') or ''
    match = re.search(r'grid[ =:(]*(\d+)[x, ]+(\d+)[x, ]*(\d+)?', name, re.I)
    return int(match.group(1 if axis == 'X' else 2)) if match else None


def bank_type(name):
    match = re.search(r'mul_mat_vec_q<\s*(?:\(ggml_type\)\s*)?(17|18|23|IQ2_XS|IQ3_XXS|IQ4_XS)\b', name, re.I)
    return bool(match)


def kernel_family(name, y=None):
    name = name.lower()
    if any(tag in name for tag in ('indexer', 'topk', 'top_k', 'dsa')):
        return 'DSA indexer'
    if any(tag in name for tag in ('shared', 'shexp', 'she_expert')):
        return 'shared expert'
    if 'mul_mat_vec_q' in name or 'mmvq' in name:
        return BANK if bank_type(name) and (y in (8, 9) or y is None) else DENSE
    if any(tag in name for tag in ('attention', 'attn', 'fattn')):
        return 'attention / flash-attn'
    if 'norm' in name:
        return 'norms'
    if 'rope' in name:
        return 'rope'
    if any(tag in name for tag in ('cpy', 'get_rows', 'set_rows')):
        return 'cpy / get_rows / set_rows'
    return 'other'


def api_family(name):
    name = name.lower()
    if 'streamwaitvalue' in name:
        return 'cudaStreamWaitValue / cuStreamWaitValue32'
    for tag, label in (('graphlaunch', 'cudaGraphLaunch'),
                       ('memcpyasync', 'cudaMemcpyAsync'),
                       ('eventsynchronize', 'cudaEventSynchronize')):
        if tag in name:
            return label
    return 'rest'


def totals(rows, classify):
    result = defaultdict(float)
    for row in rows:
        duration = time_ns(row, 'Total Time', 'Time')
        if duration is None:
            raise ValueError('summary row has no Total Time with supported units')
        result[classify(column(row, 'Name', 'Kernel Name') or '')] += duration
    return result


def kernel_totals(summary_rows, trace_rows):
    """Use trace durations to apportion summary names that merge grid shapes."""
    by_name = defaultdict(list)
    for row in trace_rows:
        by_name[column(row, 'Name', 'Kernel Name') or ''].append(row)
    families = defaultdict(float)
    split = defaultdict(lambda: [0.0, 0.0])
    fallback = False
    for row in summary_rows:
        name = column(row, 'Name', 'Kernel Name') or ''
        duration = time_ns(row, 'Total Time', 'Time')
        if duration is None:
            raise ValueError('summary row has no Total Time with supported units')
        count = float((column(row, 'Instances', 'Num Instances') or '0').replace(',', ''))
        y = grid(row, 'Y')
        matches = by_name[name] if y is None and bank_type(name) else []
        if matches and all(grid(r, 'Y') is not None for r in matches):
            weights = defaultdict(float)
            counts = defaultdict(int)
            for item in matches:
                key = (kernel_family(name, grid(item, 'Y')), grid(item, 'Y'))
                weights[key] += time_ns(item, 'Duration') or 0
                counts[key] += 1
            weight = sum(weights.values())
            for (family, dim), value in weights.items():
                share = value / weight if weight else counts[(family, dim)] / len(matches)
                families[family] += duration * share
                if family == BANK:
                    split[dim][0] += duration * share
                    split[dim][1] += counts[(family, dim)]
        else:
            family = kernel_family(name, y)
            families[family] += duration
            if family == BANK:
                fallback |= y is None
                split[y][0] += duration
                split[y][1] += count
    if fallback:
        print('  Caveat: grid Y unavailable in kern_sum and matching trace; type-only bank classification; early/late unknown.')
    return families, split


def gap_attribution(rows):
    """Classify >0.5 ms kernel-union gaps by nearest surrounding bank kernels."""
    devices = defaultdict(list)
    for row in rows:
        name = column(row, 'Name', 'Kernel Name') or ''
        if name.lstrip().lower().startswith(('[cuda', '[memcpy', '[memset', 'memcpy', 'memset')):
            continue
        start = time_ns(row, 'Start', 'Start Time')
        duration = time_ns(row, 'Duration')
        end = start + duration if duration is not None else time_ns(row, 'End', 'End Time')
        devices[column(row, 'Device', 'Device ID', 'GPU ID') or 'unknown'].append((start, end, row))
    result = defaultdict(lambda: [0, 0.0])
    for intervals in devices.values():
        intervals.sort(key=lambda item: (item[0], item[1]))
        following = [None] * len(intervals)
        bank = None
        for i in range(len(intervals) - 1, -1, -1):
            row = intervals[i][2]
            if kernel_family(column(row, 'Name', 'Kernel Name') or '', grid(row, 'Y')) == BANK:
                bank = row
            following[i] = bank
        previous = None
        frontier = intervals[0][0]
        for i, (start, end, row) in enumerate(intervals):
            if start - frontier > 500000:
                after = following[i]
                py = grid(previous, 'Y') if previous else None
                ay = grid(after, 'Y') if after else None
                if ay == 9:
                    label = 'pre-first-bank-kernel'
                    subset = 'pre-first subset: no preceding bank' if previous is None else 'pre-first subset: preceding bank present (includes token boundaries)'
                    result[subset][0] += 1
                    result[subset][1] += start - frontier
                elif py == 9 and ay == 8:
                    label = 'early-to-late: real-lookup wait (6144-wide early down)' if grid(previous, 'X') == 6144 else 'early-to-late: other early down'
                else:
                    label = 'unattributed'
                result[label][0] += 1
                result[label][1] += start - frontier
            if kernel_family(column(row, 'Name', 'Kernel Name') or '', grid(row, 'Y')) == BANK:
                previous = row
            frontier = max(frontier, end)
    return result


def trace_metrics(rows):
    """Union kernels on each device, assign each gap to the following layer.

    A cuda_gpu_trace normally has no layer column. Never infer layer identity
    from repeated kernel names. Device boundaries are never joined by a gap.
    """
    devices = defaultdict(list)
    for row in rows:
        name = column(row, 'Name', 'Kernel Name') or ''
        if name.lstrip().lower().startswith(('[cuda', '[memcpy', '[memset', 'memcpy', 'memset')):
            continue
        start = time_ns(row, 'Start', 'Start Time')
        duration = time_ns(row, 'Duration')
        end = time_ns(row, 'End', 'End Time')
        if start is None or (duration is None and end is None):
            raise ValueError('trace row needs Start and Duration (or End), with units')
        end = start + duration if duration is not None else end
        if end < start:
            raise ValueError('negative kernel duration')
        device = column(row, 'Device', 'Device ID', 'GPU ID') or 'unknown'
        layer = column(row, 'Layer', 'Layer ID') or 'unassigned'
        devices[device].append((start, end, layer))
    buckets = defaultdict(lambda: [[0, 0.0] for _ in BUCKETS])
    busy = span = long_gaps = 0.0
    for device, intervals in devices.items():
        intervals.sort()
        frontier = intervals[0][0]
        span += max(end for _, end, _ in intervals) - frontier
        for start, end, layer in intervals:
            if start > frontier:
                gap = start - frontier
                if gap > 500000:
                    long_gaps += gap
                index = 0 if gap < 50000 else 1 if gap < 500000 else 2 if gap <= 5000000 else 3
                bucket = buckets[(device, layer)][index]
                bucket[0] += 1
                bucket[1] += gap
            busy += max(0, end - max(start, frontier))
            frontier = max(frontier, end)
    return busy, span, buckets, long_gaps


def analyze(directory, tokens, prefix=None, decode_passes=None):
    decode_passes = tokens if decode_passes is None else decode_passes
    reports = load_reports(directory, prefix)
    result = dict(tokens=tokens, busy=None, gaps=None, waits=None, api=None)
    print('\nTrace: ' + directory + ((' prefix=' + prefix) if prefix else ''))
    print('tokens={} ({}); all captured activity divided by decode tokens'.format(
        tokens, 'default: 128 x 3 requests' if tokens == 384 else 'explicit --tokens'))
    print('decode-passes={}; steady estimate = captured per-token value x tokens/passes ({:.6f}); prompt work is not subtracted'.format(decode_passes, tokens / decode_passes))
    print('Missing reports: ' + (', '.join(r for r in REPORTS if r not in reports) or 'none'))
    trace = trace_metrics(reports['cuda_gpu_trace']) if 'cuda_gpu_trace' in reports else None
    if 'cuda_gpu_kern_sum' in reports:
        families, split = kernel_totals(reports['cuda_gpu_kern_sum'], reports.get('cuda_gpu_trace', []))
        total = sum(families.values())
        result['busy'] = total / tokens / 1e6
        wall = trace[1] if trace else None
        print('GPU families: ms/token | % GPU wall (first-to-last kernel per device) | % kernel time')
        for family in FAMILIES:
            value = families[family]
            share = '{:.2f}%'.format(100 * value / wall) if wall else 'n/a'
            print('  {}: {:.6f} | {} | {:.2f}%'.format(
                family, value / tokens / 1e6, share, 100 * value / total if total else 0))
        print('Bank split: captured ms/token | steady ms/token | launches | captured launches/token | steady launches/token')
        for y, label in ((9, 'early (Y=9)'), (8, 'late (Y=8)'), (None, 'unknown Y')):
            value, count = split[y]
            if y is not None or count or value:
                print('  {}: {:.6f} | {:.6f} | {:g} | {:.6f} | {:.6f}'.format(
                    label, value / tokens / 1e6, value / decode_passes / 1e6,
                    count, count / tokens, count / decode_passes))
        bank_count = sum(item[1] for item in split.values())
        print('  bank total launches: {:g} | captured/token {:.6f} | steady/token {:.6f}'.format(
            bank_count, bank_count / tokens, bank_count / decode_passes))
        print('Steady GPU families: ms/token (total / decode-passes)')
        for family in FAMILIES:
            print('  {}: {:.6f}'.format(family, families[family] / decode_passes / 1e6))
        print('  Shared experts require identifiable names; generic GEMM kernels remain other.')
    if trace:
        busy, span, buckets, long_gaps = trace
        result['busy'] = busy / tokens / 1e6
        result['gaps'] = sum(b[1] for group in buckets.values() for b in group) / tokens / 1e6
        result['waits'] = long_gaps / tokens / 1e6
        print('Gaps: device / following layer / bucket: count | count/token | ms/token')
        if not buckets:
            print('  none')
        for (device, layer), group in sorted(buckets.items()):
            for label, (count, duration) in zip(BUCKETS, group):
                print('  {} / {} / {}: {} | {:.6f} | {:.6f}'.format(
                    device, layer, label, count, count / tokens, duration / tokens / 1e6))
        print('Gap attribution (>0.5ms; nearest bank kernels, per device): count | captured ms/token | steady ms/token')
        attribution = gap_attribution(reports['cuda_gpu_trace'])
        for label in ('pre-first-bank-kernel', 'pre-first subset: no preceding bank',
                      'pre-first subset: preceding bank present (includes token boundaries)',
                      'early-to-late: real-lookup wait (6144-wide early down)',
                      'early-to-late: other early down', 'unattributed'):
            count, duration = attribution[label]
            print('  {}: {} | {:.6f} | {:.6f}'.format(label, count, duration / tokens / 1e6, duration / decode_passes / 1e6))
        early = [v for k, v in attribution.items() if k.startswith('early-to-late:')]
        print('  early-to-late total: {} | {:.6f} | {:.6f}'.format(
            sum(v[0] for v in early), sum(v[1] for v in early) / tokens / 1e6,
            sum(v[1] for v in early) / decode_passes / 1e6))
        print('  Attribution describes temporal placement, not proven CPU/I/O causality; missing grids stay unattributed.')
        print('  Layer labels absent => unassigned. Overlapping kernels merged per device.')
        print('  >0.5ms host-hook wait candidate sanity: {:.3f} ms/token; expected ~42; delta {:+.3f}.'.format(
            result['waits'], result['waits'] - 42))
        print('  Long gaps alone do not identify host-hook causes; copies may occupy kernel gaps.')
        print('  GPU busy uses interval union; family times sum kernel durations (can overlap).')
    if 'cuda_api_sum' in reports:
        apis = totals(reports['cuda_api_sum'], api_family)
        result['api'] = sum(apis.values()) / tokens / 1e6
        print('CUDA API: ms/token (host durations; may overlap GPU work)')
        for family in ('cudaStreamWaitValue / cuStreamWaitValue32', 'cudaGraphLaunch',
                       'cudaMemcpyAsync', 'cudaEventSynchronize', 'rest'):
            print('  {}: {:.6f}'.format(family, apis[family] / tokens / 1e6))
    print(summary(result))
    steady = {key: (value * tokens / decode_passes if key in ('busy', 'gaps', 'waits', 'api') and value is not None else value) for key, value in result.items()}
    print('Steady estimate: ' + summary(steady))
    return result


def summary(result):
    def value(key):
        return 'n/a' if result[key] is None else '{:.6f}'.format(result[key])
    return ('tokens={} gpu_busy={} ms/token gaps={} ms/token '
            '(of which >0.5ms: {}) api={} ms/token').format(
                result['tokens'], value('busy'), value('gaps'), value('waits'), value('api'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('directory2', nargs='?')
    parser.add_argument('--tokens', type=int, default=384, help='decode tokens per trace (default 384; win2m: 64)')
    parser.add_argument('--tokens2', type=int, help='second trace tokens; defaults to --tokens')
    parser.add_argument('--decode-passes', type=int, help='decode passes in first trace; defaults to tokens')
    parser.add_argument('--decode-passes2', type=int, help='decode passes in second trace; defaults to tokens2')
    parser.add_argument('--prefix', help='nsys export basename, e.g. glm-short')
    parser.add_argument('--prefix2', help='second nsys export basename, e.g. glm-long')
    args = parser.parse_args()
    if any(value is not None and value <= 0 for value in (args.tokens, args.tokens2, args.decode_passes, args.decode_passes2)):
        parser.error('token counts must be positive')
    try:
        first = analyze(args.directory, args.tokens, args.prefix, args.decode_passes)
        if args.directory2:
            second = analyze(args.directory2, args.tokens2 or args.tokens, args.prefix2, args.decode_passes2)
            delta = {key: (second[key] - first[key] if first[key] is not None and second[key] is not None else None)
                     for key in ('busy', 'gaps', 'waits', 'api')}
            delta['tokens'] = second['tokens'] - first['tokens']
            print('Delta (second - first), per-token metrics: ' + summary(delta))
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
