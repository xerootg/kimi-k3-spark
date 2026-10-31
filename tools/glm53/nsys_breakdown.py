#!/usr/bin/env python3
"""Summarize nsys CSVs using only the Python standard library.

Use directories containing report-name CSVs, or --prefix glm-short/glm-long
for win2m.sh's shared export directory. No pre/post-capture idle is counted.
"""

import argparse
import csv
from collections import defaultdict


FAMILIES = ('routed expert GEMV', 'shared expert', 'attention / flash-attn',
            'DSA indexer', 'norms', 'rope', 'cpy / get_rows / set_rows', 'other')
BUCKETS = ('<50 us', '50-500 us', '0.5-5 ms', '>5 ms')
REPORTS = ('cuda_gpu_kern_sum', 'cuda_gpu_trace', 'cuda_api_sum')


def normalized(value):
    return ' '.join(value.strip().lower().split())


def column(row, *names):
    for name in names:
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


def kernel_family(name):
    name = name.lower()
    if any(tag in name for tag in ('indexer', 'topk', 'top_k', 'dsa')):
        return 'DSA indexer'
    if any(tag in name for tag in ('shared', 'shexp', 'she_expert')):
        return 'shared expert'
    if 'mul_mat_vec_q' in name or 'mmvq' in name:
        return 'routed expert GEMV'
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


def analyze(directory, tokens, prefix=None):
    reports = load_reports(directory, prefix)
    result = dict(tokens=tokens, busy=None, gaps=None, waits=None, api=None)
    print('\nTrace: ' + directory + ((' prefix=' + prefix) if prefix else ''))
    print('tokens={} ({}); all captured activity divided by decode tokens'.format(
        tokens, 'default: 128 x 3 requests' if tokens == 384 else 'explicit --tokens'))
    print('Missing reports: ' + (', '.join(r for r in REPORTS if r not in reports) or 'none'))
    trace = trace_metrics(reports['cuda_gpu_trace']) if 'cuda_gpu_trace' in reports else None
    if 'cuda_gpu_kern_sum' in reports:
        families = totals(reports['cuda_gpu_kern_sum'], kernel_family)
        total = sum(families.values())
        result['busy'] = total / tokens / 1e6
        wall = trace[1] if trace else None
        print('GPU families: ms/token | % GPU wall (first-to-last kernel per device) | % kernel time')
        for family in FAMILIES:
            value = families[family]
            share = '{:.2f}%'.format(100 * value / wall) if wall else 'n/a'
            print('  {}: {:.6f} | {} | {:.2f}%'.format(
                family, value / tokens / 1e6, share, 100 * value / total if total else 0))
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
    parser.add_argument('--prefix', help='nsys export basename, e.g. glm-short')
    parser.add_argument('--prefix2', help='second nsys export basename, e.g. glm-long')
    args = parser.parse_args()
    if args.tokens <= 0 or (args.tokens2 is not None and args.tokens2 <= 0):
        parser.error('token counts must be positive')
    try:
        first = analyze(args.directory, args.tokens, args.prefix)
        if args.directory2:
            second = analyze(args.directory2, args.tokens2 or args.tokens, args.prefix2)
            delta = {key: (second[key] - first[key] if first[key] is not None and second[key] is not None else None)
                     for key in ('busy', 'gaps', 'waits', 'api')}
            delta['tokens'] = second['tokens'] - first['tokens']
            print('Delta (second - first), per-token metrics: ' + summary(delta))
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
