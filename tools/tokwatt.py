#!/usr/bin/env python3
"""Cabinet power attribution for epoch-timestamped decode windows (stdlib only)."""
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

METRIC = 'homeassistant_sensor_power_w{entity="sensor.server_cabinet_power"}'
STEP = 10
BASELINE = 120
# Instant selectors otherwise repeat samples for Prometheus's lookback period.
QUERY = f'{METRIC} and (time() - timestamp({METRIC}) <= 20)'
MARKER = re.compile(r'^LEG (\S+) start=(\S+) end=(\S+) tokens=(\S+)\s*$')


@dataclass
class Window:
    name: str
    start: float
    end: float
    tokens: int

    def __post_init__(self):
        if not (math.isfinite(self.start) and math.isfinite(self.end)
                and self.end > self.start and self.tokens >= 0):
            raise ValueError('windows require finite start < end and nonnegative tokens')


def parse_legs(lines):
    windows = []
    for number, line in enumerate(lines, 1):
        if not line.startswith('LEG '):
            continue
        match = MARKER.fullmatch(line.strip())
        if not match:
            raise ValueError(f'malformed LEG marker on line {number}')
        name, start, end, tokens = match.groups()
        try:
            windows.append(Window(name, float(start), float(end), int(tokens)))
        except ValueError as exc:
            raise ValueError(f'invalid LEG marker on line {number}: {exc}') from exc
    if not windows:
        raise ValueError('no timestamped LEG markers found; relative server logs are insufficient')
    return windows


def fetch_samples(url, window):
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.query or parts.fragment:
        raise ValueError('Prometheus URL must be an http(s) base URL without query or fragment')
    params = urlencode(dict(query=QUERY, start=window.start - BASELINE,
                            end=window.end, step=STEP))
    request_url = url.rstrip('/') + '/api/v1/query_range?' + params
    with urlopen(request_url, timeout=30) as response:
        payload = json.load(response)
    if payload.get('status') != 'success':
        raise ValueError(f"Prometheus query failed: {payload.get('error', 'unknown error')}")
    for warning in payload.get('warnings', []):
        print(f'Prometheus warning: {warning}', file=sys.stderr)
    data = payload.get('data', {})
    if data.get('resultType') != 'matrix':
        raise ValueError('expected a Prometheus matrix response')
    results = data.get('result', [])
    if len(results) > 1:
        raise ValueError('power selector matched multiple series; refusing to sum duplicate cabinet meters')
    return results[0].get('values', []) if results else []


def analyze(window, samples):
    """Equal-weight finite, nonnegative samples in half-open intervals; no filling."""
    clean = {}
    for timestamp, value in samples:
        timestamp, value = float(timestamp), float(value)
        if math.isfinite(timestamp) and math.isfinite(value) and value >= 0:
            clean[timestamp] = value
    before = [v for t, v in clean.items() if window.start - BASELINE <= t < window.start]
    during = [v for t, v in clean.items() if window.start <= t < window.end]
    watts = statistics.mean(during) if during else None
    baseline = statistics.median(before) if before else None
    delta = watts - baseline if watts is not None and baseline is not None else None
    seconds = window.end - window.start
    rate = window.tokens / seconds
    efficiency = rate / delta if delta is not None and delta > 0 else None
    cabinet = rate * 3_600_000 / watts if watts is not None and watts > 0 else None
    expected = math.ceil(seconds / STEP)
    flags = []
    if len(before) < BASELINE // STEP:
        flags.append('missing-baseline')
    if len(during) < expected:
        flags.append('missing-decode')
    if delta is not None and delta <= 0:
        flags.append('nonpositive-delta')
    if seconds < STEP:
        flags.append('sub-cadence')
    return dict(watts=watts, baseline=baseline, delta=delta, rate=rate,
                efficiency=efficiency, cabinet=cabinet,
                coverage=f'{len(during)}/{expected},{len(before)}/12',
                status=','.join(flags) or 'ok')


def report(rows):
    print('window  seconds  tokens  tok/s  cabinet_W  baseline_W  incremental_W  tok/J_inc (=tok/s/W)  cabinet_tok/kWh  samples(decode,base)  status')
    for window, result in rows:
        def fmt(key):
            value = result[key]
            return 'NA' if value is None else f'{value:.4f}'
        values = '  '.join(fmt(key) for key in
                           ('rate', 'watts', 'baseline', 'delta', 'efficiency', 'cabinet'))
        print(f'{window.name}  {window.end-window.start:.3f}  {window.tokens}  '
              f'{values}  {result["coverage"]}  {result["status"]}')


def demo():
    window = Window('synthetic', 1_700_000_000, 1_700_000_060, 300)
    samples = [(window.start + offset, 468 if offset < 0 else 538)
               for offset in range(-120, 60, 10)]
    result = analyze(window, samples)
    assert result['baseline'] == 468 and result['delta'] == 70
    assert result['rate'] == 5 and math.isclose(result['efficiency'], 1 / 14)
    assert math.isclose(result['cabinet'], 300 * 3_600_000 / (538 * 60))
    sparse = analyze(window, samples[:-1] + [(window.end - 10, 'NaN')])
    assert 'missing-decode' in sparse['status'] and sparse['delta'] == 70
    assert analyze(window, [])['efficiency'] is None
    assert analyze(window, [(window.start, 400)])['baseline'] is None
    assert analyze(window, [(window.start - 10, 538), (window.start, 468)])['efficiency'] is None
    assert analyze(window, [(window.start - 10, 468), (window.start, 468)])['efficiency'] is None
    assert parse_legs([f'LEG test start={window.start} end={window.end} tokens=300'])[0].tokens == 300
    report([(window, result), (Window('missing-sample', window.start, window.end, 300), sparse)])
    print('DEMO PASS: known energy, missing/NaN samples, absent baseline, zero/negative delta, LEG parsing')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prom-url', default=os.environ.get('PROM_URL'), help='defaults to PROM_URL')
    parser.add_argument('--legs-log', type=Path, help='log containing LEG name start=epoch end=epoch tokens=N')
    parser.add_argument('--start', type=float, help='decode start, Unix epoch seconds')
    parser.add_argument('--end', type=float, help='decode end, Unix epoch seconds (exclusive)')
    parser.add_argument('--tokens', type=int, help='actual decoded tokens')
    parser.add_argument('--name', default='decode')
    parser.add_argument('--demo', action='store_true', help='synthetic self-test; no network')
    args = parser.parse_args(argv)
    explicit = any(value is not None for value in (args.start, args.end, args.tokens))
    if args.demo:
        if args.legs_log or explicit:
            parser.error('--demo cannot be combined with window inputs')
        demo()
        return 0
    if args.legs_log and explicit:
        parser.error('choose --legs-log or --start/--end/--tokens')
    if not args.legs_log and not all(value is not None for value in (args.start, args.end, args.tokens)):
        parser.error('supply --legs-log or all of --start, --end, --tokens')
    if not args.prom_url:
        parser.error('supply --prom-url or set PROM_URL')
    try:
        if args.legs_log:
            windows = parse_legs(args.legs_log.read_text().splitlines())
        else:
            windows = [Window(args.name, args.start, args.end, args.tokens)]
        ordered = sorted(windows, key=lambda w: w.start)
        for left, right in zip(ordered, ordered[1:]):
            if left.end > right.start:
                raise ValueError('decode windows overlap; attribution would double-count energy')
        rows = [(window, analyze(window, fetch_samples(args.prom_url, window))) for window in windows]
        report(rows)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'tokwatt: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
