#!/usr/bin/env python3
"""Compare production serveidx wire bytes with every glmdump dn row (stdlib only).

W=~/glm53-work-exact python3 tools/glm53/cuda_serve_check.py
Use --no-server to build/verify the v1 store and parse dumps without Docker/SYCL.
Outputs: $W/tmp/cuda_serve_check/{experts.k3xs*,dumps/,actual/,report.json,run.log}.
No float conversion is used in the byte comparison or NaN payload checks.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time

D, F, NE, ALIGN = 6144, 2048, 8, 4096
SPECS = {3: ((17, 17, 18), 12091392),
         8: ((18, 18, 23), 16318464),
         75: ((17, 17, 23), 13959168)}
BLOCK = {17: 74, 18: 98, 23: 136}
HEADER = struct.Struct('<4sIIIIIQQ24x')
ENTRY = struct.Struct('<IIIIQQQQQ8x')
ROW = D * 4


def require(ok, message):
    if not ok:
        raise ValueError(message)


def nan_info(row):
    words = [w[0] for w in struct.iter_unpack('<I', row)]
    nans = [w for w in words if (w & 0x7fffffff) > 0x7f800000]
    return len(nans), len(nans) == D, sorted(set(nans))


def parse_dump(path):
    x, ids, dn = [(path / (s + '.bin')).read_bytes() for s in ('x', 'ids', 'dn')]
    require(len(x) > 0 and len(x) % ROW == 0, f'{path}: invalid x length')
    cases = len(x) // ROW
    require(len(ids) == cases * NE * 4, f'{path}: invalid ids length')
    require(len(dn) == cases * NE * ROW, f'{path}: invalid dn length')
    labels = {}
    for line in (path / 'kind.txt').read_text().splitlines():
        number, label = line.split(maxsplit=1)
        number = int(number)
        require(number not in labels, f'{path}: duplicate kind {number}')
        labels[number] = label
    require(set(labels) == set(range(cases)), f'{path}: kind labels do not cover cases')
    result = []
    for i in range(cases):
        es = struct.unpack_from('<8i', ids, i * NE * 4)
        require(all(0 <= e < NE for e in es), f'{path}: case {i}: id outside record slots 0..7')
        result.append((x[i * ROW:(i + 1) * ROW], es,
                       dn[i * NE * ROW:(i + 1) * NE * ROW], labels[i]))
    return result


def build_store(records, out):
    layers, offset = [], ALIGN * 2
    for layer, (types, stride) in SPECS.items():
        src = records / f'L{layer}.records'
        sizes = [D * F // 256 * BLOCK[t] for t in types]
        require(sum(sizes) == stride, f'L{layer}: record layout disagrees with stride')
        require(src.stat().st_size == NE * stride, f'{src}: expected {NE * stride} bytes')
        record_size = (stride + ALIGN - 1) // ALIGN * ALIGN
        layers.append(dict(layer=layer, types=types, sizes=sizes,
                           record_size=record_size, base_offset=offset))
        offset += NE * record_size
    store = out / 'experts.k3xs'
    with store.open('wb') as dst:
        dst.truncate(offset)
        dst.write(HEADER.pack(b'K3XS', 1, len(layers), NE, ALIGN, ENTRY.size, ALIGN, ALIGN * 2))
        for i, layer in enumerate(layers):
            dst.seek(ALIGN + i * ENTRY.size)
            dst.write(ENTRY.pack(layer['layer'], *layer['types'], *layer['sizes'],
                                 layer['record_size'], layer['base_offset']))
            stride = sum(layer['sizes'])
            with (records / f"L{layer['layer']}.records").open('rb') as src:
                for e in range(NE):
                    dst.seek(layer['base_offset'] + e * layer['record_size'])
                    dst.write(src.read(stride))
    # Verify stored records, including alignment padding, against the original slots.
    with store.open('rb') as dst:
        for layer in layers:
            stride = sum(layer['sizes'])
            with (records / f"L{layer['layer']}.records").open('rb') as src:
                for e in range(NE):
                    dst.seek(layer['base_offset'] + e * layer['record_size'])
                    require(dst.read(stride) == src.read(stride), 'store record copy differs')
                    require(dst.read(layer['record_size'] - stride) == bytes(layer['record_size'] - stride),
                            'nonzero store padding')
    sidecar = []
    for layer in layers:
        item = {k: layer[k] for k in ('layer', 'record_size', 'base_offset')}
        item['n_expert'] = NE
        for i, name in enumerate(('gate', 'up', 'down')):
            dims = [F, D, NE] if name == 'down' else [D, F, NE]
            item[name] = dict(dims=dims, type=layer['types'][i], offset=0,
                              n_bytes=layer['sizes'][i] * NE, shard='')
            item[name + '_size'] = layer['sizes'][i]
        sidecar.append(item)
    Path(str(store) + '.json').write_text(json.dumps(dict(n_expert=NE, records_offset=ALIGN * 2,
                                                        total_bytes=offset, layers=sidecar), indent=2) + '\n')
    Path(str(store) + '.set').write_text(''.join(f'{layer} {e}\n' for layer in SPECS for e in range(NE)))
    print(f'store v1: 3 layers, 24 experts, {offset} bytes; copied records verified', flush=True)


def recv_all(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError(f'server closed after {len(data)}/{size} bytes')
        data.extend(chunk)
    return bytes(data)


def inner(args):
    out = Path('/w/tmp/cuda_serve_check')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    env = dict(os.environ, G53X_EXACT='1', G53X_MUFU='/w/mufu',
               K3X_SETOUT=str(out / 'served.set'))
    store = str(out / 'experts.k3xs')
    report, failed = [], False
    (out / 'actual').mkdir(exist_ok=True)
    with (out / 'server.log').open('w') as log:
        srv = subprocess.Popen(['/w/g53x', store + '.set', store, '-', '-', 'serveidx', str(port)],
                               stdout=log, stderr=log, env=env)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while 'serving on' not in (out / 'server.log').read_text():
            if srv.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('server failed to start:\n' + (out / 'server.log').read_text())
            time.sleep(0.2)
        with socket.create_connection(('127.0.0.1', port), timeout=args.request_timeout) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for layer in SPECS:
                cases = parse_dump(out / 'dumps' / f'L{layer}')
                differences, nan_rows, all_nan_rows, nan_bad = [], 0, 0, 0
                total = 0
                with (out / 'actual' / f'L{layer}.dn.bin').open('wb') as actual:
                    for number, (x, es, want, label) in enumerate(cases):
                        hdr = [NE] + [layer * NE + e for e in es] + [0] * 8
                        hdr[16] = number
                        sock.sendall(struct.pack('<17i', *hdr) + x)
                        got = recv_all(sock, len(want))
                        actual.write(got)
                        diff = sum(a != b for a, b in zip(got, want))
                        total += diff
                        reasons = []
                        for row in range(NE):
                            w, g = want[row * ROW:(row + 1) * ROW], got[row * ROW:(row + 1) * ROW]
                            count, all_nan, payloads = nan_info(w)
                            nan_rows += bool(count)
                            all_nan_rows += all_nan
                            if count and w != g:
                                nan_bad += 1
                                reasons.append(f'row {row} expert {es[row]} CUDA NaN row bytes changed; '
                                               f'CUDA NaN words={[hex(p) for p in payloads]}, '
                                               f'wire NaN words={[hex(p) for p in nan_info(g)[2]]}')
                            if w != g and all(v == 0 for v in struct.unpack('<6144f', w)) and nan_info(g)[1]:
                                reasons.append(f'row {row} expert {es[row]}: zero CUDA row replaced by all-NaN missing sentinel')
                        if diff:
                            differences.append(dict(case=number, label=label, bytes_differing=diff, details=reasons))
                            print(f'  DIFF L{layer} case {number} ({label}): {diff} bytes', flush=True)
                            for reason in reasons:
                                print('    ' + reason, flush=True)
                summary = dict(layer=layer, cases=len(cases), rows=len(cases) * NE, bytes_differing=total,
                               cuda_nan_rows=nan_rows, cuda_all_nan_rows=all_nan_rows,
                               changed_nan_rows=nan_bad, differences=differences)
                report.append(summary)
                failed |= bool(total)
                print(f'L{layer}: cases={len(cases)} rows={len(cases) * NE} bytes differing={total}; '
                      f'CUDA NaN rows={nan_rows}, all-NaN={all_nan_rows}, changed={nan_bad}', flush=True)
                if all_nan_rows:
                    print(f'  L{layer}: engine treats these {all_nan_rows} real all-NaN rows as missing '
                          'and falls back to its local copy, even if wire bytes match.', flush=True)
        served = sorted(tuple(map(int, line.split())) for line in (out / 'served.set').read_text().splitlines())
        require(served == [(layer, e) for layer in SPECS for e in range(NE)], 'served set differs from all 24 experts')
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('FAIL: serveidx wire bytes differ from CUDA' if failed else
              'PASS: every serveidx wire row is byte-identical to CUDA (including NaN payloads)', flush=True)
        return int(failed)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait()


def host(args):
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent.parent
    work = Path(os.environ.get('W', '~/glm53-work')).expanduser().resolve()
    out = work / 'tmp' / 'cuda_serve_check'
    out.mkdir(parents=True, exist_ok=True)
    build_store(args.records or script_dir / 'records', out)
    for layer in SPECS:
        source = (args.dumps or root / 'evidence' / 'cuda-dumps') / f'L{layer}'
        cases = parse_dump(source)
        dest = out / 'dumps' / f'L{layer}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('x.bin', 'ids.bin', 'dn.bin', 'kind.txt'):
            if (source / name).resolve() != (dest / name).resolve():
                shutil.copyfile(source / name, dest / name)
        nan_rows = sum(bool(nan_info(w[i * ROW:(i + 1) * ROW])[0])
                       for _, _, w, _ in cases for i in range(NE))
        print(f'L{layer}: parsed cases={len(cases)} rows={len(cases) * NE} CUDA NaN rows={nan_rows}', flush=True)
    if args.no_server:
        print('DRY RUN PASS: store records verified and all dumps parsed; serveidx comparison NOT RUN', flush=True)
        return 0
    cmd = ('python3 /w/tools/cuda_serve_check.py --inner '
           f'--startup-timeout {args.startup_timeout} --request-timeout {args.request_timeout}')
    env = dict(os.environ, W=str(work), G53X_EXACT='1', G53X_MUFU='/w/mufu')
    with (out / 'run.log').open('w') as log:
        with subprocess.Popen([str(script_dir / 'dsycl.sh'), cmd], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-server', action='store_true')
    parser.add_argument('--records', type=Path, help='record directory (default: alongside this script)')
    parser.add_argument('--dumps', type=Path, help='CUDA dump root (default: evidence/cuda-dumps)')
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--request-timeout', type=float, default=300)
    parser.add_argument('--inner', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.startup_timeout > 0 and args.request_timeout > 0, 'timeouts must be positive')
    return inner(args) if args.inner else host(args)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
