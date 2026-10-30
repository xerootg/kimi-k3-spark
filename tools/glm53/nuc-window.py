#!/usr/bin/env python3
"""Remote stdlib driver for nuc-window.sh; executed on cluster hosts only."""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess as sp
import time
import urllib.request

STORE = '/srv/glm-5.3/store/glm53-q2-v2.k3xs'
BUNDLE = '/home/xero/k3-bitexact/sycl/lib'
MUFU = '/home/xero/glm53-mufu'

def run(*cmd, **kw):
    return sp.run(cmd, check=True, **kw)

def pairs(path):
    return list(dict.fromkeys(tuple(map(int, s.split())) for s in Path(path).read_text().splitlines()
                              if s.strip() and not s.lstrip().startswith('#')))

def write_set(path, rows):
    path.write_text(''.join(f'{l} {e}\n' for l, e in rows))

def identity(pid):
    try:
        s = Path(f'/proc/{pid}/stat').read_text()
        return [s[s.rindex(')') + 2:].split()[19], os.readlink(f'/proc/{pid}/exe')]
    except (FileNotFoundError, ProcessLookupError):
        return None

def owned(w, name):
    p = w / (name + '.pid.json')
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d['pid'] if identity(d['pid']) == d['identity'] else None

def remember(w, name, pid):
    ident = identity(pid)
    if ident is None:
        raise RuntimeError('child exited before PID recording')
    (w / (name + '.pid.json')).write_text(json.dumps(dict(pid=pid, identity=ident)))

def terminate(w, name):
    pid = owned(w, name)
    if pid:
        os.kill(pid, signal.SIGINT if name == 'spark' else signal.SIGTERM)
        for _ in range(120):
            if owned(w, name) is None or Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0] == 'Z':
                break
            time.sleep(.5)
        else:
            raise RuntimeError(f'{name} still running; refusing SIGKILL')
    (w / (name + '.pid.json')).unlink(missing_ok=True)

def stage(a, w):
    if owned(w, 'nuc'):
        raise RuntimeError('stop this harness before staging')
    env = dict(os.environ, LD_LIBRARY_PATH=f'{BUNDLE}:{w}/lib')
    result = run('ldd', str(w / 'g53x'), env=env, capture_output=True, text=True)
    print(result.stdout, end='')
    if 'not found' in result.stdout:
        raise RuntimeError('unresolved runtime libraries; repair bundle/compiler image')
    # UR/Level Zero adapters are dlopened: ldd(g53x) alone cannot gate them.
    for directory in (Path(BUNDLE), w / 'lib'):
        for lib in sorted(directory.glob('*.so*')):
            if lib.name.startswith(('libsycl', 'libur_', 'libze_')):
                dep = run('ldd', str(lib), env=env, capture_output=True, text=True)
                if 'not found' in dep.stdout:
                    raise RuntimeError(f'unresolved adapter runtime {lib}: {dep.stdout}')
    for name in ('rcp', 'ex2p', 'ex2n'):
        assert Path(MUFU, name + '.bin').is_file(), f'missing MUFU {name}'
    # Metadata only; never copy the model over WAN. Resident bytes are g+u+d,
    # exactly the counter used by g53x's decimal K3X_MAX_GB cap.
    sizes = {}
    with open(STORE, 'rb') as f:
        h = f.read(64)
        assert h[:4] == b'K3XS' and struct.unpack_from('<I', h, 12)[0] == 256
        count = struct.unpack_from('<I', h, 8)[0]
        f.seek(struct.unpack_from('<Q', h, 24)[0])
        for _ in range(count):
            r = f.read(64)
            layer = struct.unpack_from('<I', r)[0]
            n = sum(struct.unpack_from('<3Q', r, 16))
            for e in range(256):
                if 3 <= layer < 78:
                    sizes[layer, e] = n
    placement = Path(a.placement)
    if placement.exists():
        candidates = pairs(placement)
        rule = f'placement order: {placement}'
    else:
        # No measured hotness available: explicit low-ID hotness proxy, spread
        # across layers, rather than claiming these are observed hot experts.
        candidates = sorted(sizes, key=lambda p: (p[1], p[0]))
        rule = 'synthetic hotness proxy: expert ID ascending, then layer ascending'
    selected, used = [], 0
    for p in candidates:
        if p not in sizes:
            raise RuntimeError(f'invalid placement pair {p}')
        if used + sizes[p] > 44_000_000_000:
            break
        selected.append(p)
        used += sizes[p]
    assert selected, 'empty resident selection'
    write_set(w / 'resident.set', selected)
    print(f'{rule}; N={len(selected)} resident bytes={used}; K3X_MAX_GB=44')

def start(a, w):
    if not owned(w, 'nuc'):
        # Run a detached Python launcher under the same render-group identity as
        # g53x. It records the exec'd child PID, not sudo/nohup's wrapper PID.
        (w / 'launch.failed').unlink(missing_ok=True)
        (w / 'server.log').write_text('')
        run('sudo', '-n', 'prlimit', '--memlock=unlimited:unlimited', '--',
            'setpriv', '--reuid=xero', '--regid=render', '--init-groups', 'sh', '-c',
            'nohup python3 "$1/driver.py" launch --work "$1" --port "$2" >"$1/launcher.log" 2>&1 </dev/null &',
            'nuc-window', str(w), str(a.port))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        log = (w / 'server.log').read_text(errors='replace') if (w / 'server.log').exists() else ''
        if owned(w, 'nuc') and f'serving on {a.port} (busy-poll)' in log:
            served = set(pairs(w / 'served.set'))
            resident = pairs(w / 'resident.set')
            assert set(resident) <= served, 'resident set rejected by server; do not probe cold rows'
            write_set(w / 'remote.set', resident)
            print(f'ready: K3X_SETOUT served={len(served)}; resident owners={len(resident)}')
            return
        if (w / 'launch.failed').exists():
            raise RuntimeError((w / 'launch.failed').read_text() + '\n' + log[-4000:])
        time.sleep(2)
    raise RuntimeError('readiness timeout; inspect server.log; use stop before retry')

def launch(a, w):
    # Separate lock keeps duplicate start invocations from launching two servers.
    with open(w / 'launch.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if owned(w, 'nuc'):
            return
        (w / 'launch.failed').unlink(missing_ok=True)
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
            env = dict(os.environ, LD_LIBRARY_PATH=f'{BUNDLE}:{w}/lib', G53X_EXACT='2', G53X_MUFU=MUFU,
                       K3X_MAX_GB='44', K3X_SETOUT=str(w / 'served.set'), K3X_DISK_TIER='1',
                       K3X_DISK_STORE=STORE, K3X_DISK_LRU_GB='1', K3X_DISK_DEADLINE_MS='20')
            with open(w / 'server.log', 'w') as log:
                p = sp.Popen(['taskset', '-c', '4-11', 'choom', '-n', '1000', '--', str(w / 'g53x'),
                              str(w / 'resident.set'), STORE, '6144', '-', 'serveidx', str(a.port)],
                             env=env, stdout=log, stderr=log)
                # Wait through exec wrappers before capturing executable identity.
                for _ in range(100):
                    ident = identity(p.pid)
                    if ident and ident[1] == str(w / 'g53x'):
                        break
                    assert p.poll() is None, 'launcher child exited'
                    time.sleep(.05)
                else:
                    raise RuntimeError('g53x did not exec')
                remember(w, 'nuc', p.pid)
                code = p.wait()
                (w / 'launch.failed').write_text(f'g53x exited {code}')
        except Exception as e:
            (w / 'launch.failed').write_text(str(e))
            raise

def probe(a, w):
    by_layer = collections.defaultdict(list)
    for l, e in pairs(w / 'remote.set'):
        by_layer[l].append(e)
    l = next(l for l, es in by_layer.items() if len(es) >= 2)
    gids = [l * 256 + e for e in by_layer[l][:2]]
    x = struct.pack('<6144f', *(0.05 * math.sin(i * .13) for i in range(6144)))
    def recv(s, n):
        out = bytearray()
        while len(out) < n:
            b = s.recv(n - len(out))
            if not b:
                raise RuntimeError('short reply')
            out.extend(b)
        return out
    with socket.create_connection((a.ip, a.port), timeout=30) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for k in (1, 2):
            times = []
            for i in range(210):
                header = struct.pack('<17i', k, *(gids[:k] + [0] * (16-k)))
                t = time.perf_counter_ns()
                s.sendall(header + x)
                reply = recv(s, k * 6144 * 4)
                elapsed = (time.perf_counter_ns() - t) / 1e6
                assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', reply)), 'NaN/refused resident row'
                if i >= 10:
                    times.append(elapsed)
            times.sort()
            print(f'k={k} n=200 p50={times[99]:.3f} ms p99={times[197]:.3f} ms gids={gids[:k]}')

def build(w):
    tree = Path('/home/xero/llama-k3-bs')
    source = tree / 'llama/src/llama-moe-cache.cpp'
    if not source.exists():
        source = tree / 'src/llama-moe-cache.cpp'
    text = source.read_text()
    if 'remote layer' not in text or 'rows used=' not in text:
        patch = w / '0013.patch'
        assert patch.exists(), '0013 missing: set REMOTE_PATCH on laptop'
        run('git', '-C', str(tree), 'apply', '--check', str(patch))
        run('git', '-C', str(tree), 'apply', str(patch))
    with open(w / 'build.log', 'w') as log:
        env = dict(os.environ, CUDACXX='/usr/local/cuda/bin/nvcc', PATH='/usr/local/cuda/bin:' + os.environ['PATH'])
        run('cmake', '--build', str(tree / 'build'), '-j', '12', '--target', 'llama-server', env=env, stdout=log, stderr=log)
    print('remote-tier source markers present; llama-server build succeeded')

def completion():
    data = json.dumps(dict(prompt='Write a Python quicksort function and explain it.', temperature=0,
                           seed=1, n_predict=128, cache_prompt=False)).encode()
    req = urllib.request.Request('http://127.0.0.1:18080/completion', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def ab(a, w):
    # No pkill: wait for production shutdown; reject all unrelated native servers.
    for _ in range(120):
        ps = sp.check_output(['ps', '-eo', 'comm='], text=True).splitlines()
        if not any(s.strip() == 'llama-server' for s in ps):
            break
        time.sleep(5)
    else:
        raise RuntimeError('llama-server still running; no processes were killed')
    out = w / time.strftime('ab-%Y%m%d-%H%M%S')
    out.mkdir()  # never overwrite an earlier run
    base = Path('/srv/glm-5.3')
    import shutil
    snapshot = out / 'base.k3sn'
    shutil.copyfile(base / 'snapshot/glm53-q2.k3sn', snapshot)
    fixture = None
    for name, on in [('off1', False), ('on1', True), ('off2', False), ('on2', True)]:
        import shutil
        snap = out / (name + '.k3sn')
        shutil.copyfile(snapshot, snap)
        env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_EXPERT_STORE_REMOTE')}
        env.update(LLAMA_LOG_COLORS='0', LLAMA_EXPERT_STORE=STORE, LLAMA_EXPERT_STORE_SNAPSHOT=str(snap),
                   LLAMA_EXPERT_STORE_THREADS='16', LLAMA_EXPERT_STORE_SELF_PREFETCH='1', LLAMA_EXPERT_STORE_SPLIT='1',
                   LLAMA_EXPERT_STORE_SECOND_CHANCE='3', LLAMA_EXPERT_STORE_RESERVE_MIB='4096',
                   LLAMA_EXPERT_STORE_BYTE_STRIDE='1', LLAMA_EXPERT_STORE_PREFETCH='1',
                   LLAMA_EXPERT_STORE_PREFETCH_INPUT='3', LLAMA_EXPERT_STORE_SPLIT_SKIP_ZERO='1')
        if on:
            env.update(LLAMA_EXPERT_STORE_REMOTE=f'{a.ip}:{a.port}', LLAMA_EXPERT_STORE_REMOTE_SET=str(w / 'remote.set'),
                       LLAMA_EXPERT_STORE_REMOTE_DEADLINE_US='2500', LLAMA_EXPERT_STORE_REMOTE_ROWS=str(a.rows),
                       LLAMA_EXPERT_STORE_REMOTE_DMA_LATENCY='0')
        cmd = ['/home/xero/llama-k3-bs/build/bin/llama-server', '-m', 'models/GLM-5.3-UD-Q2_K_XL-00001-of-00007.gguf',
               '--host', '127.0.0.1', '--port', '18080', '--ctx-size', '65536', '--parallel', '1',
               '--batch-size', '2048', '--ubatch-size', '512', '-ngl', '999', '--n-cpu-moe', '999',
               '--moe-cache-mib', '80000', '--cache-ram', '0', '--threads', '8', '--threads-batch', '20',
               '--flash-attn', 'on', '--no-warmup', '--fit', 'off', '--load-mode', 'mmap', '-lv', '4']
        with open(out / (name + '.log'), 'w') as log:
            p = sp.Popen(cmd, cwd=base, env=env, stdout=log, stderr=log)
            remember(w, 'spark', p.pid)
            try:
                deadline = time.monotonic() + 1800
                while True:
                    assert p.poll() is None, f'{name} server died'
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=2) as r:
                            if json.load(r).get('status') == 'ok':
                                break
                    except Exception:
                        pass
                    assert time.monotonic() < deadline, 'health timeout'
                    time.sleep(3)
                for i in range(3):
                    d = completion()
                    (out / f'{name}.r{i}.json').write_text(json.dumps(d))
                    content = d['content'].encode()
                    digest = hashlib.md5(content).hexdigest()
                    if i:
                        print(f'{name} r{i} {d["timings"]["predicted_per_second"]:.3f} tok/s md5 {digest}', flush=True)
                        assert d['timings']['predicted_n'] == 128, 'not 128 decoded tokens'
                        assert digest.startswith('9449ae5738'), 'fixture md5 gate FAILED'
                        if fixture is None:
                            fixture = content
                        assert content == fixture, 'A/B byte equality FAILED'
            finally:
                terminate(w, 'spark')
                p.wait(timeout=10)
        lines = [s for s in (out / (name + '.log')).read_text(errors='replace').splitlines() if 'remote layer' in s]
        print('\n'.join(lines) if lines else f'{name}: no remote layer stats', flush=True)
        if on:
            import re
            assert any(int(n) > 0 for s in lines for n in re.findall(r'rows used=(\d+)', s)), 'no remote rows used: fallback-only run'
        time.sleep(5)
    print(f'PASS off/on/off/on rows={a.rows}; artifacts {out}')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['stage', 'start', 'launch', 'stopped', 'stop', 'probe', 'build', 'ab'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--port', type=int, default=19600)
    ap.add_argument('--ip', default='192.168.4.249')
    ap.add_argument('--rows', type=int, choices=[1, 2], default=1)
    ap.add_argument('--placement', default='/srv/glm-5.3/placement/nuc2-ram.set')
    a = ap.parse_args()
    if a.command == 'ab':
        def interrupted(signum, frame):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, interrupted)
    w = Path(a.work).resolve()
    w.mkdir(parents=True, exist_ok=True)
    with open(w / ('launch-command.lock' if a.command == 'launch' else 'command.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.command == 'stopped':
            assert not owned(w, 'nuc'), 'stop this harness before staging'
        elif a.command == 'stop':
            terminate(w, 'nuc')
            print('owned NUC server stopped (or already absent)')
        elif a.command == 'build':
            build(w)
        else:
            globals()[a.command](a, w)

if __name__ == '__main__':
    main()
