#!/usr/bin/env python3
"""Bounded local GH200 window; no SSH, fetch, image update, or snapshot replacement."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads((Path(__file__).with_name('production.json')).read_text())
K = ['kubectl', '--request-timeout=30s', '-n', 'kimi-k3']
LABEL = 'app.kubernetes.io/name=kimi-k3-gh200'
DEPLOY = 'deployment/kimi-k3-gh200'
SERVER = None
HOG = None
DEADLINE = None


def run(cmd, timeout=60, **kw):
    if DEADLINE is not None:
        timeout = min(timeout, max(1, DEADLINE - time.monotonic()))
    capture = kw.pop('capture_output', False)
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc = subprocess.Popen(cmd, start_new_session=True, **kw)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def request(path, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:18080/' + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def gold(body, out, name):
    d = request('completion', body)
    (out / (name + '.json')).write_text(json.dumps(d))
    md5 = hashlib.md5(d['content'].encode()).hexdigest()
    print(name, d['timings']['predicted_per_second'], 'tok/s md5', md5, flush=True)
    if not md5.startswith('ac78af0e5d'):
        raise RuntimeError('gold mismatch: rollback; do not interpret throughput')


def preflight():
    if os.uname().machine != 'aarch64' or os.uname().nodename.split('.')[0] != 'gh200-dev':
        raise RuntimeError('must run natively on node gh200-dev (aarch64)')
    for exe in ['kubectl', 'git', 'tar', 'patch', 'cmake', 'nvcc', 'numactl']:
        if not shutil.which(exe):
            raise RuntimeError('missing offline prerequisite: ' + exe)
    base = Path(os.environ['NATIVE_BASE']).expanduser().resolve(strict=True)
    fixture = Path(os.environ['K3_REQUEST_JSON']).expanduser().resolve(strict=True)
    body = json.loads(fixture.read_text())
    if not (body.get('n_predict') == 800 and body.get('temperature') == 0
            and body.get('cache_prompt') is False and isinstance(body.get('prompt'), str)
            and not body.get('stream', False)):
        raise RuntimeError('need exact nonstreaming raw K3 800-token greedy gold JSON')
    sha = next(s.split('=', 1)[1] for s in (ROOT/'versions.env').read_text().splitlines()
               if s.startswith('LLAMA_SHA='))
    run(['git', '-C', str(base), 'cat-file', '-e', sha + '^{tree}'])
    for path in [PROFILE['model'], PROFILE['store'], PROFILE['snapshot']]:
        if not Path(path).is_file():
            raise RuntimeError('missing documented host artifact: ' + path)
    live = json.loads(run(K + ['get', DEPLOY, '-o', 'json'], capture_output=True).stdout)
    c = live['spec']['template']['spec']['containers'][0]
    if c['image'] != PROFILE['image'] or c['args'] != PROFILE['args']:
        raise RuntimeError('live image/args differ from reviewed evidence')
    env = {x['name']: x.get('value') for x in c.get('env', []) if 'value' in x}
    if env != PROFILE['env'] or live['spec'].get('replicas') != 1:
        raise RuntimeError('live env/replica count differs from reviewed production')
    if live['spec']['template']['spec']['nodeSelector'].get('kubernetes.io/hostname') != 'gh200-dev':
        raise RuntimeError('unexpected node selector')
    return base, sha, body, fixture, live


def leg(name, stride, binary, out, body):
    global SERVER, HOG
    snap = out / (name + '.k3sn')
    shutil.copyfile(out/'frozen.k3sn', snap)
    assert hashlib.sha256(snap.read_bytes()).digest() == hashlib.sha256((out/'frozen.k3sn').read_bytes()).digest()
    # Isolate experiment from inherited LLAMA/GGML knobs and production API secret.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_'))}
    env.update(PROFILE['env'])
    env.update(LLAMA_EXPERT_STORE=PROFILE['store'], LLAMA_EXPERT_STORE_SNAPSHOT=str(snap),
               LLAMA_EXPERT_STORE_BYTE_STRIDE=str(stride))
    args = PROFILE['args'][:]
    for key, value in [('-m', PROFILE['model']), ('--host', '127.0.0.1'), ('--port', '18080')]:
        args[args.index(key) + 1] = value
    logpath = out / (name + '.log')
    with logpath.open('w') as log:
        SERVER = subprocess.Popen([str(binary)] + args, env=env, stdout=log, stderr=log)
        try:
            ready = time.monotonic() + 300
            while True:
                if SERVER.poll() is not None or time.monotonic() > ready:
                    raise RuntimeError(name + ': died or exceeded 5-minute load budget')
                try:
                    if request('health', timeout=2).get('status') == 'ok':
                        break
                except Exception:
                    time.sleep(2)
            text = logpath.read_text(errors='replace')
            expected = [6451200, 7827456, 7139328] if stride else [6451200, 10137600]
            if any(('stride %d B (%s)' % (n, 'bytes' if stride else 'block padded')) not in text for n in expected):
                raise RuntimeError(name + ': expected stride modes absent; refusing silent fallback')
            if 'cache snapshot restored:' not in text:
                raise RuntimeError(name + ': snapshot restore not confirmed')
            for i in range(1, 4):
                gold(body, out, name + '.r' + str(i))
            if name == 'ST1':
                # Bounded concurrent prefill shape gate; not a prefill output hash gate.
                prefill = dict(prompt='A simple sentence about a tree. ' * 2048,
                               n_predict=64, temperature=0, cache_prompt=False)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    replies = list(pool.map(lambda _: request('completion', prefill, 480), range(2)))
                (out/'concurrent-prefill.json').write_text(json.dumps(replies))
                gold(body, out, name + '.after-concurrent')
                # Reproduce documented host/HBM valve pressure sequentially; no sysctl changes.
                for node, gib in [(0, 12), (1, 14)]:
                    code = ('import mmap,time; m=mmap.mmap(-1,%d); '
                            '[m.__setitem__(i,1) for i in range(0,len(m),mmap.PAGESIZE)]; '
                            'time.sleep(180)' % (gib * 1024**3))
                    HOG = subprocess.Popen(['numactl', '--membind=' + str(node), sys.executable, '-c', code])
                    try:
                        time.sleep(10)
                        if HOG.poll() is not None:
                            raise RuntimeError('pressure process failed')
                        gold(body, out, name + '.pressure-node' + str(node))
                        if HOG.poll() is not None:
                            raise RuntimeError('pressure expired before gold completed')
                    finally:
                        stop(HOG)
                        HOG = None
                    time.sleep(60)
                    gold(body, out, name + '.rearm-node' + str(node))
        finally:
            stop(SERVER)
            SERVER = None
    lines = logpath.read_text(errors='replace').splitlines()
    for line in [s for s in lines if 'hit rate' in s][-4:]:
        print(name, line, flush=True)
    if not any('hit rate' in s for s in lines):
        raise RuntimeError(name + ': no hit-rate evidence')
    if name == 'ST1':
        (out/'valve-evidence.txt').write_text('\n'.join(s for s in lines if 'memory valve' in s))
        print('Review valve-evidence.txt for releases, low points and re-arm; no adoption from decode alone.', flush=True)


def main():
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized', action='store_true', help='owner has authorized the serving outage')
    args = parser.parse_args()
    if not args.authorized:
        parser.error('requires --authorized and NATIVE_BASE / K3_REQUEST_JSON; see comments in win-stride.sh')
    base, sha, body, fixture, live = preflight()
    out = ROOT/'tools/gh200/runs'/time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out.mkdir(parents=True, exist_ok=False)
    (out/'deployment-before.json').write_text(json.dumps(live))
    shutil.copyfile(fixture, out/'request.json')
    # Keep 15 minutes for termination and restoring unchanged production (<90 min target).
    DEADLINE = time.monotonic() + 75*60
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('75-minute window budget')))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(InterruptedError('interrupted; restoring production')))
    signal.alarm(75*60)
    scaled = False
    try:
        scaled = True  # rollback even if scale response times out after succeeding
        run(K + ['scale', DEPLOY, '--replicas=0'])
        run(K + ['wait', 'pod', '-l', LABEL, '--for=delete', '--timeout=240s'], timeout=250)
        shutil.copyfile(PROFILE['snapshot'], out/'frozen.k3sn')
        print('frozen snapshot sha256', hashlib.sha256((out/'frozen.k3sn').read_bytes()).hexdigest(), flush=True)
        source = out/'source'
        source.mkdir()
        archive = out/'source.tar'
        with archive.open('wb') as f:
            run(['git', '-C', str(base), 'archive', sha], stdout=f)
        run(['tar', '-xf', str(archive), '-C', str(source)])
        for patch in sorted((ROOT/'patches').glob('*.patch')):
            run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=source)
        run(['cmake', '-S', str(source), '-B', str(source/'build'), '-DGGML_CUDA=ON',
             '-DCMAKE_CUDA_ARCHITECTURES=90', '-DCMAKE_BUILD_TYPE=Release',
             '-DLLAMA_CURL=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF'], timeout=180)
        run(['cmake', '--build', str(source/'build'), '--target', 'llama-server', '-j16'], timeout=1200)
        binary = source/'build/bin/llama-server'
        for name, stride in [('ST0', 0), ('ST1', 1), ('ST0b', 0), ('ST1b', 1)]:
            leg(name, stride, binary, out, body)
    finally:
        signal.alarm(0)
        DEADLINE = None
        stop(HOG)
        stop(SERVER)
        if scaled:
            run(K + ['scale', DEPLOY, '--replicas=1'])
            run(K + ['rollout', 'status', DEPLOY, '--timeout=600s'], timeout=610)
            run(K + ['wait', 'pod', '-l', LABEL, '--for=condition=Ready', '--timeout=60s'], timeout=70)
        print('Rollback: unchanged Deployment image; artifacts', out, flush=True)


if __name__ == '__main__':
    main()
