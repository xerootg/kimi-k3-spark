#!/usr/bin/env python3
"""Host-side probes. JSONL is flushed per observation; errors never become timings."""
import argparse
import concurrent.futures
import contextlib
import glob
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import struct
import subprocess
import sys
import time

D = 6144
HDR = struct.Struct('<17i')

def emit(hop, **kw):
    print(json.dumps(dict(hop=hop, **kw)), flush=True)

def recv(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n-len(b))
        if not c:
            raise ConnectionError('short reply')
        b.extend(c)
    return b

def connect(host, port):
    s = socket.create_connection((host, port), timeout=10)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s

def client(host, port, label, samples, seconds, ks=(1,2,4,8), compute=False, cold=False, gap=0):
    rng = random.Random(53)
    x = struct.pack('<6144f', *(rng.gauss(0, .25) for _ in range(D)))
    for k in ks:
        with connect(host, port) as s:
            end = time.monotonic()+seconds
            for i in range(samples+10):
                ids = [3*256 + ((i*k+j)%256 if cold else j) for j in range(k)]
                payload = HDR.pack(k, *ids, *([0]*(16-k)))+x if k else b'P'
                if gap:
                    time.sleep(gap)
                start = time.perf_counter_ns()
                s.sendall(payload)
                result = recv(s, k*D*4 if k else 1)
                us = (time.perf_counter_ns()-start)/1000
                if compute:
                    floats = struct.unpack('<%df' % (k*D), result)
                    if not all(math.isfinite(v) for v in floats):
                        raise RuntimeError('non-finite reply / missing sentinel')
                if i>=10:
                    emit(f'{label}/k={k}', us=us, request_bytes=len(payload), reply_bytes=len(result))
                if time.monotonic()>end:
                    emit(f'{label}/k={k}', status='case time cap; p99 may be undersampled')
                    break

def netserver(port):
    with socket.socket() as ls:
        ls.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        ls.bind(('',port)); ls.listen()
        print('READY', flush=True)
        while True:
            s,_=ls.accept()
            with s:
                s.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                try:
                    while True:
                        first=recv(s,1)
                        if first==b'P': s.sendall(b'P')
                        elif first==b'B':
                            # Client times a fixed byte count, including final acknowledgement.
                            n=struct.unpack('<Q',recv(s,8))[0]
                            while n:
                                b=recv(s,min(n,262144)); n-=len(b)
                            s.sendall(b'K')
                        else:
                            header=first+recv(s,67); k=HDR.unpack(header)[0]
                            if k not in (1,2,4,8): raise ValueError('k')
                            recv(s,D*4); s.sendall(bytes(k*D*4))
                except (ConnectionError, OSError): pass

def bulk(host,port,label):
    with connect(host,port) as s:
        size=128*1024*1024; block=bytes(262144)
        start=time.perf_counter_ns(); s.sendall(b'B'+struct.pack('<Q',size))
        for _ in range(size//len(block)): s.sendall(block)
        recv(s,1); us=(time.perf_counter_ns()-start)/1000
        emit(label+'/bulk', us=us, bytes=size, gbps=size*8/us/1000)

@contextlib.contextmanager
def server(cmd, log, env=None):
    # Only this child process group is signalled. Never search by process name or port.
    with open(log,'w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=f,env=env,start_new_session=True)
        try:
            end=time.monotonic()+50
            while True:
                text=Path(log).read_text()
                if 'serving on' in text or 'READY' in text: break
                if p.poll() is not None: raise RuntimeError('server failed: '+text[-2000:])
                if time.monotonic()>end: raise TimeoutError('startup: '+log)
                time.sleep(.1)
            yield p
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try: p.wait(3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL); p.wait()

def metadata():
    cmds=[['uname','-a'], ['ip','-j','address'], ['ip','-j','route'], ['lsblk','-o','NAME,MODEL,SIZE,MOUNTPOINTS'], ['sudo','-n','fuser','-v','/dev/cpu_dma_latency']]
    for iface in glob.glob('/sys/class/net/*'):
        n=Path(iface).name
        for opts in ('-c','-i','--show-eee'): cmds.append(['sudo','-n','ethtool',opts,n])
    for cmd in cmds:
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=4)
            emit('metadata',cmd=cmd,rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except Exception as e: emit('metadata',cmd=cmd,error=str(e))
    for pattern in ('/sys/devices/system/cpu/cpu*/cpuidle/state*/*','/sys/devices/system/cpu/cpufreq/policy*/energy_performance_preference'):
        for p in glob.glob(pattern):
            try: emit('metadata',path=p,value=Path(p).read_text().strip())
            except OSError: pass

def render_command(cmd, env):
    # Use the same device permissions for benches and serveidx.
    return ['sudo','-n','-u','xero','-g','render','env',
            *[f'{k}={env[k]}' for k in ('LD_LIBRARY_PATH','G53X_MUFU','G53X_EXACT')],
            'taskset','-c','4-11',*cmd]

def local(args):
    base=Path(__file__).resolve().parent
    env=dict(os.environ, LD_LIBRARY_PATH=os.path.expanduser('~/k3-bitexact/sycl/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''), G53X_MUFU=str(base/'mufu'))
    prefix=base/'L3'
    setfile=base/'set.txt'; setfile.write_text(''.join(f'3 {i}\n' for i in range(8)))
    # Extract the SAME eight real records used by serveidx, never modify source store.
    with open(args.store,'rb') as f:
        h=f.read(64); magic,ver,nl,ne,align,entry,index,data=struct.unpack('<4sIIIIIQQ24x',h)
        if magic!=b'K3XS' or ver!=2 or ne!=256 or entry!=64: raise ValueError('expected GLM v2')
        f.seek(index); entries=[struct.unpack('<IIIIQQQQQQ',f.read(64)) for _ in range(nl)]
        e=next(e for e in entries if e[0]==3)
        layer,tg,tu,td,gs,us,ds,rec,offset,stride=e
        emit('dataset', layer=layer, experts=ne, types=[tg,tu,td], record_bytes=rec, stride=stride, base_offset=offset, store=args.store)
        if rec!=12091392 or (tg,tu,td)!=(17,17,18): raise ValueError('unexpected L3')
        for name,size,delta in [('gate',gs,0),('up',us,gs),('down',ds,gs+us)]:
            with open(str(prefix)+'_'+name+'.bin','wb') as out:
                for j in range(8):
                    f.seek(offset+j*stride+delta); b=f.read(size)
                    if len(b)!=size: raise ValueError('short store')
                    out.write(b)
    if args.mode=='igpu':
        for binary,exact in [('g53x',0),('g53x',1),('g53c',0)]:
            env['G53X_EXACT']=str(exact)
            for k in (1,2,4,8):
                cmd=[str(base/binary),str(prefix),str(tg),str(tu),str(td)]
                if binary=='g53c':cmd+=['8']
                cmd+=['bench',str(k),str(args.samples)]
                # Generated sources emit raw samples; native bench replicates eight experts into 64 slots.
                label=f'{socket.gethostname()}/{binary}/exact={exact}/bench/k={k}'
                try:
                    r=subprocess.run(render_command(cmd,env) if binary=='g53x' else ['taskset','-c','4-11',*cmd],env=env,capture_output=True,text=True,timeout=args.bench_seconds)
                    (base/(f'{binary}-{exact}-{k}.log')).write_text(r.stdout+r.stderr)
                    for line in r.stdout.splitlines():
                        if line.startswith('SURVEY '):emit(label,us=float(line.split('us=')[1]))
                    if r.returncode: emit(label,status='failed',rc=r.returncode,error=(r.stdout+r.stderr)[-2000:])
                except subprocess.TimeoutExpired as e:
                    raw=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else (e.stdout or '')
                    err=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else (e.stderr or '')
                    (base/(f'{binary}-{exact}-{k}.log')).write_text(raw+err)
                    for line in raw.splitlines():
                        if line.startswith('SURVEY '):emit(label,us=float(line.split('us=')[1]))
                    emit(label,status='bench timeout; no complete distribution')
        emit('cpu/server',status='unavailable: g53c has bench/check only; k3cs is K3 math')
    if args.mode=='session':
        env.update(G53X_EXACT=str(args.exact),K3X_DISK_TIER=str(int(args.tier!='resident')),K3X_MAX_GB='0.2')
        log=str(base/f'session-{args.exact}-{args.tier}.log')
        cmd=['sudo','-n','-u','xero','-g','render','env',*[f'{k}={env[k]}' for k in ('LD_LIBRARY_PATH','G53X_MUFU','G53X_EXACT','K3X_DISK_TIER','K3X_MAX_GB')], 'taskset','-c','4-11',str(base/'g53x'),str(setfile),args.store,'6144','-','serveidx',str(args.port)]
        with server(cmd,log):
            emit('session',status='READY')
            sys.stdin.readline()  # EOF closes only the server this session started.
        counts=[]
        for line in Path(log).read_text().splitlines():
            if line.startswith('SURVEY_DISK'):counts.append(dict(x.split('=') for x in line.split()[1:]))
        reads=sum(int(c['reads']) for c in counts)
        emit('session',status='disk counters',requests=len(counts),reads=reads)
        if not counts or (args.tier=='lru' and reads) or (args.tier=='cold' and not reads):
            raise RuntimeError('disk counter gate failed')
        return
    for exact in (0,1):
        for tier in (('resident',) if args.mode=='igpu' else ('resident','lru','cold')):
            env.update(G53X_EXACT=str(exact),K3X_DISK_TIER=str(int(tier!='resident')),K3X_MAX_GB='0.2')
            log=str(base/f'{args.mode}-{exact}-{tier}.log')
            cmd=['sudo','-n','-u','xero','-g','render','env',*[f'{k}={env[k]}' for k in ('LD_LIBRARY_PATH','G53X_MUFU','G53X_EXACT','K3X_DISK_TIER','K3X_MAX_GB')], 'taskset','-c','4-11',str(base/'g53x'),str(setfile),args.store,'6144','-','serveidx',str(args.port)]
            with server(cmd,log):
                label=f'{socket.gethostname()}/exact={exact}/{tier}'+('/prototype' if tier!='resident' else '')
                if args.mode=='igpu':
                    client('127.0.0.1',args.port,label+'/loopback',args.samples,args.serve_seconds,compute=True,gap=args.gap)
            # Read counters verify LRU hits and cold misses independently of timing.
            counts=[]
            for line in Path(log).read_text().splitlines():
                if line.startswith('SURVEY_DISK'): counts.append(dict(x.split('=') for x in line.split()[1:]))
            emit(label,status='disk counters',requests=len(counts),reads=sum(int(c['reads']) for c in counts),log=log)

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['netserver','netclient','client','metadata','igpu','session'])
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=29653)
    p.add_argument('--label',default='net');p.add_argument('--samples',type=int,default=1000);p.add_argument('--seconds',type=float,default=8)
    p.add_argument('--ks',default='1,2,4,8');p.add_argument('--compute',action='store_true');p.add_argument('--cold',action='store_true');p.add_argument('--gap',type=float,default=0)
    p.add_argument('--bench-seconds',type=float,default=20);p.add_argument('--serve-seconds',type=float,default=8)
    p.add_argument('--exact',type=int,default=1);p.add_argument('--tier',choices=['resident','lru','cold'],default='resident')
    p.add_argument('--store',default='/srv/glm-5.3/store/glm53-q2-v2.k3xs');p.add_argument('--address');p.add_argument('--spark-dir')
    a=p.parse_args()
    if a.mode=='metadata':metadata()
    elif a.mode=='netserver':
        import threading
        def eof():
            sys.stdin.readline()
            os._exit(0)
        threading.Thread(target=eof,daemon=True).start()
        netserver(a.port)
    elif a.mode in ('client','netclient'):
        client(a.host,a.port,a.label,a.samples,a.seconds,tuple(map(int,a.ks.split(','))) if a.mode=='client' else (0,1,2,4,8),a.compute,a.cold,a.gap)
        if a.mode=='netclient':bulk(a.host,a.port,a.label)
    else:local(a)
if __name__=='__main__':
    def stop(sig,frame): raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as e:
        emit('error',status='failed',error=str(e));sys.exit(1)
