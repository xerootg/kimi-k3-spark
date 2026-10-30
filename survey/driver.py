#!/usr/bin/env python3
"""Laptop controller: SSH control always originates here, never between hosts."""
import argparse
import concurrent.futures
import contextlib
import datetime
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import threading

ROOT=Path(__file__).resolve().parents[1]
HOSTS={'spark':'192.168.4.209','nuc2':'192.168.4.249','nuc3':'192.168.4.250'}
SSH=['ssh','-oBatchMode=yes','-oConnectTimeout=5','-oServerAliveInterval=5','-oServerAliveCountMax=2']
CHILDREN=set()
def ssh(host,args,**kw):
    return subprocess.run([*SSH,'xero@'+HOSTS[host],shlex.join(list(map(str,args)))],**kw)
def record(path,obj):
    with open(path,'a') as f:f.write(json.dumps(obj)+'\n')
def remote(host,args,path,limit):
    with open(path,'a') as f:
        r=ssh(host,['timeout','--signal=TERM','--kill-after=5',str(limit),*args],stdout=f,stderr=subprocess.STDOUT,timeout=limit+15)
    if r.returncode:record(path,dict(hop='error',status='remote failed',host=host,rc=r.returncode,command=args))
    return r.returncode
@contextlib.contextmanager
def session(host,args,log,limit=150):
    p=subprocess.Popen([*SSH,'xero@'+HOSTS[host],shlex.join(['timeout','--signal=TERM','--kill-after=5',str(limit),*map(str,args)])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    CHILDREN.add(p)
    try:
        # Outer host timeout bounds readline even if the remote command never reports readiness.
        with open(log,'a') as f:
            while True:
                line=p.stdout.readline();f.write(line);f.flush()
                if not line:raise RuntimeError('server exited before READY: '+str(log))
                if 'READY' in line:break
            yield
            p.stdin.close()
            for line in p.stdout:f.write(line)
        p.wait(timeout=10)
        if p.returncode:raise RuntimeError('server session failed: '+str(log))
    finally:
        if p.stdin and not p.stdin.closed:p.stdin.close()
        if p.poll() is None:
            try:p.wait(timeout=8)
            except subprocess.TimeoutExpired:p.terminate()
        CHILDREN.discard(p)
def parallel(fn,items):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for result in ex.map(fn,items):pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('section',choices=['net','nvme','igpu','e2e','all'],nargs='?',default='all');ap.add_argument('--results');ap.add_argument('--skip-build',action='store_true');a=ap.parse_args()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=Path(a.results).resolve() if a.results else ROOT/'survey/results'/stamp;out.mkdir(parents=True,exist_ok=True)
    work=out/'build';work.mkdir(exist_ok=True)
    rd='/home/xero/glm53-survey/'+stamp
    samples=int(os.environ.get('SURVEY_SAMPLES','1000'))
    case_seconds=float(os.environ.get('SURVEY_CASE_SECONDS','4'))
    serve_seconds=float(os.environ.get('SURVEY_SERVE_SECONDS','8'))
    bench_seconds=float(os.environ.get('SURVEY_BENCH_SECONDS','20'))
    nvme_seconds=float(os.environ.get('SURVEY_NVME_SECONDS','4'))
    manifest=dict(timestamp=stamp,section=a.section,samples=samples,case_seconds=case_seconds,serve_seconds=serve_seconds,bench_seconds=bench_seconds,nvme_seconds=nvme_seconds,remote_dir=rd,hosts=HOSTS,budget_us=3000,budget_source='K3 deadline assumption; not measured GLM window',disk='survey-only eight-slot repacked LRU prototype',bench='64 slots replicated from the same eight real experts')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Results:',out,flush=True)
    sections=['net','nvme','igpu','e2e'] if a.section=='all' else [a.section]
    needs_gpu=any(s in sections for s in ('igpu','e2e'))
    if needs_gpu:
        subprocess.run([sys.executable,str(ROOT/'survey/prepare.py'),str(work)],check=True)
        W=Path(os.environ.get('W',str(Path.home()/'glm53-work'))).resolve()
        if not a.skip_build:
            # Build the generated source directly, without changing production tools or binaries.
            import shutil
            dest=W/'survey-build';dest.mkdir(parents=True,exist_ok=True)
            for p in work.iterdir():shutil.copy2(p,dest/p.name)
            cmd='icpx -fsycl -O2 -ffp-contract=off -fno-fast-math -std=c++17 -pthread -I/w/llama/ggml/src -I/bitexact -o /w/survey-build/g53x /w/survey-build/g53x.cpp'
            with open(out/'build.log','w') as f:subprocess.run([str(ROOT/'tools/glm53/dsycl.sh'),cmd],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
            subprocess.run(['g++','-O2','-static','-std=c++17','-march=alderlake','-mavxvnni','-pthread','-I'+str(ROOT/'tools/glm53/ref'),str(work/'g53c.cpp'),'-o',str(dest/'g53c')],check=True,timeout=90)
        binaries=[W/'survey-build/g53x',W/'survey-build/g53c']
        for p in binaries:
            if not p.is_file():raise FileNotFoundError(p)
    if 'nvme' in sections:
        # A static pread fallback needs neither a NUC compiler nor matching glibc.
        subprocess.run(['g++','-O2','-static','-std=c++17','-pthread',
                        '-I'+str(ROOT/'tools/bitexact'),str(ROOT/'survey/glm_readbench.cpp'),
                        '-o',str(work/'glm_readbench')],check=True,timeout=90)
    for host in HOSTS:
        ssh(host,['mkdir','-p',rd],check=True,timeout=15)
        subprocess.run(['scp','-q',str(ROOT/'survey/probe.py'),'xero@'+HOSTS[host]+':'+rd+'/'],check=True,timeout=20)
        remote(host,['python3',rd+'/probe.py','metadata'],out/(host+'-metadata.jsonl'),45)
        if host!='spark':
            if needs_gpu:
                subprocess.run(['scp','-q',*map(str,binaries),'xero@'+HOSTS[host]+':'+rd+'/'],check=True,timeout=30)
                ssh(host,['mkdir','-p',rd+'/mufu'],check=True,timeout=10)
                # Tables (2.2 GB) are staged once per host in ~/glm53-mufu (rsync, no timeout) and linked per run.
                # Staged from the Spark over the LAN (the laptop is across a WAN); never pushed from here.
                for table in ('rcp.bin','ex2p.bin','ex2n.bin'):
                    ssh(host,['test','-s','/home/xero/glm53-mufu/'+table],check=True,timeout=10)
                ssh(host,['sh','-c','ln -sf ~/glm53-mufu/rcp.bin ~/glm53-mufu/ex2p.bin ~/glm53-mufu/ex2n.bin '+rd+'/mufu/'],check=True,timeout=10)
            if 'nvme' in sections:
                subprocess.run(['scp','-q',str(ROOT/'survey/glm_readbench.cpp'),str(ROOT/'tools/bitexact/expert_disk.h'),str(work/'glm_readbench'),'xero@'+HOSTS[host]+':'+rd+'/'],check=True,timeout=15)
    def probe(*args):return ['python3',rd+'/probe.py',*map(str,args)]
    timer=None
    def section_limit(seconds):
        nonlocal timer
        if timer:timer.cancel()
        timer=threading.Timer(float(os.environ.get('SURVEY_SECTION_SECONDS',seconds)),lambda:os.kill(os.getpid(),signal.SIGTERM));timer.daemon=True;timer.start()
    try:
        if 'net' in sections:
            section_limit(240)
            for dest in ('nuc2','nuc3'):
                with session(dest,probe('netserver'),out/(dest+'-netserver.log'),180):
                    for src in ('spark', 'nuc3' if dest=='nuc2' else 'nuc2'):
                        remote(src,probe('netclient','--host',HOSTS[dest],'--label',src+'->'+dest+'/ethernet','--samples',samples),out/'net.jsonl',50)
                    if dest=='nuc3':
                        # Discover connected /30 peers; verify route uses TB before measuring.
                        r=ssh('nuc2',['ip','-j','-4','addr'],capture_output=True,text=True,check=True,timeout=10)
                        import ipaddress
                        other=ssh('nuc3',['ip','-j','-4','addr'],capture_output=True,text=True,check=True,timeout=10)
                        peer_ips={x['local'] for i in json.loads(other.stdout) if i['ifname'].startswith('thunderbolt') for x in i.get('addr_info',[])}
                        (out/'tb-addresses.json').write_text(other.stdout)
                        found=False
                        for iface in json.loads(r.stdout):
                            if not iface['ifname'].startswith('thunderbolt'):continue
                            for addr in iface.get('addr_info',[]):
                                if addr.get('prefixlen')!=30:continue
                                net=ipaddress.ip_interface(addr['local']+'/30')
                                peer=next(str(x) for x in net.network.hosts() if str(x)!=addr['local'])
                                if peer not in peer_ips:continue
                                reverse=ssh('nuc3',['ip','-j','route','get',addr['local']],capture_output=True,text=True,check=True,timeout=10)
                                if not json.loads(reverse.stdout)[0]['dev'].startswith('thunderbolt'):continue
                                route=ssh('nuc2',['ip','-j','route','get',peer],capture_output=True,text=True,check=True,timeout=10)
                                if json.loads(route.stdout)[0]['dev']!=iface['ifname']:continue
                                found=True
                                remote('nuc2',probe('netclient','--host',peer,'--label','nuc2->nuc3/'+iface['ifname'],'--samples',samples),out/'net.jsonl',50)
                                with session('nuc2',probe('netserver','--port',29654),out/'tb-server.log',75):
                                    remote('nuc3',probe('netclient','--host',addr['local'],'--port',29654,'--label','nuc3->nuc2/'+iface['ifname'],'--samples',samples),out/'net.jsonl',50)
                        if not found:record(out/'net.jsonl',dict(hop='thunderbolt',status='missing connected /30; no Ethernet substitution'))
        if 'nvme' in sections:
            section_limit(180)
            def nvme(host):
                # Probe compiler/liburing locally on each NUC; runtime ring denial falls back per worker.
                build='cd '+shlex.quote(rd)+' && if command -v g++ >/dev/null; then (g++ -O2 -std=c++17 -pthread -DK3X_HAVE_URING glm_readbench.cpp -luring -o glm_readbench.native && mv glm_readbench.native glm_readbench) || echo "native liburing build unavailable; using static pread"; else echo "no remote compiler; using static pread"; fi; test -x glm_readbench'
                if remote(host,['bash','-c',build],out/(host+'-nvme-build.log'),45):
                    record(out/(host+'-nvme.jsonl'),dict(hop='error',status='failed',error='NVMe build/preflight failed; sweep skipped'))
                    return
                for records in (1,8):
                    cases=[(12091392,1)]+[(p,q) for p in (524288,1048576,4194304) for q in (1,4,8,16)]
                    for piece,qd in cases:
                        remote(host,[rd+'/glm_readbench','/srv/glm-5.3/store/glm53-q2-v2.k3xs',piece,qd,records,samples,nvme_seconds],out/(host+'-nvme.jsonl'),int(nvme_seconds+10))
            parallel(nvme,('nuc2','nuc3'))
        if 'igpu' in sections:
            section_limit(480)
            parallel(lambda h:remote(h,probe('igpu','--samples',samples,'--bench-seconds',bench_seconds,'--serve-seconds',serve_seconds),out/(h+'-igpu.jsonl'),int(os.environ.get('SURVEY_SECTION_SECONDS','480'))),('nuc2','nuc3'))
        if 'e2e' in sections:
            section_limit(480)
            # Spark uses CPU/network only; serialize boxes so client load doesn't cross-contaminate.
            for h in ('nuc2','nuc3'):
                for exact in (0,1):
                    for tier in ('resident','lru','cold'):
                        label=f'spark->{h}/exact={exact}/{tier}'+('/prototype' if tier!='resident' else '')
                        with session(h,probe('session','--exact',exact,'--tier',tier),out/(h+f'-{exact}-{tier}-session.jsonl'),int(8*case_seconds+120)):
                            remote('spark',probe('client','--host',HOSTS[h],'--label',label,'--samples',samples,'--seconds',case_seconds,'--compute','--ks','1,2,3,4,5,6,7,8',*(['--cold'] if tier=='cold' else [])),out/'e2e.jsonl',int(8*case_seconds+60))
    finally:
        if timer:timer.cancel()
        for h in HOSTS:
            target=out/(h+'-raw');target.mkdir(exist_ok=True)
            # Pull logs only, not generated slices or binaries.
            r=ssh(h,['find',rd,'-maxdepth','1','-name','*.log','-print'],capture_output=True,text=True,timeout=15)
            for p in r.stdout.splitlines():
                if p.startswith(rd+'/'):subprocess.run(['scp','-q','xero@'+HOSTS[h]+':'+p,str(target)],timeout=15)
    subprocess.run([sys.executable,str(ROOT/'survey/analyze.py'),str(out)],check=True)
if __name__=='__main__':
    def stop(sig,frame):
        for p in list(CHILDREN):
            if p.stdin and not p.stdin.closed:p.stdin.close()
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop)
    main()
