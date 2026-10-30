#!/usr/bin/env python3
"""Aggregate raw JSONL, supplied server logs and remote trace TSV. Units: microseconds."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import re

def percentile(v,p):
    return v[max(0,math.ceil(len(v)*p)-1)]  # nearest rank; no interpolation

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('results',type=Path)
    ap.add_argument('--budget-us',type=float,help='owner-measured post-to-needed window; overrides K3 assumption')
    ap.add_argument('--log',type=Path,help='owner-supplied GLM server log (never fetched or signalled by runner)')
    ap.add_argument('--trace',type=Path,help='existing REMOTE_TRACE TSV, nanosecond timestamps')
    a=ap.parse_args();groups=collections.defaultdict(list);notes=set();errors=0;bulk=[]
    manifest=json.loads((a.results/'manifest.json').read_text()) if (a.results/'manifest.json').exists() else {}
    budget=a.budget_us if a.budget_us is not None else manifest.get('budget_us',3000)
    for f in sorted(a.results.rglob('*.jsonl')):
        if '-raw' in str(f.parent):continue
        for line in f.read_text(errors='replace').splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:
                if line.strip():notes.add('Non-JSON diagnostics present; inspect raw files.')
                continue
            hop=r.get('hop','unknown')
            if f.name.startswith(('nuc2-nvme','nuc3-nvme')):hop=f.name.split('-')[0]+'/'+hop
            if 'uring_workers' in r:
                notes.add(f"{f.name}: io_uring workers={r['uring_workers']}, pread workers={r['pread_workers']} (per case).")
            if 'gbps' in r:bulk.append((hop,r['gbps']));continue
            if 'us' in r:
                us=float(r['us'])
                if math.isfinite(us) and us>=0:groups[hop].append(us)
                else:errors+=1
            if 'status' in r:
                if 'failed' in r['status']:
                    errors+=1
                    notes.add(f"{f.name}: {hop}: rc={r.get('rc','?')} {r.get('error',r.get('command',''))}")
                if r['status']!='disk counters':notes.add(hop+': '+r['status'])
                elif '-lru-' in f.name and r.get('reads')!=0:notes.add('INVALID LRU: disk reads reported in '+f.name)
                elif '-cold-' in f.name and not r.get('reads'):notes.add('INVALID cold: no disk reads in '+f.name)
    if not any('/nvme/records=' in k for k in groups):notes.add('Missing NVMe record distributions (inspect *-nvme-build.log).')
    log=a.log or (a.results/'spark-server.log')
    if log.exists():
        for line in log.read_text(errors='replace').splitlines():
            m=re.search(r'([\d.]+) tokens per second',line)
            if m and 'eval time' in line and 'prompt eval' not in line:
                groups['spark/log/mean-layer-from-decode'].append(1e6/float(m[1])/75)
            m=re.search(r'wait at the real lookup ([\d.]+) ms',line)
            if m:groups['spark/log/mean-real-lookup-wait'].append(float(m[1])*1000)
            m=re.search(r'per layer fill: plan ([\d.]+) ms, reads ([\d.]+) ms.*between fills ([\d.]+) ms',line)
            if m:
                for label,n in zip(('fill-plan','fill-reads','between-fills'),m.groups()):groups['spark/log/mean-'+label].append(float(n)*1000)
        notes.add('Log rows are distributions of reported averages, not individual layer observations.')
    else:notes.add('Spark server log absent: actual layer window and lookup wait unknown.')
    if a.trace:
        with a.trace.open() as f:
            for r in csv.DictReader(f,delimiter='\t'):
                for label,left,right in [('post-to-real-window','t_post','t_late_hook'),('real-lookup-wait','t_resolve','t_wait_end')]:
                    lo,hi=int(r[left]),int(r[right])
                    if lo>0 and hi>=lo:groups['spark/trace/'+label].append((hi-lo)/1000)
        notes.add('Trace windows are reported separately; choose --budget-us for the required lower-tail window. No automatic substitution of their median.')
    print('All latency columns in us; nearest-rank percentiles.')
    print(f'Budget: {budget:.1f} us ({"owner supplied" if a.budget_us is not None else "ASSUMED K3 deadline; GLM window unmeasured"}).')
    print('| hop | p50 | p90 | p99 | max | samples | p99 budget |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for hop,v in sorted(groups.items()):
        v.sort();qs=[percentile(v,p) for p in (.5,.9,.99)]+[v[-1]]
        decision='—'
        if hop.startswith('spark->') and '/exact=' in hop:
            decision='fits' if qs[2]<=budget else 'EXCEEDS'
            if len(v)<1000:decision+='; p99 weak'
            if '/exact=0/' in hop:decision+='; fast math only'
            if '/prototype/' in hop:decision+='; prototype'
        print('| '+hop+' | '+' | '.join(f'{n:.1f}' for n in qs)+f' | {len(v)} | {decision} |')
    for hop,gbps in bulk:print(f'Bulk {hop}: {gbps:.3f} Gbit/s (one stream, one transfer; excluded from latency percentiles).')
    if any(len(v)<1000 for k,v in groups.items() if '/log/' not in k):notes.add('Below 1,000 samples: tail estimate is weak; rerun that section with a higher time cap. At 1,000 samples only ~10 observations cover the top 1%.')
    for host in ('nuc2','nuc3'):
        for exact in (0,1):
            for tier in ('resident','lru','cold'):
                missing=[k for k in range(1,9) if not any(
                    hop.startswith(f'spark->{host}/exact={exact}/{tier}/') and hop.endswith(f'/k={k}')
                    for hop in groups)]
                if missing:notes.add(f'Missing Spark end-to-end: {host} exact={exact} {tier} k={missing}')
    if not any('g53c' in k for k in groups):notes.add('CPU comparison absent; g53c is bench-only, no GLM CPU server exists in this checkout.')
    print(f'Errors: {errors}. Raw files remain authoritative; incomplete cases are never imputed.')
    for note in sorted(notes):print('NOTE:',note)
if __name__=='__main__':main()
