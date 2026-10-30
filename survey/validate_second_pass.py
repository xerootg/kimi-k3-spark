#!/usr/bin/env python3
"""Offline regressions using owner evidence; no SSH, sockets or hardware probes."""
import collections
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import driver
import probe

base = Path(__file__).resolve().parent
raw = base/'results/20261030T074347386706Z'
errors = collections.Counter()
for f in raw.glob('*.jsonl'):
    for line in f.read_text().splitlines():
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if 'failed' in r.get('status', ''): errors[f.name] += 1
assert dict(errors) == {'nuc2-nvme.jsonl':26, 'nuc3-nvme.jsonl':26,
                        'nuc2-igpu.jsonl':12, 'nuc3-igpu.jsonl':12}
assert sum(errors.values()) == 76
for f in raw.glob('*session.jsonl'):
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    counters = rows[-1]
    assert counters['status'] == 'disk counters' and counters['requests'] > 0
    assert (counters['reads'] > 0) == ('-cold-' in f.name)
cmd = probe.render_command(['/run/g53x','bench'], dict(LD_LIBRARY_PATH='runtime',G53X_MUFU='tables',G53X_EXACT='1'))
assert cmd[:8] == ['sudo','-n','-u','xero','-g','render','env','LD_LIBRARY_PATH=runtime']
assert cmd[-4:] == ['-c','4-11','/run/g53x','bench']
# Exercise actual controller command construction with every external operation mocked.
calls=[]
def fake_run(cmd, **kw):
    calls.append(cmd)
    return subprocess.CompletedProcess(cmd,0,stdout='',stderr='')
with patch.object(sys,'argv',['driver.py','nvme','--results',str(base/'.validation-build/controller')]), patch.object(driver.subprocess,'run',fake_run):
    driver.main()
assert any(c[0]=='g++' and '-static' in c and str(base/'glm_readbench.cpp') in c for c in calls)
assert any(c[0]=='scp' and any(str(x).endswith('/glm_readbench') for x in c[1:-1]) for c in calls)
builds=[c[-1] for c in calls if c[0]=='ssh' and 'command -v g++' in c[-1]]
assert len(builds)==2 and all('test -x glm_readbench' in c for c in builds)
r=subprocess.run([sys.executable,str(base/'analyze.py'),str(raw)],capture_output=True,text=True,check=True)
assert 'Errors: 76.' in r.stdout
assert 'Missing Spark end-to-end' not in r.stdout
assert 'Missing NVMe record distributions' in r.stdout
assert '| spark->nuc2/ethernet/k=0 | 103.9 | 108.0 | 205.6 | 1128.3 | 1000 | — |' in r.stdout
assert 'nuc2-nvme.jsonl: nuc2/error: rc=127' in r.stdout
print('PASS: all 76 errors reconciled; all 12 disk counter gates verified.')
print('PASS: static NVMe staging, render bench invocation, analyzer coverage and network budget regression; no network used.')
