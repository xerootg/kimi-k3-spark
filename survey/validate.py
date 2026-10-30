#!/usr/bin/env python3
"""Hardware-free checks. Creates synthetic data only inside survey/.validation/."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

base=Path(__file__).resolve().parent
scratch=base/'.validation';scratch.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=scratch) as tmp:
    out=Path(tmp)
    (out/'manifest.json').write_text(json.dumps({'budget_us':3000}))
    with (out/'e2e.jsonl').open('w') as f:
        for tier,mult in [('resident',1),('lru',2),('cold',5)]:
            for i in range(1,1001):
                f.write(json.dumps({'hop':f'spark->nuc2/exact=1/{tier}/prototype/k=1','us':i*mult})+'\n')
    (out/'spark-server.log').write_text('eval time = 23500 ms / 100 tokens ( 4.25 tokens per second)\nexpert store: split GEMV: wait at the real lookup 1.40 ms\n')
    trace=out/'trace.tsv';trace.write_text('t_post\tt_late_hook\tt_resolve\tt_wait_end\n1000\t3001000\t3001000\t3101000\n')
    r=subprocess.run([sys.executable,str(base/'analyze.py'),str(out),'--trace',str(trace)],capture_output=True,text=True,check=True)
    assert '| 500.0 | 900.0 | 990.0 | 1000.0 | 1000 | fits' in r.stdout
    assert '| 2500.0 | 4500.0 | 4950.0 | 5000.0 | 1000 | EXCEEDS' in r.stdout
    assert '1400.0' in r.stdout
    print(r.stdout,end='')
print('PASS: synthetic nearest-rank tails, counts, log/trace parsing, fits/exceeds; no hardware probes executed.')
