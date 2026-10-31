#!/usr/bin/env python3
"""Offline K3 capacities, large offsets, and 64 KiB valve boundary checks."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('window', Path(__file__).with_name('win-stride.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

class Checks(unittest.TestCase):
    def test_k3(self):
        groups = [(80,(50,50,50)), (10,(66,66,50)), (2,(50,50,66))]
        total_weight = sum(n*sum(ts)*43008 for n,ts in groups)
        counts = []
        for n, ts in groups:
            rec = sum(ts)*43008
            old = math.ceil(rec/math.lcm(4096,*ts))*math.lcm(4096,*ts)
            stride = math.ceil(rec/4096)*4096
            self.assertEqual(stride, rec)
            budget = int(float(424000*1024**2)*n*rec/total_weight)
            counts.append((budget//old-1,budget//stride-1))
            off = 0
            for size in ts:
                for slot in [0,1,1027,7302,58435,67197]:
                    for block in [0,1,43007]:
                        ptr = slot*stride+off+block*size
                        self.assertEqual(ptr%2,0)
                        if slot > 1027:
                            self.assertGreater(ptr, 2**32)
                off += size*43008
            # A valve release must never erase a neighbour on 64 KiB GH200 pages.
            for first,last in [(0,0),(1,1),(123,125),(7000,7100)]:
                lo,hi=first*stride,(last+1)*stride
                a,b=math.ceil(lo/65536)*65536,hi//65536*65536
                self.assertTrue(lo<=a<=b<=hi)
        self.assertEqual(counts, [(58436,58436),(5639,7303),(1027,1459)])
        print('PASS: K3 65102 -> 67198 (+2096, 3.220%); 64-bit offsets, 64 KiB valve interiors')

    def test_production_profile(self):
        self.assertEqual(w.PROFILE['env']['LLAMA_EXPERT_STORE_GPU_SLOTS'], '0')
        self.assertEqual(w.PROFILE['env']['LLAMA_EXPERT_STORE_HBM_MIB'], '24576')
        args=w.PROFILE['args']
        self.assertEqual(args[args.index('--ctx-size')+1], '524288')
        self.assertEqual(args[args.index('--parallel')+1], '2')

    def test_failure_rolls_back(self):
        # Fail immediately after scale-down: finally must scale up and wait Ready.
        import tempfile
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]/'.stride-check') as tmp:
            root=Path(tmp)
            (root/'versions.env').write_text('LLAMA_SHA=stub\n')
            fixture=root/'request.json'; fixture.write_text('{}')
            calls=[]
            def fake_run(cmd, **kw):
                calls.append(cmd)
                if 'delete' in ' '.join(cmd):
                    raise RuntimeError('injected wait failure')
            with patch.object(w,'ROOT',root), patch.object(w,'preflight',return_value=(root,'stub',{},fixture,{})), patch.object(w,'run',side_effect=fake_run), patch('sys.argv',['win-stride.py','--authorized']):
                with self.assertRaisesRegex(RuntimeError,'injected'):
                    w.main()
            self.assertTrue(any('--replicas=0' in c for c in calls))
            self.assertTrue(any('--replicas=1' in c for c in calls))
            self.assertTrue(any('--for=condition=Ready' in c for c in calls))
        print('PASS: injected post-scale failure restores replicas=1 and waits Ready (mock kubectl)')

if __name__=='__main__':
    unittest.main()
