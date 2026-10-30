"""Synthetic validation; no K3 binary routing dumps are checked in."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('place', Path(__file__).with_name('place.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PlacementTests(unittest.TestCase):
    def test_partition_capacity_and_hotness(self):
        meta = {k:(p.RECORD,p.RECORD,10,10,0) for k in p.KEYS}
        hot = {k:len(p.KEYS)-i for i,k in enumerate(p.KEYS)}
        owners = p.place(hot,meta,6859,[44e9,18e9])
        self.assertEqual(set(owners),set(p.KEYS))
        self.assertTrue(all(owners[k]=='spark' for k in p.KEYS[:6859]))
        for b,gb in [('nuc2',44),('nuc3',18)]:
            self.assertLessEqual(sum(meta[k][0] for k in owners if owners[k]==b+'-ram'),gb*1e9)
        self.assertTrue(any(v.endswith('-ssd') for v in owners.values()))

    def test_trace_roundtrip_and_truncation(self):
        steps = p.synthetic(4,0,53)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'trace.bin'
            with path.open('wb') as f:
                for st,layers in steps:
                    for l,ids in layers.items():
                        f.write(struct.pack('<4i8i',st,l,3,8,*ids))
            loaded,_,dropped = p.load_trace(path)
            self.assertEqual(loaded,steps)
            self.assertEqual(dropped,0)
            with path.open('ab') as f:
                f.write(b'bad')
            with self.assertRaises(ValueError):
                p.load_trace(path)

    def test_global_row_cap_fallback_and_predictor(self):
        steps = p.synthetic(4,0,53)
        meta = {k:(p.RECORD,p.RECORD,None,None,None) for k in p.KEYS}
        owners = {k:('nuc2-ram' if k[1]%2 else 'nuc3-ram') for k in p.KEYS}
        r = p.replay(steps,{},owners,meta,rows=2,deadline=2200)
        self.assertEqual(r['remote_rows_per_token'],150)
        self.assertTrue(all(v['maximum']<=2 for v in r['rows'].values()))
        slow = p.replay(steps,{},owners,meta,rows=2,deadline=1)
        self.assertGreater(slow['fallback_rows_per_token'],0)
        self.assertGreater(slow['spark_reads_per_token'],0)
        local = {k:'spark' for k in p.KEYS}
        a = p.replay(steps,{},local,meta,precision=1,slots=1)
        self.assertGreater(a['prefetch_reads_per_token'],0)

    def test_v2_idx_offsets(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'header'
            h=struct.pack('<4sIIIIIQQ',b'K3XS',2,75,256,524288,64,64,524288)
            with path.open('wb') as f:
                f.write(h.ljust(64,b'\0'))
                for l in p.LAYERS:
                    f.write(struct.pack('<4I6Q',l,10,10,10,4000000,4000000,4091392,p.RECORD,524288+l*256*12582912,12582912))
            m=p.metadata(path)
            self.assertEqual(m[3,1][4]-m[3,0][4],12582912)


if __name__=='__main__':
    unittest.main(verbosity=2)
