import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import linux_framegen_compare as compare


class ComparisonTests(unittest.TestCase):
    def fixture(self, folder):
        rows=[]
        for name,value in [('previous-real',0.2),('real',0.6),('generated',0.4),
                           ('previous-hudless',0.2),('hudless',0.6)]:
            a=np.full((2,3,4),value,dtype='<f2');a[...,3]=np.nan
            a.tofile(folder/(name+'.raw'))
            rows.append(dict(name=name,width=3,height=2,format_name='R16G16B16A16_SFLOAT'))
        for name,channels,dtype,fmt in [('depth',1,'<f4','R32_SFLOAT'),('motion',2,'<f2','R16G16_SFLOAT')]:
            np.zeros((2,3,channels),dtype=dtype).tofile(folder/(name+'.raw'))
            rows.append(dict(name=name,width=3,height=2,format_name=fmt))
        (folder/'inputs.json').write_text(json.dumps(dict(images=rows,reset=0,current_taa_id=2,previous_taa_id=1)))

    def test_midpoint_and_alpha_is_not_color(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);r=compare.analyze(p)
            self.assertFalse(r['failures'])
            self.assertLess(r['midpoint']['mae'],0.0002)
            self.assertEqual(r['midpoint']['finite_rgb_components'],18)
            self.assertEqual(r['hud_color']['differing_rgb_pixels'],0)

    def test_nan_cannot_disappear_in_a_finite_only_score(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            a=np.fromfile(p/'generated.raw',dtype='<f2');a[0]=np.nan;a.tofile(p/'generated.raw')
            a=np.fromfile(p/'motion.raw',dtype='<f2');a[0]=np.inf;a[2]=2;a.tofile(p/'motion.raw')
            r=compare.analyze(p)
            self.assertEqual(r['midpoint']['nonfinite_comparison_pixels'],1)
            self.assertEqual(r['motion']['motion']['nonfinite_pixels'],1)
            self.assertEqual(r['motion']['motion']['out_of_range_pixels'],1)
            self.assertEqual(len(r['failures']),2)

    def test_truncated_dump_and_reset_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            (p/'generated.raw').write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'expected'):compare.analyze(p)
            self.fixture(p)
            j=json.loads((p/'inputs.json').read_text());j['reset']=1
            (p/'inputs.json').write_text(json.dumps(j))
            with self.assertRaisesRegex(ValueError,'non-reset'):compare.analyze(p)

    def test_regression_is_numerical(self):
        baseline={'midpoint':{'mae':0.01}}
        self.assertFalse(compare.compare_baseline({'midpoint':{'mae':0.011}},baseline,0.002))
        self.assertTrue(compare.compare_baseline({'midpoint':{'mae':0.02}},baseline,0.002))
