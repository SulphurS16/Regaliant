import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('nexus',Path(__file__).with_name('linux-nexus-test.py'))
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

class NexusTests(unittest.TestCase):
    def test_three_runs_reuse_only_normal_stores_and_keep_validation_separate(self):
        self.assertEqual(len(n.CASES),3)
        self.assertEqual([r[2] for r in n.CASES],[False,True,False])
        for label,case,warm,val in n.CASES:
            c=n.command(Path('/binary'),Path('/save'),Path('/shader'),label,case,warm,val)
            self.assertIn('--relative-measure',c)
            self.assertEqual('--load-shots' in c,val)
            self.assertIn('--nexus-replay',c)
            self.assertEqual('--validation' in c,val)
            self.assertEqual('--warm-required' in c,warm)
        self.assertEqual(n.CASES[-1][1],'vertices')

    def test_partial_child_summary_does_not_abort_remaining_cases(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'summary.json').write_text('{partial')
            self.assertFalse(n.load_summary(root)['valid'])
        self.assertEqual(n.load_summary(None),{})
