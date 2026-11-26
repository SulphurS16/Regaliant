import importlib.util
from pathlib import Path
import unittest


def module(file):
    spec=importlib.util.spec_from_file_location(file,Path(__file__).with_name(file+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

playtest=module('linux-playtest')
report=module('linux-session-report')


class Playtest(unittest.TestCase):
    def test_incomplete_diagnostic_or_invalid_timing_fails(self):
        r = dict(exit_code=0, validation_errors=0, crash_bytes=0, measurement_valid=True)
        self.assertFalse(playtest.failed(r))
        r['measurement_valid'] = False
        self.assertTrue(playtest.failed(r))
        r.update(ui_probe=True, ui_probe_complete=True, capture_manifests=['a', 'b'])
        self.assertFalse(playtest.failed(r))
        r['ui_probe_complete'] = False
        self.assertTrue(playtest.failed(r))
        r.update(ui_probe_complete=True, capture_manifests=['a'])
        self.assertTrue(playtest.failed(r))

    def test_equal_presentation_different_guest_resolution(self):
        a,b=playtest.case_environment('1440'),playtest.case_environment('4k')
        self.assertEqual({k for k in a if a[k]!=b[k]}, {'DS_RELAY_RENDER_SIZE'})
        self.assertEqual(a['DS_LINUX_COALESCE_SPARSE'], '0')
        self.assertEqual(playtest.case_environment('diagnostic')['DS_RELAY_UI_PROBE'], '1')
        self.assertEqual(playtest.case_environment('4k',True)['DS_PIPELINE_BINARIES'], '0')

    def test_actual_render_size_and_rates_not_original_mode_size(self):
        text='''[ 38000.000][vblank] I: render size 3840x2160 set (the mode's was 2560x1440)
[ 40684.000][main] I: Linux heartbeat: 40s flips=2000 recovered_faults=30000
[ 80684.000][main] I: Linux heartbeat: 80s flips=3200 recovered_faults=270000
[ 51000.000][vblank] I: hitch: flip 2100 took 3000 ms (median 18): pipeline 6000 ms, GPU wait 20 ms; longest: pipeline 2000 ms on worker
'''
        r=report.analyze(text)
        self.assertEqual(r['windows'][0]['rates']['flips'],30)
        self.assertEqual(r['windows'][0]['rates']['recovered_faults'],6000)
        self.assertEqual(r['windows'][0]['render_size_at_start']['width'],3840)
        self.assertEqual(r['longest_hitches'][0]['spans_ms']['pipeline'],6000)
        self.assertEqual(r['longest_hitches'][0]['wall_ms'],3000)
