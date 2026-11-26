import unittest
from linux_play_metrics import analyze

class PlayMetricsTests(unittest.TestCase):
    def test_rates_and_missing_telemetry(self):
        text='Linux heartbeat: 5s flips=10 draws=100 upload_bytes=0 uffd_ns=0 recovered_faults=0\nLinux heartbeat: 35s flips=910 draws=1900 upload_bytes=900000000 uffd_ns=1800000000 recovered_faults=0\n'
        a=analyze(text)['windows'][0]
        self.assertEqual(a['fps'],30)
        self.assertEqual(a['draws_per_flip'],2)
        self.assertEqual(a['upload_MB_per_flip'],1)
        self.assertEqual(a['uffd_aggregate_ms_per_flip'],2)
        self.assertNotIn('fg_compute_ms_per_flip',a)

    def test_extra_counters_merge_by_second(self):
        text='Linux heartbeat: 1s flips=0\nLinux play: 1s fg_ns=0 fg_pace_ns=0 pb_source_calls=0\nLinux heartbeat: 31s flips=600\nLinux play: 31s fg_ns=600000000 fg_pace_ns=6000000000 pb_source_calls=2\n'
        a=analyze(text)['windows'][0]
        self.assertEqual(a['fg_compute_ms_per_flip'],1)
        self.assertEqual(a['fg_pace_ms_per_flip'],10)
        self.assertEqual(a['pb_source_calls'],2)

    def test_load_residency_and_adaptive_rates_are_deltas_not_lifetime_totals(self):
        text = 'Linux heartbeat: 1s flips=10\nLinux play: 1s texture_zero_parts=100\nLinux adaptive: 1s fg_repeats=2\nLinux heartbeat: 2s flips=40\nLinux play: 2s texture_zero_parts=104\nLinux adaptive: 2s fg_repeats=3\n'
        row=analyze(text)['per_second'][0]
        self.assertEqual(row['texture_zero_parts_per_s'],4)
        self.assertEqual(row['fg_repeats_per_s'],1)

    def test_adaptive_repeats_are_not_counted_as_fresh_frames(self):
        import tempfile
        from pathlib import Path
        from linux_play_metrics import adaptive_report
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);folder=run/'user/linux-runs/test';folder.mkdir(parents=True)
            (folder/'fg-adaptive.csv').write_text('slot_ns,flip,generated,repeat,age_ms,present_ok\n0,100,false,false,30,true\n10000000,101,true,false,31,true\n20000000,101,true,true,41,false\n')
            r=adaptive_report(run,100,102)
            self.assertEqual(r['counts'],{'real':0,'generated':1,'repeated':1})
            self.assertEqual(r['fresh_fps'],50)
            self.assertEqual(r['slot_fps'],100)
