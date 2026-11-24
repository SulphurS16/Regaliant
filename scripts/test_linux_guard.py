"""No GPU and no heat load: fake sensors/clock plus real sleeping process groups."""
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import linux_guard as guard


class ThermalTests(unittest.TestCase):
    def test_sustained_pause_and_critical_abort(self):
        s = guard.ThermalState(103, 97, 60)
        self.assertEqual(s.step(0, 104), 'sample')
        self.assertEqual(s.step(9, 104), 'sample')
        self.assertEqual(s.step(10, 102), 'sample')  # Short spike reset.
        self.assertEqual(s.step(12, 103), 'sample')
        self.assertEqual(s.step(22, 103), 'pause')
        self.assertEqual(s.step(82, 104), 'sample')  # Not a sustained105 C abort.
        self.assertEqual(s.step(83, 97), 'sample')
        self.assertEqual(s.step(84, 96), 'resume')
        self.assertEqual(s.paused_s, 62)
        self.assertEqual(s.step(90, 105), 'sample')
        self.assertEqual(s.step(100, 105), 'pause')
        self.assertEqual(s.step(149, 105), 'sample')
        self.assertEqual(s.step(150, 105), 'abort')

    def test_thermal_defaults_and_overrides(self):
        p = guard.Policy({})
        self.assertEqual((p.pause,p.resume,p.hold,p.abort_temp,p.abort),(103,97,10,105,60))
        p = guard.Policy(dict(DS_RELAY_TEMP_PAUSE='99',DS_RELAY_TEMP_RESUME='90',DS_RELAY_TEMP_HOLD_S='2',DS_RELAY_TEMP_ABORT='104'))
        self.assertEqual((p.pause,p.resume,p.hold,p.abort_temp),(99,90,2,104))

    def test_sensor_preference_fallback_and_disappearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name, label, temp in [(0, 'amdgpu', 'edge', 110000), (1, 'coretemp', 'Core 0', 99000), (2, 'coretemp', 'Package id 0', 80000)]:
                h = root/f'hwmon{i}'; h.mkdir()
                (h/'name').write_text(name)
                (h/'temp1_label').write_text(label)
                (h/'temp1_input').write_text(str(temp))
            self.assertEqual(guard.temperature(guard.sensor_paths(root)), 80)
            (root/'hwmon2/temp1_label').unlink()
            self.assertEqual(guard.temperature(guard.sensor_paths(root)), 99)
            paths = guard.sensor_paths(root)
            paths[0].unlink()
            with self.assertRaises(guard.GuardError): guard.temperature(paths)
            with self.assertRaises(guard.GuardError): guard.temperature([])

    def test_cpu_capacity_and_explicit_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                p = root/f'cpu{i}'; (p/'topology').mkdir(parents=True)
                (p/'cpu_capacity').write_text('1024' if i >= 4 else '700')
                (p/'topology/thread_siblings_list').write_text(str(i))
            self.assertEqual(guard.select_cpus(allowed=set(range(12)), root=root), list(range(4, 12)))
            self.assertEqual(guard.select_cpus('2-3,9', set(range(12)), root), [2, 3, 9])
            with self.assertRaises(guard.GuardError): guard.select_cpus('15', set(range(12)), root)
            with self.assertRaises(ValueError): guard.cpu_list('4-2')

    def test_record_fsyncs_every_line(self):
        with tempfile.TemporaryFile(mode='w+') as f, patch.object(guard.os, 'fsync') as sync:
            guard.record(f, event='pause', temperature_c=96)
            guard.record(f, event='resume', temperature_c=84)
            self.assertEqual(sync.call_count, 2)
            f.seek(0)
            self.assertEqual([json.loads(l)['event'] for l in f], ['pause', 'resume'])

    def test_cooldown_times_out_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(guard.time, 'sleep'), patch.object(guard.time, 'monotonic', side_effect=[0, 1, 2, 3, 4, 5, 6]):
            p = guard.Policy({})
            p.cooldown = 1
            with patch.object(guard, 'temperature', return_value=100):
                with self.assertRaises(guard.GuardError): p.wait_cool(Path(tmp)/'monitor.jsonl')
            self.assertIn('cooldown-abort', (Path(tmp)/'monitor.jsonl').read_text())

    def test_pause_abort_reaps_real_sleeping_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Short timeout only for this test. Cold preflight, hot during execution.
            env = {**os.environ, 'DS_RELAY_TEMP_ABORT_S': '.05', 'DS_RELAY_TEMP_HOLD_S': '0'}
            calls = iter([80, 80])
            with patch.object(guard, 'temperature', side_effect=lambda paths: next(calls, 106)):
                p = guard.Process(['timeout', '-k', '1s', '10s', sys.executable, '-c', 'import time; time.sleep(20)'], output=Path(tmp), env=env, start_new_session=True, stdout=subprocess.DEVNULL)
                self.assertEqual(p.wait(timeout=12), 75)
            data = json.loads((Path(tmp)/'thermal.json').read_text())
            self.assertFalse(data['measurement_valid'])
            self.assertTrue(data['aborted'])
            self.assertGreaterEqual(data['pauses'], 1)
            rows = [json.loads(l) for l in (Path(tmp)/'monitor.jsonl').read_text().splitlines()]
            aborted = next(r for r in rows if r['event'] == 'abort')
            self.assertGreaterEqual(sum(g['states'].get('T', 0) for g in aborted['owned_groups'].values()), 2)
            with self.assertRaises(ProcessLookupError): os.killpg(p.pid, 0)

    def test_wall_timeout_still_ends_a_paused_wrapper_below_critical(self):
        self.assertEqual(guard.timeout_seconds(['timeout','-k','5s','1.5m','app']),90)
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, 'DS_RELAY_TEMP_HOLD_S':'0'}
            calls = iter([80])
            with patch.object(guard, 'temperature', side_effect=lambda paths: next(calls,104)):
                p = guard.Process(['timeout','-k','1s','1s',sys.executable,'-c','import time; time.sleep(20)'],output=Path(tmp),env=env,start_new_session=True,stdout=subprocess.DEVNULL)
                self.assertEqual(p.wait(timeout=12),75)
            report=json.loads((Path(tmp)/'thermal.json').read_text())
            self.assertEqual(report['aborted'],'wall timeout during thermal pause')
            with self.assertRaises(ProcessLookupError): os.killpg(p.pid,0)

    def test_new_process_group_descendant_is_owned(self):
        p = subprocess.Popen([sys.executable, '-c', 'import subprocess,time; subprocess.Popen(["sleep","20"],start_new_session=True); time.sleep(20)'], start_new_session=True)
        try:
            deadline = time.monotonic()+3
            groups = {p.pid}
            while len(groups) < 2 and time.monotonic() < deadline:
                groups = guard.descendant_groups(p.pid)
                time.sleep(.02)
            self.assertEqual(len(groups), 2)
            self.assertNotIn(os.getpgrp(), groups)
        finally:
            for pg in guard.descendant_groups(p.pid):
                try: os.killpg(pg, signal.SIGKILL)
                except ProcessLookupError: pass
            p.wait()


if __name__ == '__main__': unittest.main()
