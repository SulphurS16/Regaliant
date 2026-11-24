"""CPU-only checks for relay accounting and process cleanup (no Vulkan/game access)."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import linux_waits
import linux_nan_summary

spec = importlib.util.spec_from_file_location('relay', Path(__file__).with_name('linux-relay.py'))
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)


def stat(user, kernel, start):
    fields = ['0'] * 50
    fields[0], fields[11], fields[12], fields[19] = 'S', str(user), str(kernel), str(start)
    return '12 (name with ) spaces) ' + ' '.join(fields)


class Accounting(unittest.TestCase):
    def test_short_suite_is_two_sessions_in_both_wrappers(self):
        for mode in ['offscreen', 'present']:
            self.assertEqual(relay.suite_labels(mode, False), ['mprotect-1', 'present-fifo-cold'])
            self.assertGreater(len(relay.suite_labels(mode, True)), 2)

    def test_stat_comm_and_pid_reuse(self):
        self.assertEqual(relay.task_stat(stat(10, 20, 99)), ('S', 10, 20, 99))
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp)
            task = proc/'task'/'12'
            task.mkdir(parents=True)
            (task/'comm').write_text('BPE JobWorkerTh')
            (task/'wchan').write_text('__futex_wait')
            (task/'schedstat').write_text('1000000000 2000000000 3')
            (task/'status').write_text('voluntary_ctxt_switches: 5\nnonvoluntary_ctxt_switches: 6\n')
            output = io.StringIO()
            sampler = relay.ThreadSampler(output)
            (task/'stat').write_text(stat(10, 20, 99))
            sampler.sample(proc, 5, 'gameplay-measure')
            (task/'stat').write_text(stat(20, 30, 99))
            sampler.sample(proc, 10, 'gameplay-measure')
            (task/'stat').write_text(stat(0, 0, 100))  # Reused tid: no negative CPU delta.
            sampler.sample(proc, 15, 'gameplay-measure')
            self.assertEqual(len(output.getvalue().splitlines()), 2)
            self.assertAlmostEqual(sum(next(iter(sampler.totals.values()))[:2]), 20/sampler.ticks)
            sampler.sample(proc, 20, 'capture')  # No mixed-phase interval.
            self.assertEqual(len(output.getvalue().splitlines()), 2)

    def test_measurement_excludes_capture_and_missing_gameplay(self):
        text = '''[1000.000][x] I: Linux heartbeat: flips=1 recovered_faults=1 direct_faults=1
[2000.000][x] I: relay step: gameplay-measure-begin
[3000.000][x] I: Linux heartbeat: flips=10 recovered_faults=100 direct_faults=90
[8000.000][x] I: Linux heartbeat: flips=50 recovered_faults=200 direct_faults=180
[9000.000][x] I: relay step: gameplay-measure-end
[10000.000][x] I: Linux heartbeat: flips=500 recovered_faults=999 direct_faults=999
'''
        metrics = relay.log_metrics(text)
        self.assertEqual(metrics['flips_per_second'], 8)
        self.assertEqual(metrics['recovered_faults_per_second'], 20)
        self.assertNotIn('flips_per_second', relay.log_metrics('[1000.0][x] I: Linux heartbeat: flips=10'))

    def test_clean_profile_cannot_inherit_diagnostics(self):
        inherited = {k: '1' for k in ['DS_NAN_EARLY', 'DS_NAN_PROBE', 'DS_PROFILE_GPU', 'DS_BATCH_STAMPS', 'DS_SCREENSHOT', 'DS_LINUX_INTERVAL_SHOTS']}
        clean = relay.session_environment(inherited, 'clean')
        for k in inherited: self.assertEqual(clean[k], '0', k)
        self.assertEqual(clean['DS_RELAY_GAMEPLAY_SECONDS'], '60')
        diagnostic = relay.session_environment(clean, 'diagnostic')
        self.assertEqual(diagnostic['DS_NAN_EARLY'], '1')
        self.assertEqual(diagnostic['DS_RELAY_GAMEPLAY_SECONDS'], '0')
        self.assertEqual(diagnostic['DS_PROFILE_GPU'], '0')
        self.assertEqual(clean['DS_NAN_EARLY'], '0')  # Profiles do not mutate one another.
        good = dict(effective_validation=False, exit_code=0, validation_errors=0, crash_bytes=0, flips_per_second=10,
                    timelines=[{'frames': 15}], nan_probe={'checked_targets': 0, 'early_gameplay_targets': 0})
        self.assertFalse(relay.session_failed(good, 'clean'))
        self.assertTrue(relay.session_failed({**good, 'thermal': {'measurement_valid': False}}, 'clean'))
        self.assertTrue(relay.session_failed({**good, 'effective_validation': True}, 'clean'))
        self.assertTrue(relay.session_failed({**good, 'effective_validation': None}, 'clean'))
        self.assertTrue(relay.session_failed(good, 'diagnostic'))
        good['nan_probe'] = {'checked_targets': 1, 'early_gameplay_targets': 1}
        self.assertTrue(relay.session_failed(good, 'clean'))
        self.assertFalse(relay.session_failed(good, 'diagnostic'))

    def test_gpu_timing_is_separate_from_clean_and_nan_probes(self):
        clean = relay.session_environment({}, 'clean')
        timed = relay.session_environment(clean, 'gpu-timing')
        self.assertEqual(timed['DS_PROFILE_GPU'], '1')
        self.assertEqual(timed['DS_PROFILE_GPU_STAMPS'], 'seg')
        self.assertEqual(timed['DS_BATCH_STAMPS'], '1')
        self.assertEqual(timed['DS_NAN_EARLY'], '0')
        self.assertEqual(timed['DS_NAN_PROBE'], '0')
        self.assertEqual(timed['DS_VK_VALIDATION'], '0')
        self.assertEqual(clean['DS_PROFILE_GPU'], '0')
        # Regression: run_session intentionally strips rates from instrumented runs.
        timed_result = dict(effective_validation=False, exit_code=0, validation_errors=0, crash_bytes=0,
                            timelines=[{'frames': 15, 'gpu_timestamp_spans': 200}], nan_probe={'checked_targets': 0})
        self.assertFalse(relay.session_failed(timed_result, 'gpu-timing'))
        self.assertTrue(relay.session_failed(timed_result, 'clean'))
        timed_result['timelines'][0]['gpu_timestamp_spans'] = 0
        self.assertTrue(relay.session_failed(timed_result, 'gpu-timing'))

    def test_tracking_comparison_changes_only_policy(self):
        clean = relay.session_environment({}, 'clean')
        legacy = relay.session_environment({}, 'legacy')
        self.assertEqual(relay.session_environment({}, 'warm'), clean)
        self.assertEqual({k for k in clean if clean[k] != legacy[k]}, {'DS_LINUX_VOLATILE_BACKOFF', 'DS_LINUX_WRITE_WATCH'})
        self.assertEqual((clean['DS_LINUX_VOLATILE_BACKOFF'], legacy['DS_LINUX_VOLATILE_BACKOFF']), ('0', '1'))

    def test_repeated_tracking_profiles_and_present_modes(self):
        clean = relay.session_environment({}, 'clean-1')
        self.assertEqual(relay.session_environment({}, 'clean-2'), clean)
        self.assertEqual(relay.session_environment({}, 'warm-1'), clean)
        old = relay.session_environment({}, 'mprotect-1')
        self.assertEqual({k for k in clean if clean[k] != old[k]}, {'DS_LINUX_WRITE_WATCH'})
        for mode in ('fifo', 'mailbox'):
            env = relay.session_environment({}, 'present-'+mode)
            self.assertEqual(env['DS_LINUX_PRESENT_MODE'], mode)
            self.assertEqual(env['DS_RELAY_GAMEPLAY_SECONDS'], '60')
            self.assertEqual(env['DS_RELAY_CLOSE_WINDOW'], '1')
            self.assertEqual(env['DS_WINDOW_SHOT'], '0')
        failed = dict(exit_code=0, validation_errors=0, crash_bytes=0,
                      write_watch_requested='uffd', write_watch='mprotect')
        self.assertTrue(relay.session_failed(failed, 'clean-1'))

    def test_presentation_percentiles_keep_stalls_and_completion_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'user/gameplay-presentation.csv'
            path.parent.mkdir()
            path.write_text('kind,id,monotonic_ns,slot_ms,acquire_ms,record_submit_ms,gpu_ready_ms,capture_queue_ms,present_ms\n'
                            'submit,1,1000000000,1,2,3,4,5,6\n'
                            'submit,2,1010000000,1,2,3,4,5,6\n'
                            'wait_complete,1,1000000000,0,0,0,0,0,0\n'
                            'wait_complete,2,1016666667,0,0,0,0,0,0\n'
                            'wait_complete,5,1100000000,0,0,0,0,0,0\n')
            r = relay.present_times.analyze(root)
            self.assertEqual(r['submit']['mean_ms'], 10)
            self.assertFalse(r['submit']['physical_display_timestamps'])
            self.assertEqual(r['wait_complete']['samples'], 2)
            self.assertEqual(r['wait_complete']['nonconsecutive_ids'], 1)
            self.assertAlmostEqual(r['wait_complete']['p99_ms'], 83.333333)
            self.assertNotIn('displayed', r)
            self.assertEqual(r['pacing'], r['wait_complete'])
            self.assertFalse(r['target']['p99_met'])
            self.assertFalse(relay.present_times.pacing_target({'samples': 0})['measured'])
            self.assertEqual(r['stages']['gpu_ready_ms']['mean_ms'], 4)

    def test_present_aggregation_and_diagnostic_separation(self):
        p = {'pacing': {'samples': 60, 'p99_ms': 25}, 'movement-2': {'pacing': {'samples': 60, 'p99_ms': 30}}}
        metrics = dict(measurement_valid=True, flips_per_second=50, presentation=p)
        r = relay.comparisons_for({'present-fifo-warm': metrics, 'present-validation': dict(metrics, measurement_valid=False)}, 'present')
        self.assertEqual(list(r), ['present-fifo-warm'])
        self.assertEqual(r['present-fifo-warm']['stationary']['p99_ms'], 25)
        self.assertEqual(r['present-fifo-warm']['movement-2']['pacing']['p99_ms'], 30)
        for label in ('present-fifo-cold', 'present-fifo-warm', 'present-mailbox-cold', 'present-mailbox-warm'):
            e = relay.session_environment({'VK_INSTANCE_LAYERS':'VK_LAYER_KHRONOS_validation'}, label)
            self.assertEqual(e['DS_VK_VALIDATION'], '0')
            self.assertNotIn('VK_INSTANCE_LAYERS', e)
            self.assertEqual(e['DS_PROFILE_GPU'], '0')
            self.assertEqual(e['DS_BATCH_STAMPS'], '0')
            self.assertEqual(e['DS_LINUX_WRITE_WATCH'], 'mprotect')
        self.assertEqual(relay.session_environment({}, 'present-validation')['DS_VK_VALIDATION'], '1')
        old = relay.session_environment({}, 'sparse-original')
        new = relay.session_environment({}, 'mprotect-1')
        self.assertEqual({k for k in old if old[k] != new[k]}, {'DS_LINUX_COALESCE_SPARSE'})

    def test_pipeline_reuse_uses_actual_hits_and_interval_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user = root/'user'
            user.mkdir()
            report = dict(binary_mode=True, binary_hits=98, binary_refused=1, missing=1, invalid=0, other_driver=0, other_code=0,
                          captured=2, source_create_calls=2, source_create_ms=15., binary_total_ms=3., identity=['gpu', 'uuid'])
            (user/'pipeline-reuse.json').write_text(json.dumps(report))
            (user/'movement-2-pipeline-reuse.json').write_text(json.dumps({'begin':report,'end':report}))
            out = relay.pipeline_reuse(root)
            self.assertEqual(out['binary_hit_fraction'], .98)
            self.assertTrue(out['reuse_verified'])
            self.assertFalse(relay.reuse_verified({**out, 'binary_hit_fraction':.8}))
            self.assertFalse(relay.reuse_verified({**out, 'binary_mode':False}))
            self.assertEqual(out['intervals']['movement-2']['source_create_ms'], 0)
            (user/'pipeline-reuse.json').write_text('{')
            self.assertFalse(relay.pipeline_reuse(root)['available'])
        self.assertEqual(relay.session_environment({}, 'present-fifo-uffd')['DS_LINUX_WRITE_WATCH'], 'uffd')
        diag = relay.session_environment({}, 'present-fifo-gpu-timing')
        self.assertEqual(diag['DS_PROFILE_GPU'], '1')
        self.assertEqual(diag['DS_VK_VALIDATION'], '0')
        self.assertEqual(diag['DS_RELAY_CLOSE_WINDOW'], '1')
        self.assertEqual(diag['DS_LINUX_PRESENT_MODE'], 'fifo')

    def test_slow_frame_attribution_retains_upstream_gaps(self):
        def row(kind, id, ns, acquire=0):
            return dict(kind=kind,id=str(id),monotonic_ns=str(ns),slot_ms='0',acquire_ms=str(acquire),
                        record_submit_ms='1',gpu_ready_ms='0',capture_queue_ms='0',present_ms='0')
        rows = [row('submit',1,1_000_000_000), row('submit',2,1_033_000_000,2),
                row('wait_complete',1,1_005_000_000), row('wait_complete',2,1_038_000_000)]
        slow = relay.present_times.slow_frames(rows)
        self.assertEqual(slow[0]['completion_ms'], 33)
        self.assertEqual(slow[0]['outside_current_present_ms'], 30)
        self.assertTrue(slow[0]['consecutive'])

    def test_traffic_is_measured_inside_stationary_window(self):
        text = '''[1000.0][x] I: relay step: gameplay-measure-begin
[2000.0][x] I: Linux heartbeat: flips=10 upload_bytes=8192 writeback_bytes=4096 unarmable_pages=12
[7000.0][x] I: Linux heartbeat: flips=20 upload_bytes=49152 writeback_bytes=8192 unarmable_pages=22
[8000.0][x] I: relay step: gameplay-measure-end
'''
        metrics = relay.log_metrics(text)
        self.assertEqual(metrics['upload_bytes_per_flip'], 4096)
        self.assertEqual(metrics['upload_bytes_per_second'], 8192)
        self.assertEqual(metrics['unarmable_pages_per_flip'], 1)

    def test_validation_is_verified_from_device_log(self):
        self.assertIsNone(relay.effective_validation('no device initialized'))
        self.assertIs(relay.effective_validation('Vulkan device: NVIDIA (validation true; driver 1)'), True)
        self.assertIs(relay.effective_validation('Vulkan device: NVIDIA (validation false; driver 1)'), False)
        env = relay.session_environment({'VK_INSTANCE_LAYERS': 'VK_LAYER_KHRONOS_validation'}, 'clean')
        self.assertNotIn('VK_INSTANCE_LAYERS', env)
        self.assertEqual(env['DS_RECORD_PROFILE'], '0')

    def test_timeout_collects_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            binary = repo/'target/release/dsport'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\nsleep 30\n')
            binary.chmod(0o755)
            # This sleeping shell fixture has no thermal load; sensor behavior is tested separately.
            with patch.object(relay.linux_guard, 'temperature', return_value=50):
                metrics = relay.run_session(repo, repo/'unused-game', repo/'run', 'offscreen', '1', '1', repo/'cache')
            self.assertEqual(metrics['exit_code'], 124)
            self.assertEqual((repo/'run/exit-code.txt').read_text(), '124\n')
            self.assertTrue((repo/'run/thread-summary.txt').exists())


class TraceAccounting(unittest.TestCase):
    def test_upload_prologue_is_not_hidden_in_batch_coverage(self):
        self.assertEqual(linux_waits.classify('GPU', 'gpu-gfx #123 prologue'), 'gpu_upload_prologue')
        self.assertEqual(linux_waits.classify('GPU', 'gpu-gfx #123'), 'gpu_batch_span')
        coverage = linux_waits.partition([(0, 100, 'gpu_batch_span'), (0, 70, 'gpu_upload_prologue')], 0, 100)
        self.assertEqual(dict(coverage), {'gpu_upload_prologue': 70, 'gpu_batch_span': 30})

    def test_slot_wait_initial_state_and_releaser_phases(self):
        for state in ('submit', 'gpu', 'release'):
            self.assertEqual(linux_waits.classify('gpu-gfx', f'wait: batch slot (#12, initial {state}, 2 slots, compute false)'), 'batch_slot_'+state)
        self.assertEqual(linux_waits.classify('gpu-release', 'wait: GPU completion #12 owner gfx'), 'gpu_completion')
        self.assertEqual(linux_waits.classify('gpu-release', 'wait: submission #12'), 'submission_wait')
        self.assertEqual(linux_waits.classify('gpu-release', 'release pool reset #12'), 'release_reset')

    def test_nested_and_cross_frame_spans_are_not_double_counted(self):
        spans = [(0, 100, 'record_or_other'), (10, 70, 'gpu_completion'),
                 (11, 69, 'gpu_completion'), (50, 90, 'batch_slot')]
        c = linux_waits.partition(spans, 20, 80)
        self.assertEqual(dict(c), {'gpu_completion': 30, 'batch_slot': 30})
        c = linux_waits.partition(spans, 80, 120)
        self.assertEqual(dict(c), {'batch_slot': 10, 'record_or_other': 10, 'untraced': 20})
        self.assertEqual(sum(c.values()), 40)

    def test_separate_lanes_and_missing_flips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root/'trace.json'
            events = [{'ph': 'M', 'name': 'thread_name', 'tid': 1, 'args': {'name': 'gpu-gfx'}},
                      {'ph': 'M', 'name': 'thread_name', 'tid': 2, 'args': {'name': 'GPU'}},
                      {'ph': 'i', 'name': 'flip 1', 'ts': 0}, {'ph': 'i', 'name': 'flip 2', 'ts': 1000},
                      {'ph': 'X', 'name': 'wait: batch #1', 'tid': 1, 'ts': 0, 'dur': 1000},
                      {'ph': 'X', 'name': 'gpu-gfx #1', 'tid': 2, 'ts': 0, 'dur': 1000}]
            trace.write_text(json.dumps({'traceEvents': events}))
            self.assertEqual(linux_waits.analyze(trace, root/'out')['mean_frame_ms'], 1)
            self.assertEqual(len((root/'out/frame-waits.csv').read_text().splitlines()), 3)
            trace.write_text(json.dumps({'traceEvents': []}))
            self.assertEqual(linux_waits.analyze(trace, root/'out')['frames'], 0)

    def test_nan_origin_summary_keeps_inherited_and_transitions_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'user/nan-12/passes.jsonl'
            path.parent.mkdir(parents=True)
            common = {'id': 1, 'queue': 'gpu-gfx', 'kind': 'draw', 'packet': '0x100',
                      'shaders': ['0x200'], 'actual': 'key', 'requested': 'view', 'before': None,
                      'after': {'nan_components': 2}}
            records = [dict(common, classification='ALREADY_NAN_OR_UNKNOWN_BEFORE'),
                       dict(common, id=2, classification='FINITE_TO_NAN')]
            path.write_text('\n'.join(map(json.dumps, records))+'\n')
            result = linux_nan_summary.summarize(root)
            self.assertEqual((result['finite_to_nan'], result['already_or_unknown']), (1, 1))
            self.assertIn('FINITE_TO_NAN event 2', (root/'nan-summary.txt').read_text())

    def test_early_summary_requires_gameplay_and_tolerates_missing_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'user/nan-0/passes.jsonl'
            path.parent.mkdir(parents=True)
            common = dict(id=1, early=True, phase=0, queue='gpu-gfx', kind='dispatch',
                          packet='0x100', shaders=['0x200'], actual='key', requested='view',
                          before=None, after={'nan_components': 0, 'zero_components': 9}, classification='finite')
            records = [common, dict(common, id=2, phase=2),
                       dict(id=3, early=True, phase=2, error='no cached image')]
            path.write_text('\n'.join(map(json.dumps, records))+'\n')
            result = linux_nan_summary.summarize(root)
            self.assertEqual(result['checked_targets'], 2)
            self.assertEqual(result['early_gameplay_targets'], 1)
            self.assertEqual(result['errors'], 1)
            self.assertIn('phase 2 early True', (root/'nan-summary.txt').read_text())


class PublicConfiguration(unittest.TestCase):
    def test_game_path_required_and_overridable(self):
        import os, subprocess
        env = {k: v for k, v in os.environ.items() if not k.startswith(('DS_', 'DSPORT_'))}
        script = str(Path(__file__).with_name('linux-gpu-test.sh'))
        no_path = subprocess.run([script, '--dry-run'], env=env, capture_output=True, text=True)
        self.assertEqual(no_path.returncode, 2)
        self.assertIn('DS_GAME_DIR', no_path.stderr)
        for args, extra in [(['--app', '/path/to/game', '--dry-run'], {}),
                            (['/path/to/game', '--dry-run'], {}),
                            (['--present', '--dry-run'], {'DS_GAME_DIR': '/path/to/game'})]:
            out = subprocess.run([script]+args, env=dict(env, **extra), capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('/path/to/game', out.stdout)

    def test_readonly_root_rejected_before_output(self):
        import os, subprocess
        script = Path(__file__).with_name('linux-gpu-test.sh').resolve()
        repo = script.parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, DS_GAME_DIR=tmp, DS_READONLY_ROOTS=str(repo))
            result = subprocess.run([str(script)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Unsafe relay output', result.stderr)


if __name__ == '__main__': unittest.main()
