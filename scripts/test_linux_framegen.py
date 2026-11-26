import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

fg=module('linux-framegen-test')
times=module('linux-present-times')


class FrameGeneration(unittest.TestCase):
    def test_relay_keeps_unsafe_recording_and_capture_out_of_clean(self):
        for on in (False,True):
            e=fg.steady.relay.session_environment(fg.environment(on),'steady-fg-on-uffd')
            self.assertEqual(e['DS_FRAME_GENERATION'],'fsr3' if on else 'off')
            self.assertEqual(e['DS_LINUX_WRITE_WATCH'],'uffd')
            for key in ('DS_FG_SHOTS','DS_FG_INPUT_DUMP','DS_PAR_WHOLE_RUN','DS_RECORD_TSC','DS_RECORD_STATIC_BIND','DS_VK_VALIDATION','DS_PROFILE_GPU'):
                self.assertEqual(e[key],'0')
            self.assertEqual(e['SDL_AUDIO_DRIVER'],'dummy')
        e=fg.steady.relay.session_environment(fg.environment(True,True),'steady-fg-validation-uffd')
        self.assertEqual(e['DS_VK_VALIDATION'],'1')
        self.assertEqual(e['DS_FG_SHOTS'],'1')
        self.assertEqual(e['DS_RELAY_STEADY_DIAGNOSTIC'],'1')

    def test_missing_input_and_native_create_diagnostics_are_collected(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir()
            (run/'user/dsport.log').write_text('FG TAA scheduled [1]: 3840x2160\nFG HUDless deferred [1]: no TAA inputs yet\n')
            (run/'console.log').write_text('FSR3 bridge: begin optical flow context\nFSR3 bridge: end optical flow context, code 5\n')
            (run/'user/framegen-state.json').write_text('{"stages":{"TAA scheduled":1},"generated":0}')
            r=fg.diagnostics(run)
            self.assertEqual(r['states'][0]['stages']['TAA scheduled'],1)
            self.assertIn('no TAA inputs yet',(run/'framegen-stages.txt').read_text())
            self.assertTrue(any('code 5' in line for line in r['last_events']))

    def test_diagnostic_requires_previous_colour_pair_and_real_files(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);folder=run/'user/fg-inputs-1';folder.mkdir(parents=True)
            names=['hudless','real','generated','depth','motion','previous-real','previous-hudless']
            (folder/'inputs.json').write_text(json.dumps({'images':[{'name':n} for n in names]}))
            self.assertFalse(fg.input_pairs(run)[0]['complete'])
            for name in names:
                for suffix in ('.raw','.png'): (folder/(name+suffix)).touch()
            self.assertTrue(fg.input_pairs(run)[0]['complete'])
            (folder/'previous-real.raw').unlink()
            self.assertFalse(fg.input_pairs(run)[0]['complete'])

    def test_new_dump_requires_raw_motion_and_recovery_mask(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);folder=run/'user/fg-inputs-1';folder.mkdir(parents=True)
            names=['hudless','real','generated','depth','motion','previous-real','previous-hudless','motion-raw','motion-status']
            (folder/'inputs.json').write_text(json.dumps({'schema':2,'images':[{'name':n} for n in names]}))
            for name in names:
                (folder/(name+'.raw')).touch()
                if name!='motion-status': (folder/(name+'.png')).touch()
            self.assertTrue(fg.input_pairs(run)[0]['complete'])
            (folder/'motion-raw.raw').unlink()
            self.assertFalse(fg.input_pairs(run)[0]['complete'])

    def test_completed_ids_separate_presented_rate_from_real_rate(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);user=run/'user';user.mkdir()
            fields=['kind','id','monotonic_ns','slot_ms','acquire_ms','record_submit_ms','gpu_ready_ms','capture_queue_ms','present_ms','generated','fg_snapshot_age_ms','fg_compute_completion_ms']
            with (user/'stationary-presentation.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for i in range(1,7):
                    for kind in ('submit','wait_complete'):
                        row=dict.fromkeys(fields,0)
                        row.update(kind=kind,id=i,monotonic_ns=i*16_666_667+(1_000_000 if kind=='wait_complete' else 0),generated=str(i%2==1).lower(),fg_snapshot_age_ms=20,fg_compute_completion_ms=3)
                        w.writerow(row)
            r=times.analyze(run)['stationary']
            self.assertAlmostEqual(r['wait_complete']['presented_fps'],60,places=4)
            self.assertEqual(r['frame_generation']['generated_completions'],3)
            self.assertEqual(r['frame_generation']['real_completions'],3)
            self.assertEqual(r['frame_generation']['compute_and_completion']['samples'],3)
            self.assertEqual(r['enqueue_to_completion']['mean_ms'],1)

    def test_missing_completion_is_not_claimed_as_presented_fps(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir()
            (run/'user/stationary-presentation.csv').write_text('kind,id,monotonic_ns,slot_ms,acquire_ms,record_submit_ms,gpu_ready_ms,capture_queue_ms,present_ms,generated,fg_snapshot_age_ms,fg_compute_completion_ms\nsubmit,1,1,0,0,0,0,0,0,true,0,0\n')
            r=times.analyze(run)['stationary']
            self.assertNotIn('presented_fps',r['wait_complete'])
            self.assertEqual(r['frame_generation']['generated_completions'],0)
            self.assertFalse(r['target']['measured'])

    def test_invalid_or_warming_fps_is_not_reported_as_measurement(self):
        row=dict(invalid_reasons=['walking compiled pipelines inside measurement'],diagnostic=False,valid_steady_state=False,phases={'walking':dict(flips_per_second=999,presentation={})})
        self.assertNotIn('999',fg.summarize({'test':row}))
        row.update(invalid_reasons=[],valid_steady_state=True,warmup_only=True)
        self.assertNotIn('999',fg.summarize({'test':row}))
