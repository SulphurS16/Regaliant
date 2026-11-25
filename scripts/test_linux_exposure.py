import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('exposure',Path(__file__).with_name('linux-exposure-check.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


class ExposureTests(unittest.TestCase):
    def test_overrides_cannot_reenable_fg_or_redirect_output(self):
        self.assertEqual(e.overrides(['DS_LINUX_WRITE_WATCH=mprotect','DS_NAN_PRESERVE=0']),
                         {'DS_LINUX_WRITE_WATCH':'mprotect','DS_NAN_PRESERVE':'0'})
        for value in ['DSPORT_HOME=/game','DS_FRAME_GENERATION=fsr3','DS_SCREENSHOT=1','DS_LINUX_WRITE_WATCH=no']:
            with self.assertRaises(ValueError):e.overrides([value])
        self.assertEqual(e.overrides(['DS_META_CLAIMS=2','DS_META_CLAIM_REBIND=1']),
                         {'DS_META_CLAIMS':'2','DS_META_CLAIM_REBIND':'1'})
        with self.assertRaises(ValueError):e.overrides(['DS_META_CLAIMS=4'])

    def test_inherited_experiments_and_driver_paths_are_isolated(self):
        env=e.environment({'PATH':'/bin','DS_NAN_PROBE':'1','DS_FRAME_GENERATION':'fsr3',
            'DS_READONLY_ROOTS':'/protected','DS_RELAY_CPUS':'0-3','__GL_SHADER_DISK_CACHE_PATH':'/player'},
            Path('/output'),Path('/game'),[3600,4200],{'DS_LINUX_WRITE_WATCH':'mprotect'},False)
        self.assertEqual(env['DS_FRAME_GENERATION'],'off')
        self.assertEqual(env['DS_LINUX_WRITE_WATCH'],'mprotect')
        self.assertEqual(env['DS_SCREENSHOT'],'3600,4200')
        self.assertEqual(env['__GL_SHADER_DISK_CACHE_PATH'],'/output/driver/nvidia')
        self.assertNotIn('DS_TAP_CS',env)
        self.assertEqual(env['DS_READONLY_ROOTS'],'/protected')
        self.assertEqual(env['DS_RELAY_CPUS'],'0-3')

    def test_settings_keep_visual_choices_but_block_save_import_and_fg(self):
        text=e.settings('Setting_PC_Bloom=0\nSetting_PC_ImportSaves=1\nSetting_PC_SaveFolder=2\nSetting_PC_FrameGeneration=1\nSetting_PC_AntiAliasing=3\n')
        self.assertIn('Setting_PC_Bloom=0',text)
        self.assertNotIn('Setting_PC_ImportSaves=',text)
        self.assertNotIn('Setting_PC_SaveFolder=',text)
        self.assertIn('Setting_PC_AntiAliasing=0',text)
        self.assertIn('Setting_PC_FrameGeneration=0',text)

    def test_readiness_uses_first_stationary_heartbeat_not_loading_hud(self):
        text='relay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(768)\nLinux heartbeat: 60s flips=1800\n'
        self.assertIsNone(e.evidence(text)['begin_upper_flip'])
        text+='relay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 65s flips=1950\nLinux heartbeat: 70s flips=2100\n'
        self.assertEqual(e.evidence(text)['begin_upper_flip'],1950)
        self.assertTrue(e.evidence(text+'relay step: gameplay-measure-end;')['ended'])

    def test_tail_preserves_partial_lines_and_incremental_phase(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'linux-runs/session';folder.mkdir(parents=True)
            log=folder/'dsport.log';log.write_text('relay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 90s flips=')
            tail=e.LogTail(root);state=e.evidence(tail.read())
            self.assertIsNone(state['begin_upper_flip'])
            with log.open('a') as f:f.write('3000\n')
            state=e.evidence(tail.read(),state)
            self.assertEqual(state['begin_upper_flip'],3000)
            self.assertEqual(tail.read(),'')

    def test_taps_decode_bit_patterns_without_reporting_nan_as_json_number(self):
        text='tap cs 0x100893aa00 res 0 for flip 3600: gpu [#3f800000 2] guest [#7fc00000 0]\n'
        self.assertEqual(e.evidence(text)['exposure'],[{'flip':3600,'gpu':1.0,'guest':None}])
        self.assertEqual(e.float_word('#00000000'),0.0)
        self.assertIsNone(e.float_word('#deadbeef'))

    def test_image_statistics_ignore_alpha_and_report_fixed_centre(self):
        from PIL import Image
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'sample.png'
            a=np.zeros((8,8,4),dtype=np.uint8);a[2:6,2:6,:3]=255
            Image.fromarray(a).save(path)
            r=e.brightness(path)
            self.assertAlmostEqual(r['full']['mean_rgb'],.25)
            self.assertAlmostEqual(r['centre']['mean_rgb'],1.)
            self.assertAlmostEqual(r['full']['mean_luminance_linear'],.25)
            self.assertEqual(r['full']['near_white_fraction'],.25)

    def test_report_rejects_fallback_missing_early_and_post_stationary_samples(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir();shot=run/'sample.png';shot.write_bytes(b'fixture')
            text=f'Linux write watch: mprotect\nrelay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 90s flips=3700\nscreenshot of flip 3600 (buffer 0x100): {shot}\nrelay step: gameplay-measure-end;\n'
            with patch.object(e,'brightness',return_value=dict(width=3840,height=2160)):
                result=e.report(run,[3600,3900],{'DS_LINUX_WRITE_WATCH':'uffd'},0,text,'done')
            self.assertFalse(result['valid'])
            self.assertTrue(any('tracker' in f for f in result['failures']))
            self.assertTrue(any('before confirmed' in f for f in result['failures']))
            self.assertTrue(any('missing capture' in f for f in result['failures']))
            self.assertTrue(any('stationary window' in f for f in result['failures']))
            json.dumps(result,allow_nan=False)

    def test_valid_report_needs_real_capture_and_backend_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir();shot=run/'sample.png';shot.write_bytes(b'fixture')
            text=f'Linux write watch: uffd async WP + PAGEMAP_SCAN\nrelay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 90s flips=3000\nscreenshot of flip 3600 (buffer 0x100): {shot}\n'
            with patch.object(e,'brightness',return_value=dict(width=3840,height=2160)):
                result=e.report(run,[3600],{'DS_LINUX_WRITE_WATCH':'uffd'},0,text,'done')
            self.assertTrue(result['valid'],result['failures'])

    def test_capture_stop_75_requires_shutdown_and_complete_valid_samples(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir();shot=run/'sample.png';shot.write_bytes(b'fixture')
            text=f'Linux write watch: uffd async WP + PAGEMAP_SCAN\nrelay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 90s flips=3000\nscreenshot of flip 3600 (buffer 0x100): {shot}\n'
            end='Linux shutdown complete: status 75; process-lifetime workers end together\n'
            env={'DS_LINUX_WRITE_WATCH':'uffd'}
            with patch.object(e,'brightness',return_value=dict(width=3840,height=2160)):
                self.assertTrue(e.report(run,[3600],env,75,text+end,'fixed captures complete')['valid'])
                for code,log,reason,flips in [(75,text,'fixed captures complete',[3600]),
                    (75,text+end,'runtime ended before fixed captures',[3600]),
                    (75,text+end,'fixed captures complete',[3600,3900]),
                    (-15,text+end,'fixed captures complete',[3600])]:
                    self.assertFalse(e.report(run,flips,env,code,log,reason)['valid'])
                (run/'thermal.json').write_text(json.dumps({'aborted':'thermal abort'}))
                self.assertFalse(e.report(run,[3600],env,75,text+end,'fixed captures complete')['valid'])

    def test_metadata_switch_requires_matching_runtime_policy(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'user').mkdir();shot=run/'sample.png';shot.write_bytes(b'fixture')
            text=f'Linux write watch: uffd\nrelay step: gameplay-measure-begin; menu stack Some(0) scene(s), HUD bytes Some(880)\nLinux heartbeat: 90s flips=3000\nscreenshot of flip 3600 (buffer 0x100): {shot}\n'
            env={'DS_LINUX_WRITE_WATCH':'uffd','DS_META_CLAIMS':'1','DS_META_CLAIM_REBIND':'1'}
            with patch.object(e,'brightness',return_value=dict(width=3840,height=2160)):
                self.assertFalse(e.report(run,[3600],env,0,text,'done')['valid'])
                text+='images: metadata claim mask 1 (1 colour, 2 depth; 3 upstream default); claim rebind invalidation true\n'
                self.assertTrue(e.report(run,[3600],env,0,text,'done')['valid'])


if __name__=='__main__':unittest.main()
