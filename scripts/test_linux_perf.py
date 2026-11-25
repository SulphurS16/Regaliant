import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import linux_guard
import linux_perf


class Sampling(unittest.TestCase):
    def test_select_exact_gfx_and_reject_missing_or_ambiguous(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for tid,name in [(11,'gpu-gfx-par0'),(12,'gpu-gfx')]:
                q=p/'task'/str(tid);q.mkdir(parents=True);(q/'comm').write_text(name+'\n')
            self.assertEqual(linux_perf.gfx_tid(p),12)
            (p/'task/11/comm').write_text('gpu-gfx\n')
            with self.assertRaises(ValueError): linux_perf.gfx_tid(p)
            (p/'task/12/comm').write_text('other\n');(p/'task/11/comm').write_text('other\n')
            with self.assertRaises(ValueError): linux_perf.gfx_tid(p)

    def test_bounded_thread_only_command_and_top_self(self):
        argv=linux_perf.command(123,Path('/tmp/example'))
        self.assertEqual(argv[0],'timeout')
        self.assertEqual(argv[argv.index('-t')+1],'123')
        self.assertNotIn('-p',argv)
        self.assertIn('dwarf,16384',argv)
        r=linux_perf.top_self('# overhead;symbol\n 1.20%;[.] slow\n24.30%;[.] fast::function\n')
        self.assertEqual(r[0]['self_percent'],24.3)
        self.assertEqual(r[1]['symbol'],'[.] slow')

    def test_fake_perf_capture_report_and_missing_tool(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);fake=p/'perf'
            fake.write_text('#!/usr/bin/env python3\nimport sys,pathlib\nif sys.argv[1]=="record": pathlib.Path(sys.argv[sys.argv.index("-o")+1]).write_bytes(b"fixture")\nelse: print("50.00%;[.] function_a\\n25.00%;[.] function_b")\n')
            fake.chmod(0o755)
            with patch.dict(os.environ,PATH=str(p)+os.pathsep+os.environ['PATH']),patch.object(linux_perf,'gfx_tid',return_value=123):
                linux_perf.preflight(p)
                sample=linux_perf.GfxSample(os.getpid(),p)
                sample.process.wait(timeout=5)
                r=sample.finish()
                self.assertTrue(r['valid'])
                self.assertEqual(len(r['top20_self']),2)
                self.assertIn('function_a',(p/'gpu-gfx-top20.txt').read_text())
                self.assertEqual(json.loads((p/'gpu-gfx-perf-command.json').read_text())['tid'],123)
            with patch.object(linux_perf.shutil,'which',return_value=None):
                with self.assertRaisesRegex(ValueError,'requires perf'): linux_perf.preflight(p)

    def test_perf_refusal_is_not_reported_as_a_valid_sample(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);fake=p/'perf'
            fake.write_text('#!/usr/bin/env python3\nimport sys\nprint("permission denied")\nsys.exit(2)\n')
            fake.chmod(0o755)
            with patch.dict(os.environ,PATH=str(p)+os.pathsep+os.environ['PATH']),patch.object(linux_perf,'gfx_tid',return_value=123):
                with self.assertRaisesRegex(ValueError,'denied/unavailable'): linux_perf.preflight(p)
                sample=linux_perf.GfxSample(os.getpid(),p)
                sample.process.wait(timeout=5)
                result=sample.finish()
                self.assertFalse(result['valid'])
                self.assertEqual(result['exit_code'],2)
                self.assertIn('FAILED/INCOMPLETE',(p/'gpu-gfx-top20.txt').read_text())

    def test_shared_lock_refuses_a_second_holder_without_editing_note(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cpu-heavy.lock';p.write_text('owner note\n')
            with linux_guard.cpu_heavy_lock(p):
                with self.assertRaisesRegex(linux_guard.GuardError,'lock is held'):
                    with linux_guard.cpu_heavy_lock(p): self.fail('second lock acquired')
            with linux_guard.cpu_heavy_lock(p): pass
            self.assertEqual(p.read_text(),'owner note\n')


if __name__=='__main__': unittest.main()
