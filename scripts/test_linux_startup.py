import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('startup', Path(__file__).with_name('linux-startup-test.py'))
startup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(startup)


class Startup(unittest.TestCase):
    def test_missing_or_failed_compiles_are_not_a_fixed_crash(self):
        keys = ['c0000000000000001', 'c0000000000000002']
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            run = session/'user'/'linux-runs'/'one'
            run.mkdir(parents=True)
            report = run/'pipeline-reuse.json'
            report.write_text(json.dumps({'ledger': [{'key': keys[0], 'outcome': 'source_failed'},
                                                    {'key': keys[1], 'outcome': 'hit'}]}))
            rows = [{'expected_compiles': startup.compile_outcomes(session, keys)}]
            self.assertEqual(startup.missing_compiles(rows, keys), [keys[0]])
            rows.append({'expected_compiles': {keys[0]: ['source', 'captured']}})
            self.assertEqual(startup.missing_compiles(rows, keys), [])
            self.assertEqual(startup.missing_compiles([{'failed': True}], keys), keys)


if __name__ == '__main__':
    unittest.main()
