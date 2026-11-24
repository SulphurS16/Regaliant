import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('reuse', Path(__file__).with_name('linux-pipeline-reuse.py'))
reuse = importlib.util.module_from_spec(spec); spec.loader.exec_module(reuse)


def rows(outcome, count=120, fp='same', ms=0):
    return [dict(key=f'c{i}', fingerprint=fp, outcome=outcome, ms=ms) for i in range(count)]


class PipelineReuse(unittest.TestCase):
    def test_matching_fresh_entries_pass_and_new_entries_are_separate(self):
        first = dict(ledger=rows('captured'))
        second = dict(ledger=rows('hit') + [dict(key='new', fingerprint='new', outcome='source', ms=9000)])
        c = reuse.compare(first, second)
        self.assertTrue(c['passed']); self.assertEqual(c['new_or_changed_second'], 1)
        self.assertEqual(c['matched_source_ms'], 0)

    def test_fresh_refusal_source_work_or_incomplete_ledger_fails(self):
        first = dict(ledger=rows('captured'))
        second = dict(ledger=rows('hit'))
        for field in ('fresh_replay_failed', 'ledger_dropped', 'store_write_failures'):
            bad = dict(second, **{field:1})
            self.assertFalse(reuse.compare(first,bad)['passed'])
        bad = dict(ledger=rows('hit') + rows('source', count=1, ms=1100))
        self.assertFalse(reuse.compare(first,bad)['passed'])
        bad = dict(ledger=rows('refused'))
        self.assertFalse(reuse.compare(first,bad)['passed'])

    def test_changed_fingerprints_do_not_count_as_hits(self):
        c = reuse.compare(dict(ledger=rows('captured')), dict(ledger=rows('hit',fp='changed')))
        self.assertEqual(c['matched_pipelines'],0)
        self.assertFalse(c['passed'])

    def test_profile_is_hidden_capped_and_validation_free(self):
        env = reuse.relay.session_environment({}, 'cache-reuse')
        self.assertEqual(env['DS_VK_VALIDATION'],'0')
        self.assertEqual(env['DS_LINUX_COALESCE_SPARSE'],'0')
