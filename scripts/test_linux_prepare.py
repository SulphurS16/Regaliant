import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('linux-prepare-shaders.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class Preparation(unittest.TestCase):
    def test_bounded_command_keeps_player_cache_and_policy(self):
        env = dict(XDG_CACHE_HOME='/cache', DS_PLAIN_MODES='0', DS_MAX_RECONV='0')
        original = env.copy()
        cmd = prepare.command(Path('/repo'), Path('/game'), 600, env)
        self.assertEqual(env, original)
        self.assertEqual(cmd[:4], ['timeout', '-k', '10s', '630'])
        self.assertEqual(cmd[-4:], ['--prepare-shaders', '2', '--prepare-seconds', '600'])

    def test_guard_logs_cannot_touch_game_or_link_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game, target = root/'game', root/'target'
            game.mkdir(); target.mkdir()
            (game/'linked').symlink_to(target, target_is_directory=True)
            for out in [game/'logs', target/'logs']:
                with self.assertRaises(ValueError): prepare.protect_output(out, game, {})
            prepare.protect_output(root/'safe', game, {})
            with self.assertRaises(ValueError):
                prepare.protect_output(root/'safe', game, {'DS_READONLY_ROOTS': str(root)})

    def test_bounded_inputs(self):
        for seconds, env in [(0, {}), (86401, {}), (600, {'DS_COMPILE_WORKERS': '0'})]:
            with self.assertRaises(ValueError): prepare.command(Path('/repo'), Path('/game'), seconds, env)
