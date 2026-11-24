#!/usr/bin/env python3
"""Prepare the player's shader store, with the relay's CPU and thermal limits."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import linux_guard


def protect_output(output, game, env):
    forbidden = [game.resolve()] + [Path(p).resolve() for p in env.get('DS_READONLY_ROOTS', '').split(':') if p]
    if game.is_dir():
        for directory, dirs, files in os.walk(game, followlinks=False):
            for name in dirs + files:
                p = Path(directory)/name
                if p.is_symlink(): forbidden.append(p.resolve())
    outputs = [output.resolve()]
    if output.exists():
        for directory, dirs, files in os.walk(output, followlinks=False):
            outputs.extend((Path(directory)/n).resolve() for n in dirs + files if (Path(directory)/n).is_symlink())
    if any(p.is_relative_to(root) for p in outputs for root in forbidden):
        raise ValueError('Preparation monitor output must stay outside game/read-only roots and their symlink targets')


def command(repo, game, seconds, env):
    # Guard sets the same worker count for all driver compiler pools. XDG/DSPORT_HOME,
    # DS_LINUX_SHADER_CACHE and player diagnostic switches stay exactly as for normal play.
    workers = int(env.get('DS_COMPILE_WORKERS', '2'))
    if not 1 <= workers <= 64: raise ValueError('DS_COMPILE_WORKERS must be 1..64')
    if not 1 <= seconds <= 86400: raise ValueError('--seconds must be 1..86400')
    return ['timeout', '-k', '10s', str(seconds + 30), str(repo/'target/release/dsport'),
            '--app', str(game), '--prepare-shaders', str(workers), '--prepare-seconds', str(seconds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', default=os.environ.get('DS_GAME_DIR'), help='game folder or .zar; defaults to DS_GAME_DIR')
    parser.add_argument('--seconds', type=int, default=14400, help='bounded attempt; rerun to resume (default four hours)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not args.app: parser.error('set DS_GAME_DIR or pass --app /path/to/PPSA01341-app0')
    repo = Path(__file__).resolve().parent.parent
    game = Path(args.app).resolve(strict=True)
    env = os.environ.copy()
    env['CARGO_HOME'] = str(repo/'.cargo-home')
    env.setdefault('DS_RELAY_FPS_LIMIT', '30')
    output = repo/'local-data/linux-port/shader-preparation'/f'{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}'
    protect_output(output.parent, game, env)
    cmd = command(repo, game, args.seconds, env)
    if args.dry_run:
        print(json.dumps(dict(command=cmd, monitor=str(output), fps_limit=env['DS_RELAY_FPS_LIMIT'],
                             paths='inherit player XDG/DSPORT_HOME and DS_LINUX_SHADER_CACHE',
                             cpus=linux_guard.select_cpus(env.get('DS_RELAY_CPUS')), nice=10,
                             thermal='95 C pause / 85 C resume / 60 s abort, configurable DS_RELAY_TEMP_*'), indent=2))
        return 0
    if not (repo/'target/release/dsport').is_file(): parser.error('build dsport --release first (see docs/linux.md)')
    print(f'Thermal monitor: {output}', flush=True)
    return linux_guard.run(cmd, output, env=env, cwd=repo)


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        sys.exit(main())
    except (OSError, ValueError, linux_guard.GuardError, KeyboardInterrupt) as e:
        print(f'Preparation stopped: {e}. Rerun the same command to resume.', file=sys.stderr)
        sys.exit(75)
