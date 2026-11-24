#!/usr/bin/env python3
"""Bounded, thermally guarded native-resolution comparisons and floating-UI diagnostics."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import linux_guard


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


relay = module('playtest_relay', 'linux-relay.py')
prepare = module('prepare_paths', 'linux-prepare-shaders.py')


def case_environment(case, pipeline_cache=False):
    if case not in ('1440', '4k', 'diagnostic'): raise ValueError('unknown playtest case')
    env = dict(DS_RELAY_RENDER_SIZE='2560x1440' if case == '1440' else '3840x2160',
               DS_RELAY_WINDOW_SIZE='2400x1350', DS_RELAY_FULL='0', DS_RELAY_IDLE_COMPARE='0',
               DS_RELAY_UI_PROBE='1' if case == 'diagnostic' else '0', DS_TRACE_CLEARS='0',
               DS_CAPTURE_COUNT='1', DS_LINUX_COALESCE_SPARSE='0')
    if pipeline_cache: env['DS_PIPELINE_BINARIES'] = '0'
    return env


def failed(result):
    if result['exit_code'] or result['validation_errors'] or result['crash_bytes']: return True
    if result.get('ui_probe'):
        return not result.get('ui_probe_complete') or len(result.get('capture_manifests', [])) < 2
    return not result.get('measurement_valid')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', default=os.environ.get('DS_GAME_DIR'))
    p.add_argument('--case', choices=('1440', '4k', 'diagnostic'), default='4k')
    p.add_argument('--repeat', type=int, choices=(1, 2), default=1, help='second run reuses the first run caches')
    p.add_argument('--pipeline-cache', action='store_true', help='test VkPipelineCache instead of driver-refused KHR binaries')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.app: p.error('set DS_GAME_DIR or pass --app')
    for var in ('DS_RELAY_SAVE_SEED', 'DS_RELAY_SHADER_SEED'):
        if not os.environ.get(var): p.error(f'{var} is required; use an isolated prior relay snapshot')
    repo = Path(__file__).resolve().parent.parent
    game = Path(args.app).resolve(strict=True)
    seed = Path(os.environ['DS_RELAY_SAVE_SEED']).resolve(strict=True)
    shaders = Path(os.environ['DS_RELAY_SHADER_SEED']).resolve(strict=True)
    if not (seed/'saves').is_dir() or not shaders.is_dir(): p.error('save seed needs saves/; shader seed must be a directory')
    output = repo/'local-data/linux-port/playtest'/f'run-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}'
    prepare.protect_output(output.parent, game, os.environ)
    env = case_environment(args.case, args.pipeline_cache)
    plan = dict(case=args.case, repetitions=args.repeat, environment=env, output=str(output),
                duration_s=300 if args.case == 'diagnostic' else 180,
                visible=args.case != 'diagnostic', fullscreen=args.case != 'diagnostic',
                diagnostics='target dumps/clear trace; timings invalid' if args.case == 'diagnostic' else 'no GPU timestamps or validation; lightweight waits/thread accounting')
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return 0
    if not (repo/'target/release/dsport').is_file(): p.error('build --release -p dsport first')
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    # Writes only into our isolated snapshot, never the player's live settings or save.
    snapshot = output/'seed'; snapshot.mkdir()
    shutil.copytree(seed/'saves', snapshot/'saves')
    text = (seed/'pcsettings.txt').read_text() if (seed/'pcsettings.txt').exists() else ''
    lines = [l for l in text.splitlines() if not l.strip().startswith(('Setting_PC_WindowMode', 'Setting_PC_SaveFolder', 'Setting_PC_ImportSaves'))]
    lines.append('Setting_PC_WindowMode=1')
    (snapshot/'pcsettings.txt').write_text('\n'.join(lines)+'\n')
    cache = output/'shaders'; shutil.copytree(shaders, cache)
    os.environ.update(env, CARGO_HOME=str(repo/'.cargo-home'))
    results = []
    for i in range(args.repeat):
        label = f'present-fifo-{args.case}-{i+1}'
        result = relay.run_session(repo, game, output/label, 'offscreen' if args.case == 'diagnostic' else 'present',
            str(plan['duration_s']), str(plan['duration_s']+30), cache, label, snapshot, output/'driver')
        results.append(result)
        if failed(result): break
    (output/'summary.txt').write_text('Read each metrics.json, thread-summary.txt, waits-*/wait-summary.txt and pipeline-reuse.json.\n'
        'Diagnostic: capture-*.txt, targets-*/stats.txt/PNGs, UI clear probe metadata and trace clears in dsport.log.\n'
        'Compare equal saves and render sizes; never report diagnostic timings as FPS. Thermal pauses invalidate timing.\n'
        + '\n'.join(f"Run {i+1}: exit {r['exit_code']}, fps {r.get('flips_per_second')}, valid {r.get('measurement_valid')}" for i,r in enumerate(results))+'\n')
    print(output)
    return int(any(failed(r) for r in results))


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try: sys.exit(main())
    except (OSError, ValueError, linux_guard.GuardError, KeyboardInterrupt) as e:
        print(f'Playtest stopped: {e}', file=sys.stderr); sys.exit(75)
