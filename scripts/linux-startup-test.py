#!/usr/bin/env python3
"""Ten bounded hidden menu startups with private copies of a suspect shader store."""
import argparse
import json
import os
import re
from pathlib import Path
import signal
import sys
import time
import linux_guard
import linux_snapshot
from importlib.util import spec_from_file_location, module_from_spec


def module(name):
    spec = spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    m = module_from_spec(spec); spec.loader.exec_module(m); return m


relay = module('linux-relay')
prepare = module('linux-prepare-shaders')


def compile_outcomes(session, expected):
    """Successful calls, not just 'compile input' log lines or quarantine skips."""
    found = {key: [] for key in expected}
    for path in (session/'user').rglob('pipeline-reuse.json'):
        for row in json.loads(path.read_text()).get('ledger', []):
            if row.get('key') in found:
                found[row['key']].append(row['outcome'])
    return found


def missing_compiles(rows, expected):
    return [key for key in expected if not any(
        set(row.get('expected_compiles', {}).get(key, [])) & {'source', 'hit'} for row in rows)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', default=os.environ.get('DS_GAME_DIR'))
    p.add_argument('--shader-seed', default=os.environ.get('DS_RELAY_SHADER_SEED'), required=False)
    p.add_argument('--save-seed', default=os.environ.get('DS_RELAY_SAVE_SEED'))
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--expect-pipeline', action='append', default=[], metavar='KEY',
                   help='require a successful source/binary creation of this key somewhere in the suite; repeatable')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.app or not args.shader_seed: p.error('set DS_GAME_DIR and --shader-seed (or DS_RELAY_SHADER_SEED)')
    if not 1 <= args.count <= 20 or not 5 <= args.seconds <= 120: p.error('count must be 1..20, seconds 5..120')
    if any(not re.fullmatch('[cg][0-9a-f]{16}', key) for key in args.expect_pipeline):
        p.error('--expect-pipeline must be c or g followed by 16 lowercase hexadecimal digits')
    game = Path(args.app).resolve(strict=True)
    seed = Path(args.shader_seed).resolve(strict=True)
    save = Path(args.save_seed).resolve(strict=True) if args.save_seed else None
    if not seed.is_dir() or (save is not None and not (save/'saves').is_dir()):
        p.error('shader seed must be a directory; save seed must contain saves/')
    repo = Path(__file__).resolve().parent.parent
    out = repo/'local-data/linux-port/startup'/f'run-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}'
    prepare.protect_output(out.parent, game, os.environ)
    env = dict(DS_RELAY_FULL='0', DS_RELAY_UI_PROBE='0', DS_RELAY_STEADY='0', DS_LINUX_HIDDEN='1',
        DS_RELAY_THERMAL_MODE='observe', DS_RELAY_SESSION_TEMP='70', DS_RELAY_COOLDOWN_S='600',
        DS_RELAY_RENDER_SIZE='2560x1440', DS_OUTPUT_SIZE='3840x2160', DS_LINUX_INTERVAL_SHOTS='0',
        DS_PAR_WHOLE_RUN='0', SDL_AUDIO_DRIVER='dummy', CARGO_HOME=str(repo/'.cargo-home'))
    # Alternate fresh copies of the *same* source with a persistent private store. A fatal
    # compiler quarantine can heal only the latter; never silently delete evidence in seed.
    plan = dict(output=str(out), shader_seed=str(seed), save_seed=str(save), count=args.count,
        seconds=args.seconds, timeout=args.seconds+30, environment=env, expected_pipelines=args.expect_pipeline,
        sessions=['persistent' if i % 2 == 0 else 'reset' for i in range(args.count)],
        note='No input, hidden offscreen, silent, capped 30fps. Boot/menu observation, not a gameplay or FPS test.')
    if args.dry_run: print(json.dumps(plan, indent=2)); return 0
    with linux_guard.cpu_heavy_lock():
        out.mkdir(parents=True)
        (out.parent/'latest.txt').write_text(str(out)+'\n')
        (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
        os.environ.update(env)
        rows=[]
        for i, kind in enumerate(plan['sessions'], 1):
            cache = out/('persistent-shaders' if kind == 'persistent' else f'reset-{i}-shaders')
            if not cache.exists(): linux_snapshot.copy_tree(seed, cache)
            session = out/f'start-{i:02}-{kind}'
            try:
                m = relay.run_session(repo, game, session, 'offscreen', str(args.seconds), str(args.seconds+30), cache, 'startup', save,
                                      out/('persistent-driver' if kind == 'persistent' else f'reset-{i}-driver'))
                text='\n'.join(p.read_text(errors='replace') for p in (session/'user').rglob('dsport.log'))
                flips=max((int(n) for n in re.findall(r'Linux heartbeat:.*?\bflips=(\d+)',text)),default=0)
                rows.append(dict(session=session.name, exit=m['exit_code'], crash_bytes=m['crash_bytes'], video_flips=flips,
                    thermal=m['thermal'], pipeline_reuse=m.get('pipeline_reuse'), failed=m['exit_code']!=0 or bool(m['crash_bytes']) or flips==0))
                rows[-1]['expected_compiles'] = compile_outcomes(session, args.expect_pipeline)
                rows[-1]['pending_compiles'] = [str(p.relative_to(cache)) for p in sorted(cache.rglob('*.pending'))]
            except (OSError, ValueError, linux_guard.GuardError) as e:
                rows.append(dict(session=session.name, failed=True, error=str(e)))
            (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
            lines=[f"{r['session']}: exit={r.get('exit')} failed={r['failed']} video_flips={r.get('video_flips',0)} {r.get('error','')}" for r in rows]
            missing = missing_compiles(rows, args.expect_pipeline)
            (out/'summary.txt').write_text('\n'.join(lines)+f'\nExpected pipelines without a successful creation: {missing}\nInspect user/linux-runs/*/crash.log for RIP bytes, current mappings/file offsets, TID and active compile keys/programs/SPIR-V paths. fault-*.spv are saved before driver calls. Private shaders/pipelines/crashed.txt records attributed NVIDIA faults; compile-pending records interrupted/failed source compiles. Compare reset versus persistent starts; the source seed is never changed.\n')
            print(lines[-1], flush=True)
            # Abort the suite on guard failure, not on a game crash.
            if 'error' in rows[-1] or rows[-1].get('thermal',{}).get('aborted'): break
        print(out)
        return int(len(rows)!=args.count or any(r['failed'] for r in rows) or bool(missing_compiles(rows, args.expect_pipeline)))


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try: sys.exit(main())
    except (OSError, ValueError, linux_guard.GuardError, KeyboardInterrupt) as e:
        print(f'Startup relay stopped: {e}', file=sys.stderr); sys.exit(75)
