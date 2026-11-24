#!/usr/bin/env python3
"""Two guarded, hidden runs from the same save; verify fresh pipeline binary replay."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import linux_snapshot
import signal
import sys
import time
import linux_guard


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


relay = module('linux-relay')
prepare = module('linux-prepare-shaders')


def compare(first, second):
    identity = lambda r: (r['key'], r['fingerprint'])
    captured = {identity(r) for r in first.get('ledger', []) if r['outcome'] == 'captured'}
    rows = [r for r in second.get('ledger', []) if identity(r) in captured]
    hits = {identity(r) for r in rows if r['outcome'] == 'hit'}
    sources = {identity(r) for r in rows if r['outcome'] in ('source', 'source_failed')}
    refused = {identity(r) for r in rows if r['outcome'] == 'refused'}
    seen = hits | sources | refused
    ms = sum(r['ms'] for r in rows if r['outcome'] in ('source', 'source_failed'))
    all_seen = {identity(r) for r in second.get('ledger', []) if r['outcome'] in ('hit', 'source', 'source_failed', 'refused')}
    coverage = len(seen)/max(1,len(all_seen))
    storage_bad = first.get('store_write_failures', 0) + second.get('store_write_failures', 0)
    fresh_bad = first.get('fresh_replay_failed', 0) + second.get('fresh_replay_failed', 0)
    fraction = len(hits - sources - refused) / max(1, len(seen))
    return dict(matched_pipelines=len(seen), captured_first=len(captured), reused=len(hits-sources-refused),
        matched_hit_fraction=fraction, matched_coverage=coverage, matched_source_ms=ms, matched_refusals=len(refused), fresh_replay_failed=fresh_bad, store_write_failures=storage_bad,
        new_or_changed_second=len({identity(r) for r in second.get('ledger', [])} - captured),
        passed=len(seen)>=100 and coverage>=0.95 and fraction>=0.99 and ms<=1000 and not fresh_bad and not storage_bad
            and not first.get('ledger_dropped', 0) and not second.get('ledger_dropped', 0),
        note='Matched (key,fingerprint) captured in run 1 only; new/changed pipelines reported separately. Creation time is aggregate call time, not CPU time.')


def read_report(run):
    files = sorted((run/'user').rglob('pipeline-reuse.json'))
    if len(files) != 1: raise ValueError(f'expected one pipeline report in {run}, found {len(files)}')
    return json.loads(files[0].read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--app', default=os.environ.get('DS_GAME_DIR'))
    p.add_argument('--seconds', type=int, default=300, help='each run, 60..600 seconds')
    p.add_argument('--cold-driver', action='store_true', help='second run gets a new driver-cache directory; port binaries retained')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not args.app: p.error('set DS_GAME_DIR or --app')
    if not 60 <= args.seconds <= 600: p.error('--seconds must be 60..600')
    for k in ('DS_RELAY_SAVE_SEED', 'DS_RELAY_SHADER_SEED'):
        if not os.environ.get(k): p.error(f'set {k} to an isolated prior relay snapshot')
    repo = Path(__file__).resolve().parent.parent
    game = Path(args.app).resolve(strict=True)
    save = Path(os.environ['DS_RELAY_SAVE_SEED']).resolve(strict=True)
    shader = Path(os.environ['DS_RELAY_SHADER_SEED']).resolve(strict=True)
    if not (save/'saves').is_dir() or not shader.is_dir(): p.error('save seed must contain saves/; shader seed must be a directory')
    output = repo/'local-data/linux-port/pipeline-reuse'/f'run-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}-{os.getpid()}'
    prepare.protect_output(output.parent, game, os.environ)
    env = dict(DS_PIPELINE_BINARIES='1', DS_PB_VERIFY_CAPTURE='1', DS_RELAY_FULL='0', DS_RELAY_UI_PROBE='0',
               DS_RELAY_RENDER_SIZE='2560x1440', DS_RELAY_WINDOW_SIZE='800x450', DS_SPIRV_NONCE='0',
               DS_SIM_PB_REFUSE='0', DS_PB_NO_INTERNAL='0', __GL_SHADER_DISK_CACHE='1',
               __GL_SHADER_DISK_CACHE_SIZE=os.environ.get('__GL_SHADER_DISK_CACHE_SIZE','8589934592'),
               __GL_SHADER_DISK_CACHE_SKIP_CLEANUP=os.environ.get('__GL_SHADER_DISK_CACHE_SKIP_CLEANUP','1'))
    plan = dict(output=str(output), seconds_each=args.seconds, timeout_each=args.seconds+30,
                visible=False, audio='null', fps_cap=30, workers=int(os.environ.get('DS_COMPILE_WORKERS', '2')), environment=env,
                binary_seed='fresh; copy translations/specs, omit .pb/.shared and old binaries-mode.txt', cold_driver=args.cold_driver)
    if args.dry_run: print(json.dumps(plan, indent=2)); return 0
    if not (repo/'target/release/dsport').is_file(): p.error('build --release -p dsport first')
    output.mkdir(parents=True)
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    cache = output/'shaders'
    linux_snapshot.copy_tree(shader, cache, ignore=shutil.ignore_patterns('*.pb', '*.shared', 'binaries-mode.txt'))
    os.environ.update(env, CARGO_HOME=str(repo/'.cargo-home'))
    results = []
    for n in (1,2):
        run = output/f'run-{n}'
        driver = output/('driver-2' if n == 2 and args.cold_driver else 'driver')
        m = relay.run_session(repo, game, run, 'offscreen', str(args.seconds), str(args.seconds+30), cache, 'cache-reuse', save, driver)
        results.append(m)
        if m['exit_code'] or m['validation_errors'] or m['crash_bytes']: break
    if len(results) == 2:
        c = compare(read_report(output/'run-1'), read_report(output/'run-2'))
        c['sessions_complete'] = all(any('saved game resumed, movement' in s for s in r['selfplay']) for r in results)
        c['thermal_valid'] = all(r['thermal']['measurement_valid'] for r in results)
        c['passed'] &= c['sessions_complete'] and c['thermal_valid']
    else: c = dict(passed=False, error='session failed; inspect logs')
    (output/'comparison.json').write_text(json.dumps(c, indent=2)+'\n')
    (output/'summary.txt').write_text(json.dumps(c, indent=2)+'\nInspect run-*/user/linux-runs/*/pipeline-reuse.json (full ledger, fresh failures, driver cache environment), cache inventories, dsport.log, frame PNGs and monitor.jsonl. These runs verify reuse, not uncapped FPS.\n')
    print(output)
    return 0 if c['passed'] else 1


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    try: sys.exit(main())
    except (OSError, ValueError, linux_guard.GuardError, KeyboardInterrupt) as e:
        print(f'Pipeline reuse test stopped: {e}', file=sys.stderr); sys.exit(75)
