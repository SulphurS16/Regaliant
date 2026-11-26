#!/usr/bin/env python3
"""Visible native-4K FSR3 validation; --steady adds a warmed, capture-free off/on comparison."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import linux_guard
import linux_snapshot
import importlib.util

spec=importlib.util.spec_from_file_location('steady',Path(__file__).with_name('linux-steady.py'))
steady=importlib.util.module_from_spec(spec);spec.loader.exec_module(steady)


def environment(on, diagnostic=False):
    return dict(steady.environment(), DS_FRAME_GENERATION='fsr3' if on else 'off',
        DS_FG_SHOTS='1' if diagnostic else '0', DS_FG_INPUT_DUMP='1' if diagnostic else '0',
        DS_PAR_WHOLE_RUN='0', DS_RECORD_TSC='0', DS_RECORD_STATIC_BIND='0',
        DS_RELAY_THERMAL_MODE='observe', DS_RELAY_SESSION_TEMP=os.environ.get('DS_RELAY_SESSION_TEMP','70'),
        DS_RELAY_COOLDOWN_S=os.environ.get('DS_RELAY_COOLDOWN_S','600'))


def input_pairs(run):
    pairs=[]
    required={'hudless','real','generated','depth','motion','previous-real','previous-hudless'}
    for path in sorted((run/'user').rglob('fg-inputs-*/inputs.json')):
        try: value=json.loads(path.read_text())
        except (OSError,ValueError): continue
        names={image.get('name') for image in value.get('images',[])}
        complete=required<=names and all((path.parent/(name+suffix)).is_file()
            for name in required for suffix in ('.raw','.png'))
        # Schema 2 preserves the guest input separately from the repaired SDK input.
        if value.get('schema', 1) >= 2:
            complete = complete and {'motion-raw','motion-status'} <= names and all(
                (path.parent/(name+'.raw')).is_file() for name in ('motion-raw','motion-status'))
        pairs.append(dict(directory=str(path.parent.relative_to(run)),complete=complete,**value))
    return pairs


def diagnostics(run):
    lines=[]
    for path in [*(run/'user').rglob('dsport.log'), run/'console.log']:
        if path.is_file():
            lines.extend(line for line in path.read_text(errors='replace').splitlines()
                if 'FG ' in line or 'FSR3' in line or 'Linux frame generation:' in line)
    (run/'framegen-stages.txt').write_text('\n'.join(lines)+'\n')
    states=[]
    for path in sorted((run/'user').rglob('framegen-state.json')):
        try: states.append(json.loads(path.read_text()))
        except (OSError,ValueError): pass
    return dict(states=states,last_events=lines[-20:])


def review(run, metrics, on, diagnostic):
    result=steady.report(run,metrics,diagnostic)
    text='\n'.join(p.read_text(errors='replace') for p in (run/'user').rglob('dsport.log'))
    reasons=result['invalid_reasons']
    if on:
        result['framegen_diagnostics']=diagnostics(run)
        if 'FSR3 FG context:' not in text: reasons.append('FSR3 context never created')
        if 'frame generation disabled for this session' in text: reasons.append('FSR3 fell back to real frames; inspect log')
        for phase in ('stationary','walking'):
            fg=result['phases'].get(phase,{}).get('presentation',{}).get('frame_generation',{})
            if fg.get('generated_completions',0)<10: reasons.append(phase+' fewer than 10 generated completions')
        shots=list((run/'user').rglob('generated-*.png'))
        if diagnostic and not shots: reasons.append('no generated-frame PNGs')
        if diagnostic:
            result['input_pairs']=input_pairs(run)
            if not any(pair['complete'] for pair in result['input_pairs']):
                reasons.append('no complete current/previous FSR input pair dump')
            from linux_framegen_compare import analyze
            for pair in result['input_pairs']:
                if not pair['complete']: continue
                folder=run/pair['directory']
                try:
                    comparison=analyze(folder)
                    (folder/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False)+'\n')
                    pair['comparison']=comparison
                    reasons.extend(folder.name+': '+error for error in comparison['failures'])
                except (OSError,ValueError,KeyError) as e:
                    reasons.append(folder.name+': offline comparison failed: '+str(e))
        result['generated_pngs']=[str(p.relative_to(run)) for p in shots]
    if 'Linux whole-run recording: true' in text: reasons.append('unsafe whole-run recording enabled')
    result['frame_generation']='FSR3' if on else 'off'
    result['valid_steady_state']=not reasons and not diagnostic
    (run/'framegen-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize(results):
    lines=['Experimental FSR3 frame generation, native 3840x2160, FIFO. Whole-run OFF.',
        'KHR completion is CPU-observed presentation completion, not physical scanout.',
        'No input-to-photon latency claim. Snapshot age includes recording, generation and pacing.',
        'PNG display previews are opaque; raw generated alpha is an SDK inpainting mask, not transparency.',
        'Inspect generated-*.png beside window-*.png and steady boundary PNGs: HUD, disocclusion, colours.',
        'Validation/capture session is diagnostic, never an FPS measurement.\n']
    for name,r in results.items():
        lines.append(name+': '+('FAILED: '+', '.join(r['invalid_reasons']) if r['invalid_reasons'] else ('UNMEASURED warm-up' if r.get('warmup_only') else 'diagnostic complete' if r['diagnostic'] else 'valid measurement')))
        for state in r.get('framegen_diagnostics',{}).get('states',[]):
            lines.append('  FG gates: '+str(state))
        for pair in r.get('input_pairs',[]):
            lines.append(f"  Paired inputs: {pair['directory']} complete={pair['complete']}; formats/ranges/NaNs in inputs.json.")
            if 'comparison' in pair:
                lines.append('    Midpoint RGB reference: '+str(pair['comparison']['midpoint']))
                lines.append('    Raw/prepared motion: '+str(pair['comparison']['motion']))
        if r.get('framegen_diagnostics'): lines.append('  Creation and input diagnostics: framegen-stages.txt (native SDK stages also in console.log).')
        for label,p in r.get('phases',{}).items():
            present=p.get('presentation',{})
            if r.get('valid_steady_state') and not r.get('warmup_only'):
                lines.append(f"  {label}: real guest fps {p.get('flips_per_second')}; presented {present.get('wait_complete',{})}")
                lines.append('    generation/latency: '+str(present.get('frame_generation',{})))
            else:
                f=present.get('frame_generation',{})
                lines.append(f"  {label}: unmeasured; real/generated completions {f.get('real_completions',0)}/{f.get('generated_completions',0)}; raw timings retained in JSON only")
    lines.append('\nVisual acceptance is manual and still required even when all numerical checks pass.')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',default=os.environ.get('DS_GAME_DIR'))
    parser.add_argument('--save-seed',default=os.environ.get('DS_RELAY_SAVE_SEED','local-data/linux-port/save-seed'))
    parser.add_argument('--shader-seed',default=os.environ.get('DS_RELAY_SHADER_SEED'))
    parser.add_argument('--steady',action='store_true',help='after short validation, warm the 30s stationary/60s walking route, then measure FG off/on')
    parser.add_argument('--warmup-passes',type=int,default=4)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not args.app: parser.error('set DS_GAME_DIR or pass --app /path/to/PPSA01341-app0')
    if not 1<=args.warmup_passes<=12: parser.error('--warmup-passes must be 1..12')
    repo=Path(__file__).resolve().parent.parent
    game=Path(args.app).resolve(strict=True);seed=Path(args.save_seed).resolve(strict=True)
    verified=None if args.shader_seed else steady.latest_seed(repo/'local-data/linux-port/pipeline-reuse')
    shaders=Path(args.shader_seed).resolve(strict=True) if args.shader_seed else verified/'shaders'
    if not (seed/'saves').is_dir() or not shaders.is_dir(): parser.error('save seed needs saves/ and shader seed must be a directory')
    output=repo/'local-data/linux-port/framegen'/f'run-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}'
    steady.prepare.protect_output(output.parent,game,os.environ)
    plan=dict(output=str(output),game=str(game),save_seed=str(seed),shader_seed=str(shaders),environment=environment(True,True),
        sessions=['steady-fg-validation-uffd']+(['unmeasured route warm-up','steady-a1-fg-off-uffd','steady-b1-fg-on-uffd'] if args.steady else []),
        build='matching runtime in this checkout target, nice10/-j4; ABI 2 adapter refreshed from existing SDK source',
        cpu='shared lock, 8 logical CPUs/nice10/workers2; cool <=70C each session, 105C/60s abort',
        window='visible fullscreen native4K; null audio; scripted save resume and walking; no input grab',
        timeout_s=630,diagnostic_route_s=10,steady_route_s=90,
        read_only='game and seeds; every run uses private save/shader/driver copies',
        requires='source-built target/release/upscalers/libregaliant_fsrfg.so and matching backend; Vulkan validation layer for diagnostic')
    if args.dry_run: print(json.dumps(plan,indent=2));return 0
    try: import numpy  # fail before a game session, not while reporting its captures
    except ImportError: parser.error('offline FG comparison requires numpy')
    if not (repo/'target/release/upscalers/libregaliant_fsrfg.so').is_file(): parser.error('run python3 scripts/linux-build-framegen.py first')
    # A merge changes the adapter ABI as well as Rust. Refresh just the small bridge
    # and input shader with the existing SDK/backend before taking the session lock.
    spec=importlib.util.spec_from_file_location('fg_build',repo/'scripts/linux-build-framegen.py')
    build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)
    stamp=repo/'target/release/upscalers/fsr-framegen-adapter.sha256'
    if not stamp.is_file() or stamp.read_text().strip()!=build.adapter_digest(repo):
        subprocess.run([sys.executable,str(repo/'scripts/linux-build-framegen.py'),'--bridge-only'],check=True)
    output.mkdir(parents=True);print(output,flush=True)
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    results={}
    def persist():
        (output/'summary.txt').write_text(summarize(results))
        (output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    try:
        with linux_guard.cpu_heavy_lock():
            # The reviewer merges source from a worktree. Build the matching runtime
            # here too: ABI 1 binaries must not silently run against an ABI 2 bridge.
            # Pin target to this checkout even if the shell exports CARGO_TARGET_DIR.
            build_env=os.environ.copy()
            cache=next((p/'.cargo-home' for p in (repo,*repo.parents) if (p/'.cargo-home').is_dir()),repo/'.cargo-home')
            build_env.setdefault('CARGO_HOME',str(cache))
            build_env['CARGO_TARGET_DIR']=str(repo/'target')
            with (output/'runtime-build.log').open('w') as log:
                code=linux_guard.run(['timeout','-k','5s','900s','cargo','build','--release','-p','dsport','-j4'],
                    output/'runtime-build',env=build_env,cwd=repo,stdout=log,stderr=log)
            if code: raise ValueError('matching runtime build failed; see runtime-build.log')
            snapshot=output/'seed';snapshot.mkdir()
            linux_snapshot.copy_tree(seed/'saves',snapshot/'saves')
            (snapshot/'pcsettings.txt').write_text(steady.settings((seed/'pcsettings.txt').read_text() if (seed/'pcsettings.txt').is_file() else ''))
            master=output/'prepared-shaders';driver=output/'prepared-driver'
            linux_snapshot.copy_tree(shaders,master)
            if verified and (verified/'driver').is_dir(): linux_snapshot.copy_tree(verified/'driver',driver)
            def run(label,name,on,diagnostic=False,warming=False):
                os.environ.update(environment(on,diagnostic),CARGO_HOME=str(repo/'.cargo-home'))
                cache,dc=master,driver
                if not warming:
                    cache=output/(name+'-shaders');dc=output/(name+'-driver')
                    linux_snapshot.copy_tree(master,cache)
                    if driver.exists(): linux_snapshot.copy_tree(driver,dc)
                metrics=steady.relay.run_session(repo,game,output/name,'present','600','630',cache,label,snapshot,dc)
                r=review(output/name,metrics,on,diagnostic);r['warmup_only']=warming
                results[name]=r;persist();return r
            # Verify SDK/WSI validity before warming or doing long performance sessions.
            name='steady-fg-validation-uffd'
            r=run(name,name,True,True)
            if r['invalid_reasons']: raise ValueError('FG diagnostic failed; no long runs started; see summary.txt')
            if args.steady:
                label='steady-fg-on-uffd'
                for attempt in range(args.warmup_passes):
                    r=run(label,f'warmup-{attempt+1}',True,warming=True)
                    failures=[x for x in r['invalid_reasons'] if not x.endswith('compiled pipelines inside measurement')]
                    if failures: raise ValueError('route warm-up failed: '+', '.join(failures))
                    if not r['invalid_reasons']: break
                else: raise ValueError('route is not warm; no performance sessions started')
                for name,on in [('steady-a1-fg-off-uffd',False),('steady-b1-fg-on-uffd',True)]:
                    run(name,name,on)
    except (OSError,ValueError,linux_guard.GuardError,KeyboardInterrupt) as e:
        persist()
        with (output/'summary.txt').open('a') as f: f.write(f'\nSTOPPED: {e}\n')
        raise
    print(summarize(results))
    return int(any(r['invalid_reasons'] for r in results.values() if not r.get('warmup_only')))

if __name__=='__main__':
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try: sys.exit(main())
    except (OSError,ValueError,linux_guard.GuardError,KeyboardInterrupt) as e:
        print(f'Frame-generation relay stopped: {e}',file=sys.stderr);sys.exit(75)
