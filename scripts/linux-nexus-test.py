#!/usr/bin/env python3
"""Three Nexus runs: refresh the seed, replay it with adaptive FG, validate shaders.
No builds, no git writes, no writes to the supplied save or shader seeds.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import importlib.util
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

CASES=(('seed-off','off',False,False), ('warm-adaptive','fg-adaptive',True,False),
       ('validation-census','vertices',False,True))

def command(binary, save, shader, label, case, warm, validation):
    cmd=[sys.executable,str(HERE/'linux-exposure-check.py'),'--binary',str(binary),
         '--save-seed',str(save),'--shader-seed',str(shader),'--play-case',case,'--label',label,
         '--nexus-replay','--relative-measure','--flips','300,600' if validation else '3000']
    if warm:cmd+=['--warm-required']
    if validation:cmd+=['--validation','--seconds','900','--load-shots','60,300,600,900,1200']
    return cmd

def load_summary(run):
    try:
        return json.loads((run/'summary.json').read_text()) if run else {}
    except (OSError,ValueError) as e:
        return {'valid':False,'failures':['Incomplete child summary: '+str(e)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,default=HERE.parent/'target/release/dsport')
    p.add_argument('--save-seed',type=Path,required=True)
    p.add_argument('--shader-seed',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not os.environ.get('DS_GAME_DIR'):p.error('set DS_GAME_DIR to the read-only game directory')
    root=HERE.parent/'local-data/linux-port/nexus'/('run-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+str(os.getpid()))
    shader=a.shader_seed.resolve();save=a.save_seed.resolve();binary=a.binary.resolve()
    if a.dry_run:
        print(json.dumps([dict(command=command(binary,save,shader,label,case,warm,val), shader_seed_from=('supplied seed' if i==0 else 'seed-off/shaders'), driver_seed_from=(None if i==0 else 'seed-off/driver')) for i,(label,case,warm,val) in enumerate(CASES)],indent=2));return 0
    # Child owns the shared CPU lock, guards, private copies, timeout and shutdown.
    if not shader.is_dir() or not (save/'saves').is_dir() or not binary.is_file():p.error('binary and immutable shader/save seeds required')
    spec=importlib.util.spec_from_file_location('prepare',HERE/'linux-prepare-shaders.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    prepare.protect_output(root.parent,Path(os.environ['DS_GAME_DIR']).resolve(),os.environ)
    root.mkdir(parents=True)
    (root.parent/'latest.txt').write_text(str(root)+'\n')
    rows=[]
    rebuilt=set()
    driver=None
    for label,case,warm,val in CASES:
        full_label=root.name+'-'+label
        cmd=command(binary,save,shader,full_label,case,warm,val)
        if driver: cmd+=['--driver-seed',str(driver)]
        if warm and driver is None:
            code=1
            (root/(label+'.log')).write_text('Warm session skipped: seed-off did not leave both application and driver caches.\n')
        else:
            with (root/(label+'.log')).open('w') as out:
                child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT)
                try:
                    code=child.wait(timeout=1080 if val else 780)
                except subprocess.TimeoutExpired:
                    code=124
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:child.wait(timeout=25)
                        except subprocess.TimeoutExpired:child.kill();child.wait()
        matches=list((HERE.parent/'local-data/linux-port/exposure').glob('*-'+full_label))
        run=matches[0] if len(matches)==1 else None
        summary=load_summary(run)
        if warm and driver is None:
            summary={'valid':False,'failures':['warm session skipped: seed-off cache handoff unavailable']}
        reports=summary.get('pipeline_reuse',[])
        def entries(outcome):
            return {(r['key'],r['fingerprint']) for q in reports for r in q.get('ledger',[]) if r['outcome']==outcome}
        if label=='seed-off':
            rebuilt=entries('captured')
        if warm:
            replayed=rebuilt & entries('hit')
            refused=rebuilt & entries('refused')
            summary['previously_captured']={'hit':len(replayed),'refused':len(refused),'refused_examples':sorted(refused)[:10]}
            if refused:
                summary['valid']=False
                summary.setdefault('failures',[]).append(f'{len(refused)} identical keys freshly captured in seed-off refused on replay')
        rows.append(dict(label=label,exit=code,run=str(run) if run else None,summary=summary))
        # Only the first normal session supplies caches to subsequent runs.
        if run and (run/'shaders').is_dir() and (run/'driver').is_dir() and label == 'seed-off':
            shader=run/'shaders'
            driver=run/'driver'
        (root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        lines=[]
        for row in rows:
            s=row['summary'];lines.append(f"{row['label']}: exit {row['exit']}, valid={s.get('valid',False)}; {row['run']}")
            for q in s.get('pipeline_reuse',[]):
                lines.append(f"  binary hits {q.get('binary_hits')}, source creates {q.get('source_create_calls')}, other-device/contract {q.get('other_driver')}, refused {q.get('binary_refused')}")
            windows=[w for w in s.get('play_metrics',{}).get('windows',[]) if w['first_flip'] >= (s.get('stationary_begin_upper_flip') or 10**12)]
            if windows:
                w=windows[-1];lines.append(f"  last stationary window: {w['fps']:.2f} real fps, {w.get('pb_source_calls','?')} source creates")
            if 'previously_captured' in s: lines.append(f"  replay of first-run replacements: {s['previously_captured']}")
            if 'outlier_files' in s: lines.append("  vertex census: "+', '.join(s['outlier_files']))
            pacing=s.get('presentation',{}).get('exit',{}).get('pacing',{})
            if pacing.get('samples'):lines.append(f"  KHR completions {pacing.get('presented_fps',0):.2f}/s, p50/p99 {pacing['p50_ms']:.2f}/{pacing['p99_ms']:.2f}ms (includes captures)")
            a=s.get('adaptive',{})
            if a.get('fresh_fps') is not None:lines.append(f"  adaptive {a['fresh_fps']:.2f} fresh images/s, {a['slot_fps']:.2f} slots/s; {a['counts']}; age all slots {a['content_age_ms']} ms, fresh {a.get('fresh_content_age_ms')} ms")
            lines.extend('  '+f for f in s.get('failures',[]))
        lines+=['See each summary.txt for stationary rates. Validation/census holds the early after-load PNGs.',
                'Adaptive: fg-adaptive.csv records application slots. KHR present timing disabled to match preparation. Validation/census is not timed.',
                'Current adaptive playback delays a real interval +4ms to keep midpoints before their newer endpoint; repeated slots are explicitly counted.']
        (root/'summary.txt').write_text('\n'.join(lines)+'\n')
        print(f"{label}: exit {code}, valid={summary.get('valid',False)}; {run}",flush=True)
    print(root)
    return int(any(r['exit'] or not r['summary'].get('valid') for r in rows))

if __name__=='__main__':
    def interrupted(sig, frame): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGHUP,interrupted)
    try:sys.exit(main())
    except (KeyboardInterrupt,subprocess.TimeoutExpired):sys.exit(75)
