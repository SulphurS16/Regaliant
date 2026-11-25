#!/usr/bin/env python3
"""Compare a supplied Linux binary at fixed video flips in an isolated, stationary save.

No build or git operations. Absolute flip numbers use the capture protocol present in old
binaries; loading speed is not assumed. Samples outside confirmed stationary gameplay fail.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import sys
import time

import linux_guard
import linux_snapshot
from importlib.util import spec_from_file_location, module_from_spec


def module(name):
    spec = spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    m = module_from_spec(spec); spec.loader.exec_module(m); return m


prepare = module('linux-prepare-shaders')
# Keep arbitrary path, input, capture and safety overrides out of a diagnostic comparison.
SWITCHES = {'DS_LINUX_WRITE_WATCH': {'uffd', 'mprotect'},
            'DS_NAN_PRESERVE': {'0', '1'}, 'DS_PLAIN_MODES': {'0', '1'},
            'DS_MAX_RECONV': {'0', '1'}, 'DS_META_CLAIMS': {'0', '1', '2', '3'},
            'DS_META_CLAIM_REBIND': {'0', '1'}}


def overrides(items):
    result = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or key not in SWITCHES or value not in SWITCHES[key]:
            raise ValueError('--env supports '+', '.join(k+'='+ '/'.join(sorted(v)) for k,v in SWITCHES.items()))
        result[key] = value
    return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def settings(text):
    # Do not inherit imports/relocations or a saved FG/upscaler choice. No changes to seed.
    values = dict(WindowMode=0, RenderScale=6, AntiAliasing=0, AspectRatio=1, FrameRateLimit=0,
                  FrameGeneration=0)
    skip = tuple('Setting_PC_'+k+'=' for k in [*values, 'SaveFolder', 'ImportSaves'])
    lines = [l for l in text.splitlines() if not l.strip().startswith(skip)]
    return '\n'.join(lines+[f'Setting_PC_{k}={v}' for k,v in values.items()])+'\n'


def environment(inherited, run, game, flips, extra, tap):
    env = {k:v for k,v in inherited.items() if not k.startswith(('DS_', 'DSPORT_', '__GL_', 'MESA_', 'CUDA_CACHE_', 'VK_', 'SDL_'))}
    # Preserve only supervision/readonly configuration, not a developer's render experiment.
    for k,v in inherited.items():
        if k.startswith('DS_RELAY_TEMP_') or k in ('DS_RELAY_CPUS','DS_COMPILE_WORKERS','DS_CPU_HEAVY_LOCK','DS_RELAY_COOLDOWN_S','DS_READONLY_ROOTS'):
            env[k] = v
    env.update(DS_GAME_DIR=str(game), DSPORT_HOME=str(run/'user'),
        DS_LINUX_SHADER_CACHE=str(run/'shaders'), DS_FRAME_GENERATION='off',
        DS_LINUX_HIDDEN='1', SDL_AUDIO_DRIVER='dummy', DS_GPU='1', DS_SECTIONS='1', DS_C4='0',
        DS_UPSCALER='off', DS_RENDER_SIZE='3840x2160', DS_OUTPUT_SIZE='3840x2160',
        DS_PAR_WHOLE_RUN='0', DS_LINUX_COALESCE_SPARSE='0', DS_VK_VALIDATION='0',
        VK_LOADER_LAYERS_DISABLE='~implicit~', DS_SCREENSHOT=','.join(map(str,flips)),
        DS_RELAY_STEADY='0', DS_RELAY_GAMEPLAY_SECONDS='180', DS_RELAY_SHORT='1',
        DS_RELAY_IDLE_COMPARE='0', DS_RELAY_FPS_LIMIT='30', DS_RELAY_CLOSE_WINDOW='0',
        DS_LINUX_INTERVAL_SHOTS='0', DS_FRAME_CAP='30', DS_UNCAPPED='0',
        DS_NAN_PROBE='0', DS_PROFILE_GPU='0', DS_PROFILE_GPU_STAMPS='0',
        DS_LINUX_WRITE_WATCH='uffd', DS_RELAY_THERMAL_MODE='observe',
        __GL_SHADER_DISK_CACHE_PATH=str(run/'driver/nvidia'), __GL_SHADER_DISK_CACHE_SIZE=str(8<<30),
        __GL_SHADER_DISK_CACHE_SKIP_CLEANUP='1', MESA_SHADER_CACHE_DIR=str(run/'driver/mesa'),
        MESA_SHADER_CACHE_MAX_SIZE='8G', CUDA_CACHE_PATH=str(run/'driver/cuda'))
    for k in ('CACHE','DATA','CONFIG','STATE'): env['XDG_'+k+'_HOME'] = str(run/('xdg-'+k.lower()))
    for k in ('TMPDIR','TMP','TEMP'): env[k] = str(run/'tmp')
    if tap:
        # Existing, cross-version diagnostic: TAA's resource 0 is a BUFFER descriptor at
        # pass constants +0xf0, NOT the first float of fg-inputs/taa_words (a different block).
        env.update(DS_TAP_CS='100893aa00:0:16', DS_TAP_FROM=str(max(1,flips[0]-120)))
    env.update(extra)
    return env


def float_word(word):
    if word == '#deadbeef': return None  # Runtime's unreadable guest-memory marker.
    value = struct.unpack('<f',struct.pack('<I',int(word[1:],16)))[0] if word.startswith('#') else float(word)
    return value if math.isfinite(value) else None


def evidence(text, result=None):
    if result is None: result = dict(shots={}, begin_upper_flip=None, ended=False, failed=False, tracking=[], identity=[], exposure=[])
    active = result.get('_active', False)
    for line in text.splitlines():
        if 'relay step: gameplay-measure-begin;' in line:
            active = 'menu stack Some(0)' in line and bool(re.search(r'HUD bytes Some\(([89]\d\d|[1-9]\d{3,})\)',line))
        if active and result['begin_upper_flip'] is None:
            h = re.search(r'Linux heartbeat:.*?\bflips=(\d+)',line)
            if h: result['begin_upper_flip'] = int(h[1])
        if 'relay step: gameplay-measure-end;' in line: result['ended'] = True; active = False
        if 'Linux self-play result: FAILED' in line or 'FATAL' in line: result['failed'] = True
        shot = re.search(r'screenshot of flip (\d+) \(buffer .*?\): (.+\.png)',line)
        if shot: result['shots'][int(shot[1])] = shot[2]
        if 'Linux write watch:' in line: result['tracking'].append(line)
        if any(s in line for s in ('Vulkan device ids:', 'Vulkan: NaN,', 'translation store:', 'Vulkan device:', 'maximal reconvergence', 'images: metadata claim mask')):
            result['identity'].append(line)
        tap = re.search(r'tap cs 0x100893aa00 res 0 for flip (\d+): gpu \[([^]]+)\] guest \[([^]]+)\]',line)
        if tap:
            try: result['exposure'].append(dict(flip=int(tap[1]),gpu=float_word(tap[2].split()[0]),guest=float_word(tap[3].split()[0])))
            except (ValueError,IndexError,OverflowError): pass
    result['_active'] = active
    return result


class LogTail:
    def __init__(self, root):
        self.root, self.offsets, self.partial = root, {}, {}

    def read(self):
        lines=[]
        for path in sorted(self.root.glob('linux-runs/*/dsport.log')):
            with path.open('r', errors='replace') as f:
                f.seek(self.offsets.get(path,0)); text=self.partial.get(path,'')+f.read(); self.offsets[path]=f.tell()
            complete, sep, pending=text.rpartition('\n')
            self.partial[path]=pending
            if sep: lines.append(complete)
        return '\n'.join(lines)


def brightness(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im: rgb=np.asarray(im.convert('RGB'),dtype=np.float64)/255.0
    h,w=rgb.shape[:2]
    def stats(a):
        linear=np.where(a<=0.04045,a/12.92,((a+0.055)/1.055)**2.4)
        return dict(mean_rgb=float(a.mean()), mean_luma_srgb=float((a@[.2126,.7152,.0722]).mean()),
                    mean_luminance_linear=float((linear@[.2126,.7152,.0722]).mean()),
                    near_white_fraction=float(np.all(a>=.95,axis=2).mean()), near_black_fraction=float(np.all(a<=.01,axis=2).mean()))
    return dict(width=w,height=h,full=stats(rgb),centre=stats(rgb[h//4:3*h//4,w//4:3*w//4]))


def report(run, flips, env, code, text, stop_reason):
    data=evidence(text); failures=[]; samples=[]
    # Old and current bootstraps deliberately use 75 for the stop-file protocol. Only
    # accept that exit for our completed captures AND the runtime's completed shutdown;
    # thermal cancellation, a timeout/kill, or an unrelated 75 must still fail.
    capture_stop = (code == 75 and stop_reason == 'fixed captures complete'
                    and 'Linux shutdown complete: status 75;' in text)
    if code != 0 and not capture_stop: failures.append(f'runtime exit {code}')
    if data['failed']: failures.append('runtime/self-play failure')
    begin=data['begin_upper_flip']
    if begin is None: failures.append('no confirmed stationary gameplay heartbeat')
    if data['ended']: failures.append('180s stationary window ended; samples are not a fixed stationary comparison')
    if env.get('DS_TAP_CS') and not data['exposure']: failures.append('no TAA exposure taps; verify the program address before drawing a conclusion')
    wanted=env['DS_LINUX_WRITE_WATCH']
    if not any('Linux write watch: '+wanted in l for l in data['tracking']): failures.append('requested tracker was not active (possible uffd fallback)')
    if any(k in env for k in ('DS_META_CLAIMS','DS_META_CLAIM_REBIND')):
        policy=f"images: metadata claim mask {env.get('DS_META_CLAIMS','3')} (1 colour, 2 depth; 3 upstream default); claim rebind invalidation {'true' if env.get('DS_META_CLAIM_REBIND') == '1' else 'false'}"
        if policy not in text: failures.append('metadata control not confirmed (binary may predate these switches)')
    thermal_path=run/'thermal.json'
    thermal=json.loads(thermal_path.read_text()) if thermal_path.exists() else {}
    if thermal.get('aborted'): failures.append('thermal guard aborted')
    crashes=[str(p.relative_to(run)) for p in (run/'user').rglob('crash.log') if p.stat().st_size]
    if crashes: failures.append('nonempty crash report')
    for flip in flips:
        path=Path(data['shots'].get(flip,''))
        if flip not in data['shots'] or not path.is_file(): failures.append(f'missing capture at flip {flip}'); continue
        if not path.resolve().is_relative_to(run.resolve()): failures.append('capture escaped output directory'); continue
        if begin is None or flip<=begin: failures.append(f'flip {flip} is before confirmed stationary gameplay')
        row=dict(flip=flip,path=str(path.relative_to(run)),sha256=digest(path),**brightness(path))
        if (row['width'],row['height'])!=(3840,2160): failures.append(f'flip {flip} is not 3840x2160')
        row['exposure_nearby']=[s for s in data['exposure'] if abs(s['flip']-flip)<=2]
        samples.append(row)
    return dict(valid=not failures,failures=failures,exit=code,stop_reason=stop_reason,samples=samples,
        completed_capture_stop=capture_stop,
        stationary_begin_upper_flip=begin,tracking=data['tracking'],identity=data['identity'],
        exposure=data['exposure'],thermal=thermal,crash_reports=crashes,
        note='Fixed absolute guest video flips, not seconds or generated/present IDs. Review pose/UI. '
             'Mean RGB is encoded SDR code value, not physical luminance. Optional taps use a delayed GPU ring '
             '(32 taps, no per-slot fence); treat GPU/guest mismatch as a lead, not proof of coherence loss.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',required=True,type=Path)
    p.add_argument('--app',default=os.environ.get('DS_GAME_DIR'))
    p.add_argument('--save-seed',type=Path,default=Path('local-data/linux-port/save-seed'))
    p.add_argument('--shader-seed',type=Path,help='optional immutable store; otherwise start a private empty store')
    p.add_argument('--flips',default='3600,3900,4200',help='absolute video-flip captures; identical list for every matrix arm')
    p.add_argument('--seconds',type=int,default=600,help='wall bound including startup (30..1800)')
    p.add_argument('--env',action='append',default=[],metavar='DS_NAME=VALUE')
    p.add_argument('--tap-exposure',action='store_true',help='separate diagnostic: GPU/guest TAA exposure taps (changes synchronization)')
    p.add_argument('--trace-clears',action='store_true',help='log metadata fills/transfers/resolves for three flips ending at the first capture')
    p.add_argument('--label',default='check')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not args.app: p.error('set DS_GAME_DIR or pass --app /path/to/PPSA01341-app0')
    flips=sorted(set(int(s) for s in args.flips.split(',')))
    if not 1<=len(flips)<=8 or flips[0]<1 or flips[-1]>50000 or not 30<=args.seconds<=1800: p.error('invalid flip list or seconds')
    if not re.fullmatch('[A-Za-z0-9_-]+',args.label): p.error('label must contain letters, digits, _ or -')
    extra=overrides(args.env)
    repo=Path(__file__).resolve().parent.parent
    binary=args.binary.resolve(strict=True); game=Path(args.app).resolve(strict=True); seed=args.save_seed.resolve(strict=True)
    shader=args.shader_seed.resolve(strict=True) if args.shader_seed else None
    if not binary.is_file() or not os.access(binary,os.X_OK) or not (seed/'saves').is_dir(): p.error('executable binary and save seed containing saves/ required')
    if shader is not None and not shader.is_dir(): p.error('shader seed must be a directory')
    run=repo/'local-data/linux-port/exposure'/f'run-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}-{args.label}'
    prepare.protect_output(run.parent,game,os.environ)
    if any(run.is_relative_to(s) for s in (seed, binary, *((shader,) if shader else ()))): p.error('output must not be inside an input seed')
    env=environment(os.environ,run,game,flips,extra,args.tap_exposure)
    if args.trace_clears: env['DS_TRACE_CLEARS_FLIPS']=f'{max(1,flips[0]-2)}:{flips[0]}'
    env['CARGO_HOME']=str(repo/'.cargo-home')
    command=['timeout','-k','5s',str(args.seconds+15)+'s',str(binary),'--app',str(game),'--offscreen','--relay-continue','--run-seconds',str(args.seconds)]
    plan=dict(label=args.label,binary=str(binary),binary_sha256=digest(binary),game=str(game),save_seed=str(seed),shader_seed=str(shader) if shader else None,
        save_hashes={str(p.relative_to(seed)):digest(p) for p in sorted((seed/'saves').rglob('*')) if p.is_file()},
        flips=flips,command=command,environment={k:v for k,v in env.items() if k.startswith(('DS_','DSPORT_','SDL_','XDG_','__GL_','MESA_','CUDA_','VK_','CARGO_'))},output=str(run),
        note='No build, hidden offscreen native4K, FG off, 30fps, stationary <=180s after readiness. Stop after fixed captures; bootstrap polls stop file, so final flip may overshoot. Not a performance test.')
    if args.dry_run: print(json.dumps(plan,indent=2)); return 0
    import numpy  # Fail before launching if image dependencies are missing.
    from PIL import Image
    with linux_guard.cpu_heavy_lock():
        run.mkdir(parents=True)
        (run.parent/'latest.txt').write_text(str(run)+'\n')
        (run/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        (run/'user').mkdir()
        linux_snapshot.copy_tree(seed/'saves',run/'user/saves')
        for name in ('pcsettings.txt','controls.txt','display.txt'):
            src=seed/name
            if src.is_file(): (run/'user'/name).write_bytes(src.read_bytes())
        pc=run/'user/pcsettings.txt'
        pc.write_text(settings(pc.read_text() if pc.exists() else ''))
        if shader: linux_snapshot.copy_tree(shader,run/'shaders')
        else: (run/'shaders').mkdir()
        for d in ('tmp','driver/nvidia','driver/mesa','driver/cuda','xdg-cache','xdg-config','xdg-data','xdg-state'): (run/d).mkdir(parents=True,exist_ok=True)
        reason='runtime ended before fixed captures'; process=None
        tail=LogTail(run/'user'); state=None
        try:
            with (run/'console.log').open('wb') as console:
                process=linux_guard.Process(command,output=run,env=env,start_new_session=True,stdout=console,stderr=console)
                stop_at=None
                while process.poll() is None:
                    state=evidence(tail.read(),state)
                    complete=all(n in state['shots'] for n in flips)
                    if stop_at is None and (complete or state['ended'] or state['failed']):
                        reason='fixed captures complete' if complete else 'stationary window ended or runtime failed'
                        process.stop_file.write_text('exposure check finished\n'); stop_at=time.monotonic()
                    if stop_at is not None and time.monotonic()-stop_at>15:
                        process.stop(); break
                    time.sleep(.25)
                code=process.wait()
        except (OSError, ValueError, linux_guard.GuardError, KeyboardInterrupt) as error:
            (run/'summary.json').write_text(json.dumps(dict(valid=False, failures=[str(error)]),indent=2)+'\n')
            (run/'summary.txt').write_text('INVALID: '+str(error)+'\n')
            raise
        finally:
            if process: process.stop(); process.wait()
        text='\n'.join(log.read_text(errors='replace') for log in sorted((run/'user').glob('linux-runs/*/dsport.log')))
        result=report(run,flips,env,code,text,reason)
        result['binary_sha256']=plan['binary_sha256']
        result['pipeline_reuse']=[json.loads(p.read_text()) for p in (run/'user').glob('linux-runs/*/pipeline-reuse.json')]
        result['compile_warnings']=[l for l in text.splitlines() if any(s in l for s in ('compiled ahead:', 'pipelines made in the background:', 'shader preparation:'))]
        clear_lines=[l for l in text.splitlines() if 'trace clears:' in l]
        (run/'clear-trace.txt').write_text('\n'.join(clear_lines)+'\n')
        result['clear_trace_lines']=len(clear_lines)
        if args.trace_clears and not clear_lines:
            result['valid']=False;result['failures'].append('no bounded clear trace; verify binary supports DS_TRACE_CLEARS_FLIPS')
        (run/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        lines=[f"{args.label}: {'VALID' if result['valid'] else 'INVALID'}; exit {code}; {reason}",f"binary sha256 {plan['binary_sha256']}",f"requested tracker {env['DS_LINUX_WRITE_WATCH']}; FG OFF; native4K offscreen"]
        lines += [f"flip {s['flip']}: RGB mean {s['full']['mean_rgb']:.6f}, centre {s['centre']['mean_rgb']:.6f}, white {s['full']['near_white_fraction']:.4%}; {s['path']}" for s in result['samples']]
        lines += ['Exposure samples: '+str(len(result['exposure']))+(' (diagnostic ring, see summary.json)' if args.tap_exposure else ' (use --tap-exposure in a separate run)'),*result['failures'],result['note']]
        (run/'summary.txt').write_text('\n'.join(lines)+'\n')
        print('\n'.join(lines));print(run)
        return 0 if result['valid'] else 1


if __name__=='__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try: sys.exit(main())
    except (OSError,ValueError,linux_guard.GuardError,KeyboardInterrupt) as e:
        print(f'Exposure check stopped: {e}',file=sys.stderr);sys.exit(75)
