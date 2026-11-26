#!/usr/bin/env python3
"""Native 4K saved-scene relay: clean 30s stationary/60s walking, separate GPU diagnosis."""
import argparse
import hashlib
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import linux_guard
import linux_perf
import linux_snapshot


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


relay = module('linux-relay')
prepare = module('linux-prepare-shaders')


def latest_seed(root):
    for run in sorted(root.glob('run-*'), reverse=True):
        try: c = json.loads((run/'comparison.json').read_text())
        except (OSError, ValueError): continue
        # A thermally interrupted FPS test can still establish successful binary replay.
        if (c.get('matched_pipelines', 0) >= 100 and c.get('matched_hit_fraction', 0) >= .99
                and not c.get('matched_refusals', 0) and not c.get('store_write_failures', 0)
                and not c.get('fresh_replay_failed', 0) and (run/'shaders').is_dir()):
            return run
    raise ValueError('No verified reuse seed found; set DS_RELAY_SHADER_SEED or --shader-seed')


def settings(text):
    values = dict(WindowMode=1, RenderScale=6, AntiAliasing=0, AspectRatio=1, FrameRateLimit=2)
    skip = tuple('Setting_PC_'+k for k in [*values, 'SaveFolder', 'ImportSaves'])
    lines = [l for l in text.splitlines() if not l.strip().startswith(skip)]
    return '\n'.join(lines+[f'Setting_PC_{k}={v}' for k,v in values.items()])+'\n'


def environment():
    return dict(DS_RELAY_STEADY='1', DS_RELAY_RENDER_SIZE='3840x2160', DS_OUTPUT_SIZE='3840x2160',
        DS_RELAY_WINDOW_SIZE='2400x1350', DS_UPSCALER='off', DS_FRAME_CAP='0', DS_UNCAPPED='1',
        DS_RELAY_FULL='0', DS_RELAY_UI_PROBE='0', DS_RELAY_IDLE_COMPARE='0', DS_PIPELINE_BINARIES='1',
        DS_PB_VERIFY_CAPTURE='0', DS_SPIRV_NONCE='0', DS_SIM_PB_REFUSE='0', DS_RELAY_TARGETS='0',
        DS_LINUX_INTERVAL_SHOTS='0', DS_PRESENT_STATS='1', SDL_AUDIO_DRIVER='dummy')


def fps_stats(intervals):
    ms = [v/1000 for v in intervals if v > 0]
    p = relay.present_times.percentiles(ms)
    if ms:
        fps = sorted(1000/v for v in ms)
        import math
        p.update(mean_fps=1000/p['mean_ms'], fps_percentiles={f'p{q}':fps[max(0, math.ceil(len(fps)*q/100)-1)] for q in (1, 10, 50, 90, 99)})
    return p


def fault_cost(counters, flips):
    n = counters.get('fault_samples',0)
    if not n: return dict(samples=0, estimated_cpu_ms_per_flip=None, estimated_wall_ms_per_flip=None)
    faults = counters['direct_faults']
    return dict(samples=n, mean_cpu_us=counters['fault_sample_cpu_ns']/n/1000,
        mean_wall_us=counters['fault_sample_wall_ns']/n/1000,
        estimated_cpu_ms_per_flip=counters['fault_sample_cpu_ns']/n*faults/max(flips,1)/1e6,
        estimated_wall_ms_per_flip=counters['fault_sample_wall_ns']/n*faults/max(flips,1)/1e6,
        note='Sampled direct recovery including mprotect/TLB, excludes kernel signal entry/return and service faults. Sum across threads, not frame critical path. Timing syscalls perturb samples; not total fault cost.')


def phase_report(data, reuse):
    seconds, count = data['seconds'], data['counters']['flips']
    threads = []
    groups = {}
    system = {}
    for t in data['threads']:
        name = t['name']
        group = 'job-workers' if name.startswith('BPE JobWorker') else relay.category(name)
        cpu = 1000*t['cpu_s']/max(count,1)
        groups[group] = groups.get(group,0)+cpu
        if 'system_s' in t: system[group] = system.get(group,0)+1000*t['system_s']/max(count,1)
        threads.append(dict(t, cpu_ms_per_flip=cpu, runqueue_ms_per_flip=1000*t['runqueue_s']/max(count,1),
                            not_running_ms_per_flip=1000*max(0,seconds-t['cpu_s']-t['runqueue_s'])/max(count,1)))
    begin, end = reuse.get('begin',{}), reuse.get('end',{})
    pipeline = {k:max(0,end.get(k,0)-begin.get(k,0)) for k in ('source_create_calls','source_create_ms','source_create_started','binary_hits','binary_total_ms','cache_create_calls','cache_create_ms','binary_refused','missing','store_write_failures')}
    pipeline.update(source_active_begin=begin.get('source_create_active'),source_active_end=end.get('source_create_active'))
    warm = all(k in begin and k in end for k in ('source_create_calls','source_create_ms','source_create_started','source_create_active')) and all(pipeline[k]==0 for k in ('source_create_calls','source_create_ms','source_create_started','source_active_begin','source_active_end'))
    return dict(seconds=seconds, flips=count, flips_per_second=count/max(seconds,.001),
        frame_times=fps_stats([v[0] for v in data['intervals_us_and_flip']]),
        rates_per_second={k:v/max(seconds,.001) for k,v in data['counters'].items()},
        per_flip={k:v/max(count,1) for k,v in data['counters'].items()},
        cpu_ms_per_flip_by_group=groups, system_cpu_ms_per_flip_by_group=system, threads=sorted(threads,key=lambda t:-t['cpu_ms_per_flip']),
        fault_cost=fault_cost(data['counters'],count),
        jobs=data['job_workers'], pipelines=pipeline, fully_warm=warm,
        upscaler=data['upscaler'])


def phase_thermal(data, thermal, monitor):
    """Never subtract a pause from FPS. Invalidate only the phase it overlaps.
    Old logs lack exact boundary clocks: preserve their conservative invalidation.
    """
    if thermal.get('aborted'): return dict(valid=False, reason='thermal/guard abort')
    if thermal.get('measurement_valid'): return dict(valid=True, paused_s=0)
    begin, end = data.get('begin_monotonic_ns'), data.get('end_monotonic_ns')
    if begin is None or end is None or not monitor.exists():
        return dict(valid=False, reason='thermal pause/abort (missing phase clocks)')
    start=None; pauses=[]
    for line in monitor.read_text().splitlines():
        try: event=json.loads(line)
        except ValueError: continue
        if event.get('event')=='pause': start=event['monotonic_s']
        elif event.get('event')=='resume' and start is not None:
            pauses.append((start,event['monotonic_s']));start=None
    if start is not None: pauses.append((start,float('inf')))
    if len(pauses)!=thermal.get('pauses'): return dict(valid=False, reason='incomplete thermal log')
    overlap=sum(max(0,min(b,end/1e9)-max(a,begin/1e9)) for a,b in pauses)
    return dict(valid=overlap==0,paused_s=overlap,session_pauses=thermal.get('pauses',0))


def phase_cpu_environment(data, monitor):
    begin, end = data.get('begin_monotonic_ns',0)/1e9, data.get('end_monotonic_ns',0)/1e9
    rows=[]
    if monitor.exists():
        for line in monitor.read_text().splitlines():
            try: r=json.loads(line)
            except ValueError: continue
            if begin <= r.get('monotonic_s',-1) <= end and r.get('event') != 'cooldown': rows.append(r)
    temps=[r['temperature_c'] for r in rows if 'temperature_c' in r]
    cpus={}
    for r in rows:
        for cpu, fields in r.get('cpu_telemetry',{}).items():
            for k,v in fields.items():
                if v is not None: cpus.setdefault(cpu,{}).setdefault(k,[]).append(v)
    for fields in cpus.values():
        for k,v in list(fields.items()):
            fields[k]=dict(min=min(v),max=max(v),mean=sum(v)/len(v),delta=v[-1]-v[0] if 'throttle' in k else None)
    return dict(samples=len(rows),temperature_c=dict(min=min(temps),max=max(temps),mean=sum(temps)/len(temps)) if temps else None,
                cpus=cpus,trace='monitor.jsonl',note='Cumulative throttle deltas are per logical CPU, not additive package totals; unavailable fields omitted.')


def report(run, metrics, diagnostic=False):
    logs=sorted((run/'user').rglob('dsport.log'))
    text='\n'.join(p.read_text(errors='replace') for p in logs)
    phases={}
    flip_ranges={}
    for p in sorted((run/'user').rglob('*-steady.json')):
        data=json.loads(p.read_text()); label=data['phase']
        reuse=json.loads(p.with_name(label+'-pipeline-reuse.json').read_text())
        iv=data['intervals_us_and_flip']
        if iv: flip_ranges[label]=(iv[0][1],iv[-1][1])
        phases[label]=phase_report(data,reuse)
        self_path=p.with_name(label+'-cpu-self.json')
        phases[label]['recording_self']=json.loads(self_path.read_text()) if self_path.exists() else {}
        phases[label]['presentation']=metrics['presentation'].get(label,{})
        phases[label]['thermal']=phase_thermal(data,metrics['thermal'],run/'monitor.jsonl')
        phases[label]['cpu_environment']=phase_cpu_environment(data,run/'monitor.jsonl')
    for trace in metrics['timelines']:
        trace['thermal']=phase_thermal(trace,metrics['thermal'],run/'monitor.jsonl')
        match=re.search(r'timeline-(\d+)',trace.get('source_file',''))
        if match:
            flip=int(match[1])
            within=[label for label,(a,b) in flip_ranges.items() if a<=flip<=b]
            preceding=sorted((b,label) for label,(a,b) in flip_ranges.items() if b<flip)
            trace['phase']=within[0] if within else (preceding[-1][1] if preceding else 'unknown')
    sizes=re.findall(r'render size (\d+)x(\d+) set',text)
    swapchains=re.findall(r'swapchain[^\n]*?(\d+)x(\d+)',text)
    reasons=[]
    if metrics['exit_code'] or metrics['validation_errors'] or metrics['crash_bytes']: reasons.append('runtime/validation/crash failure')
    if metrics['thermal'].get('aborted'): reasons.append('thermal/guard abort')
    validation = 'validation' in metrics.get('profile','')
    if metrics['effective_validation'] is not validation: reasons.append('requested validation state not confirmed')
    if not sizes or sizes[-1] != ('3840','2160'): reasons.append('native guest render size not confirmed')
    if not swapchains or swapchains[-1] != ('3840','2160'): reasons.append('3840x2160 swapchain not confirmed; wrong monitor/size')
    if 'SDL window mode 1 applied' not in text: reasons.append('fullscreen not confirmed')
    if 'steady route complete: stationary then walking' not in text: reasons.append('route incomplete')
    for label, seconds in [('stationary',5 if diagnostic else 30), ('walking',5 if diagnostic else 60)]:
        p=phases.get(label,{})
        if not p:
            reasons.append(label+' not measured (gameplay/phase not reached)')
            continue
        if not p.get('thermal',{}).get('valid'): reasons.append(label+' thermal pause/abort')
        if p.get('seconds',0)<seconds or p.get('flips',0)<30: reasons.append(label+' insufficient samples')
        if p.get('upscaler') != 'Off': reasons.append(label+' upscaler not disabled')
        if not diagnostic and not p.get('recording_self',{}).get('valid'): reasons.append(label+' missing/invalid TSC self profile')
        if not validation and not p.get('fully_warm',False): reasons.append(label+' compiled pipelines inside measurement')
        if not p.get('presentation',{}).get('wait_complete',{}).get('samples',0): reasons.append(label+' no KHR completion samples')
    for trace in metrics['timelines']:
        if not trace['thermal']['valid']:
            reason=trace.get('phase','trace')+' thermal pause/abort'
            if reason not in reasons: reasons.append(reason)
    if diagnostic and not validation and 'sample' not in metrics.get('profile','') and not any(t.get('gpu_timestamp_spans',0)>0 for t in metrics['timelines']): reasons.append('no diagnostic GPU timestamps')
    if not diagnostic:
        for label in ('stationary','walking'):
            if not any(t.get('phase')==label and t.get('frames',0)>=2 and not t.get('gpu_timestamp_spans',0) for t in metrics['timelines']):
                reasons.append(label+' missing adjacent CPU wait trace')
    if metrics['write_watch_requested'] != metrics['write_watch']: reasons.append('requested tracking backend unavailable/fell back')
    profile=metrics.get('profile','')
    if re.match(r'steady-[abc]\d+-(baseline|whole-run|whole-run-quick)-',profile):
        kind=candidate_of(profile)
        expected=f"Linux whole-run recording: {str(kind.startswith('whole-run')).lower()}"
        if expected not in text: reasons.append('recording candidate switches not confirmed')
        quick = kind=='whole-run-quick'
        enabled=os.environ.get('DS_RELAY_QUICK_WINS','tsc,static-bind').split(',')
        expected=f"Linux recording quick wins: tsc={str(quick and 'tsc' in enabled).lower()} static_bind={str(quick and 'static-bind' in enabled).lower()}"
        if expected not in text: reasons.append('quick-win switches not confirmed')

    whole=whole_run_counts(text)
    if candidate_of(profile).startswith('whole-run') and whole['runs'] == 0:
        reasons.append('whole-run path not exercised')

    screenshots=sorted(str(p.relative_to(run)) for p in (run/'user').rglob('*steady-*.png'))
    if len(screenshots)<4: reasons.append('missing boundary screenshots')
    gpu={}
    with (run/'gpu-clocks.csv').open() as f:
        for r in csv.DictReader(f):
            if not r.get('phase','').startswith('steady-'): continue
            key=(r['phase'],r['index'].strip())
            row=gpu.setdefault(str(key),{})
            for k in ('gpu_util_pct','power_W','graphics_MHz'):
                try: row.setdefault(k,[]).append(float(r[k]))
                except ValueError: pass
    for d in gpu.values():
        for k,v in list(d.items()): d[k]=dict(samples=len(v),mean=sum(v)/len(v),min=min(v),max=max(v))
    cooled=cooldown_report(run/'monitor.jsonl')
    if re.match(r'steady-[abc]\d+-',metrics.get('profile','')) and not cooled['valid']: reasons.append('session cooling to <=70C not confirmed')
    result=dict(translation_identity=[l for l in text.splitlines() if 'fingerprint' in l.lower() or 'pipeline binaries:' in l.lower()],whole_run=whole,cooldown=cooled,valid_steady_state=not reasons and not diagnostic, diagnostic=diagnostic, invalid_reasons=reasons,
        recording_candidate=candidate_of(metrics.get('profile','')), address_wake=True, pair=int(re.search(r'steady-[abc](\d+)-',metrics.get('profile','')).group(1)) if re.search(r'steady-[abc](\d+)-',metrics.get('profile','')) else 1,
        exit_code=metrics['exit_code'], selfplay=metrics.get('selfplay',[]),
        phases=phases, gpu_samples=gpu, waits=metrics['timelines'], display=[l for l in text.splitlines() if any(t in l for t in ('SDL display:', 'swapchain ', 'SDL window mode', 'pc settings: state:'))], requested_tracking=metrics['write_watch_requested'],
        actual_tracking=metrics['write_watch'], screenshots=screenshots,
        note='CPU work and concurrent wait lanes are not additive frame wall time. GPU utilization is sampled device-wide duty, not per-frame active execution. Diagnostic GPU batch spans include barriers/waits, and are not clean FPS. KHR completion is not physical scanout.')
    (run/'steady-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def needs_warm_retry(result):
    reasons=result['invalid_reasons']
    paused=any(r in ('stationary thermal pause/abort','walking thermal pause/abort') for r in reasons)
    return bool(reasons) and all(r.endswith('compiled pipelines inside measurement')
        or r in ('stationary thermal pause/abort','walking thermal pause/abort')
        or (paused and r.endswith(('insufficient samples','no KHR completion samples'))) for r in reasons)


def tracking_comparison(results):
    if any('-baseline-' in label or r.get('recording_candidate','baseline') != 'baseline' for label,r in results.items()):
        return dict(performance_valid=False,pairs={},correctness_verified=False,reason='recording comparison uses mprotect throughout; see recording-comparison.json')
    groups={}
    for label,r in results.items():
        if r['diagnostic'] or r.get('warmup_only') or r.get('recording_candidate','baseline') != 'baseline': continue
        groups.setdefault(r.get('pair',1),{})[r.get('actual_tracking','unknown')] = (label,r)
    pairs={}
    for index,selected in sorted(groups.items()):
        if not all(k in selected and selected[k][1]['valid_steady_state'] for k in ('mprotect','uffd')):
            pairs[str(index)] = dict(performance_valid=False,reason='pair needs fully warm, cooled, valid mprotect and uffd sessions')
            continue
        a,b=selected['mprotect'][1],selected['uffd'][1]
        phases={}
        for label in ('stationary','walking'):
            old,new=a['phases'][label],b['phases'][label]
            phases[label]=dict(fps_before=old['flips_per_second'],fps_after=new['flips_per_second'],
                fps_change_pct=100*(new['flips_per_second']/old['flips_per_second']-1),
                temperature_before=old.get('cpu_environment',{}).get('temperature_c'),
                temperature_after=new.get('cpu_environment',{}).get('temperature_c'),
                faults_s_before=old['rates_per_second'].get('faults'),faults_s_after=new['rates_per_second'].get('faults'),
                cpu_ms_per_flip_before=old['cpu_ms_per_flip_by_group'],cpu_ms_per_flip_after=new['cpu_ms_per_flip_by_group'],
                system_cpu_ms_per_flip_before=old.get('system_cpu_ms_per_flip_by_group'),system_cpu_ms_per_flip_after=new.get('system_cpu_ms_per_flip_by_group'),
                frame_ms_before=old['frame_times'],frame_ms_after=new['frame_times'])
        pairs[str(index)]=dict(performance_valid=True,baseline=selected['mprotect'][0],candidate=selected['uffd'][0],phases=phases,
                              cooldown_before=a.get('cooldown'),cooldown_after=b.get('cooldown'))
    valid=[v for v in pairs.values() if v['performance_valid']]
    # Preserve single-pair fields for readers of older reports, but never silently replace
    # a failed pair with the other pair. Both outcomes and temperatures are explicit.
    return dict(**(valid[0] if valid else dict(performance_valid=False,reason='no complete valid pair')),
        pairs=pairs,correctness_verified=False,note='Paired A,B,A,B; each session cools before launch. Compare temperatures and PNGs; no temperature normalization or cross-pair substitution.')


def recording_comparison(results):
    groups={}
    for label,r in results.items():
        if r['diagnostic'] or r.get('warmup_only'): continue
        groups.setdefault(r.get('pair',1),{})[r.get('recording_candidate',candidate_of(label))]=(label,r)
    pairs={}
    for index,group in sorted(groups.items()):
        for base,candidate in [('baseline','whole-run'),('baseline','whole-run-quick'),('whole-run','whole-run-quick')]:
            if candidate not in group: continue
            name=f'{candidate} pair {index}' if base=='baseline' else f'quick-wins incremental pair {index}'
            if base not in group or not all(group[k][1]['valid_steady_state'] for k in (base,candidate)):
                pairs[name]=dict(performance_valid=False,reason='requires two warm, cooled, successful sessions');continue
            a,b=group[base][1],group[candidate][1]
            if a.get('actual_tracking')!='mprotect' or b.get('actual_tracking')!='mprotect':
                pairs[name]=dict(performance_valid=False,reason='recording A/B requires mprotect in both');continue
            phases={}
            for phase in ('stationary','walking'):
                old,new=a['phases'][phase],b['phases'][phase]
                before,after=(p.get('recording_self',{}) for p in (old,new))
                rows_before={r['name']:r['self_ms_per_flip'] for r in before.get('rows',[])}
                rows_after={r['name']:r['self_ms_per_flip'] for r in after.get('rows',[])}
                phases[phase]=dict(fps_before=old['flips_per_second'],fps_after=new['flips_per_second'],
                    fps_change_pct=100*(new['flips_per_second']/old['flips_per_second']-1),
                    temperature_before=old.get('cpu_environment',{}).get('temperature_c'),temperature_after=new.get('cpu_environment',{}).get('temperature_c'),
                    gfx_cpu_before=before.get('thread_cpu_ms_per_flip',0),gfx_cpu_after=after.get('thread_cpu_ms_per_flip',0),
                    self_ms_before=rows_before,self_ms_after=rows_after,
                    self_ms_delta={k:rows_after.get(k,0)-rows_before.get(k,0) for k in rows_before.keys()|rows_after.keys()},
                    frame_ms_before=old['frame_times'],frame_ms_after=new['frame_times'])
            pairs[name]=dict(performance_valid=True,baseline=group[base][0],candidate=group[candidate][0],phases=phases)
    return dict(pairs=pairs,correctness_verified=False,note='Cooled A/B/C/A/B/C, including incremental B-to-C comparison; PNG review required before enabling by default.')


def cooldown_report(path):
    try: rows=[r for l in path.read_text().splitlines() if (r:=json.loads(l)).get('event') in ('cooldown','cooldown-abort')]
    except (OSError,ValueError): return dict(valid=False,reason='cooldown evidence missing')
    if not rows: return dict(valid=False,reason='empty cooldown trace')
    last=rows[-1]
    target=last.get('target_c')
    return dict(valid=last.get('event')=='cooldown' and target is not None and 0<target<=70 and last['temperature_c']<=target,
        start_temperature_c=rows[0].get('temperature_c'),end_temperature_c=last.get('temperature_c'),target_c=target,
        seconds=last.get('monotonic_s',0)-rows[0].get('monotonic_s',0))


def summary(results, sessions=None):
    lines=['Native 3840x2160, TAA/no upscaler, fullscreen FIFO; 30s stationary + 60s walking.',
           'Clean intervals exclude screenshots and all timelines/GPU timestamps. Diagnostic session is separate.',
           'CPU ms/flip is work; overlapping thread waits must NOT be summed. See per-run steady-summary.json, frame-waits.csv and thread-times.csv.']
    if sessions is not None:
        lines.append('\nPlanned sessions (independent; a failure does not cancel later sessions):')
        for label,entry in sessions.items():
            lines.append(f'  {label}: {entry["state"]}; attempts '+', '.join(entry['attempts']))
    for label,r in results.items():
        lines.append(f'\n{label}: warmup-only={r.get("warmup_only",False)}, valid steady state={r["valid_steady_state"]}; '+', '.join(r['invalid_reasons']))
        if 'sampling' in r:
            lines.append('  gpu-gfx sampling: '+str(r['sampling'].get('valid'))+'; see gpu-gfx-top20.txt and gpu-gfx-perf-stacks.txt')
            lines += [f'    {v["self_percent"]:.2f}% self CPU: {v["symbol"]}' for v in r['sampling'].get('top20_self',[])]
        if 'cooldown' in r: lines.append('  pre-launch cooling: '+str(r['cooldown']))
        lines.append(f'  runtime exit: {r.get("exit_code")}; '+str(r.get('selfplay',[])))
        if 'whole_run' in r: lines.append('  whole-run coverage (session totals): '+str(r['whole_run']))
        for phase,p in r['phases'].items():
            if not p.get('thermal',{}).get('valid',False):
                lines.append(f'{phase}: INVALID thermal pause/abort; raw timings retained in JSON only, excluded from performance comparison.')
                continue
            if r.get('warmup_only') or r.get('diagnostic') or not p.get('fully_warm',False):
                state = 'UNMEASURED warm-up' if r.get('warmup_only') else ('DIAGNOSTIC (not performance)' if r.get('diagnostic') else 'INVALID compile-contaminated')
                lines.append(f'{phase}: {state}; source counters {p.get("pipelines",{})}. Raw timings retained in JSON only.')
                continue
            cost=p.get('fault_cost',{})
            if cost.get('samples'):
                lines.append(f'  {phase} direct-fault recovery estimate: {cost["estimated_cpu_ms_per_flip"]:.3f} CPU ms/flip, {cost["estimated_wall_ms_per_flip"]:.3f} summed wall ms/flip ({cost["samples"]} samples; EXCLUDES kernel signal entry/return).')
            t=p['frame_times']; present=p['presentation'].get('wait_complete',{})
            lines.append(f'{phase}: {p["flips_per_second"]:.2f} flips/s; frame ms p50/p90/p99/max '+
                '/'.join(f'{t.get(k,0):.2f}' for k in ('p50_ms','p90_ms','p99_ms','max_ms'))+
                '; KHR completion ms '+ '/'.join(f'{present.get(k,0):.2f}' for k in ('p50_ms','p90_ms','p99_ms','max_ms')))
            lines.append(f'  faults/s {p["rates_per_second"].get("faults",0):.0f}; upload/writeback MB/flip '+
                '/'.join(f'{p["per_flip"].get(k,0)/1e6:.2f}' for k in ('upload_bytes','writeback_bytes'))+
                f'; pipeline source calls/ms {p["pipelines"]["source_create_calls"]}/{p["pipelines"]["source_create_ms"]:.1f}')
            lines.append('  CPU temperature/clock/throttle trace monitor.jsonl: '+str(p.get('cpu_environment',{}).get('temperature_c')))
            lines.append('  CPU ms/flip '+str({k:round(v,2) for k,v in p['cpu_ms_per_flip_by_group'].items()}))
            if p.get('system_cpu_ms_per_flip_by_group'):
                lines.append('  of which kernel CPU ms/flip '+str({k:round(v,2) for k,v in p['system_cpu_ms_per_flip_by_group'].items()}))
            own=p.get('recording_self',{})
            if own:
                lines.append(f'  Adjacent gpu-gfx self profile: valid={own.get("valid")}, {own.get("flips")} flips; thread CPU {own.get("thread_cpu_ms_per_flip",0):.2f} ms/flip. Self TSC elapsed includes preemption/overhead; child scopes/waits excluded:')
                lines += [f'    {v["self_ms_per_flip"]:.3f} ms/flip {v["name"]} ({v["calls"]} calls)' for v in own.get('top10',[])]
        if r['waits']:
            for trace in r['waits']:
                if not trace.get('thermal',{}).get('valid',True):
                    lines.append(f'  {trace.get("phase","unknown")} trace: INVALID thermal pause; raw trace retained, timings omitted.')
                    continue
                if trace.get('label_waits'):
                    lines.append('    gpu-gfx label producers: '+str(trace['label_waits'])+' (whole waits only)')
                lines.append(f'  {"GPU diagnostic" if trace.get("gpu_timestamp_spans",0) else ("Detailed CPU trace" if trace.get("cpu_detail_spans",0) else "Adjacent CPU-only trace")} {trace.get("phase","unknown")}: {trace["frames"]} flips, {trace["mean_frame_ms"]:.2f} ms/flip (NOT baseline FPS)')
                for lane,cats in trace.get('mean_ms_by_track',{}).items():
                    if lane in ('GPU','gpu-gfx','gpu-submit','gpu-release','vblank','vk submit') or lane.startswith('GPU segments /'):
                        lines.append('    '+lane+': '+', '.join(f'{k} {v:.2f}' for k,v in sorted(cats.items(),key=lambda x:-x[1])[:5])+' ms/flip')
    lines+=['\nCompare the stationary/walking before/after PNGs for correct geometry, lighting and HUD. Automated timings cannot establish visual correctness.',
            'A 60Hz display limits FIFO to its refresh. Offscreen timing is NOT equivalent to visible presentation. The game is uncapped for this test; normal player settings are untouched.']
    recording=recording_comparison(results)
    for key,pair in recording['pairs'].items():
        if not pair['performance_valid']:
            lines.append(f'Recording {key}: INVALID: '+pair['reason']); continue
        for phase,p in pair['phases'].items():
            lines.append(f'Recording {key} {phase}: {p["fps_before"]:.2f} -> {p["fps_after"]:.2f} fps ({p["fps_change_pct"]:+.1f}%); temperatures A={p["temperature_before"]}, B={p["temperature_after"]}; gpu-gfx CPU {p["gfx_cpu_before"]:.2f} -> {p["gfx_cpu_after"]:.2f} ms/flip. PNG correctness requires review.')
    comparison=tracking_comparison(results)
    for index,pair in comparison['pairs'].items():
        if not pair['performance_valid']:
            lines.append(f'Tracking pair {index}: INVALID: '+pair['reason']); continue
        for phase,p in pair['phases'].items():
            lines.append(f'Tracking pair {index} {phase}: mprotect {p["fps_before"]:.2f} -> uffd {p["fps_after"]:.2f} fps ({p["fps_change_pct"]:+.1f}%); temperatures A={p["temperature_before"]}, B={p["temperature_after"]}. PNG correctness requires review.')
    if not comparison['pairs']: lines.append('Tracking A/B: '+comparison.get('reason','no paired results yet'))
    return '\n'.join(lines)+'\n'


def whole_run_counts(text):
    rows=re.findall(r'whole-run recording: enabled (true|false), runs (\d+), commands (\d+)',text)
    return dict(runs=max((int(n) for _,n,_ in rows),default=0),commands=max((int(n) for _,_,n in rows),default=0))


def candidate_of(label):
    return next((k for k in ('whole-run-quick','whole-run') if '-'+k+'-' in label), 'baseline')


def session_plan(variants, gpu_timing=True, sampling=False, candidate=None):
    if candidate:
        return [f'steady-{arm}{pair}-{v}-mprotect' for pair in (1,2) for arm,v in [('a','baseline'),('b','whole-run'),('c','whole-run-quick')]] + ['steady-validation-whole-run-quick-mprotect'] + (['steady-diagnostic-whole-run-quick-mprotect'] if gpu_timing else [])
    return [f'steady-{chr(97+i)}{pair}-{v}' for pair in (1,2) for i,v in enumerate(variants)] + (['steady-sample-mprotect'] if sampling else []) + (['steady-diagnostic-'+variants[0]] if gpu_timing else [])


def warm_store(labels, limit, run_one, persist):
    """A complete zero-source cycle proves this route/store for every variant. No timing
    session starts with an unknown seed. Private store grows; caller's seed is read-only.
    """
    results, sessions, clean = {}, {}, set()
    for i in range(limit):
        label=labels[i % len(labels)]
        name=f'warmup-{i+1}-'+label.removeprefix('steady-')
        sessions[name]=dict(state='warming',attempts=[name]);persist(results,sessions)
        try: result=run_one(label,0,name)
        except Exception as e:
            sessions[name]['state']='failed';persist(results,sessions)
            raise ValueError(f'warm-up {name} failed; no performance sessions started: {e}') from e
        result['warmup_only']=True;result['valid_steady_state']=False;results[name]=result
        reasons=result['invalid_reasons']
        failures=[r for r in reasons if not r.endswith('compiled pipelines inside measurement')]
        if failures:
            sessions[name]['state']='failed';persist(results,sessions)
            raise ValueError(f'warm-up {name} failed; no performance sessions started: '+', '.join(failures))
        if not reasons: clean.add(label)
        else: clean.clear()
        sessions[name]['state']='warm' if not reasons else 'compiled (unmeasured)';persist(results,sessions)
        if len(clean)==len(labels): return results,sessions
    raise ValueError(f'Warm store not established after {limit} dedicated route passes; no performance sessions started. Reuse prepared-shaders next time; inspect warm-up source calls.')


def run_suite(labels, run_one, persist):
    """Every session has its own process/timeout/cooldown. Preserve partial results even
    when launch, gameplay or analysis fails. Explicit cancellation alone ends the suite.
    """
    results={}
    sessions={label:dict(state='pending',attempts=[]) for label in labels}
    persist(results,sessions)
    try:
        for label in labels:
            diagnostic=any(k in label for k in ('diagnostic','sample','validation'))
            for attempt in range(1 if diagnostic else 2):
                name=label if attempt==0 else label.replace('steady-','steady-warm-',1)
                sessions[label]['state']='running'
                sessions[label]['attempts'].append(name)
                persist(results,sessions)
                try:
                    result=run_one(label,attempt,name)
                except Exception as e:
                    # run_session reaps its process before analysis; retain the traceback
                    # without sacrificing the other independent measurements.
                    result=dict(valid_steady_state=False,diagnostic=diagnostic,recording_candidate=candidate_of(label),
                        invalid_reasons=[f'session launch/analysis failed: {type(e).__name__}: {e}'],
                        exit_code=None,selfplay=[],phases={},waits=[],actual_tracking='unknown',pair=int(re.search(r'steady-[abc](\d+)-',label).group(1)) if re.search(r'steady-[abc](\d+)-',label) else 1,error_traceback=traceback.format_exc())
                retry=not diagnostic and not re.match(r'steady-[abc]\d+-',label) and attempt==0 and needs_warm_retry(result)
                if retry:
                    result['warmup_only']=True
                    result['retry_reasons']=result['invalid_reasons']
                results[name]=result
                sessions[label]['state']='warming' if retry else ('failed' if result['invalid_reasons'] else 'completed')
                persist(results,sessions)
                if result.get('exit_code')==130:
                    sessions[label]['state']='cancelled'
                    raise KeyboardInterrupt('session interrupted')
                if not retry: break
    except KeyboardInterrupt:
        for entry in sessions.values():
            if entry['state'] in ('pending','running','warming'):
                entry['state']='cancelled' if entry['attempts'] else 'not run (user cancelled)'
        persist(results,sessions)
        raise
    return results,sessions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',default=os.environ.get('DS_GAME_DIR'))
    parser.add_argument('--save-seed',default=os.environ.get('DS_RELAY_SAVE_SEED'))
    parser.add_argument('--shader-seed',default=os.environ.get('DS_RELAY_SHADER_SEED'))
    parser.add_argument('--record-candidate',choices=('whole-run',),help='cooled A/B/C/A/B/C: baseline, whole-run, whole-run plus quick wins; all remain off in normal play')
    parser.add_argument('--warmup-passes',type=int,default=9,help='bounded route passes to establish a common warm store before any measurement (default9)')
    parser.add_argument('--quick-wins',default='tsc,static-bind',help='whole-run-quick switches: tsc,static-bind, or either alone')
    parser.add_argument('--tracking',choices=('mprotect','uffd','both'),default='uffd')
    parser.add_argument('--perf-gpu-gfx',action='store_true',help='separate stationary gpu-gfx sampling run; requires perf; never included in clean FPS')
    parser.add_argument('--thermal-pause',action='store_true',help='use preparation-style SIGSTOP cooling instead of as-played measurement')
    parser.add_argument('--no-gpu-timing',action='store_true',help='omit the separate short diagnostic session')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if not 1 <= args.warmup_passes <= 12: parser.error('--warmup-passes must be 1..12')
    if not set(args.quick_wins.split(',')) <= {'tsc','static-bind'}: parser.error('--quick-wins must be tsc,static-bind or either alone')
    if args.record_candidate and args.perf_gpu_gfx: parser.error('--perf-gpu-gfx is separate from recording A/B')
    if not args.app: parser.error('set DS_GAME_DIR or pass --app /path/to/PPSA01341-app0')
    repo=Path(__file__).resolve().parent.parent
    game=Path(args.app).resolve(strict=True)
    seed=Path(args.save_seed or repo/'local-data/linux-port/save-seed').resolve(strict=True)
    verified=None if args.shader_seed else latest_seed(repo/'local-data/linux-port/pipeline-reuse')
    shaders=Path(args.shader_seed).resolve(strict=True) if args.shader_seed else verified/'shaders'
    if not (seed/'saves').is_dir() or not shaders.is_dir(): parser.error('seed needs saves/ and a warm shader store')
    output=repo/'local-data/linux-port/steady'/f'run-{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}-{os.getpid()}'
    prepare.protect_output(output.parent,game,os.environ)
    variants=['mprotect'] if args.record_candidate else (['mprotect','uffd'] if args.tracking=='both' else [args.tracking])
    os.environ['DS_RELAY_THERMAL_MODE'] = 'pause' if args.thermal_pause else 'observe'
    os.environ.setdefault('DS_RELAY_SESSION_TEMP','70')
    os.environ.setdefault('DS_RELAY_COOLDOWN_S','600')
    plan=dict(output=str(output),save_seed=str(seed),shader_seed=str(shaders),environment={**environment(), **{k:os.environ[k] for k in ('DS_RELAY_THERMAL_MODE','DS_RELAY_SESSION_TEMP','DS_RELAY_COOLDOWN_S')}},
        sessions=session_plan(variants,not args.no_gpu_timing,args.perf_gpu_gfx,args.record_candidate),
        recording_candidate=args.record_candidate,
        visible=True,fullscreen=True,render_pixels=[3840,2160],audio='null',stationary_s=30,walking_s=60,
        timing='no capture/validation/probes during clean intervals; adjacent32-flip TSC self profiles always; optional separate20s perf sampling and sampled fault recovery; GPU diagnostic LAST: 5s/5s CPU profile, bucket census and32-flip GPU segment traces, never baseline FPS',
        timeout_each_s=630,run_seconds=600,warm_retry='Dedicated unmeasured same-route passes through all variants before A/B/C. Require a complete zero-compile cycle; stop loudly if not warm. Every timed session copies the same prepared store.',warmup_passes=args.warmup_passes,quick_wins=args.quick_wins,cpu_lock='~/.cache/cpu-heavy.lock (DS_CPU_HEAVY_LOCK overrides); taken by this script, do not double-wrap with flock',cooldown='Before EACH launch, including repeats/diagnostics: <=70C (DS_RELAY_SESSION_TEMP), bounded600s (DS_RELAY_COOLDOWN_S). No session starts on cooldown failure.',cpu='DS_RELAY_CPUS or 8 logical CPUs; nice10; workers2; thermal observe per second, no SIGSTOP,105/60s abort; --thermal-pause restores preparation cooling')
    if args.dry_run: print(json.dumps(plan,indent=2)); return 0
    if not (repo/'target/release/dsport').is_file(): parser.error('build release dsport first')
    output.mkdir(parents=True); print(f'Steady relay output: {output}',flush=True); (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    snapshot=output/'seed'; snapshot.mkdir()
    linux_snapshot.copy_tree(seed/'saves',snapshot/'saves')
    (snapshot/'pcsettings.txt').write_text(settings((seed/'pcsettings.txt').read_text() if (seed/'pcsettings.txt').exists() else ''))
    os.environ.update(environment(),CARGO_HOME=str(repo/'.cargo-home'))
    caches={}
    master=output/'prepared-shaders'
    master_driver=output/'prepared-driver'
    os.environ['DS_RELAY_QUICK_WINS']=args.quick_wins
    def run_one(label,attempt,name):
        diagnostic=any(k in label for k in ('diagnostic','sample','validation'))
        warming=name.startswith('warmup-')
        cache_key = 'warmup' if warming else name
        if cache_key not in caches:
            cache=master if warming else output/(name+'-shaders')
            driver=master_driver if warming else output/(name+'-driver')
            if not warming:
                linux_snapshot.copy_tree(master,cache)
                if master_driver.is_dir(): linux_snapshot.copy_tree(master_driver,driver)
            caches[cache_key]=(cache,driver)
        cache,driver=caches[cache_key]
        run=output/name
        print(f'{name}: cooling to <=70C before launch; telemetry in {run}/monitor.jsonl',flush=True)
        metrics=relay.run_session(repo,game,run,'present','600','630',cache,label,snapshot,driver)
        result=report(run,metrics,diagnostic)
        if 'sample' in label:
            sample = run/'gpu-gfx-perf.json'
            result['sampling'] = linux_perf.finalize_phase(run) if sample.exists() else dict(valid=False,error='stationary sampling never started')
            if not result['sampling']['valid']:
                result['invalid_reasons'].append('gpu-gfx sampling failed; see gpu-gfx-perf*.log/json')
                result['valid_steady_state'] = False
        if warming:
            result['warmup_only']=True
            result['valid_steady_state']=False
        (run/'steady-summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def persist(results,sessions):
        (output/'summary.txt').write_text(summary(results,sessions))
        (output/'sessions.json').write_text(json.dumps(sessions,indent=2)+'\n')
        (output/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
        (output/'tracking-comparison.json').write_text(json.dumps(tracking_comparison(results),indent=2)+'\n')
        (output/'recording-comparison.json').write_text(json.dumps(recording_comparison(results),indent=2)+'\n')

    started_suite = False
    try:
        with linux_guard.cpu_heavy_lock():
            if args.perf_gpu_gfx: linux_perf.preflight(output)
            started_suite = True
            linux_snapshot.copy_tree(shaders,master)
            if verified and (verified/'driver').is_dir(): linux_snapshot.copy_tree(verified/'driver',master_driver)
            warm_labels=plan['sessions'][:3] if args.record_candidate else plan['sessions'][:len(variants)]
            warm_results,warm_sessions=warm_store(warm_labels,args.warmup_passes,run_one,
                lambda r,ss: persist(r,{**ss,**{l:dict(state='not run (waiting for warm store)',attempts=[]) for l in plan['sessions']}}))
            with (repo/'target/release/dsport').open('rb') as binary:
                binary_hash=hashlib.file_digest(binary,'sha256').hexdigest()
            (output/'warmup.json').write_text(json.dumps(dict(passed=True,sessions=list(warm_results),
                shader_store=str(master),driver_store=str(master_driver),binary_sha256=binary_hash,translation_identity=next(reversed(warm_results.values())).get('translation_identity',[])),indent=2)+'\n')
            results,sessions=run_suite(plan['sessions'],run_one,lambda r,ss: persist({**warm_results,**r},{**warm_sessions,**ss}))
            results={**warm_results,**results};sessions={**warm_sessions,**sessions}
    except (OSError,ValueError,linux_guard.GuardError,KeyboardInterrupt) as e:
        if not (output/'warmup.json').exists():
            (output/'warmup.json').write_text(json.dumps(dict(passed=False,error=str(e)),indent=2)+'\n')
        if not started_suite:
            sessions={label:dict(state='not run (preflight/lock failure)',attempts=[]) for label in plan['sessions']}
            persist({},sessions)
            with (output/'summary.txt').open('a') as f: f.write(f'No game started: {e}\n')
            print(output,flush=True)
        with (output/'summary.txt').open('a') as f: f.write(f'\nSUITE STOPPED: {e}\n')
        raise

    print(output)
    print(summary(results,sessions))
    return int(any(r['invalid_reasons'] for r in results.values() if not r.get('warmup_only')))


if __name__=='__main__':
    def interrupted(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGHUP,interrupted)
    try: sys.exit(main())
    except (OSError,ValueError,linux_guard.GuardError,KeyboardInterrupt) as e:
        print(f'Steady relay stopped: {e}',file=sys.stderr);sys.exit(75)
