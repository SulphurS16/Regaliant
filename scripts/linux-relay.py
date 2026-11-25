#!/usr/bin/env python3
"""Bounded GPU relay, /proc CPU accounting and comparable cache-reuse sessions."""
import csv
import hashlib
import json
import os
import re
import shutil
import linux_snapshot
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import linux_guard
import linux_perf
import linux_waits
import linux_nan_summary
import importlib.util
_spec = importlib.util.spec_from_file_location('present_times', Path(__file__).with_name('linux-present-times.py'))
present_times = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(present_times)


def task_stat(text):
    # comm can contain spaces and parentheses; all following fields are numeric/state.
    tail = text[text.rindex(')') + 2:].split()
    return tail[0], int(tail[11]), int(tail[12]), int(tail[19])


def category(name):
    if name.startswith('linux-fault'): return 'fault-service'
    if name.startswith(('gpu-', 'vk-', 'pipe-', 'trcache', 'exact-specs')): return 'renderer'
    if name == 'vblank' or name.startswith(('present-', 'shot-', 'linux-')): return 'host/diagnostics'
    if name.startswith(('GAME_', 'BPE ', 'Core.', 'Nexus', 'apr', 'Bink', 'snd_', 'nca')): return 'guest'
    return 'other/driver'


class ThreadSampler:
    fields = ['elapsed_s', 'interval_s', 'phase', 'tid', 'name', 'category', 'user_s', 'system_s',
              'cpu_pct_one_core', 'sched_cpu_s', 'runqueue_s', 'voluntary_switches',
              'involuntary_switches', 'state', 'wchan', 'cpus_allowed', 'nice', 'last_cpu', 'policy']

    def __init__(self, output):
        self.writer = csv.DictWriter(output, fieldnames=self.fields)
        self.writer.writeheader()
        self.previous = {}
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.periods = defaultdict(dict)

    def sample(self, proc, elapsed, phase):
        for task in sorted((proc/'task').iterdir()):
            try:
                name = (task/'comm').read_text().strip()
                stat_text = (task/'stat').read_text()
                state, user, kernel, start = task_stat(stat_text)
                fields = stat_text[stat_text.rindex(')')+2:].split()
                sched = list(map(int, (task/'schedstat').read_text().split()))
                status = dict(line.split(':', 1) for line in (task/'status').read_text().splitlines() if ':' in line)
                vol = int(status.get('voluntary_ctxt_switches', 0))
                inv = int(status.get('nonvoluntary_ctxt_switches', 0))
                wchan = (task/'wchan').read_text().strip()
            except (OSError, ValueError, IndexError):
                continue
            key = (task.name, start)
            now = (elapsed, user, kernel, sched[0], sched[1], vol, inv, phase)
            old = self.previous.get(key)
            self.previous[key] = now
            # Exclude intervals spanning a diagnostic/measurement boundary.
            if old is None or old[7] != phase: continue
            dt = elapsed - old[0]
            if dt <= 0: continue
            du, dk = (user - old[1])/self.ticks, (kernel - old[2])/self.ticks
            dc, dq = (sched[0] - old[3])/1e9, (sched[1] - old[4])/1e9
            self.writer.writerow(dict(zip(self.fields, [f'{elapsed:.3f}', f'{dt:.3f}', phase, task.name, name,
                category(name), f'{du:.6f}', f'{dk:.6f}', f'{100*(du+dk)/dt:.3f}', f'{dc:.6f}',
                f'{dq:.6f}', vol-old[5], inv-old[6], state, wchan, status.get('Cpus_allowed_list', '').strip(),
                fields[16] if len(fields)>16 else '', fields[36] if len(fields)>36 else '', fields[38] if len(fields)>38 else ''])))
            totals = self.totals[(phase, category(name), task.name, name)]
            for i, val in enumerate([du, dk, dc, dq]): totals[i] += val
            self.periods[phase][elapsed] = dt

    def report(self, path):
        lines = ['Measured thread CPU (100% = one core); task state/wchan sampled every 5 seconds.',
                 'Direct signal work is charged to the faulting guest/renderer thread, not fault-service.',
                 'runqueue_s is schedstat runnable delay; zero may mean kernel schedstats is disabled.',
                 'Blocked/sleeping time is not CPU time. Short-lived threads between samples are absent.',
                 'Startup and phase-boundary intervals excluded; phase changes have <=5s sampling latency.']
        groups = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for (phase, cat, tid, name), values in self.totals.items():
            for i, val in enumerate(values): groups[phase, cat][i] += val
        for (phase, cat), (user, kernel, cpu, queue) in sorted(groups.items()):
            wall = sum(self.periods[phase].values())
            lines.append(f'{phase:22} {cat:18} user {user:.2f}s sys {kernel:.2f}s CPU {100*(user+kernel)/max(wall, .001):.1f}% across {wall:.1f}s; sched CPU {cpu:.2f}s runnable wait {queue:.2f}s')
        lines.append('\nPer-thread CPU, heaviest first (phase/category/tid/name/user/system/sched CPU/runqueue seconds):')
        for key, values in sorted(self.totals.items(), key=lambda kv: sum(kv[1][:2]), reverse=True):
            lines.append(f'{key}: ' + ' '.join(f'{x:.3f}' for x in values))
        path.write_text('\n'.join(lines)+'\n')


def snapshot_cgroup(proc, run, elapsed, phase):
    # Capture ancestor limits too: the leaf may be unlimited inside a capped parent.
    try:
        cg = next(line[3:] for line in (proc/'cgroup').read_text().splitlines() if line.startswith('0::'))
        root = Path('/sys/fs/cgroup').resolve()
        path = (root/cg.lstrip('/')).resolve()
        if not path.is_relative_to(root): return
        values = {}
        while path.is_relative_to(root):
            for key in ['cpu.max', 'cpu.stat', 'cpu.pressure', 'cpuset.cpus.effective']:
                try: values[str(path/key)] = (path/key).read_text()
                except OSError: pass
            if path == root: break
            path = path.parent
        with (run/'cgroup-samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(elapsed_s=elapsed, phase=phase, files=values))+'\n')
    except (OSError, StopIteration): pass


def cache_inventory(cache):
    files = [p for p in cache.rglob('*') if p.is_file()] if cache.exists() else []
    return f'{len(files)} files, {sum(p.stat().st_size for p in files)} bytes\n'


def save_inventory(user):
    return {str(p.relative_to(user)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted((user/'saves').rglob('*')) if p.is_file()}


def log_metrics(text):
    result = {}
    heartbeats = []
    begin, end = None, None
    for line in text.splitlines():
        t = re.match(r'\[\s*([\d.]+)\]', line)
        if not t: continue
        seconds = float(t[1])/1000
        for tag in ['in the game: the tutorial', 'saved game resumed in gameplay', 'gameplay-measure-begin', 'gameplay-measure-end', 'selfplay-result']:
            if 'relay step: '+tag in line: result[tag] = seconds
        if 'relay step: gameplay-measure-begin' in line: begin = seconds
        if 'relay step: gameplay-measure-end' in line: end = seconds
        if 'Linux heartbeat:' in line:
            fields = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', line)}
            heartbeats.append((seconds, fields))
    samples = [(t, f) for t, f in heartbeats if begin is not None and end is not None and begin < t < end]
    if len(samples) >= 2:
        (t0, first), (t1, last) = samples[0], samples[-1]
        result['measurement_seconds'] = t1-t0
        for key in first:
            result[key+'_per_second'] = (last[key]-first[key])/(t1-t0)
            if last.get('flips', 0) > first.get('flips', 0):
                result[key+'_per_flip'] = (last[key]-first[key])/(last['flips']-first['flips'])
    return result


def profile_kind(profile):
    if profile.startswith('steady-'): return 'steady'
    if profile == 'present-validation': return 'present-validation'
    if profile == 'present-fifo-gpu-timing': return 'gpu-timing'
    if profile in ('sparse-original', 'round9-route'): return 'clean'
    if profile.startswith(('clean', 'warm', 'mprotect')): return 'clean'
    if profile.startswith('present'): return 'present'
    return profile


def cpu_environment(run, elapsed):
    # Read-only samples: thermal/clock differences can explain an apparent cache regression.
    row = dict(elapsed_s=elapsed, frequencies_khz={}, temperatures_millidegrees={})
    for p in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'):
        try: row['frequencies_khz'][p.parts[-3]] = int(p.read_text())
        except (OSError, ValueError): pass
    for p in Path('/sys/class/hwmon').glob('hwmon*/temp*_input'):
        try: row['temperatures_millidegrees'][str(p)] = int(p.read_text())
        except (OSError, ValueError): pass
    try: row['loadavg'] = Path('/proc/loadavg').read_text().strip()
    except OSError: pass
    with (run/'cpu-environment.jsonl').open('a') as f: linux_guard.record(f, **row)


def session_environment(inherited, profile):
    # Pin instrumentation per child, even if a caller bypasses the shell wrapper.
    label, profile = profile, profile_kind(profile)
    env = inherited.copy()
    env.update({key: '0' for key in (
        'DS_NAN_PROBE', 'DS_NAN_EARLY', 'DS_PROFILE_GPU', 'DS_PROFILE_GPU_STAMPS',
        'DS_BATCH_STAMPS', 'DS_CAPTURE_CODE', 'DS_CAPTURE_TARGETS_FRAME', 'DS_RELAY_TARGETS',
        'DS_CULLREAD_DIAG', 'DS_RECORD_PROFILE', 'DS_STEP_LAPS', 'DS_OP_CENSUS', 'DS_WW_CENSUS', 'DS_RELAY_CPU_PROFILE', 'DS_LINUX_FAULT_TIMING', 'DS_LINUX_INTERVAL_SHOTS', 'DS_SCREENSHOT')})
    env.update(DS_RELAY_GAMEPLAY_SECONDS='0', DS_VK_VALIDATION='1', DS_LINUX_VOLATILE_BACKOFF='0', DS_LINUX_WRITE_WATCH='mprotect', DS_LINUX_VBLANK_GRID='0', DS_FRAME_STALLS='1', DS_LINUX_COALESCE_SPARSE='0', DS_RELAY_CLOSE_WINDOW='0', DS_WINDOW_SHOT='0')
    if profile in ('clean', 'warm', 'legacy'):
        env.update(DS_RELAY_GAMEPLAY_SECONDS='60', DS_VK_VALIDATION='0')
    elif profile == 'gpu-timing':
        env.update(DS_RELAY_GAMEPLAY_SECONDS='20', DS_VK_VALIDATION='0', DS_PROFILE_GPU='1', DS_PROFILE_GPU_STAMPS='seg', DS_BATCH_STAMPS='1')
    elif profile == 'diagnostic':
        env.update(DS_NAN_EARLY='1', DS_CAPTURE_CODE='1', DS_RECORD_PROFILE='1')
    elif profile == 'host-profile':
        env.update(DS_RELAY_GAMEPLAY_SECONDS='30', DS_VK_VALIDATION='0', DS_STEP_LAPS='1', DS_OP_CENSUS='1', DS_WW_CENSUS='1', DS_RELAY_CPU_PROFILE='1')
    elif profile == 'steady':
        env.update(DS_VK_VALIDATION='0', DS_RELAY_STEADY='1', DS_RELAY_IDLE_COMPARE='0',
                   DS_RELAY_CLOSE_WINDOW='1', DS_LINUX_PRESENT_MODE='fifo', DS_RELAY_UI_PROBE='0',
                   DS_SCENE_PROBE='0', DS_TRACE_CLEARS='0', DS_PB_VERIFY_CAPTURE='0', DS_CAPTURE='',
                   DS_CP_CLASSES='0', DS_WB_BUCKETS='0', DS_WB_BUCKETS_SPLIT='0', DS_WB_SAME='0', DS_FE2_HOLD='0', DS_RECORD_PROFILE='0')
        env['DS_LINUX_WRITE_WATCH'] = 'uffd' if label.endswith('-uffd') else 'mprotect'
        env.pop('DS_SYNC_INPUT_BATCH', None)  # Removed after a device-lost relay.
        env['DS_LINUX_ADDRESS_WAKE'] = '1'  # Verified promise wake-return semantics; normal play default too.
        env.pop('DS_RECORD_FAST', None)  # Removed after a no-gain A/B.
        env['DS_PAR_WHOLE_RUN'] = '1' if '-whole-run-' in label else '0'
        env['DS_PIPELINE_CRUMBS'] = '0'  # Filesystem diagnostics belong in startup relay only.
        if 'sample' in label: env['DS_LINUX_FAULT_TIMING'] = '1'
        if 'diagnostic' in label:
            env.update(DS_PROFILE_GPU='1', DS_PROFILE_GPU_STAMPS='seg', DS_BATCH_STAMPS='1', DS_RELAY_STEADY_DIAGNOSTIC='1',
                       DS_CP_CLASSES='1', DS_WB_BUCKETS='1', DS_WB_BUCKETS_SPLIT='1', DS_STEP_LAPS='1', DS_OP_CENSUS='1', DS_WW_CENSUS='1', DS_RECORD_PROFILE='1')
        else: env['DS_RELAY_STEADY_DIAGNOSTIC'] = '0'
    elif profile == 'startup':
        env.update(DS_VK_VALIDATION='0', DS_PIPELINE_CRUMBS='1', DS_FRAME_CAP='30', DS_UNCAPPED='0', DS_RELAY_IDLE_COMPARE='0')
    elif profile == 'cache-reuse':
        env.update(DS_VK_VALIDATION='0', DS_RELAY_GAMEPLAY_SECONDS='30')
    elif profile not in ('present', 'present-validation', 'resume'):
        raise ValueError(f'unknown relay profile: {profile}')
    if profile == 'legacy': env.update(DS_LINUX_VOLATILE_BACKOFF='1', DS_LINUX_WRITE_WATCH='mprotect')
    if label.startswith(('clean', 'warm')) or label.endswith('-uffd'): env['DS_LINUX_WRITE_WATCH'] = 'uffd'
    if label == 'sparse-original': env['DS_LINUX_COALESCE_SPARSE'] = '0'
    if profile in ('present', 'present-validation'):
        env.update(DS_RELAY_GAMEPLAY_SECONDS='10' if profile == 'present-validation' else '60',
                   DS_RELAY_CLOSE_WINDOW='1', DS_LINUX_PRESENT_MODE='fifo' if 'fifo' in label else 'mailbox',
                   DS_VK_VALIDATION='1' if profile == 'present-validation' else '0')
    if label.startswith('present-'):
        env.update(DS_RELAY_CLOSE_WINDOW='1', DS_LINUX_PRESENT_MODE='fifo' if 'fifo' in label else 'mailbox')
    if env['DS_VK_VALIDATION'] == '0':
        env.pop('VK_INSTANCE_LAYERS', None)
        env.pop('VK_LOADER_LAYERS_ENABLE', None)
        env['VK_LOADER_LAYERS_DISABLE'] = '*validation*'
    return env


def effective_validation(text):
    states = re.findall(r'Vulkan device:.*?validation (true|false)', text)
    return states[-1] == 'true' if states else None


def session_failed(metrics, profile):
    if profile in ('mprotect-2', 'present-fifo-warm', 'present-mailbox-warm') and not reuse_verified(metrics.get('pipeline_reuse', {})):
        return True
    visible = profile.startswith('present-')
    profile = profile_kind(profile)
    if visible and not metrics.get('window_closed', False): return True
    if not metrics.get('thermal', {}).get('measurement_valid', True): return True
    if metrics['exit_code'] or metrics['validation_errors'] or metrics['crash_bytes']: return True
    if metrics.get('write_watch_requested') in ('uffd', 'mprotect') and metrics.get('write_watch') != metrics['write_watch_requested']: return True
    if metrics.get('uffd_errors_per_second', 0) != 0 or metrics.get('uffd_errors_total', 0): return True
    if metrics.get('resume_requested') and not any(('saved game resumed, movement, attacks and menu' in t or 'saved game resumed, movement and attacks (short relay)' in t) for t in metrics.get('selfplay', [])): return True
    if profile in ('clean', 'warm', 'legacy', 'gpu-timing'):
        if (metrics.get('effective_validation') is not False or not any(t['frames'] for t in metrics['timelines'])
                or metrics['nan_probe']['checked_targets'] != 0): return True
        if profile == 'gpu-timing':
            # Rates are deliberately omitted for instrumented sessions. Require GPU
            # timestamp evidence, not a field that run_session just removed.
            return not any(t.get('gpu_timestamp_spans', 0) for t in metrics['timelines'])
        return 'flips_per_second' not in metrics
    if profile in ('present', 'present-validation'):
        if profile == 'present' and metrics.get('effective_validation') is not False: return True
        p = metrics.get('presentation', {})
        return (not metrics.get('window_closed', False) or p.get('wait_complete', {}).get('samples', 0) < 30
                or p.get('movement-2', {}).get('wait_complete', {}).get('samples', 0) < 30)
    if profile == 'host-profile':
        return metrics.get('effective_validation') is not False or not any(t['frames'] for t in metrics['timelines'])
    if profile == 'resume':
        return not any(('saved game resumed, movement, attacks and menu' in t or 'saved game resumed, movement and attacks (short relay)' in t) for t in metrics['selfplay'])
    if profile == 'diagnostic':
        return not metrics['nan_probe'].get('early_gameplay_targets', 0)
    return False


def session_input(profile, saved_user):
    # Startup reproduction leaves the title at its first menu; no gameplay or capture route.
    return [] if profile == 'startup' else ['--relay-continue' if saved_user is not None else '--relay-newgame']


def run_session(repo, game, run, mode, seconds, hard, cache, profile="clean", saved_user=None, driver_cache=None):
    run.mkdir()
    env = session_environment(os.environ, profile)
    env['DS_LINUX_SHADER_CACHE'] = str(cache)
    env.update(DS_LINUX_HIDDEN='1' if mode == 'offscreen' else '0', DS_LINUX_WINDOW_SIZE=env.get('DS_RELAY_WINDOW_SIZE', '800x450'))
    if env.get('DS_RELAY_RENDER_SIZE'): env['DS_RENDER_SIZE'] = env['DS_RELAY_RENDER_SIZE']
    if env.get('DS_RELAY_UI_PROBE') == '1':
        env.update(DS_CAPTURE_TARGETS_FRAME='1', DS_CAPTURE_CODE='1', DS_VK_VALIDATION='0', DS_RELAY_CLOSE_WINDOW='0')
    gentle = env.get('DS_RELAY_FULL', '0') != '1'
    env['DS_RELAY_FPS_LIMIT'] = '0' if profile_kind(profile) in ('clean', 'present') else '30'
    if run.name == 'seed-newgame': env['DS_RELAY_FPS_LIMIT'] = '30'
    if gentle:
        env.update(DS_RELAY_GAMEPLAY_SECONDS='30', DS_RELAY_SHORT='1')
        if mode == 'offscreen': env['DS_RELAY_IDLE_COMPARE'] = '1'
    if profile_kind(profile) == 'steady': env.update(DS_RELAY_IDLE_COMPARE='0', DS_RELAY_FPS_LIMIT='0', DS_RELAY_SHORT='1')
    if profile == 'cache-reuse': env.update(DS_RELAY_IDLE_COMPARE='0', DS_RELAY_FPS_LIMIT='30')
    if env.get('DS_RELAY_UI_PROBE') == '1':
        env.update(DS_RELAY_GAMEPLAY_SECONDS='0', DS_RELAY_IDLE_COMPARE='0', DS_RELAY_FPS_LIMIT='30')
    linux_snapshot.check_writable(cache/"pipelines"/"binaries")
    driver_cache = driver_cache or run.parent/('driver-'+profile.replace('warm', 'clean'))
    driver_cache.mkdir(exist_ok=True)
    for name in ('mesa', 'nvidia', 'cuda'): (driver_cache/name).mkdir(exist_ok=True)
    env.update(MESA_SHADER_CACHE_DIR=str(driver_cache/'mesa'), __GL_SHADER_DISK_CACHE_PATH=str(driver_cache/'nvidia'), CUDA_CACHE_PATH=str(driver_cache/'cuda'))
    # Every comparison copies the same immutable relay save/settings, never player data.
    env['DSPORT_HOME'] = str(run/'user')
    (run/'user').mkdir()
    if saved_user is not None:
        linux_snapshot.copy_tree(saved_user/'saves', run/'user'/'saves')
        for name in ('pcsettings.txt', 'display.txt', 'controls.txt'):
            p = saved_user/name
            if p.is_file():
                text = p.read_text()
                if name == 'pcsettings.txt':
                    text = '\n'.join(l for l in text.splitlines() if not l.strip().startswith(('Setting_PC_SaveFolder', 'Setting_PC_ImportSaves')))+'\n'
                (run/'user'/name).write_text(text)
    linux_snapshot.check_writable(run/'user'/'saves')
    (run/'saves-before.json').write_text(json.dumps(save_inventory(run/'user'), indent=2)+'\n')
    command = ['timeout', '-k', '5s', hard+'s', str(repo/'target/release/dsport'), '--app', str(game),
               *session_input(profile, saved_user), '--run-seconds', seconds] + (['--offscreen'] if mode == 'offscreen' else [])
    (run/'command.json').write_text(json.dumps({'argv': command, 'DSPORT_HOME': env['DSPORT_HOME'],
        'shader_cache': str(cache), 'environment': {k: v for k, v in env.items() if k.startswith(('DS_', 'XDG_', 'SDL_', '__GL_', 'MESA_'))}}, indent=2)+'\n')
    (run/'cache-before.txt').write_text(cache_inventory(cache))
    (run/'driver-cache-before.txt').write_text(cache_inventory(driver_cache))
    code = 125
    log_offset, phase = 0, 'startup/menu'
    perf = None
    gfx_sample = None
    gfx_attempted = False
    perf_attempted = False
    perf_log = None
    with (run/'console.log').open('wb') as console, (run/'thread-times.csv').open('w') as cpu, (run/'gpu-clocks.csv').open('w') as clocks:
        sampler = ThreadSampler(cpu)
        clocks.write('elapsed_s,index,pstate,graphics_MHz,memory_MHz,gpu_util_pct,memory_util_pct,power_W,phase\n')
        started = time.monotonic()
        process = linux_guard.Process(command, output=run, stdout=console, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try:
            while True:
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                elapsed = time.monotonic()-started
                logs = sorted((run/'user').rglob('dsport.log'))
                if logs:
                    with logs[-1].open(errors='replace') as log:
                        log.seek(log_offset)
                        new = log.read()
                        log_offset = log.tell()
                    for line in new.splitlines():
                        if 'relay step: steady-' in line:
                            match = re.search(r'relay step: steady-(stationary|walking)-(begin|end);', line)
                            if match: phase = 'steady-'+match[1] if match[2] == 'begin' else 'settle'
                        elif 'relay step: gameplay-target-capture-begin' in line: phase = 'capture'
                        elif 'relay step: gameplay-target-capture-end' in line: phase = 'settle'
                        elif 'relay step: gameplay-measure-begin' in line: phase = 'gameplay-measure'
                        elif 'relay step: gameplay-measure-end' in line: phase = 'movement/menu'
                        elif 'relay step: gameplay-timeline-begin' in line: phase = 'timeline'
                        elif 'relay step: gameplay-timeline-end' in line: phase = 'movement/menu'
                        elif 'relay step: movement-measure-' in line: phase = 'movement' if '-begin;' in line else 'movement/menu'
                try:
                    pid = int((run/'user'/'active-pid').read_text().strip())
                    if os.getpgid(pid) == process.pid:
                        proc = Path('/proc')/str(pid)
                        if 'sample' in profile and phase == 'steady-stationary' and not gfx_attempted:
                            gfx_attempted = True
                            try: gfx_sample = linux_perf.GfxSample(pid, run)
                            except (OSError, ValueError) as e:
                                (run/'gpu-gfx-perf.json').write_text(json.dumps(dict(valid=False,error=str(e)))+'\n')
                        if gfx_sample is not None and (gfx_sample.process.poll() is not None or phase != 'steady-stationary'):
                            gfx_sample.finish(interrupted=phase != 'steady-stationary')
                            gfx_sample = None
                        if not perf_attempted and ((profile == 'host-profile' and phase == 'gameplay-measure') or (profile.startswith('steady-diagnostic') and phase == 'steady-stationary')):
                            perf_attempted = True
                            if shutil.which('perf'):
                                perf_log = (run/'perf-console.log').open('wb')
                                perf_seconds = 5 if profile.startswith('steady-diagnostic') else 20
                                argv = ['timeout', '-k', '5s', str(perf_seconds+5)+'s', 'perf', 'record', '-e', 'cpu-clock:u', '-F', '99',
                                        '--call-graph', 'dwarf,8192', '-p', str(pid), '-o', str(run/'perf.data'), '--', 'sleep', str(perf_seconds)]
                                (run/'perf-command.json').write_text(json.dumps(argv)+'\n')
                                perf = subprocess.Popen(argv, stdout=perf_log, stderr=subprocess.STDOUT, start_new_session=True)
                            else: (run/'perf-console.log').write_text('perf unavailable; no host stack samples collected\n')
                        sampler.sample(proc, elapsed, phase)
                        for name in ['status', 'maps', 'cgroup']:
                            (run/f'proc-{name}.txt').write_text((proc/name).read_text())
                        snapshot_cgroup(proc, run, elapsed, phase)
                        cpu.flush()
                        os.fsync(cpu.fileno())
                except (OSError, ValueError):
                    pass
                cpu_environment(run, elapsed)
                if shutil.which('nvidia-smi'):
                    try:
                        query = subprocess.run(['nvidia-smi', '--query-gpu=index,pstate,clocks.current.graphics,clocks.current.memory,utilization.gpu,utilization.memory,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                        for line in query.stdout.splitlines(): clocks.write(f'{elapsed:.3f},'+line+','+phase+'\n')
                        if query.returncode: (run/'gpu-clocks-error.txt').write_text(query.stderr)
                        clocks.flush()
                        os.fsync(clocks.fileno())
                    except subprocess.TimeoutExpired:
                        (run/'gpu-clocks-error.txt').write_text('nvidia-smi timed out after 2s\n')
        except KeyboardInterrupt:
            code = 130
        finally:
            # Always reap, including timeout/interruption and any children left after exit_group.
            process.stop()
            process.wait()
            sampler.report(run/'thread-summary.txt')
            if gfx_sample is not None: gfx_sample.finish(interrupted=True)
            if perf is not None:
                try: os.killpg(perf.pid, signal.SIGINT)
                except ProcessLookupError: pass
                try: perf.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(perf.pid, signal.SIGKILL); perf.wait()
                if perf_log: perf_log.close()
                (run/'perf-exit-code.txt').write_text(str(perf.returncode)+'\n')
                with (run/'perf-report.txt').open('w') as output:
                    try: subprocess.run(['perf', 'report', '--stdio', '--no-children', '--percent-limit', '1', '-i', str(run/'perf.data')], stdout=output, stderr=subprocess.STDOUT, timeout=15)
                    except (OSError, subprocess.TimeoutExpired) as e: output.write(str(e))
    (run/'exit-code.txt').write_text(str(code)+'\n')
    (run/'cache-after.txt').write_text(cache_inventory(cache))
    (run/'driver-cache-after.txt').write_text(cache_inventory(driver_cache))
    (run/'saves-after.json').write_text(json.dumps(save_inventory(run/'user'), indent=2)+'\n')
    logs = sorted((run/'user').rglob('dsport.log'))
    text = '\n'.join(p.read_text(errors='replace') for p in logs)
    markers = ['Vulkan', 'vulkan:', 'SDL', 'sceVideoOutOpen', 'flip #', 'screenshot', 'unresolved', 'FATAL',
               'failed', 'Linux heartbeat', 'fault', 'fe2:', 'resident:', 'entering eboot', 'relay step:',
               'self-play result', 'prepare-play:', 'text:', 'shutdown', 'present', 'pipeline', 'cull read:', 'view mask', 'timeline', 'relay gameplay frame times', 'NaN probe']
    (run/'milestones.txt').write_text('\n'.join(l for l in text.splitlines() if any(m in l for m in markers))+'\n')
    validation = [l for l in text.splitlines() if 'vulkan:' in l or 'validation' in l.lower() or 'VUID-' in l]
    (run/'cache-events.txt').write_text('\n'.join(l for l in text.splitlines() if any(k in l for k in ('pipeline cache:', 'translation store', 'precompile', 'pipeline store', 'pipeline binaries:')))+'\n')
    (run/'validation.txt').write_text('\n'.join(validation)+'\n')
    (run/'frames.txt').write_text('\n'.join(str(p.relative_to(run)) for p in sorted((run/'user').rglob('*.png')))+'\n')
    metrics = log_metrics(text)
    metrics['thermal'] = json.loads((run/'thermal.json').read_text())
    metrics['idle_comparison'] = [json.loads(p.read_text()) for p in (run/'user').rglob('job-idle-comparison.json')]
    if not metrics['thermal']['measurement_valid']:
        (run/'TIMING_INVALID.txt').write_text('Thermal pause or guard failure: all frame-time, fps and CPU-per-second samples in this session are INVALID. Raw diagnostics retained; excluded from comparison.json.\n')
    metrics['idle_comparison_valid'] = metrics['thermal']['measurement_valid'] and code == 0
    if metrics['idle_comparison']:
        (run/'job-idle-summary.json').write_text(json.dumps(dict(valid=metrics['idle_comparison_valid'], samples=metrics['idle_comparison'], note='Stationary legacy then bounded-sleep; thermal pauses invalidate all samples. Warm-up/order can affect results.'), indent=2)+'\n')
    metrics['profile'] = profile
    metrics['resume_requested'] = saved_user is not None and profile != 'startup'
    metrics['write_watch'] = 'uffd' if 'Linux write watch: uffd async WP' in text else 'mprotect'
    metrics['write_watch_requested'] = env['DS_LINUX_WRITE_WATCH']
    metrics['uffd_errors_total'] = max([int(n) for n in re.findall(r'uffd_errors=(\d+)', text)] or [0])
    if re.search(r' W: Linux async write watch: 0x', text): metrics['uffd_errors_total'] = max(1, metrics['uffd_errors_total'])
    metrics['window_closed'] = 'SDL window-close received; flushing completed saves' in text
    metrics['presentation'] = present_times.analyze(run)
    metrics['presentation_device'] = [l for l in text.splitlines() if any(k in l for k in ('SDL display:', 'present modes:', 'swapchain:', 'pacing:'))]
    metrics['stalls'] = summarize_stalls(run)
    metrics['pipeline_reuse'] = pipeline_reuse(run)
    metrics['frame_times'] = [l for l in text.splitlines() if 'relay gameplay frame times:' in l]
    metrics['effective_validation'] = effective_validation(text)
    metrics['measurement_valid'] = profile_kind(profile) in ('clean', 'legacy', 'present') and 'flips_per_second' in metrics and metrics['effective_validation'] is False and metrics['thermal']['measurement_valid'] and code == 0
    if not metrics['thermal']['measurement_valid'] or profile_kind(profile) not in ('clean', 'legacy', 'present'):
        for key in list(metrics):
            if key.endswith(('_per_second', '_per_flip')): del metrics[key]
    if env.get('DS_RELAY_UI_PROBE') == '1':
        metrics['measurement_valid'] = False
        metrics['ui_probe'] = True
        metrics['ui_probe_complete'] = all(f'UI clear probe completed frame {i}' in text for i in range(2))
        for key in list(metrics):
            if key.endswith(('_per_second', '_per_flip')): del metrics[key]
    metrics['nan_probe'] = linux_nan_summary.summarize(run)
    metrics['timelines'] = []
    for trace in sorted((run/'user').rglob('timeline-*.json')):
        output = run/('waits-'+trace.stem)
        try: metrics['timelines'].append(linux_waits.analyze(trace, output))
        except (OSError, ValueError, KeyError) as e:
            output.mkdir(exist_ok=True)
            (output/'wait-summary.txt').write_text(f'Analysis failed: {e}\n')
    metrics['exit_code'] = code
    metrics['selfplay'] = [p.read_text(errors='replace') for p in (run/'user').rglob('selfplay.txt')]
    metrics['capture_manifests'] = [str(p.relative_to(run)) for p in (run/'user').glob('linux-runs/*/targets-*/stats.txt')]
    metrics['validation_errors'] = sum(' E: vulkan:' in l or 'VUID-' in l for l in validation)
    metrics['crash_bytes'] = sum(p.stat().st_size for p in (run/'user').rglob('crash.log'))
    (run/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    (run/'summary.txt').write_text(f'Runtime exit {code}; thermal timing valid: {metrics["thermal"]["measurement_valid"]}; read metrics.json, nan-summary.txt, waits-*/wait-summary.txt and thread-summary.txt. Targets: {metrics["capture_manifests"]}.\nValidation errors: {metrics["validation_errors"]}; crash bytes: {metrics["crash_bytes"]}.\n')
    return metrics


def isolated_scatter(repo, run):
    output = run/'scatter-check'
    output.mkdir()
    env = session_environment(os.environ, 'diagnostic')
    env.update(DS_NAN_EARLY='0', DS_RECORD_PROFILE='0', DS_LINUX_HIDDEN='1', SDL_AUDIO_DRIVER='dummy')
    command = ['timeout', '-k', '5s', '600s', str(repo/'target/release/dsport'),
               '--scatter-check', str(run/'scatter-fixture'), str(output)]
    (output/'command.json').write_text(json.dumps({'argv': command}, indent=2)+'\n')
    with (output/'console.log').open('wb') as log:
        p = linux_guard.Process(command, output=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try: code = p.wait()
        finally:
            p.stop()
            p.wait()
    text = (output/'dsport.log').read_text(errors='replace') if (output/'dsport.log').exists() else ''
    errors = [l for l in text.splitlines() if ' E: vulkan:' in l or 'VUID-' in l]
    (output/'validation.txt').write_text('\n'.join(errors)+'\n')
    (output/'result.json').write_text(json.dumps(dict(exit_code=code, validation_errors=len(errors)))+'\n')
    return code == 0 and not errors


def real_audio_check(repo, run):
    # No GPU or game launch. Exercise a real SDL device with silence for 20 seconds;
    # the window run remains on dummy audio. Never alter the desktop's default sink.
    output = run/'audio-check'
    output.mkdir()
    env = os.environ.copy()
    env.pop('SDL_AUDIO_DRIVER', None)
    env.pop('SDL_AUDIODRIVER', None)
    env['DS_AUDIO_CHECK_SECONDS'] = '20'
    command = ['timeout', '-k', '5s', '30s', str(repo/'target/release/dsport'), '--audio-check']
    (output/'command.json').write_text(json.dumps({'argv': command, 'DS_AUDIO_CHECK_SECONDS': '20', 'SDL_AUDIO_DRIVER': 'auto'}, indent=2)+'\n')
    with (output/'console.log').open('wb') as log:
        p = linux_guard.Process(command, output=output, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        try: code = p.wait()
        finally:
            p.stop()
            p.wait()
    text = (output/'console.log').read_text(errors='replace')
    drivers = re.findall(r'driver=(\S+)', text)
    driver = drivers[-1] if drivers else None
    ok = code == 0 and driver not in (None, 'dummy', 'disk') and 'SDL audio check complete:' in text
    result = dict(exit_code=code, driver=driver, stream_consumed_silence=ok, audibility_verified=False)
    (output/'summary.txt').write_text(json.dumps(result, indent=2)+'\nRead console.log for per-second queue/paused/consumption evidence. Silence is intentional.\n')
    return ok


def tracking_benchmark(repo, run):
    output = run/'tracking-bench'
    output.mkdir()
    rows = []
    for trial in range(1, 4):
        for backend in ('mprotect', 'uffd'):
            env = os.environ.copy()
            env.update(DS_LINUX_WRITE_WATCH=backend, DS_LINUX_VOLATILE_BACKOFF='0')
            cmd = ['timeout', '-k', '5s', '30s', str(repo/'target/release/dsport'), '--write-watch-check']
            with (output/f'{backend}-{trial}.log').open('w') as log:
                p = linux_guard.Process(cmd, output=output/f'{backend}-{trial}', stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                try: code = p.wait()
                finally:
                    p.stop()
                    p.wait()
            if code == 75: raise linux_guard.GuardError('Tracking check stopped by thermal guard; no further workloads started')
            text = (output/f'{backend}-{trial}.log').read_text(errors='replace')
            m = re.search(r'backend=(\w+) rounds=(\d+) writes=(\d+) wall_s=([\d.]+) cpu_s=([\d.]+) signals=(\d+)', text)
            rows.append(dict(backend=backend, trial=trial, exit_code=code, result=m.groups() if m else None))
    (output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    return all(r['exit_code'] == 0 and r['result'] and r['result'][0] == r['backend'] for r in rows)


def summarize_stalls(run):
    rows = []
    for p in (run/'user').rglob('frame-stalls.jsonl'):
        for line in p.read_text().splitlines():
            row = json.loads(line)
            if 'frame' in row: rows.append(row)
    rows.sort(key=lambda r: r['duration_ms'], reverse=True)
    report = dict(frames_over_250ms=len(rows), longest=rows[:20])
    (run/'stalls-summary.json').write_text(json.dumps(report, indent=2)+'\n')
    return dict(frames_over_250ms=len(rows), max_ms=rows[0]['duration_ms'] if rows else None)


def comparisons_for(results, mode):
    # Actual session keys, not nonexistent mprotect/clean/warm buckets in a present run.
    output = {}
    for name, m in results.items():
        if not m.get('measurement_valid'): continue
        row = {k: m[k] for k in ('flips_per_second', 'recovered_faults_per_second', 'upload_bytes_per_flip',
                'uffd_ns_per_flip', 'saved game resumed in gameplay') if k in m}
        row['pipeline_reuse'] = m.get('pipeline_reuse', {})
        row['idle_comparison'] = m.get('idle_comparison', [])
        if name.startswith('present'):
            p = m['presentation']
            row['stationary'] = p.get('pacing', {})
            for phase in ('movement-1', 'movement-2'):
                row[phase] = p.get(phase, {})
        output[name] = row
    return output


def reuse_verified(r):
    return bool(r.get('available') and r.get('binary_mode') and r.get('binary_hit_fraction', 0) >= .95)


def pipeline_reuse(run):
    reports = list((run/'user').rglob('pipeline-reuse.json'))
    if not reports: return {'available': False}
    try: r = json.loads(reports[-1].read_text())
    except (OSError, ValueError) as e: return {'available':False, 'error':str(e)}
    hits, source = r['binary_hits'], r['source_create_calls']
    result = {k: r[k] for k in ('binary_hits', 'binary_refused', 'missing', 'invalid', 'other_driver', 'other_code',
                               'captured', 'source_create_calls', 'source_create_ms', 'binary_total_ms', 'identity')}
    result['store_write_failures'] = r.get('store_write_failures', 0)
    result.update(available=True, binary_mode=r['binary_mode'], binary_hit_fraction=hits/max(hits+source, 1))
    intervals = {}
    for p in (run/'user').rglob('*-pipeline-reuse.json'):
        t = json.loads(p.read_text())
        intervals[p.name.removesuffix('-pipeline-reuse.json')] = {k: t['end'][k]-t['begin'][k]
            for k in ('binary_hits', 'source_create_calls', 'source_create_ms', 'binary_refused')}
    result['intervals'] = intervals
    result['reuse_verified'] = reuse_verified(result)
    (run/'pipeline-reuse-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def prepare_cache(repo, game, run, cache, seed):
    output = run/'preparation'
    output.mkdir()
    env = session_environment(os.environ, 'mprotect-1')
    env.update(DSPORT_HOME=str(output/'user'), DS_LINUX_SHADER_CACHE=str(cache),
               DS_PREPARE_SEED=str(seed), DS_PREPARE_LEVELS='8-1', DS_PREPARE_TOUR='2:2:0.25',
               DS_PREPARE_SHOTS='1', DS_LINUX_HIDDEN='1', SDL_AUDIO_DRIVER='dummy', DS_RELAY_FPS_LIMIT='30')
    command = ['timeout', '-k', '10s', '630s', str(repo/'target/release/dsport'), '--app', str(game),
               '--prepare-shaders', os.environ.get('DS_COMPILE_WORKERS', '2'), '--prepare-seconds', '600']
    (output/'command.json').write_text(json.dumps(dict(argv=command, environment={k:v for k,v in env.items() if k.startswith(('DS_', 'DSPORT_', 'XDG_', 'SDL_'))}), indent=2)+'\n')
    with (output/'console.log').open('wb') as log:
        p = linux_guard.Process(command, output=output, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try: code = p.wait()
        finally:
            p.stop()
            p.wait()
    state_path = output/'user/prepare/linux-full/state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    complete = code == 0 and 'tour-1' in state.get('completed', [])
    result = dict(exit_code=code, selected_level_complete=complete, whole_game_prepared=False,
                  state=state, note='Bounded shared tour of8-1 from a copied relay save. Exit12 means resumable partial progress, never full preparation.')
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    # A bounded partial tour still yields a useful cache; errors or no completed tour
    # remain visible and fail the final relay check, but do not discard measurements.
    return complete


def suite_labels(mode, full):
    if not full: return ['mprotect-1', 'present-fifo-cold']
    if mode == 'offscreen': return ['mprotect-1', 'mprotect-2', 'clean-1', 'round9-route', 'gpu-timing', 'host-profile']
    return ['present-fifo-cold', 'present-fifo-warm', 'present-mailbox-cold', 'present-mailbox-warm', 'present-fifo-uffd', 'present-fifo-gpu-timing', 'present-validation']


def main():
    repo, game, run, root = map(Path, sys.argv[1:5])
    mode, seconds, hard = sys.argv[5:]
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    # Optional immutable developer inputs. A fresh checkout makes its own isolated save.
    seed = Path(os.environ['DS_RELAY_SAVE_SEED']).resolve() if os.environ.get('DS_RELAY_SAVE_SEED') else None
    shader_seed = Path(os.environ['DS_RELAY_SHADER_SEED']).resolve() if os.environ.get('DS_RELAY_SHADER_SEED') else None
    if seed is not None and not (seed/'saves').is_dir():
        raise SystemExit(f'DS_RELAY_SAVE_SEED must contain saves/: {seed}')
    if shader_seed is not None and not shader_seed.is_dir():
        raise SystemExit(f'DS_RELAY_SHADER_SEED is not a directory: {shader_seed}')
    full = os.environ.get('DS_RELAY_FULL') == '1'
    if not full and seed is None:
        raise linux_guard.GuardError('Short suite needs DS_RELAY_SAVE_SEED containing saves/ (a prior relay user directory); use --full explicitly to create one. No game launched.')
    benchmark_ok = tracking_benchmark(repo, run) if full else True
    if not benchmark_ok:
        message = f'CPU write-watch check failed; inspect {run}/tracking-bench. No GPU run started.\n'
        (root/'summary.txt').write_text(message)
        (run/'summary.txt').write_text(message)
        return 1
    scatter_ok = None
    if full and mode == 'offscreen' and (run/'scatter-fixture').is_dir():
        scatter_ok = isolated_scatter(repo, run)
        if not scatter_ok: raise linux_guard.GuardError('Scatter failed or stopped: see scatter-check; no further sessions started')
    elif mode == 'offscreen':
        (run/'scatter-skipped.txt').write_text('No DS_SCATTER_FIXTURE supplied. Full game checks still run; isolated scatter was not checked.\n')
    results = {}
    prepared = run/'prepared-shaders'
    if shader_seed is None: prepared.mkdir()
    else: linux_snapshot.copy_tree(shader_seed, prepared)
    if seed is None:
        bootstrap = run_session(repo, game, run/'seed-newgame', 'offscreen', '900', '960', prepared, 'round9-route')
        if session_failed(bootstrap, 'round9-route') or not (run/'seed-newgame/user/saves').is_dir():
            (run/'summary.txt').write_text('Could not create the isolated tutorial save; inspect seed-newgame.\n')
            shutil.copy2(run/'summary.txt', root/'summary.txt')
            return 1
        seed = run/'seed-newgame/user'

    preparation_ok = prepare_cache(repo, game, run, prepared, seed) if full else True
    if full and json.loads((run/'preparation/thermal.json').read_text()).get('aborted'):
        raise linux_guard.GuardError('Preparation stopped by thermal guard; no further sessions started')
    shader_seed = prepared
    labels = suite_labels(mode, full)
    for label in labels:
        print(f'GPU relay {label}: {run/label}', flush=True)
        family = label.removesuffix('-cold').removesuffix('-warm')
        if label in ('mprotect-1', 'mprotect-2'): family = 'mprotect'
        if label in ('present-fifo-uffd', 'present-fifo-gpu-timing'): family = 'present-fifo'
        if not full: family = 'gentle'
        cache = run/('cache-'+family)/'shaders'
        if not cache.exists(): linux_snapshot.copy_tree(shader_seed, cache)
        # Cold/warm repeat the same route from immutable saves, sharing BOTH cache types.
        driver = run/('driver-'+family)
        results[label] = run_session(repo, game, run/label, 'present' if label.startswith('present') else 'offscreen',
                                    '900' if label == 'round9-route' else seconds,
                                    '960' if label == 'round9-route' else hard,
                                    cache, label, None if label == 'round9-route' else seed, driver)
        if results[label]['exit_code'] != 0: break
    audio_ok = real_audio_check(repo, run) if full and mode == 'present' and all(v['exit_code'] == 0 for v in results.values()) else True
    comparisons = comparisons_for(results, mode)
    (run/'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    lines = ['Only validation-free, probe-free measurements have measurement_valid=true.',
             'Saved-scene runs now require stable real gameplay HUD/menu and 120 flips before measuring.',
             'Round9-route repeats the original new-game/movement workflow; compare its boundary PNG before comparing FPS.',
             'All caches begin with a bounded shared-tour preparation; mprotect-1/2 and visible cold/warm share stores across processes.',
             'Pipeline-reuse counts actual binary hits separately from source creates (which may hit the driver cache); intervals identify source creation during movement. Warm checks require >=95% binary hits.',
             'Present cold/warm share renderer AND driver caches, with the same flip-counted route from the same save.',
             'Movement-1 and movement-2 contain no captures; second pass revisits the area, but collision/combat can change the pose.',
             'KHR wait_complete is the pacing measure; CPU enqueue time remains separate. No physical scanout timestamps are claimed.',
             'gpu-timing/present-fifo-gpu-timing/host-profile/present-validation are diagnostic only, excluded from the comparison.']
    if not full:
        lines = ['SHORT suite: no preparation or stress checks. Saved scene/cache copies; one stationary A/B and one visible FIFO run.',
                 'The whole 30s offscreen window mixes old/new job policies; compare per-policy CPU/fps in job-idle-summary.json.',
                 'Any thermal pause invalidates the whole session. No captured/timestamp/validation work inside measured movement.']
    for key, value in comparisons.items(): lines.append(key+': '+json.dumps(value))
    (run/'measurement.txt').write_text('\n'.join(lines)+'\n')
    outcomes = {k: dict(exit=v['exit_code'], failed_checks=session_failed(v, k), write_watch=v['write_watch']) for k,v in results.items()}
    summary = f"""GPU relay ({mode}, {'FULL' if full else 'SHORT'}): {run}
Thermal/CPU policy: monitor.jsonl (fsync every line), thermal.json in every workload. Any pause invalidates ALL timing in that session; invalid sessions are excluded from comparisons. Short suite skips preparation, scatter, stress and extra profiles.
Idle before/after CPU seconds per gameplay second and FPS: job-idle-comparison.json / comparison.json. This is a stationary in-session A/B, not proof across all scenes.
Session outcomes: {json.dumps(outcomes)}
Bounded preparation complete: {preparation_ok if full else 'SKIPPED (short suite)'}; CPU stress checks: {benchmark_ok if full else 'SKIPPED (short suite)'}; scatter: {scatter_ok}; real SDL silence stream: {audio_ok if full and mode == 'present' else 'SKIPPED' }.
Read comparison.json/measurement.txt: actual profile names, steady FPS, readiness time and KHR completion percentiles. Timing sessions have validation OFF (verified from device log), no NaN probes or GPU timestamps. Diagnostic sessions are separate.
Correctness: resumed/measurement boundary PNGs and each movement pass before/after; selfplay result and save hashes. Black/loading captures or runtime exit alone do not establish gameplay.
Visible cold/warm reuse renderer and driver caches; both movement passes are capture-free. Warm movement goal p50<=16.7ms, p99<=33ms, max<=100ms is reported, not silently treated as achieved. KHR wait completion is a CPU observation, not physical scanout.
Every frame >=250ms has retained operation/wait reasons in user/linux-runs/*/frame-stalls.jsonl and frame-stalls.trace.json; stalls-summary.json ranks them. Categories overlap; do not add them. pread events show file, offset, sizes, setup and kernel/retry cost. No fabricated cross-thread stack samples.
Pipeline reuse proof: pipeline-reuse-summary.json and per-interval *-pipeline-reuse.json; source creation is not necessarily compiler time. Full-suite preparation/summary.json records bounded selected-level completion (not full coverage).
Host-profile records existing CPU step/packet/write-watch counters to cpu-gameplay-profile.txt, reset after loading. It optionally attaches perf for20s only during stationary gameplay; inspect perf-console.log, perf-exit-code.txt and perf-report.txt. Missing tools or denied events are explicit. gpu-timing has a separate GPU segment timeline; instrumentation changes batching and is not a performance baseline.
Per-thread CPU/wchan/runqueue in thread-times.csv/summary, GPU clocks/util/power in gpu-clocks.csv, CPU thermal/clock/load/cgroup evidence retained. Sparse page-entry->range counts and compilation/cache statistics remain in dsport.log.
Visible sessions:800x450, null audio, scripted controls, no focus/grab/fullscreen requests; finish through SDL close. Full suite adds a separate validation session and a20s real SDL silence check without changing sinks.
All argv/env/cache paths in command.json. Every process bounded and reaped. Default player presentation is FIFO; tracking remains mprotect; uffd removed signals but did not match throughput. No commits or game-tree writes.
"""
    (run/'summary.txt').write_text(summary)
    (root/'summary.txt').write_text(summary)
    print(summary)
    return int(not preparation_ok or scatter_ok is False or not audio_ok or len(results) != len(labels) or any(v['failed_checks'] for v in outcomes.values()))


if __name__ == '__main__':
    try: sys.exit(main())
    except (linux_guard.GuardError, KeyboardInterrupt) as e:
        message = f'Relay stopped safely: {e}\n'
        for directory in sys.argv[3:5]:
            (Path(directory)/'summary.txt').write_text(message)
        print(message, file=sys.stderr)
        sys.exit(75)
