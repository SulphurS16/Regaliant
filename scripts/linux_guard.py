"""CPU placement and fail-closed thermal supervision for every relay child.

No sensors or a stalled/cold-down-failing sensor means no workload, not unguarded work.
Monitor records are flushed and fsynced individually, including transitions.
"""
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


class GuardError(RuntimeError):
    pass


def cpu_list(value):
    result = set()
    for part in value.split(','):
        bounds = part.split('-')
        if len(bounds) > 2: raise ValueError('CPU set must use numbers/ranges, e.g. 0-3,8-11')
        lo, hi = int(bounds[0]), int(bounds[-1])
        if lo < 0 or hi < lo: raise ValueError('invalid CPU range')
        result.update(range(lo, hi + 1))
    if not result: raise ValueError('empty CPU set')
    return result


def select_cpus(value=None, allowed=None, root=Path('/sys/devices/system/cpu')):
    allowed = set(os.sched_getaffinity(0) if allowed is None else allowed)
    if value:
        wanted = cpu_list(value)
        if not wanted <= allowed: raise GuardError('DS_RELAY_CPUS contains CPUs outside the allowed affinity')
        return sorted(wanted)
    def read(cpu, path, default=0):
        try: return int((root/f'cpu{cpu}'/path).read_text())
        except (OSError, ValueError): return default
    def rank(cpu):
        # core_type: Intel Core=2, Atom=1. Capacity/frequency are fallbacks on other CPUs.
        kind = read(cpu, 'topology/core_type')
        capacity = read(cpu, 'cpu_capacity')
        freq = read(cpu, 'cpufreq/cpuinfo_max_freq')
        try: siblings = cpu_list((root/f'cpu{cpu}/topology/thread_siblings_list').read_text().strip()) & allowed
        except (OSError, ValueError): siblings = {cpu}
        return (-kind, -capacity, -freq, cpu != min(siblings or {cpu}), cpu)
    return sorted(sorted(allowed, key=rank)[:8])


def sensor_paths(root=Path('/sys/class/hwmon')):
    packages, fallback = [], []
    for hw in sorted(root.glob('hwmon*')):
        try: name = (hw/'name').read_text().strip()
        except OSError: continue
        for p in sorted(hw.glob('temp*_input')):
            try: label = p.with_name(p.name.replace('_input', '_label')).read_text().strip()
            except OSError: label = ''
            if label == 'Package id 0': packages.append(p)
            if name in ('coretemp', 'k10temp'): fallback.append(p)
    return packages or fallback


def temperature(paths):
    if not paths: raise GuardError('No CPU package/coretemp/k10temp sensor: relay not started')
    values = []
    for p in paths:
        try: v = float(p.read_text()) / 1000
        except (OSError, ValueError) as e: raise GuardError(f'CPU sensor unavailable: {p}: {e}') from e
        if not math.isfinite(v) or not -20 <= v <= 150: raise GuardError(f'Invalid CPU temperature: {p}: {v}')
        values.append(v)
    return max(values)


def record(f, **row):
    row.update(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), monotonic_s=time.monotonic(), load=os.getloadavg())
    f.write(json.dumps(row)+'\n')
    f.flush()
    os.fsync(f.fileno())


class Policy:
    def __init__(self, env=None):
        env = os.environ if env is None else env
        self.cpus = select_cpus(env.get('DS_RELAY_CPUS'))
        self.workers = int(env.get('DS_COMPILE_WORKERS', '2'))
        self.pause = float(env.get('DS_RELAY_TEMP_PAUSE', '95'))
        self.resume = float(env.get('DS_RELAY_TEMP_RESUME', '85'))
        self.abort = float(env.get('DS_RELAY_TEMP_ABORT_S', '60'))
        self.cooldown = float(env.get('DS_RELAY_COOLDOWN_S', '120'))
        if not (1 <= self.workers <= 16 and 0 < self.resume < self.pause <= 110
                and 0 < self.abort <= 600 and 0 < self.cooldown <= 600):
            raise GuardError('Invalid worker count or thermal thresholds/time limits')
        self.paths = sensor_paths()

    def command(self, command):
        return ['nice', '-n', '10', 'taskset', '-c', ','.join(map(str, self.cpus)), *command]

    def environment(self, env):
        env = env.copy()
        env.update(DS_COMPILE_WORKERS=str(self.workers), DS_COMPILE_AHEAD=str(self.workers),
                   DS_ASYNC_GFX_WORKERS=str(self.workers), DS_PRECOMPILE_WORKERS=str(self.workers),
                   CARGO_BUILD_JOBS=str(self.workers), RUST_TEST_THREADS=str(self.workers))
        return env

    def wait_cool(self, output):
        started = time.monotonic()
        with output.open('a') as f:
            while True:
                temp = temperature(self.paths)
                record(f, event='cooldown', temperature_c=temp, cpus=self.cpus, sensors=list(map(str, self.paths)))
                if temp < self.resume: return
                if time.monotonic()-started >= self.cooldown:
                    record(f, event='cooldown-abort', temperature_c=temp)
                    raise GuardError('CPU did not cool below resume threshold; no next session started')
                time.sleep(2)


class ThermalState:
    """Pure hysteresis state machine, tested with a fake clock and sensor values."""
    def __init__(self, pause, resume, abort):
        self.pause, self.resume, self.abort = pause, resume, abort
        self.since = None
        self.pauses = 0
        self.paused_s = 0.0

    def step(self, now, temp):
        if self.since is None:
            if temp > self.pause:
                self.since = now
                self.pauses += 1
                return 'pause'
        elif temp < self.resume:
            self.paused_s += now-self.since
            self.since = None
            return 'resume'
        elif now-self.since >= self.abort:
            # Bound the entire pause, even if temperature sits between thresholds.
            return 'abort'
        return 'sample'


def descendant_groups(pid):
    table = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = p.read_text().rsplit(')', 1)[1].split()
            table[int(p.parent.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, ValueError, IndexError): pass
    owned = {pid}
    while True:
        extra = {p for p, (parent, _) in table.items() if parent in owned} - owned
        if not extra: break
        owned.update(extra)
    return {pid} | {table[p][1] for p in owned if p in table}


def group_activity(groups):
    result = {}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            f = p.read_text().rsplit(')', 1)[1].split()
            pg = int(f[2])
            if pg not in groups: continue
            row = result.setdefault(pg, dict(states={}, cpu_ticks=0))
            row['states'][f[0]] = row['states'].get(f[0], 0) + 1
            row['cpu_ticks'] += int(f[11]) + int(f[12])
        except (OSError, ValueError, IndexError): pass
    return result


class Process:
    """Popen interface with an independent 2s monitor; monitors never share game affinity."""
    def __init__(self, command, *, output, env=None, **kwargs):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.policy = Policy(env)
        self.policy.wait_cool(self.output/'monitor.jsonl')
        self.state = ThermalState(self.policy.pause, self.policy.resume, self.policy.abort)
        self.aborted = None
        self.finished = threading.Event()
        self.groups = set()
        self.started = time.monotonic()
        self.stop_file = self.output/'stop-request'
        child_env = self.policy.environment(env or os.environ)
        child_env['DS_RELAY_STOP_FILE'] = str(self.stop_file)
        # GNU timeout remains the independent wall bound. All callers supply it.
        with (self.output/'guard-command.json').open('w') as f:
            record(f, argv=self.policy.command(command), workers=self.policy.workers, cpus=self.policy.cpus,
                   environment={k:v for k,v in child_env.items() if k.startswith(('DS_', 'DSPORT_', 'CARGO_','RUST_TEST'))})
        self.p = subprocess.Popen(self.policy.command(command), env=child_env, **kwargs)
        self.pid = self.p.pid
        self.groups.add(self.pid)
        self.thread = threading.Thread(target=self._monitor, name='relay-thermal', daemon=True)
        self.thread.start()

    def send(self, sig):
        self.groups.update(descendant_groups(self.pid))
        for pg in self.groups:
            if pg == os.getpgrp(): continue
            try: os.killpg(pg, sig)
            except ProcessLookupError: pass

    def stop(self):
        if self.p.poll() is None:
            self.stop_file.write_text('relay stop\n')
            self.send(signal.SIGCONT)
            try: self.p.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
        # TERM also invokes the preparation parent's cancellation handler.
        self.send(signal.SIGTERM)
        self.send(signal.SIGCONT)
        try: self.p.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
        self.send(signal.SIGKILL)
        self.p.wait()

    def _monitor(self):
        with (self.output/'monitor.jsonl').open('a') as f:
            try:
                while self.p.poll() is None:
                    self.groups.update(descendant_groups(self.pid))
                    temp = temperature(self.policy.paths)
                    action = self.state.step(time.monotonic(), temp)
                    if action == 'pause': self.send(signal.SIGSTOP)
                    elif action == 'resume': self.send(signal.SIGCONT)
                    record(f, event=action, temperature_c=temp, pid=self.pid, cpus=self.policy.cpus,
                           paused=self.state.since is not None, owned_groups=group_activity(self.groups))
                    if action == 'abort':
                        self.aborted = 'thermal pause exceeded bounded cooling time'
                        break
                    if self.finished.wait(2): break
            except Exception as e:
                self.aborted = str(e)
                record(f, event='guard-error', error=self.aborted)
            finally:
                if self.aborted: self.stop()
                paused = self.state.paused_s + (time.monotonic()-self.state.since if self.state.since is not None else 0)
                report = dict(pauses=self.state.pauses, paused_s=paused, aborted=self.aborted,
                              measurement_valid=self.state.pauses == 0 and not self.aborted,
                              cpus=self.policy.cpus, nice=10, compile_workers=self.policy.workers,
                              elapsed_s=time.monotonic()-self.started)
                with (self.output/'thermal.json').open('w') as result: record(result, **report)

    def wait(self, timeout=None):
        code = self.p.wait(timeout=timeout)
        self.finished.set()
        self.thread.join()
        return 75 if self.aborted else code

    def poll(self): return self.p.poll()


def run(command, output, env=None, **kwargs):
    p = Process(command, output=output, env=env, start_new_session=True, **kwargs)
    try: return p.wait()
    finally:
        p.stop()
        p.wait()


if __name__ == '__main__':
    def interrupted(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        directory, *command = sys.argv[1:]
        sys.exit(run(command, Path(directory)))
    except (GuardError, ValueError, KeyboardInterrupt) as e:
        print(f'Relay guard: {e}', file=sys.stderr)
        sys.exit(75)
