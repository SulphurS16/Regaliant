"""Bounded gpu-gfx CPU sampling, separate from clean timing. No root/sysctl changes."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def preflight(output):
    if not shutil.which('perf'):
        raise ValueError('--perf-gpu-gfx requires perf on the reviewer host; install it or omit the option')
    command = ['perf', 'record', '-e', 'cpu-clock:u', '-F', '199', '--call-graph', 'dwarf,16384',
               '-o', str(output/'perf-preflight.data'), '--', 'sleep', '0.05']
    with (output/'perf-preflight.log').open('w') as log:
        try: result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=10)
        except subprocess.TimeoutExpired as e: raise ValueError('perf preflight timed out; no game started') from e
    if result.returncode:
        raise ValueError('perf sampling denied/unavailable; see '+str(output/'perf-preflight.log'))


def gfx_tid(proc):
    matches = []
    for task in (proc/'task').iterdir():
        try:
            if (task/'comm').read_text().strip() == 'gpu-gfx': matches.append(int(task.name))
        except FileNotFoundError: pass
    if len(matches) != 1: raise ValueError(f'expected one gpu-gfx thread, found {matches}')
    return matches[0]


def command(tid, output, seconds=20):
    # timeout bounds the profiler even if the relay dies. -t excludes the idle parallel
    # recorders; DWARF unwinds a normal line-table release build without frame pointers.
    return ['timeout', '-k', '3s', f'{seconds+3}s', 'perf', 'record', '-e', 'cpu-clock:u',
            '-F', '199', '--call-graph', 'dwarf,16384', '-t', str(tid),
            '-o', str(output/'gpu-gfx-perf.data'), '--', 'sleep', str(seconds)]


def top_self(text):
    result = []
    # perf report --no-children: self percentage, then DSO and symbol separated by ';'.
    for line in text.splitlines():
        match = re.match(r'^\s*([0-9.]+)%?\s*;\s*(.+?)\s*$', line)
        if match: result.append(dict(self_percent=float(match[1]), symbol=match[2]))
    return sorted(result, key=lambda v: -v['self_percent'])[:20]


class GfxSample:
    def __init__(self, pid, output):
        self.output = output
        self.tid = gfx_tid(Path('/proc')/str(pid))
        self.begin = time.monotonic_ns()
        self.log = (output/'gpu-gfx-perf-console.log').open('wb')
        argv = command(self.tid, output)
        (output/'gpu-gfx-perf-command.json').write_text(json.dumps(dict(argv=argv,pid=pid,tid=self.tid,
            begin_monotonic_ns=self.begin,phase='stationary',event='cpu-clock:u',frequency_hz=199,
            note='Self percentages of sampled userspace CPU, not frame wall time or waits.'),indent=2)+'\n')
        try: self.process = subprocess.Popen(argv,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            self.log.close()
            raise

    def finish(self, interrupted=False):
        # Stop at the phase boundary too: never mix walking/captures into stationary samples.
        if self.process.poll() is None:
            try: os.killpg(self.process.pid,signal.SIGINT)
            except ProcessLookupError: pass
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid,signal.SIGKILL); self.process.wait()
            interrupted = True
        self.log.close()
        status = dict(exit_code=self.process.returncode,interrupted=interrupted,tid=self.tid,
                      begin_monotonic_ns=self.begin,end_monotonic_ns=time.monotonic_ns(),valid=False)
        try:
            argv = ['perf','report','--stdio','--no-children','--percent-limit','0',
                    '--sort','dso,symbol','--field-separator',';','-i',str(self.output/'gpu-gfx-perf.data')]
            result = subprocess.run(argv,capture_output=True,text=True,timeout=30)
            (self.output/'gpu-gfx-perf-self.txt').write_text(result.stdout+result.stderr)
            status['top20_self'] = top_self(result.stdout)
            status['valid'] = self.process.returncode == 0 and not interrupted and result.returncode == 0 and bool(status['top20_self'])
            with (self.output/'gpu-gfx-perf-stacks.txt').open('w') as out:
                subprocess.run(['perf','report','--stdio','--no-children','--percent-limit','0.5',
                                '-i',str(self.output/'gpu-gfx-perf.data')],stdout=out,stderr=subprocess.STDOUT,timeout=30)
        except (OSError,subprocess.TimeoutExpired) as e: status['error'] = str(e)
        (self.output/'gpu-gfx-perf.json').write_text(json.dumps(status,indent=2)+'\n')
        lines = ['gpu-gfx stationary CPU sampling: '+('VALID' if status['valid'] else 'FAILED/INCOMPLETE'),
                 'Self % is sampled userspace CPU, not frame wall time. Sampling run is excluded from FPS comparisons.']
        lines += [f'{r["self_percent"]:6.2f}%  {r["symbol"]}' for r in status.get('top20_self',[])]
        (self.output/'gpu-gfx-top20.txt').write_text('\n'.join(lines)+'\n')
        return status
