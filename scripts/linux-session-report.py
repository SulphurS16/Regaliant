#!/usr/bin/env python3
"""Read-only summary of player logs; does not infer CPU time from overlapping hitch spans."""
import argparse
import collections
import json
from pathlib import Path
import re


def analyze(text, intervals=((40, 80), (70, 115))):
    beats, sizes, hitches, gpu_hitches, audio = [], [], [], [], []
    for line in text.splitlines():
        timestamp = re.match(r'\[\s*([\d.]+)\]', line)
        if not timestamp: continue
        time_s = float(timestamp[1])/1000
        if 'Linux heartbeat:' in line:
            fields = {k: int(v) for k,v in re.findall(r'(\w+)=(\d+)', line)}
            elapsed = int(re.search(r'heartbeat: (\d+)s', line)[1])
            beats.append((time_s, elapsed, fields))
        if m := re.search(r'render size (\d+)x(\d+)', line):
            sizes.append(dict(time_s=time_s, width=int(m[1]), height=int(m[2]), message=line.split('I: ',1)[-1]))
        if m := re.search(r'hitch: flip (\d+) took (\d+) ms', line):
            hitches.append(dict(time_s=time_s, flip=int(m[1]), wall_ms=int(m[2]),
                spans_ms={k:int(v) for k,v in re.findall(r'(pipeline|translation|upload|GPU wait|CP wait|memory) (\d+) ms',line.split('; longest:')[0])},
                detail=line.split('; longest:')[-1]))
        if m := re.search(r'hitch GPU: flip (\d+) took (\d+) ms: GPU busy ([\d.]+) ms.*median frame (\d+) ms, GPU busy ([\d.]+)', line):
            gpu_hitches.append(dict(time_s=time_s, flip=int(m[1]), wall_ms=int(m[2]), gpu_busy_ms=float(m[3]), median_frame_ms=int(m[4]), median_gpu_busy_ms=float(m[5])))
        if 'audio:' in line and 'ran dry' in line: audio.append(line)
    windows=[]
    for begin,end in intervals:
        sample=[b for b in beats if begin <= b[1] <= end]
        if len(sample)<2: continue
        a,b=sample[0],sample[-1]; seconds=b[0]-a[0]
        windows.append(dict(requested_seconds=[begin,end], actual_seconds=seconds,
            rates={k:(b[2][k]-a[2][k])/seconds for k in a[2].keys() & b[2].keys()},
            render_size_at_start=next((s for s in reversed(sizes) if s['time_s']<=a[0]), None)))
    return dict(windows=windows, render_size_changes=sizes, hitch_count=len(hitches),
        longest_hitches=sorted(hitches,key=lambda h:h['wall_ms'],reverse=True)[:12],
        gpu_hitches=gpu_hitches, audio_underruns=audio,
        limitation='Hitch CPU spans overlap across threads; GPU numbers are hitch samples, not a continuous profile. No per-thread CPU times can be recovered from this log alone.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    args=parser.parse_args()
    report=analyze(args.log.read_text(errors='replace'))
    reuse=args.log.with_name('pipeline-reuse.json')
    if reuse.exists():
        data=json.loads(reuse.read_text())
        report['pipeline_reuse']={k:v for k,v in data.items() if k not in ('store','events')}
        report['sampled_binary_failure_reasons']=dict(collections.Counter(e['reason'] for e in data.get('events',[])))
    print(json.dumps(report,indent=2))
