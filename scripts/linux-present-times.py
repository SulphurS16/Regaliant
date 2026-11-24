#!/usr/bin/env python3
"""Gameplay-only presentation percentiles; keep CPU completion apart from scanout time."""
import csv
import json
import math
from pathlib import Path


def percentiles(values):
    s = sorted(values)
    if not s: return {'samples': 0}
    return dict(samples=len(s), mean_ms=sum(s)/len(s), max_ms=s[-1],
                **{f'p{q}_ms': s[max(0, math.ceil(len(s)*q/100)-1)] for q in (50, 90, 95, 99)})


def pacing_target(pacing):
    if not pacing.get('samples'): return {'measured': False, 'reason': 'no KHR completion samples'}
    return dict(measured=True, p50_met=pacing['p50_ms'] <= 16.7, p99_met=pacing['p99_ms'] <= 33,
                no_stall_over_100ms=pacing['max_ms'] <= 100)


def slow_frames(rows):
    """Correlate long KHR intervals with enqueue gaps and the present call's costs.
    This distinguishes supply gaps from WSI blocking, not causal GPU attribution.
    """
    names = ('slot_ms', 'acquire_ms', 'record_submit_ms', 'gpu_ready_ms', 'capture_queue_ms', 'present_ms')
    submits = {int(r['id']): r for r in rows if r['kind'] == 'submit'}
    completions = sorted((int(r['monotonic_ns']), int(r['id'])) for r in rows if r['kind'] == 'wait_complete')
    result = []
    for (a, ai), (b, bi) in zip(completions, completions[1:]):
        gap = (b-a)/1e6
        if gap < 25: continue
        row = dict(id=bi, previous_id=ai, completion_ms=gap, consecutive=bi == ai+1)
        if bi in submits and ai in submits:
            now, before = submits[bi], submits[ai]
            costs = {k: float(now[k]) for k in names}
            enqueue = (int(now['monotonic_ns'])-int(before['monotonic_ns']))/1e6
            row.update(costs, enqueue_interval_ms=enqueue,
                       outside_current_present_ms=max(0, enqueue-sum(costs.values())))
        result.append(row)
    return result


def analyze(run):
    result = {'note': 'wait_complete (KHR present_wait/present_id) is the pacing measure: CPU observation of completion, not physical scanout. submit is CPU enqueue. Intervals include ID gaps; duplicate timestamps are excluded. Both movement passes exclude captures. Missing completion samples are reported, never replaced by submit times.'}
    overall = result
    for path in sorted((run/'user').rglob('*-presentation.csv')):
        phase = path.name.removesuffix('-presentation.csv')
        result = overall if phase == 'gameplay' else overall.setdefault(phase, {})
        with path.open() as f: rows = list(csv.DictReader(f))
        for kind in ('submit', 'wait_complete'):
            samples = sorted((int(r['monotonic_ns']), int(r['id'])) for r in rows if r['kind'] == kind)
            intervals, gaps = [], 0
            for (a, ai), (b, bi) in zip(samples, samples[1:]):
                if bi != ai+1: gaps += 1
                if b > a: intervals.append((b-a)/1e6)
            result[kind] = dict(percentiles(intervals), observations=len(samples), nonconsecutive_ids=gaps,
                                physical_display_timestamps=False)
        enqueued = {int(r['id']): int(r['monotonic_ns']) for r in rows if r['kind'] == 'submit'}
        result['enqueue_to_completion'] = percentiles([(int(r['monotonic_ns'])-enqueued[int(r['id'])])/1e6
            for r in rows if r['kind'] == 'wait_complete' and int(r['id']) in enqueued and int(r['monotonic_ns']) >= enqueued[int(r['id'])]])
        slow = slow_frames(rows)
        (run/f'{phase}-slow-frames.json').write_text(json.dumps(slow, indent=2)+'\n')
        result['intervals_over_25ms'] = len(slow)
        result['slow_frame_stages'] = {k: percentiles([r[k] for r in slow if k in r])
            for k in ('enqueue_interval_ms', 'outside_current_present_ms', 'slot_ms', 'acquire_ms', 'record_submit_ms', 'gpu_ready_ms', 'present_ms')}
        result['pacing'] = result['wait_complete']
        result['target'] = pacing_target(result['pacing'])
        result['stages'] = {name: percentiles([float(r[name]) for r in rows if r['kind'] == 'submit'])
                            for name in ('slot_ms', 'acquire_ms', 'record_submit_ms', 'gpu_ready_ms', 'capture_queue_ms', 'present_ms')}
    (run/'presentation-summary.json').write_text(json.dumps(overall, indent=2)+'\n')
    return overall


if __name__ == '__main__':
    import sys
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))
