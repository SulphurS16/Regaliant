#!/usr/bin/env python3
"""Per-flip wall-time coverage from runtime Chrome traces; never sum concurrent lanes."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def classify(track, name):
    # Specific waits win over enclosing batch/record spans and duplicate hitch reports.
    n = name.lower()
    if track == 'GPU': return 'gpu_upload_prologue' if n.endswith(' prologue') else 'gpu_batch_span'
    if track.startswith('GPU segments /'):
        if 'write-back' in n: return 'gpu_writeback'
        if 'draws ' in n: return 'gpu_draw_segment'
        if 'cs ' in n: return 'gpu_compute_segment'
        return 'gpu_barrier_or_other'
    if name == 'replay commands': return 'driver_recording'
    if name == 'submit uploads': return 'texture_upload_submission'
    if name == 'submit pagetable': return 'pagetable_submission'
    if track == 'vk submit':
        if name == 'prepare submit jobs': return 'submit_preparation'
        if name == 'submit specs': return 'submit_specs'
        return 'submit_jobs'
    if track.startswith('Vulkan /'):
        return 'vulkan_queue_lock' if n.startswith('queue lock:') else 'vulkan_call_and_preparation'
    if 'wait: pacing' in n: return 'ready_frame_pacing'
    if 'idle: no ready frame' in n: return 'no_ready_frame'
    if 'wait: batch slot' in n:
        for state in ('submit', 'gpu', 'release'):
            if f'initial {state},' in n: return 'batch_slot_'+state
        return 'batch_slot'
    if 'wait: gpu completion' in n: return 'gpu_completion'
    if 'wait: submission' in n: return 'submission_wait'
    if n.startswith('release diagnostics'): return 'release_diagnostics'
    if n.startswith('release pool reset'): return 'release_reset'
    if 'wait: label' in n or 'cp wait: label' in n: return 'label'
    if 'wait: chunk' in n or 'wait: parallel' in n: return 'parallel'
    if 'wait: another thread' in n: return 'memory_lock'
    if 'gpu wait:' in n or 'wait: batch' in n or 'wait: own batches' in n: return 'gpu_completion'
    if n.startswith('wait:'): return 'other_wait'
    if n.startswith('idle:'): return 'no_submission'
    if n.startswith('memory:'): return 'memory_work'
    if track == 'gpu-complete': return 'completion_effects'
    if track == 'vblank': return 'videoout_work'
    return 'record_or_other'


PRIORITY = {name: i for i, name in enumerate(['untraced', 'record_or_other', 'videoout_work',
    'completion_effects', 'memory_work', 'no_submission', 'other_wait', 'gpu_completion',
    'memory_lock', 'parallel', 'label', 'batch_slot', 'batch_slot_submit', 'batch_slot_gpu', 'batch_slot_release', 'submission_wait', 'release_diagnostics', 'release_reset', 'no_ready_frame', 'ready_frame_pacing',
    'submit_jobs', 'submit_preparation', 'submit_specs', 'driver_recording', 'texture_upload_submission', 'pagetable_submission', 'vulkan_queue_lock', 'vulkan_call_and_preparation', 'gpu_batch_span', 'gpu_upload_prologue', 'gpu_barrier_or_other', 'gpu_compute_segment', 'gpu_draw_segment', 'gpu_writeback'])}


def partition(spans, start, end):
    """Exclusive coverage: overlapping nested spans are counted once at highest priority."""
    edges = [(start, 0, 'untraced'), (end, 0, 'untraced')]
    for a, b, category in spans:
        a, b = max(a, start), min(b, end)
        if a < b: edges.extend([(a, 1, category), (b, -1, category)])
    active = defaultdict(int)
    result = defaultdict(float)
    prev = start
    for t, delta, category in sorted(edges):
        if t > prev:
            selected = max((k for k, n in active.items() if n), key=PRIORITY.get, default='untraced')
            result[selected] += t-prev
        active[category] += delta
        prev = t
    return result


def analyze(path, output):
    data = json.loads(path.read_text())
    events = data['traceEvents']
    tracks = {e['tid']: e['args']['name'] for e in events if e.get('ph') == 'M' and e.get('name') == 'thread_name'}
    flips = sorted((e['ts'], e['name']) for e in events if e.get('ph') == 'i' and e['name'].startswith('flip '))
    spans = defaultdict(list)
    details = defaultdict(list)
    for e in events:
        if e.get('ph') != 'X' or e.get('dur', 0) <= 0: continue
        track = tracks.get(e['tid'], str(e['tid']))
        if track.endswith((' queued', ' subs')): continue
        if track == 'GPU segments': track += ' / ' + e['name'].split(':', 1)[0]
        a, b = e['ts'], e['ts']+e['dur']
        cat = classify(track, e['name'])
        spans[track].append((a, b, cat))
        # Keep the exact blocking operation/address/producer for subsequent diagnosis.
        if cat in ('batch_slot', 'batch_slot_submit', 'batch_slot_gpu', 'batch_slot_release', 'submission_wait', 'label', 'gpu_completion', 'memory_lock', 'other_wait'):
            details[track, e['name']].append((a, b, cat))
    gpu_timestamp_spans = sum(len(v) for k, v in spans.items() if k == 'GPU' or k.startswith('GPU segments /'))
    output.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: defaultdict(float))
    with (output/'frame-waits.csv').open('w') as f:
        writer = csv.writer(f)
        cats = list(PRIORITY)
        writer.writerow(['ending_flip', 'start_us', 'frame_ms', 'track'] + [c+'_ms' for c in cats])
        for (a, _), (b, flip) in zip(flips, flips[1:]):
            for track, ranges in sorted(spans.items()):
                coverage = partition(ranges, a, b)
                writer.writerow([flip, a, (b-a)/1000, track] + [coverage[c]/1000 for c in cats])
                for c, t in coverage.items(): totals[track][c] += t/1000
    frames = max(0, len(flips)-1)
    wall = (flips[-1][0]-flips[0][0])/1000 if frames else 0
    lines = [f'Source: {path}', f'{frames} complete flip intervals, {wall:.3f} ms; mean {wall/max(frames, 1):.3f} ms/flip.',
        'Each lane partitions frame WALL time; nested/duplicate spans are counted once. Concurrent lanes MUST NOT be summed.',
        'Batch slot suffix = state at wait entry (not a decomposition of time spent later): submit not yet handed to Vulkan, gpu completion pending, release GPU already done. Match batch IDs to gpu-release tracks.',
        'Recording/submit spans are wall time, not CPU utilization. Untraced includes sleep and uninstrumented work.',
        'vk submit includes job preparation and submit_specs; it is NOT just vkQueueSubmit. Vulkan lanes split queue-lock wait from call/preparation under lock.',
        'GPU upload prologues include copies and their trailing visibility barrier, not just DMA. Batch spans and prologues overlap: categories partition their union; inspect longest-prologues.txt for raw spans.',
        'GPU batch-span coverage includes in-command waits/barriers, is not device occupancy, and only covers timestamped batches.',
        'GPU segment lanes separate executors; mixed draw/compute segments are classified as draws. Inspect raw trace for shaders/barrier sites.',
        'The intervals are between displayed flips, not exact causal attribution of pipelined work to a game frame.',
        'No ready frame = videoout timer waited while no flip was ready; ready_frame_pacing = ready at wait start (a readiness change during wait is not timed).',
        '\nMean milliseconds per flip, by lane:']
    for track, values in sorted(totals.items()):
        lines.append(track + ': ' + ', '.join(f'{c}={t/frames:.3f}' for c, t in sorted(values.items(), key=lambda x: -x[1]) if t > 0))
    prologues = sorted([(e['dur']/1000, e['name']) for e in events if e.get('ph') == 'X' and tracks.get(e.get('tid')) == 'GPU' and e['name'].endswith(' prologue')], reverse=True)
    (output/'longest-prologues.txt').write_text('Copy prologues including trailing barriers; raw overlapping spans, do not sum.\n' + ''.join(f'{ms:.3f} ms {name}\n' for ms, name in prologues[:100]))
    if frames:
        ranked = [(sum(partition(v, flips[0][0], flips[-1][0]).get(c, 0) for c in PRIORITY if c != 'untraced')/1000, k)
                  for k, v in details.items()]
        gpu = [(min(e['ts']+e['dur'], flips[-1][0])-max(e['ts'], flips[0][0]), e['name'])
               for e in events if e.get('ph') == 'X' and tracks.get(e['tid'], '').startswith('GPU segments')
               and e['ts'] < flips[-1][0] and e['ts']+e['dur'] > flips[0][0]]
        lines.append('\nLongest GPU segments in this window (including waits/barriers):')
        for us, name in sorted(gpu, reverse=True)[:25]: lines.append(f'{us/1000:.3f} ms {name}')
        lines.append('\nLargest individual blocking spans (duplicate descriptions may overlap; do not sum):')
        for ms, (track, name) in sorted(ranked, reverse=True)[:40]: lines.append(f'{ms:.3f} ms {track}: {name}')
    else: lines.append('INSUFFICIENT EVIDENCE: fewer than two flip markers.')
    (output/'wait-summary.txt').write_text('\n'.join(lines)+'\n')
    return {'frames': frames, 'mean_frame_ms': wall/max(frames, 1), 'tracks': len(totals), 'gpu_timestamp_spans': gpu_timestamp_spans,
            'mean_ms_by_track': {track: {c: t/max(frames, 1) for c, t in values.items()} for track, values in totals.items()}}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
