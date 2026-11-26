"""Read-only rates from play logs. Aggregate worker timings are not frame critical paths."""
import json
import re
import sys
from pathlib import Path


def analyze(text, window=30):
    heartbeats=[]; extras={}; events=[]; partials=[]; stats=[]; timestamp=0.
    for line in text.splitlines():
        m=re.match(r'\[\s*([\d.]+)\]',line)
        if m: timestamp=float(m[1])/1000
        h=re.search(r'Linux (heartbeat|play|adaptive): (\d+)s',line)
        if h:
            row={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)};row['seconds']=int(h[2])
            if h[1]=='heartbeat':heartbeats.append(row)
            else:extras.setdefault(row['seconds'],{}).update(row)
        if 'partially resident texture:' in line:partials.append(timestamp)
        if any(x in line for x in ('FSR3 FG context:', 'frame generation disabled', 'open \'/app0/movies/', 'Linux shutdown complete:')):
            events.append(dict(seconds=timestamp,line=line))
        if any(x in line for x in ('pipelines compiled ahead:', 'pipeline binaries:', 'images: ')) and (
            'pipelines created' in line or 'pipelines made from them' in line or re.search(r'images: \d+ cached',line)):
            stats.append(dict(seconds=timestamp,line=line.strip()))
    for row in heartbeats:row.update(extras.get(row['seconds'],{}))
    per_second=[]
    for a,b in zip(heartbeats,heartbeats[1:]):
        dt=b['seconds']-a['seconds']
        if dt<=0: continue
        row={'seconds':b['seconds'],'flip':b.get('flips'), 'interval_s':dt}
        for k in ('texture_zero_parts','texture_missing_parts','texture_remaps','texture_gpu_upload_bytes','pb_hits','pb_source_calls','fg_packets','fg_real_shown','fg_generated_shown','fg_repeats','fg_drops'):
            if k in a and k in b:row[k+'_per_s']=(b[k]-a[k])/dt
        per_second.append(row)
    windows=[]
    if heartbeats:
        a=heartbeats[0]
        for b in heartbeats[1:]:
            dt=b['seconds']-a['seconds']
            if dt<window:continue
            flips=b.get('flips',0)-a.get('flips',0)
            delta={k:b[k]-a[k] for k in b.keys() & a.keys() if k!='seconds'}
            if flips>0:
                row=dict(start_s=a['seconds'],end_s=b['seconds'],fps=flips/dt,flips=flips,
                         first_flip=a['flips'],last_flip=b['flips'])
                for name,key,scale in [('draws_per_flip','draws',1),('upload_MB_per_flip','upload_bytes',1e6),
                    ('writeback_MB_per_flip','writeback_bytes',1e6),('uffd_aggregate_ms_per_flip','uffd_ns',1e6),
                    ('uffd_scans_per_flip','uffd_scans',1),('fg_compute_ms_per_flip','fg_ns',1e6),
                    ('fg_pace_ms_per_flip','fg_pace_ns',1e6),('source_create_ms_per_flip','pb_source_ns',1e6)]:
                    if key in delta:row[name]=delta[key]/flips/scale
                for k in ['pb_source_calls','pb_hits','pb_refused','pb_missing','fg_generated','fg_dispatches','fg_retired','fg_pool_deferred','texture_remaps']:
                    if k in delta:row[k]=delta[k]
                if 'ww_samples' in delta and delta['ww_samples']>0:
                    row['uffd_sample_us']={k:delta[k]/delta['ww_samples']/1000 for k in ('ww_mapping_ns','ww_lock_ns','ww_work_ns')}
                    row['uffd_gauges']={k:b[k] for k in ('ww_regions','ww_journal','ww_writers')}
                row['partial_warnings']=sum(a['seconds']<=t<b['seconds'] for t in partials)
                row['faults_per_s']=delta.get('recovered_faults',0)/dt
                windows.append(row)
            a=b
    return dict(windows=windows,per_second=per_second,events=events,cache_snapshots=stats,
        note='Heartbeat seconds are runtime-relative; event timestamps include bootstrap (~1s offset). No screenshot-to-time association. Worker ns are aggregate wall durations, not CPU or critical-path time. Captures and scene changes invalidate performance comparisons.')


if __name__=='__main__':
    print(json.dumps(analyze(Path(sys.argv[1]).read_text(errors='replace')),indent=2))


def adaptive_report(run, first_flip=0, last_flip=2**64):
    """Separate fresh images from repeated slots; CPU enqueue ages, not scanout latency."""
    import csv
    import math
    rows=[]
    for p in (Path(run)/'user').glob('linux-runs/*/fg-adaptive.csv'):
        with p.open() as f:
            rows.extend(r for r in csv.DictReader(f) if (r['present_ok']=='true' or r['repeat']=='true') and first_flip<=int(r['flip'])<=last_flip)
    if len(rows)<2:return {'samples':len(rows)}
    rows.sort(key=lambda r:int(r['slot_ns']))
    seconds=(int(rows[-1]['slot_ns'])-int(rows[0]['slot_ns']))/1e9
    if seconds<=0:return {'samples':len(rows)}
    counts={name:sum(test(r) for r in rows[1:]) for name,test in (
        ('real',lambda r:r['repeat']=='false' and r['generated']=='false'),
        ('generated',lambda r:r['repeat']=='false' and r['generated']=='true'),
        ('repeated',lambda r:r['repeat']=='true'))}
    def ages(samples):
        values=sorted(float(r['age_ms']) for r in samples)
        return {f'p{q}':values[max(0,math.ceil(len(values)*q/100)-1)] for q in (50,90,99)} if values else {}
    return dict(samples=len(rows),seconds=seconds,counts=counts,
        fresh_fps=(counts['real']+counts['generated'])/seconds,
        slot_fps=(len(rows)-1)/seconds,
        content_age_ms=ages(rows), fresh_content_age_ms=ages(r for r in rows if r['repeat']=='false'),
        note='Private-copy content timestamps to the present call; KHR completion adds WSI latency. Repeated slots are not fresh frames. Not controller-to-photon latency.')
