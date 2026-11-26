"""Summarize serialized pass probes without mistaking inherited NaNs for their origin."""
import json
from collections import Counter
from pathlib import Path


def summarize(run):
    paths = sorted((run/'user').rglob('nan-*/passes.jsonl'))
    result = {'files': len(paths), 'records': 0, 'checked_targets': 0, 'finite_to_nan': 0, 'new_nan_locations': 0, 'already_or_unknown': 0, 'errors': 0, 'early_gameplay_targets': 0}
    lines = ['Counts are float NaN COMPONENTS (not pixels); integer encodings are excluded.',
             'FINITE_TO_NAN requires a finite before-image and the SAME actual cache key after the packet.',
             'NEW_NAN_COMPONENTS detects newly poisoned component locations even when before/after total counts are equal.',
             'Each queue is flushed around the packet; other queues and guest CPU remain live. This perturbs execution.',
             'Events are probe ordinals, not capture draw numbers. Match queue + packet address in capture-<flip>.txt for draw/dispatch indices.',
             'Already-poisoned inputs/targets may originate in an earlier frame. A transition here is not proof of the original source.',
             'Reports keep first 8 inherited/unknown and first 8 finite-to-NaN target/subresource cases separately.',
             'Early mode keeps two uses/program/output layout per phase, max 64 packets/phase, including finite/zero results.',
             'Early input CPU/mirror/images in event-*/before are PRE-operation; event root images/mirror are POST. CPU prefixes remain PRE.',
             'Fogscatter deliberately writes NaN sentinels outside its depth limit. A finite-to-NaN is not automatically a bug. Inspect upstream depth and consumer sampling.']
    for path in paths:
        records = []
        for line in path.read_text().splitlines():
            try: records.append(json.loads(line))
            except json.JSONDecodeError:
                result['errors'] += 1
                lines.append(f'Incomplete JSON record in {path} (possible interrupted capture).')
        result['records'] += len(records)
        result['checked_targets'] += sum(r.get('after') is not None for r in records)
        result['early_gameplay_targets'] += sum(r.get('early', False) and r.get('phase', 0) >= 2 and r.get('after') is not None for r in records)
        classifications = Counter(r.get('classification', 'read_error') for r in records)
        result['finite_to_nan'] += classifications['FINITE_TO_NAN']
        result['new_nan_locations'] += classifications['NEW_NAN_COMPONENTS']
        result['already_or_unknown'] += classifications['ALREADY_NAN_OR_UNKNOWN_BEFORE']
        result['errors'] += classifications['read_error']
        lines.append(f'\n{path.relative_to(run)}: {dict(classifications)}')
        # Keep the first onset/inherited observation of each actual key and requested view.
        seen = set()
        for r in sorted(records, key=lambda x: x['id']):
            if 'error' in r: continue
            c = r.get('classification')
            if c not in ('FINITE_TO_NAN', 'NEW_NAN_COMPONENTS', 'ALREADY_NAN_OR_UNKNOWN_BEFORE') and not r.get('early'): continue
            key = (c, r['actual'], r['requested'])
            if key in seen and not r.get("early"): continue
            seen.add(key)
            lines.append(f"{c} event {r['id']} {r['queue']} {r['kind']} packet {r['packet']} shaders {r['shaders']} phase {r.get('phase')} early {r.get('early', False)}\n  {r['requested']} -> {r['actual']}\n  before {r['before']} after {r['after']}; added {r.get('new_nan_components')} cleared {r.get('cleared_nan_components')} first added {r.get('first_new_nan_component')}; before error {r.get('before_error')}")
        errors = Counter(r['error'] for r in records if 'error' in r)
        lines.append(f'Readback errors/skipped formats: {dict(errors)}')
        lines.append(f'Native outputs and inputs: {path.parent.relative_to(run)}')
    if not paths: lines.append('NO NAN PROBE EVIDENCE: expected in clean/present; otherwise no eligible packet was captured.')
    elif result['already_or_unknown'] and not result['finite_to_nan'] and not result['new_nan_locations']:
        lines.append('No observed finite-to-NaN transition. Inspect inherited inputs/history and unknown-before records; this frame alone cannot establish the original producer.')
    (run/'nan-summary.txt').write_text('\n'.join(lines)+'\n')
    return result
