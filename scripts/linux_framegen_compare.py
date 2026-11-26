#!/usr/bin/env python3
"""Read an FSR input dump; score generated RGB against its two real neighbours.

The arithmetic midpoint is a reproducible regression reference, not ground truth
for moving/disoccluded pixels. Alpha is the SDK's inpainting weight and is excluded.
Requires numpy; previews additionally require Pillow. Never writes into the input.
"""
import argparse
import json
from pathlib import Path
import numpy as np


FORMATS = {'R16G16B16A16_SFLOAT': ('<f2', 4), 'R16G16_SFLOAT': ('<f2', 2),
           'R32_SFLOAT': ('<f4', 1), 'R16_UNORM': ('<u2', 1), 'R32_UINT': ('<u4', 1)}


def load(directory):
    meta = json.loads((directory / 'inputs.json').read_text())
    images = {}
    for row in meta['images']:
        name = row['name']
        if name in images or Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('invalid/duplicate image name')
        fmt = row['format_name']
        if fmt not in FORMATS:
            raise ValueError(f'unsupported format {fmt}')
        dtype, channels = FORMATS[fmt]
        w, h = row['width'], row['height']
        if not isinstance(w, int) or not isinstance(h, int) or min(w, h) <= 0:
            raise ValueError('invalid dimensions')
        path = directory / (name + '.raw')
        expected = w * h * channels * np.dtype(dtype).itemsize
        if path.stat().st_size != expected:
            raise ValueError(f'{name}: expected {expected} bytes, got {path.stat().st_size}')
        # mmap keeps 4K captures bounded in memory; calculate RGB metrics in row chunks.
        images[name] = np.memmap(path, dtype=dtype, mode='r', shape=(h, w, channels))
    return meta, images


def rgb_metrics(previous, current, generated):
    if previous.shape != current.shape or current.shape != generated.shape or current.shape[-1] != 4:
        raise ValueError('real neighbours/generated must have equal RGBA extents')
    errors, invalid = [], dict(previous=0, current=0, generated=0, reference=0)
    finite_samples, total, squared, maximum, bad_pixels = 0, 0.0, 0.0, 0.0, 0
    for y in range(0, len(current), 64):
        a, b, g = [x[y:y+64, :, :3].astype(np.float32) for x in (previous, current, generated)]
        reference = (a + b) * 0.5
        for name, data in [('previous', a), ('current', b), ('generated', g), ('reference', reference)]:
            invalid[name] += int((~np.isfinite(data)).sum())
        valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(g) & np.isfinite(reference)
        bad_pixels += int((~valid.all(axis=-1)).sum())
        diff = np.abs(g - reference)[valid]
        finite_samples += len(diff)
        if len(diff):
            total += float(diff.sum(dtype=np.float64))
            squared += float(np.square(diff, dtype=np.float64).sum())
            maximum = max(maximum, float(diff.max()))
            errors.append(diff)
    errors = np.concatenate(errors) if errors else np.array([], dtype=np.float32)
    return dict(mae=total/finite_samples if finite_samples else None,
                rmse=(squared/finite_samples)**0.5 if finite_samples else None,
                p95_abs_error=float(np.percentile(errors, 95)) if finite_samples else None,
                max_abs_error=maximum if finite_samples else None,
                finite_rgb_components=finite_samples, nonfinite_rgb_components=invalid,
                nonfinite_comparison_pixels=bad_pixels)


def analyze(directory):
    meta, images = load(Path(directory))
    required = {'previous-real', 'real', 'generated', 'previous-hudless', 'hudless', 'depth', 'motion'}
    if not required <= images.keys():
        raise ValueError('missing pair images: ' + ', '.join(sorted(required-images.keys())))
    previous_id, current_id = meta.get('previous_taa_id', 0), meta.get('current_taa_id', 0)
    if meta.get('reset') or previous_id <= 0 or current_id != previous_id + 1:
        raise ValueError('dump is not an ordered non-reset interpolation pair')
    result = dict(schema=1, reference='(previous-real RGB + real RGB)/2 in stored SDR sRGB code values',
                  caveat='Arithmetic midpoint is not motion-compensated ground truth; compare the same scene/route.',
                  current_taa_id=meta['current_taa_id'], previous_taa_id=meta['previous_taa_id'],
                  midpoint=rgb_metrics(images['previous-real'], images['real'], images['generated']))
    failures = []
    if result['midpoint']['nonfinite_comparison_pixels']:
        failures.append('non-finite RGB in generated/real reference')
    motion = {}
    for name in ('motion-raw', 'motion'):
        if name not in images:
            continue
        v = images[name].astype(np.float32)
        finite = np.isfinite(v).all(axis=-1)
        bad = (~finite)
        out = finite & (np.abs(v) > 1).any(axis=-1)
        finite_values = v[np.isfinite(v)]
        motion[name] = dict(nonfinite_pixels=int(bad.sum()), out_of_range_pixels=int(out.sum()),
                            finite_min=float(finite_values.min()) if finite_values.size else None,
                            finite_max=float(finite_values.max()) if finite_values.size else None)
        if name == 'motion' and (bad.any() or out.any()):
            failures.append('invalid prepared motion')
    result['motion'] = motion
    if 'motion-status' in images:
        counts, n = np.unique(images['motion-status'], return_counts=True)
        result['motion_recovery_status_counts'] = {str(int(k)): int(v) for k, v in zip(counts, n)}
        status = images['motion-status'][..., 0]
        raw, fixed = images['motion-raw'], images['motion']
        same = np.equal(raw.view('<u2'), fixed.view('<u2')).all(axis=-1)
        result['unchanged_motion_modified_pixels'] = int(((status == 0) & ~same).sum())
        if result['unchanged_motion_modified_pixels']:
            failures.append('valid guest motion was modified')
    # Report HUD colour differences without pretending an inferred mask proves absence of HUD.
    a, b = images['real'][..., :3], images['hudless'][..., :3]
    if a.shape != b.shape:
        raise ValueError('HUDless/display extent mismatch')
    hud_finite = np.isfinite(a).all() and np.isfinite(b).all()
    result['hud_color'] = dict(equal_rgb_pixels=int((a == b).all(axis=-1).sum()),
                              differing_rgb_pixels=int((a != b).any(axis=-1).sum()),
                              mean_abs_difference=float(np.abs(a.astype(np.float32)-b).mean()) if hud_finite else None)
    for name in ('hudless', 'previous-hudless', 'depth'):
        if not np.isfinite(images[name][..., :3]).all():
            failures.append(f'non-finite {name}')
    result['failures'] = failures
    return result


def compare_baseline(result, baseline, max_mae_increase):
    a, b = result['midpoint']['mae'], baseline['midpoint']['mae']
    if a is None or b is None:
        return ['midpoint MAE unavailable']
    return ['midpoint MAE regression'] if a > b + max_mae_increase else []


def previews(directory, output):
    from PIL import Image
    _, images = load(directory)
    a, b, g = [images[n][..., :3].astype(np.float32) for n in ('previous-real', 'real', 'generated')]
    reference = (a+b)*0.5
    for name, v in [('midpoint', reference), ('generated-rgb', g), ('error-x8', np.abs(g-reference)*8)]:
        invalid = ~np.isfinite(v).all(axis=-1)
        rgb = (np.clip(np.nan_to_num(v), 0, 1)*255).astype(np.uint8)
        rgb[invalid] = [255, 0, 255]  # visibly distinguish undefined pixels from black
        Image.fromarray(rgb).save(output/(name+'.png'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', type=Path)
    p.add_argument('--output', type=Path, required=True, help='separate report directory')
    p.add_argument('--baseline', type=Path, help='previous comparison.json from the same scene')
    p.add_argument('--max-mae-increase', type=float, default=0.002, help='absolute SDR code-value MAE increase')
    p.add_argument('--previews', action='store_true')
    args = p.parse_args()
    if args.output.resolve() == args.inputs.resolve():
        p.error('output must be separate from the raw input dump')
    if args.max_mae_increase < 0:
        p.error('MAE tolerance must be nonnegative')
    result = analyze(args.inputs)
    if args.baseline:
        result['failures'] += compare_baseline(result, json.loads(args.baseline.read_text()), args.max_mae_increase)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.previews:
        previews(args.inputs, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(bool(result['failures']))


if __name__ == '__main__':
    raise SystemExit(main())
