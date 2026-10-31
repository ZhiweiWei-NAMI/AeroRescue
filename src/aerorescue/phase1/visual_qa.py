"""Machine visual QA and layered contact sheets for phase-1 aerial/ground pairs.

Two independent gates over what :mod:`aerorescue.phase1.scene_renderer` produced:

* ``run_machine_visual_qa`` refuses a rendered section that is not a physically
  plausible sensor pair (wrong shape, empty frame, flat/degenerate RGB, non-unit
  normals, camera inside a solid, co-visibility that contradicts the declared
  regime).  Every check is measured on the rendered buffers, never inferred from
  the camera metadata.
* ``render_contact_sheet`` lays the sections out for human review: per section one
  row of six panels (UAV RGB, ground RGB, UAV depth, UAV normals, true-3-D
  co-visibility / blind-spot overlay, top-down ENU frusta with the branch Q bars).

The pair bundle is the dict returned by ``SceneMultichannelRenderer.render_pair``::

    {'uav': render, 'ground': render, 'covisibility': {...},
     'target_entity_ids': [...], 'regime': 'A'}

A ``render`` is the dict returned by ``SceneMultichannelRenderer.render_view``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

EXPECTED = {'rgb': (384, 640, 3), 'depth': (384, 640),
            'normal_cam': (384, 640, 3), 'normal_world': (384, 640, 3),
            'instance_id': (384, 640)}
MIN_VALID_RATIO = 0.35        # a section may not be mostly empty sky
NORMAL_UNIT_TOL = 1e-3
RGB_MIN_STD = 4.0             # grey-level spread of a real scene
RGB_MIN_UNIQUE = 64           # distinct colours; a flat terrain + road view is legitimately low
NEAR_CLIP_M = 0.1


def _as_array(render, key):
    value = render.get(key)
    return None if value is None else np.asarray(value)


def _view_metrics(render, name):
    """Measured statistics of one rendered view (no metadata shortcuts)."""
    metrics = {'name': name}
    shapes = {}
    for key in ('rgb', 'depth', 'normal_cam', 'normal_world', 'instance_id'):
        array = _as_array(render, key)
        shapes[key] = None if array is None else list(array.shape)
    metrics['shapes'] = shapes
    metrics['dtypes'] = {key: str(np.asarray(render[key]).dtype) for key in EXPECTED if key in render}
    metrics['finite_channels'] = all(np.isfinite(np.asarray(render[key])).all() for key in EXPECTED if key in render)
    rgb = _as_array(render, 'rgb')
    depth = _as_array(render, 'depth')
    normal_cam = _as_array(render, 'normal_cam')
    instance = _as_array(render, 'instance_id')
    if rgb is None or depth is None:
        metrics['error'] = 'missing rgb/depth'
        return metrics
    valid = np.asarray(render.get('valid_mask', depth > 0), bool) & (depth > 0)
    metrics['valid_ratio'] = float(valid.mean())
    metrics['depth_min_m'] = float(depth[valid].min()) if valid.any() else 0.0
    metrics['depth_max_m'] = float(depth[valid].max()) if valid.any() else 0.0
    metrics['depth_median_m'] = float(np.median(depth[valid])) if valid.any() else 0.0
    metrics['near_clip_ratio'] = float((valid & (depth < NEAR_CLIP_M + 1e-3)).sum() / max(valid.sum(), 1))
    metrics['rgb_mean'] = [round(float(v), 2) for v in rgb.reshape(-1, 3).mean(0)]
    metrics['rgb_std'] = [round(float(v), 2) for v in rgb.reshape(-1, 3).std(0)]
    metrics['rgb_unique_colours'] = int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0])
    metrics['rgb_dynamic_range'] = int(rgb.max()) - int(rgb.min())
    if normal_cam is not None and valid.any():
        norm = np.linalg.norm(normal_cam[valid], axis=1)
        metrics['normal_unit_error'] = float(np.abs(norm - 1.0).max())
        metrics['normal_facing_camera_ratio'] = float((normal_cam[valid][:, 2] < 0).mean())
        rotation = np.asarray(render.get('R_c_from_w'), float)
        world = _as_array(render, 'normal_world')
        if rotation.shape == (3, 3) and world is not None:
            expected = world[valid] @ rotation.T
            metrics['normal_frame_error'] = float(np.abs(expected - normal_cam[valid]).max())
    if instance is not None:
        metrics['instance_ids'] = int(np.unique(instance).size)
        metrics['instance_sky_consistency'] = float(((instance == 0) == (~valid)).mean())
        metrics['entity_pixels'] = int(valid.sum())
    metrics['cam_inside_entity'] = render.get('cam_inside_entity')
    stats = render.get('stats') or {}
    metrics['render_stats'] = {k: stats[k] for k in ('triangles', 'samples', 'near_clipped') if k in stats}
    return metrics


def _check_view(metrics, reasons, prefix, regime=None):
    shapes = metrics.get('shapes', {})
    for key, expected in EXPECTED.items():
        got = shapes.get(key)
        if got is None:
            if key in EXPECTED:
                reasons.append(f'{prefix}: missing {key}')
            continue
        if tuple(got) != expected:
            reasons.append(f'{prefix}: {key} shape {tuple(got)} != {expected}')
    if 'error' in metrics:
        return
    if not metrics['finite_channels']:
        reasons.append(f'{prefix}: nonfinite sensor channel')
    for key, dtype in {'rgb':'uint8','depth':'float32','normal_cam':'float32','normal_world':'float32','instance_id':'uint32'}.items():
        if metrics['dtypes'].get(key) != dtype:
            reasons.append(f'{prefix}: {key} dtype must be {dtype}')
    min_std = 2.0 if str(regime) in ('B', 'water') else RGB_MIN_STD
    min_unique = 24
    if metrics['valid_ratio'] < MIN_VALID_RATIO:
        reasons.append(f"{prefix}: valid depth ratio {metrics['valid_ratio']:.3f} < {MIN_VALID_RATIO} (empty frame)")
    if metrics['rgb_dynamic_range'] < 24 or max(metrics['rgb_std']) < min_std:
        reasons.append(f"{prefix}: RGB is flat (range {metrics['rgb_dynamic_range']}, std {metrics['rgb_std']})")
    if metrics['rgb_unique_colours'] < min_unique:
        reasons.append(f"{prefix}: only {metrics['rgb_unique_colours']} distinct colours (degenerate render)")
    mean = float(np.mean(metrics['rgb_mean']))
    if mean < 3.0 or mean > 252.0:
        reasons.append(f'{prefix}: RGB mean {mean:.1f} is black/white')
    if max(metrics['rgb_std']) < 1e-6:
        reasons.append(f'{prefix}: RGB is a single colour')
    if 'normal_unit_error' not in metrics:
        reasons.append(f'{prefix}: no valid pixel to test normals')
    else:
        if metrics['normal_unit_error'] > NORMAL_UNIT_TOL:
            reasons.append(f"{prefix}: normal unit error {metrics['normal_unit_error']:.2e} > {NORMAL_UNIT_TOL}")
        if metrics.get('normal_frame_error', 0.0) > 1e-4:
            reasons.append(f"{prefix}: normal_cam != R_cw @ normal_world ({metrics['normal_frame_error']:.2e})")
    if metrics.get('cam_inside_entity'):
        reasons.append(f"{prefix}: camera is inside solid entity {metrics['cam_inside_entity']}")
    if metrics['near_clip_ratio'] > 0.5:
        reasons.append(f"{prefix}: {metrics['near_clip_ratio']:.2f} of pixels sit on the near plane")
    if metrics.get('instance_sky_consistency', 1.0) < 0.999:
        reasons.append(f"{prefix}: instance_id 0 does not coincide with invalid depth")


def run_machine_visual_qa(pair_render_bundle, regime=None):
    """Assert that one rendered aerial/ground section is a usable sensor pair.

    Returns ``{'passed': bool, 'metrics': dict, 'reasons': list[str]}``.  ``passed``
    is False only for a measured defect; soft observations (a target that is out of
    both frames, an unusually low overlap for the regime) are reported under
    ``metrics['warnings']`` so the caller can decide.
    """
    bundle = pair_render_bundle or {}
    regime = regime or bundle.get('regime') or 'A'
    reasons, warnings = [], []
    metrics = {'regime': regime, 'views': {}, 'warnings': warnings}
    for name in ('uav', 'ground'):
        render = bundle.get(name)
        if not isinstance(render, dict):
            reasons.append(f'{name}: render missing from pair bundle')
            continue
        view = _view_metrics(render, name)
        metrics['views'][name] = view
        _check_view(view, reasons, name, regime=regime)
    if len(metrics['views']) == 2:
        uav, ground = metrics['views']['uav'], metrics['views']['ground']
        metrics['pair'] = {
            'depth_ratio': round(uav['depth_median_m'] / max(ground['depth_median_m'], 1e-6), 3),
            'valid_ratio_min': round(min(uav['valid_ratio'], ground['valid_ratio']), 3),
        }
        covis = bundle.get('covisibility') or {}
        metrics['pair']['covis_pixels'] = int(covis.get('covis_pixels', 0))
        metrics['pair']['overlap_uav'] = round(float(covis.get('overlap_uav', 0.0)), 5)
        metrics['pair']['overlap_ground'] = round(float(covis.get('overlap_ground', 0.0)), 5)
        metrics['pair']['blind_pixels'] = int(covis.get('blind_pixels', 0))
        targets = covis.get('per_entity') or {}
        metrics['targets'] = {eid: {k: row.get(k) for k in
                                    ('uav_pixels', 'uav_tokens', 'ground_pixels', 'ground_state')}
                              for eid, row in targets.items()}
        if covis:
            if regime in ('A', 'B') and metrics['pair']['covis_pixels'] <= 0:
                warnings.append(f"authored regime {regime}: measured zero overlap; classify as C")
            if covis.get('covis_mask_uav') is not None:
                mask = np.asarray(covis['covis_mask_uav'], bool)
                if mask.shape != (uav['shapes']['depth'][0], uav['shapes']['depth'][1]):
                    reasons.append(f"co-visibility mask shape {mask.shape} != UAV frame")
                else:
                    overlap = mask.sum() / max(int(np.asarray(bundle['uav'].get('valid_mask')).sum()), 1)
                    metrics['pair']['mask_overlap_recomputed'] = round(float(overlap), 5)
                    if abs(overlap - metrics['pair']['overlap_uav']) > 1e-3:
                        reasons.append('co-visibility mask and reported overlap disagree')
        for eid, row in (targets or {}).items():
            if not row.get('present'):
                warnings.append(f'target {eid} is not in the instance registry')
            elif not row.get('uav_pixels') and not row.get('ground_pixels'):
                warnings.append(f'target {eid} has no pixel in either view (out of both frames)')
    metrics['passed_checks'] = len(reasons) == 0
    return {'passed': not reasons, 'metrics': metrics, 'reasons': reasons}


# --------------------------------------------------------------------------- contact sheet

def _record_view(record, *names):
    for name in names:
        if isinstance(record.get(name), dict):
            return record[name]
    return None


def _covis_overlay(uav, covis):
    """UAV RGB with true co-visible surface in green and aerial-only blind pixels in red."""
    base = np.asarray(uav['rgb'], np.float32) * 0.55 + 90.0
    mask = np.asarray(covis.get('covis_mask_uav', np.zeros(base.shape[:2])), bool)
    blind = np.asarray(covis.get('blind_mask_uav', np.zeros(base.shape[:2])), bool)
    base[mask] = 0.35 * base[mask] + 0.65 * np.array([40.0, 235.0, 90.0])
    base[blind] = 0.45 * base[blind] + 0.55 * np.array([240.0, 60.0, 50.0])
    return np.clip(base, 0, 255).astype(np.uint8)


def _depth_image(depth, valid):
    depth = np.asarray(depth, np.float32)
    finite = valid & np.isfinite(depth) & (depth > 0)
    if not finite.any():
        return np.zeros(depth.shape + (3,), np.uint8)
    near, far = np.percentile(depth[finite], [2, 98])
    span = max(far - near, 1e-6)
    normalised = 1.0 - np.clip((depth - near) / span, 0, 1)   # bright = near, dark = far
    normalised = np.where(finite, normalised, 0.0)
    try:
        import matplotlib
        colour = matplotlib.colormaps['inferno'](normalised)[..., :3]
    except Exception:  # pragma: no cover - matplotlib is a declared dependency
        colour = np.repeat(normalised[..., None], 3, axis=2)
    colour = colour * finite[..., None]
    return (np.clip(colour, 0, 1) * 255).astype(np.uint8)


def _normal_image(normal_cam, valid):
    normal = np.asarray(normal_cam, np.float32)
    image = np.clip((normal * 0.5 + 0.5) * 255.0, 0, 255)
    image[~valid] = 0
    return image.astype(np.uint8)


def _frustum_panel(axis, record):
    """Top-down ENU plot of both camera frusta, drawn from the real pose and FOV."""
    uav_pose = record.get('uav_pose') or (record.get('uav') or {}).get('cam_pose_enu')
    ground_pose = record.get('ground_pose') or (record.get('ground') or {}).get('cam_pose_enu')
    uav_rpy = record.get('uav_rpy') or (record.get('uav') or {}).get('cam_rpy_deg') or [0, -90, 0]
    ground_rpy = record.get('ground_rpy') or (record.get('ground') or {}).get('cam_rpy_deg') or [0, 0, 0]
    fx = (record.get('uav') or {}).get('K', (457.0, 457.0))[0]
    width = (record.get('uav') or {}).get('width', 640)
    hfov = 2.0 * math.degrees(math.atan(0.5 * width / max(fx, 1e-6)))
    span = float(record.get('frustum_range_m', 45.0))
    for pose, rpy, colour, label in ((uav_pose, uav_rpy, '#2d7dd2', 'UAV'),
                                     (ground_pose, ground_rpy, '#2a9d4a', 'GND')):
        if pose is None:
            continue
        e, n = float(pose[0]), float(pose[1])
        yaw = math.radians(float(rpy[0]))
        for sign in (-1.0, 1.0):
            angle = yaw + sign * math.radians(hfov) / 2.0
            axis.plot([e, e + span * math.cos(angle)], [n, n + span * math.sin(angle)],
                      color=colour, lw=0.9, alpha=0.9)
        axis.plot([e], [n], marker='o', ms=4, color=colour)
        axis.annotate(label, (e, n), textcoords='offset points', xytext=(4, 4), fontsize=6, color=colour)
    axis.set_aspect('equal', adjustable='datalim')
    axis.tick_params(labelsize=5)
    axis.set_xlabel('E (m)', fontsize=6, labelpad=0)
    axis.set_ylabel('N (m)', fontsize=6, labelpad=0)
    axis.set_title('ENU frusta', fontsize=7, pad=2)
    axis.grid(alpha=0.25, lw=0.4)


def _branch_panel(axis, record):
    branch = record.get('branch_q') or record.get('branch_q_hat') or {}
    if not branch:
        axis.axis('off')
        axis.text(0.5, 0.5, 'no branch Q', ha='center', va='center', fontsize=6, color='#888888')
        axis.set_title(r'$\hat{Q}$ by branch', fontsize=7, pad=2)
        return
    labels = list(branch)[:8]
    values = [float(branch[k]) for k in labels]
    colours = ['#2a9d4a' if v >= 0 else '#c0392b' for v in values]
    axis.bar(range(len(values)), values, color=colours, width=0.65)
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels([str(k)[:10] for k in labels], rotation=45, ha='right', fontsize=5)
    axis.tick_params(axis='y', labelsize=5)
    axis.set_title(r'$\hat{Q}$ by branch', fontsize=7, pad=2)
    axis.grid(axis='y', alpha=0.25, lw=0.4)


def render_contact_sheet(records_with_renders, out_png_path, dpi=170, title=None):
    """Six-column review sheet: one row per section, six synchronised panels.

    ``records_with_renders`` is a list of dicts with ``title``, ``uav``, ``ground``,
    ``covisibility`` and optional ``branch_q`` / ``uav_pose`` / ``ground_pose`` /
    ``uav_rpy`` / ``ground_rpy``.  Returns the written path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    records = [r for r in (records_with_renders or []) if isinstance(r, dict)]
    if not records:
        raise ValueError('render_contact_sheet needs at least one record with renders')
    rows = len(records)
    figure = plt.figure(figsize=(6 * 2.35, rows * 1.85 + 0.5), dpi=dpi)
    outer = figure.add_gridspec(rows, 6, wspace=0.06, hspace=0.22,
                                left=0.015, right=0.99, top=0.94, bottom=0.02)
    headers = ['UAV RGB', 'Ground RGB', 'UAV depth', 'UAV normals',
               'co-vis (green) / blind (red)', 'ENU frusta + branch Q']
    for row, record in enumerate(records):
        uav = _record_view(record, 'uav', 'aerial')
        ground = _record_view(record, 'ground', 'ego')
        covis = record.get('covisibility') or record.get('covis') or {}
        axes = []
        for column in range(5):
            axes.append(figure.add_subplot(outer[row, column]))
        axes[0].imshow(uav['rgb'])
        axes[1].imshow(ground['rgb'])
        axes[2].imshow(_depth_image(uav['depth'], np.asarray(uav.get('valid_mask', uav['depth'] > 0), bool)))
        axes[3].imshow(_normal_image(uav['normal_cam'], np.asarray(uav.get('valid_mask', uav['depth'] > 0), bool)))
        axes[4].imshow(_covis_overlay(uav, covis))
        for column, axis in enumerate(axes):
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(headers[column], fontsize=8)
        axes[0].set_ylabel(str(record.get('title', f'section {row}'))[:38], fontsize=7)
        nested = outer[row, 5].subgridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.55)
        _frustum_panel(figure.add_subplot(nested[0]), record)
        _branch_panel(figure.add_subplot(nested[1]), record)
    if title:
        figure.suptitle(title, fontsize=10)
    out_png_path = Path(out_png_path)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png_path, dpi=dpi, facecolor='white')
    plt.close(figure)
    return out_png_path


def write_qa_report(pair_render_bundle, regime, out_json_path):
    """Run the QA gate and persist the measured metrics next to the dataset."""
    report = run_machine_visual_qa(pair_render_bundle, regime)
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
