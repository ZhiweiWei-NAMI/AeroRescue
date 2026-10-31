#!/usr/bin/env python3
"""Materialize and verify the 13 rescue scenes with explicit forced rules.

For every scene the two authored variants are exercised:

* ``configs/<SCENE>.json``                     -- the base scenario config
* ``configs/<SCENE>_upstream_allocated.json``  -- the allocated upstream product

Each variant runs in its own subprocess so a slow or failing upstream
materialization cannot hide the other results.  The report records, per
variant, whether ``materialize(cfg, directory)`` succeeded, the native
objective truth status after the forced-rule sequence, and the action
failures that were observed (never patched).

Usage:
    python tools/verify_all_13_scenes.py [--output runs/verify/scenes_13_matrix.json]
                                         [--only SC01 SC05 ...]
                                         [--timeout-s 1800]
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENES = ['SC01', 'SC03', 'SC04', 'SC05', 'SC07', 'SC08', 'SC09', 'SC10',
          'SC11', 'SC12', 'SC13', 'SC17', 'GT04']

# The canonical config used for the 13-scene pass.  SC01's allocated product is
# a background-only scene (no authored mission entities), so its base config is
# the one that carries the rescue chain.
CANONICAL = {scene: f'{scene}_upstream_allocated.json' for scene in SCENES}
CANONICAL['SC01'] = 'SC01.json'

# Why a scene's canonical config is not simply its allocated product.
NOTES = {
    'SC01': 'The allocated SC01 product is a background-only scene (empty buildings/stations/humans), '
            'so the base config carries the seven mission objectives; the allocated variant is still '
            'materialized and reported.',
    'SC05': 'SC05 is a shore-free open-water source (source_water() publishes banks_en=[]), so the '
            'receiving station is placed on the published waterborne boundary-handoff region '
            '(RESV_SC05_WEST_HANDOFF) with capacity 2 and receiver_confirmation=true instead of a '
            'fabricated shoreline.',
    'GT04': 'The base GT04 config pins the frozen step3 catalog, while the allocated config pins '
            'catalog_allocated.json; one seed directory cannot hold both products. The base config '
            'therefore consumes its own frozen product (upstream/GT04-1-base) via scene_json.',
    'SC17': 'The SC17 allocation targets large factory reservations; generation now uses a bounded '
            'consecutive-miss search limit and a vectorised terrain-support query, so the seed product '
            'is produced once and consumed from cache.',
}


def _variants(scene):
    return [('base', f'{scene}.json'),
            ('upstream_allocated', f'{scene}_upstream_allocated.json')]


def run_one(config_path, output_path):
    """Materialize one config and run the forced-rule sequence."""
    sys.path.insert(0, str(ROOT))
    from rescue_scenes.paths import expand
    from rescue_scenes.backgrounds import materialize
    from rescue_scenes.core.api import Environment
    from rescue_scenes.forced_rules import ForcedRuleDriver

    config_path = Path(config_path).resolve()
    raw = config_path.read_bytes()
    cfg = expand(json.loads(raw))
    started = time.time()
    result = {'config': str(config_path.relative_to(ROOT)), 'materialize_ok': False,
              'error': None, 'stage': 'materialize'}
    with tempfile.TemporaryDirectory(prefix='rescue-13-') as tmp:
        directory = Path(tmp)
        _, bundle = materialize(cfg, directory)
        result['materialize_ok'] = True
        result['materialize_s'] = round(time.time() - started, 2)
        result['stage'] = 'simulate'
        core = deepcopy(bundle['core'])
        core['semantics']['path'] = str(directory / core['semantics']['path'])
        env = Environment.from_config(core)
        sim = env.sim
        driver = ForcedRuleDriver(env, cfg)
        while not driver.done:
            driver.advance(sim.t + cfg['tick_s'])
        report = driver.snapshot()
        objectives = report['objectives']
        # ``partial`` is a legitimate intermediate native result (a bounded
        # work step), never a silent success and never a driver failure.
        failed = [a for a in report['actions'] if a['status'] not in ('succeeded', 'partial')]
        partial = [a for a in report['actions'] if a['status'] == 'partial']
        result.update(
            materialize_ok=True,
            stage='done',
            all_objectives_satisfied=report['all_objectives_satisfied'],
            rule_targets_satisfied=report['rule_targets_satisfied'],
            within_original_horizon=report['within_original_horizon'],
            sim_time_s=report['sim_time_s'],
            objective_total=len(objectives),
            objective_satisfied=sum(bool(o['satisfied']) for o in objectives),
            unsatisfied=[{'id': o['id'], 'kind': o['kind'], 'progress': o['progress']}
                         for o in objectives if not o['satisfied']],
            objective_kinds=sorted({o['kind'] for o in objectives}),
            actions=len(report['actions']),
            action_failures=[{'action': a['action'], 'actor': a['actor'], 'status': a['status'],
                              'failure': a['failure'], 'feedback': a['feedback']}
                             for a in failed],
            partial_actions=[{'action': a['action'], 'actor': a['actor'],
                              'feedback': a['feedback']} for a in partial],
            rules=len(report['rules']),
            resource_gaps=report['resource_gaps'],
            capability_overrides=len(report['capability_overrides']),
        )
    result['seconds'] = round(time.time() - started, 2)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'runs/verify/scenes_13_matrix.json')
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--timeout-s', type=int, default=1800,
                        help='per-variant wall clock guard for the subprocess')
    parser.add_argument('--one', nargs=2, metavar=('CONFIG', 'OUTPUT'),
                        help='internal: run a single config')
    args = parser.parse_args()

    if args.one:
        run_one(args.one[0], args.one[1])
        return 0

    scenes = args.only or SCENES
    matrix = {'schema': 'rescue.scenes13.verify/v1',
              'generated_at': datetime.now(timezone.utc).isoformat(),
              'scenes': {}, 'summary': {}}
    for scene in scenes:
        entry = {'scene': scene, 'canonical_config': f'configs/{CANONICAL[scene]}',
                 'notes': NOTES.get(scene), 'variants': {}}
        for variant, name in _variants(scene):
            config = ROOT / 'configs' / name
            if not config.is_file():
                entry['variants'][variant] = {'config': f'configs/{name}', 'skipped': 'missing config'}
                continue
            with tempfile.TemporaryDirectory(prefix='rescue-13-run-') as tmp:
                out = Path(tmp) / 'result.json'
                command = [sys.executable, str(Path(__file__).resolve()),
                           '--one', str(config), str(out)]
                try:
                    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                          timeout=args.timeout_s)
                except subprocess.TimeoutExpired:
                    entry['variants'][variant] = {'config': f'configs/{name}',
                                                  'materialize_ok': None,
                                                  'error': f'timeout after {args.timeout_s}s',
                                                  'stage': 'timeout'}
                    continue
                if out.is_file():
                    record = json.loads(out.read_text())
                else:
                    record = {'config': f'configs/{name}', 'materialize_ok': False,
                              'error': (proc.stderr or proc.stdout or 'no result')[-4000:],
                              'stage': 'subprocess'}
                record['returncode'] = proc.returncode
                entry['variants'][variant] = record
        canonical = entry['variants'].get(
            'base' if CANONICAL[scene] == f'{scene}.json' else 'upstream_allocated', {})
        entry['canonical_materialize_ok'] = canonical.get('materialize_ok')
        entry['canonical_all_objectives_satisfied'] = canonical.get('all_objectives_satisfied')
        entry['pass'] = bool(canonical.get('materialize_ok')
                             and canonical.get('all_objectives_satisfied')
                             and canonical.get('within_original_horizon'))
        matrix['scenes'][scene] = entry
        print(f"{scene}: materialize={canonical.get('materialize_ok')} "
              f"objectives={canonical.get('objective_satisfied')}/{canonical.get('objective_total')} "
              f"pass={entry['pass']}", flush=True)

    passed = [s for s in scenes if matrix['scenes'][s]['pass']]
    materialized = [s for s in scenes
                    if matrix['scenes'][s]['canonical_materialize_ok']]
    variants_ok = [s for s in scenes
                   if all(v.get('materialize_ok') for v in matrix['scenes'][s]['variants'].values()
                          if not v.get('skipped'))]
    matrix['summary'] = {
        'scenes_requested': len(scenes),
        'scenes_materialized': len(materialized),
        'scenes_objectives_satisfied': len(passed),
        'all_13_materialize': set(SCENES).issubset(materialized),
        'all_13_objectives_satisfied': set(SCENES).issubset(passed),
        'all_variants_materialized': len(variants_ok) == len(scenes),
        'failing_scenes': [s for s in scenes if s not in passed],
        'variant_materialize_failures': [s for s in scenes if s not in variants_ok],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, ensure_ascii=False, indent=2))
    print(json.dumps(matrix['summary'], ensure_ascii=False, indent=2))
    return 0 if len(passed) == len(scenes) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
