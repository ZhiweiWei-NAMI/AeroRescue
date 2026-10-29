"""Opt-in scene-v0.6 layout import; execution stays in rescue_scenes.

The upstream owns capacity packing, measured geometry and damage activation.
Layout owns the existing incident/deployment sampling. No upstream public
navigation request or evaluator-only population candidate becomes an agent task.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import math
from pathlib import Path
import struct
import subprocess
import os
import signal
import sys

from .backgrounds import Grid
from .paths import GENERATOR_ROOT, PRIVATE_ROOT, PYTHON
from .layout import (EPS, Layout, expand_rules, footprint_classes,
                     graph_for, inside, polygon_gap, project, rectangle,
                     shortest_route, source_network)

from .paths import ROOT  # relocatable data root
UPSTREAM = GENERATOR_ROOT
PIN = '4a9dccbc1c5377cefabdc3279255c267bb24bf53'
DAMAGE = {'D0': ('none', 'stable'), 'D1': ('minor', 'compromised'),
          'D2': ('major', 'unstable'), 'D3': ('destroyed', 'collapsed')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def route_a():
    # Reuse the pinned implementation, including its nonuniform T0/T1 lookup
    # and physical road surfaces. No copy of its generator/core is installed.
    path = str(UPSTREAM)
    if path not in sys.path:
        sys.path.insert(0, path)
    module = importlib.import_module('rescue_scene.route_a')
    if not Path(module.__file__).resolve().is_relative_to(UPSTREAM.resolve()):
        raise ImportError('rescue_scene.route_a was loaded from a different upstream')
    return module


def _exact_xy_key(e, n):
    # Keep signed zero and exact float coordinates; never round XY.
    return (type(e), e.hex() if isinstance(e, float) else e,
            type(n), n.hex() if isinstance(n, float) else n)


def _cached_surface_lookup(lookup):
    """Memoize one terrain instance's already-loaded, fixed physical layers."""
    cache = {}

    def surface_at(e, n, triangles, bounds):
        # Road and water lookups at the same XY must remain independent.
        key = (id(triangles), id(bounds), *_exact_xy_key(e, n))
        if key not in cache:
            value = lookup(e, n, triangles, bounds)
            if len(cache) >= 65536:
                cache.clear()
            # Upstream normal() exposes the returned list to callers. Store an
            # immutable copy so mutating that result cannot poison later hits.
            cache[key] = None if value is None else (value[0], tuple(value[1]))
        value = cache[key]
        return None if value is None else (value[0], list(value[1]))

    return surface_at


class UpstreamGrid:
    """Grid facade also reopened by the unchanged engine through its descriptor."""

    def __init__(self, descriptor):
        self.descriptor = Path(descriptor).resolve()
        spec = read(self.descriptor)
        self.base = Grid(spec['grid'])
        self.path = self.base.path
        self.e, self.n = self.base.e, self.base.n
        self.u, self.classes, self.names = self.base.u, self.base.classes, self.base.names
        self.bounds = self.base.bounds
        self.airspace_datum_u_m = spec.get('airspace_datum_u_m')
        self.terrain = route_a().RouteATerrain(self.path)
        self.terrain.enable_physical_layers(spec['physical'])
        # Terrain/physical layers are fixed for this descriptor's grid instance.
        # Install only after loading: height/region_at/normal share the same pure
        # surface lookup without changing the pinned upstream class or formula.
        self.terrain._surface_at = _cached_surface_lookup(self.terrain._surface_at)
        self._height_cache = {}

    def core(self, en):
        return self.base.core(en)

    def cell(self, e, n):
        return self.base.cell(e, n)

    def classification(self, e, n):
        if getattr(self.terrain, 'triangle_classes', None) is not None:
            return self.terrain.region_at(e, n)
        return self.base.classification(e, n)

    def height(self, e, n):
        key = _exact_xy_key(e, n)
        if key in self._height_cache:
            return self._height_cache[key]
        value = self.terrain.height(e, n)
        if value is None:
            raise ValueError(f'Position outside upstream support: {e}, {n}')
        if len(self._height_cache) >= 65536:
            self._height_cache.clear()
        self._height_cache[key] = value
        return value

    def enu(self, xy, agl=0):
        e, n = xy[0] + self.e[0], xy[1] + self.n[0]
        datum = self.airspace_datum_u_m if agl > 0 and self.airspace_datum_u_m is not None else self.height(e, n)
        return [e, n, datum + agl]


def scene_id(config):
    """Upstream scene id; only GT04 differs from the project scene id/name."""
    spec = config['placement']['upstream_generated_scene']
    upstream_id = spec.get('upstream_scene_id') or config['background']['id']
    if config['background']['id'] == 'GT04' and upstream_id != 'GT04_RURAL_CONTINUATION':
        raise ValueError('GT04 requires upstream_generated_scene.upstream_scene_id=GT04_RURAL_CONTINUATION')
    return upstream_id


def generation_config(config):
    """Generation input rules, unless the background root ships its own (scenes after the first four)."""
    spec = config['placement']['upstream_generated_scene']
    root = Path(spec['background_root']) / scene_id(config) / '01_structure' / 'layout'
    local = root / f'{scene_id(config)}_route_a_rules.json'
    if local.is_file():
        return str(local)
    return spec['generation_config'] if 'generation_config' in spec else spec['config']


def load_scene(config):
    """Generate into the new lane-owned directory, or consume a frozen product."""
    spec = config['placement']['upstream_generated_scene']
    output = ROOT / 'tools/generated/upstream' / f"{config['background']['id']}-{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    invocation = {'mode': 'consume', 'returncode': None}
    if 'scene_json' not in spec and (output/'scene.json').is_file() and (output/'scene-manifest.json').is_file():
        provenance=read(output/'scene-manifest.json')['generation']['provenance']
        def json_hash(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        requested_targets=sorted(spec.get('coverage_targets',[]))
        if (provenance['catalog_sha256'] != json_hash(spec['catalog']) or
                provenance['config_sha256'] != json_hash(generation_config(config)) or
                provenance['coverage_target_ids'] != requested_targets or
                provenance['input_roots'] != spec['input_roots']):
            raise ValueError('Existing upstream inputs differ from requested catalog/config/coverage/input_roots; retained unchanged: '+str(output))
        spec = dict(spec, scene_json=str(output/'scene.json'))
    if 'scene_json' in spec:
        path = Path(spec['scene_json']).resolve()
        if path.name == 'scene-manifest.json':
            manifest = read(path)
            path = (path.parent / manifest['generation']['scene_file']).resolve()
        # Adapter audits sit beside the consumed product, so consuming another
        # generation tree (e.g. a different catalog) never rewrites this seed's.
        output = path.parent
    else:
        command = [spec.get('python', PYTHON), '-m', 'rescue_scene', 'generate',
                   '--background-root', spec['background_root'], '--scene', scene_id(config),
                   '--catalog', spec['catalog'], '--config', generation_config(config),
                   '--seed', str(config['seed']), '--output', str(output)]
        targets = spec.get('coverage_targets', [])
        for target in targets:
            command += ['--coverage-target', target]
        for root in spec['input_roots']:
            command += ['--input-root', root]
        proc=subprocess.Popen(command,cwd=UPSTREAM,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              text=True,start_new_session=True)
        try:
            stdout,stderr=proc.communicate(timeout=spec.get('timeout_s',180))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL)
            stdout,stderr=proc.communicate()
            write(output/'adapter-invocation.json',{'mode':'generate','status':'timeout',
                  'timeout_s':spec.get('timeout_s',180),'argv':command,'cwd':str(UPSTREAM),
                  'pid':proc.pid,'returncode':proc.returncode,'cleanup':'task-owned subprocess group killed and waited',
                  'stdout_tail':stdout[-4000:],'stderr_tail':stderr[-4000:]})
            raise RuntimeError('Upstream generator timeout; compact diagnostic retained: '+str(output/'adapter-invocation.json'))
        proc.stdout,proc.stderr=stdout,stderr
        (output / 'generator.stdout.log').write_text(proc.stdout, encoding='utf-8')
        (output / 'generator.stderr.log').write_text(proc.stderr, encoding='utf-8')
        invocation = {'mode': 'generate', 'argv': command, 'cwd': str(UPSTREAM), 'returncode': proc.returncode}
        write(output / 'adapter-invocation.json', invocation)
        if proc.returncode not in (0, 2):
            raise RuntimeError(f'Upstream generator exited {proc.returncode}; outputs/logs retained in {output}')
        path = output / 'scene.json'
    scene = read(path)
    manifest_path = path.parent / 'scene-manifest.json'
    manifest = read(manifest_path)
    if scene['scene_id'] != scene_id(config) or scene['seed'] != config['seed']:
        raise ValueError('Upstream scene/seed and target config differ')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['generation']['scene_sha256']:
        raise ValueError('Upstream scene bytes differ from their manifest')
    descriptor = output / 'upstream-grid.json'
    terrain = manifest['terrain']
    resolve = lambda p: str((manifest_path.parent / p).resolve())
    write(descriptor, {'schema': 'rescue.upstream-grid/v1', 'grid': resolve(terrain['grid']),
                       'physical': resolve(terrain['support_mesh']['file']), 'upstream_commit': PIN})
    return spec, scene, manifest, path, output, descriptor, invocation


def imported_row(inst, manifest, separation, baseline):
    """Keep full matrices/current hulls; a reservation is never a physical mesh."""
    poly = deepcopy(inst['polygon_xy_m'])
    reservation = inst.get('coverage_reservation')
    kind = 'human' if inst['role'] == 'person' else 'building' if inst['role'] == 'building' else 'facility'
    relations = {'upstream_role': inst['role']}
    if 'part_of' in inst:
        relations['parent'] = inst['part_of']
    if 'caused_by' in inst:
        relations['caused_by'] = inst['caused_by']
    asset = manifest['assets'][inst['asset_id']]
    reserved = reservation['polygon_xy_m'] if reservation else poly
    if baseline is not None:
        reserved = route_a().hull(reserved + baseline['polygon_xy_m'])
    return {'kind': kind, 'asset_kind': inst['role'], 'asset_id': inst['asset_id'],
            'acceptance_id': inst['acceptance_id'], 'revision': inst['revision'],
            'enu': deepcopy(inst['pose_enu']), 'yaw_deg': inst['yaw_deg'],
            'matrix_enu': deepcopy(inst['matrix_enu']), 'support': deepcopy(inst['support']),
            'dimensions_m': deepcopy(asset['measurement']['dimensions_enu_m']),
            'actual_bbox_enu_m': deepcopy(inst['actual_bbox_enu_m']),
            'footprint_en': deepcopy(reserved),
            'asset_footprint_en': poly, 'region': inst.get('plot_id'),
            'clearance_m': separation / 2, 'relations': relations,
            'state': inst['state'], 'upstream_instance_id': inst['id'],
            'coverage_reservation': deepcopy(reservation), 'visible': inst['visible']}


def generation_damage_audit(scene, rule_log):
    """Trace source state selection without rerunning or revising its draws."""
    normal = {i['id']: i for i in scene['normal_environment']}
    rejected = [r for r in rule_log if r['stage'] == 'disaster' and r['decision'] == 'reject']
    buildings = {}
    for inst in scene['instances']:
        if inst['kind'] != 'building':
            continue
        bid = inst['id']; baseline = normal[bid]['polygon_xy_m']
        # generation.py measures event reach at the D0 hull vertex mean before
        # activation, rather than at the potentially displaced damaged hull.
        center = [sum(p[k] for p in baseline) / len(baseline) for k in (0, 1)]
        events = []
        forcing = []
        for event in scene['events']:
            distance = math.dist(center, event['center_en_m'])
            events.append({**deepcopy(event), 'distance_m': distance,
                           'within_radius': distance <= event['radius_m']})
            anchor = event.get('coverage_anchor_instance_id')
            if anchor == bid or (anchor is None and event['id'] == inst.get('caused_by') and
                                  'coverage-conditioned' in event.get('intensity_source', '')):
                forcing.append(event['id'])
        buildings[bid] = {'generated_state': inst['state'], 'asset_id': inst['asset_id'],
                          'baseline_asset_id': normal[bid]['asset_id'], 'baseline_centroid_en_m': center,
                          'caused_by': inst.get('caused_by'),
                          'event': next((e for e in events if e['id'] == inst.get('caused_by')), None),
                          'impacting_events': [e for e in events if e['within_radius']],
                          'coverage_forced': bool(forcing), 'coverage_forcing_events': forcing,
                          'rejected_damage_draws': [deepcopy(r) for r in rejected if r.get('instance_id') == bid]}
    allocation = scene.get('provenance', {}).get('asset_allocation')
    return {'buildings': buildings,
            'rejection_assignment': 'Only explicit rule-log instance_id is used; unbound draws are not attributed to buildings.',
            'unbound_rejected_damage_draws': [deepcopy(r) for r in rejected if 'instance_id' not in r],
            'state_model': ('Exact independently placed allocation states; normal_environment retains the same source state/geometry for each instance. No D0-to-damage event, fragility prediction or task outcome is inferred.' if allocation else
                            'Source authored event radius and configured damage_weights, exact variant availability and scatter fit; coverage anchors are forced, not physical predictions.'),
            'coverage_exclusion': 'An explicit event coverage_anchor_instance_id excludes that building only; a coverage-conditioned caused_by event without an anchor also excludes its building. Neighbours of an explicit anchor retain their own random draws.'}


def population_classification(inst):
    """Evaluator population roles come from source facts, never asset presence."""
    fields = inst['truth_fields']
    ordinary = {'human.mobility': 'mobile', 'human.location_context': 'outside',
                'human.injury': 'none', 'human.needs': [], 'human.buriedness': 0.,
                'human.alive': True, 'human.carrier_id': None,
                'human.handover_status': 'not_started', 'human.support_received': []}
    missing = sorted(set(ordinary) - fields.keys())
    evidence = [key for key, values in (
        ('human.mobility', ('limited', 'immobile', 'trapped')),
        ('human.injury', ('minor', 'serious', 'critical')),
        ('human.location_context', ('water', 'in_water', 'in water', 'afloat')),
        ('human.alive', (False,))) if key in fields and fields[key] in values]
    if isinstance(fields.get('human.buriedness'), (int, float)) and fields['human.buriedness'] > 0:
        evidence.append('human.buriedness')
    if isinstance(fields.get('human.needs'), list) and fields['human.needs']:
        evidence.append('human.needs')
    issues = ['missing source truth fields: ' + ', '.join(missing)] if missing else []
    healthy = not missing and all(fields[key] == value for key, value in ordinary.items())
    if healthy and 'caused_by' not in inst:
        role = 'ordinary'
    elif evidence:
        role = 'affected'
    else:
        role = None
        issues.append('source fields do not establish an ordinary or affected population role')
    if 'caused_by' in inst and healthy:
        issues.append('caused_by event conflicts with wholly non-casualty source fields')
    return {'population_role': role, 'caused_by_present': 'caused_by' in inst, 'caused_by': inst.get('caused_by'),
            'casualty_evidence_fields': evidence, 'source_truth_fields': deepcopy(fields),
            'inconsistencies': issues}


class ImportedIncidentLayout(Layout):
    """Feed fixed upstream records to the existing incident/deployment sampler."""

    def __init__(self, config, grid):
        # Layout validates the source sidecar against its upstream name, while
        # the project id and configured core hazard profile remain independent.
        source_config = deepcopy(config)
        source_config['background']['id'] = scene_id(config)
        for group in ('buildings', 'debris', 'stations'):
            source_config['placement'].setdefault(group, [])
        super().__init__(source_config, grid)
        self.config = config
        self.incident_audit = {'kind': 'none', 'selected': None, 'rejected': [],
                               'source_policy': 'No building-damage incident configured; retain the authored mission.'}

    def candidate(self, rule, kind):
        if kind == 'building' and getattr(self, 'fixed_source', None) and rule['id'] == self.fixed_source['upstream_instance_id']:
            return deepcopy(self.fixed_source), None
        return super().candidate(rule, kind)

    def place(self, rule, kind):
        if kind == 'station' and rule['id'] in self.records:
            return  # A source facility has already been bound to this service.
        if kind == 'building' and 'source_acceptance_id' in rule:
            if rule.get('occupants'):
                raise ValueError(f"{rule['id']}: source building binding cannot add authored occupants")
            candidates = [(eid, row) for eid, row in self.records.items()
                          if row.get('upstream_instance_id') == eid and row['kind'] == 'building'
                          and row['acceptance_id'] == rule['source_acceptance_id']
                          and row['state'] == rule['damage_grade']]
            if 'different_plot_from' in rule:
                other = self.records[rule['different_plot_from']]['region']
                candidates = [(eid, row) for eid, row in candidates if row['region'] != other]
            candidates = [(eid, row) for eid, row in candidates
                          if row['region'] in self.regions
                          and self.regions[row['region']]['kind'] in rule['allowed_region_kinds']
                          and ('allowed_region_ids' not in rule or row['region'] in rule['allowed_region_ids'])
                          and (not rule.get('require_nearest_access') or 'access' in row['relations'])]
            if not candidates:
                raise ValueError(f"{rule['id']}: no unused upstream building matching acceptance={rule['source_acceptance_id']}, state={rule['damage_grade']} and plot relations")
            eid, row = min(candidates, key=lambda item: hashlib.sha256(
                f"{self.config['seed']}:building:{rule['id']}:{item[0]}".encode()).digest())
            if rule['id'] in self.records:
                raise ValueError(f"{rule['id']}: source building binding would overwrite an existing entity")
            self.records.pop(eid); self.records[rule['id']] = row
            self.aliases[eid] = rule['id']
            row['relations']['source_binding'] = {'source_acceptance_id': rule['source_acceptance_id'],
                                                'source_instance_id': eid, 'authored_role': rule['id']}
            if 'different_plot_from' in rule:
                row['relations']['different_plot_from'] = rule['different_plot_from']
            for other in self.records.values():
                if other['relations'].get('parent') == eid:
                    other['relations']['parent'] = rule['id']
            return
        if 'source_region_id' in rule:
            # SC08's original display GLB has no METADATA_RESERVATION nodes.
            # An explicit deployment may select its published triangle-region
            # boundary instead; never synthesize a reservation or reuse a dog.
            rid = rule['source_region_id']
            source = next((r for r in self.source['reservations'] if r['id'] == rid), None)
            if source is None or source['kind'] not in rule['allowed_region_kinds']:
                raise ValueError(f"{rule['id']}: source_region_id={rid!r} is not an allowed source reservation")
            if ('allowed_region_ids' in rule and rid not in rule['allowed_region_ids']) or rule.get('allow_unreserved_surface'):
                raise ValueError(f"{rule['id']}: source_region_id requires deployment wholly in that allowed reservation")
            if source.get('holes_xy_m'):
                raise ValueError(f"{rule['id']}: source reservation holes require explicit footprint support")
            polygon = deepcopy(source['polygon_en_m'])
            if polygon[0] == polygon[-1]:
                polygon.pop()
            self.regions[rid] = {'id': rid, 'kind': source['kind'], 'polygon_en': polygon}
            self.sources.setdefault('deployment_source_regions', {})[rid] = {
                'layout_json': self.sources['layout_json'], 'source_pointer': f"/reservations/{self.source['reservations'].index(source)}",
                'boundary': 'published primary polygon; whole-footprint source-class and clearance checks retained'}
            rule = {**rule, 'allowed_region_ids': [rid]}
        named = rule.get('anchors') or ([rule['source_anchor']] if 'source_anchor' in rule else [])
        if not named:
            return super().place(rule, kind)
        # Anchored rules (SC13 rope-team-1: the rig ledge) stand on the published source anchor,
        # with the same draw budget/records as place(), one anchor chosen per draw.
        reason = 'anchor draw'
        for _ in range(self.rules['max_draws_per_entity']):
            name = self.rng.choice(named)
            row, reason = super().candidate({**rule, 'mode': 'source_anchor', 'source_anchor': name}, kind)
            if row is not None:
                self.records[rule['id']] = row
                return
        raise ValueError(f"{rule['id']}: exhausted {self.rules['max_draws_per_entity']} draws; rule={rule}; constraint={reason}")

    def approach(self, row, source_access):
        accesses = self.access_for(row['region'])
        access = min(accesses, key=lambda a: math.dist(a['points'][-1], source_access['to_enu'][:2])) if accesses else None
        start, end = source_access['to_enu'][:2], source_access['from_enu'][:2]
        # Shared walks arrive as upstream edge ids; doors facing a street end on a retained source carriageway.
        walk_sources = {e['id'] for e in self.scene['road_edges'] if e['road_id'] == source_access['road_id']} | {source_access['road_id']}
        entrance_roads = [r['id'] for r in self.roads if r['source_id'] in walk_sources]
        if access is not None:
            row['relations']['access'] = access['id']
        row['relations'].update(approach_start_en=list(start),
                                entrance_en=list(end), entrance_road_ids=entrance_roads,
                                approach_waypoints_en=[list(start), list(end)],
                                entrance_heading_deg=math.degrees(math.atan2(end[1]-start[1], end[0]-start[0])),
                                upstream_exterior_approach=deepcopy(source_access), doorway_verified=False,
                                upstream_access_id=source_access.get('id'),
                                upstream_access_pointer=f"/accesses/{self.scene['accesses'].index(source_access)}",
                                adapter_approach_id=source_access['instance_id']+':approach')

    def carriageway_stand(self, eid, vehicle_id):
        """A real body-sized road stop, separate from the foot-only entrance."""
        row = self.records[eid]; vehicle = self.records[vehicle_id]
        access = self.accesses[row['relations']['access']]; landing = access['points'][0]
        roads = [r for r in self.roads if not r['is_access'] and r['source_id'] == access['road_source']
                 and 'wheeled' in r.get('allowed_mobilities', ['foot', 'legged', 'wheeled'])]
        length, width = vehicle['dimensions_m'][:2]
        candidates = []
        for road in roads:
            a, b = road['points']; distance = math.dist(a, b)
            steps = math.ceil(distance/self.spacing)
            points = [project(landing, a, b)[0]]
            points += [[a[k]+(b[k]-a[k])*j/steps for k in (0, 1)] for j in range(steps+1)]
            for point in {tuple(p) for p in points}:
                candidates.append((math.dist(point, landing), road['id'], point, road))
        rejected = []
        for _, _, point, road in sorted(candidates, key=lambda c: (c[0], c[1], c[2])):
            a, b = road['points']; heading = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
            poly = rectangle(point, length, width, heading)
            if any(not (self.grid.e[0] <= p[0] <= self.grid.e[-1] and self.grid.n[0] <= p[1] <= self.grid.n[-1]) for p in poly):
                rejected.append('vehicle body outside source bounds'); continue
            if set(footprint_classes(self.grid, poly, self.spacing)) != {'ROAD_CARRIAGEWAY'}:
                rejected.append('vehicle body not wholly on carriageway'); continue
            blockers = [oid for oid, other in self.records.items() if oid != vehicle_id and
                        polygon_gap(poly, other['footprint_en']) <= max(vehicle['clearance_m'], other['clearance_m'])+EPS]
            if blockers:
                rejected.append('vehicle body clearance: '+', '.join(blockers)); continue
            return {'vehicle_id': vehicle_id, 'vehicle_dimensions_m': [length, width],
                    'stand_en': list(point), 'heading_deg': heading, 'footprint_en': poly,
                    'entrance_ids': [road['id']], 'start_en': list(point),
                    'waypoints_en': [list(point), list(point)], 'plot_access_id': access['id'],
                    'selection': 'nearest body-clear carriageway stop to the source plot-access landing',
                    'rejected_candidates': rejected}
        raise ValueError(f'{eid}: no body-clear carriageway receiving/loading stand for {vehicle_id}; rejections={rejected}')

    def loading_stand(self, building_id, vehicle_id):
        if 'upstream_instance_id' not in self.records[building_id]:
            return super().loading_stand(building_id, vehicle_id)
        # Layout.generate calls this after deployment. Keep the real foot
        # entrance while reusing S5c's measured-body stand/path candidates from
        # the vehicle-compatible plot access, not the foot-only shared walk.
        relation = self.records[building_id]['relations']
        foot = deepcopy(relation); start = self.accesses[relation['access']]['points'][-1]
        try:
            relation['approach_start_en'] = list(start)
            door = relation['entrance_en']
            relation['entrance_heading_deg'] = math.degrees(math.atan2(door[1]-start[1], door[0]-start[0]))
            super().loading_stand(building_id, vehicle_id)
            audit = deepcopy(relation['loading_stand'])
            chosen = next(c for c in audit['candidates'] if c['selected'])
        finally:
            relation.clear(); relation.update(foot)
        relation['loading_stand'] = audit
        if chosen['valid']:
            relation['vehicle_approach'] = {
                'vehicle_id': vehicle_id, 'vehicle_dimensions_m': audit['vehicle_dimensions_m'],
                'stand_en': chosen['stand_en'], 'heading_deg': chosen['heading_deg'],
                'footprint_en': chosen['footprint_en'], 'entrance_ids': [relation['access']],
                'start_en': list(start), 'waypoints_en': chosen['path_en'],
                'plot_access_id': relation['access'], 'selection': 'existing S5c valid measured-body stand and drivable plot-access path'}
        else:
            # S5c records an invalid corridor stop when all candidates fail.
            # Never publish that as reachable: retain its rejections and use
            # a separately checked real road stop, with normal carry limits.
            relation['vehicle_approach'] = self.carriageway_stand(building_id, vehicle_id)

    def receiving_stand(self, station_id, vehicle_id):
        row = self.records[station_id]; relation = row['relations']
        stand = self.carriageway_stand(station_id, vehicle_id)
        access = self.accesses[relation['access']]
        graph = graph_for([r for r in self.roads if 'foot' in r.get('allowed_mobilities', ['foot', 'legged', 'wheeled'])])
        current = tuple(round(v, 7) for v in access['points'][0])
        target = tuple(round(v, 7) for v in relation['approach_start_en'])
        route = shortest_route(graph, current, target)
        if route is None:
            raise ValueError(f'{station_id}: no source foot network from plot access to facility approach')
        points = [stand['stand_en']]
        if math.dist(points[-1], current) > EPS:
            points.append(list(current))
        for rid in route[0]:
            road = self.by_road[rid]
            current = next(tuple(round(v, 7) for v in p) for p in road['points']
                           if tuple(round(v, 7) for v in p) != current)
            points.append(list(current))
        points.append(list(relation['entrance_en']))
        relation['vehicle_approach'] = stand
        relation['receiving_foot_leg'] = {'waypoints_en': points, 'road_ids': route[0],
            'door_en': relation['entrance_en'], 'doorway_verified': False,
            'meaning': 'staffed handover occurs at the carriageway service point; onward foot/carry route to the unchanged facility is published, not pre-executed',
            'runtime_clearance_verified': False}
        row['enu'] = [*stand['stand_en'], self.grid.height(*stand['stand_en'])]

    def person_candidate(self, building, settings):
        bid = building['id']; polygon = building['polygon_xy_m']
        associated = [p for p in self.scene['instances'] if p['role'] == 'person' and p.get('part_of') == bid]
        interior = [p for p in associated if inside(p['pose_enu'][:2], polygon)]
        if interior:
            person = min(interior, key=lambda p: hashlib.sha256(f"{self.config['seed']}:person:{p['id']}".encode()).digest())
            return {'id': person['id'], 'enu': person['pose_enu'], 'existing_instance': True,
                    'source': {'kind': 'upstream_person_inside_building', 'instance_id': person['id']}}, None
        cohort = next((c for c in self.scene['population_truth_candidates'] if c['building_id'] == bid), None)
        if cohort is None:
            return None, 'no upstream population cohort or physical person inside current footprint'
        represented = sum(p.get('caused_by') == building.get('caused_by') and
                          p['truth_fields']['human.mobility'] in ('immobile', 'trapped') for p in associated)
        if settings['candidate_pool'] == 'uninstantiated_affected':
            available = cohort['affected_candidate_count'] - represented
            index = represented + 1
        elif settings['candidate_pool'] == 'uninstantiated_population':
            if cohort['affected_candidate_count'] <= 0:
                return None, 'no affected-person candidate in upstream building cohort'
            available = cohort['population'] - cohort['physical_people_instantiated']
            index = cohort['physical_people_instantiated'] + 1
        else:
            raise ValueError('Unsupported upstream incident candidate_pool: ' + settings['candidate_pool'])
        if available <= 0:
            return None, (f"no available {settings['candidate_pool']} candidate inside {bid}; "
                          f"affected={cohort['affected_candidate_count']}, already represented outdoors={represented}")
        # This is the existing G02 abstract interior position, not a claimed
        # measured doorway, surviving room or certified human support anchor.
        center = [sum(p[k] for p in polygon) / len(polygon) for k in (0, 1)]
        return {'id': f'{bid}-affected-{index:04d}', 'enu': [*center, self.grid.height(*center)],
                'existing_instance': False,
                'source': {'kind': 'upstream_population_truth_candidate', 'building_id': bid,
                           'candidate_index': index, 'cohort': deepcopy(cohort),
                           'candidate_pool': settings['candidate_pool'], 'represented_affected_people': represented,
                           'physical_person_inside_footprint': False,
                           'state_assignment': f"configured {self.config['background']['id']} authored initial condition; source cohort counts retained unchanged",
                           'position_model': 'G02 abstract footprint centroid'}}, None

    def incident(self, rule, buildings, agents):
        settings = self.rules['upstream_generated_scene'].get('incident')
        if not settings:
            raise ValueError(f"{rule['id']}: debris requires an explicit upstream_generated_scene.incident; omit debris for a non-building mission")
        if settings.get('kind', 'building_damage') != 'building_damage':
            raise ValueError(f"{rule['id']}: unsupported upstream incident kind {settings.get('kind')!r}")
        damaged = [i for i in self.scene['instances'] if i['role'] == 'building' and i['state'] in settings['damage_states']]
        forced = [i for i in damaged if self.damage_audit['buildings'][i['id']]['coverage_forced']]
        candidates = [i for i in damaged if not self.damage_audit['buildings'][i['id']]['coverage_forced']]
        candidates.sort(key=lambda i: hashlib.sha256(f"{self.config['seed']}:incident:{i['id']}".encode()).digest())
        self.incident_audit = {'kind': settings.get('kind', 'building_damage'), 'rule': rule['intent'], 'rejected': [], 'selected': None,
            'source_policy': 'Seed-ordered existing generated damage; coverage-forced states are always excluded, including consume mode.',
            'excluded_coverage_forced': [{'building_id': i['id'], 'state': i['state'],
                'event_ids': self.damage_audit['buildings'][i['id']]['coverage_forcing_events']} for i in forced]}
        stage = agents[rule['staging_agent']]['spawn']
        entries = [p for p, links in self.graph.items() if len(links) == 1 and
                   min(p[0]-self.grid.e[0], self.grid.e[-1]-p[0], p[1]-self.grid.n[0], self.grid.n[-1]-p[1])
                   <= stage['boundary_distance_m'] and not self.by_road[links[0][2]]['is_access']]
        debris_length = self.model(rule)['footprint_size_m'][0]
        for building in candidates:
            bid = building['id']; person, reason = self.person_candidate(building, settings)
            if reason:
                self.incident_audit['rejected'].append({'building_id': bid, 'reason': reason})
                continue
            self.fixed_source = deepcopy(self.records[bid])
            access = next((a for a in self.scene['accesses'] if a['instance_id'] == bid), None)
            if access is None or not self.access_for(self.fixed_source['region']):
                self.incident_audit['rejected'].append({'building_id': bid, 'reason': 'no published source exterior/plot access for the configured route-blockage incident'})
                continue
            self.approach(self.fixed_source, access)
            target = self.accesses[self.fixed_source['relations']['access']]['points'][-1]
            routes = [shortest_route(self.graph, entry, tuple(round(v, 7) for v in target)) for entry in entries]
            # Apply the incident's existing route/reach/span condition before
            # spending its joint draw budget on a fixed, ineligible building.
            reachable = any(route and any(not self.by_road[rid]['is_access'] and
                            polygon_gap(building['polygon_xy_m'], self.by_road[rid]['footprint_en']) <= rule['collapse_reach_m'] and
                            math.dist(*self.by_road[rid]['points']) > debris_length + 2*rule['segment_end_clearance_m']
                            for rid in route[0]) for route in routes)
            if not reachable:
                self.incident_audit['rejected'].append({'building_id': bid, 'reason': 'no carriageway segment within collapse reach on a boundary-to-access public shortest route'})
                continue
            original = dict(self.records)
            chosen_rule = {**rule, 'source': bid}
            source_rule = {'id': bid, 'occupants': []}
            try:
                super().incident(chosen_rule, {bid: source_rule}, agents)
            except ValueError as error:
                self.records = original
                if not str(error).startswith(f"{rule['id']} (source {bid}, staging {rule['staging_agent']}): exhausted"):
                    raise
                self.incident_audit['rejected'].append({'building_id': bid, 'reason': str(error)})
                continue
            buildings[bid] = source_rule  # generate() keeps this existing record; no G02 clone.
            rule['source'] = bid
            self.aliases[settings['source_role']] = bid
            self.casualty = person
            if not person['existing_instance']:
                row = self.record({'asset_kind': 'person', 'clearance_m': 0.}, person['enu'][:2],
                                  self.records[bid]['yaw_deg'], self.records[bid]['region'], 'human',
                                  {'parent': bid, 'location_context': settings['person']['location_context'],
                                   'candidate_source': person['source'],
                                   **(settings['person'].get('fields') or {})})
                self.records[person['id']] = row
            self.records[person['id']]['relations'].update(parent=bid, candidate_source=person['source'],
                                                         location_context=settings['person']['location_context'])
            self.incident_audit['selected'] = {'kind': settings.get('kind', 'building_damage'),
                'source_building_id': bid, 'person_id': person['id'],
                'person_source': person['source'], 'blocked_segment': self.records[rule['id']]['relations']['road'],
                'staging_agent': rule['staging_agent'], 'staging_enu': self.records[rule['staging_agent']]['enu'],
                'boundary_entry_en': self.records[rule['staging_agent']]['relations']['boundary_entry_en']}
            return
        raise ValueError(f"{rule['id']} ({rule['intent']}): no existing upstream building/person satisfies rule; "
                         f"rejections={self.incident_audit['rejected']}")


def road_entity_chains(entities, source_ids, protected):
    """Join consecutive source spans for emission only, never the layout graph.

    Every bend and upstream edge record survives. Junctions, referenced roads,
    blocked spans and differing road attributes remain entity boundaries.
    """
    roads = [entity for entity in entities if entity['class'] == 'road']
    degree = {}
    for road in roads:
        for point in (road['geometry']['coords'][0], road['geometry']['coords'][-1]):
            key = tuple(point); degree[key] = degree.get(key, 0) + 1
    result = []; previous = None; previous_attrs = None
    for entity in entities:
        attrs = entity['attributes']
        eligible = (entity['class'] == 'road' and attrs.get('attr.source_road_id') in source_ids
                    and entity['id'] not in protected and not attrs.get('attr.upstream_blocked_by'))
        common = {key: value for key, value in attrs.items() if key != 'attr.upstream_road_edges'}
        if eligible and previous is not None:
            points = entity['geometry']['coords']
            join = previous['geometry']['coords'][-1]
            if join == points[0] and degree[tuple(join)] == 2 and common == previous_attrs:
                previous['geometry']['coords'].extend(points[1:])
                previous['attributes']['attr.upstream_road_edges'].extend(attrs['attr.upstream_road_edges'])
                continue
        if eligible:
            entity = deepcopy(entity)  # Edge lists also belong to the unchanged layout audit.
        result.append(entity)
        previous = entity if eligible else None
        previous_attrs = common if eligible else None
    return result


def adapt(config):
    spec, scene, manifest, path, output, descriptor, invocation = load_scene(config)
    grid = UpstreamGrid(descriptor)
    rules = read(path.parent / 'rules.json')
    layout = ImportedIncidentLayout(config, grid)
    if config['background']['id'] == 'SC05' and layout.water is not None:
        # A pre-airborne UAV measures altitude above the published sea surface,
        # not above the seabed. height() still supplies bathymetry for depth checks.
        grid.airspace_datum_u_m = layout.water['datum_U_m']
        write(descriptor, {**read(descriptor), 'airspace_datum_u_m': grid.airspace_datum_u_m})
    layout.scene = scene; layout.aliases = {}
    layout.damage_audit = generation_damage_audit(scene, read(path.parent / 'rule-log.json'))
    write(output / 'adapter-damage-audit.json', layout.damage_audit)
    population_audit = {
        'rule': 'ordinary: no caused_by and complete non-casualty source fields (mobile/outside/none, no needs, burial, carrier, support or handover); affected: positive source casualty evidence; incident: the explicitly selected incident person. Conflicting/unclassified fields are recorded, never guessed; no role means existing objectives remain.',
        'visibility': 'population_role is pinned evaluator truth only; agent observations/task rules receive no new field.',
        'people': {i['id']: population_classification(i) for i in scene['instances'] if i['role'] == 'person'}}
    normal = {i['id']: i for i in scene['normal_environment']}
    agent_kinds = {agent['id']: agent['kind'] for agent in config['agents']}
    for inst in scene['instances']:
        row = imported_row(inst, manifest, rules['separation_m'], normal.get(inst['id']))
        if inst['id'] in agent_kinds:
            if inst.get('solid_kind') != 'platform' or inst['role'] != agent_kinds[inst['id']]:
                raise ValueError(f"{inst['id']}: configured {agent_kinds[inst['id']]} agent does not match upstream role={inst['role']}, solid_kind={inst.get('solid_kind')}")
            row['kind'] = agent_kinds[inst['id']]
        layout.records[inst['id']] = row
    binding = spec.get('incident_binding')
    if binding is not None:
        if binding.get('kind') != 'allocated_field_operation':
            raise ValueError(f"{config['background']['id']}: unsupported incident_binding kind {binding.get('kind')!r}")
        if spec.get('incident') or layout.rules['debris']:
            raise ValueError(f"{config['background']['id']}: allocated_field_operation cannot also configure a building-damage incident/debris")
        requested = binding.get('source_acceptance_ids', [])
        eligible = [inst for inst in scene['instances'] if inst['acceptance_id'] in requested]
        layout.incident_audit = {'kind': binding['kind'], 'source_acceptance_ids': deepcopy(requested),
                                 'tasks': deepcopy(binding.get('tasks', [])), 'selected': None, 'rejected': [],
                                 'source_policy': 'Bind one existing generated allocated subject; tasks are configured scope, not task creation or successful execution.'}
        if not eligible:
            reason = f"{config['background']['id']}: no generated instance matches incident_binding source_acceptance_ids={requested}"
            layout.incident_audit['rejected'].append({'reason': reason})
            write(output / 'adapter-incident-selection.json', layout.incident_audit)
            raise ValueError(reason)
        subject = min(eligible, key=lambda inst: hashlib.sha256(f"{config['seed']}:binding:{inst['id']}".encode()).digest())
        layout.incident_audit['selected'] = {'kind': binding['kind'], 'source_instance_ids': [subject['id']],
                                            'source_acceptance_ids': [subject['acceptance_id']],
                                            'tasks': deepcopy(binding.get('tasks', [])),
                                            'task_execution_verified': False}
    # Bind an existing medical shell to the already configured staffed service.
    # Its source matrix/footprint stay intact; the staffed receiving point is
    # bound below to a real road stop, without placing a second facility shell.
    for rule in expand_rules(layout.rules['stations']):
        if 'source_acceptance_id' not in rule:
            continue  # Explicit authored mission station, sampled by Layout.
        facilities = [i for i in scene['instances'] if i['role'] == 'facility' and i['acceptance_id'] == rule['source_acceptance_id']]
        if not facilities:
            raise ValueError(f"{rule['id']}: no existing upstream facility for {rule['source_acceptance_id']}")
        inst = min(facilities, key=lambda i: hashlib.sha256(f"{config['seed']}:station:{i['id']}".encode()).digest())
        row = layout.records.pop(inst['id']); layout.records[rule['id']] = row
        layout.aliases[inst['id']] = rule['id']; row['kind'] = 'station'
        row['upstream_pose_enu'] = deepcopy(row['enu'])
    for row in layout.records.values():
        if 'parent' in row['relations']:
            parent = row['relations']['parent']; row['relations']['parent'] = layout.aliases.get(parent, parent)
    # Retain the existing carriageway widths (not road + sidewalk width), add
    # upstream shared paths and exterior approaches, then use the same splitter.
    source = deepcopy(layout.source)
    source_ids = {r['id'] for r in source['roads']} | {a['id'] for a in source['accesses']}
    if config['background']['id'] != 'SC01':
        # Keep each source edge attached to its actual centreline segment as the
        # junction splitter subdivides it. Matching by road_id alone repeats all
        # 3600 SC11 track edges on every segment (quadratic JSON), and spreads a
        # local blocked_by claim over the entire road. Keep SC01's frozen output.
        def segment_key(rid, a, b):
            return rid, tuple(a[:2]), tuple(b[:2])

        edges_by_segment = {}
        for edge in scene['road_edges']:
            key = segment_key(edge['road_id'], edge['from_enu'], edge['to_enu'])
            edges_by_segment.setdefault(key, []).append(edge)
        for road in source['roads']:
            line = road.get('centerline_en_m', road.get('centerline_enu_m'))
            road['upstream_edges_by_segment'] = [
                edges_by_segment.get(segment_key(road['id'], a, b), [])
                for a, b in zip(line, line[1:])]
    # Generated walks/approaches stand on the scene's ground: a list names the candidate
    # surfaces, None keeps the background's own canonical terrain class.
    surfaces = spec.get('walkable_ground_surfaces') or [spec['walkable_ground_surface']]
    ground = [x for x in surfaces if x is not None]
    for edge in scene['road_edges']:
        if edge['road_id'] in source_ids:
            continue
        walk = {'id': edge['id'], 'centerline_enu_m': [edge['from_enu'], edge['to_enu']],
                'carriageway_width_m': edge['width_m'], 'allowed_mobilities': ['foot', 'legged']}
        if config['background']['id'] != 'SC01':
            walk['upstream_edges_by_segment'] = [[edge]]
        if ground:
            walk['surface'] = ground[0]
        source['roads'].append(walk)
    for access in scene['accesses']:
        approach = {'id': access['instance_id'] + ':approach',
                    'centerline_enu_m': [access['from_enu'], access['to_enu']],
                    'carriageway_width_m': access['width_m'], 'allowed_mobilities': ['foot', 'legged']}
        if ground:
            approach['surface'] = ground[0]
        source['roads'].append(approach)
    layout.roads, layout.accesses = source_network(source, layout.regions, config['placement']['approach_depth_m'])
    for road in layout.roads:
        if road['source_id'] not in source_ids:
            road['is_access'] = True  # Shared walks/approaches are not carriageway debris candidates.
    layout.by_road = {r['id']: r for r in layout.roads}
    layout.graph = graph_for(layout.roads)
    access_gaps = []
    for inst in scene['instances']:
        if inst['kind'] == 'building':
            row = layout.records[layout.aliases.get(inst['id'], inst['id'])]
            source_access = next((a for a in scene['accesses'] if a['instance_id'] == inst['id']), None)
            if source_access is None:
                access_gaps.append({'instance_id': inst['id'], 'missing': 'source exterior access',
                                    'effect': 'retain source building geometry/truth; no entrance, approach or interior-access permission invented'})
            else:
                layout.approach(row, source_access)
                if 'access' not in row['relations']:
                    access_gaps.append({'instance_id': inst['id'], 'missing': 'source plot-access adapter connection',
                                        'effect': 'retain published exterior approach; no vehicle plot access or receiving stand invented'})
    try:
        generated = layout.generate()
    except ValueError:
        if hasattr(layout, 'incident_audit'):
            if layout.sources.get('unconnected_accesses'):
                layout.incident_audit['unconnected_source_accesses'] = deepcopy(layout.sources['unconnected_accesses'])
            write(output / 'adapter-incident-selection.json', layout.incident_audit)
        raise
    write(output / 'adapter-incident-selection.json', layout.incident_audit)
    rows = generated['entities']
    selected = layout.incident_audit['selected'] if hasattr(layout, 'casualty') else None
    for rule in expand_rules(layout.rules['stations']):
        if 'source_acceptance_id' in rule and selected is not None:
            layout.receiving_stand(rule['id'], selected['staging_agent'])
    for row in rows.values():
        if 'support' not in row:
            row['support'] = {'kind': 'existing_layout_source_support', 'height_m': row['enu'][2]}
    entities = []

    def add(eid, cls, typ, points, attrs, authored=False):
        points = [*points, points[0]] if typ == 'polygon' else points
        entities.append({'id': eid, 'class': cls, 'geometry': {'type': typ, 'coords': [grid.core(p) for p in points]},
                         'attributes': attrs, 'label_source': 'authored' if authored else 'upstream'})

    def navigation_fields(row):
        relation = row['relations']
        fields = {'building.entrance_ids': relation['entrance_road_ids'],
                  'attr.entrance_xy': grid.core(relation['entrance_en']),
                  'attr.entrance_heading_deg': relation['entrance_heading_deg'],
                  'attr.approach_start_xy': grid.core(relation['approach_start_en']),
                  'attr.approach_waypoints_xy': [grid.core(p) for p in relation['approach_waypoints_en']],
                  'attr.interior_access_via_entrance': True, 'attr.doorway_verified': False}
        if 'vehicle_approach' in relation:
            stand = relation['vehicle_approach']
            fields['attr.vehicle_approach'] = {'entrance_ids': stand['entrance_ids'],
                'stand_xy': grid.core(stand['stand_en']), 'heading_deg': stand['heading_deg'],
                'start_xy': grid.core(stand['start_en']),
                'waypoints_xy': [grid.core(p) for p in stand['waypoints_en']]}
        return fields

    for road in generated['roads']:
        attrs = {'road.width_m': road['width_m'], 'road.surface': road.get('surface', 'asphalt'),
                 'attr.source_road_id': road['source_id']}
        for key in ('source_surface', 'allowed_mobilities', 'professional_only', 'source_pointer'):
            if key in road:
                attrs['attr.' + key] = road[key]
        if 'upstream_road_edges' in road:
            # source_network copies the originating segment's edges through
            # every junction cut, not the edge list of the entire source road.
            matching = road['upstream_road_edges']
        else:
            matching = [e for e in scene['road_edges'] if e['road_id'] == road['source_id'] or e['id'] == road['source_id']]
        attrs['attr.upstream_road_edges'] = matching
        attrs['attr.upstream_blocked_by'] = sorted({eid for edge in matching for eid in edge['blocked_by']})
        add(road['id'], 'road', 'polyline', road['points'], attrs, road.get('authored', False))
    if 'water' in generated:
        water = generated['water']
        add(water['id'], 'water', 'polygon', water['polygon_en'],
            {'attr.datum_U_m': water['datum_U_m'], 'attr.source_stage_id': water['id']})
    for region in generated['regions'].values():
        isolated = region['kind'] == 'metadata_only_cliff_side_professional_only'
        if region['kind'] in config['background']['plot_region_kinds'] or isolated:
            attrs = {'attr.role': 'candidate_building_plot', 'attr.reservation_kind': region['kind']}
            if isolated:
                attrs['attr.ground_access_isolated'] = True
            add(region['id'], 'open_area', 'polygon', region['polygon_en'], attrs)
    opts = {'cause': 'upstream_generated_layout_with_existing_incident', 'pinned_buildings': {},
            'pinned_occupants': {}, 'pinned_humans': {}, 'placed_humans': {},
            'outdoor_count': 0, 'no_debris_from': [], 'obstacles': []}
    if selected is None:
        opts['cause'] = 'upstream_generated_layout_with_authored_mission'
    metadata_gaps = list(access_gaps)
    # These explicit mappings come from the workbook, not from mesh size.
    building_metadata = spec.get('building_metadata', {})
    for inst in scene['instances']:
        eid = layout.aliases.get(inst['id'], inst['id']); row = rows[eid]
        if row['kind'] == 'station':
            continue  # Emitted below with the configured service fields.
        if eid in agent_kinds:
            continue  # build_agents creates the platform, never a duplicate facility shell.
        attrs = {'entity.pose': grid.core(row['enu']), 'entity.height_m': row['dimensions_m'][2],
                 'attr.acceptance_id': inst['acceptance_id'], 'attr.upstream_role': inst['role'],
                 'attr.upstream_state': inst['state'], 'attr.upstream_support': inst['support']}
        if 'caused_by' in inst:
            attrs['attr.upstream_caused_by'] = inst['caused_by']
        if inst['motion_blocker'] and row['kind'] != 'human':
            attrs['attr.blocks_ground_navigation'] = True
            attrs['attr.navigation_footprint'] = {'type': 'polygon', 'coords': [grid.core(p) for p in row['asset_footprint_en']]}
        if inst['kind'] == 'building' and 'upstream_exterior_approach' in row['relations']:
            attrs.update(navigation_fields(row))
        if row['kind'] == 'human':
            fields = deepcopy(inst['truth_fields'])
            role = population_audit['people'][inst['id']]['population_role']
            if role is not None:
                fields['attr.population_role'] = role
            opts['placed_humans'][eid] = {'xy': grid.core(row['enu']), 'location_context': fields['human.location_context']}
            opts['pinned_humans'][eid] = fields
            if 'part_of' in inst:
                parent = inst['part_of']
                opts['pinned_humans'][eid]['entity.parent_id'] = layout.aliases.get(parent, parent)
        elif row['kind'] == 'building':
            family = inst['asset_id'].rsplit('-', 1)[0]
            meta = building_metadata[family]
            attrs.update(meta)
            # The adapter supplies the explicit workbook/config mappings; do
            # not invoke the ingester's legacy authored-default fill for them.
            for key in ('building.use', 'building.structure', 'building.floors', 'building.capacity'):
                if key not in attrs:
                    raise ValueError(f'{eid}: missing explicit building_metadata field {key}; source outputs remain in {path.parent}')
            attrs['attr.damage_grade'] = inst['state']
            attrs['attr.navigation_footprint'] = {'type': 'polygon', 'coords': [grid.core(p) for p in row['asset_footprint_en']]}
            attrs['attr.population_instantiation_policy'] = 'import physical source instances; no G02 resampling; one explicitly selected cohort candidate may be materialized'
            add(eid, 'building', 'polygon', row['asset_footprint_en'], attrs)
            # Source generation, not core G04 exposure x fragility, governs
            # these grades. Unpinned D1-D3 roof/wall numbers remain synthetic.
            damage, structural = DAMAGE[inst['state']]
            opts['pinned_buildings'][eid] = {'building.visual_damage': damage, 'building.structural_condition': structural,
                                              'building.render_variant': inst['asset_id']}
            if inst['state'] == 'D0':
                opts['pinned_buildings'][eid].update({'building.roof_loss_ratio': 0.0,
                                                     'building.wall_damage': {'crack_ratio': 0.0, 'breach_ratio': 0.0}})
            # This is an explicit no-duplicate-generation instruction, never a
            # claim that the building's population is zero or known to agents.
            opts['pinned_occupants'][eid] = 0
            opts['no_debris_from'].append(eid)  # scatter already belongs to this mesh
        else:
            add(eid, 'facility', 'polygon', row['asset_footprint_en'], attrs)
    if selected is not None:
        casualty = layout.casualty
        person_fields = {'human.' + key: value for key, value in spec['incident']['person'].items()}
        person_fields['entity.parent_id'] = selected['source_building_id']
        person_fields['attr.population_candidate_source'] = casualty['source']
        person_fields['attr.population_role'] = 'incident'
        source_classification = population_audit['people'].get(casualty['id'])
        population_audit['people'][casualty['id']] = {
            'population_role': 'incident', 'source_classification': source_classification,
            'candidate_source': casualty['source'], 'configured_state': deepcopy(spec['incident']['person']),
            'state_model': f"Cohort affected count is a source Bernoulli draw from trap_probability_by_state; the selected person mobility/injury/buriedness/needs are authored {config['background']['id']} configuration, not predicted entrapment."}
        opts['placed_humans'][casualty['id']] = {'xy': grid.core(casualty['enu']),
                                               'location_context': spec['incident']['person']['location_context']}
        opts['pinned_humans'].setdefault(casualty['id'], {}).update(person_fields)
    for rule in generated['debris_rules']:
        row = rows[rule['id']]; rel = row['relations']
        opts['obstacles'].append({'id': rule['id'], 'pose': grid.core(row['enu']), 'road': rel['road'],
                                  'source': rel['source'], 'volume_m3': rule['volume_m3'], 'height_m': rule['height_m'],
                                  'work': rule['work'], 'blocked_width_m': rel['blocked_width_m']})
    for eid, rule in generated['building_rules'].items():
        row = rows[eid]
        if 'upstream_instance_id' in row:
            continue
        add(eid, 'building', 'polygon', row['footprint_en'],
            {'building.use': 'residential', 'building.structure': 'masonry', 'building.floors': 1,
             'building.capacity': max(2, len(rule['occupants'])), 'attr.damage_grade': rule['damage_grade'],
             'building.entrance_ids': [row['relations']['access']],
             'attr.navigation_footprint': {'type': 'polygon', 'coords': [grid.core(p) for p in row['asset_footprint_en']]},
             'attr.entrance_xy': grid.core(row['relations']['entrance_en']),
             'attr.entrance_heading_deg': row['relations']['entrance_heading_deg'],
             'attr.approach_start_xy': grid.core(row['relations']['approach_start_en']),
             'attr.approach_waypoints_xy': [grid.core(p) for p in row['relations'].get('approach_waypoints_en', [])],
             'attr.plot_id': row['region'], 'attr.access_id': row['relations']['access']}, True)
        damage, structural = DAMAGE[rule['damage_grade']]
        opts['pinned_buildings'][eid] = {'building.visual_damage': damage, 'building.structural_condition': structural}
        opts['pinned_occupants'][eid] = len(rule['occupants'])
        opts['no_debris_from'].append(eid)
        for index, person in enumerate(rule['occupants'], 1):
            if person['location_context'] != 'interior':
                raise ValueError(f'{eid}: occupant rule requires location_context=interior')
            opts['pinned_humans'][f'human-{eid}-{index}'] = {f'human.{k}': v for k, v in person.items()}
    for rule in expand_rules(layout.rules['stations']):
        row = rows[rule['id']]; relation = row['relations']
        attrs = {'station.kind': rule['kind'], 'station.capacity': rule['capacity'], 'station.power_W': rule['power_W'],
                 'attr.receiver_confirmation': rule['receiver_confirmation'], 'attr.plot_id': row['region']}
        if 'access' in relation:
            attrs['attr.access_id'] = relation['access']
        if 'upstream_instance_id' in row:
            if 'upstream_exterior_approach' not in relation:
                raise ValueError(f"{rule['id']}: source station has no published exterior approach; receiving access is not invented")
            attrs.update({'attr.upstream_instance_id': row['upstream_instance_id'],
                          'attr.blocks_ground_navigation': True, **navigation_fields(row),
                          'attr.navigation_footprint': {'type': 'polygon', 'coords': [grid.core(p) for p in row['asset_footprint_en']]}})
        if 'water_access_polygon_en' in relation:
            polygon = relation['water_access_polygon_en']
            attrs['station.access_region'] = {'type': 'polygon', 'coords': [grid.core(p) for p in [*polygon, polygon[0]]]}
        add(rule['id'], 'station', 'point', [row['enu']], attrs, True)
    for group in ('facilities', 'equipment'):
        for rule in expand_rules(config['placement'].get(group, [])):
            row = rows[rule['id']]
            attrs = {**rule['fields'], 'attr.plot_id': row['region'], 'entity.pose': grid.core(row['enu'])}
            if 'parent' in row['relations']:
                attrs['entity.parent_id'] = row['relations']['parent']
            add(rule['id'], 'facility', 'polygon', row['asset_footprint_en'], attrs, True)
    for rule in expand_rules(config['placement'].get('humans', [])):
        eid = rule['id']; row = rows[eid]
        fields = {**{f'human.{k}': v for k, v in rule['human'].items()}, **rule.get('fields', {})}
        opts['placed_humans'][eid] = {'xy': grid.core(row['enu']), 'location_context': fields['human.location_context']}
        opts['pinned_humans'][eid] = fields
    for rule in config['placement'].get('hazards', []):
        row = rows[rule['geometry_from']]
        polygon = row['asset_footprint_en'] if rule['geometry_kind'] == 'asset_footprint' else row['footprint_en']
        opts.setdefault('pinned_hazards', {})[rule['id']] = {
            'geometry': {'type': 'polygon', 'coords': [grid.core(p) for p in [*polygon, polygon[0]]]},
            'fields': deepcopy(rule['fields'])}
    if 'station_stock' in config['placement']:
        opts['station_stock'] = deepcopy(config['placement']['station_stock'])
    if 'public_brief_zh' in config:
        opts['public_brief_zh'] = config['public_brief_zh']
    listing = read(config['background']['scene_list'])
    source_roles = {}
    for inst in scene['instances']:
        source_roles[inst['role']] = source_roles.get(inst['role'], 0) + 1
    consumption = {
        'source_scene_json': str(path), 'source_scene_sha256': manifest['generation']['scene_sha256'],
        'generated_instance_count': len(scene['instances']), 'generated_counts_by_role': source_roles,
        'generated_instance_ids': [inst['id'] for inst in scene['instances']],
        'empty_generation': not scene['instances'],
        'generated_nonplatform_instance_count': sum(inst.get('solid_kind') != 'platform' for inst in scene['instances']),
        'imported_runtime_entities': {eid: row['upstream_instance_id'] for eid, row in rows.items()
                                      if 'upstream_instance_id' in row},
        'authored_mission_placements': {eid: {'kind': row['kind'], 'asset_id': row['asset_id'],
                                            'enu': deepcopy(row['enu']), 'region': row['region']}
                                        for eid, row in rows.items() if 'upstream_instance_id' not in row},
        'authored_hazards': deepcopy(config['placement'].get('hazards', [])),
        'meaning': 'Generated source instances and separately sampled/authored mission placements are distinct evidence. Empty generation is no environment-generation evidence; platform-only generation is platform presence only. Neither proves meaningful scene coverage, visible geometry, task execution or mission success.',
        'meaningful_scene_generation_verified': False,
        'background_support_source': config['background']['geometry_source'],
    }
    if 'provenance' in config:
        consumption['mission_provenance'] = deepcopy(config['provenance'])
    # Kept beside a consumed product as well as in the runtime bundle audit.
    if config['background']['id'] != 'SC01':
        write(output / 'adapter-consumption-audit.json', consumption)
    upstream_audit = {'commit': PIN, 'scene_json': str(path), 'manifest': str(path.parent / 'scene-manifest.json'),
                      'runtime_grid': str(descriptor), 'invocation': invocation, 'summary': scene['summary'],
                      'normal_environment': scene['normal_environment'], 'events': scene['events'],
                      'relations': scene['relations'], 'accesses': scene['accesses'], 'road_edges': scene['road_edges'],
                      'coverage_plan': scene['coverage_plan'], 'gaps': scene['gaps'],
                      'population_truth_candidates': scene['population_truth_candidates'],
                      'population_classification': population_audit,
                      'generation_damage': layout.damage_audit,
                      'navigation': {eid: {k: deepcopy(row['relations'][k]) for k in (
                          'upstream_access_id', 'upstream_access_pointer', 'adapter_approach_id',
                          'upstream_exterior_approach', 'entrance_road_ids', 'entrance_en',
                          'entrance_heading_deg', 'approach_start_en', 'approach_waypoints_en',
                          'doorway_verified', 'loading_stand', 'vehicle_approach', 'receiving_foot_leg') if k in row['relations']}
                          for eid, row in rows.items() if 'upstream_exterior_approach' in row['relations']},
                      'metadata_gaps': metadata_gaps, 'metadata_sources': spec.get('metadata_sources', {}),
                      'provenance': scene['provenance'],
                      'entity_aliases': layout.aliases, 'incident': layout.incident_audit,
                      'incident_policy': f"existing incident sampler ({layout.incident_audit['kind']}) uses a fixed imported damaged building and existing person/cohort; a source facility is the station when configured; only route debris and agent deployments are sampled",
                      'state_activation': 'upstream authored_generation applied before bootstrap; not a runtime action',
                      'coverage_evidence': 'generated placement only; render/task validation remains pending'}
    if config['background']['id'] != 'SC01':
        upstream_audit['consumption'] = consumption
        upstream_audit['coverage_evidence'] = (
            'no generated instances; authored mission placement is separate evidence' if not scene['instances']
            else 'generated platform presence only; authored mission placement is separate evidence'
            if consumption['generated_nonplatform_instance_count'] == 0
            else 'generated instance placement only; render/task validation remains pending')
    if selected is None:
        upstream_audit['incident_policy'] = 'No building-damage incident selected; retain authored mission placement and imported truth without inventing debris or a trapped casualty.'
    audit = {'background_entry': next(r for r in listing['scenes'] if r['sid'] == config['background']['id']),
             'layout': rows, 'slots': {eid: {'enu': r['enu'], 'plot': r['region'], 'role': r['kind'], **r['relations']} for eid, r in rows.items()},
             'layout_network': {k: generated[k] for k in ('roads', 'regions', 'accesses')},
             'layout_sources': {**generated['sources'], 'upstream_scene': str(path), 'upstream_manifest_digest': manifest['digest']},
             'layout_pairwise': generated['pairwise'], 'upstream': upstream_audit,
             'transform': {'core_xy_offset': [-grid.e[0], -grid.n[0]], 'render': 'ENU -> [E,U,-N]',
                           'height': 'pinned RouteATerrain T0/T1 plus physical road support',
                           'yaw': 'degrees, forward East=0, North=90; full upstream matrix retained'},
             'grid_sha256': manifest['terrain']['grid_sha256'], 'geometry_source': config['background']['geometry_source'],
             'assumptions': config['assumptions']}
    if config['background']['id'] in ('SC07', 'SC12', 'SC13', 'SC17'):
        audit['source_features'] = deepcopy(layout.source['features'])
    if 'water' in generated:
        audit['layout_network']['water'] = generated['water']
        audit['transform']['height'] += '; afloat mission poses use source water datum'
    if spec.get('road_entity_chain_sources'):
        # SC11 opts in its unblocked track: wind/target movement introduces no
        # local road closures. Do not generalize this to flood/fire/debris roads,
        # whose runtime passability is deliberately entity-wide.
        protected = {slot['road'] for slot in opts['obstacles']}
        for entity in entities:
            attrs = entity['attributes']
            protected.update(attrs.get('building.entrance_ids', []))
            protected.update(attrs.get('attr.vehicle_approach', {}).get('entrance_ids', []))
            if 'attr.access_id' in attrs:
                protected.add(attrs['attr.access_id'])
        entities = road_entity_chains(entities, spec['road_entity_chain_sources'], protected)
    doc = {'schema': 'rescue.upstream-semantics/v1', 'source': {'kind': 'upstream_annotation',
           'producer': 'rescue_scenes.upstream_scene', 'terrain_ref': str(grid.path),
           'note': f'Pinned scene-v0.6 {PIN}; {path}; source events already applied; existing incident is authored.'},
           'frame': {'crs': 'local_ENU_m', 'bounds_m': grid.bounds}, 'entities': entities}
    return grid, doc, opts, audit


def bind_assets(bundle, directory, snapshot, models, asset_audit):
    """Bind imported instances through existing AAS types/poses without changing
    engine/presentation/assets. The wrapper cancels the engine's display frame,
    then applies the exact upstream matrix to the untouched GLB roots.
    """
    from .glb_runtime import optimize, merge_by_material, opaque_blend, compact_textures, display_lods, semantic_textures, externalize_images, quantize_geometry, self_contained_level
    from .presentation import BASE, AGENT, FIELDS, POSITION, ORIENTATION

    upstream = bundle['audit']['upstream']; manifest_path = Path(upstream['manifest'])
    manifest = read(manifest_path); grid = UpstreamGrid(upstream['runtime_grid'])
    mapping, presentation, source_assets = {}, [], {}
    shared_bindings = {}  # (asset, exact wrapper) -> (typeId, model): identical wrappers draw as one instanced batch
    agent_kinds = {agent['id']: agent['kind'] for agent in bundle['config']['agents']}
    agent_ids = set(agent_kinds)
    if bundle['config']['display'].get('human_asset_policy') == 'delivered_base_only':
        # Reuse the already staged faithful static BASE human for automatic
        # display states too; mobility truth does not imply a delivered pose.
        for kind in ('human_prone', 'human_supine'):
            previous = asset_audit.get('assets', {}).get(kind, {})
            models[kind] = deepcopy(models['human'])
            asset_audit['assets'][kind] = {
                **deepcopy(asset_audit['assets']['human']),
                'substitution': {'requested_asset_id': previous.get('asset_id'),
                                 'delivered_asset_id': 'HUM-001-C01-BASE',
                                 'reason': 'BASE-only mission display; no pose-specific geometry claimed'}}
        if 'boat_loaded' in models:
            models['boat_loaded'] = deepcopy(models['boat'])
            asset_audit['assets']['boat_loaded'] = {
                **deepcopy(asset_audit['assets']['boat']),
                'display_gap': 'Use the existing boat without its embedded posed person; carriage truth remains authoritative, but no on-deck body is depicted.'}

    def multiply(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

    # Raw GLB Y-up -> ENU, matching upstream route_a.glb_primitives.
    cframe = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    for eid, row in bundle['audit']['layout'].items():
        if 'upstream_instance_id' not in row:
            continue
        asset = manifest['assets'][row['asset_id']]
        source = (manifest_path.parent / asset['file']).resolve()
        if row['asset_id'] not in source_assets:
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != asset['sha256']:
                raise ValueError('Upstream asset bytes changed: ' + str(source))
            length, tag = struct.unpack_from('<II', raw, 12)
            if tag != 0x4E4F534A:
                raise ValueError('Upstream asset lacks a GLB JSON chunk: ' + str(source))
            # The viewer cannot draw the skinned source (it falls back to a marker); the
            # project's existing static export of the same accepted asset keeps its frame.
            static = PRIVATE_ROOT / 'assets/glb' / f"{row['asset_id']}_static.glb"
            if json.loads(raw[20:20 + length]).get('skins') and static.is_file():
                source_assets.setdefault('static_substitutions', {})[row['asset_id']] = str(static)
                raw = static.read_bytes(); length, tag = struct.unpack_from('<II', raw, 12)
            doc = json.loads(raw[20:20 + length]); binary = b''
            if 20 + length < len(raw):
                size, tag = struct.unpack_from('<II', raw, 20 + length)
                if tag != 0x004E4942:
                    raise ValueError('Upstream asset has an unexpected GLB chunk: ' + str(source))
                binary = raw[28 + length:28 + length + size]
            if len(doc.get('buffers', [])) != 1 or 'uri' in doc['buffers'][0]:
                raise ValueError('Upstream asset is not a single embedded-buffer GLB: ' + str(source))
            # Same display texture budget as every staged project model (assets.normalize_glb).
            doc, binary, texture_record = optimize(doc, binary, 1024)
            # Display-only batching: primitives baked to asset space and joined per material
            # (identical vertices/UVs/materials; checked in dev/step3/merge_check). Footprints,
            # navigation and rescue truth never read this copy.
            doc, binary, merge_record = merge_by_material(doc, binary)
            doc, binary, alpha_record = opaque_blend(doc, binary)
            doc, binary, compact_record = compact_textures(doc, binary)
            source_assets.setdefault('display_records', {})[row['asset_id']] = {'textures': texture_record, 'merge': merge_record, 'alpha_mode': alpha_record, 'texture_payload': compact_record}
            # Display-only distance levels (same asset, same damage state, decimated geometry), computed on the
            # compacted copy (cache key). Declared to the viewer through asset.extras.aasLod on each wrapper.
            levels = display_lods(doc, binary, Path(os.environ.get('RESCUE_LOD_CACHE', str(ROOT / 'tools/generated/display_lod_cache'))))
            # Semantic texture budget, then images as content-addressed files shared across assets/levels.
            doc, binary, semantic_record = semantic_textures(doc, binary)
            doc, binary, image_files = externalize_images(doc, binary)
            doc, binary, quantize_record = quantize_geometry(doc, binary)
            source_assets['display_records'][row['asset_id']] |= {'semantic_textures': semantic_record, 'quantize': quantize_record}
            for name, data in image_files.items():
                image_path = Path(directory) / 'assets' / name; image_path.parent.mkdir(parents=True, exist_ok=True); image_path.write_bytes(data)
                models['upstream_img:' + name] = {'url': '/data/rescue_assets/' + name, 'file': str(image_path)}
            # One shared geometry binary per formal asset; every instance wrapper references it,
            # so the viewer downloads/caches the mesh once, not per instance.
            shared = Path(directory) / 'assets' / f"upstream-asset-{row['asset_id']}-{hashlib.sha256(binary).hexdigest()[:8]}.bin"
            shared.parent.mkdir(parents=True, exist_ok=True); shared.write_bytes(binary)
            models['upstream_bin:' + row['asset_id']] = {'url': '/data/rescue_assets/' + shared.name, 'file': str(shared)}
            lods = []
            for level, (template, lod_binary, lod_record) in enumerate(levels, 1):
                template = self_contained_level(template, doc.get('images'))
                lod_bin = shared.with_name(f"{shared.stem}-lod{level}-{hashlib.sha256(lod_binary).hexdigest()[:8]}.bin"); lod_bin.write_bytes(lod_binary)
                models[f"upstream_lod_bin:{row['asset_id']}:{level}"] = {'url': '/data/rescue_assets/' + lod_bin.name, 'file': str(lod_bin)}
                lods.append((template, lod_bin.name, lod_record))
            source_assets['display_records'][row['asset_id']]['lod'] = [record for _, _, record in lods]
            source_assets[row['asset_id']] = (doc, shared.name, lods)
        # One source read/hash/parse per formal asset; wrappers remain independent.
        original, shared_name, lods = source_assets[row['asset_id']]
        doc = deepcopy(original)
        doc['buffers'][0]['uri'] = shared_name
        theta = math.radians(row['yaw_deg']); c, s = math.cos(theta), math.sin(theta)
        # (Rz(yaw) * Rx(90deg)) inverse.
        inverse = [[c, s, 0, 0], [0, 0, 1, 0], [s, -c, 0, 0], [0, 0, 0, 1]]
        published = [*row['enu'][:2], grid.height(*row['enu'][:2])]
        if eid in agent_ids:
            display = bundle['config']['display']
            if agent_kinds[eid] == 'uav' and grid.airspace_datum_u_m is not None:
                published[2] = grid.airspace_datum_u_m + display['uav_agl_m']
            else:
                published[2] += display['uav_agl_m'] if agent_kinds[eid] == 'uav' else display['agent_ground_offset_m']
        elif row['kind'] == 'human':
            published[2] += bundle['config']['display']['person_offset_m']
        matrix = deepcopy(row['matrix_enu'])
        if eid in agent_ids and agent_kinds[eid] == 'uav' and grid.airspace_datum_u_m is not None:
            # The open-water UAV flies at the configured datum-relative AGL.
            # Do not cancel that runtime altitude with its old staging wrapper.
            matrix[2][3] = published[2]
        for k in range(3):
            matrix[k][3] -= published[k]
        wrapper = multiply(inverse, multiply(matrix, cframe))
        scene = doc['scenes'][doc.get('scene', 0)]
        nodes = doc.setdefault('nodes', [])
        index = len(nodes)
        nodes.append({'name': 'upstream_exact_parent_transform', 'children': list(scene['nodes']),
                      'matrix': [wrapper[i][j] for j in range(4) for i in range(4)]})
        scene['nodes'] = [index]
        # Equal to 1e-12 (float noise only) -> one shared wrapper; the first instance's exact matrix is used.
        key = (row['asset_id'], json.dumps([[round(v, 12) for v in r] for r in wrapper]) + (':agent' if eid in agent_ids else ''))
        if key not in shared_bindings:
            stem = 'upstream-' + hashlib.sha256(key[0].encode() + key[1].encode()).hexdigest()[:16]
            declared = []
            for level, (template, lod_name, lod_record) in enumerate(lods, 1):
                lod_doc = deepcopy(template)
                lod_doc['buffers'][0]['uri'] = lod_name
                lod_doc['nodes'].append({'name': 'upstream_exact_parent_transform', 'children': [0], 'matrix': [wrapper[i][j] for j in range(4) for i in range(4)]})
                lod_doc['scenes'][0]['nodes'] = [len(lod_doc['nodes']) - 1]
                lod_body = json.dumps(lod_doc, separators=(',', ':')).encode(); lod_body += b' ' * ((-len(lod_body)) % 4)
                lod_bytes = struct.pack('<4sII', b'glTF', 2, 20 + len(lod_body)) + struct.pack('<II', len(lod_body), 0x4E4F534A) + lod_body
                lod_file = Path(directory) / 'assets' / f"{stem}-lod{level}-{hashlib.sha256(lod_bytes).hexdigest()[:8]}.glb"; lod_file.write_bytes(lod_bytes)
                models[f'{stem}:lod{level}'] = {'url': '/data/rescue_assets/' + lod_file.name, 'file': str(lod_file)}
                declared.append({'url': '/data/rescue_assets/' + lod_file.name, 'minDistance': lod_record['min_distance_m'], 'selfContained': True, 'triangles': lod_record['triangles']})
            if declared:
                doc.setdefault('asset', {}).setdefault('extras', {})['aasLod'] = declared
            target = Path(directory) / 'assets' / (stem + '.glb')
            body = json.dumps(doc, separators=(',', ':')).encode(); body += b' ' * ((-len(body)) % 4)
            target.write_bytes(struct.pack('<4sII', b'glTF', 2, 20 + len(body)) + struct.pack('<II', len(body), 0x4E4F534A) + body)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            final = target.with_name(target.stem + '-' + digest[:8] + '.glb'); target.replace(final)
            typ = 'rescue:Upstream_' + stem.split('-', 1)[1]
            snapshot['registry']['types'].append({'id': typ, 'parents': [AGENT if eid in agent_ids else BASE], 'abstract': False})
            snapshot['details']['effective_fields'][typ] = list(FIELDS if eid in agent_ids else FIELDS[:-1])
            models[typ] = {'url': '/data/rescue_assets/' + final.name, 'scale': 1, 'file': str(final)}
            presentation.append({'typeId': typ, 'positionField': POSITION, 'orientationField': ORIENTATION, 'frame': 'enu',
                                 'visual': {'kind': 'model', 'asset': models[typ]['url'], 'scale': 1}})
            shared_bindings[key] = (typ, digest)
        typ, digest = shared_bindings[key]
        mapping[eid] = typ
        asset_audit['assets'][eid] = {'asset_id': row['asset_id'], 'source': str(source), 'source_sha256': asset['sha256'],
                                     'display_source': source_assets.get('static_substitutions', {}).get(row['asset_id'], str(source)),
                                     'upstream_matrix_enu': row['matrix_enu'], 'wrapper_matrix_yup': wrapper,
                                     'generated_file_sha256': digest, 'display_runtime': source_assets['display_records'][row['asset_id']],
                                     'evidence': 'binding prepared; no render/task acceptance'}
    snapshot['details']['statistics']['types'] = len(snapshot['registry']['types'])
    return mapping, presentation
