"""Metric semantic adapter for seeded, source-derived background layouts.

The core uses positive XY metres; rendering is ENU. Terrain heights come from the
canonical grid; afloat layout poses use the source water datum. This reader keeps the v0.5 runtime
stdlib-only and accepts numeric, non-pickled NumPy .npy members inside .npz.
"""
from __future__ import annotations
import ast
import bisect
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct
import zipfile


def read_npy(raw):
    if raw[:6] != b'\x93NUMPY':
        raise ValueError('Not a NumPy array')
    major = raw[6]
    fmt = '<H' if major == 1 else '<I'
    size = struct.calcsize(fmt)
    length = struct.unpack_from(fmt, raw, 8)[0]
    header = ast.literal_eval(raw[8 + size:8 + size + length].decode('latin1'))
    if header['fortran_order']:
        raise ValueError('Fortran-order grids require explicit conversion')
    dtype = header['descr']
    endian, kind, width = dtype[0], dtype[1], int(dtype[2:])
    data = raw[8 + size + length:]
    shape = header['shape']
    count = math.prod(shape)
    if kind == 'U':
        stride = width * 4
        values = [data[i*stride:(i+1)*stride].decode('utf-32-le' if endian != '>' else 'utf-32-be').rstrip('\0') for i in range(count)]
    elif kind == 'S':
        values = [data[i*width:(i+1)*width].rstrip(b'\0').decode() for i in range(count)]
    else:
        formats = {('f',4):'f', ('f',8):'d', ('i',1):'b', ('i',2):'h', ('i',4):'i', ('i',8):'q',
                   ('u',1):'B', ('u',2):'H', ('u',4):'I', ('u',8):'Q'}
        code = formats[(kind, width)]
        values = list(struct.unpack(('>' if endian == '>' else '<') + str(count) + code, data))
    return shape, values


class Grid:
    def __new__(cls, path):
        # Only the opt-in adapter emits this descriptor. Ordinary NPZ readers
        # retain their original interpolation and output bytes.
        if cls is Grid and Path(path).name == 'upstream-grid.json':
            from .upstream_scene import UpstreamGrid
            return UpstreamGrid(path)
        return super().__new__(cls)

    def __init__(self, path):
        self.path = Path(path).resolve()
        with zipfile.ZipFile(self.path) as archive:
            arrays = {key: read_npy(archive.read(key + '.npy')) for key in ('E','N','U','cell_class','class_names')}
        self.e, self.n = arrays['E'][1], arrays['N'][1]
        self.u, self.classes = arrays['U'][1], arrays['cell_class'][1]
        self.names = arrays['class_names'][1]
        if arrays['U'][0] != (len(self.n),len(self.e)) or arrays['cell_class'][0] != (len(self.n)-1,len(self.e)-1):
            raise ValueError('Canonical grid shape mismatch')
        self.bounds = [0, 0, self.e[-1]-self.e[0], self.n[-1]-self.n[0]]

    def core(self, en):
        return [en[0]-self.e[0], en[1]-self.n[0]]

    def enu(self, xy, agl=0):
        e, n = xy[0]+self.e[0], xy[1]+self.n[0]
        return [e, n, self.height(e,n)+agl]

    def cell(self, e, n):
        if not self.e[0] <= e <= self.e[-1] or not self.n[0] <= n <= self.n[-1]:
            raise ValueError(f'Position outside canonical terrain: {e}, {n}')
        return min(len(self.e)-2,bisect.bisect_right(self.e,e)-1), min(len(self.n)-2,bisect.bisect_right(self.n,n)-1)

    def height(self, e, n):
        i,j = self.cell(e,n)
        x=(e-self.e[i])/(self.e[i+1]-self.e[i]); y=(n-self.n[j])/(self.n[j+1]-self.n[j])
        w=len(self.e)
        return (1-y)*((1-x)*self.u[j*w+i]+x*self.u[j*w+i+1])+y*((1-x)*self.u[(j+1)*w+i]+x*self.u[(j+1)*w+i+1])

    def classification(self,e,n):
        i,j=self.cell(e,n)
        return self.names[int(self.classes[j*(len(self.e)-1)+i])]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()).hexdigest()


def adapt(config):
    """Generate semantic geometry, truth inputs and a reproducible layout audit."""
    if 'upstream_generated_scene' in config['placement']:
        from .upstream_scene import adapt as adapt_upstream
        return adapt_upstream(config)
    from .layout import expand_rules, generate_layout

    bg = config['background']; grid = Grid(bg['grid'])
    report = json.loads(Path(bg['build_report']).read_text())
    listing = json.loads(Path(bg['scene_list']).read_text())
    entry = next(row for row in listing['scenes'] if row['sid'] == bg['id'])
    if report['scene_id'] != bg['id']:
        raise ValueError('Build report belongs to another background')
    generated = generate_layout(config, grid)
    layout = generated['entities']; entities = []

    def add(eid, cls, typ, points, attrs, authored=False):
        if typ == 'polygon':
            points = [*points, points[0]]
        entities.append(dict(id=eid, **{'class': cls},
            geometry=dict(type=typ, coords=[grid.core(p) for p in points]), attributes=attrs,
            label_source='authored' if authored else 'upstream'))

    for road in generated['roads']:
        attrs = {'road.width_m': road['width_m'], 'road.surface': road.get('surface', 'asphalt'),
                 'attr.source_road_id': road['source_id']}
        attrs.update({'attr.'+k: road[k] for k in
                      ('source_surface', 'allowed_mobilities', 'professional_only', 'source_pointer') if k in road})
        add(road['id'], 'road', 'polyline', road['points'], attrs, road.get('authored', False))
    if 'water' in generated:
        water = generated['water']
        add(water['id'], 'water', 'polygon', water['polygon_en'],
            {'attr.datum_U_m': water['datum_U_m'], 'attr.source_stage_id': water['id']})
    for region in generated['regions'].values():
        isolated = region['kind'] == 'metadata_only_cliff_side_professional_only'
        if region['kind'] in bg['plot_region_kinds'] or isolated:
            add(region['id'], 'open_area', 'polygon', region['polygon_en'],
                {'attr.role': 'candidate_building_plot', 'attr.reservation_kind': region['kind'],
                 **({'attr.ground_access_isolated': True} if isolated else {})})
    opts = {'cause': 'seeded_scene_layout', 'pinned_buildings': {}, 'pinned_occupants': {},
            'pinned_humans': {}, 'outdoor_count': 0, 'no_debris_from': [], 'obstacles': []}
    if 'public_brief_zh' in config:
        opts['public_brief_zh'] = config['public_brief_zh']
    damage_states = {'D0': ('none', 'stable'), 'D1': ('minor', 'compromised'),
                     'D2': ('major', 'unstable'), 'D3': ('destroyed', 'collapsed')}
    for eid, rule in generated['building_rules'].items():
        row = layout[eid]
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
        damage, structural = damage_states[rule['damage_grade']]
        opts['pinned_buildings'][eid] = {'building.visual_damage': damage, 'building.structural_condition': structural}
        opts['pinned_occupants'][eid] = len(rule['occupants'])
        opts['no_debris_from'].append(eid)
        for i, person in enumerate(rule['occupants'], 1):
            hid = f'human-{eid}-{i}'
            if person['location_context'] != 'interior':
                raise ValueError(f'{hid}: occupant rule requires location_context=interior of {eid}')
            opts['pinned_humans'][hid] = {f'human.{k}': v for k, v in person.items()}
    for rule in generated['debris_rules']:
        row = layout[rule['id']]; relation = row['relations']
        opts['obstacles'].append({'id': rule['id'], 'pose': grid.core(row['enu']),
            'road': relation['road'], 'source': relation['source'], 'volume_m3': rule['volume_m3'],
            'height_m': rule['height_m'], 'work': rule['work'], 'blocked_width_m': relation['blocked_width_m']})
    for rule in expand_rules(config['placement']['stations']):
        eid = rule['id']; row = layout[eid]
        attrs = {'station.kind': rule['kind'], 'station.capacity': rule['capacity'], 'station.power_W': rule['power_W'],
                 'attr.receiver_confirmation': rule['receiver_confirmation'], 'attr.plot_id': row['region']}
        if 'access' in row['relations']:
            attrs['attr.access_id'] = row['relations']['access']
        if 'water_access_polygon_en' in row['relations']:
            polygon = row['relations']['water_access_polygon_en']
            attrs['station.access_region'] = {'type': 'polygon', 'coords': [grid.core(p) for p in [*polygon, polygon[0]]]}
        add(eid, 'station', 'point', [row['enu'][:2]],
            attrs, True)
    for group in ('facilities', 'equipment'):
        for rule in expand_rules(config['placement'].get(group, [])):
            row = layout[rule['id']]
            attrs = {**rule['fields'], 'attr.plot_id': row['region'], 'entity.pose': grid.core(row['enu'])}
            if 'parent' in row['relations']:
                attrs['entity.parent_id'] = row['relations']['parent']
            add(rule['id'], 'facility', 'polygon', row['asset_footprint_en'], attrs, True)
    for rule in expand_rules(config['placement'].get('humans', [])):
        eid = rule['id']; row = layout[eid]
        fields = {**{f'human.{k}': v for k, v in rule['human'].items()}, **rule.get('fields', {})}
        # People are not an upstream semantic class: the core creates them in G02 at the generated position.
        opts.setdefault('placed_humans', {})[eid] = {'xy': grid.core(row['enu'][:2]), 'location_context': fields['human.location_context']}
        opts['pinned_humans'][eid] = fields
    for rule in config['placement'].get('hazards', []):
        row = layout[rule['geometry_from']]
        polygon = row['asset_footprint_en'] if rule['geometry_kind'] == 'asset_footprint' else row['footprint_en']
        opts.setdefault('pinned_hazards', {})[rule['id']] = {
            'geometry': {'type': 'polygon', 'coords': [grid.core(p) for p in [*polygon, polygon[0]]]},
            'fields': deepcopy(rule['fields'])}
    if 'station_stock' in config['placement']:
        opts['station_stock'] = deepcopy(config['placement']['station_stock'])
    doc = {'schema': 'rescue.upstream-semantics/v1',
        'source': {'kind': 'upstream_annotation', 'producer': 'rescue_scenes.backgrounds',
                   'terrain_ref': str(grid.path),
                   # The upstream schema allows no extra source keys; exact refs are also in audit.layout_sources.
                   'note': f"Tagged GLB reservations ({bg['aas_full_glb']}) and same-source road/access layout ({bg['layout_json']}); "
                           'disaster and deployment rules are authored.'},
        'frame': {'crs': 'local_ENU_m', 'bounds_m': grid.bounds}, 'entities': entities}
    slots = {eid: {'enu': row['enu'], 'plot': row['region'], 'role': row['kind'], **row['relations']}
             for eid, row in layout.items()}
    audit = {'background_entry': entry, 'slots': slots, 'layout': layout,
        'layout_network': {key: generated[key] for key in ('roads', 'regions', 'accesses')},
        'layout_sources': generated['sources'], 'layout_pairwise': generated['pairwise'],
        'transform': {'core_xy_offset': [-grid.e[0], -grid.n[0]], 'render': 'ENU -> [E,U,-N]',
                      'height': 'bilinear canonical grid', 'yaw': 'degrees, forward East=0, North=90'},
        'grid_sha256': hashlib.sha256(grid.path.read_bytes()).hexdigest(),
        'geometry_source': bg['geometry_source'], 'assumptions': config['assumptions']}
    if 'water' in generated:
        audit['layout_network']['water'] = generated['water']
        audit['transform']['height'] = 'bilinear canonical terrain; afloat layout poses use source water datum'
    if bg['id'] in ('SC07', 'SC12', 'SC13', 'SC17'):
        source = json.loads(Path(bg['layout_json']).read_text())
        audit['source_features'] = source['features']
    return grid, doc, opts, audit


def materialize(config, directory):
    from .core.agents.embodiment import EMBODIMENTS

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    grid, doc, opts, audit = adapt(config)
    if 'upstream_generated_scene' in config['placement']:
        config = deepcopy(config)
        config['background']['grid'] = audit['upstream']['runtime_grid']
        # scenario_id selects a core hazard profile, not an upstream terrain id.
        # In particular GT04's terrain is GT04_RURAL_CONTINUATION, not a profile.
    semantics = directory/'semantics.json'
    semantics.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
    agents = []
    for row in config['agents']:
        spec = deepcopy(row); spawn = spec.pop('spawn')
        generated = audit['layout'][spec['id']]
        spec['start'] = grid.core(generated['enu'])
        overrides = spec.setdefault('overrides', {})
        if overrides.get('mobility', EMBODIMENTS[spec['kind']]['mobility']) == 'wheeled':
            overrides['length_m'] = generated['dimensions_m'][0]
        if 'navigation_access_of' in spawn:
            target_id = spawn['navigation_access_of']
            if 'upstream' in audit:
                target_id = audit['upstream']['entity_aliases'].get(target_id, target_id)
            target = audit['layout'][target_id]['relations']['access']
            for goal in spec['overrides']['navigation_goals']:
                if goal['purpose'] == 'access':
                    goal['target'] = target
        agents.append(spec)
    core = {'scenario_id': config['scenario_id'], 'seed': config['seed'],
            # Resolve relative to bundle_path in the plugin. Keeping output
            # directory names out of the document makes bundle bytes repeatable.
            'semantics': {'source': 'upstream', 'path': 'semantics.json'},
            'bounds_m': grid.bounds, 'agents': agents, 'tick_s': config['tick_s'],
            'until_s': config['until_s'], 'truth_options': opts}
    bundle = {'config': config, 'core': core, 'audit': audit, 'config_digest': digest(config)}
    path = directory/'bundle.json'
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding='utf-8')
    return path, bundle
