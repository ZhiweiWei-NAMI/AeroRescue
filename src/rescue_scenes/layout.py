"""Seeded placement on tagged background regions and the source road network.

Only GLB reservation extras are read; mesh geometry is never reinterpreted.
All coordinates here are EN metres, and yaw is forward from East towards North.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import struct

EPS = 1e-7
from .paths import ROOT  # data/code root is relocatable (RESCUE_SCENE_ROOT)


def edges(poly):
    return zip(poly, poly[1:] + poly[:1])


def project(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0., min(1., ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return [a[0]+t*dx, a[1]+t*dy], t


def intersection(a, b, c, d):
    ux, uy, vx, vy = b[0]-a[0], b[1]-a[1], d[0]-c[0], d[1]-c[1]
    det = ux*vy-uy*vx
    if abs(det) < EPS:
        return None
    wx, wy = c[0]-a[0], c[1]-a[1]
    t, u = (wx*vy-wy*vx)/det, (wx*uy-wy*ux)/det
    if -EPS <= t <= 1+EPS and -EPS <= u <= 1+EPS:
        return [a[0]+t*ux, a[1]+t*uy]
    return None


def inside(p, poly):
    if any(math.dist(p, project(p, a, b)[0]) < EPS for a, b in edges(poly)):
        return True
    result = False
    for a, b in edges(poly):
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:
            result = not result
    return result


def polygon_gap(a, b):
    """Zero for touching/intersecting polygons, otherwise minimum edge gap."""
    ax, ay = zip(*a); bx, by = zip(*b)
    box_gap = math.hypot(max(min(ax)-max(bx), min(bx)-max(ax), 0),
                         max(min(ay)-max(by), min(by)-max(ay), 0))
    if box_gap < EPS:
        if inside(a[0], b) or inside(b[0], a):
            return 0.
        if any(intersection(p, q, r, s) is not None for p, q in edges(a) for r, s in edges(b)):
            return 0.
    return min([math.dist(p, project(p, r, s)[0]) for p in a for r, s in edges(b)] +
               [math.dist(p, project(p, r, s)[0]) for p in b for r, s in edges(a)])


def rectangle(center, length, width, yaw):
    angle = math.radians(yaw); c, s = math.cos(angle), math.sin(angle)
    return [[center[0]+c*x-s*y, center[1]+s*x+c*y]
            for x, y in [(-length/2, -width/2), (length/2, -width/2),
                         (length/2, width/2), (-length/2, width/2)]]


def road_polygon(road, extra=0.):
    a, b = road['points']
    return rectangle([(a[0]+b[0])/2, (a[1]+b[1])/2], math.dist(a, b),
                     road['width_m']+2*extra, math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))


def footprint_samples(poly, spacing):
    """Rectangular footprint boundary AND interior, at most spacing metres."""
    a, b, _, d = poly
    ni = max(1, math.ceil(math.dist(a, b)/spacing))
    nj = max(1, math.ceil(math.dist(a, d)/spacing))
    for i in range(ni+1):
        for j in range(nj+1):
            yield [a[k]+i/ni*(b[k]-a[k])+j/nj*(d[k]-a[k]) for k in (0, 1)]


def footprint_classes(grid, poly, spacing):
    classes = {grid.classification(*p) for p in footprint_samples(poly, spacing)}
    # Include narrow breakline cells touched between samples (e.g. 2 cm curbs).
    xs, ys = zip(*poly)
    i0, j0 = grid.cell(min(xs), min(ys)); i1, j1 = grid.cell(max(xs), max(ys))
    stride = len(grid.e)-1
    for j in range(j0, j1+1):
        for i in range(i0, i1+1):
            name = grid.names[int(grid.classes[j*stride+i])]
            if name in classes:
                continue
            cell = [[grid.e[i], grid.n[j]], [grid.e[i+1], grid.n[j]],
                    [grid.e[i+1], grid.n[j+1]], [grid.e[i], grid.n[j+1]]]
            if polygon_gap(poly, cell) < EPS:
                classes.add(name)
    return sorted(classes)


def read_reservations(path):
    with Path(path).open('rb') as handle:
        magic, version, _ = struct.unpack('<4sII', handle.read(12))
        length, kind = struct.unpack('<II', handle.read(8))
        if magic != b'glTF' or version != 2 or kind != 0x4E4F534A:
            raise ValueError(f'Background reservations require GLB 2 JSON: {path}')
        raw = handle.read(length)
    doc = json.loads(raw)
    regions = {}
    for node in doc['nodes']:
        extra = node.get('extras', {})
        if extra.get('role') != 'METADATA_RESERVATION':
            continue
        value = extra['polygon_en_m']
        polygon = json.loads(value) if isinstance(value, str) else value
        if polygon[0] == polygon[-1]:
            polygon = polygon[:-1]
        regions[node['name']] = {'id': node['name'], 'kind': extra['kind'],
                                 'polygon_en': polygon, 'true_z0_m': extra['true_z0_m']}
    return regions, hashlib.sha256(raw).hexdigest()


def source_pad_walks(layout, regions):
    """Realize declared pedestrian graph joins inside their source-graded pads.

    A road ending on a pad is not geometrically incident to a trail starting at
    its far edge. Only an explicit source graph AND the matching graded source
    rectangle authorize this traversal; proximity or a reservation alone do not.
    These are logical foot paths on existing ground, not new physical surfaces.
    """
    authored = layout.get('features', {}).get('authored_geometry', {})
    graph = authored.get('pedestrian_network', {}).get('graph', [])
    neighbors = {}
    for chain in graph:
        if not isinstance(chain, str):
            continue
        nodes = chain.split('--')
        for a, b in zip(nodes, nodes[1:]):
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    walks = []
    for pad, fields in authored.get('reservations_and_entry', {}).items():
        rect = fields.get('rectangle_en')
        if pad not in neighbors or not isinstance(rect, list) or len(rect) != 4:
            continue
        xmin, ymin, xmax, ymax = rect
        polygon = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        if not any(region['kind'] in {'metadata_only_assembly_reservation', 'metadata_only_camp_reservation'}
                   and {tuple(p) for p in region['polygon_en']} == {tuple(p) for p in polygon}
                   for region in regions.values()):
            continue
        landings = []
        for road in layout['roads']:
            node = road['id']
            pointer = road.get('source_pointer', '')
            fragment = pointer.partition('#/pedestrian_network/')[2]
            if fragment:
                node = fragment.split('/')[0]
            if node not in neighbors[pad]:
                continue
            line = road.get('centerline_en_m', road.get('centerline_enu_m', []))
            for point in (line[:1] + line[-1:]):
                if xmin <= point[0] <= xmax and ymin <= point[1] <= ymax:
                    landings.append((road, list(point[:2])))
        for index, (left, a) in enumerate(landings):
            for right, b in landings[index + 1:]:
                if left['id'] == right['id'] or math.dist(a, b) < EPS:
                    continue
                allowed = [m for m in ('foot', 'legged')
                           if m in left.get('allowed_mobilities', ()) and m in right.get('allowed_mobilities', ())]
                if not allowed:
                    continue
                walks.append({'source_id': f"{pad}:walk:{left['id']}:{right['id']}",
                    'points': [a, b],
                    'width_m': min(left.get('carriageway_width_m', left.get('width_m')),
                                   right.get('carriageway_width_m', right.get('width_m')),
                                   xmax-xmin, ymax-ymin),
                    'sidewalk_m': 0., 'is_access': True, 'surface': 'graded_earth',
                    'allowed_mobilities': allowed,
                    'professional_only': bool(left.get('professional_only') or right.get('professional_only')),
                    'source_surface': pad,
                    'source_pointer': left.get('source_pointer', '').partition('#')[0] + '#/pedestrian_network/graph',
                    'upstream_road_edges': []})
    return walks


def source_plot_bindings(source, regions):
    """Bind locked source plots by their whole polygons, never identifier spelling."""
    def ring_key(polygon):
        points = [tuple(p[:2]) for p in polygon]
        if points[0] == points[-1]:
            points.pop()
        variants = [tuple(ring[i:] + ring[:i]) for ring in (points, points[::-1]) for i in range(len(ring))]
        return min(variants)

    bindings = {}
    for plot in source['placement_regions']:
        e0, e1 = plot['bounds_E']; n0, n1 = plot['bounds_N']
        key = ring_key([[e0, n0], [e1, n0], [e1, n1], [e0, n1]])
        matches = [rid for rid, region in regions.items() if region['kind'] == 'dry_placement'
                   and ring_key(region['polygon_en']) == key]
        if len(matches) == 1:
            bindings[plot['id']] = matches[0]
    return bindings


def source_ribbon_polygon(line, width):
    """The published source/common.py ribbon's full XY boundary."""
    left, right = [], []
    for index, point in enumerate(line):
        a, b = line[max(0, index - 1)], line[min(len(line) - 1, index + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dx, dy)
        nx, ny = -dy / norm * width / 2, dx / norm * width / 2
        left.append([point[0] + nx, point[1] + ny])
        right.append([point[0] - nx, point[1] - ny])
    return left + right[::-1]


def source_network(layout, regions, approach_depth, unconnected_accesses=None, *, plot_bindings=None):
    """Read roads/accesses, then split at every junction and access landing."""
    originals = {r['id']: r for r in layout['roads']}
    def centerline(row):
        key = 'centerline_en_m' if 'centerline_en_m' in row else 'centerline_enu_m'
        return [p[:2] for p in row[key]]

    raw = []
    for road in originals.values():
        line = centerline(road)
        for index, (a, b) in enumerate(zip(line, line[1:])):
            raw.append({'source_id': road['id'], 'points': [a, b],
                        **({'upstream_road_edges': road['upstream_edges_by_segment'][index]}
                           if 'upstream_edges_by_segment' in road else {}),
                        # GT04 publishes the measured carriageway as width_m;
                        # other Route-A sidecars name it carriageway_width_m.
                        'width_m': road['carriageway_width_m'] if 'carriageway_width_m' in road else road['width_m'],
                        'sidewalk_m': road['sidewalk_each_m'] if 'sidewalk_each_m' in road else 0.,
                        'is_access': False, **{k: road[k] for k in
                            ('surface', 'source_surface', 'allowed_mobilities', 'professional_only', 'source_pointer') if k in road}})
    raw.extend(source_pad_walks(layout, regions))
    accesses = {}
    plot_bindings = layout.get('plot_reservation_bindings', {}) if plot_bindings is None else plot_bindings
    for access in layout['accesses']:
        plot_id = next((ref if ref in regions else plot_bindings[ref]
                        for ref in access['connects'] if ref in regions or ref in plot_bindings), None)
        rid = next((ref for ref in access['connects'] if ref in originals), None)
        if plot_id is None or rid is None:
            # A missing/ambiguous source polygon binding remains a real gap;
            # neither identifier similarity nor proximity authorizes a link.
            if unconnected_accesses is not None:
                unconnected_accesses.append({'access_id': access['id'], 'connects': list(access['connects']),
                    'missing': 'loaded source reservation' if plot_id is None else 'source road',
                    'effect': 'omit unconnected plot access; no identifier alias or connector invented'})
            continue
        if 'points_en_m' in access:
            # GT04 publishes an actual curved track, in plot->road order. Keep
            # every point; the public access API is road->plot. Its landing is
            # allowed only on the explicitly connected road's physical ribbon.
            authored = [list(p[:2]) for p in access['points_en_m']]
            road_line = centerline(originals[rid])
            gap, endpoint, landing = min((math.dist(point, q), endpoint, q)
                for endpoint in (0, -1) for point in [authored[endpoint]]
                for q in [min((project(point, a, b)[0] for a, b in zip(road_line, road_line[1:])),
                              key=lambda p: math.dist(point, p))])
            points = authored if endpoint == 0 else list(reversed(authored))
            road_width = originals[rid].get('carriageway_width_m', originals[rid].get('width_m'))
            road_support = source_ribbon_polygon(road_line, road_width)
            # Source roads are resampled/rounded independently of access ends
            # (~9/21 mm in plots 04/05). Traverse only their existing overlap,
            # never snap across a missing surface or to a different source road.
            if gap > road_width / 2 + EPS or not inside(points[0], road_support) or not inside(points[-1], regions[plot_id]['polygon_en']):
                if unconnected_accesses is not None:
                    unconnected_accesses.append({'access_id': access['id'], 'connects': list(access['connects']),
                        'missing': 'declared road overlap or source plot endpoint',
                        'effect': 'omit source track without a physically supported endpoint binding'})
                continue
            poly = source_ribbon_polygon(points, access['width_m'])
            points = [landing] + (points if gap > EPS else points[1:])
            end, previous = points[-1], points[-2]
            norm = math.dist(previous, end)
            direction = [(end[k] - previous[k]) / norm for k in (0, 1)]
            yaw = math.degrees(math.atan2(direction[1], direction[0]))
            corridor_center = [end[k] + direction[k] * approach_depth / 2 for k in (0, 1)]
            accesses[access['id']] = {'id': access['id'], 'plot': plot_id, 'road_source': rid,
                'points': points, 'source_points_en_m': authored, 'width_m': access['width_m'], 'polygon_en': poly,
                'approach_polygon_en': rectangle(corridor_center, approach_depth, access['width_m'], yaw),
                'road_attachment': {'kind': 'declared_road_physical_overlap', 'distance_m': gap, 'road_source': rid}}
            for a, b in zip(points, points[1:]):
                if math.dist(a, b) < EPS:
                    continue
                raw.append({'source_id': access['id'], 'points': [a, b],
                    'width_m': access['width_m'], 'sidewalk_m': 0., 'is_access': True,
                    **{k: access.get(k, originals[rid].get(k)) for k in
                       ('surface', 'source_surface', 'allowed_mobilities', 'professional_only', 'source_pointer')
                       if k in access or k in originals[rid]}})
            continue
        if 'polygon_en_m' in access:
            poly = access['polygon_en_m']
        else:
            # SC03 annotates the same spur as a metric centreline and width.
            a, b = centerline(access)[0], centerline(access)[-1]
            poly = rectangle([(a[k]+b[k])/2 for k in (0, 1)], math.dist(a, b),
                             access['width_m'], math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])))
        if poly[0] == poly[-1]:
            poly = poly[:-1]
        center = [sum(p[k] for p in poly)/len(poly) for k in (0, 1)]
        line = centerline(originals[rid])
        landing = min((project(center, a, b)[0] for a, b in zip(line, line[1:])),
                      key=lambda p: math.dist(center, p))
        norm = math.dist(center, landing)
        direction = [(center[k]-landing[k])/norm for k in (0, 1)]
        extent = max(sum((p[k]-landing[k])*direction[k] for k in (0, 1)) for p in poly)
        end = [landing[k]+extent*direction[k] for k in (0, 1)]
        yaw = math.degrees(math.atan2(direction[1], direction[0]))
        corridor_center = [end[k]+direction[k]*approach_depth/2 for k in (0, 1)]
        accesses[access['id']] = {'id': access['id'], 'plot': plot_id, 'road_source': rid,
            'points': [landing, end], 'width_m': access['width_m'], 'polygon_en': poly,
            'approach_polygon_en': rectangle(corridor_center, approach_depth, access['width_m'], yaw)}
        raw.append({'source_id': access['id'], 'points': [landing, end],
                    'width_m': access['width_m'], 'sidewalk_m': 0., 'is_access': True,
                    **{k: access.get(k, originals[rid].get(k)) for k in
                       ('surface', 'source_surface', 'allowed_mobilities', 'professional_only', 'source_pointer')
                       if k in access or k in originals[rid]}})
    cuts = [[list(p) for p in r['points']] for r in raw]
    for i, road in enumerate(raw):
        a, b = road['points']
        for j in range(i):
            c, d = raw[j]['points']
            p = intersection(a, b, c, d)
            if p is not None:
                cuts[i].append(p); cuts[j].append(p)
            else:
                for p in (c, d):
                    if math.dist(p, project(p, a, b)[0]) < EPS:
                        cuts[i].append(p)
                for p in (a, b):
                    if math.dist(p, project(p, c, d)[0]) < EPS:
                        cuts[j].append(p)
    counts = {}; roads = []
    for road, points in zip(raw, cuts):
        a, b = road['points']
        points = sorted({tuple(round(v, 7) for v in p) for p in points},
                        key=lambda p: project(p, a, b)[1])
        for p, q in zip(points, points[1:]):
            if math.dist(p, q) < EPS:
                continue
            sid = road['source_id']; counts[sid] = counts.get(sid, 0)+1
            # Preserve only this raw segment's upstream edges on each junction
            # piece. The adapter must not rematch the whole source road later.
            roads.append({**road, 'id': f'{sid}#{counts[sid]}', 'points': [list(p), list(q)]})
    # The terminal access segment keeps the source id as a public navigation
    # target; any preceding pieces still have deterministic junction ids.
    for road in roads:
        if road['is_access'] and road['source_id'] in accesses:
            end = accesses[road['source_id']]['points'][-1]
            if any(math.dist(p, end) < EPS for p in road['points']):
                road['id'] = road['source_id']
        road['footprint_en'] = road_polygon(road)
    return roads, accesses


def source_water(layout, rule):
    """Use the published water extent, never a synthetic shore or ground pad."""
    if layout['scene_id'] == 'SC05':
        source = layout['features']['waterborne_layout']
        state = source['water_state']
        if rule['stage_id'] != state['primary_state']:
            raise ValueError(f"SC05: water stage {rule['stage_id']} differs from the source state")
        return {'id': state['primary_state'], 'datum_U_m': state['datum_u_m'],
                'polygon_en': source['footprint']['polygon_en_m'], 'banks_en': []}
    # SC03's two directed banks meet at the north/south scene boundaries.
    stage = next(row for row in layout['water'] if row['id'] == rule['stage_id'])
    banks = [[p[:2] for p in stage['waterline_polylines_enu_m'][i]] for i in rule['bank_indices']]
    south, north = layout['bounds_enu_m']['N']
    if (len(banks) != 2 or abs(banks[0][0][1]-south) > EPS or abs(banks[0][-1][1]-north) > EPS
            or abs(banks[1][0][1]-north) > EPS or abs(banks[1][-1][1]-south) > EPS):
        raise ValueError(f"{stage['id']}: configured banks must run south->north and north->south at scene boundaries")
    return {'id': stage['id'], 'datum_U_m': stage['datum_U_m'],
            'polygon_en': banks[0]+banks[1], 'banks_en': banks}


def graph_for(roads):
    graph = {}
    for road in roads:
        a, b = (tuple(p) for p in road['points'])
        length = math.dist(a, b)
        graph.setdefault(a, []).append((b, length, road['id']))
        graph.setdefault(b, []).append((a, length, road['id']))
    return graph


def shortest_route(graph, start, goal):
    queue = [(0., start)]; distance = {start: 0.}; previous = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        if node == goal:
            route = []
            while node != start:
                node, rid = previous[node]; route.append(rid)
            return list(reversed(route)), cost
        for neighbor, length, rid in graph[node]:
            value = cost+length
            if neighbor not in distance or value < distance[neighbor]-EPS:
                distance[neighbor] = value; previous[neighbor] = (node, rid)
                heapq.heappush(queue, (value, neighbor))
    return None


def expand_rules(rules):
    for rule in rules:
        for i in range(rule['count']):
            yield {**rule, 'id': rule['id'] if rule['count'] == 1 else f"{rule['id']}-{i+1}"}


class Layout:
    def __init__(self, config, grid):
        self.config, self.grid = config, grid
        self.rules = config['placement']; self.rng = random.Random(config['seed'])
        self.spacing = self.rules['sample_spacing_m']
        if not 0 < self.spacing <= 1:
            raise ValueError('placement.sample_spacing_m must be in (0, 1] metres')
        bg = config['background']
        self.regions, metadata_hash = read_reservations(bg['aas_full_glb'])
        raw = Path(bg['layout_json']).read_bytes(); source = json.loads(raw)
        self.source = source
        if source['scene_id'] != bg['id']:
            raise ValueError(f"{bg['id']}: layout_json belongs to {source['scene_id']}")
        unconnected_accesses = []
        plot_bindings, plot_source_raw = {}, None
        if bg.get('plot_reservation_source'):
            plot_source_raw = Path(bg['plot_reservation_source']).read_bytes()
            plot_bindings = source_plot_bindings(json.loads(plot_source_raw), self.regions)
            # The upstream adapter copies this source and calls source_network
            # again after appending delivered walks/approaches. Keep the measured
            # binding through that existing rebuild without renaming connects.
            source['plot_reservation_bindings'] = plot_bindings
        self.roads, self.accesses = source_network(source, self.regions, self.rules['approach_depth_m'],
                                                  unconnected_accesses, plot_bindings=plot_bindings)
        self.water = source_water(source, self.rules['water']) if 'water' in self.rules else None
        self.by_road = {r['id']: r for r in self.roads}
        self.graph = graph_for(self.roads)
        self.keepouts = [(r['id'], r['polygon_en']) for r in self.regions.values()
                         if r['kind'] in bg['keepout_region_kinds']]
        self.access_keepouts = []
        for access in self.accesses.values():
            self.access_keepouts.extend([(access['id'], access['polygon_en']),
                                         (access['id']+':approach', access['approach_polygon_en'])])
        asset_path = Path(self.rules['asset_map'])
        if not asset_path.is_absolute():
            asset_path = ROOT/asset_path
        asset_raw = asset_path.read_bytes(); self.assets = json.loads(asset_raw)
        self.order = self.assets['sizeM_order']
        production = asset_path.with_name('production_map.json')  # produced assets, same record format
        if production.is_file():
            extra = json.loads(production.read_text())
            assert extra['sizeM_order'] == self.order
            self.assets['kinds'].update(extra['kinds']); self.assets['assets'].update(extra['assets'])
        self.records = {}; self.models = {}
        self.sources = {'aas_full_glb': bg['aas_full_glb'], 'layout_json': bg['layout_json'],
                        'glb_json_sha256': metadata_hash, 'layout_sha256': hashlib.sha256(raw).hexdigest(),
                        'asset_map': str(asset_path), 'asset_map_sha256': hashlib.sha256(asset_raw).hexdigest()}
        if plot_source_raw is not None:
            self.sources['plot_reservation_binding'] = {'source': bg['plot_reservation_source'],
                'sha256': hashlib.sha256(plot_source_raw).hexdigest(), 'bindings': plot_bindings,
                'method': 'unique exact whole source-plot polygon match to loaded dry_placement reservation'}
        if unconnected_accesses:
            self.sources['unconnected_accesses'] = unconnected_accesses

    def model(self, rule):
        key = (rule['asset_kind'], tuple(rule.get('reserve_damage_grades', [])), rule.get('damage_grade'),
               tuple(rule.get('reserve_states', [])))
        if key in self.models:
            return self.models[key]
        base = self.assets['kinds'][rule['asset_kind']]
        selected = self.assets['assets'][base['states'][rule['damage_grade']]] if 'damage_grade' in rule else base
        states = rule.get('reserve_damage_grades', []) + rule.get('reserve_states', [])
        variants = {state: self.assets['assets'][base['states'][state]] for state in states}
        reference = self.assets['assets'][base['states']['D0']] if rule.get('reserve_damage_grades') else base
        origin = reference.get('origin_m', self.config.get('display', {}).get('model_origins', {}).get(rule['asset_kind']))
        center = [origin[i] if origin is not None else (reference['boundsM']['min'][i]+reference['boundsM']['max'][i])/2 for i in (0, 2)]
        turn = {'+x': 0., '-x': math.pi, '+z': math.pi/2, '-z': -math.pi/2}[reference['forward']]
        c, s = math.cos(turn), math.sin(turn)

        def local(row):
            lo, hi = row['boundsM']['min'], row['boundsM']['max']
            result = []
            for x, z in [(lo[0], lo[2]), (hi[0], lo[2]), (hi[0], hi[2]), (lo[0], hi[2])]:
                x -= center[0]; z -= center[1]
                result.append([c*x+s*z, s*x-c*z])
            # Reorder as rectangle a,b,c,d in normalized forward/lateral axes.
            xs, ys = zip(*result)
            return [[min(xs), min(ys)], [max(xs), min(ys)], [max(xs), max(ys)], [min(xs), max(ys)]]

        variant_bounds = {grade: local(row) for grade, row in variants.items()}
        selected_bounds = local(selected)
        all_points = selected_bounds + [p for poly in variant_bounds.values() for p in poly]
        # A symmetric envelope keeps core polygon centroid == shared D0 display
        # origin, while covering asymmetric D2/D3 spread without model scaling.
        length, width = [2*max(abs(p[i]) for p in all_points) for i in (0, 1)]
        dimensions = [selected['sizeM'][self.order.index(name)] for name in
                      ('length_along_forward', 'width_horizontal_perpendicular', 'height_Y')]
        result = {'asset_id': selected['asset_id'], 'asset_kind': rule['asset_kind'],
                  'asset_forward': selected['forward'], 'reference_asset_id': reference['asset_id'],
                  'dimensions_m': dimensions, 'footprint_size_m': [length, width],
                  'scale': [1., 1., 1.], 'local_asset_polygon': selected_bounds,
                  'local_variants': variant_bounds}
        self.models[key] = result
        return result

    def access_for(self, region):
        return [a for a in self.accesses.values() if a['plot'] == region]

    def yaw(self, rule, center, region):
        rotation = rule['rotation']; choices = rotation['choices_deg']
        if rotation['policy'] == 'seeded_choice':
            return self.rng.choice(choices)
        if rotation['policy'] == 'face_access':
            accesses = self.access_for(region)
            if not accesses:
                raise ValueError(f"{rule['id']}: face_access requires an access for region {region}")
            target = min((a['points'][-1] for a in accesses), key=lambda p: math.dist(center, p))
        elif rotation['policy'] == 'face_nearest_road':
            target = min((project(center, *r['points'])[0] for r in self.roads if not r['is_access']),
                         key=lambda p: math.dist(center, p))
        else:
            raise ValueError(f"{rule['id']}: unsupported rotation rule {rotation['policy']}")
        angle = math.degrees(math.atan2(target[1]-center[1], target[0]-center[0]))
        delta = lambda yaw: abs((yaw-angle+180) % 360-180)
        best = min(delta(yaw) for yaw in choices)
        return self.rng.choice([yaw for yaw in choices if delta(yaw) <= best+EPS])

    def record(self, rule, center, yaw, region, kind, relations):
        model = self.model(rule); angle = math.radians(yaw); c, s = math.cos(angle), math.sin(angle)
        def world(poly):
            return [[center[0]+c*x-s*y, center[1]+s*x+c*y] for x, y in poly]
        row = {'kind': kind, **{k: v for k, v in model.items() if not k.startswith('local_')},
                'enu': [*center, self.grid.height(*center)], 'yaw_deg': yaw % 360,
                'footprint_en': rectangle(center, *model['footprint_size_m'], yaw),
                'asset_footprint_en': world(model['local_asset_polygon']),
                'variant_footprints_en': {grade: world(poly) for grade, poly in model['local_variants'].items()},
                'region': region, 'classes': [], 'clearance_m': rule['clearance_m'], 'relations': relations}
        if rule.get('support_footprint_only'):
            anchor = self.source['anchors'][rule['source_anchor']]
            # The source records the actual support in the asset's X/Z plane.
            turn = {'+x': 0., '-x': math.pi, '+z': math.pi/2, '-z': -math.pi/2}[model['asset_forward']]
            ct, st = math.cos(turn), math.sin(turn)
            points = [[ct*x+st*z, st*x-ct*z] for x, z in anchor['support_only_footprint']]
            xs, ys = zip(*points)
            row['support_footprint_en'] = world([[min(xs), min(ys)], [max(xs), min(ys)],
                                                [max(xs), max(ys)], [min(xs), max(ys)]])
        return row

    def check(self, rule, row, in_region=True, roadway=False):
        poly = row['footprint_en']; grid = self.grid
        if any(not (grid.e[0] <= p[0] <= grid.e[-1] and grid.n[0] <= p[1] <= grid.n[-1]) for p in poly):
            return 'whole footprint inside canonical grid bounds'
        support = row.get('support_footprint_en', poly)
        samples = list(footprint_samples(support, self.spacing))
        if in_region and row['region'] in self.regions:
            region = self.regions[row['region']]['polygon_en']
            if any(not inside(p, region) for p in samples):
                return f"whole footprint inside reservation {row['region']}"
        if not roadway:
            keepouts = self.keepouts + (self.access_keepouts if rule['keep_access_clear'] else [])
            for eid, keepout in keepouts:
                if polygon_gap(poly, keepout) <= rule['clearance_m']+EPS:
                    return f'access/keep-out clearance from {eid}'
            for road in self.roads:
                if road['is_access']:
                    continue
                extra = rule['road_clearance_m']+(road['sidewalk_m'] if rule['exclude_sidewalk'] else 0.)
                if polygon_gap(poly, road_polygon(road, extra)) < EPS:
                    return f"off-carriageway road clearance from {road['id']}"
        for eid, other in self.records.items():
            if row['relations'].get('parent') == eid:
                continue
            gap = max(rule['clearance_m'], other['clearance_m'])
            if polygon_gap(poly, other['footprint_en']) <= gap+EPS:
                return f'footprint separation {gap} m from {eid}'
            connector = other['relations'].get('approach_connector_en')
            if connector and polygon_gap(poly, connector) <= gap+EPS:
                return f'keep the access approach to {eid} clear'
            connector = row['relations'].get('approach_connector_en')
            if connector and polygon_gap(connector, other['footprint_en']) <= gap+EPS:
                return f'access approach must avoid footprint {eid}'
        classes = footprint_classes(grid, support, self.spacing)
        if not set(classes) <= set(rule['allowed_classes']):
            return f"whole footprint classes {classes} within {rule['allowed_classes']}"
        if self.water is not None and 'dry_freeboard_m' in rule:
            if any(grid.height(*p) < self.water['datum_U_m']+rule['dry_freeboard_m'] for p in samples):
                return 'whole footprint on dry ground above the source water datum'
        row['classes'] = classes
        nearest = min(((polygon_gap(poly, r['footprint_en']), r['id'])
                       for r in self.roads if not r['is_access']), default=(None, None))
        row['nearest_road_clearance_m'], row['nearest_road'] = nearest
        if rule.get('require_nearest_access'):
            nearest_id = min(self.roads, key=lambda r: math.dist(row['enu'][:2], project(row['enu'], *r['points'])[0]))['source_id']
            if nearest_id not in [a['id'] for a in self.access_for(row['region'])]:
                return 'nearest public-map road is an access of its own plot'
        return None

    def candidate(self, rule, kind):
        if rule.get('mode') in ('in_water', 'on_water_near_shore'):
            return self.water_candidate(rule, kind)
        if rule.get('mode') == 'shore_landing':
            return self.shore_candidate(rule)
        pools = [r for r in self.regions.values() if r['kind'] in rule['allowed_region_kinds']
                 and ('allowed_region_ids' not in rule or r['id'] in rule['allowed_region_ids'])]
        if rule.get('mode') == 'source_anchor':
            anchor = self.source['anchors'][rule['source_anchor']]
            region = anchor['region']
            if region not in {r['id'] for r in pools}:
                return None, 'source anchor belongs to an allowed reservation'
            relations = {'source_anchor': rule['source_anchor']}
            if 'ground_normal_enu' in anchor:
                relations['ground_normal_enu'] = anchor['ground_normal_enu']
            if 'parent' in rule:
                relations.update(parent=rule['parent'], local_offset_m=rule['local_offset_m'])
            row = self.record(rule, anchor['en_m'], anchor['yaw_deg'], region, kind, relations)
            reason = self.check(rule, row)
            return (None, reason) if reason else (row, None)
        if 'different_plot_from' in rule:
            other = self.records[rule['different_plot_from']]['region']
            pools = [r for r in pools if r['id'] != other]
        if 'near' in rule:
            anchor = self.records[rule['near']]['enu']
            low, high = rule['distance_m']; angle = self.rng.uniform(0, math.tau)
            radius = math.sqrt(self.rng.uniform(low*low, high*high))
            center = [anchor[0]+radius*math.cos(angle), anchor[1]+radius*math.sin(angle)]
            matches = [r for r in pools if inside(center, r['polygon_en'])]
            if not matches and not rule['allow_unreserved_surface']:
                return None, 'nearby center must belong to an allowed reservation'
            region = self.rng.choice(matches)['id'] if matches else 'unreserved_surface'
        else:
            if not pools:
                return None, f"no reservation of kinds {rule['allowed_region_kinds']} satisfying plot relations"
            selected = self.rng.choice(pools); region = selected['id']; xs, ys = zip(*selected['polygon_en'])
            center = [self.rng.uniform(min(xs), max(xs)), self.rng.uniform(min(ys), max(ys))]
        # Grid bounds are checked before the height lookup in record().
        if not (self.grid.e[0] <= center[0] <= self.grid.e[-1] and self.grid.n[0] <= center[1] <= self.grid.n[-1]):
            return None, 'sampled center inside canonical bounds'
        relations = {'near': rule['near']} if 'near' in rule else {}
        if region in self.regions and self.access_for(region):
            relations['access'] = min(self.access_for(region), key=lambda a: math.dist(center, a['points'][-1]))['id']
        if 'different_plot_from' in rule:
            relations['different_plot_from'] = rule['different_plot_from']
        row = self.record(rule, center, self.yaw(rule, center, region), region, kind, relations)
        if kind in ('building', 'station') and 'access' in relations:
            access = self.accesses[relations['access']]; start = access['points'][-1]
            middle = [(start[k]+center[k])/2 for k in (0, 1)]
            angle = math.degrees(math.atan2(center[1]-start[1], center[0]-start[0]))
            relations['approach_connector_en'] = rectangle(middle, math.dist(start, center), access['width_m'], angle)
        reason = self.check(rule, row)
        if reason is None and kind == 'building':
            # The placement envelope reserves all requested damage variants;
            # the entrance follows the envelope of the displayed asset state.
            hits = [p for a, b in edges(row['asset_footprint_en'])
                    if (p := intersection(start, center, a, b)) is not None]
            relations['entrance_en'] = min(hits, key=lambda p: math.dist(start, p))
            relations['entrance_heading_deg'] = angle % 360
            relations['approach_start_en'] = list(start)
        return (None, reason) if reason else (row, None)

    def check_water(self, rule, row):
        samples = list(footprint_samples(row['footprint_en'], self.spacing))
        if any(not inside(p, self.water['polygon_en']) for p in samples):
            return 'whole footprint inside the source water polygon'
        depths = [self.water['datum_U_m']-self.grid.height(*p) for p in samples]
        low, high = rule['water_depth_m']
        if min(depths) < low or max(depths) > high:
            return f'whole-footprint water depths [{min(depths):.3f}, {max(depths):.3f}] within [{low}, {high}] m'
        row['enu'][2] = self.water['datum_U_m']
        row['relations'].update(water_id=self.water['id'], water_depth_range_m=[min(depths), max(depths)],
                                height_source='source_water_datum')
        return None

    def water_candidate(self, rule, kind):
        if rule['mode'] == 'in_water':
            if 'region_from' in rule:
                region = self.regions[self.records[rule['region_from']]['relations']['search_region']]
            else:
                pools = [r for r in self.regions.values() if r['kind'] in rule['allowed_region_kinds']
                         and ('allowed_region_ids' not in rule or r['id'] in rule['allowed_region_ids'])]
                region = self.rng.choice(pools)
            xs, ys = zip(*region['polygon_en'])
            center = [self.rng.uniform(min(xs), max(xs)), self.rng.uniform(min(ys), max(ys))]
            region_id = region['id']; relations = {'location_context': 'water'}
        else:
            landing = self.records[rule['near']]['relations']
            tangent = landing['shore_tangent_en']; normal = landing['water_normal_en']
            along = self.rng.uniform(*rule['along_shore_m']); outward = self.rng.uniform(*rule['water_setback_m'])
            center = [landing['water_berth_en'][k]+along*tangent[k]+outward*normal[k] for k in (0, 1)]
            region_id = self.water['id']; relations = {'near': rule['near'], 'shore_landing': rule['near']}
        if not (self.grid.e[0] <= center[0] <= self.grid.e[-1] and self.grid.n[0] <= center[1] <= self.grid.n[-1]):
            return None, 'water spawn center inside canonical bounds'
        row = self.record(rule, center, self.yaw(rule, center, region_id), region_id, kind, relations)
        reason = self.check(rule, row, in_region=rule['mode'] == 'in_water')
        if reason is None:
            reason = self.check_water(rule, row)
        return (None, reason) if reason else (row, None)

    def shore_candidate(self, rule):
        pools = [r for r in self.regions.values() if r['kind'] in rule['search_region_kinds']]
        region = self.rng.choice(pools); xs, ys = zip(*region['polygon_en'])
        target = [sum(xs)/len(xs), self.rng.uniform(min(ys), max(ys))]
        bank = self.water['banks_en'][rule['bank_index']]
        a, b = min(zip(bank, bank[1:]), key=lambda pair: math.dist(target, project(target, *pair)[0]))
        shore = project(target, a, b)[0]
        tangent = [(b[k]-a[k])/math.dist(a, b) for k in (0, 1)]
        normal = rule['water_normal_en']
        dry_setback = self.rng.uniform(*rule['dry_setback_m']); wet_setback = self.rng.uniform(*rule['water_setback_m'])
        center = [shore[k]-dry_setback*normal[k] for k in (0, 1)]
        berth = [shore[k]+wet_setback*normal[k] for k in (0, 1)]
        road = min((r for r in self.roads if not r['is_access']), key=lambda r: math.dist(center, project(center, *r['points'])[0]))
        start = project(center, *road['points'])[0]; access_id = rule['id']+':access'
        angle = math.degrees(math.atan2(center[1]-start[1], center[0]-start[0]))
        connector = rectangle([(start[k]+center[k])/2 for k in (0, 1)], math.dist(start, center), rule['access_width_m'], angle)
        relations = {'access': access_id, 'road_source': road['source_id'], 'search_region': region['id'],
                     'shore_en': shore, 'shore_tangent_en': tangent, 'water_normal_en': normal,
                     'water_berth_en': berth, 'water_id': self.water['id'], 'dry_access_en': [start, center],
                     'approach_connector_en': connector}
        row = self.record(rule, center, self.yaw(rule, center, 'unreserved_surface'), 'unreserved_surface', 'station', relations)
        reason = self.check(rule, row, in_region=False)
        if reason:
            return None, reason
        dry_samples = [*footprint_samples(row['footprint_en'], self.spacing), *footprint_samples(connector, self.spacing)]
        if any(self.grid.height(*p) < self.water['datum_U_m']+rule['dry_freeboard_m'] for p in dry_samples):
            return None, 'receiving footprint and road-to-shore access remain dry above source datum'
        if not set(footprint_classes(self.grid, connector, self.spacing)) <= set(rule['access_classes']):
            return None, 'dry landing access uses configured traversable grid classes'
        berth_rule = {**rule, 'asset_kind': rule['landing_asset_kind'], 'allowed_classes': rule['water_classes']}
        del berth_rule['dry_freeboard_m']
        wet = self.record(berth_rule, berth, math.degrees(math.atan2(tangent[1], tangent[0])), self.water['id'], 'boat', {})
        reason = self.check(berth_rule, wet, in_region=False)
        if reason is None:
            reason = self.check_water(rule, wet)
        if reason:
            return None, reason
        if math.dist(center, berth) > rule['handover_distance_m']:
            return None, 'water berth within existing physical handover distance of dry receiver'
        relations['water_access_polygon_en'] = wet['footprint_en']
        return row, None

    def place(self, rule, kind):
        reason = 'no placement draws'
        for _ in range(self.rules['max_draws_per_entity']):
            row, reason = self.candidate(rule, kind)
            if row is not None:
                self.records[rule['id']] = row
                if rule.get('mode') == 'shore_landing':
                    relation = row['relations']
                    road = {'id': relation['access'], 'source_id': relation['access'],
                            'points': relation['dry_access_en'], 'width_m': rule['access_width_m'],
                            'sidewalk_m': 0., 'is_access': True, 'authored': True}
                    road['footprint_en'] = road_polygon(road)
                    self.roads.append(road)
                    self.access_keepouts.append((road['id'], road['footprint_en']))
                return
        raise ValueError(f"{rule['id']}: exhausted {self.rules['max_draws_per_entity']} draws; rule={rule}; constraint={reason}")

    def incident(self, rule, buildings, agents):
        """Jointly sample source, route blockage and boundary staging, not tasks."""
        source_rule = buildings[rule['source']]; stage_rule = agents[rule['staging_agent']]['spawn']
        vehicle_rule = {**stage_rule, 'id': rule['staging_agent']}
        if not stage_rule['allow_on_carriageway']:
            raise ValueError(f"{vehicle_rule['id']}: network_boundary_entry requires the explicit allow_on_carriageway spawn rule")
        entries = [p for p, links in self.graph.items() if len(links) == 1 and
                   min(p[0]-self.grid.e[0], self.grid.e[-1]-p[0], p[1]-self.grid.n[0], self.grid.n[-1]-p[1])
                   <= stage_rule['boundary_distance_m'] and not self.by_road[links[0][2]]['is_access']]
        reason = 'a boundary network entry, source plot access and reachable adjacent blockage'
        if not entries:
            raise ValueError(f"{vehicle_rule['id']}: no degree-one public-road entry within {stage_rule['boundary_distance_m']} m of terrain boundary")
        debris_model = self.model(rule)
        for _ in range(self.rules['incident_draws']):
            source, reason = self.candidate(source_rule, 'building')
            if source is None:
                continue
            entry = self.rng.choice(entries); neighbor, length, stage_road = self.graph[entry][0]
            setback = self.rng.uniform(*stage_rule['boundary_setback_m'])
            if setback+vehicle_rule['clearance_m'] >= length:
                reason = 'vehicle setback inside boundary entry segment'; continue
            center = [entry[k]+setback/length*(neighbor[k]-entry[k]) for k in (0, 1)]
            yaw = math.degrees(math.atan2(neighbor[1]-entry[1], neighbor[0]-entry[0]))
            self.records[source_rule['id']] = source
            vehicle = self.record(vehicle_rule, center, yaw, stage_road, 'ground_vehicle',
                                  {'boundary_entry_en': list(entry), 'entry_road': stage_road})
            reason = self.check(vehicle_rule, vehicle, in_region=False, roadway=True)
            if reason:
                del self.records[source_rule['id']]; continue
            target_access = self.accesses[source['relations']['access']]
            route = shortest_route(self.graph, entry, tuple(round(v, 7) for v in target_access['points'][-1]))
            if route is None:
                reason = 'public-map route from boundary entry to source plot access'
                del self.records[source_rule['id']]; continue
            path, route_length = route
            candidates = [self.by_road[rid] for rid in path if rid != stage_road and not self.by_road[rid]['is_access']
                and polygon_gap(source['asset_footprint_en'], self.by_road[rid]['footprint_en']) <= rule['collapse_reach_m']
                and math.dist(*self.by_road[rid]['points']) > debris_model['footprint_size_m'][0]+2*rule['segment_end_clearance_m']]
            if not candidates:
                reason = f"source footprint within {rule['collapse_reach_m']} m collapse reach of a segment on staging-to-access shortest route"
                del self.records[source_rule['id']]; continue
            road = self.rng.choice(candidates); a, b = road['points']; span = math.dist(a, b)
            _, t = project(source['enu'], a, b)
            margin = (debris_model['footprint_size_m'][0]/2+rule['segment_end_clearance_m'])/span
            low = max(margin, t-rule['along_road_window_m']/span)
            high = min(1-margin, t+rule['along_road_window_m']/span)
            if low > high:
                reason = 'debris footprint and end clearance inside the matching segment'
                del self.records[source_rule['id']]; continue
            fraction = self.rng.uniform(low, high)
            center = [a[k]+fraction*(b[k]-a[k]) for k in (0, 1)]
            yaw = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
            debris = self.record(rule, center, yaw, road['id'], 'obstacle',
                                 {'source': source_rule['id'], 'road': road['id'], 'target_access': target_access['id'],
                                  'staging_agent': vehicle_rule['id'], 'public_shortest_route': path})
            gap = polygon_gap(source['asset_footprint_en'], debris['asset_footprint_en'])
            if gap > rule['collapse_reach_m']:
                reason = f"actual debris footprint within {rule['collapse_reach_m']} m of source footprint"
                del self.records[source_rule['id']]; continue
            self.records[vehicle_rule['id']] = vehicle
            reason = self.check(rule, debris, in_region=False, roadway=True)
            if reason:
                del self.records[source_rule['id']]; del self.records[vehicle_rule['id']]; continue
            # Match the physical cross-road footprint, leaving its actual gap.
            debris['relations'].update(collapse_gap_m=gap, collapse_reach_m=rule['collapse_reach_m'],
                                       blocked_width_m=min(road['width_m'], debris_model['footprint_size_m'][1]))
            source['relations']['blocked_road'] = road['id']
            vehicle['relations'].update(target_access=target_access['id'], target_building=source_rule['id'],
                                        public_shortest_route=path, public_route_length_m=route_length-setback)
            self.records[rule['id']] = debris
            return
        raise ValueError(f"{rule['id']} (source {rule['source']}, staging {rule['staging_agent']}): exhausted {self.rules['incident_draws']} joint draws; rule={rule}; constraint={reason}")

    def loading_stand(self, building_id, vehicle_id):
        """Choose a loading pose from the placed asset geometry, without moving it."""
        building = self.records[building_id]; vehicle = self.records[vehicle_id]
        relation = building['relations']; polygon = building['asset_footprint_en']
        center = building['enu'][:2]; start = relation['approach_start_en']
        length, width = vehicle['dimensions_m'][:2]
        clearance = max(building['clearance_m'], vehicle['clearance_m'])
        allowed = set(self.rules.get('vehicle_approach_classes', ['PLOT_SURFACE', 'ACCESS_APRON_LANE']))
        angle = math.radians(relation['entrance_heading_deg'])
        nose = [relation['entrance_en'][0]-(1.5+length/2)*math.cos(angle),
                relation['entrance_en'][1]-(1.5+length/2)*math.sin(angle)]
        others = [(eid, row['footprint_en'], max(clearance, row['clearance_m']))
                  for eid, row in self.records.items()
                  if eid not in {building_id, vehicle_id} and row['relations'].get('parent') != building_id]

        def check(poly, travel=False):
            if any(not (self.grid.e[0] <= p[0] <= self.grid.e[-1]
                        and self.grid.n[0] <= p[1] <= self.grid.n[-1]) for p in poly):
                return ['outside canonical grid']
            reasons = []
            classes = set(footprint_classes(self.grid, poly, self.spacing))
            if classes - allowed:
                reasons.append('non-drivable classes: '+', '.join(sorted(classes-allowed)))
            gap = polygon_gap(poly, polygon)
            if gap < EPS or travel and gap+EPS < clearance:
                reasons.append(f'displayed building envelope clearance: {gap:.3f} m')
            for eid, reserved, required in others:
                gap = polygon_gap(poly, reserved)
                if gap <= required+EPS:
                    reasons.append(f'reserved footprint {eid}: {gap:.3f} m <= {required:g} m clearance')
            return reasons

        # Offset corner nodes leave room for the real body to turn. The existing
        # shortest_route selects straight legs; no new navigation engine is used.
        radius = math.hypot(length, width)/2+clearance
        u = [(polygon[1][k]-polygon[0][k])/math.dist(polygon[0], polygon[1]) for k in (0, 1)]
        v = [(polygon[3][k]-polygon[0][k])/math.dist(polygon[0], polygon[3]) for k in (0, 1)]
        corners = [[p[k]+radius*(sx*u[k]+sy*v[k]) for k in (0, 1)]
                   for p, sx, sy in zip(polygon, (-1, 1, 1, -1), (-1, -1, 1, 1))]
        leg_cache = {}

        def leg_reasons(a, b):
            key = tuple(sorted((tuple(a), tuple(b))))
            if key not in leg_cache:
                if math.dist(a, b) < EPS:
                    leg_cache[key] = []
                else:
                    middle = [(a[k]+b[k])/2 for k in (0, 1)]
                    yaw = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
                    leg_cache[key] = check(rectangle(middle, math.dist(a, b), width, yaw), travel=True)
            return leg_cache[key]

        def path_to(stand, heading):
            direction = [math.cos(math.radians(heading)), math.sin(math.radians(heading))]
            # Align beyond the end of the face before the final parallel leg,
            # so turning is outside the building rather than beside its wall.
            back = min(sum((p[k]-stand[k])*direction[k] for k in (0, 1)) for p in polygon)-length/2-clearance
            lead = [stand[k]+back*direction[k] for k in (0, 1)]
            nodes = [start, lead, stand, *corners, nose]
            graph = {i: [] for i in range(len(nodes))}; rejected = set()
            for i, a in enumerate(nodes):
                for j in range(i):
                    # The stand must be reached along the selected face heading.
                    if 2 in (i, j) and 1 not in (i, j):
                        continue
                    reasons = leg_reasons(a, nodes[j])
                    if reasons:
                        rejected.update(reasons)
                        continue
                    distance = math.dist(a, nodes[j])
                    graph[i].append((j, distance, j)); graph[j].append((i, distance, i))
            route = shortest_route(graph, 0, 2)
            if route is None:
                return None, None, ['no drivable path via access/corridor and clearance corners', *sorted(rejected)]
            indices, distance = route
            return [list(nodes[i]) for i in [0, *indices]], distance, []

        def candidate(name, stand, headings, face=None):
            row = {'id': name, 'stand_en': stand, 'face_point_en': face,
                   'distance_to_casualty_m': math.dist(center, stand), 'heading_options': []}
            footprint = rectangle(stand, length, width, headings[0])
            stand_reasons = check(footprint)  # Reversing heading leaves the same parked rectangle.
            for heading in headings:
                reasons = list(stand_reasons)
                path, distance = None, None
                if not reasons:
                    if name == 'corridor':
                        reasons = leg_reasons(start, stand)
                        if not reasons:
                            path, distance = [list(start), list(stand)], math.dist(start, stand)
                    else:
                        path, distance, reasons = path_to(stand, heading)
                row['heading_options'].append({'heading_deg': heading % 360,
                    'valid': not reasons, 'rejection_reasons': reasons, 'footprint_en': footprint,
                    'path_en': path, 'path_length_m': distance})
            valid = [option for option in row['heading_options'] if option['valid']]
            chosen = min(valid, key=lambda option: option['path_length_m']) if valid else row['heading_options'][0]
            row.update(chosen)
            if not valid:
                row['rejection_reasons'] = list(dict.fromkeys(reason for option in row['heading_options']
                                                            for reason in option['rejection_reasons']))
            return row

        candidates = [candidate('corridor', nose, [relation['entrance_heading_deg']], relation['entrance_en'])]
        for index, (a, b) in enumerate(edges(polygon)):
            face, _ = project(center, a, b)
            edge_length = math.dist(a, b)
            normal = [(b[1]-a[1])/edge_length, -(b[0]-a[0])/edge_length]
            if sum(normal[k]*(face[k]-center[k]) for k in (0, 1)) < 0:
                normal = [-value for value in normal]
            stand = [face[k]+(1.5+width/2)*normal[k] for k in (0, 1)]
            heading = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
            candidates.append(candidate(f'face-{index+1}', stand, [heading, heading+180], face))
        valid = [row for row in candidates if row['valid']]
        if valid:
            nearest = min(row['distance_to_casualty_m'] for row in valid)
            chosen = min((row for row in valid if row['distance_to_casualty_m'] <= nearest+EPS),
                         key=lambda row: row['path_length_m'])
        else:
            chosen = candidates[0]
        for row in candidates:
            row['selected'] = row is chosen
            if row['valid'] and row is not chosen:
                row['selection_reason'] = ('farther from casualty' if row['distance_to_casualty_m'] > chosen['distance_to_casualty_m']+EPS
                                           else 'longer path at equal casualty distance')
        relation['loading_stand'] = {'vehicle_id': vehicle_id, 'vehicle_asset_id': vehicle['asset_id'],
            'vehicle_dimensions_m': [length, width], 'clearance_m': clearance,
            'selected_candidate': chosen['id'], 'candidates': candidates,
            'selection_reason': 'nearest valid stand, then shortest path' if valid else 'no valid candidate; retain corridor stop'}
        relation['entrance_en'] = chosen['stand_en']
        relation['entrance_heading_deg'] = chosen['heading_deg']
        relation['approach_waypoints_en'] = chosen['path_en'] if chosen['path_en'] is not None else [list(start), list(nose)]

    def generate(self):
        buildings = {r['id']: r for r in expand_rules(self.rules['buildings'])}
        agents = {a['id']: a for a in self.config['agents']}
        debris = list(expand_rules(self.rules['debris']))
        for rule in debris:
            self.incident(rule, buildings, agents)
        for rule in buildings.values():
            if rule['id'] not in self.records:
                self.place(rule, 'building')
        for rule in expand_rules(self.rules['stations']):
            self.place(rule, 'station')
        for group in ('facilities', 'equipment'):
            for rule in expand_rules(self.rules.get(group, [])):
                self.place(rule, 'facility')
        for agent in self.config['agents']:
            if agent['id'] not in self.records:
                self.place({**agent['spawn'], 'id': agent['id']}, agent['kind'])
        for rule in expand_rules(self.rules.get('humans', [])):
            self.place(rule, 'human')
        # Core G02 places interior occupants at the building polygon centroid.
        # Record that same pose here; do not invent a separate outdoor person.
        for rule in buildings.values():
            parent = self.records[rule['id']]
            for i, _ in enumerate(rule['occupants'], 1):
                person_rule = {'asset_kind': 'person', 'clearance_m': 0.}
                person = self.record(person_rule, parent['enu'][:2], parent['yaw_deg'], parent['region'], 'human',
                                     {'parent': rule['id'], 'location_context': 'interior'})
                person['classes'] = footprint_classes(self.grid, person['footprint_en'], self.spacing)
                person['nearest_road'] = parent['nearest_road']
                person['nearest_road_clearance_m'] = min(polygon_gap(person['footprint_en'], r['footprint_en']) for r in self.roads if not r['is_access'])
                self.records[f"human-{rule['id']}-{i}"] = person
        from .core.agents.embodiment import EMBODIMENTS
        wheels, carriers = [], []
        for agent in self.config['agents']:
            spec = {**EMBODIMENTS[agent['kind']], **agent.get('overrides', {})}
            if spec['mobility'] == 'wheeled':
                wheels.append(agent['id'])
                if 'C22' in spec['capabilities']:
                    carriers.append(agent['id'])
        for building_id in buildings:
            available = carriers or wheels
            if available:
                vehicle_id = min(available, key=lambda eid: self.records[eid]['relations'].get('target_building') != building_id)
                self.loading_stand(building_id, vehicle_id)
        pairs = []
        for eid, row in self.records.items():
            row['overlap_check'] = {'ok': True, 'pairs': []}
        items = list(self.records.items())
        for i, (eid, row) in enumerate(items):
            for oid, other in items[:i]:
                parent, other_parent = row['relations'].get('parent'), other['relations'].get('parent')
                interior = parent == oid or other_parent == eid or (parent is not None and parent == other_parent)
                gap = polygon_gap(row['footprint_en'], other['footprint_en'])
                required = max(row['clearance_m'], other['clearance_m'])
                ok = interior or gap > required+EPS
                pair = {'a': oid, 'b': eid, 'overlap': gap < EPS, 'gap_m': gap,
                        'required_gap_m': required, 'allowed_interior': interior, 'ok': ok}
                pairs.append(pair)
                row['overlap_check']['pairs'].append(pair); other['overlap_check']['pairs'].append(pair)
                row['overlap_check']['ok'] &= ok; other['overlap_check']['ok'] &= ok
        result = {'entities': self.records, 'roads': self.roads, 'regions': self.regions,
                'accesses': self.accesses, 'sources': self.sources, 'pairwise': pairs,
                'building_rules': buildings, 'debris_rules': debris}
        if self.water is not None:
            result['water'] = self.water
        return result


def generate_layout(config, grid):
    return Layout(config, grid).generate()
