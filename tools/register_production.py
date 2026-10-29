#!/usr/bin/env python3
"""Register READY production assets (assets/production/*/asset.json) in assets/production_map.json.

Same record format as lane A2's assets/asset_map.json (kinds/assets, boundsM, sizeM, forward, up), so
the layout generator and display staging read both maps; A2's map is never modified. Bounds are
measured here from each exported GLB (node transforms x POSITION accessor min/max), not copied.
"""
import json
import re
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
# display kind -> (production dir, manifest asset id or None for single-asset manifests, default state).
KINDS = {'clearance_loader': ('P1_clearance_loader', None, 'BASE'),
         'human_prone': ('P2_HUM-001-C01_poses', 'HUM-001-C01-P-PRONE', 'PRONE'),
         'human_supine': ('P2_HUM-001-C01_poses', 'HUM-001-C01-P-SUPINE', 'SUPINE'),
         'ambulance': ('P5_RSC-008-C01_ambulance', 'RSC-008-C01-P-CLOSED', 'CLOSED'),
         'ambulance_loaded': ('P5_RSC-008-C01_ambulance', 'RSC-008-C01-P-LOADED', 'LOADED'),
         'boat': ('P3_rescue_boat', None, 'BASE'),
         'boat_loaded': ('P3_rescue_boat', None, 'LOADED'),
         'human_in_water': ('P4_person_in_water', None, 'FLOATING'),
         'fire_unit': ('P6_fire_unit', None, 'BASE'),
         'fire_unit_deployed': ('P6_fire_unit', None, 'DEPLOYED'),
         'fire_pump': ('P6_fire_unit', None, 'PORTABLE-PUMP'),
         'rope_rig': ('P7_cliff_rope_rescue', None, 'STOWED'),
         'rope_rig_rigged': ('P7_cliff_rope_rescue', None, 'RIGGED'),
         'human_snow_casualty': ('P8_snow_casualty_sled', 'NEW-HUM-S01', 'CASUALTY'),
         'sled': ('P8_snow_casualty_sled', 'NEW-SLED-001', 'SLED_EMPTY'),
         'sled_loaded': ('P8_snow_casualty_sled', 'NEW-SLED-001', 'SLED_LOADED'),
         'tank': ('P9_tank_bund', 'NEW-TANK-001-C01', 'INTACT'),
         'tank_leaking': ('P9_tank_bund', 'NEW-TANK-001-C01', 'LEAKING'),
         'bund': ('P9_tank_bund', 'NEW-BUND-001', 'INTACT'),
         'bund_leaking': ('P9_tank_bund', 'NEW-BUND-001', 'LEAKING')}


def matrix(node):
    if 'matrix' in node:
        m = node['matrix']; return [[m[c*4+r] for c in range(4)] for r in range(4)]
    x, y, z, w = node.get('rotation', [0, 0, 0, 1]); sx, sy, sz = node.get('scale', [1, 1, 1]); t = node.get('translation', [0, 0, 0])
    r = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return [[r[i][0]*sx, r[i][1]*sy, r[i][2]*sz, t[i]] for i in range(3)]+[[0, 0, 0, 1]]


def mul(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def bounds(path):
    data = path.read_bytes(); length = struct.unpack_from('<I', data, 12)[0]
    doc = json.loads(data[20:20+length]); lo = [math.inf]*3; hi = [-math.inf]*3
    def walk(index, parent):
        node = doc['nodes'][index]; world = mul(parent, matrix(node))
        for prim in doc['meshes'][node['mesh']]['primitives'] if 'mesh' in node else []:
            acc = doc['accessors'][prim['attributes']['POSITION']]
            for corner in [[(acc['min'], acc['max'])[(k >> i) & 1][i] for i in range(3)] for k in range(8)]:
                p = [sum(world[i][j]*corner[j] for j in range(3))+world[i][3] for i in range(3)]
                for i in range(3): lo[i] = min(lo[i], p[i]); hi[i] = max(hi[i], p[i])
        for child in node.get('children', []): walk(child, world)
    identity = [[float(i == j) for j in range(4)] for i in range(4)]
    for root in doc['scenes'][doc.get('scene', 0)]['nodes']: walk(root, identity)
    return lo, hi


def main():
    out = {'schema': 'rescue.production-map/v1', 'sizeM_order': ['length_along_forward', 'width_horizontal_perpendicular', 'height_Y'],
           'kinds': {}, 'assets': {}}
    for kind, (folder, asset_id, default) in KINDS.items():
        base = ROOT/'assets/production'/folder
        manifest = json.loads((base/'asset.json').read_text())
        if asset_id: manifest = next(a for a in manifest['assets'] if a['id'] == asset_id)
        forward = re.search(r'[+-][XYZxyz]', manifest['forward_axis']).group().lower()  # e.g. 'glTF -Z (Blender +Y)'
        states = {}
        for row in manifest['states']:
            lo, hi = bounds(base/row['glb']); ext = [hi[i]-lo[i] for i in range(3)]
            along = 0 if forward[1] == 'x' else 2
            aid = manifest['id'] if manifest['id'].endswith(row['state']) else f"{manifest['id']}-{row['state']}"
            out['assets'][aid] = {'asset_id': aid, 'file': str((base/row['glb']).relative_to(ROOT)), 'label_zh': manifest['label_zh'],
                                  'boundsM': {'min': lo, 'max': hi}, 'sizeM': [ext[along], ext[2-along], ext[1]], 'forward': forward,
                                  'up': re.search(r'[+-][XYZxyz]', manifest['up']).group().lower(), 'external_uris': [], 'production_status': manifest['status'],
                                  'excel_status': manifest['excel_status'], 'manifest': str((base/'asset.json').relative_to(ROOT))}
            states[row['state']] = aid
        out['kinds'][kind] = {**out['assets'][states[default]], 'states': states}
    (ROOT/'assets/production_map.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: [v['states'], [round(x, 3) for x in v['sizeM']], v['forward']] for k, v in out['kinds'].items()}))


if __name__ == '__main__':
    main()
