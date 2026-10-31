"""Same-section four-channel scene rasteriser for phase-1 aerial/ground pairs.

One scene graph, one camera model, one Z-buffer: every render of a section draws the
*whole* scene (terrain + roads + buildings + debris + casualties + vehicles) so the
aerial and ground cameras of a pair see the same geometry, not two different worlds.

Frozen conventions (contracts ③ and ⑧ of docs/PHASE1_LOCAL_EXECUTION_PLAN.md)
------------------------------------------------------------------------------
* Resolution 640x384 by default: divisible by 32 in both axes, i.e. 12x20 = 240
  post-merger visual tokens for Qwen3.5 (patch 16, merge 2).
* Camera optics: ``+X`` right, ``+Y`` down, ``+Z`` forward; ``depth`` is the linear
  optical ``Z_c`` in metres, ``0.0`` (with ``valid_mask == False``) for sky.
* Camera pose: ``yaw`` = compass bearing from East towards North (deg), ``pitch``
  negative = looking down, ``roll`` about the optical axis (deg).  The world-to-camera
  rotation is ``R_cw = (Rz(yaw) @ Ry(-pitch) @ Rx(roll) @ B)^T`` with ``B`` the
  upright zero-attitude frame ``(X_c, Y_c, Z_c) = (South, Down, East)``.
* GLB assets are glTF +Y up.  The scene background is authored with
  ``X = East, Y = Up, Z = -North``; dynamic entities reuse the AAS display chain of
  ``engine.py`` (``grid.enu`` + ``Rz(yaw) @ Rx(90deg)`` + ground-normal tilt).
* Entity visibility follows ``engine.py`` 578-658: cleared/moved debris is not drawn,
  a casualty carried by a vehicle is drawn only by the carrier's LOADED variant, and
  a person/vehicle whose display state has a dedicated variant is drawn with it.
* No text, no borders, no debug ids are ever painted into ``rgb``.

The rasteriser is a batched PyTorch triangle-bbox sweep (CUDA or CPU) with a packed
int64 Z-buffer key ``(Z_c * 1e5) << 24 | triangle_slot``: one pass over the visible
triangles yields depth, camera/world normals, instance ids and shaded RGB together.
"""
from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:  # the AAS scene runtime ships these helpers; never re-implement the glTF reader
    from rescue_scenes.glb_runtime import _local_matrix, _read_accessor, read_glb
except ImportError:  # pragma: no cover - editable checkout without src on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rescue_scenes.glb_runtime import _local_matrix, _read_accessor, read_glb

WIDTH, HEIGHT = 640, 384
PATCH_MERGE = 32                      # post-merger token pitch (patch 16 x merge 2)
SKY_ID, BACKGROUND_ID = 0, 1          # 0 = sky/invalid, 1 = terrain/road background
NEAR_M = 0.1
_SLOT_BITS = 24
_DEPTH_SCALE = 1e5                    # 0.01 mm depth quantisation inside the key
_KEY_EMPTY = np.int64((1 << 62) - 1)

# Sun and sky of the renderer (fixed, documented, not fitted to any image).
SUN_AZIMUTH_DEG, SUN_ELEVATION_DEG = 235.0, 42.0
SUN_RADIANCE = 1.02
SKY_ZENITH = np.array([0.30, 0.44, 0.63], np.float32)
SKY_HORIZON = np.array([0.74, 0.79, 0.85], np.float32)
SURFACE_COLOR = {                     # semantic albedo of the box/cylinder fallbacks
    'human': (0.72, 0.24, 0.20), 'obstacle': (0.46, 0.42, 0.37),
    'platform': (0.84, 0.85, 0.86), 'station': (0.26, 0.55, 0.50),
    'building': (0.56, 0.54, 0.50), 'item': (0.80, 0.66, 0.28),
    'hazard': (0.62, 0.36, 0.22), 'facility': (0.44, 0.50, 0.54),
    'road': (0.16, 0.16, 0.17), 'open_area': (0.36, 0.33, 0.28),
    'water': (0.05, 0.14, 0.20),
}
ROAD_COLOR = {'asphalt': (0.13, 0.13, 0.14), 'concrete': (0.46, 0.46, 0.44),
              'gravel': (0.40, 0.37, 0.32), 'dirt': (0.33, 0.27, 0.20),
              'paved': (0.30, 0.30, 0.31)}
# Real 3-D fallback extents (length, width, height) when an entity has no delivered GLB.
DEFAULT_EXTENT = {'human': (1.72, 0.55, 1.70), 'obstacle': (3.0, 3.0, 1.4),
                  'uav': (1.0, 1.0, 0.45), 'ground_vehicle': (5.6, 2.2, 2.4),
                  'boat': (4.6, 1.9, 1.1), 'robot_dog': (0.85, 0.45, 0.50),
                  'specialist_team': (0.9, 0.7, 1.80), 'station': (6.0, 4.0, 3.0),
                  'building': (12.0, 10.0, 6.0), 'item': (0.8, 0.8, 0.7),
                  'hazard': (2.0, 1.6, 1.6), 'facility': (6.0, 3.0, 2.6)}
_FORWARD_TURN = {'+x': 0.0, '-x': math.pi, '+z': math.pi / 2, '-z': -math.pi / 2}
_STATIC_CLASSES = {'road', 'open_area', 'water', 'building', 'station', 'facility', 'item', 'hazard'}


# --------------------------------------------------------------------------- geometry helpers

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _homogeneous(rot, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, t
    return m


def camera_rotation(yaw_deg, pitch_deg, roll_deg):
    """World(ENU)-from-camera rotation for the (right, down, forward) optical frame."""
    base = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return rot_z(math.radians(yaw_deg)) @ rot_y(-math.radians(pitch_deg)) \
        @ rot_x(math.radians(roll_deg)) @ base


def _tilt_rotation(normal):
    """Minimal rotation carrying +U onto the measured ground normal (engine.py tilt quat)."""
    n = np.asarray(normal, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    axis = np.cross([0.0, 0.0, 1.0], n)
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3) if n[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return _axis_angle(axis / s, math.atan2(s, float(n[2])))


def _axis_angle(axis, angle):
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[c + x * x * t, x * y * t - z * s, x * z * t + y * s],
                     [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
                     [z * x * t - y * s, z * y * t + x * s, c + z * z * t]])


def _position_xy(entity):
    """engine.py ``position_xy``: truth pose when present, else the geometry centroid."""
    fields = entity.get('fields', {})
    pose = fields.get('entity.pose', fields.get('human.pose'))
    if pose is not None:
        return [float(pose[0]), float(pose[1])]
    points = entity['geometry']['coords']
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    return [sum(p[k] for p in points) / len(points) for k in (0, 1)]


def _face_normals(verts, faces):
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    n = np.cross(a, b)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _box_mesh(length, width, height):
    """Axis-aligned box in the glTF Y-up asset frame: X = width, Y = height, Z = length."""
    hx, hz = width / 2.0, length / 2.0
    v = np.array([[-hx, 0, -hz], [hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz],
                  [-hx, height, -hz], [hx, height, -hz], [hx, height, hz], [-hx, height, hz]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return v, f


def _cylinder_mesh(radius, height, segments=20):
    """Upright cylinder in the glTF Y-up asset frame (a person's real standing volume)."""
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(a), np.zeros(segments), radius * np.sin(a)], 1)
    top = ring + np.array([0.0, height, 0.0])
    v = np.concatenate([ring, top, [[0, 0, 0], [0, height, 0]]])
    f = []
    for i in range(segments):
        j = (i + 1) % segments
        f += [[i, j, segments + j], [i, segments + j, segments + i]]
        f += [[2 * segments, j, i], [2 * segments + 1, segments + i, segments + j]]
    return v, np.array(f)


# --------------------------------------------------------------------------- asset access

def _asset_root():
    try:
        from rescue_scenes.paths import PRIVATE_ROOT
        return Path(PRIVATE_ROOT)
    except Exception:  # pragma: no cover - explicit root always wins
        return Path(__file__).resolve().parents[3] / 'scene_data' / 'rescue-data'


def _resolve_asset(root, relative):
    for base in (Path(root), Path(root).parent / 'scene_sim'):
        path = base / relative
        if path.is_file():
            return path
    return None


def _load_asset_maps(root):
    kinds, assets = {}, {}
    for name in ('asset_map.json', 'production_map.json'):
        path = Path(root) / 'assets' / name
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding='utf-8'))
        kinds.update(doc.get('kinds', {}))
        rows = doc.get('assets', {})
        assets.update(rows if isinstance(rows, dict) else {r['asset_id']: r for r in rows})
    return kinds, assets


def _texture_rgb(doc, binary, texture_index, cache, key):
    entry = doc['textures'][texture_index]
    source = entry.get('source')
    if source is None or source in cache:
        return cache.get(source)
    image = doc['images'][source]
    if 'bufferView' not in image:
        cache[source] = None
        return None
    view = doc['bufferViews'][image['bufferView']]
    start = view.get('byteOffset', 0)
    raw = binary[start:start + view['byteLength']]
    try:
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'), np.float32) / 255.0
    except Exception:  # pragma: no cover - a broken embedded image must not kill a render
        arr = None
    cache[source] = arr
    return arr


def _primitive_colors(doc, binary, prim, uv, cache, key):
    """Per-vertex linear RGB = baseColorFactor * baseColorTexture(uv) * COLOR_0."""
    material = doc['materials'][prim['material']] if 'material' in prim else {}
    pbr = material.get('pbrMetallicRoughness', {})
    factor = np.array(list(pbr.get('baseColorFactor', [1, 1, 1, 1]))[:3], np.float32)
    count = len(uv) if uv is not None else len(_read_accessor(doc, binary, prim['attributes']['POSITION']))
    color = np.repeat(factor[None, :], count, 0)
    if uv is not None and pbr.get('baseColorTexture'):
        image = _texture_rgb(doc, binary, pbr['baseColorTexture']['index'], cache, key)
        if image is not None:
            h, w, _ = image.shape
            px = np.clip((np.clip(uv[:, 0], 0, 1) * (w - 1)).astype(np.int32), 0, w - 1)
            py = np.clip((np.clip(uv[:, 1], 0, 1) * (h - 1)).astype(np.int32), 0, h - 1)
            color = color * image[py, px]
    if 'COLOR_0' in prim.get('attributes', {}):
        vc = _read_accessor(doc, binary, prim['attributes']['COLOR_0'])
        color = color * vc[:, :3].astype(np.float32)
    return np.clip(color, 0.0, 1.0)


def _subdivide_textured_triangles(pos_world, index_array, uv, max_edge_m=8.0, max_passes=3, max_faces=250000):
    """Vectorised 1->4 midpoint subdivision for large textured triangles so 2K textures
    are sampled densely across ground/facade surfaces instead of only at 3 corner vertices."""
    if uv is None or len(index_array) == 0:
        return pos_world, index_array, uv
    verts = np.asarray(pos_world, dtype=np.float32)
    faces = np.asarray(index_array, dtype=np.int64)
    uvs = np.asarray(uv, dtype=np.float32)
    for _ in range(max_passes):
        if len(faces) >= max_faces:
            break
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        e01 = np.linalg.norm(v1 - v0, axis=1)
        e12 = np.linalg.norm(v2 - v1, axis=1)
        e20 = np.linalg.norm(v0 - v2, axis=1)
        big = np.maximum(np.maximum(e01, e12), e20) > max_edge_m
        if not np.any(big):
            break
        keep_faces = faces[~big]
        sub_f = faces[big]
        n_v = len(verts)
        m_sub = len(sub_f)
        # Create midpoint vertices m01, m12, m20
        m01_v = 0.5 * (verts[sub_f[:, 0]] + verts[sub_f[:, 1]])
        m12_v = 0.5 * (verts[sub_f[:, 1]] + verts[sub_f[:, 2]])
        m20_v = 0.5 * (verts[sub_f[:, 2]] + verts[sub_f[:, 0]])
        m01_u = 0.5 * (uvs[sub_f[:, 0]] + uvs[sub_f[:, 1]])
        m12_u = 0.5 * (uvs[sub_f[:, 1]] + uvs[sub_f[:, 2]])
        m20_u = 0.5 * (uvs[sub_f[:, 2]] + uvs[sub_f[:, 0]])
        verts = np.concatenate([verts, m01_v, m12_v, m20_v], axis=0)
        uvs = np.concatenate([uvs, m01_u, m12_u, m20_u], axis=0)
        i0, i1, i2 = sub_f[:, 0], sub_f[:, 1], sub_f[:, 2]
        im01 = np.arange(n_v, n_v + m_sub, dtype=np.int64)
        im12 = np.arange(n_v + m_sub, n_v + 2 * m_sub, dtype=np.int64)
        im20 = np.arange(n_v + 2 * m_sub, n_v + 3 * m_sub, dtype=np.int64)
        f_new = np.concatenate([
            np.stack([i0, im01, im20], axis=1),
            np.stack([im01, i1, im12], axis=1),
            np.stack([im20, im12, i2], axis=1),
            np.stack([im01, im12, im20], axis=1),
        ], axis=0)
        faces = np.concatenate([keep_faces, f_new], axis=0)
    return verts, faces, uvs


def bake_glb(path, skip_roles=('METADATA_RESERVATION',)):
    """Flatten a GLB into asset-frame triangles: verts (V,3), faces (F,3), face RGB (F,3).

    Geometry keeps the delivered node transforms; per-face colour is the material
    base colour times the sampled base-colour texture (and vertex colour) at the
    face centroid UV.  Metadata prisms (clear-volume reservations) are display
    aids, not surfaces, so they are skipped by role.
    """
    doc, binary = read_glb(Path(path))
    nodes, meshes = doc.get('nodes', []), doc.get('meshes', [])
    scene = doc['scenes'][doc.get('scene', 0)]
    textures, verts, faces, colors = {}, [], [], []
    base = 0
    stack = [(index, np.eye(4)) for index in scene['nodes']]
    while stack:
        index, parent = stack.pop()
        node = nodes[index]
        world = parent @ _local_matrix(node)
        if (node.get('extras') or {}).get('role') in skip_roles:
            continue
        stack.extend((child, world) for child in node.get('children', []))
        if 'mesh' not in node:
            continue
        linear = world[:3, :3]
        mirrored = np.linalg.det(linear) < 0
        for prim in meshes[node['mesh']]['primitives']:
            if prim.get('mode', 4) != 4 or 'POSITION' not in prim.get('attributes', {}):
                continue
            position = _read_accessor(doc, binary, prim['attributes']['POSITION'])
            if 'indices' in prim:
                index_array = _read_accessor(doc, binary, prim['indices']).ravel().astype(np.int64)
            else:
                index_array = np.arange(len(position), dtype=np.int64)
            if index_array.size < 3:
                continue
            index_array = index_array.reshape(-1, 3)
            if mirrored:
                index_array = index_array[:, ::-1]
            uv = (_read_accessor(doc, binary, prim['attributes']['TEXCOORD_0'])
                  if 'TEXCOORD_0' in prim['attributes'] else None)
            pos_world = position @ linear.T + world[:3, 3]
            material = doc['materials'][prim['material']] if 'material' in prim else {}
            has_tex = (
                uv is not None
                and 'COLOR_0' not in prim.get('attributes', {})
                and bool(material.get('pbrMetallicRoughness', {}).get('baseColorTexture'))
            )
            if has_tex:
                pos_world, index_array, uv = _subdivide_textured_triangles(pos_world, index_array, uv)
            v_col = _primitive_colors(doc, binary, prim, uv, textures, path)
            face_col = v_col[index_array].mean(1)
            verts.append(pos_world)
            faces.append(index_array + base)
            colors.append(face_col)
            base += len(pos_world)
    if not verts:
        raise ValueError(f'{path}: no triangle geometry')
    verts = np.concatenate(verts).astype(np.float32)
    faces = np.concatenate(faces).astype(np.int32)
    colors = np.concatenate(colors).astype(np.float32)
    return verts, faces, colors


def _triangulate(points):
    """Fan triangulation of a simple polygon given as (k,2) ENU points."""
    faces = [[0, i, i + 1] for i in range(1, len(points) - 1)]
    return np.array(faces, np.int32)


def _safe_grid_enu(grid, xy, agl=0.0):
    x, y = float(xy[0]), float(xy[1])
    try:
        res = grid.enu([x, y], agl)
        return float(res[0]), float(res[1]), float(res[2])
    except ValueError:
        b = getattr(grid, 'bounds', [0.0, 0.0, float(grid.e[-1] - grid.e[0]), float(grid.n[-1] - grid.n[0])])
        xc = max(float(b[0]) + 0.05, min(float(b[2]) - 0.05, x))
        yc = max(float(b[1]) + 0.05, min(float(b[3]) - 0.05, y))
        res = grid.enu([xc, yc], agl)
        return float(res[0] + (x - xc)), float(res[1] + (y - yc)), float(res[2])


def _ribbon(points, width, grid, lift=0.04, step=4.0):
    """Road ribbon: the polyline swept to ``width`` and draped on the terrain."""
    points = np.asarray(points, float)
    verts, faces = [], []
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        pieces = max(1, int(math.ceil(length / step)))
        direction = (b - a) / max(length, 1e-9)
        normal = np.array([-direction[1], direction[0]])
        for k in range(pieces):
            for t in (k / pieces, (k + 1) / pieces):
                centre = a + (b - a) * t
                for side in (-0.5, 0.5):
                    xy = centre + normal * width * side
                    e, n, u = _safe_grid_enu(grid, xy, 0.0)
                    verts.append([e, n, u + lift])
    count = len(verts) // 4
    for i in range(count):
        o = 4 * i
        faces += [[o, o + 1, o + 3], [o, o + 3, o + 2]]
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


# --------------------------------------------------------------------------- renderer

class SceneMultichannelRenderer:
    """Full-scene RGB + depth + normal + instance rasteriser for one scene section."""

    def __init__(self, cfg, world=None, layout=None, *, grid=None, agents=None,
                 width=WIDTH, height=HEIGHT, device=None, asset_root=None,
                 near=NEAR_M, sun=None, fog_scale_m=420.0):
        self.cfg = cfg
        self.width, self.height = int(width), int(height)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.near = float(near)
        self.display = dict(cfg.get('display', {}))
        self.grid = grid if grid is not None else self._load_grid(cfg['background']['grid'])
        self.asset_root = Path(asset_root) if asset_root else _asset_root()
        self.kinds, self.assets = _load_asset_maps(self.asset_root)
        self.world = world
        self.agents = dict(agents or {})
        self.layout = layout or {}
        self.sun = np.asarray(sun, float) if sun is not None else np.array([
            math.cos(math.radians(SUN_ELEVATION_DEG)) * math.cos(math.radians(SUN_AZIMUTH_DEG)),
            math.cos(math.radians(SUN_ELEVATION_DEG)) * math.sin(math.radians(SUN_AZIMUTH_DEG)),
            math.sin(math.radians(SUN_ELEVATION_DEG))], float)
        self.fog_scale_m = float(fog_scale_m)
        self.water_u = float((cfg.get('placement', {}).get('water') or {}).get('datum_U_m', 0.0)) \
            if (cfg.get('placement', {}).get('water')) else None
        self.entity_id_to_id = {}
        self.id_to_entity_id = {}
        self.entity_boxes = {}
        self._mesh_cache = {}
        self._yaw = {}
        self._previous_xy = {}
        self._static = None
        self._dynamic = None
        self._dynamic_key = None
        self._grid_tensors()
        self.last_stats = {}
        self.rebuild()

    # -- construction helpers ---------------------------------------------------------
    @classmethod
    def from_environment(cls, cfg, env, bundle=None, **kwargs):
        layout = (bundle or {}).get('audit', {}).get('layout')
        return cls(cfg, world=env.sim.world, layout=layout, agents=env.sim.agents, **kwargs)

    @staticmethod
    def _load_grid(path):
        from rescue_scenes.backgrounds import Grid
        return Grid(path)

    def _grid_tensors(self):
        grid = self.grid
        self._ge = torch.as_tensor(np.asarray(grid.e, np.float32), device=self.device)
        self._gn = torch.as_tensor(np.asarray(grid.n, np.float32), device=self.device)
        self._gu = torch.as_tensor(np.asarray(grid.u, np.float32).reshape(len(grid.n), len(grid.e)),
                                   device=self.device)

    def terrain_height(self, e, n):
        """Vectorised bilinear terrain height on the canonical grid (torch tensors)."""
        ge, gn = self._ge, self._gn
        i = (torch.searchsorted(ge, e.contiguous(), right=True) - 1).clamp(0, len(ge) - 2)
        j = (torch.searchsorted(gn, n.contiguous(), right=True) - 1).clamp(0, len(gn) - 2)
        dx = ((e - ge[i]) / (ge[i + 1] - ge[i])).clamp(0.0, 1.0)
        dy = ((n - gn[j]) / (gn[j + 1] - gn[j])).clamp(0.0, 1.0)
        u00 = self._gu[j, i]
        u10 = self._gu[j, i + 1]
        u01 = self._gu[j + 1, i]
        u11 = self._gu[j + 1, i + 1]
        return (1 - dy) * ((1 - dx) * u00 + dx * u10) + dy * ((1 - dx) * u01 + dx * u11)

    def _instance(self, entity_id):
        if entity_id not in self.entity_id_to_id:
            index = len(self.entity_id_to_id) + 2
            self.entity_id_to_id[entity_id] = index
            self.id_to_entity_id[index] = entity_id
        return self.entity_id_to_id[entity_id]

    def _mesh(self, path):
        path = Path(path)
        if path not in self._mesh_cache:
            self._mesh_cache[path] = bake_glb(path)
        return self._mesh_cache[path]

    # -- scene assembly ----------------------------------------------------------------
    def rebuild(self, world=None, layout=None):
        if world is not None:
            self.world = world
        if layout is not None:
            self.layout = layout
        self.entity_boxes.clear()
        self._static = self._build_static()
        self._dynamic_key = None
        self._dynamic = self._build_dynamic()
        return self

    def _entities(self):
        return dict(getattr(self.world, 'entities', {}) or {})

    def _pose(self, entity_id, entity):
        """engine.py display pose: grid.enu + travel yaw + ground-normal tilt."""
        row = self.layout.get(entity_id, {})
        fields = entity.get('fields', {})
        kind = fields.get('platform.kind') if entity['cls'] == 'platform' else entity['cls']
        agent = self.agents.get(entity_id)
        if agent is not None and getattr(agent, 'kind', None) == 'uav':
            agl = float(self.display.get('uav_agl_m', 0.0))
        elif agent is not None:
            agl = float(self.display.get('agent_ground_offset_m', 0.0))
        elif entity['cls'] == 'platform':
            agl = float(self.display.get('uav_agl_m' if kind == 'uav' else 'agent_ground_offset_m', 0.0))
        elif entity['cls'] == 'human':
            agl = float(self.display.get('person_offset_m', 0.0))
        else:
            agl = 0.0
        xy = _position_xy(entity)
        enu = [float(v) for v in self.grid.enu(xy, agl)]
        carrier = fields.get('human.carrier_id')
        afloat = (kind == 'boat') or (entity['cls'] == 'human'
                                      and fields.get('human.location_context') == 'water' and not carrier)
        if afloat and self.water_u is not None:
            enu[2] = self.water_u
        old = self._previous_xy.get(entity_id)
        if old is not None and math.dist(old[:2], enu[:2]) > 1e-6:
            self._yaw[entity_id] = math.atan2(enu[1] - old[1], enu[0] - old[0])
        self._previous_xy[entity_id] = enu
        yaw = self._yaw.get(entity_id, math.radians(float(row.get('yaw_deg', 0.0))))
        rotation = rot_z(yaw) @ rot_x(math.pi / 2)
        normal = (row.get('relations') or {}).get('ground_normal_enu')
        tilt = np.eye(3)
        if normal:
            tilt = _tilt_rotation(normal)
            rotation = tilt @ rotation
        offset = (row.get('relations') or {}).get('local_offset_m')
        if offset:
            enu = [enu[i] + float((rot_z(yaw) @ np.asarray(offset, float))[i]) for i in range(3)]
        return enu, yaw, rotation, tilt

    def _asset_row(self, entity_id, entity, display_kind=None):
        """Delivered asset for an entity: the layout's asset id, then the display variant.

        A variant is either a top-level delivered kind (``human_supine``,
        ``ambulance_loaded``, ``human_in_water``) or a state of the entity's own kind
        (debris ``opening``), which is exactly how ``presentation.TYPES`` names it.
        """
        row = self.layout.get(entity_id, {})
        base = self.assets.get(row.get('asset_id')) if row.get('asset_id') else None
        asset = base
        if display_kind:
            variant = self.kinds.get(display_kind)
            if variant and variant.get('file'):
                asset = variant
            else:
                kind_row = self.kinds.get((base or {}).get('asset_kind') or row.get('asset_kind') or '', {})
                candidate = (kind_row.get('states') or {}).get(display_kind)
                if candidate in self.assets:
                    asset = self.assets[candidate]
        if asset is None:
            kind = (row.get('asset_kind') or (entity.get('fields', {}).get('platform.kind')
                                              if entity['cls'] == 'platform' else entity['cls']))
            alias = {'person': 'human', 'debris': 'obstacle', 'medical_point': 'station',
                     'shelter': 'station', 'ambulance': 'ground_vehicle', 'boat': 'platform'}.get(kind, kind)
            asset = self.kinds.get(kind) or self.kinds.get(alias)
        return asset or {}, row

    def _entity_extent(self, entity, row, kind):
        dims = row.get('dimensions_m')
        if dims and len(dims) == 3:
            return float(dims[0]), float(dims[1]), float(dims[2])
        return DEFAULT_EXTENT.get(kind, DEFAULT_EXTENT['building'])

    def _append(self, store, verts, faces, color, instance_id):
        """Append one primitive; ``color`` is either one RGB or one RGB per face."""
        faces = np.asarray(faces, np.int32)
        color = np.asarray(color, np.float32).reshape(-1, 3)
        if len(color) == 1:
            color = np.repeat(color, len(faces), 0)
        elif len(color) != len(faces):
            raise ValueError(f'per-face colour count {len(color)} != {len(faces)} triangles')
        store['verts'].append(np.asarray(verts, np.float32))
        store['faces'].append(faces + store['offset'])
        store['colors'].append(color)
        store['instances'].append(np.full(len(faces), instance_id, np.int32))
        store['offset'] += len(verts)

    @staticmethod
    def _new_store():
        return {'verts': [], 'faces': [], 'colors': [], 'instances': [], 'offset': 0,
                'triangles': 0, 'drawn': []}

    @staticmethod
    def _finish_store(store):
        if not store['verts']:
            empty = np.zeros((0, 3), np.float32)
            return {'verts': torch.zeros((0, 3)), 'faces': torch.zeros((0, 3), dtype=torch.long),
                    'colors': torch.zeros((0, 3)), 'normals': torch.zeros((0, 3)),
                    'instances': torch.zeros(0, dtype=torch.int32)}
        verts = np.concatenate(store['verts'])
        faces = np.concatenate(store['faces'])
        colors = np.concatenate(store['colors'])
        instances = np.concatenate(store['instances'])
        edge_a = verts[faces[:, 1]] - verts[faces[:, 0]]
        edge_b = verts[faces[:, 2]] - verts[faces[:, 0]]
        cross = np.cross(edge_a.astype(np.float64), edge_b.astype(np.float64))
        length = np.linalg.norm(cross, axis=1)
        # Degenerate (zero-area) triangles have no normal and no coverage: drop them so
        # every rendered pixel carries a genuine unit normal.
        keep = length > 1e-9
        normals = cross[keep] / length[keep][:, None]
        return {'verts': torch.as_tensor(verts), 'faces': torch.as_tensor(faces[keep].astype(np.int64)),
                'colors': torch.as_tensor(colors[keep]), 'normals': torch.as_tensor(normals.astype(np.float32)),
                'instances': torch.as_tensor(instances[keep])}

    def _resolve_background_glb(self):
        """Prefer the high-resolution textured appearance GLB declared by console_source
        when present in delivery_root or runtime/frontend_dist; fall back to aas_full_glb."""
        bg = self.cfg.get('background', {})
        fallback = Path(bg['aas_full_glb'])
        cs_path = bg.get('console_source')
        if cs_path and Path(cs_path).is_file():
            try:
                cs = json.loads(Path(cs_path).read_text(encoding='utf-8'))
                url = (cs.get('city') or {}).get('url', '')
                if url.startswith('/data/'):
                    deliv = Path(bg.get('delivery_root', '')) / bg.get('id', '') / 'geometry' / Path(url).name
                    if deliv.is_file():
                        return deliv
                    rt = (self.asset_root.parent / 'blockout_geometry_20261023'
                          / 'marble_aas_visual_20261024' / 'runtime' / 'frontend_dist' / url.lstrip('/'))
                    if rt.is_file():
                        return rt
            except Exception:
                pass
        return fallback

    def _build_static(self):
        store = self._new_store()
        background = self._resolve_background_glb()
        verts, faces, colors = self._mesh(background)
        yup_to_enu = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
        world = verts @ yup_to_enu[:3, :3].T
        self._append(store, world, faces, colors, BACKGROUND_ID)
        store['drawn'].append('<background>')
        store['triangles'] += len(faces)
        for entity_id, entity in sorted(self._entities().items()):
            if entity['cls'] not in _STATIC_CLASSES:
                continue
            if entity['cls'] == 'road':
                self._append_road(store, entity_id, entity)
            elif entity['cls'] in ('open_area', 'water'):
                self._append_patch(store, entity_id, entity)
            else:
                self._append_entity(store, entity_id, entity, None)
        return self._finish_store(store)

    def _append_road(self, store, entity_id, entity):
        fields = entity['fields']
        points = entity['geometry']['coords']
        if entity['geometry']['type'] == 'point' or len(points) < 2:
            return
        width = float(fields.get('road.width_m', 4.0))
        verts, faces = _ribbon(points, width, self.grid)
        color = ROAD_COLOR.get(str(fields.get('road.surface', '')).lower(), ROAD_COLOR['asphalt'])
        self._append(store, verts, faces, color, self._instance(entity_id))
        store['drawn'].append(entity_id)
        store['triangles'] += len(faces)

    def _append_patch(self, store, entity_id, entity):
        points = np.asarray(entity['geometry']['coords'], float)
        if len(points) > 3 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 3:
            return
        verts = []
        for x, y in points:
            e, n, u = _safe_grid_enu(self.grid, [float(x), float(y)], 0.0)
            verts.append([e, n, (self.water_u if entity['cls'] == 'water' and self.water_u is not None
                                 else u) + 0.03])
        faces = _triangulate(points)
        color = SURFACE_COLOR['water' if entity['cls'] == 'water' else 'open_area']
        self._append(store, np.asarray(verts, np.float32), faces, color, self._instance(entity_id))
        store['drawn'].append(entity_id)
        store['triangles'] += len(faces)

    def _append_entity(self, store, entity_id, entity, kind_hint, display_kind=None):
        """One entity as real GLB triangles, or a true-size semantic box/cylinder."""
        asset, row = self._asset_row(entity_id, entity, display_kind)
        enu, yaw, rotation, _ = self._pose(entity_id, entity)
        kind = kind_hint or row.get('asset_kind') or (
            entity['fields'].get('platform.kind') if entity['cls'] == 'platform' else entity['cls'])
        path = _resolve_asset(self.asset_root, asset['file']) if asset.get('file') else None
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = enu
        if path is not None and asset.get('runtime_static', {}).get('file'):
            static_path = _resolve_asset(self.asset_root, asset['runtime_static']['file'])
            path = static_path or path
        if path is not None:
            verts, faces, colors = self._mesh(path)
            from rescue_scenes.assets import normalization_parameters
            reference=self.assets.get(row.get('reference_asset_id'),asset)
            floating=(display_kind or kind) in ('boat','boat_loaded','human_in_water')
            translate,turn,scale=normalization_parameters(asset,reference,waterline=floating)
            # Match stage_models(): world * scale * heading * origin translation.
            # Reversing heading/translation displaced non-+X buildings by metres.
            scale_matrix=np.eye(4);scale_matrix[:3,:3]=np.diag(scale)
            chain=transform@scale_matrix@_homogeneous(rot_y(turn),np.zeros(3))@_homogeneous(np.eye(3),np.asarray(translate))
            if row.get('upstream_instance_id') and row.get('matrix_enu') and entity['cls'] in _STATIC_CLASSES:
                # Upstream placements are authored at the GLB origin. Recentring
                # them translates walls away from the navigation footprint.
                yup_to_enu = _homogeneous(rot_x(math.pi / 2), np.zeros(3))
                chain = np.asarray(row['matrix_enu'], float) @ yup_to_enu
            world = verts @ chain[:3, :3].T + chain[:3, 3]
            self._append(store, world, faces, colors, self._instance(entity_id))
        else:  # no delivered GLB: keep the target on screen with its true metric volume
            length, width, height = self._entity_extent(entity, row, kind)
            if kind == 'human':
                if display_kind in ('human_prone', 'human_supine'):
                    verts, faces = _box_mesh(1.75, 0.55, 0.42)   # a lying casualty's real volume
                else:
                    verts, faces = _cylinder_mesh(max(0.22, width * 0.45), height)
            else:
                verts, faces = _box_mesh(length, width, height)
            chain = transform @ _homogeneous(rot_y(_FORWARD_TURN.get('+x', 0.0)), np.zeros(3))
            world = verts @ chain[:3, :3].T + chain[:3, 3]
            color = SURFACE_COLOR.get(kind if kind in SURFACE_COLOR else 'building')
            self._append(store, world, faces, color, self._instance(entity_id))
        store['drawn'].append(entity_id)
        store['triangles'] += len(faces)
        box = self._entity_box(entity, row, enu, yaw, kind)
        if box is not None:
            self.entity_boxes[entity_id] = box

    def _entity_box(self, entity, row, enu, yaw, kind):
        """Volumetric OBB used to prove the camera never sits inside a solid."""
        if kind in ('road', 'open_area', 'water'):
            return None
        footprint = row.get('footprint_en')
        height = (row.get('dimensions_m') or DEFAULT_EXTENT.get(kind, (0, 0, 2.0)))[2]
        if footprint and len(footprint) >= 3:
            pts = np.asarray(footprint, float)
            lo, hi = pts.min(0), pts.max(0)
            return {'e': (float(lo[0]), float(hi[0])), 'n': (float(lo[1]), float(hi[1])),
                    'u': (float(enu[2]), float(enu[2]) + float(height))}
        radius = max(float((row.get('dimensions_m') or DEFAULT_EXTENT.get(kind, (1, 1, 2)))[0]), 0.3) / 2
        return {'e': (enu[0] - radius, enu[0] + radius), 'n': (enu[1] - radius, enu[1] + radius),
                'u': (enu[2], enu[2] + float(height))}

    def _build_dynamic(self):
        store = self._new_store()
        for entity_id, entity in sorted(self._entities().items()):
            if entity['cls'] in _STATIC_CLASSES or entity['cls'] == 'cue':
                continue
            fields = entity['fields']
            if entity['cls'] == 'obstacle':
                state = str(fields.get('obstacle.state', 'present'))
                if state == 'moved':
                    state = 'cleared'
                if state == 'cleared':
                    continue                       # engine.py: cleared debris is not drawn
                self._append_entity(store, entity_id, entity, None,
                                    'opening' if state == 'partly_cleared' else None)
            elif entity['cls'] == 'human':
                if fields.get('human.carrier_id'):
                    continue                       # drawn by the carrier's LOADED variant
                mobility = str(fields.get('human.mobility', 'mobile'))
                context = fields.get('human.location_context')
                display_kind = None
                if context in (self.display.get('human_models') or {}) or context == 'water':
                    display_kind = 'human_in_water'
                elif mobility == 'trapped':
                    display_kind = 'human_prone'
                elif mobility == 'immobile':
                    display_kind = 'human_supine'
                self._append_entity(store, entity_id, entity, 'human', display_kind)
            else:
                model = (self.display.get('agent_models') or {}).get(entity_id)
                if model in ('ambulance', 'boat', 'sled') and self._carries(entity_id):
                    model = model + '_loaded'
                self._append_entity(store, entity_id, entity, model or None, model)
        return self._finish_store(store)

    def _carries(self, entity_id):
        for other in self._entities().values():
            if other['cls'] == 'human' and other['fields'].get('human.carrier_id') == entity_id:
                return True
        return False

    def _dynamic_signature(self):
        parts = []
        for entity_id, entity in sorted(self._entities().items()):
            if entity['cls'] in _STATIC_CLASSES or entity['cls'] == 'cue':
                continue
            fields = entity['fields']
            parts.append((entity_id, entity['cls'], tuple(np.round(_position_xy(entity), 3)),
                          str(fields.get('obstacle.state')), str(fields.get('human.mobility')),
                          str(fields.get('human.carrier_id')), str(fields.get('human.location_context'))))
        return tuple(parts)

    def sync_dynamic(self, world=None, force=False):
        """Re-tessellate moving entities only when their pose/state actually changed."""
        if world is not None:
            self.world = world
        key = self._dynamic_signature()
        if force or key != self._dynamic_key:
            self._dynamic = self._build_dynamic()
            self._dynamic_key = key
        return self

    # -- camera ------------------------------------------------------------------------
    def intrinsics_matrix(self, intrinsics):
        """(fx, fy, cx, cy) at render resolution from a K dict / 3x3 matrix / None."""
        if intrinsics is None:
            fx = (self.width / 2.0) / math.tan(math.radians(70.0) / 2.0)
            return fx, fx, self.width / 2.0, self.height / 2.0
        if isinstance(intrinsics, dict):
            if 'fx' in intrinsics:
                fx, fy = float(intrinsics['fx']), float(intrinsics.get('fy', intrinsics['fx']))
                cx = float(intrinsics.get('cx', 0.5 * float(intrinsics.get('width', self.width))))
                cy = float(intrinsics.get('cy', 0.5 * float(intrinsics.get('height', self.height))))
            else:
                hfov = float(intrinsics.get('hfov_deg', 70.0))
                fx = fy = (self.width / 2.0) / math.tan(math.radians(hfov) / 2.0)
                cx, cy = self.width / 2.0, self.height / 2.0
            source = float(intrinsics.get('width', self.width))
            scale = self.width / source if source > 0 else 1.0   # uniform: no aspect distortion
            return fx * scale, fy * scale, cx * scale, cy * scale
        k = np.asarray(intrinsics, float)
        return float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])

    def camera(self, cam_pose_enu, cam_rpy_deg, intrinsics=None):
        pose = np.asarray(cam_pose_enu, float).reshape(3)
        rpy = np.asarray(cam_rpy_deg, float).reshape(3)
        r_wc = camera_rotation(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        r_cw = r_wc.T
        t_cw = np.eye(4)
        t_cw[:3, :3] = r_cw
        t_cw[:3, 3] = -r_cw @ pose
        k = self.intrinsics_matrix(intrinsics)
        return {'pose': pose, 'rpy': rpy, 'R_w_from_c': r_wc, 'R_c_from_w': r_cw,
                'T_c_from_w': t_cw, 'K': k}

    # -- rasterisation -----------------------------------------------------------------
    # -- rasterisation: near clip, viewport clip, sub-pixel cull, bbox refinement ------
    def _project_tris(self, tri, k):
        """Screen coordinates and reciprocal optical depth of (N,3,3) camera triangles."""
        fx, fy, cx, cy = k
        inverse_z = 1.0 / tri[:, :, 2]
        return fx * tri[:, :, 0] * inverse_z + cx, fy * tri[:, :, 1] * inverse_z + cy, inverse_z

    @staticmethod
    def _clip_plane(poly, owner, coeff, bound, keep_greater):
        """Sutherland-Hodgman clip of a triangle soup against one linear plane.

        The plane is ``coeff . p >= bound`` (or ``<=``).  Every attribute is affine
        on the triangle, so interpolated vertices keep coverage and perspective
        depth exact.  Clipping in 3-D camera space before the perspective divide is
        what keeps the screen coordinates bounded (a near-plane vertex otherwise
        projects to +-1e5 px and destroys float32 barycentrics).
        """
        device, dtype = poly.device, poly.dtype
        weights = torch.as_tensor(coeff, dtype=dtype, device=device)
        value = poly @ weights
        inside = value >= bound if keep_greater else value <= bound
        count = inside.sum(1)
        pieces = [poly[count == 3]]
        owners = [owner[count == 3]]
        mixed = (count > 0) & (count < 3)
        if bool(mixed.any()):
            sub, flags, val = poly[mixed], inside[mixed], value[mixed]
            m = len(sub)
            slots = torch.zeros((m, 4, poly.shape[2]), dtype=dtype, device=device)
            filled = torch.zeros(m, dtype=torch.long, device=device)
            index = torch.arange(m, device=device)
            for i in range(3):
                j = (i + 1) % 3
                take = flags[:, i]
                slots[index[take], filled[take].clamp(max=3)] = sub[take, i]
                filled = filled + take.long()
                crossing = flags[:, i] != flags[:, j]
                if bool(crossing.any()):
                    denom = torch.where(crossing, val[:, j] - val[:, i],
                                        torch.ones(m, dtype=dtype, device=device))
                    t = (bound - val[:, i]) / denom
                    point = sub[:, i] + (sub[:, j] - sub[:, i]) * t[:, None]
                    slots[index[crossing], filled[crossing].clamp(max=3)] = point[crossing]
                    filled = filled + crossing.long()
            both = torch.stack([slots[:, [0, 1, 2]], slots[:, [0, 2, 3]]], 1)
            take = torch.stack([torch.ones(m, dtype=torch.bool, device=device), filled == 4], 1)
            pieces += [both[take]]
            owners += [owner[mixed].unsqueeze(1).expand(-1, 2)[take]]
        return torch.cat(pieces), torch.cat(owners)

    def _refine(self, poly, owner, cap=768, levels=6):
        """Split triangles whose pixel bbox exceeds ``cap`` (all attributes affine)."""
        final_poly, final_owner = [], []
        for _ in range(levels):
            u, v = poly[:, :, 0], poly[:, :, 1]
            x0 = torch.floor(u.min(1).values - 0.5).clamp_(0, self.width - 1).long()
            x1 = torch.ceil(u.max(1).values - 0.5).clamp_(0, self.width - 1).long()
            y0 = torch.floor(v.min(1).values - 0.5).clamp_(0, self.height - 1).long()
            y1 = torch.ceil(v.max(1).values - 0.5).clamp_(0, self.height - 1).long()
            big = ((x1 - x0 + 1) * (y1 - y0 + 1)) > cap
            if not bool(big.any()):
                break
            final_poly.append(poly[~big])
            final_owner.append(owner[~big])
            a, b, c = poly[big, 0], poly[big, 1], poly[big, 2]
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            poly = torch.stack([torch.stack([a, ab, ca], 1), torch.stack([ab, b, bc], 1),
                                torch.stack([ca, bc, c], 1), torch.stack([ab, bc, ca], 1)], 1).reshape(-1, 3, 3)
            owner = owner[big].repeat_interleave(4)
        final_poly.append(poly)
        final_owner.append(owner)
        return torch.cat(final_poly), torch.cat(final_owner)

    def _rasterize(self, points_cam, faces, k):
        """Batched triangle sweep; returns (face slot, linear depth, near-clipped, samples)."""
        height, width = self.height, self.width
        empty = (torch.zeros((height, width), dtype=torch.long, device=self.device),
                 torch.zeros((height, width), device=self.device))
        if faces.numel() == 0:
            return (*empty, 0, 0)
        fx, fy, cx, cy = k
        tri = points_cam[faces]
        near_clipped = int((tri[:, :, 2] < self.near).any(1).sum())
        owner = torch.arange(len(tri), device=self.device)
        # Near plane then the four side planes, all in 3-D: the projected coordinates
        # stay inside the image, so the bbox sweep is exact and numerically stable.
        planes = (((0.0, 0.0, 1.0), self.near, True),
                  ((fx, 0.0, cx), 0.0, True), ((fx, 0.0, cx - width), 0.0, False),
                  ((0.0, fy, cy), 0.0, True), ((0.0, fy, cy - height), 0.0, False))
        for coeff, bound, greater in planes:
            tri, owner = self._clip_plane(tri, owner, coeff, bound, greater)
            if tri.numel() == 0:
                return (*empty, near_clipped, 0)
        u, v, inverse_z = self._project_tris(tri, k)
        poly = torch.stack([u, v, inverse_z], 2)
        det = (poly[:, 1, 0] - poly[:, 0, 0]) * (poly[:, 2, 1] - poly[:, 0, 1]) \
            - (poly[:, 2, 0] - poly[:, 0, 0]) * (poly[:, 1, 1] - poly[:, 0, 1])
        buffer = torch.full((height * width,), int(_KEY_EMPTY), dtype=torch.int64, device=self.device)
        total_samples = 0
        # Triangles smaller than one pixel are *not* dropped: a detailed casualty or
        # debris mesh has millimetre triangles, and dropping them would erase the task
        # target at range.  They are splatted at their centroid (exact at that point),
        # which is the standard sub-pixel coverage fallback.
        tiny = det.abs() < 0.02
        if bool(tiny.any()):
            sub, sub_owner = poly[tiny], owner[tiny]
            centre_u = sub[:, :, 0].mean(1).floor().clamp_(0, width - 1).long()
            centre_v = sub[:, :, 1].mean(1).floor().clamp_(0, height - 1).long()
            depth = 1.0 / sub[:, :, 2].mean(1)
            quantised = (depth * _DEPTH_SCALE).clamp_(min=1).to(torch.int64)
            key = (quantised << _SLOT_BITS) | sub_owner.to(torch.int64)
            buffer.scatter_reduce_(0, centre_v * width + centre_u, key, reduce='amin', include_self=True)
            total_samples += len(sub)
        poly, owner = poly[~tiny], owner[~tiny]
        poly, owner = (self._refine(poly, owner) if poly.numel()
                       else (poly[:0], owner[:0]))
        u, v, inverse_z = poly[:, :, 0], poly[:, :, 1], poly[:, :, 2]
        x0 = torch.floor(u.min(1).values - 0.5).clamp_(0, width - 1).long()
        x1 = torch.ceil(u.max(1).values - 0.5).clamp_(0, width - 1).long()
        y0 = torch.floor(v.min(1).values - 0.5).clamp_(0, height - 1).long()
        y1 = torch.ceil(v.max(1).values - 0.5).clamp_(0, height - 1).long()
        bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
        areas = bw * bh
        cumulative = torch.cumsum(areas, 0)
        starts = cumulative - areas
        budget, start, count = 1_500_000, 0, len(areas)
        while start < count:
            end = int(torch.searchsorted(starts, starts[start] + budget, right=True).item())
            end = min(max(end, start + 1), count)
            piece = torch.repeat_interleave(torch.arange(start, end, device=self.device), areas[start:end])
            # Block offsets are relative to the chunk, not to the whole frame.
            offset = torch.repeat_interleave(starts[start:end] - starts[start], areas[start:end])
            local = torch.arange(piece.numel(), device=self.device) - offset
            px = (x0[piece] + local % bw[piece]).to(torch.float32) + 0.5
            py = (y0[piece] + local // bw[piece]).to(torch.float32) + 0.5
            u0, u1, u2 = u[piece, 0], u[piece, 1], u[piece, 2]
            v0, v1, v2 = v[piece, 0], v[piece, 1], v[piece, 2]
            area2 = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
            # Dividing by the *signed* double area already normalises the winding:
            # numerator and denominator share the sign, so inside is b >= 0 for both.
            safe = torch.where(area2.abs() < 1e-12, torch.full_like(area2, 1e-12), area2)
            b0 = ((u1 - px) * (v2 - py) - (u2 - px) * (v1 - py)) / safe
            b1 = ((u2 - px) * (v0 - py) - (u0 - px) * (v2 - py)) / safe
            b2 = 1.0 - b0 - b1
            hit = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
            start = end
            if not bool(hit.any()):
                continue
            piece, b0, b1, b2, px, py = piece[hit], b0[hit], b1[hit], b2[hit], px[hit], py[hit]
            depth = 1.0 / (b0 * inverse_z[piece, 0] + b1 * inverse_z[piece, 1] + b2 * inverse_z[piece, 2])
            quantised = (depth * _DEPTH_SCALE).clamp_(min=1).to(torch.int64)
            key = (quantised << _SLOT_BITS) | owner[piece].to(torch.int64)
            pixel = py.to(torch.int64) * width + px.to(torch.int64)
            buffer.scatter_reduce_(0, pixel, key, reduce='amin', include_self=True)
            total_samples += piece.numel()
        hit = buffer < _KEY_EMPTY
        slot = torch.where(hit, buffer & ((1 << _SLOT_BITS) - 1), torch.full_like(buffer, -1))
        depth = torch.where(hit, (buffer >> _SLOT_BITS).to(torch.float32) / _DEPTH_SCALE,
                            torch.zeros_like(buffer, dtype=torch.float32))
        return slot.reshape(height, width), depth.reshape(height, width), near_clipped, total_samples

    # -- public render -----------------------------------------------------------------
    def render_view(self, cam_pose_enu, cam_rpy_deg, intrinsics=None, *, sync=True, stats=None):
        """Synchronised RGB / depth / normals / instance ids for one camera pose."""
        if sync:
            self.sync_dynamic()
        cam = self.camera(cam_pose_enu, cam_rpy_deg, intrinsics)
        r_cw = torch.as_tensor(cam['R_c_from_w'], dtype=torch.float32, device=self.device)
        centre = torch.as_tensor(cam['pose'], dtype=torch.float32, device=self.device)
        static, dynamic = self._static, self._dynamic
        verts = torch.cat([static['verts'], dynamic['verts']]).to(self.device)
        faces = torch.cat([static['faces'], dynamic['faces'] + len(static['verts'])]).to(self.device)
        colors = torch.cat([static['colors'], dynamic['colors']]).to(self.device)
        normals = torch.cat([static['normals'], dynamic['normals']]).to(self.device)
        instances = torch.cat([static['instances'], dynamic['instances']]).to(self.device)
        points_cam = (verts - centre) @ r_cw.T
        k = cam['K']
        slot, depth, near_clipped, samples = self._rasterize(points_cam, faces, k)
        height, width = self.height, self.width
        rgb = np.empty((height, width, 3), np.float32)
        rgb[:] = SKY_HORIZON
        gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        sky = (SKY_ZENITH[None, :] * (1.0 - gradient) + SKY_HORIZON[None, :] * gradient)
        rgb[:] = sky[:, None, :]
        normal_cam = np.zeros((height, width, 3), np.float32)
        normal_world = np.zeros((height, width, 3), np.float32)
        instance = np.full((height, width), SKY_ID, np.uint32)
        valid = depth > 0
        if bool(valid.any()):
            face = slot[valid]
            depth_v = depth[valid]
            colour = colors[face]
            normal_w = normals[face]
            normal_c = normal_w @ r_cw.T
            flip = normal_c[:, 2] > 0
            normal_c[flip] *= -1.0
            normal_w[flip] *= -1.0
            rows, cols = torch.nonzero(valid, as_tuple=True)
            px = cols.to(torch.float32) + 0.5
            py = rows.to(torch.float32) + 0.5
            fx, fy, cx, cy = k
            point_cam = torch.stack([(px - cx) / fx * depth_v, (py - cy) / fy * depth_v, depth_v], 1)
            point_world = point_cam @ r_cw + centre
            ground = self.terrain_height(point_world[:, 0], point_world[:, 1])
            light = torch.as_tensor(self.sun, dtype=torch.float32, device=self.device)
            lambert = torch.clamp(normal_w @ light, min=0.0)
            hemisphere = 0.15 + 0.19 * (0.5 + 0.5 * normal_w[:, 2])
            tint = torch.tensor([1.0, 0.97, 0.92], dtype=torch.float32, device=self.device)
            shade = (lambert * SUN_RADIANCE)[:, None] * tint[None, :] + hemisphere[:, None]
            height_above = torch.clamp(point_world[:, 2] - ground, min=0.0)
            contact = 0.80 + 0.20 * torch.clamp(height_above / 0.8, max=1.0)
            is_solid = instances[face] != BACKGROUND_ID
            shade = shade * torch.where(is_solid, contact, torch.ones_like(contact))[:, None]
            shaded = colour * shade
            fog = 1.0 - torch.exp(-(depth_v / self.fog_scale_m) ** 1.5)
            shaded = shaded * (1.0 - fog)[:, None] + torch.as_tensor(SKY_HORIZON, device=self.device) * fog[:, None]
            shaded = torch.clamp(shaded, 0.0, 1.0) ** (1.0 / 2.2)
            rgb[rows.cpu().numpy(), cols.cpu().numpy()] = shaded.cpu().numpy()
            normal_cam[rows.cpu().numpy(), cols.cpu().numpy()] = normal_c.cpu().numpy()
            normal_world[rows.cpu().numpy(), cols.cpu().numpy()] = normal_w.cpu().numpy()
            instance[rows.cpu().numpy(), cols.cpu().numpy()] = instances[face].cpu().numpy().astype(np.uint32)
        depth_out = depth.cpu().numpy().astype(np.float32)
        result = {
            'rgb': np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8),
            'depth': np.where(valid.cpu().numpy(), depth_out, 0.0).astype(np.float32),
            'normal_cam': normal_cam, 'normal_world': normal_world,
            'instance_id': instance, 'valid_mask': valid.cpu().numpy(),
            'K': cam['K'], 'T_c_from_w': cam['T_c_from_w'], 'R_c_from_w': cam['R_c_from_w'],
            'R_w_from_c': cam['R_w_from_c'], 'cam_pose_enu': cam['pose'].tolist(),
            'cam_rpy_deg': cam['rpy'].tolist(), 'width': width, 'height': height,
            'cam_inside_entity': self.cam_inside(cam['pose']),
            'entity_id_to_id': dict(self.entity_id_to_id), 'id_to_entity_id': dict(self.id_to_entity_id),
            'stats': {'triangles': int(len(faces)), 'drawn_entities': len(self.entity_id_to_id),
                      'near_clipped': near_clipped, 'samples': int(samples),
                      'valid_ratio': float(valid.float().mean().item()),
                      'device': str(self.device)},
        }
        if stats is not None:
            stats.update(result['stats'])
        self.last_stats = result['stats']
        return result

    def cam_inside(self, pose):
        """True when the optical centre lies inside a solid entity's metric box."""
        for entity_id, box in self.entity_boxes.items():
            if box['e'][0] <= pose[0] <= box['e'][1] and box['n'][0] <= pose[1] <= box['n'][1] \
                    and box['u'][0] - 0.05 <= pose[2] <= box['u'][1]:
                return entity_id
        return None

    def render_pair(self, uav_pose, uav_rpy, ground_pose, ground_rpy, intrinsics,
                    target_entity_ids=(), **kwargs):
        """Render the aerial and ground cameras of one section and its co-visibility.

        ``intrinsics`` is one calibration shared by both cameras, or a ``(K_uav, K_gnd)``
        pair when the two sensors really differ (contract ③ keeps K_i and K_e separate).
        """
        if isinstance(intrinsics, (tuple, list)) and len(intrinsics) == 2:
            uav_intrinsics, ground_intrinsics = intrinsics
        else:
            uav_intrinsics = ground_intrinsics = intrinsics
        uav = self.render_view(uav_pose, uav_rpy, uav_intrinsics, **kwargs)
        ground = self.render_view(ground_pose, ground_rpy, ground_intrinsics, **kwargs)
        covis = self.compute_covisibility_and_blind_spots(uav, ground, target_entity_ids)
        return {'uav': uav, 'ground': ground, 'covisibility': covis,
                'target_entity_ids': list(target_entity_ids)}

    # -- co-visibility -----------------------------------------------------------------
    def unproject(self, render):
        """Valid pixels of a render as world points + normals (torch, device tensors)."""
        depth = torch.as_tensor(render['depth'], device=self.device)
        valid = torch.as_tensor(render['valid_mask'], device=self.device) & (depth > 0)
        rows, cols = torch.nonzero(valid, as_tuple=True)
        z = depth[rows, cols]
        fx, fy, cx, cy = render['K']
        point_cam = torch.stack([(cols.to(torch.float32) + 0.5 - cx) / fx * z,
                                 (rows.to(torch.float32) + 0.5 - cy) / fy * z, z], 1)
        r_cw = torch.as_tensor(render['R_c_from_w'], dtype=torch.float32, device=self.device)
        centre = torch.as_tensor(render['cam_pose_enu'], dtype=torch.float32, device=self.device)
        return point_cam @ r_cw + centre, rows * render['width'] + cols, z

    def project(self, points_world, render):
        """Project world points into a render; returns (u, v, z_c, inside) tensors."""
        r_cw = torch.as_tensor(render['R_c_from_w'], dtype=torch.float32, device=self.device)
        centre = torch.as_tensor(render['cam_pose_enu'], dtype=torch.float32, device=self.device)
        point_cam = (points_world - centre) @ r_cw.T
        z = point_cam[:, 2]
        fx, fy, cx, cy = render['K']
        safe = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
        u = fx * point_cam[:, 0] / safe + cx
        v = fy * point_cam[:, 1] / safe + cy
        inside = (z > self.near) & (u >= 0) & (u < render['width']) & (v >= 0) & (v < render['height'])
        return u, v, z, inside

    def compute_covisibility_and_blind_spots(self, uav_render, ground_render, target_entity_ids):
        """True 3-D same-surface co-visibility and per-target blind-spot analysis.

        A UAV pixel is co-visible when its back-projected world point lands on the
        ground camera's image, the two linear depths agree within
        ``0.15 + 0.03 Z`` and the world normals agree (``n_i . n_e >= 0.7``), i.e. the
        same physical surface is observed, not merely the same image region.
        """
        height, width = self.height, self.width
        uav_valid = torch.as_tensor(uav_render['valid_mask'], device=self.device)
        ground_valid = torch.as_tensor(ground_render['valid_mask'], device=self.device)
        ground_depth = torch.as_tensor(ground_render['depth'], device=self.device)
        points, uav_flat, _ = self.unproject(uav_render)
        rows, cols = uav_flat // width, uav_flat % width
        u, v, z_proj, inside = self.project(points, ground_render)
        covis_mask = np.zeros((height, width), bool)
        blind_mask = np.asarray(uav_render['valid_mask'], bool).copy()
        projected = int(inside.sum().item())
        overlap_i = overlap_e = 0.0
        if projected:
            uu = torch.clamp(u.long(), 0, ground_render['width'] - 1)
            vv = torch.clamp(v.long(), 0, ground_render['height'] - 1)
            gnd_flat = vv * ground_render['width'] + uu
            z_ground = ground_depth.reshape(-1)[gnd_flat]
            tolerance = 0.15 + 0.03 * z_proj
            same_surface = (z_ground > 0) & ((z_ground - z_proj).abs() <= tolerance)
            n_uav = torch.as_tensor(uav_render['normal_world'][rows.cpu().numpy(),
                                                               cols.cpu().numpy()], device=self.device)
            n_ground = torch.as_tensor(ground_render['normal_world'].reshape(-1, 3)[gnd_flat.cpu().numpy()],
                                       device=self.device)
            aligned = (n_uav * n_ground).sum(1) >= 0.7
            hit = inside & same_surface & aligned
            index = (rows[hit] * width + cols[hit])
            covis_mask.reshape(-1)[index.cpu().numpy()] = True
            blind_mask[covis_mask] = False
            blind_mask.reshape(-1)[(rows[~hit] * width + cols[~hit]).cpu().numpy()] = True
            uav_valid_count = max(int(uav_valid.sum().item()), 1)
            ground_valid_count = max(int(ground_valid.sum().item()), 1)
            overlap_i = float(hit.sum().item()) / uav_valid_count
            touched = torch.zeros(height * width, dtype=torch.bool, device=self.device)
            touched[gnd_flat[hit]] = True
            overlap_e = float(touched.sum().item()) / ground_valid_count
        per_entity = {}
        uav_ids = torch.as_tensor(uav_render['instance_id'].reshape(-1), device=self.device)
        for entity_id in target_entity_ids:
            index = uav_render['entity_id_to_id'].get(entity_id)
            if index is None:
                per_entity[entity_id] = {'present': False, 'instance_id': None}
                continue
            mask_uav = (uav_ids == index) & uav_valid.reshape(-1)
            pixels = int(mask_uav.sum().item())
            tokens = self.post_merger_token_grid(mask_uav.reshape(height, width).cpu().numpy())
            blind = int((mask_uav & torch.as_tensor(blind_mask.reshape(-1), device=self.device)).sum().item())
            ground_pixels = int((torch.as_tensor(ground_render['instance_id'].reshape(-1), device=self.device)
                                 == index).sum().item())
            entry = {'present': True, 'instance_id': index, 'uav_pixels': pixels,
                     'uav_pixel_fraction': pixels / max(int(uav_valid.sum().item()), 1),
                     'uav_tokens': int(tokens.sum()), 'uav_token_total': int(tokens.size),
                     'uav_token_ratio': float(tokens.sum()) / float(tokens.size),
                     'uav_covis_pixels': int((mask_uav & torch.as_tensor(covis_mask.reshape(-1),
                                                                         device=self.device)).sum().item()),
                     'uav_blind_pixels': blind, 'ground_pixels': ground_pixels}
            if pixels:
                points_e = points[mask_uav[uav_flat]]
                ue, ve, ze, inside_e = self.project(points_e, ground_render)
                if int(inside_e.sum().item()):
                    uu = torch.clamp(ue[inside_e].long(), 0, ground_render['width'] - 1)
                    vv = torch.clamp(ve[inside_e].long(), 0, ground_render['height'] - 1)
                    ent_gnd_flat = vv * ground_render['width'] + uu
                    z_ground = ground_depth.reshape(-1)[ent_gnd_flat]
                    tolerance = 0.15 + 0.03 * ze[inside_e]
                    visible = (z_ground > 0) & ((z_ground - ze[inside_e]).abs() <= tolerance)
                    occluded = (z_ground > 0) & (z_ground < ze[inside_e] - 0.5)
                    entry.update(projected_pixels=int(inside_e.sum().item()),
                                 ground_visible_pixels=int(visible.sum().item()),
                                 ground_occluded_pixels=int(occluded.sum().item()),
                                 ground_visible_fraction=float(visible.float().mean().item()))
                    entry['ground_state'] = ('visible' if float(visible.float().mean()) >= 0.5 else
                                             'occluded' if float(occluded.float().mean()) >= 0.5 else
                                             'partial')
                else:
                    entry.update(projected_pixels=0, ground_visible_pixels=0,
                                 ground_occluded_pixels=0, ground_visible_fraction=0.0,
                                 ground_state='out_of_view')
            else:
                entry.update(projected_pixels=0, ground_visible_pixels=0, ground_occluded_pixels=0,
                             ground_visible_fraction=0.0, ground_state='absent')
            per_entity[entity_id] = entry
        return {'covis_mask_uav': covis_mask, 'blind_mask_uav': blind_mask,
                'overlap_uav': overlap_i, 'overlap_ground': overlap_e,
                'covis_pixels': int(covis_mask.sum()), 'blind_pixels': int(blind_mask.sum()),
                'uav_valid_pixels': int(uav_valid.sum().item()),
                'ground_valid_pixels': int(ground_valid.sum().item()),
                'projected_pixels': projected, 'per_entity': per_entity}

    def post_merger_token_grid(self, mask):
        """(12, 20) post-merger token occupancy of a pixel mask (32x32 px per token)."""
        rows = self.height // PATCH_MERGE
        cols = self.width // PATCH_MERGE
        grid = np.asarray(mask, bool).reshape(rows, PATCH_MERGE, cols, PATCH_MERGE).any(axis=(1, 3))
        return grid
