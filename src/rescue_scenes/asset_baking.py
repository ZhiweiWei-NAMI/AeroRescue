"""Strict embedded GLB validation and measured runtime asset statistics."""
from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from . import glb_runtime as runtime

CATEGORIES = {
    'AGT': 'AGT_agents', 'BLD': 'BLD_buildings', 'ENV': 'ENV_environment',
    'HUM': 'HUM_humans', 'OBS': 'OBS_obstacles', 'RSC': 'RSC_equipment',
    'TER': 'TER_terrain',
}
PIPELINE = ['optimize', 'merge_by_material', 'opaque_blend', 'compact_textures',
            'semantic_textures', 'quantize_geometry']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decode_glb(raw):
    """Reject truncated chunks, bad versions, misalignment and invalid GLB buffers."""
    if len(raw) < 28:
        raise ValueError('truncated GLB')
    magic, version, length = struct.unpack_from('<4sII', raw)
    if (magic, version, length) != (b'glTF', 2, len(raw)):
        raise ValueError('invalid GLB magic/version/declared length')
    offset, chunks = 12, []
    while offset < length:
        if offset + 8 > length:
            raise ValueError('truncated chunk header')
        size, kind = struct.unpack_from('<II', raw, offset)
        if size % 4 or offset + 8 + size > length:
            raise ValueError('unaligned or truncated chunk')
        chunks.append((kind, raw[offset + 8:offset + 8 + size]))
        offset += 8 + size
    if len(chunks) != 2 or [x[0] for x in chunks] != [0x4E4F534A, 0x004E4942]:
        raise ValueError('expected JSON followed by BIN')
    doc = json.loads(chunks[0][1])
    if doc.get('asset', {}).get('version') != '2.0':
        raise ValueError('not glTF 2.0')
    binary = chunks[1][1]
    buffers = doc.get('buffers', [])
    if len(buffers) != 1 or not 0 <= len(binary) - buffers[0]['byteLength'] <= 3:
        raise ValueError('invalid embedded buffer length')
    return doc, binary


def validate(doc, binary, *, max_px=None, decode_geometry=True):
    """Validate image formats, array references/ranges and default-scene topology."""
    for kind in ('buffers', 'images'):
        if any('uri' in item for item in doc.get(kind, [])):
            raise ValueError(f'unresolved {kind} URI')
    views = doc.get('bufferViews', [])
    for view in views:
        if view.get('buffer', 0) != 0 or view.get('byteOffset', 0) < 0 or view['byteLength'] < 0:
            raise ValueError('invalid buffer view')
        if view.get('byteOffset', 0) + view['byteLength'] > doc['buffers'][0]['byteLength']:
            raise ValueError('buffer view exceeds embedded buffer')
    widths = dict(runtime._WIDTH, MAT2=4, MAT3=9, MAT4=16)
    for acc in doc.get('accessors', []):
        if 'bufferView' not in acc:
            continue  # sparse/viewless accessors are supported by glTF, preserved by baker
        view = views[acc['bufferView']]
        item = np.dtype(runtime._COMPONENTS[acc['componentType']]).itemsize * widths[acc['type']]
        stride = view.get('byteStride', item)
        needed = acc.get('byteOffset', 0) + max(0, acc['count'] - 1) * stride + (item if acc['count'] else 0)
        if stride < item or needed > view['byteLength']:
            raise ValueError('accessor exceeds buffer view')
    dimensions = []
    for image in doc.get('images', []):
        view = views[image['bufferView']]
        start = view.get('byteOffset', 0)
        payload = binary[start:start + view['byteLength']]
        with Image.open(io.BytesIO(payload)) as picture:
            if Image.MIME.get(picture.format) != image.get('mimeType'):
                raise ValueError('image MIME does not match actual payload')
            if max_px is not None and max(picture.size) > max_px:
                raise ValueError('texture exceeds size budget')
            dimensions.append(list(picture.size))
            picture.verify()
    for mesh in doc.get('meshes', []):
        for prim in mesh['primitives']:
            attrs = prim['attributes']
            count = doc['accessors'][attrs['POSITION']]['count']
            if any(doc['accessors'][i]['count'] != count for i in attrs.values()):
                raise ValueError('mismatched vertex attribute count')
            if decode_geometry and 'indices' in prim:
                indices = runtime._read_accessor(doc, binary, prim['indices'])
                if indices.size and indices.max() >= count:
                    raise ValueError('out-of-range primitive index')
    scene = doc.get('scenes', [])[doc.get('scene', 0)]
    nodes = doc.get('nodes', [])
    def visit(index, ancestors):
        if index in ancestors or index < 0 or index >= len(nodes):
            raise ValueError('cyclic/invalid scene node')
        node = nodes[index]
        if 'mesh' in node and not 0 <= node['mesh'] < len(doc.get('meshes', [])):
            raise ValueError('invalid mesh index')
        for child in node.get('children', []):
            visit(child, ancestors | {index})
    for index in scene.get('nodes', []):
        visit(index, set())
    return {'valid': True, 'external_uris': 0, 'image_dimensions': dimensions}


def measure(doc, binary, file_bytes):
    """Count default-scene draw instances, not merely unique mesh definitions."""
    nodes = doc.get('nodes', [])
    scene = doc.get('scenes', [])[doc.get('scene', 0)]
    stack = [(i, np.eye(4)) for i in scene.get('nodes', [])]
    draws = triangles = vertices = quantized = total_positions = 0
    lows, highs = [], []
    while stack:
        index, parent = stack.pop()
        node = nodes[index]
        world = parent @ runtime._local_matrix(node)
        stack.extend((i, world) for i in node.get('children', []))
        if 'mesh' not in node:
            continue
        for prim in doc['meshes'][node['mesh']]['primitives']:
            draws += 1
            acc = doc['accessors'][prim['attributes']['POSITION']]
            count = acc['count']; vertices += count; total_positions += 1
            quantized += acc['componentType'] != 5126
            n = doc['accessors'][prim['indices']]['count'] if 'indices' in prim else count
            if prim.get('mode', 4) == 4:
                triangles += n // 3
            if 'sparse' not in acc and 'bufferView' in acc:
                pos = runtime._read_accessor(doc, binary, prim['attributes']['POSITION'])
                xyz = pos @ world[:3, :3].T + world[:3, 3]
                if xyz.size and not np.isfinite(xyz).all():
                    raise ValueError('non-finite vertex positions')
                if xyz.size:
                    lows.append(xyz.min(axis=0)); highs.append(xyz.max(axis=0))
    texture_bytes = sum(doc['bufferViews'][i]['byteLength'] for i in {x['bufferView'] for x in doc.get('images', [])})
    decoded_rgba = 0; max_texture = 0
    for image in doc.get('images', []):
        view = doc['bufferViews'][image['bufferView']]; start = view.get('byteOffset', 0)
        with Image.open(io.BytesIO(binary[start:start + view['byteLength']])) as picture:
            decoded_rgba += picture.width * picture.height * 4
            max_texture = max(max_texture, *picture.size)
    return {'file_bytes': file_bytes, 'binary_bytes': len(binary), 'texture_bytes': texture_bytes,
            'decoded_texture_rgba8_bytes': decoded_rgba, 'max_texture_dimension': max_texture,
            'primitives': draws, 'triangles': triangles, 'vertices': vertices,
            'position_accessors': total_positions, 'quantized_position_accessors': quantized,
            'bbox_min': np.min(lows, axis=0).tolist() if lows else None,
            'bbox_max': np.max(highs, axis=0).tolist() if highs else None}


def world_positions(doc, binary):
    """Collect ordered world vertices by the same material/attribute merge groups."""
    grouped = {}
    stack = [(i, np.eye(4)) for i in doc['scenes'][doc.get('scene', 0)].get('nodes', [])]
    while stack:
        index, parent = stack.pop()
        node = doc['nodes'][index]; world = parent @ runtime._local_matrix(node)
        stack.extend((i, world) for i in node.get('children', []))
        if 'mesh' not in node: continue
        for primitive in doc['meshes'][node['mesh']]['primitives']:
            attributes = primitive['attributes']
            accessor = doc['accessors'][attributes['POSITION']]
            if 'sparse' in accessor or 'bufferView' not in accessor:
                return None
            key = (primitive.get('material'), tuple(sorted((name, doc['accessors'][acc]['type']) for name, acc in attributes.items())))
            position = runtime._read_accessor(doc, binary, attributes['POSITION'])
            grouped.setdefault(key, []).append(position @ world[:3, :3].T + world[:3, 3])
    return {key: np.concatenate(parts) for key, parts in grouped.items()}


def bake(source, output):
    raw = Path(source).read_bytes()
    doc, binary = decode_glb(raw)
    validate(doc, binary)
    before = measure(doc, binary, len(raw))
    original_positions = world_positions(doc, binary)
    records = {}
    for name in PIPELINE:
        options = {'max_px': 512} if name == 'optimize' else {'max_px': 512, 'quality': 85} if name == 'semantic_textures' else {}
        doc, binary, records[name] = getattr(runtime, name)(doc, binary, **options)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.glb.tmp')
    runtime.write_glb(temporary, doc, binary)
    baked_raw = temporary.read_bytes()
    doc, binary = decode_glb(baked_raw)
    audit = validate(doc, binary, max_px=512)
    after = measure(doc, binary, len(baked_raw))
    if before['triangles'] != after['triangles']:
        raise ValueError('baking changed default-scene triangle count')
    if before['bbox_min'] is not None and after['bbox_min'] is not None:
        error = float(np.max(np.abs(np.array([before['bbox_min'], before['bbox_max']]) - np.array([after['bbox_min'], after['bbox_max']]))))
        tolerance = max(1e-5, float(np.max(np.subtract(before['bbox_max'], before['bbox_min']))) / 32767 * 1.1)
        if error > tolerance:
            raise ValueError(f'baking moved geometry: bbox error {error} exceeds {tolerance}')
        audit.update(bbox_max_error_m=error, bbox_tolerance_m=tolerance)
        baked_positions = world_positions(doc, binary)
        if original_positions is not None and baked_positions is not None:
            if original_positions.keys() != baked_positions.keys():
                raise ValueError('baking changed material/vertex-attribute groups')
            vertex_error = 0.0
            for key, original in original_positions.items():
                transformed = baked_positions[key]
                if original.shape != transformed.shape:
                    raise ValueError('baking changed vertex count in a material group')
                vertex_error = max(vertex_error, float(np.max(np.abs(original - transformed), initial=0)))
            if vertex_error > tolerance:
                raise ValueError(f'baking moved vertices: error {vertex_error} exceeds {tolerance}')
            audit.update(vertex_max_error_m=vertex_error, vertex_correspondence_verified=True)
        else:
            audit['vertex_correspondence_verified'] = False
    temporary.replace(output)
    return {'before': before, 'after': after, 'steps': records, 'audit': audit,
            'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'baked_sha256': hashlib.sha256(baked_raw).hexdigest()}
