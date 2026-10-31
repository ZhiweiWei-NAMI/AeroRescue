"""AAS runtime variant of an accepted GLB: same geometry/UVs/materials, lighter GPU load.

- Embedded textures are downscaled to at most `max_px` (same image format).
- Material extensions that force extra render passes in the AAS viewport (transmission/ior/clearcoat/
  specular/emissive_strength) are dropped; transmissive glass becomes an alpha-blended surface.
The source GLB is never modified; the transform record goes into display-assets.json.
"""
import io
import json
import struct

from PIL import Image

DROP = {'KHR_materials_transmission', 'KHR_materials_ior', 'KHR_materials_clearcoat',
        'KHR_materials_specular', 'KHR_materials_emissive_strength'}


def optimize(doc, binary, max_px):
    views = doc.get('bufferViews', [])
    replaced = {}
    record = {'max_px': max_px, 'images': [], 'dropped_extensions': sorted(DROP & set(doc.get('extensionsUsed', [])))}
    for image in doc.get('images', []):
        if 'bufferView' not in image:
            continue
        view = views[image['bufferView']]
        start = view.get('byteOffset', 0)
        data = binary[start:start + view['byteLength']]
        picture = Image.open(io.BytesIO(data))
        if max(picture.size) <= max_px:
            continue
        scale = max_px / max(picture.size)
        size = (max(1, round(picture.size[0] * scale)), max(1, round(picture.size[1] * scale)))
        out = io.BytesIO()
        jpeg = image.get('mimeType') == 'image/jpeg'
        picture = picture.convert('RGB') if jpeg and picture.mode not in ('RGB', 'L') else picture
        picture.resize(size, Image.LANCZOS).save(out, 'JPEG' if jpeg else 'PNG', **({'quality': 90} if jpeg else {'optimize': True}))
        replaced[image['bufferView']] = out.getvalue()
        record['images'].append([list(Image.open(io.BytesIO(data)).size), list(size)])
    for material in doc.get('materials', []):
        ext = material.get('extensions', {})
        if 'KHR_materials_transmission' in ext:
            pbr = material.setdefault('pbrMetallicRoughness', {})
            factor = list(pbr.get('baseColorFactor', [1, 1, 1, 1]))
            factor[3] = min(factor[3], 0.35)
            pbr['baseColorFactor'] = factor
            material['alphaMode'] = 'BLEND'
        for name in DROP:
            ext.pop(name, None)
        if not ext:
            material.pop('extensions', None)
    for key in ('extensionsUsed', 'extensionsRequired'):
        if key in doc:
            doc[key] = [name for name in doc[key] if name not in DROP]
            if not doc[key]:
                doc.pop(key)
    # Repack the binary chunk with 4-byte aligned views (accessors keep their view indices).
    blob = bytearray()
    for index, view in enumerate(views):
        start = view.get('byteOffset', 0)
        data = replaced.get(index, binary[start:start + view['byteLength']])
        blob += b'\0' * ((-len(blob)) % 4)
        view['byteOffset'] = len(blob)
        view['byteLength'] = len(data)
        view['buffer'] = 0
        blob += data
    blob += b'\0' * ((-len(blob)) % 4)
    doc['buffers'] = [{'byteLength': len(blob)}]
    return doc, bytes(blob), record


_COMPONENTS = {5120: 'i1', 5121: 'u1', 5122: 'i2', 5123: 'u2', 5125: 'u4', 5126: 'f4'}
_WIDTH = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
_NORMALIZED_SCALE = {'i1': 127.0, 'u1': 255.0, 'i2': 32767.0, 'u2': 65535.0}


def _read_accessor(doc, binary, index):
    import numpy as np
    acc = doc['accessors'][index]
    if 'sparse' in acc or 'bufferView' not in acc:
        raise ValueError('sparse/viewless accessor')
    view = doc['bufferViews'][acc['bufferView']]
    dtype = np.dtype(_COMPONENTS[acc['componentType']]).newbyteorder('<')
    width = _WIDTH[acc['type']]; count = acc['count']
    stride = view.get('byteStride') or dtype.itemsize * width
    start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
    raw = np.frombuffer(binary, dtype=np.uint8, count=stride * (count - 1) + dtype.itemsize * width, offset=start)
    rows = np.lib.stride_tricks.as_strided(raw, shape=(count, dtype.itemsize * width), strides=(stride, 1))
    values = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
    if acc.get('normalized'):
        return np.maximum(values.astype('f8') / _NORMALIZED_SCALE[dtype.str[1:]], -1.0)
    return values.astype('f8') if acc['type'] != 'SCALAR' or acc['componentType'] == 5126 else values.astype('u4')


def _local_matrix(node):
    import numpy as np
    if 'matrix' in node:
        return np.array(node['matrix'], dtype='f8').reshape(4, 4).T
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m = np.eye(4); m[:3, :3] = rot * np.array(node.get('scale', [1, 1, 1]))[None, :]; m[:3, 3] = node.get('translation', [0, 0, 0])
    return m


def merge_by_material(doc, binary):
    """Display-only geometry batching: every static triangle primitive is baked into world (asset)
    space and concatenated with the other primitives that share its material and attribute set.
    Same vertices, UVs, normals, materials and textures; far fewer draw objects. Assets with skins,
    animations, morph targets, compressed or sparse geometry are returned unchanged (with the reason).
    """
    import numpy as np
    record = {'merged': False}
    reasons = [k for k in ('skins', 'animations') if doc.get(k)]
    reasons += [e for e in doc.get('extensionsUsed', []) if e in ('KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_mesh_quantization')]
    if reasons:
        record['skipped'] = reasons; return doc, binary, record
    scene = doc['scenes'][doc.get('scene', 0)]; nodes = doc.get('nodes', [])
    groups, keep, before, triangles = {}, [], 0, 0
    stack = [(i, np.eye(4)) for i in scene['nodes']]
    while stack:
        index, parent = stack.pop()
        node = nodes[index]; world = parent @ _local_matrix(node)
        stack += [(c, world) for c in node.get('children', [])]
        if 'camera' in node or node.get('extensions'):
            keep.append({k: v for k, v in node.items() if k not in ('children', 'mesh', 'matrix', 'translation', 'rotation', 'scale')} | {'matrix': world.T.ravel().tolist()})
        if 'mesh' not in node:
            continue
        for prim in doc['meshes'][node['mesh']]['primitives']:
            before += 1
            try:
                if prim.get('mode', 4) != 4 or prim.get('targets') or 'POSITION' not in prim['attributes']:
                    raise ValueError('non-triangle/morph primitive')
                attrs = {name: _read_accessor(doc, binary, acc) for name, acc in prim['attributes'].items()}
                count = len(attrs['POSITION'])
                idx = _read_accessor(doc, binary, prim['indices']).ravel() if 'indices' in prim else np.arange(count, dtype='u4')
            except (ValueError, KeyError) as exc:
                record['skipped'] = [f'primitive not mergeable: {exc}']; return doc, binary, record
            linear = world[:3, :3]; normal_m = np.linalg.inv(linear).T
            attrs['POSITION'] = attrs['POSITION'] @ linear.T + world[:3, 3]
            if 'NORMAL' in attrs:
                n = attrs['NORMAL'] @ normal_m.T; attrs['NORMAL'] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            if 'TANGENT' in attrs:
                t = attrs['TANGENT'][:, :3] @ linear.T; t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
                attrs['TANGENT'] = np.hstack([t, attrs['TANGENT'][:, 3:] * np.sign(np.linalg.det(linear))])
            idx = idx.reshape(-1, 3)
            if np.linalg.det(linear) < 0:
                idx = idx[:, ::-1]  # a mirrored node flips winding; keep front faces outward
            key = (prim.get('material'), tuple(sorted((k, v.shape[1]) for k, v in attrs.items())))
            group = groups.setdefault(key, {'attrs': {k: [] for k in attrs}, 'indices': [], 'base': 0})
            for k, v in attrs.items():
                group['attrs'][k].append(v)
            group['indices'].append(idx + group['base']); group['base'] += count; triangles += len(idx)
    blob = bytearray(); views, accessors = [], []

    def add_view(data, target=None):
        blob.extend(b'\0' * ((-len(blob)) % 4)); views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)} | ({'target': target} if target else {}))
        blob.extend(data); return len(views) - 1
    images = []
    for image in doc.get('images', []):
        image = dict(image)
        if 'bufferView' in image:
            old = doc['bufferViews'][image['bufferView']]; start = old.get('byteOffset', 0)
            image['bufferView'] = add_view(binary[start:start + old['byteLength']])
        images.append(image)
    primitives = []
    for (material, _), group in groups.items():
        attributes = {}
        for name, parts in group['attrs'].items():
            data = np.concatenate(parts).astype('<f4'); width = data.shape[1]
            acc = {'bufferView': add_view(data.tobytes(), 34962), 'componentType': 5126, 'count': len(data),
                   'type': {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}[width]}
            if name == 'POSITION':
                acc |= {'min': data.min(axis=0).tolist(), 'max': data.max(axis=0).tolist()}
            accessors.append(acc); attributes[name] = len(accessors) - 1
        index = np.concatenate(group['indices']).ravel().astype('<u4')
        accessors.append({'bufferView': add_view(index.tobytes(), 34963), 'componentType': 5125, 'count': len(index), 'type': 'SCALAR'})
        primitives.append({'attributes': attributes, 'indices': len(accessors) - 1} | ({'material': material} if material is not None else {}))
    blob.extend(b'\0' * ((-len(blob)) % 4))
    root = {'name': 'display_merged_by_material', 'mesh': 0} | ({'extras': nodes[scene['nodes'][0]]['extras']} if len(scene['nodes']) == 1 and 'extras' in nodes[scene['nodes'][0]] else {})
    out = {k: v for k, v in doc.items() if k not in ('nodes', 'meshes', 'accessors', 'bufferViews', 'buffers', 'scenes', 'scene', 'images')}
    out |= {'images': images} if images else {}
    out |= {'nodes': [root, *keep], 'meshes': [{'name': 'display_merged', 'primitives': primitives}], 'accessors': accessors,
            'bufferViews': views, 'buffers': [{'byteLength': len(blob)}], 'scenes': [{'nodes': list(range(1 + len(keep)))}], 'scene': 0}
    record.update(merged=True, primitives_before=before, primitives_after=len(primitives), triangles=triangles)
    return out, bytes(blob), record


def opaque_blend(doc, binary, min_alpha=150, opaque_fraction=0.9999):
    """Display-only: BLEND materials whose base-colour alpha is effectively 1 everywhere become OPAQUE.

    Some delivered assets (HUM-001 body/clothing) are exported with alphaMode BLEND over opaque textures,
    which makes the viewer draw whole people as sorted transparent geometry without depth writes. A
    material is switched only when baseColorFactor alpha is 1, no primitive using it has RGBA vertex
    colours, and its base-colour texture has no alpha channel or >= opaque_fraction of texels at
    alpha >= 250 with none below min_alpha. Glass (factor alpha < 1) and cut-outs (lashes, brows) stay.
    """
    import numpy as np
    views = doc.get('bufferViews', []); record = {'to_opaque': [], 'kept_blend': []}
    rgba_vertex = {p.get('material') for m in doc.get('meshes', []) for p in m['primitives']
                   if 'COLOR_0' in p.get('attributes', {}) and doc['accessors'][p['attributes']['COLOR_0']]['type'] == 'VEC4'}
    for index, material in enumerate(doc.get('materials', [])):
        if material.get('alphaMode') != 'BLEND':
            continue
        pbr = material.get('pbrMetallicRoughness', {}); name = material.get('name', str(index))
        if pbr.get('baseColorFactor', [1, 1, 1, 1])[3] < 0.999 or index in rgba_vertex:
            record['kept_blend'].append(name); continue
        stats = None
        if pbr.get('baseColorTexture'):
            image = doc['images'][doc['textures'][pbr['baseColorTexture']['index']]['source']]
            if 'bufferView' not in image:
                record['kept_blend'].append(name); continue
            view = views[image['bufferView']]
            picture = Image.open(io.BytesIO(binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]))
            if 'A' in picture.getbands() or 'transparency' in picture.info:
                alpha = np.asarray(picture.convert('RGBA'))[..., 3]
                stats = {'min_alpha': int(alpha.min()), 'fraction_ge_250': round(float((alpha >= 250).mean()), 6)}
                if stats['min_alpha'] < min_alpha or stats['fraction_ge_250'] < opaque_fraction:
                    record['kept_blend'].append(name); continue
        material['alphaMode'] = 'OPAQUE'
        record['to_opaque'].append({'material': name, **(stats or {'alpha_channel': False})})
    return doc, binary, record


def compact_textures(doc, binary, min_psnr=40.0):
    """Display-only texture payload reduction (call after optimize/merge, which leave one packed buffer):
    16-bit PNGs become 8-bit PNGs (WebGL uploads these images at 8 bits per channel anyway); opaque
    base-colour PNGs become JPEG q92 only when PSNR >= min_psnr against the PNG. Normal, metal-roughness,
    occlusion and alpha textures keep lossless PNG.
    """
    import numpy as np
    views = doc.get('bufferViews', []); record = {'to_8bit': 0, 'to_jpeg': 0, 'kept_png_low_psnr': 0, 'bytes_before': 0, 'bytes_after': 0, 'min_psnr_db': None}
    base = {doc['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']].get('source')
            for m in doc.get('materials', []) if m.get('pbrMetallicRoughness', {}).get('baseColorTexture')}
    other = {doc['textures'][t['index']].get('source') for m in doc.get('materials', [])
             for t in (m.get('normalTexture'), m.get('occlusionTexture'), m.get('emissiveTexture'), m.get('pbrMetallicRoughness', {}).get('metallicRoughnessTexture')) if t}
    # RGBA base-colour images read only by OPAQUE materials: the viewer ignores their alpha.
    readers = {}
    for m in doc.get('materials', []):
        t = m.get('pbrMetallicRoughness', {}).get('baseColorTexture')
        if t: readers.setdefault(doc['textures'][t['index']].get('source'), set()).add(m.get('alphaMode', 'OPAQUE'))
    alpha_unused = {i for i, modes in readers.items() if modes == {'OPAQUE'}}
    replaced = {}
    for i, image in enumerate(doc.get('images', [])):
        if 'bufferView' not in image or image.get('mimeType') != 'image/png':
            continue
        view = views[image['bufferView']]; data = binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]
        picture = Image.open(io.BytesIO(data)); new = None; new_mime = None
        depth, ctype = data[24], data[25]
        if i in base and i not in other and (ctype == 2 or (ctype == 6 and i in alpha_unused)):
            rgb = picture.convert('RGB'); out = io.BytesIO(); rgb.save(out, 'JPEG', quality=92)
            a = np.asarray(rgb, dtype='f8'); b = np.asarray(Image.open(io.BytesIO(out.getvalue())).convert('RGB'), dtype='f8')
            mse = float(((a - b) ** 2).mean()); psnr = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
            if psnr >= min_psnr:
                new = out.getvalue(); new_mime = 'image/jpeg'
                record['min_psnr_db'] = round(float(psnr if record['min_psnr_db'] is None else min(record['min_psnr_db'], psnr)), 2)
            else:
                record['kept_png_low_psnr'] += 1
        if new is None and depth == 16:
            out = io.BytesIO(); picture.convert('RGBA' if ctype in (4, 6) else 'RGB').save(out, 'PNG', optimize=True)
            new = out.getvalue(); record['to_8bit'] += 1
        if new is not None and len(new) < len(data):
            if new_mime:
                image['mimeType'] = new_mime; record['to_jpeg'] += 1
            replaced[image['bufferView']] = new; record['bytes_before'] += len(data); record['bytes_after'] += len(new)
    blob = bytearray()
    for index, view in enumerate(views):
        data = replaced.get(index, binary[view['byteOffset']:view['byteOffset'] + view['byteLength']])
        blob += b'\0' * ((-len(blob)) % 4); view['byteOffset'] = len(blob); view['byteLength'] = len(data); blob += data
    blob += b'\0' * ((-len(blob)) % 4); doc['buffers'] = [{'byteLength': len(blob)}]
    return doc, bytes(blob), record


def semantic_textures(doc, binary, max_px=512, quality=85):
    """Display-only texture budget judged by semantic recognisability (category, damage grade, task state),
    not pixel fidelity: damage states of one building share identical textures, so damage lives in the
    geometry. Images are capped at max_px; every image whose alpha is never read (no BLEND/MASK material
    samples it as base colour) becomes JPEG, normal and metal-roughness maps included. Call on the single
    packed buffer (after compact_textures)."""
    views = doc.get('bufferViews', []); record = {'max_px': max_px, 'jpeg_quality': quality, 'bytes_before': 0, 'bytes_after': 0, 'images': []}
    alpha_read = {doc['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']].get('source')
                  for m in doc.get('materials', []) if m.get('alphaMode', 'OPAQUE') != 'OPAQUE' and m.get('pbrMetallicRoughness', {}).get('baseColorTexture')}
    replaced = {}
    for i, image in enumerate(doc.get('images', [])):
        if 'bufferView' not in image:
            continue
        view = views[image['bufferView']]; data = binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]
        picture = Image.open(io.BytesIO(data)); size = picture.size
        if max(size) > max_px:
            scale = max_px / max(size); picture = picture.resize((max(1, round(size[0] * scale)), max(1, round(size[1] * scale))), Image.LANCZOS)
        out = io.BytesIO()
        if i in alpha_read and picture.mode in ('RGBA', 'LA', 'P'):
            picture.save(out, 'PNG', optimize=True); mime = 'image/png'
        else:
            picture.convert('RGB').save(out, 'JPEG', quality=quality); mime = 'image/jpeg'
        if len(out.getvalue()) < len(data):
            replaced[image['bufferView']] = out.getvalue(); image['mimeType'] = mime
        record['bytes_before'] += len(data); record['bytes_after'] += len(replaced.get(image['bufferView'], data))
        record['images'].append([list(size), list(picture.size), image.get('mimeType')])
    blob = bytearray()
    for index, view in enumerate(views):
        data = replaced.get(index, binary[view['byteOffset']:view['byteOffset'] + view['byteLength']])
        blob += b'\0' * ((-len(blob)) % 4); view['byteOffset'] = len(blob); view['byteLength'] = len(data); blob += data
    blob += b'\0' * ((-len(blob)) % 4); doc['buffers'] = [{'byteLength': len(blob)}]
    return doc, bytes(blob), record


def externalize_images(doc, binary):
    """Move embedded images out of the packed buffer into content-addressed files, so identical textures
    (e.g. the damage states of one building) are downloaded, decoded and uploaded once by the viewer.
    Returns (doc, geometry-only binary, {file name: bytes}); image URIs are the file names."""
    import hashlib
    views = doc.get('bufferViews', []); image_views = {img['bufferView'] for img in doc.get('images', []) if 'bufferView' in img}
    files, keep, blob = {}, {}, bytearray()
    for image in doc.get('images', []):
        if 'bufferView' not in image:
            continue
        view = views[image['bufferView']]; data = binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]
        name = f"upstream-img-{hashlib.sha256(data).hexdigest()[:16]}.{'jpg' if image.get('mimeType') == 'image/jpeg' else 'png'}"
        files[name] = data; image.pop('bufferView'); image['uri'] = name
    for index, view in enumerate(views):
        if index in image_views:
            continue
        data = binary[view['byteOffset']:view['byteOffset'] + view['byteLength']]
        blob += b'\0' * ((-len(blob)) % 4); keep[index] = len(keep); blob += data
        view['byteOffset'] = len(blob) - len(data)
    doc['bufferViews'] = [views[i] for i in sorted(keep)]
    for accessor in doc.get('accessors', []):
        if 'bufferView' in accessor:
            accessor['bufferView'] = keep[accessor['bufferView']]
    blob += b'\0' * ((-len(blob)) % 4); doc['buffers'] = [{'byteLength': len(blob)}]
    return doc, bytes(blob), files


def quantize_geometry(doc, binary):
    """Display-only KHR_mesh_quantization (core three.js GLTFLoader support, no decoder): positions as
    normalized int16 dequantized by a uniform scale + offset child node per mesh (sub-millimetre for
    building-sized meshes), normals as normalized int8, UV sets as normalized uint16 when inside [0,1],
    uint16 indices when they fit. Same triangles; the mesh keeps its place in the asset frame.
    Skinned/animated/morph or already-compressed assets are returned unchanged (with the reason)."""
    import numpy as np
    record = {'quantized': False}
    reasons = [k for k in ('skins', 'animations') if doc.get(k)] + [e for e in doc.get('extensionsUsed', []) if e in ('KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_mesh_quantization')]
    prims = [p for m in doc.get('meshes', []) for p in m['primitives']]
    if any(p.get('targets') or p.get('mode', 4) != 4 or 'indices' not in p or {k for k in p['attributes'] if not k.startswith(('TEXCOORD_', 'COLOR_'))} - {'POSITION', 'NORMAL', 'TANGENT'} for p in prims):
        reasons.append('morph/non-triangle/unindexed primitive or unsupported vertex attributes')
    if reasons:
        record['skipped'] = reasons; return doc, binary, record
    blob, views, accessors = bytearray(), [], []

    def put(array, target, stride=None, **acc):
        blob.extend(b'\0' * ((-len(blob)) % 4))
        views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': array.nbytes, 'target': target} | ({'byteStride': stride} if stride else {}))
        blob.extend(array.tobytes()); accessors.append({'bufferView': len(views) - 1} | acc); return len(accessors) - 1

    before = len(binary); dequant = {}
    for mi, mesh in enumerate(doc.get('meshes', [])):
        positions = [_read_accessor(doc, binary, p['attributes']['POSITION']) for p in mesh['primitives']]
        lo, hi = np.min([a.min(axis=0) for a in positions], axis=0), np.max([a.max(axis=0) for a in positions], axis=0)
        center, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2, 1e-6); dequant[mi] = (center.tolist(), half)
        for prim, pos in zip(mesh['primitives'], positions):
            q = np.zeros((len(pos), 4), '<i2'); q[:, :3] = np.clip(np.round((pos - center) / half * 32767), -32767, 32767)
            attributes = {'POSITION': put(q, 34962, 8, componentType=5122, normalized=True, count=len(q), type='VEC3',
                                          min=q[:, :3].min(axis=0).tolist(), max=q[:, :3].max(axis=0).tolist())}
            if 'NORMAL' in prim['attributes']:
                n = _read_accessor(doc, binary, prim['attributes']['NORMAL']); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                qn = np.zeros((len(n), 4), '<i1'); qn[:, :3] = np.clip(np.round(n * 127), -127, 127)
                attributes['NORMAL'] = put(qn, 34962, 4, componentType=5120, normalized=True, count=len(qn), type='VEC3')
            if 'TANGENT' in prim['attributes']:
                tangent = _read_accessor(doc, binary, prim['attributes']['TANGENT'])
                tangent[:, :3] /= np.maximum(np.linalg.norm(tangent[:, :3], axis=1, keepdims=True), 1e-12)
                qt = np.clip(np.round(tangent * 127), -127, 127).astype('<i1')
                attributes['TANGENT'] = put(qt, 34962, 4, componentType=5120, normalized=True, count=len(qt), type='VEC4')
            for name in sorted(k for k in prim['attributes'] if k.startswith('COLOR_')):
                color = _read_accessor(doc, binary, prim['attributes'][name])
                width = color.shape[1]
                if color.min() >= 0 and color.max() <= 1:
                    qc = np.zeros((len(color), 4), '<u1')
                    qc[:, :width] = np.round(color * 255)
                    attributes[name] = put(qc, 34962, 4, componentType=5121, normalized=True, count=len(qc), type=f'VEC{width}')
                else:
                    attributes[name] = put(color.astype('<f4'), 34962, None, componentType=5126, count=len(color), type=f'VEC{width}')
            for name in sorted(k for k in prim['attributes'] if k.startswith('TEXCOORD_')):
                uv = _read_accessor(doc, binary, prim['attributes'][name])
                if uv.min() >= 0 and uv.max() <= 1:
                    attributes[name] = put(np.round(uv * 65535).astype('<u2'), 34962, None, componentType=5123, normalized=True, count=len(uv), type='VEC2')
                else:
                    attributes[name] = put(uv.astype('<f4'), 34962, None, componentType=5126, count=len(uv), type='VEC2')
            index = _read_accessor(doc, binary, prim['indices']).ravel()
            small = index.max(initial=0) < 65535
            prim['indices'] = put(index.astype('<u2' if small else '<u4'), 34963, None, componentType=5123 if small else 5125, count=len(index), type='SCALAR')
            prim['attributes'] = attributes
    image_views = {}
    for image in doc.get('images', []):
        if 'bufferView' in image:
            view = doc['bufferViews'][image['bufferView']]; data = binary[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
            blob.extend(b'\0' * ((-len(blob)) % 4)); views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(data)}); blob.extend(data)
            image['bufferView'] = len(views) - 1
    nodes = doc['nodes']
    for node in list(nodes):
        if 'mesh' in node:
            center, half = dequant[node['mesh']]
            nodes.append({'name': 'display_dequantize', 'mesh': node.pop('mesh'), 'translation': center, 'scale': [half] * 3})
            node.setdefault('children', []).append(len(nodes) - 1)
    blob.extend(b'\0' * ((-len(blob)) % 4))
    doc['accessors'], doc['bufferViews'], doc['buffers'] = accessors, views, [{'byteLength': len(blob)}]
    for key in ('extensionsUsed', 'extensionsRequired'):
        doc[key] = sorted(set(doc.get(key, [])) | {'KHR_mesh_quantization'})
    record.update(quantized=True, bytes_before=before, bytes_after=len(blob))
    return doc, bytes(blob), record


def self_contained_level(template, images):
    """A display_lods template (buffer 0 = base images, buffer 1 = level geometry) rewritten to use the
    externalized image files: one buffer (the level geometry), so a level never fetches the base binary."""
    import copy
    level = copy.deepcopy(template); drop = {i for i, v in enumerate(level['bufferViews']) if v['buffer'] == 0}
    remap = {old: new for new, old in enumerate(i for i in range(len(level['bufferViews'])) if i not in drop)}
    level['bufferViews'] = [dict(v, buffer=0) for i, v in enumerate(level['bufferViews']) if i not in drop]
    for accessor in level['accessors']:
        accessor['bufferView'] = remap[accessor['bufferView']]
    if images:
        level['images'] = copy.deepcopy(images)
    level['buffers'] = [level['buffers'][1]]
    return level


def read_glb(path):
    raw = path.read_bytes(); n, _ = struct.unpack_from('<II', raw, 12); doc = json.loads(raw[20:20 + n])
    size, _ = struct.unpack_from('<II', raw, 20 + n)
    return doc, raw[28 + n:28 + n + size]


# (triangle ratio, switch distance in bounding radii). One level: a second at 0.08 broke HUM-001 clothing
# shells in the Blender check (dev/step3/merge_check/lod_compare_hum.png).
LOD_LEVELS = ((0.30, 8.0),)


def display_lods(doc, binary, cache, blender='blender', levels=LOD_LEVELS, min_triangles=300, max_image_diff=0.5):
    """Display-only distance levels of one merged display GLB, decimated in Blender (COLLAPSE).

    Returns [(template, lod_binary, record)] ascending by distance. A template is a glTF document whose
    buffer 0 is the full display binary (images are shared, never duplicated) and buffer 1 the level's own
    geometry; the caller fills both buffer URIs. Same materials/textures, same asset frame, same damage
    state: only the triangle count changes. Levels that would keep < min_triangles or save < 30 % are
    dropped. A level is kept only if, rendered in Blender (Cycles) as the viewer frames it at the level's
    switch distance (front and rear), its mean absolute pixel difference to the full model is at most
    max_image_diff (0-255 scale); edge-collapse tears hard-surface assets (cars, barrels), which this rejects.
    Results are cached by the display binary hash in `cache`.
    """
    import hashlib, subprocess, tempfile
    import numpy as np
    from pathlib import Path
    if doc.get('skins') or doc.get('animations') or len(doc.get('meshes', [])) != 1:
        return []
    full_tris = sum(doc['accessors'][p['indices']]['count'] // 3 for p in doc['meshes'][0]['primitives'] if 'indices' in p)
    positions = np.concatenate([_read_accessor(doc, binary, p['attributes']['POSITION']) for p in doc['meshes'][0]['primitives']])
    lo, hi = positions.min(axis=0), positions.max(axis=0); radius = float(np.linalg.norm(hi - lo) / 2)
    key = hashlib.sha256(binary + json.dumps(doc['meshes'], sort_keys=True).encode()).hexdigest()[:16]
    folder = Path(cache) / key; folder.mkdir(parents=True, exist_ok=True)
    outputs = [folder / f'lod{i + 1}.glb' for i in range(len(levels))]
    if not all(p.exists() for p in outputs):
        # Geometry-only input with texture-less materials named M<index>, which map back to the display materials by index.
        geometry = {k: v for k, v in doc.items() if k not in ('images', 'textures', 'samplers', 'extensionsUsed', 'extensionsRequired')}
        geometry['materials'] = [{'name': f'M{i}'} for i in range(len(doc.get('materials', [])))]
        with tempfile.TemporaryDirectory() as tmp:
            write_glb(Path(tmp) / 'in.glb', geometry, binary)
            job = Path(tmp) / 'job.json'
            job.write_text(json.dumps({'input': str(Path(tmp) / 'in.glb'), 'levels': [{'ratio': r, 'output': str(o)} for (r, _), o in zip(levels, outputs)]}))
            try:
                subprocess.run([blender, '-b', '--factory-startup', '--python', str(Path(__file__).with_name('lod_blender.py')), '--', str(job)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Display-only distance levels; headless hosts without Blender skip them.
                return []
    result, previous = [], full_tris
    image_views = sorted({img['bufferView'] for img in doc.get('images', []) if 'bufferView' in img})
    for (ratio, radii), path in zip(levels, outputs):
        raw_doc, raw_bin = read_glb(path)
        merged, merged_bin, _ = merge_by_material(raw_doc, raw_bin)
        names = [m.get('name', '') for m in raw_doc.get('materials', [])]
        views = [dict(doc['bufferViews'][v], buffer=0) for v in image_views]; remap = {v: i for i, v in enumerate(image_views)}
        blob = bytearray(); accessors, primitives, triangles = [], [], 0
        for prim in merged['meshes'][0]['primitives']:
            attributes = {}
            for name, acc in prim['attributes'].items():
                data = _read_accessor(merged, merged_bin, acc).astype('<f4')
                blob.extend(b'\0' * ((-len(blob)) % 4)); views.append({'buffer': 1, 'byteOffset': len(blob), 'byteLength': data.nbytes, 'target': 34962}); blob.extend(data.tobytes())
                accessors.append({'bufferView': len(views) - 1, 'componentType': 5126, 'count': len(data), 'type': {1: 'SCALAR', 2: 'VEC2', 3: 'VEC3', 4: 'VEC4'}[data.shape[1]]}
                                 | ({'min': data.min(axis=0).tolist(), 'max': data.max(axis=0).tolist()} if name == 'POSITION' else {}))
                attributes[name] = len(accessors) - 1
            index = _read_accessor(merged, merged_bin, prim['indices']).ravel().astype('<u4'); triangles += len(index) // 3
            blob.extend(b'\0' * ((-len(blob)) % 4)); views.append({'buffer': 1, 'byteOffset': len(blob), 'byteLength': index.nbytes, 'target': 34963}); blob.extend(index.tobytes())
            accessors.append({'bufferView': len(views) - 1, 'componentType': 5125, 'count': len(index), 'type': 'SCALAR'})
            material = int(names[prim['material']][1:]) if 'material' in prim and names[prim['material']][1:].isdigit() else None
            primitives.append({'attributes': attributes, 'indices': len(accessors) - 1} | ({'material': material} if material is not None else {}))
        blob.extend(b'\0' * ((-len(blob)) % 4))
        record = {'ratio': ratio, 'triangles': triangles, 'full_triangles': full_tris, 'min_distance_m': round(radii * radius, 2), 'bounding_radius_m': round(radius, 3)}
        if triangles < min_triangles or triangles > 0.7 * previous:
            continue
        root = dict(doc['nodes'][doc['scenes'][doc.get('scene', 0)]['nodes'][0]]); root['mesh'] = 0
        template = {k: v for k, v in doc.items() if k not in ('nodes', 'meshes', 'accessors', 'bufferViews', 'buffers', 'scenes', 'scene', 'images')}
        template |= {'images': [dict(img, bufferView=remap[img['bufferView']]) if 'bufferView' in img else img for img in doc.get('images', [])]} if doc.get('images') else {}
        template |= {'nodes': [root], 'meshes': [{'name': f'display_lod_{ratio}', 'primitives': primitives}], 'accessors': accessors, 'bufferViews': views,
                     'buffers': [{'byteLength': len(binary)}, {'byteLength': len(blob)}], 'scenes': [{'nodes': [0]}], 'scene': 0}
        record['image_diff'] = _lod_image_diff(doc, binary, template, bytes(blob), record['min_distance_m'], folder / path.stem, blender)
        if record['image_diff'] > max_image_diff:
            continue
        previous = triangles
        result.append((template, bytes(blob), record))
    return result


def _lod_image_diff(doc, binary, template, lod_binary, distance, stem, blender):
    """Max over front/rear of the mean absolute RGB difference between full model and level renders (cached)."""
    import copy, subprocess
    import numpy as np
    from pathlib import Path
    cached = Path(f'{stem}.diff.json')
    if cached.exists():
        return json.loads(cached.read_text())['image_diff']
    level = copy.deepcopy(template); offset = len(binary) + (-len(binary)) % 4
    for view in level['bufferViews']:
        view['byteOffset'] = view.get('byteOffset', 0) + (offset if view['buffer'] == 1 else 0); view['buffer'] = 0
    combined = binary + b'\0' * ((-len(binary)) % 4) + lod_binary; level['buffers'] = [{'byteLength': len(combined)}]
    import tempfile
    # Private scratch files: concurrent materializations may check the same new asset.
    with tempfile.TemporaryDirectory(prefix='lodcheck-') as scratch:
        full_glb, level_glb = Path(scratch) / 'full.glb', Path(scratch) / 'check.glb'
        write_glb(full_glb, doc, binary); write_glb(level_glb, level, combined)
        script = str(Path(__file__).with_name('lod_check_blender.py')); diffs = {}
        for view in ('front', 'rear'):
            frame = Path(scratch) / f'{view}.frame.json'; shots = []
            for name, glb in (('full', full_glb), ('level', level_glb)):
                png = Path(scratch) / f'{view}.{name}.png'; shots.append(png)
                subprocess.run([blender, '-b', '--factory-startup', '--python', script, '--', str(glb), str(png), str(distance), view, '32', str(frame)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800)
            a, b = (np.asarray(Image.open(p).convert('RGB'), dtype='f8') for p in shots)
            diffs[view] = round(float(np.abs(a - b).mean()), 3)
    cached.write_text(json.dumps({'image_diff': max(diffs.values()), 'views': diffs}))
    return max(diffs.values())


def write_glb(path, doc, binary):
    body = json.dumps(doc, separators=(',', ':')).encode()
    body += b' ' * ((-len(body)) % 4)
    payload = struct.pack('<II', len(body), 0x4E4F534A) + body + struct.pack('<II', len(binary), 0x004E4942) + binary
    path.write_bytes(struct.pack('<4sII', b'glTF', 2, len(payload) + 12) + payload)
