"""Baked delivery integrity and texture conversion regressions."""
import importlib.util
import io
import json
from pathlib import Path
import struct

import pytest
from PIL import Image

from rescue_scenes.asset_baking import CATEGORIES, bake, decode_glb, measure, sha256, validate
from rescue_scenes.glb_runtime import _read_accessor, compact_textures, write_glb

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / 'scene_data/assets_library'
MANIFEST = LIBRARY / 'baked_manifest.json'


@pytest.fixture(scope='module')
def manifest():
    if not MANIFEST.exists():
        pytest.skip('large local asset library has not been staged in this checkout')
    return json.loads(MANIFEST.read_text())


def test_rejected_jpeg_keeps_png_mime_and_payload():
    """A tiny lossless image can be smaller than a perfect-PSNR JPEG."""
    stream = io.BytesIO()
    Image.new('RGB', (1, 1), 'black').save(stream, 'PNG')
    png = stream.getvalue()
    doc = {'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(png)}],
           'buffers': [{'byteLength': len(png)}],
           'images': [{'bufferView': 0, 'mimeType': 'image/png'}],
           'textures': [{'source': 0}],
           'materials': [{'pbrMetallicRoughness': {'baseColorTexture': {'index': 0}}}]}
    result, binary, record = compact_textures(doc, png)
    assert result['images'][0]['mimeType'] == 'image/png'
    assert binary[:len(png)] == png
    assert record['to_jpeg'] == 0


@pytest.mark.parametrize('mutation', ['magic', 'version', 'length', 'alignment', 'chunk_type'])
def test_glb_decoder_rejects_corrupt_headers(mutation):
    document = json.dumps({'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}]}).encode()
    document += b' ' * (-len(document) % 4)
    payload = struct.pack('<II', len(document), 0x4E4F534A) + document + struct.pack('<II', 4, 0x004E4942) + bytes(4)
    raw = bytearray(struct.pack('<4sII', b'glTF', 2, 12 + len(payload)) + payload)
    if mutation == 'magic': raw[:4] = b'bad!'
    if mutation == 'version': struct.pack_into('<I', raw, 4, 1)
    if mutation == 'length': struct.pack_into('<I', raw, 8, len(raw) - 1)
    if mutation == 'alignment': struct.pack_into('<I', raw, 12, len(document) - 1)
    if mutation == 'chunk_type': struct.pack_into('<I', raw, 16, 0)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        decode_glb(raw)


def test_baking_preserves_mirrored_geometry_tangent_handedness_and_color(tmp_path):
    import numpy as np
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], dtype='<f4')
    arrays = [positions, np.array([[0, 0, 1]] * 3, dtype='<f4'),
              np.array([[1, 0, 0, -1]] * 3, dtype='<f4'),
              np.array([[0.1, 0.5, 1, 0.25]] * 3, dtype='<f4'),
              np.array([[0, 1, 2]], dtype='<u4')]
    binary = b''; views = []; accessors = []
    for array, kind in zip(arrays, ['VEC3', 'VEC3', 'VEC4', 'VEC4', 'SCALAR']):
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': array.nbytes})
        binary += array.tobytes()
        accessors.append({'bufferView': len(views) - 1, 'componentType': 5125 if kind == 'SCALAR' else 5126,
                          'count': array.size if kind == 'SCALAR' else len(array), 'type': kind})
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}],
           'bufferViews': views, 'accessors': accessors,
           'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TANGENT': 2, 'COLOR_0': 3}, 'indices': 4}]}],
           'nodes': [{'mesh': 0, 'scale': [-1, 2, 1], 'translation': [10, 20, 30]}],
           'scenes': [{'nodes': [0]}], 'scene': 0}
    original = tmp_path / 'original.glb'; output = tmp_path / 'baked.glb'
    write_glb(original, doc, binary)
    record = bake(original, output)
    result, payload = decode_glb(output.read_bytes())
    prim = result['meshes'][0]['primitives'][0]
    assert record['steps']['quantize_geometry']['quantized']
    assert record['before']['triangles'] == record['after']['triangles'] == 1
    assert record['after']['bbox_min'] == pytest.approx([8, 20, 30], abs=0.0002)
    assert record['after']['bbox_max'] == pytest.approx([10, 26, 30], abs=0.0002)
    tangent = _read_accessor(result, payload, prim['attributes']['TANGENT'])
    assert tangent == pytest.approx(np.array([[-1, 0, 0, 1]] * 3), abs=1 / 127)
    color = _read_accessor(result, payload, prim['attributes']['COLOR_0'])
    assert color == pytest.approx(arrays[3], abs=1 / 255)
    indices = _read_accessor(result, payload, prim['indices']).ravel()
    assert indices.tolist() == [2, 1, 0]


def test_delivery_selection_reassembles_complete_raw_glb_segments():
    spec = importlib.util.spec_from_file_location('stage_drive_assets_test', ROOT / 'tools/stage_and_bake_drive_assets.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    aid = 'BLD-002-G02-D2'
    row = {'acceptance_id': aid, '生产状态': '已交付', '修订号 / 文件版本': 'r16'}
    def part(i):
        return {'path': f'/delivery/part{i}.zip', 'member': f'export/{aid}.glb.part{i:02d}of02.bin',
                'bytes': 100 + i, 'crc32': i}
    folder = {'directory': 'BLD-002-G02-D2_r16_CURRENT_EXTERIOR', 'candidates': [part(2), part(1)]}
    choices, gaps = module.select_sources([row], [folder])
    assert not gaps
    assert len(choices) == 1
    assert choices[0]['source']['bytes'] == 203
    assert choices[0]['source']['segments'] == [part(1), part(2)]
    incomplete = {'directory': folder['directory'], 'candidates': [part(1)]}
    choices, gaps = module.select_sources([row], [incomplete])
    assert not choices
    assert gaps[0]['reason'] == 'no_matching_complete_glb'


def test_base_selection_prefers_documented_static_name_over_wind_snapshot():
    spec = importlib.util.spec_from_file_location('stage_static_base_test', ROOT / 'tools/stage_and_bake_drive_assets.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    row = {'acceptance_id': 'ENV-003-C01-BASE', '生产状态': '已交付', '修订号 / 文件版本': 'BASE'}
    folder = {'directory': 'ENV003C01_726_728_G08_GREEN_SHRUB_BASE_STATIC_WIND',
              'candidates': [{'path': '/delivery/current.zip', 'member': 'formats/WIND.glb', 'bytes': 100, 'crc32': 1},
                             {'path': '/delivery/current.zip', 'member': 'formats/STATIC.glb', 'bytes': 100, 'crc32': 2}]}
    choices, gaps = module.select_sources([row], [folder])
    assert not gaps
    assert choices[0]['source']['member'] == 'formats/STATIC.glb'


def test_extraction_rejects_changed_member_crc(tmp_path, monkeypatch):
    import zipfile
    spec = importlib.util.spec_from_file_location('stage_drive_extract_test', ROOT / 'tools/stage_and_bake_drive_assets.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'LIBRARY', tmp_path / 'library')
    archive = tmp_path / 'delivery.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        output.writestr('asset.glb', b'current')
    with zipfile.ZipFile(archive) as source:
        info = source.getinfo('asset.glb')
    choice = {'asset_id': 'AGT-001-C01-BASE',
              'source': {'path': str(archive), 'member': 'asset.glb', 'bytes': info.file_size, 'crc32': info.CRC ^ 1}}
    with pytest.raises(ValueError, match='identity changed'):
        module.extract(choice, [])
    assert not (module.LIBRARY / '_raw_glb/AGT-001-C01-BASE.glb').exists()


def test_configurations_have_no_drive_mount_dependency():
    for path in (ROOT / 'configs').rglob('*.json'):
        text = path.read_text()
        json.loads(text)
        assert '/mnt/data2/weizhiwei/google-drive/' not in text, str(path)


def test_category_layout_and_complete_accounting(manifest):
    assert set(manifest['by_category']) == set(CATEGORIES)
    assert not manifest['gaps'], manifest['gaps']
    ids = [r['asset_id'] for r in manifest['assets'] if r['status'] == 'baked']
    assert len(ids) == len(set(ids)) == manifest['summary']['assets']
    gaps = {r['asset_id'] for r in manifest['gaps']}
    assert not set(ids) & gaps
    assert len(set(ids) | gaps) == manifest['catalog']['delivered_geometry']
    listed = set()
    for record in manifest['assets']:
        if record['status'] != 'baked': continue
        path = ROOT / record['file']; listed.add(path)
        assert path.parent.name == CATEGORIES[record['asset_id'][:3]]
        assert path.name == record['asset_id'] + '.glb'
    actual = {p for category in CATEGORIES.values() for p in (LIBRARY / category).glob('*.glb')}
    assert listed == actual


def test_every_baked_glb_is_embedded_valid_and_matches_metrics(manifest):
    for record in manifest['assets'] + manifest['runtime']:
        assert record['status'] == 'baked', record
        path = ROOT / record['file']; raw = path.read_bytes()
        assert sha256(path) == record['baked_sha256'], str(path)
        doc, binary = decode_glb(raw)
        assert validate(doc, binary, max_px=512)['valid']
        measured = measure(doc, binary, len(raw))
        assert measured == record['after'], str(path)
        assert record['before']['triangles'] == measured['triangles']
        assert record['audit']['vertex_correspondence_verified']
        assert record['audit']['vertex_max_error_m'] <= record['audit']['bbox_tolerance_m']
        assert measured['max_texture_dimension'] <= 512
        assert set(doc.get('extensionsUsed', [])) & {
            'KHR_materials_transmission', 'KHR_materials_ior', 'KHR_materials_clearcoat',
            'KHR_materials_specular', 'KHR_materials_emissive_strength'} == set()


def test_deformation_metadata_and_nonimage_bytes_are_preserved(manifest):
    checked = 0
    for record in manifest['assets'] + manifest['runtime']:
        if record['steps']['merge_by_material']['merged']:
            continue
        source = ROOT / record.get('raw_file', record.get('backup'))
        original, original_binary = decode_glb(source.read_bytes())
        dynamic = original.get('skins') or original.get('animations') or any(
            primitive.get('targets') for mesh in original.get('meshes', []) for primitive in mesh['primitives'])
        if not dynamic:
            continue
        baked, baked_binary = decode_glb((ROOT / record['file']).read_bytes())
        for key in ('skins', 'animations', 'meshes', 'nodes', 'accessors', 'scenes', 'scene'):
            assert original.get(key) == baked.get(key), (record['file'], key)
        image_views = {image['bufferView'] for image in original.get('images', [])}
        assert len(original['bufferViews']) == len(baked['bufferViews'])
        for i, original_view in enumerate(original['bufferViews']):
            if i in image_views: continue
            baked_view = baked['bufferViews'][i]
            old_start = original_view.get('byteOffset', 0); new_start = baked_view.get('byteOffset', 0)
            assert original_view['byteLength'] == baked_view['byteLength']
            assert original_binary[old_start:old_start + original_view['byteLength']] == baked_binary[new_start:new_start + baked_view['byteLength']], (record['file'], i)
        checked += 1
    assert checked > 0


def test_reviewed_primary_selection_matches_baked_manifest(manifest):
    selections = json.loads((ROOT / 'configs/drive_asset_primary_selection.json').read_text())['entries']
    records = {record['asset_id']: record for record in manifest['assets']}
    for aid, selected in selections.items():
        record = records[aid]; source = record['source']
        assert source['directory'] == selected['directory']
        assert Path(source['path']).name == selected['archive']
        assert source['member'] == selected['member']
        assert source['crc32'] == selected['crc32']
        assert record['before']['file_bytes'] == selected['bytes']
        assert (ROOT / selected['selection_evidence']['file']).is_file()


def test_runtime_originals_and_archive_checksums(manifest):
    assert len(manifest['runtime']) == 17
    for record in manifest['runtime']:
        backup = ROOT / record['backup']
        assert sha256(backup) == record['raw_sha256']
        assert backup.stat().st_size == record['before']['file_bytes']
    for entry in manifest['archives']:
        if entry['kind'] != 'archive': continue
        path = ROOT / entry['file']
        assert entry['zip_crc_verified']
        assert path.stat().st_size == entry['bytes']
        assert sha256(path) == entry['sha256']


def test_manifest_aggregates_are_exact(manifest):
    records = [r for r in manifest['assets'] if r['status'] == 'baked']
    for stage in ('before', 'after'):
        for field, value in manifest['summary'][stage].items():
            assert value == sum(r[stage][field] for r in records)
        for category in CATEGORIES:
            selected = [r for r in records if r['category'] == category]
            for field, value in manifest['by_category'][category][stage].items():
                assert value == sum(r[stage][field] for r in selected)
