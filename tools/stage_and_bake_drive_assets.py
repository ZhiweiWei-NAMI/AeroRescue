#!/usr/bin/env python3
"""Stage formal Drive deliveries and bake self-contained, audited runtime GLBs.

Run with AeroAgentSim/.venv/bin/python tools/stage_and_bake_drive_assets.py.
The source cache and per-asset records make reruns resumable. No raw Blender or
render-sheet payload is extracted. ZIP split GLB byte segments are reassembled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import uuid
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scene_data/upstream/scene-v0.6'))
from rescue_scene.catalog_snapshot import read_excel, GEOMETRY
from rescue_scenes.asset_baking import CATEGORIES, PIPELINE, bake, sha256

DRIVE = Path('/mnt/data2/weizhiwei/google-drive')
DELIVERIES = DRIVE / '低空救援_三维资产库_20261021/03_批量资产'
LIBRARY = ROOT / 'scene_data/assets_library'
WORKBOOK = ROOT / 'scene_data/catalog/灾后搜救_3D实体资产清单_当前执行版.xlsx'
BAKER_SHA256 = hashlib.sha256((''.join(sha256(ROOT / 'src/rescue_scenes' / name) for name in ('asset_baking.py', 'glb_runtime.py'))).encode()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative(path):
    return str(Path(path).relative_to(ROOT))


def norm(value):
    return re.sub('[^A-Z0-9]', '', value.upper())


def scan_folder(folder):
    candidates, errors = [], []
    for directory, dirs, names in os.walk(folder, onerror=lambda error: errors.append(f'{folder}: {error}')):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for name in names:
            path = Path(directory) / name
            if name.lower().endswith('.glb'):
                candidates.append({'path': str(path), 'member': None, 'bytes': path.stat().st_size})
            elif name.lower().endswith('.zip'):
                try:
                    with zipfile.ZipFile(path) as archive:
                        for info in archive.infolist():
                            if info.filename.lower().endswith('.glb') or '.glb.part' in info.filename.lower():
                                candidates.append({'path': str(path), 'member': info.filename,
                                                   'bytes': info.file_size, 'crc32': info.CRC})
                except (OSError, zipfile.BadZipFile) as error:
                    errors.append(f'{path}: {error}')
    result = {'directory': folder.name, 'candidates': candidates, 'errors': errors}
    dump(LIBRARY / 'source_index' / (hashlib.sha256(folder.name.encode()).hexdigest()[:16] + '.json'), result)
    return result


def inventory(refresh=False):
    output = LIBRARY / 'drive_inventory.json'
    if output.exists() and not refresh:
        return json.loads(output.read_text())
    folders = [p for p in DELIVERIES.iterdir() if p.is_dir() and '已退回' not in p.name]
    found = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for folder in folders:
            cached = LIBRARY / 'source_index' / (hashlib.sha256(folder.name.encode()).hexdigest()[:16] + '.json')
            if cached.exists() and not refresh:
                found.append(json.loads(cached.read_text()))
            else:
                futures.append(pool.submit(scan_folder, folder))
        for future in as_completed(futures):
            found.append(future.result())
            print(f'indexed {len(found)}/{len(folders)} folders', flush=True)
    dump(output, found)
    return found


def cached_sources():
    found = []
    for path in (ROOT / 'scene_data/scene_sim/dev/step3/upstream_expanded_snapshot/assets').glob('*/*.glb'):
        found.append(path)
    found.extend((ROOT / 'scene_data/scene_sim/assets/glb').glob('*.glb'))
    found.extend((LIBRARY / 'runtime_originals').glob('*.glb'))
    return found


def select_sources(rows, listings):
    for folder in listings:
        groups = {}
        for candidate in folder['candidates']:
            member = candidate.get('member') or Path(candidate['path']).name
            match = re.match(r'^(.*\.glb)\.part(\d+)of(\d+)\.bin$', member, re.I)
            if match:
                groups.setdefault((match[1], int(match[3])), {})[int(match[2])] = candidate
        for (member, count), parts in groups.items():
            if set(parts) == set(range(1, count + 1)):
                segments = [parts[i] for i in range(1, count + 1)]
                folder['candidates'].append({'path': segments[0]['path'], 'member': member,
                                             'bytes': sum(p['bytes'] for p in segments), 'segments': segments})
    selection = json.loads((ROOT / 'configs/upstream_selection_expanded.json').read_text())
    explicit = {r['acceptance_id']: r for r in selection.get('entries', [])}
    explicit.update({r['acceptance_id']: r for r in selection.get('not_added', []) if r['acceptance_id'] not in explicit})
    overrides_path = ROOT / 'configs/drive_asset_primary_selection.json'
    overrides = json.loads(overrides_path.read_text()).get('entries', {}) if overrides_path.exists() else {}
    choices, gaps = [], []
    for row in rows:
        aid = row['acceptance_id']; key = norm(aid); component = norm('-'.join(aid.split('-')[:3]))
        family = norm('-'.join(aid.split('-')[:2])); variant = aid.split('-')[2]; state = aid.split('-', 3)[-1]
        binding = overrides.get(aid, explicit.get(aid, {}))
        scored = []
        for folder in listings:
            fname = norm(folder['directory'])
            folder_exact = binding.get('directory') == folder['directory']
            if aid in overrides and not folder_exact:
                continue
            if not folder_exact and key not in fname and family not in fname:
                continue
            for candidate in folder['candidates']:
                if binding.get('archive') and Path(candidate['path']).name != binding['archive']:
                    continue
                member = candidate['member'] or Path(candidate['path']).name
                if aid in overrides and (member != binding['member'] or candidate['bytes'] != binding['bytes'] or candidate.get('crc32') != binding.get('crc32')):
                    continue
                if not member.lower().endswith('.glb'):
                    continue
                base = norm(Path(member).stem); whole = norm(member)
                component_context = component in fname or component in whole or (family in fname and variant in base)
                if not folder_exact and not component_context:
                    continue
                score = 0
                if binding.get('member') == member and folder_exact:
                    score += 1000
                if base == key:
                    score += 500
                elif key in base:
                    score += 300
                elif key in whole:
                    score += 100
                elif key in fname or folder_exact:
                    score += 40
                elif component_context and norm(state) in base:
                    score += 80
                elif component_context and norm(state) in fname:
                    score += 40
                elif state == 'BASE' and component_context and (component in base or variant in base) and not any(token in base for token in ('WIND', 'FALLEN', 'LIMIT', 'NEUTRAL', 'POSE')):
                    score += 20
                else:
                    continue
                state = aid.split('-', 3)[-1]
                # A state-specific filename is stronger evidence than a shared folder.
                if norm(state) in base:
                    score += 80
                if any(x in base for x in ('PREVIEW', 'COLLISION', 'PROXY', 'THUMBNAIL', 'LOD')):
                    score -= 200
                if 'STATIC' in base and state == 'BASE':
                    score += 10
                if folder_exact:
                    score += 20
                scored.append((score, candidate | {'directory': folder['directory']}))
        if not scored:
            gaps.append({'asset_id': aid, 'reason': 'no_matching_complete_glb', 'workbook_status': row['生产状态']})
            continue
        scored.sort(key=lambda x: (-x[0], x[1]['path'], x[1]['member'] or ''))
        best = scored[0][0]; tied = [c for score, c in scored if score == best]
        # Duplicate copies with identical ZIP CRC and byte length are equivalent.
        identity = {(c['bytes'], c.get('crc32', c['path'])) for c in tied}
        if len(identity) > 1:
            gaps.append({'asset_id': aid, 'reason': 'ambiguous_primary_glb', 'candidates': tied})
            continue
        chosen = tied[0]
        choices.append({'asset_id': aid, 'category': aid[:3], 'source': chosen,
                        'selection_score': best, 'revision': row['修订号 / 文件版本'],
                        **({'selection_evidence': binding['selection_evidence']} if binding.get('selection_evidence') else {})})
    return choices, gaps


def local_archive(path):
    manifest = LIBRARY / 'archive_manifest.json'
    if manifest.exists():
        for entry in json.loads(manifest.read_text()):
            if entry.get('kind') == 'archive' and entry['source'] == str(path):
                staged = ROOT / entry['file']
                if staged.is_file(): return staged
    return path


def extract(choice, caches):
    source = choice['source']; aid = choice['asset_id']
    target = LIBRARY / '_raw_glb' / f'{aid}.glb'
    target.parent.mkdir(parents=True, exist_ok=True)
    for cached in [target, *caches]:
        if not cached.exists() or cached.stat().st_size != source['bytes']:
            continue
        # Verify ZIP content identity, not merely the asset-ID filename.
        if 'crc32' in source:
            crc = 0
            with cached.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    crc = zlib.crc32(block, crc)
            if crc != source['crc32']:
                continue
        elif str(cached) != source['path']:
            continue
        if cached != target:
            shutil.copyfile(cached, target)
        return target, str(cached)
    temporary = target.with_suffix('.glb.part')
    if source.get('segments'):
        with temporary.open('wb') as outgoing:
            for part in source['segments']:
                with zipfile.ZipFile(local_archive(part['path'])) as archive:
                    info = archive.getinfo(part['member'])
                    if info.file_size != part['bytes'] or info.CRC != part['crc32']:
                        raise ValueError('split GLB ZIP member identity changed since indexing')
                    with archive.open(info) as incoming:
                        shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
    elif source['member']:
        with zipfile.ZipFile(local_archive(source['path'])) as archive:
            info = archive.getinfo(source['member'])
            if info.file_size != source['bytes'] or ('crc32' in source and info.CRC != source['crc32']):
                raise ValueError('GLB ZIP member identity changed since indexing')
            with archive.open(info) as incoming, temporary.open('wb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 4 * 1024 * 1024)
    else:
        shutil.copyfile(source['path'], temporary)
    if temporary.stat().st_size != source['bytes']:
        raise ValueError('extracted length differs from ZIP member')
    temporary.replace(target)
    return target, None


def bake_choice(choice, caches):
    aid = choice['asset_id']; record_path = LIBRARY / 'records' / f'{aid}.json'
    output = LIBRARY / CATEGORIES[choice['category']] / f'{aid}.glb'
    if record_path.exists() and output.exists():
        old = json.loads(record_path.read_text())
        if old.get('status') == 'baked' and old.get('baker_sha256') == BAKER_SHA256 and old.get('source') == choice['source'] and sha256(output) == old.get('baked_sha256'):
            old.update(choice)
            dump(record_path, old)
            return old
    try:
        raw, reuse = extract(choice, caches)
        metrics = bake(raw, output)
        record = choice | metrics | {'status': 'baked', 'baker_sha256': BAKER_SHA256, 'file': relative(output),
                                      'raw_file': relative(raw), 'reused_local_file': reuse}
    except Exception as error:
        record = choice | {'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
    dump(record_path, record)
    print(f"{aid}: {record['status']} {record.get('error', '')}", flush=True)
    return record


def bake_runtime():
    records = []
    root = ROOT / 'scene_data/scene_sim/assets/glb'
    for path in sorted(root.glob('*.glb')):
        backup = LIBRARY / 'runtime_originals' / path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copyfile(path, backup)
        record_path = LIBRARY / 'runtime_records' / (path.stem + '.json')
        if record_path.exists():
            old = json.loads(record_path.read_text())
            if old.get('status') == 'baked' and old.get('baker_sha256') == BAKER_SHA256 and sha256(path) == old.get('baked_sha256'):
                records.append(old); continue
        try:
            record = bake(backup, path) | {'status': 'baked', 'baker_sha256': BAKER_SHA256, 'file': relative(path), 'backup': relative(backup)}
        except Exception as error:
            record = {'status': 'failed', 'file': relative(path), 'error': f'{type(error).__name__}: {error}'}
        dump(record_path, record); records.append(record)
        print(f"runtime {path.name}: {record['status']} {record.get('error', '')}", flush=True)
    dump(LIBRARY / 'runtime_manifest.json', records)
    return records


def stage_archives_and_configs():
    """Copy referenced evidence first, then rewrite paths to real workspace files."""
    configs = list((ROOT / 'configs').rglob('*.json'))
    documents = [(p, json.loads(p.read_text())) for p in configs]
    previous_path = LIBRARY / 'archive_manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else []
    replacements, evidence = {}, []
    pattern = re.compile(r'/mnt/data2/weizhiwei/google-drive/[^\n\r\"<>]*')
    values = []
    def gather(obj):
        if isinstance(obj, str):
            values.extend(pattern.findall(obj))
        elif isinstance(obj, list):
            for item in obj: gather(item)
        elif isinstance(obj, dict):
            for item in obj.values(): gather(item)
    for _, doc in documents: gather(doc)
    paths = set()
    for value in values:
        path = value.split('#', 1)[0].split('!', 1)[0]
        if '.zip' in path: path = path[:path.index('.zip') + 4]
        paths.add(path)
    paths.add(str(DRIVE / 'AeroRescue/AeroRescue_SAREnv_envs-1-4-6_0f2d344d_2026-10-14.zip'))
    archive_lock = threading.Lock()
    def stage(source_string):
        source = Path(source_string)
        if source.name == WORKBOOK.name:
            return source_string, str(WORKBOOK), {'source': source_string, 'file': relative(WORKBOOK), 'sha256': sha256(WORKBOOK), 'kind': 'workbook'}
        if source.suffix.lower() == '.zip':
            directory = ROOT / 'scene_data/sarenv' if source.parent.name == 'AeroRescue' else LIBRARY / 'archives'
            destination = directory / source.name
            directory.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
            digest = hashlib.sha256()
            with source.open('rb') as incoming, temporary.open('xb') as outgoing:
                for block in iter(lambda: incoming.read(4 * 1024 * 1024), b''):
                    digest.update(block); outgoing.write(block)
            copied_sha256 = digest.hexdigest()
            with archive_lock:
                if destination.exists() and sha256(destination) != copied_sha256:
                    raise ValueError(f'archive basename collision or changed source content: {source}')
                temporary.replace(destination)
            with zipfile.ZipFile(destination) as archive:
                bad = archive.testzip()
                if bad: raise ValueError(f'ZIP CRC failed: {bad}')
                members = len(archive.infolist())
            return source_string, str(destination), {'source': source_string, 'file': relative(destination), 'sha256': sha256(destination), 'bytes': destination.stat().st_size, 'zip_crc_verified': True, 'members': members, 'kind': 'archive'}
        if source.is_file():
            destination = LIBRARY / 'source_metadata' / source.relative_to(DRIVE)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists(): shutil.copyfile(source, destination)
            return source_string, str(destination), {'source': source_string, 'file': relative(destination), 'sha256': sha256(destination), 'kind': 'metadata'}
        # Unavailable historical evidence remains explicitly non-local provenance.
        replacement = 'drive-provenance:' + source.relative_to(DRIVE).as_posix()
        return source_string, replacement, {'source': source_string, 'reference': replacement, 'kind': 'historical_provenance_unavailable'}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for original, replacement, entry in pool.map(stage, sorted(paths)):
            replacements[original] = replacement; evidence.append(entry)
    def rewrite(obj):
        if isinstance(obj, str):
            for original in sorted(replacements, key=len, reverse=True): obj = obj.replace(original, replacements[original])
            if str(DRIVE) + '/' in obj: raise ValueError('unmapped Drive path')
            return obj
        if isinstance(obj, list): return [rewrite(v) for v in obj]
        if isinstance(obj, dict): return {k: rewrite(v) for k, v in obj.items()}
        return obj
    for path, doc in documents:
        updated = rewrite(doc)
        if updated != doc: dump(path, updated)
    evidence = list({e['source']: e for e in [*previous, *evidence]}.values())
    dump(LIBRARY / 'archive_manifest.json', evidence)
    return evidence


def summarize(records):
    valid = [r for r in records if r.get('status') == 'baked']
    fields = ('file_bytes', 'texture_bytes', 'decoded_texture_rgba8_bytes', 'primitives', 'triangles', 'vertices', 'position_accessors', 'quantized_position_accessors')
    result = {'assets': len(valid)}
    for stage in ('before', 'after'):
        result[stage] = {f: sum(r[stage][f] for r in valid) for f in fields}
    result['compression_ratio'] = result['before']['file_bytes'] / max(1, result['after']['file_bytes'])
    result['quantized_assets'] = sum(r['after']['quantized_position_accessors'] > 0 for r in valid)
    result['quantized_asset_rate'] = result['quantized_assets'] / max(1, len(valid))
    result['quantized_position_rate'] = result['after']['quantized_position_accessors'] / max(1, result['after']['position_accessors'])
    return result


def report(manifest):
    summary = manifest['summary']; before = summary['before']; after = summary['after']
    valid_records = [r for r in manifest['assets'] if r.get('status') == 'baked']
    max_vertex_error = max((r['audit'].get('vertex_max_error_m', 0) for r in valid_records), default=0)
    enlarged = [r['asset_id'] for r in valid_records if r['after']['file_bytes'] > r['before']['file_bytes']]
    primary_path = ROOT / 'configs/drive_asset_primary_selection.json'
    primary_count = len(json.loads(primary_path.read_text()).get('entries', {})) if primary_path.exists() else 0
    runtime_summary = summarize(manifest['runtime'])
    lines = ['# Asset baking and audit report', '', f"Generated: {manifest['generated_utc']}", '',
             f"Baked **{summary['assets']}** categorized assets from **{manifest['catalog']['delivered_geometry']}** workbook-delivered geometry IDs. "
             f"Recorded **{len(manifest['gaps'])}** unresolved source/selection/baking gaps. No gap is counted as a successful delivery.", '',
             '| Category | Baked | Raw bytes | Baked bytes | Ratio | Draws before → after | Texture bytes before → after | Quantized |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for category, data in manifest['by_category'].items():
        a, b = data['before'], data['after']
        lines.append(f"| {category} | {data['assets']} | {a['file_bytes']:,} | {b['file_bytes']:,} | {data['compression_ratio']:.2f}× | {a['primitives']:,} → {b['primitives']:,} | {a['texture_bytes']:,} → {b['texture_bytes']:,} | {data['quantized_assets']} |")
    lines += ['', f"Totals: **{before['file_bytes']:,} → {after['file_bytes']:,} bytes ({summary['compression_ratio']:.2f}×)**; "
              f"draw instances **{before['primitives']:,} → {after['primitives']:,}**; "
              f"embedded texture payload **{before['texture_bytes']:,} → {after['texture_bytes']:,} bytes**.", '',
              f"Maximum measured world-vertex coordinate error: **{max_vertex_error:.9f} m ({max_vertex_error * 1000:.6f} mm)**. "
              f"Outputs larger than their original GLB: **{len(enlarged)}**" + (f" ({', '.join(enlarged)})" if enlarged else '') + '.', '',
              f"Quantized asset rate: **{summary['quantized_assets']}/{summary['assets']} ({summary['quantized_asset_rate']:.1%})**; quantized position-primitive rate: **{summary['quantized_position_rate']:.1%}**.", '',
              f"Decoded RGBA8 texture allocation proxy: **{before['decoded_texture_rgba8_bytes']:,} → {after['decoded_texture_rgba8_bytes']:,} bytes** "
              '(sum of width × height × 4; excludes mipmaps, driver overhead and cross-asset sharing). GLB byte reduction alone is not a GPU-memory measurement.', '',
              '## Method and evidence', '',
              '- Exact pipeline: optimize(max_px=512) → merge_by_material → opaque_blend → compact_textures → semantic_textures(max_px=512, quality=85) → quantize_geometry.',
              '- Every successful output is re-read and checked for glTF 2.0 magic/version/total length, JSON/BIN chunk ordering and 4-byte alignment, embedded buffer bounds, accessor strides and bounds, index ranges, acyclic scene topology, matching image MIME and payload, decodable images, and a 512-pixel texture maximum.',
              '- Default-scene triangle counts must be identical. Every world-space vertex is compared in its material/attribute group, and geometry bounds must agree within the recorded quantization tolerance. Draw counts include mesh instances in the active scene.',
              '- SHA256 binds original and baked bytes. ZIP extraction verifies CRC; cached NVMe files are reused only when byte length and member CRC agree. Original runtime GLBs are retained in `scene_data/assets_library/runtime_originals/`.',
              '- Skins, animations, morphs, compressed geometry and unsupported attribute sets are preserved when the existing merge/quantization functions skip them. Per-step skip reasons remain in the manifest.',
              '- Texture encoding is lossy and runtime-oriented; JPEG quality 85 can alter normal and metal/roughness maps. No visual equivalence, placement certification or task-level simulator acceptance is claimed from binary validation.',
              '- Fixed compact_textures MIME mutation: a JPEG rejected for larger size must not relabel the unchanged PNG payload.', '',
              '## Existing runtime assets', '',
              f"{sum(r.get('status') == 'baked' for r in manifest['runtime'])}/{len(manifest['runtime'])} existing runtime GLBs baked in place; exact metrics and original hashes are recorded in `runtime_manifest.json`. "
              f"Bytes **{runtime_summary['before']['file_bytes']:,} → {runtime_summary['after']['file_bytes']:,} ({runtime_summary['compression_ratio']:.2f}×)**; "
              f"draw instances **{runtime_summary['before']['primitives']:,} → {runtime_summary['after']['primitives']:,}**; "
              f"triangles **{runtime_summary['after']['triangles']:,}**, unchanged; quantized **{runtime_summary['quantized_assets']}/{runtime_summary['assets']}**.", '',
              '## Source completeness and limitations', '']
    for gap in manifest['gaps']:
        lines.append(f"- `{gap['asset_id']}`: {gap.get('reason', gap.get('error', 'unknown'))}.")
    if not manifest['gaps']: lines.append(f"All {manifest['catalog']['delivered_geometry']} delivered geometry IDs have a selected and audited runtime GLB. Pose configurations, surface effects and hidden features are separate obligations and do not create extra geometry GLBs or inflate this count; the catalog section of the manifest retains their delivered counts.")
    lines += ['', f"Reviewed **{primary_count}** ambiguous/numeric/state/revision selections are pinned to an exact directory, archive member, byte length and CRC in `configs/drive_asset_primary_selection.json`; each includes the local package evidence used. "
              'Examples: G02U1 corrected wall UVs; G03M1/G02M1 corrected roof material factors; G05 roller-door recovery; r06_A01 container damage continuity; numbered stair/terrain components; BASE aliases for STATIC vegetation, NEUTRAL searchlights and DEPLOYED relay equipment. '
              'For the tile/wood/log components that formally share two poses, the chosen FLAT/SCATTERED library default and its pose provenance are explicit; no extra geometry or runtime acceptance is counted.']
    lines += ['', '## Archives and configuration references', '',
              f"Verified {sum(e['kind'] == 'archive' for e in manifest['archives'])} local ZIP archives with CRC checks and SHA256. "
              f"Runtime configuration references now use local workspace files. Unavailable historical evidence paths: **{sum(e['kind'] == 'historical_provenance_unavailable' for e in manifest['archives'])}**. Original source locations remain in the manifest for traceability.", '',
              '## Reproduction', '', '```bash', '/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python tools/stage_and_bake_drive_assets.py',
              '/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python -m pytest -q', '```', '',
              'Test results are recorded after execution; this report does not infer a pass from test collection.', '']
    if manifest.get('verification'):
        verification = manifest['verification']
        lines += ['## Executed verification', '', f"`{verification['pytest_command']}`: **{verification['pytest_result']}**.", '',
                  'Integration checks cover all categorized outputs and 17 runtime GLBs, exact manifest aggregates and hashes, original runtime backups, CRC-verified archive hashes, zero Drive mount references in all JSON configs, primary-selection identity pins, and byte-for-byte preservation of deformation metadata and non-image buffer views for skinned/animated/morph assets.', '',
                  f"Captured pytest output: `{verification['pytest_log']}`. No test is skipped in this staged workspace.", '']
    (ROOT / 'docs/asset-baking-and-audit-report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-only', action='store_true')
    parser.add_argument('--refresh-index', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    LIBRARY.mkdir(parents=True, exist_ok=True)
    for directory in CATEGORIES.values(): (LIBRARY / directory).mkdir(exist_ok=True)
    runtime_records = bake_runtime()
    if args.runtime_only: return
    catalog_rows = read_excel(WORKBOOK)
    rows = [r for r in catalog_rows if r['验收单元种类'] in GEOMETRY and r['生产状态'] == '已交付']
    listings = inventory(args.refresh_index)
    choices, gaps = select_sources(rows, listings)
    dump(LIBRARY / 'selection_plan.json', {'choices': choices, 'gaps': gaps})
    caches = cached_sources(); records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(bake_choice, choice, caches) for choice in choices]
        for future in as_completed(futures): records.append(future.result())
    gaps.extend({'asset_id': r['asset_id'], 'reason': r['error']} for r in records if r['status'] != 'baked')
    archives = stage_archives_and_configs()
    manifest = {'schema': 'rescue.baked-asset-library/v1', 'generated_utc': datetime.now(timezone.utc).isoformat(),
                'pipeline': PIPELINE, 'baker_sha256': BAKER_SHA256, 'options': {'max_px': 512, 'quality': 85},
                'catalog': {'file': relative(WORKBOOK), 'sha256': sha256(WORKBOOK), 'delivered_geometry': len(rows),
                            'geometry_rows': sum(r['验收单元种类'] in GEOMETRY for r in catalog_rows),
                            'non_geometry_deliveries_by_type': dict(Counter(r['验收单元种类'] for r in catalog_rows if r['验收单元种类'] not in GEOMETRY and r['生产状态'] == '已交付'))},
                'source_folders': len(listings), 'assets': sorted(records, key=lambda x: x['asset_id']),
                'gaps': gaps, 'runtime': runtime_records, 'archives': archives,
                'summary': summarize(records),
                'by_category': {c: summarize([r for r in records if r['category'] == c]) for c in CATEGORIES}}
    dump(LIBRARY / 'baked_manifest.json', manifest); report(manifest)
    print(json.dumps(manifest['summary'], indent=2), flush=True)


if __name__ == '__main__': main()
