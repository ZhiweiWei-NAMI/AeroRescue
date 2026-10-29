"""Blender-side display LOD decimation (run: blender -b --factory-startup --python lod_blender.py -- <job.json>).

job.json: {"input": <glb whose materials are named M<index>>, "levels": [{"ratio": r, "output": <glb>}, ...]}
Each level re-imports the input (vertices welded) and decimates only connected parts with more than
MIN_PART faces: first a planar dissolve (shape-exact), then edge-collapse decimation towards the level's
ratio. Small loose parts (debris, tiles, frames, rubble) are kept unchanged so damage pieces never vanish.
Exports texture-less M<index> materials, so the caller can rebind the original materials by index.
"""
import json
import math
import sys

import bmesh
import bpy

MIN_PART = 200


def select_large_parts(bm):
    """Select faces of connected components with more than MIN_PART faces; return (total, selected) face counts."""
    seen, total, chosen = set(), len(bm.faces), 0
    for face in bm.faces:
        face.select_set(False)
    for face in bm.faces:
        if face.index in seen:
            continue
        stack, part = [face], []
        seen.add(face.index)
        while stack:
            f = stack.pop(); part.append(f)
            for edge in f.edges:
                for g in edge.link_faces:
                    if g.index not in seen:
                        seen.add(g.index); stack.append(g)
        if len(part) > MIN_PART:
            chosen += len(part)
            for f in part:
                f.select_set(True)
    return total, chosen


job = json.load(open(sys.argv[sys.argv.index('--') + 1]))
for level in job['levels']:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=job['input'], merge_vertices=True)  # weld split vertices so decimation can reduce flat-shaded parts
    for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
        bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True); bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_mode(type='FACE')
        bm = bmesh.from_edit_mesh(obj.data); bm.faces.ensure_lookup_table()
        total, chosen = select_large_parts(bm); bmesh.update_edit_mesh(obj.data)
        if chosen:
            # Ratio on the large parts so the whole asset approaches the level ratio where possible.
            target = max(level['ratio'] * total - (total - chosen), 0) / chosen
            bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(1.0), delimit={'MATERIAL', 'SEAM', 'UV'})
            bpy.ops.mesh.quads_convert_to_tris()
            bpy.ops.mesh.decimate(ratio=min(1.0, max(target, 0.02)))
        bpy.ops.object.mode_set(mode='OBJECT')
    options = dict(filepath=level['output'], export_format='GLB', export_materials='EXPORT', export_texcoords=True,
                   export_normals=True, export_tangents=False, export_yup=True, export_apply=True, export_animations=False,
                   export_skins=False, export_morph=False, export_cameras=False, export_lights=False, export_extras=False)
    known = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    bpy.ops.export_scene.gltf(**{k: v for k, v in options.items() if k in known})
