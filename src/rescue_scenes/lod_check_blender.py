"""Render one display GLB as the AAS viewer would frame it at distance D (1280x800, 45 deg vertical FOV),
cropped to the object, for full-vs-LOD comparison at the LOD switch distance.
usage: blender -b --factory-startup --python lod_check_blender.py -- <glb> <out.png> <distance_m> <rear|front> [samples] [frame.json]
frame.json (optional, read if present, written if absent): {center, radius, ground} so a level is framed exactly like the full model."""
import bpy, sys, math
from mathutils import Vector
import json, os
args = sys.argv[sys.argv.index('--') + 1:]; glb, out, dist, view = args[:4]; dist = float(dist)
samples = int(args[4]) if len(args) > 4 else 32; frame = args[5] if len(args) > 5 else None
bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=glb)
objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))); hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
center = (lo + hi) / 2; radius = (hi - lo).length / 2
if frame and os.path.exists(frame):
    f = json.load(open(frame)); center = Vector(f['center']); radius = f['radius']; lo = Vector((0, 0, f['ground']))
elif frame:
    json.dump({'center': list(center), 'radius': radius, 'ground': lo.z}, open(frame, 'w'))
sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = samples; sc.cycles.seed = 0; sc.render.threads_mode = 'FIXED'; sc.render.threads = 4
sc.render.resolution_x, sc.render.resolution_y = 1280, 800; sc.render.filepath = out
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True; w.node_tree.nodes['Background'].inputs[1].default_value = 0.6
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN')); sun.data.energy = 3; sun.rotation_euler = (math.radians(50), 0, math.radians(30)); sc.collection.objects.link(sun)
g = bpy.data.meshes.new('g'); s = radius * 4; g.from_pydata([(-s, -s, lo.z), (s, -s, lo.z), (s, s, lo.z), (-s, s, lo.z)], [], [(0, 1, 2, 3)]); sc.collection.objects.link(bpy.data.objects.new('g', g))
direction = Vector((-0.45, 0.75, 0.48) if view == 'rear' else (0.45, -0.75, 0.48)).normalized()
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam
cam.location = center + direction * dist; cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam.data.sensor_fit = 'VERTICAL'; cam.data.angle = math.radians(45)
half = min(0.5, 1.6 * radius / (2 * dist * math.tan(math.radians(22.5))))  # crop to ~1.6x the object's projected radius
sc.render.use_border = True; sc.render.use_crop_to_border = True
sc.render.border_min_x, sc.render.border_max_x = 0.5 - half * 800 / 1280, 0.5 + half * 800 / 1280
sc.render.border_min_y, sc.render.border_max_y = 0.5 - half, 0.5 + half
bpy.ops.render.render(write_still=True)
