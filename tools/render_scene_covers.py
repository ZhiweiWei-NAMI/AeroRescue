#!/usr/bin/env python3
"""Render a scene cover image from a display GLB with headless Blender (Cycles CPU).

Run one scene per process, e.g.:

    blender -b --factory-startup -t 16 \
        --python tools/render_scene_covers.py -- \
        --sid SC01 --glb /abs/path/SC01.glb \
        --jpg  $FD/data/rescue_covers/SC01.jpg \
        --thumb src/rescue_scenes/templates/rescue-sc01/thumbnail.png \
        --samples 48 --width 1280 --height 720

The GLB is imported into an empty scene (the glTF importer converts Y-up),
an oblique camera is placed from the south-west at ~35 degrees elevation to
frame the whole world bounding box, and a sun + Nishita sky light the scene.
Cycles renders on CPU with denoising and AgX.  The render is written as a
display-referred PNG by Blender, then system python3 + Pillow produces the
full-resolution JPEG (quality 88) and the 640x360 PNG thumbnail.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser(description="Render an AeroRescue scene cover")
    p.add_argument("--sid", required=True)
    p.add_argument("--glb", required=True)
    p.add_argument("--jpg", required=True)
    p.add_argument("--thumb", required=True)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--margin", type=float, default=1.06)
    p.add_argument("--sun-energy", type=float, default=4.0)
    p.add_argument("--sky-strength", type=float, default=0.5)
    p.add_argument("--elevation-deg", type=float, default=35.0)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--probe", action="store_true", help="Import and report only")
    return p.parse_args(argv)


def world_bbox():
    """World-space axis-aligned bounds of all renderable mesh objects."""
    lo = Vector((math.inf, math.inf, math.inf))
    hi = Vector((-math.inf, -math.inf, -math.inf))
    found = 0
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        for c in corners:
            lo.x = min(lo.x, c.x)
            lo.y = min(lo.y, c.y)
            lo.z = min(lo.z, c.z)
            hi.x = max(hi.x, c.x)
            hi.y = max(hi.y, c.y)
            hi.z = max(hi.z, c.z)
        found += 1
    if found == 0 or not all(map(math.isfinite, (*lo, *hi))):
        # Fallback: unit cube at the origin.
        lo = Vector((-5.0, -5.0, 0.0))
        hi = Vector((5.0, 5.0, 10.0))
    return lo, hi, found


def setup_camera(
    center: Vector, size: Vector, diagonal: float, elevation_deg: float, margin: float
):
    """Place a camera south-west of the bbox, elevated, framed on the whole bbox."""
    cam_data = bpy.data.cameras.new("CoverCam")
    cam_data.lens = 35.0
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("CoverCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    elev = math.radians(elevation_deg)
    # Direction from the scene centre towards the camera: south-west, elevated.
    az = math.radians(225.0)  # SW: -X, -Y
    horiz = math.cos(elev)
    direction = Vector((horiz * math.cos(az), horiz * math.sin(az), math.sin(elev)))
    direction.normalize()

    cam_data.clip_start = max(0.05, diagonal * 1.0e-4)
    cam_data.clip_end = diagonal * 8.0 + 100.0

    # Build the camera basis: look at the centre, Y up.
    quat = (-direction).to_track_quat("-Z", "Y")
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = quat
    right = quat @ Vector((1.0, 0.0, 0.0))
    up = quat @ Vector((0.0, 1.0, 0.0))
    forward = quat @ Vector((0.0, 0.0, -1.0))

    tan_h = math.tan(cam_data.angle_x * 0.5)
    tan_v = math.tan(cam_data.angle_y * 0.5)

    # For each bbox corner p (relative to centre), camera at centre + d*direction:
    #   depth = d - dot(p, direction)
    #   |x| <= tan_h * depth, |y| <= tan_v * depth
    # -> d >= dot(p, direction) + |x|/tan_h  (and likewise for y).
    d_req = 0.0
    half = size * 0.5
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rel = Vector((sx * half.x, sy * half.y, sz * half.z))
                x = abs(rel.dot(right))
                y = abs(rel.dot(up))
                along = rel.dot(direction)
                d_req = max(d_req, along + x / tan_h, along + y / tan_v)

    distance = d_req * margin
    cam.location = center + direction * distance
    cam_data.clip_start = max(0.05, min(diagonal * 1.0e-3, distance * 0.01))
    cam_data.clip_end = distance + diagonal * 4.0 + 100.0
    return cam, distance


def setup_lighting(sun_energy: float, sky_strength: float):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("CoverWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = sky_strength

    # Soft vertical sky gradient.  A Nishita sky turns black below the horizon,
    # which leaves floating terrain scenes sitting in a black void; the
    # gradient keeps the backdrop readable from every camera elevation.
    tex = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -1.0
    mr.inputs["From Max"].default_value = 1.0
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.30, 0.38, 0.46, 1.0)
    ramp.color_ramp.elements[1].position = 0.52
    ramp.color_ramp.elements[1].color = (0.64, 0.72, 0.82, 1.0)
    top = ramp.color_ramp.elements.new(1.0)
    top.color = (0.16, 0.34, 0.60, 1.0)

    nt.links.new(tex.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    sun_data = bpy.data.lights.new("CoverSun", type="SUN")
    sun_data.energy = sun_energy
    sun_data.angle = math.radians(2.0)
    sun = bpy.data.objects.new("CoverSun", sun_data)
    scene.collection.objects.link(sun)
    # Sun in the south-east so the camera (south-west) gets side lighting.
    light_dir = Vector((-0.55, 0.55, -0.72)).normalized()
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = light_dir.to_track_quat("-Z", "Y")


def setup_render(width: int, height: int, samples: int, threads: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception as exc:  # pragma: no cover
        print("DENOISER_FALLBACK", repr(exc))
    try:
        scene.cycles.denoising_use_gpu = False
    except Exception:
        pass
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.volume_bounces = 1
    scene.cycles.transparent_max_bounces = 4
    scene.render.use_persistent_data = False

    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15

    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


PIL_CONVERT = r"""
import sys
from PIL import Image
src, jpg, thumb, tw, th, quality = sys.argv[1:7]
im = Image.open(src).convert("RGB")
im.save(jpg, "JPEG", quality=int(quality), optimize=True, subsampling=1)
im.resize((int(tw), int(th)), Image.LANCZOS).save(thumb, "PNG", optimize=True)
print("PIL_OK", jpg, thumb)
"""


def save_outputs(render_png: str, jpg: str, thumb: str):
    img = bpy.data.images.get("Render Result")
    if img is None:
        raise RuntimeError("no Render Result image")
    img.save_render(render_png, scene=bpy.context.scene)

    py = shutil.which("python3") or "/usr/bin/python3"
    tw, th = 640, 360
    cmd = [py, "-c", PIL_CONVERT, render_png, jpg, thumb, str(tw), str(th), "88"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Pillow conversion failed: %s\n%s\n%s"
            % (" ".join(cmd), proc.stdout, proc.stderr)
        )
    print(proc.stdout.strip())


def main() -> int:
    args = parse_args()
    glb = os.path.realpath(args.glb)
    if not os.path.isfile(glb):
        print("RENDER_FAIL %s missing glb %s" % (args.sid, glb))
        return 2

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)

    if not hasattr(bpy.ops.import_scene, "gltf"):
        print("RENDER_FAIL %s glTF importer unavailable" % args.sid)
        return 3
    bpy.ops.import_scene.gltf(filepath=glb)

    lo, hi, n_mesh = world_bbox()
    center = (lo + hi) * 0.5
    size = hi - lo
    diagonal = size.length
    if diagonal <= 1.0e-6:
        diagonal = 20.0
    print(
        "SCENE %s meshes=%d bbox_min=%s bbox_max=%s diag=%.3f"
        % (
            args.sid,
            n_mesh,
            tuple(round(v, 3) for v in lo),
            tuple(round(v, 3) for v in hi),
            diagonal,
        )
    )

    if args.probe:
        return 0

    cam, distance = setup_camera(
        center, size, diagonal, args.elevation_deg, args.margin
    )
    bpy.context.scene.camera = cam
    print(
        "CAMERA %s loc=%s dist=%.3f clip=(%.3f, %.3f)"
        % (
            args.sid,
            tuple(round(v, 3) for v in cam.location),
            distance,
            cam.data.clip_start,
            cam.data.clip_end,
        )
    )

    setup_lighting(args.sun_energy, args.sky_strength)
    setup_render(args.width, args.height, args.samples, args.threads)

    for path in (args.jpg, args.thumb):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    tmpdir = tempfile.mkdtemp(prefix="aerorescue_cover_%s_" % args.sid)
    render_png = os.path.join(tmpdir, "%s_render.png" % args.sid)
    try:
        t_render = time.time()
        bpy.ops.render.render(write_still=False)
        render_seconds = time.time() - t_render
        save_outputs(render_png, args.jpg, args.thumb)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(
        "RENDER_OK sid=%s jpg=%s jpg_bytes=%d thumb=%s thumb_bytes=%d "
        "render_seconds=%.1f total_seconds=%.1f"
        % (
            args.sid,
            args.jpg,
            os.path.getsize(args.jpg),
            args.thumb,
            os.path.getsize(args.thumb),
            render_seconds,
            time.time() - t0,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback

        traceback.print_exc()
        sid = "?"
        if "--sid" in sys.argv:
            try:
                sid = sys.argv[sys.argv.index("--sid") + 1]
            except Exception:
                pass
        print("RENDER_FAIL %s %r" % (sid, exc))
        sys.exit(1)
