"""Read-only adapter: upstream ground-template sidecar -> rescue.upstream-semantics/v1.

Input: a ground-template deliverable directory (e.g. GT02_HILLSIDE_BACKGROUND_SOURCE) whose
metadata/regions.json uses schema "aerorescue-ground-template/1.0" (ENU metres, Blender [E,N,U]).
Only the 2-D semantics of that sidecar are used (U07: elevation does not drive this phase):
classed regions, road/stair corridors with carriageway widths, junctions, placement plots,
tree anchors and the detention pond. The raw GLB terrain mesh is never loaded.

The upstream deliverable deliberately supplies NO buildings, people, damaged roads, vehicles
or debris. Buildings are therefore placed here as explicitly *authored* entities on the
upstream placement plots (anchor, heading, max footprint), and their missing semantic
attributes are left for semantics.ingest() to fill and report in `missing`. People, damage,
obstacles and hazards come from the world-truth generator (G02-G07), never from the terrain.

Geometric operations performed by this adapter are recorded in source.note / adapter_log:
origin shift into [0,W]x[0,H], centreline down-sampling, and snapping of corridor endpoints
onto nearby corridors (<= 3 m) so the road graph is connected. This is not certified
navigation; it is a declared adapter assumption.

Usage: python3 -m rescue_env.world.upstream_gt <deliverable_dir> <out.json>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

SNAP_M = 3.0
SPACING_M = 2.0

# upstream region class -> (semantic class or None to skip, note)
REGION_CLASS = {
    "natural_ground": ("slope", "上游山坡自然地面"),
    "embankment": ("slope", "上游路堤边坡"),
    "retaining_structure_zone": ("facility", "上游挡土结构区"),
    "paved_public": ("open_area", "上游硬化公共场地"),
    "green_public": ("open_area", "上游绿地/滞洪洼地"),
    "channel": ("open_area", "上游排水沟渠（非水域）"),
    "drainage_swale": ("open_area", "上游排水浅沟（非水域）"),
    "bank": ("open_area", "上游植被保留带（保留区，非已确认树木）"),
    "placement_plot": ("open_area", "上游建筑候选地块"),
    "carriageway": (None, "由道路中心线表示"),
    "corridor_reserve": (None, "道路保留带并入道路"),
    "stair_ramp_reserve": (None, "由台阶中心线表示"),
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _downsample(points, spacing=SPACING_M):
    out = [points[0]]
    for p in points[1:-1]:
        if _dist(p, out[-1]) >= spacing:
            out.append(p)
    if _dist(points[-1], out[-1]) > 1e-6:
        out.append(points[-1])
    return out


def _project(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return (a[0] + t * dx, a[1] + t * dy), t


def _ring(poly):
    pts = [(float(x), float(y)) for x, y, *_ in poly]
    if len(pts) > 2 and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def convert(deliverable: str | Path) -> dict:
    root = Path(deliverable)
    src = root / "metadata" / "regions.json"
    raw = src.read_bytes()
    d = json.loads(raw)
    if d.get("schema") != "aerorescue-ground-template/1.0":
        raise ValueError(f"unsupported sidecar schema {d.get('schema')!r}")
    xs = [p[0] for r in d["regions"] for p in r["polygon_xy_m"]]
    ys = [p[1] for r in d["regions"] for p in r["polygon_xy_m"]]
    ox, oy = -min(xs), -min(ys)
    W, H = max(xs) + ox, max(ys) + oy
    sh = lambda p: (round(float(p[0]) + ox, 3), round(float(p[1]) + oy, 3))  # noqa: E731
    log = [f"origin shift (+{ox:.3f}, +{oy:.3f}) m: upstream local ENU -> [0,{W:.1f}]x[0,{H:.1f}]"]
    entities = []

    # Corridors: roads and stairs as polylines (2-D), down-sampled, then endpoint snapping.
    lines = {}
    for c in d["corridors"]:
        lines[c["id"]] = _downsample([sh(p) for p in c["centerline_enu_m"]])
    snapped = 0
    for cid, pts in lines.items():
        for end in (0, -1):
            p = pts[end]
            best = None
            for oid, opts in lines.items():
                if oid == cid:
                    continue
                for i, (a, b) in enumerate(zip(opts, opts[1:])):
                    q, _ = _project(p, a, b)
                    dd = _dist(p, q)
                    if dd <= SNAP_M and (best is None or dd < best[0]):
                        best = (dd, oid, i, q)
            if best and best[0] > 1e-6:
                _, oid, i, q = best
                q = (round(q[0], 3), round(q[1], 3))
                lines[oid].insert(i + 1, q)
                pts[end] = q
                snapped += 1
            elif best:
                pts[end] = best[3]
    log.append(f"corridor centrelines down-sampled to ~{SPACING_M} m; {snapped} endpoint(s) snapped onto another corridor (<= {SNAP_M} m)")
    for c in d["corridors"]:
        stair = c["class"] == "stair_ramp"
        entities.append({"id": f"road-{c['id'].lower()}", "class": "road",
                         "geometry": {"type": "polyline", "coords": [list(p) for p in lines[c["id"]]]},
                         "attributes": {"road.width_m": float(c.get("carriageway_width_m") or 3.0),
                                        "road.surface": "asphalt" if "2lane" in c["class"] else "concrete",  # catalog domain has no "stairs"; step height + upstream class carry it
                                        **({"road.step_height_m": 0.18} if stair else {}),
                                        "attr.upstream_id": c["id"], "attr.upstream_class": c["class"],
                                        "attr.reserved_width_m": c.get("reserved_width_m")},
                         "label_source": "upstream"})

    # Classed regions (skip those represented by corridors). Upstream polygon semantics are
    # "union of CCW components minus CW holes": each component becomes its own entity; the
    # flattened polygon_xy_m is never used as one footprint. Our schema has single rings, so
    # holes are kept as declared attributes (not subtracted) and counted in the adapter log.
    holes_kept = 0
    for r in d["regions"]:
        cls, note = REGION_CLASS.get(r["class"], ("unknown_upstream", "未映射的上游类别"))
        if cls is None:
            continue
        comps = r.get("polygon_components_xy_m") or [r["polygon_xy_m"]]
        holes = [[list(sh(p)) for p in hole] for hole in (r.get("holes_xy_m") or [])]
        holes_kept += len(holes)
        for ci, comp in enumerate(comps, 1):
            ring = _ring([sh(p) for p in comp])
            if len(ring) < 4:
                continue
            entities.append({"id": f"region-{r['region_id']}" + (f"-c{ci}" if len(comps) > 1 else ""),
                             "class": cls if cls != "unknown_upstream" else r["class"],
                             "geometry": {"type": "polygon", "coords": [list(p) for p in ring]},
                             "attributes": {"attr.upstream_class": r["class"], "attr.upstream_role": r.get("role"),
                                            "attr.upstream_name": r.get("name"), "attr.note_zh": note,
                                            "attr.component": f"{ci}/{len(comps)}",
                                            **({"attr.holes_xy_m": holes} if holes else {})},
                             "label_source": "upstream"})
    log.append(f"regions split into CCW components; {holes_kept} hole ring(s) kept as attributes, not subtracted")

    # Detention pond -> water (upstream drainage polygon).
    pond = (d.get("drainage") or {}).get("pond_polygon_xy_m")
    if pond:
        entities.append({"id": "water-pond", "class": "water",
                         "geometry": {"type": "polygon", "coords": [list(p) for p in _ring([sh(p) for p in pond])]},
                         "attributes": {"attr.upstream_class": "pond_polygon", "attr.note_zh": "上游滞洪池"},
                         "label_source": "upstream"})

    # Upstream tree anchors are OPTIONAL placement anchors, not confirmed existing trees. As a declared,
    # deterministic downstream authoring rule, 2 of every 5 anchors (by stable id hash) are instantiated
    # as authored trees; the rest stay unused. Instantiated trees say which upstream anchor they came from.
    anchors = (d.get("optional_tree_anchors") or {}).get("tree_anchor_enu_m") or []
    used = 0
    for i, p in enumerate(anchors, 1):
        if int(hashlib.sha256(f"tree-anchor-{i}".encode()).hexdigest(), 16) % 5 >= 2:
            continue
        used += 1
        entities.append({"id": f"tree-{i}", "class": "tree", "geometry": {"type": "point", "coords": [list(sh(p))]},
                         "attributes": {"attr.upstream_anchor": f"optional_tree_anchor[{i - 1}]",
                                        "attr.note_zh": "作者按固定规则由上游可选树锚点实例化；非上游确认的现存树木"},
                         "label_source": "authored"})
    log.append(f"optional tree anchors: {len(anchors)} upstream; {used} instantiated as authored trees (fixed 2/5 hash rule), rest unused")

    # Buildings: upstream plots are placement CANDIDATES, not existing buildings. As a declared
    # downstream authoring rule, a building is placed on a plot only if "building" is an allowed
    # candidate category and the plot's max footprint is at least 4 m x 4 m; other plots remain
    # open candidates (staging/parking). Footprint = 90% of max_footprint, centred on the anchor.
    skipped = []
    for p in d["plots"]:
        mf = p.get("max_footprint_m") or [0, 0]
        w, h = (float(mf[0]), float(mf[1])) if isinstance(mf, (list, tuple)) else (float(mf), float(mf))
        if "building" not in (p.get("candidate_categories") or []) or min(w, h) < 4.0:
            skipped.append(p["id"])
            continue
        ax, ay = sh(p["anchor_enu_m"])
        w, h = w * 0.9, h * 0.9
        a = math.radians(float(p.get("heading_deg_ccw_from_east") or 0.0))
        corners = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
            dx, dy = sx * w / 2, sy * h / 2
            corners.append([round(ax + dx * math.cos(a) - dy * math.sin(a), 3), round(ay + dx * math.sin(a) + dy * math.cos(a), 3)])
        entities.append({"id": f"building-{p['id'].lower()}", "class": "building",
                         "geometry": {"type": "polygon", "coords": corners},
                         "attributes": {"attr.upstream_plot": p["id"], "attr.frontage_corridor": p.get("frontage_corridor_id"),
                                        "attr.note_zh": "作者放置于上游建筑候选地块；上游不提供建筑"},
                         "label_source": "authored"})

    log.append(f"authored buildings on {len(d['plots']) - len(skipped)} plot(s); left as open candidates: {', '.join(skipped) or 'none'}")
    # Coordination post / station: not in upstream; authored on the paved public area if present.
    paved = next((e for e in entities if e["attributes"].get("attr.upstream_class") == "paved_public"), None)
    if paved:
        cx = sum(p[0] for p in paved["geometry"]["coords"][:-1]) / (len(paved["geometry"]["coords"]) - 1)
        cy = sum(p[1] for p in paved["geometry"]["coords"][:-1]) / (len(paved["geometry"]["coords"]) - 1)
        entities.append({"id": "station-post", "class": "station",
                         "geometry": {"type": "point", "coords": [[round(cx, 3), round(cy, 3)]]},
                         "attributes": {"attr.note_zh": "作者设定的协调/补给点（上游未提供）"}, "label_source": "authored"})
        # Authored medical receiving point beside it: rescue handover needs a compatible receiver.
        entities.append({"id": "station-medical", "class": "station",
                         "geometry": {"type": "point", "coords": [[round(cx + 6.0, 3), round(cy, 3)]]},
                         "attributes": {"station.kind": "medical", "station.capacity": 12, "station.power_W": 500.0,
                                        "attr.note_zh": "作者设定的医疗/救援接收点（上游未提供）"},
                         "label_source": "authored"})
    sha = hashlib.sha256(raw).hexdigest()
    return {
        "schema": "rescue.upstream-semantics/v1",
        "source": {"kind": "upstream_annotation", "producer": f"{d.get('template_id')} ground-template sidecar (read-only)",
                   "terrain_ref": f"{root.name}/metadata/regions.json sha256:{sha[:16]}",
                   "note": ("上游地形语义（道路/地块/区域/树/池）只读接入；建筑与协调点为作者标注；不加载原始网格；"
                            "未做自动分割。" + " | ".join(log))},
        "frame": {"crs": "local_ENU_m", "bounds_m": [0, 0, round(W, 3), round(H, 3)]},
        "entities": entities,
    }


def adapter_record(deliverable: str | Path, doc: dict) -> dict:
    """Explicit, separate record of the transform and adapter assumptions (schema-external)."""
    d = json.loads((Path(deliverable) / "metadata" / "regions.json").read_text())
    xs = [p[0] for r in d["regions"] for p in r["polygon_xy_m"]]
    ys = [p[1] for r in d["regions"] for p in r["polygon_xy_m"]]
    return {"source_dir": str(deliverable), "source_schema": d.get("schema"), "template_id": d.get("template_id"),
            "axes_in": d.get("axes"), "transform": {"type": "translate_xy", "dx_m": -min(xs), "dy_m": -min(ys),
                                                    "z": "not applied in this 2D-layout phase; elevation availability/provenance recorded under \"elevation\" for later use",
                                                    "inverse": "x_enu = x - dx_m, y_enu = y - dy_m"},
            "note": doc["source"]["note"], "upstream_limitations": d.get("limitations", []),
            "not_used": ["terrain_grid.json (elevation)", "raw GLB mesh", "camera_fit.json", "flood_demo.json"],
            "integration_scope_zh": "仅接入山坡的二维语义布局（区域/道路/地块/树锚点/池塘）；未接入高程、未加载网格，不是完整地形集成。",
            "elevation": _elevation_provenance(Path(deliverable))}


def _elevation_provenance(deliverable: Path) -> dict:
    """Record (not use) upstream elevation so a later phase can consume it without re-discovery."""
    grid_path = deliverable / "metadata" / "terrain_grid.json"
    if not grid_path.exists():
        return {"available": False}
    raw = grid_path.read_bytes()
    g = json.loads(raw)
    z = [v for v in g.get("z_m", []) if isinstance(v, (int, float))]
    return {"available": True, "used": False, "file": "metadata/terrain_grid.json",
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "schema": g.get("schema"),
            "grid": {"nx": g.get("nx"), "ny": g.get("ny"), "spacing_m": g.get("spacing_m"),
                     "origin_xy_m": g.get("origin_xy_m"), "bounds_xy_m": g.get("bounds_xy_m")},
            "z_range_m": [min(z), max(z)] if z else None, "interpolation": g.get("interpolation"),
            "same_xy_frame_as_regions": True,
            "note_zh": "高程可用但本阶段未使用；坐标需同样平移 (dx, dy) 才能与本语义布局对齐。"}


if __name__ == "__main__":
    doc = convert(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    out.with_name(out.name.replace(".semantics.json", ".adapter.json")).write_text(
        json.dumps(adapter_record(sys.argv[1], doc), ensure_ascii=False, indent=1))
    print(len(doc["entities"]), "entities;", doc["frame"]["bounds_m"])
