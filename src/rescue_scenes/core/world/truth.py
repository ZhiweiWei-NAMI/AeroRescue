"""Ground truth, authored generative rules and evaluator-only world objectives.

No task records, beliefs or allocations are created here. Numeric model parameters
are authored and uncalibrated; semantic labels are not perception outputs.
"""
from collections import ChainMap, OrderedDict
from copy import deepcopy
import heapq
import math

from ..catalog import load
from .profiles import PROFILES, OVERLAYS
from .semantics import SemanticLayer, ingest, rng_for

_EPS = 1e-7
BUILDING_NAVIGATION_FIELDS = ("attr.navigation_footprint", "attr.entrance_xy",
                              "attr.entrance_heading_deg", "attr.approach_start_xy",
                              "attr.approach_waypoints_xy", "attr.interior_access_via_entrance",
                              "attr.vehicle_approach")


def _exact_key(value):
    """Cache inputs without conflating types, signed zero or dictionary order."""
    if isinstance(value, dict):
        return (type(value), tuple((_exact_key(k), _exact_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_exact_key(v) for v in value))
    if isinstance(value, float):
        return (type(value), value.hex())
    return (type(value), value)


class _PreparedPolygon(tuple):
    """Query-local immutable coordinates and the original ordered edge pairs."""

    def __new__(cls, polygon):
        obj = super().__new__(cls, map(tuple, polygon))
        obj.edges = tuple(zip(obj, obj[1:] + obj[:1]))
        if obj:
            x0 = x1 = obj[0][0]
            y0 = y1 = obj[0][1]
            magnitude = abs(x0)
            # Keep first-on-tie extrema and the original x/y comparison order,
            # including signed zero and nonfinite coordinates.
            for point in obj:
                x, y = point[0], point[1]
                if x < x0:
                    x0 = x
                if y < y0:
                    y0 = y
                if x > x1:
                    x1 = x
                if y > y1:
                    y1 = y
                ax, ay = abs(x), abs(y)
                if ax > magnitude:
                    magnitude = ax
                if ay > magnitude:
                    magnitude = ay
            obj.bounds = (x0, y0, x1, y1)
        else:
            obj.bounds, magnitude = None, 0
        # This only broadens the rejection box; exact predicates keep _EPS.
        obj.padding = _EPS * (1 + magnitude) * 8
        return obj


def _prepare_polygon(polygon):
    return polygon if isinstance(polygon, _PreparedPolygon) else _PreparedPolygon(polygon)


def _outside_polygon_box(a, b, polygon):
    if polygon.bounds is None:
        return True
    x0, y0, x1, y1 = polygon.bounds
    # _intersection admits t/u just outside [0, 1]. Account for segment
    # extension as well as the polygon's edge extension and roundoff.
    pad = polygon.padding + _EPS * (abs(b[0]-a[0]) + abs(b[1]-a[1])) * 8
    return (max(a[0], b[0]) < x0-pad or min(a[0], b[0]) > x1+pad or
            max(a[1], b[1]) < y0-pad or min(a[1], b[1]) > y1+pad)


def _footprint_candidates(a, b, rows):
    """Reject prepared footprint rows in order, sharing segment box arithmetic."""
    high_x, low_x = max(a[0], b[0]), min(a[0], b[0])
    high_y, low_y = max(a[1], b[1]), min(a[1], b[1])
    extension = _EPS * (abs(b[0]-a[0]) + abs(b[1]-a[1])) * 8
    for row in rows:
        polygon = row[1]
        if polygon.bounds is None:
            continue
        x0, y0, x1, y1 = polygon.bounds
        pad = polygon.padding + extension
        # Identical arithmetic/comparisons to _outside_polygon_box: this is
        # only a broad phase, never a replacement for the exact predicates.
        if not (high_x < x0-pad or low_x > x1+pad or
                high_y < y0-pad or low_y > y1+pad):
            yield row


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _center(geometry):
    coords = geometry.get("coords", [])
    if not coords:
        return (0.0, 0.0)
    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return tuple(sum(p[i] for p in coords) / len(coords) for i in (0, 1))


def _projection(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx*dx + dy*dy
    u = max(0.0, min(1.0, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / denom)) if denom else 0.0
    return (a[0]+u*dx, a[1]+u*dy), u


def _projection_tree(rows, positions):
    """Index the actual floating-point projection ranges of compatible edges."""
    records = []
    for eid, edge, _, _, _ in rows:
        a, b = positions[edge['a']], positions[edge['b']]
        # _projection returns a + u*(b-a), not b, even when u == 1.
        # Monotonic floating-point multiplication/addition for 0 <= u <= 1
        # keeps q inside these bounds, including endpoint cancellation error.
        dx, dy = b[0]-a[0], b[1]-a[1]
        first = (a[0] + 0.0*dx, a[1] + 0.0*dy)
        end = (a[0] + 1.0*dx, a[1] + 1.0*dy)
        box = (min(first[0], end[0]), min(first[1], end[1]),
               max(first[0], end[0]), max(first[1], end[1]))
        records.append((eid, a, b, box))
    tree = []

    def split(items):
        box = (min(row[3][0] for row in items), min(row[3][1] for row in items),
               max(row[3][2] for row in items), max(row[3][3] for row in items))
        index = len(tree)
        tree.append(None)
        if len(items) <= 8:
            tree[index] = (box, (), items)
        else:
            axis = 0 if box[2]-box[0] >= box[3]-box[1] else 1
            items.sort(key=lambda row: row[3][axis]/2 + row[3][axis+2]/2)
            middle = len(items)//2
            children = (split(items[:middle]), split(items[middle:]))
            tree[index] = (box, children, ())
        return index

    if records:
        split(records)
    return tree


def _nearest_projection(point, tree, contacts):
    """Exact original (distance, edge id, q, u) minimum, with box pruning."""
    queue = [(0., 0)] if tree else []
    best = None
    while queue:
        lower, index = heapq.heappop(queue)
        # Strict comparison preserves all distance ties and their edge-id order.
        if best is not None and lower > best[0]:
            break
        _, children, records = tree[index]
        for eid, a, b, _ in records:
            q, u = _projection(point, a, b)
            candidate = (_distance(point, q), eid, q, u)
            if (best is None or candidate < best) and not contacts(point, q):
                best = candidate
        for child in children:
            x0, y0, x1, y1 = tree[child][0]
            # L-infinity distance to the enclosing box is <= the original
            # hypot(point-q), without squaring or a rounded sqrt bound.
            lower = max(0., x0-point[0], point[0]-x1, y0-point[1], point[1]-y1)
            if best is None or lower <= best[0]:
                heapq.heappush(queue, (lower, child))
    return best


def _intersection(a, b, c, d):
    """Non-parallel segment intersection, including endpoints."""
    ux, uy, vx, vy = b[0]-a[0], b[1]-a[1], d[0]-c[0], d[1]-c[1]
    det = ux*vy - uy*vx
    if abs(det) < _EPS:
        return None
    wx, wy = c[0]-a[0], c[1]-a[1]
    t, u = (wx*vy-wy*vx)/det, (wx*uy-wy*ux)/det
    if -_EPS <= t <= 1+_EPS and -_EPS <= u <= 1+_EPS:
        return (a[0]+t*ux, a[1]+t*uy)
    return None


def _inside(p, polygon):
    """Boundary-inclusive point in a semantic polygon."""
    if len(polygon) < 3:
        return False
    polygon = _prepare_polygon(polygon)
    if _outside_polygon_box(p, p, polygon):
        return False
    inside = False
    for a, b in polygon.edges:
        q, _ = _projection(p, a, b)
        if _distance(p, q) < _EPS:
            return True
        if (a[1] > p[1]) != (b[1] > p[1]):
            if p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:
                inside = not inside
    return inside


def _hits(geometry, polygon):
    coords = geometry.get("coords", [])
    if any(_inside(p,polygon) for p in coords):
        return True
    return any(_intersection(a,b,c,d) is not None
               for a,b in zip(coords,coords[1:])
               for c,d in zip(polygon,polygon[1:]+polygon[:1]))


def _segment_samples(a, b, polygons):
    cuts = [0.0, 1.0]
    for poly in polygons:
        for c, d in zip(poly, poly[1:] + poly[:1]):
            q = _intersection(a, b, c, d)
            if q is not None:
                _, u = _projection(q, a, b)
                cuts.append(u)
    cuts = sorted(set(cuts))
    return [a, b] + [(a[0]+(b[0]-a[0])*(s+t)/2, a[1]+(b[1]-a[1])*(s+t)/2)
                     for s, t in zip(cuts, cuts[1:])]


def footprint_entry(start, goal, polygon):
    """First contact with a building, including a start already on its footprint."""
    polygon = _prepare_polygon(polygon)
    if _outside_polygon_box(start, goal, polygon):
        return None
    if _inside(start, polygon):
        return start
    hits = [point for a, b in polygon.edges
            if (point := _intersection(start, goal, a, b)) is not None]
    return min(hits, key=lambda point: _distance(start, point)) if hits else None


def _footprint_intersects(start, goal, polygon):
    """The exact footprint_entry presence test, without finding nearest contact."""
    polygon = _prepare_polygon(polygon)
    if _outside_polygon_box(start, goal, polygon):
        return False
    if _inside(start, polygon):
        return True
    # Keep the same intersection arithmetic and edge order. A presence query
    # needs neither the remaining hits nor their distances from the start.
    return any(_intersection(start, goal, a, b) is not None for a, b in polygon.edges)


def isolated_ground_route(start, goal, regions):
    """A source-isolated ledge admits local work but no ordinary exit connector."""
    for eid, polygon in regions:
        local = _inside(start, polygon) and _inside(goal, polygon)
        if local or footprint_entry(start, goal, polygon) is not None:
            return {"ok": local, "waypoints": [list(start), list(goal)] if local else [list(start)],
                    "length_m": _distance(start, goal) if local else 0.,
                    "road_ids": [None] if local else [], "blocked_by": [] if local else [eid],
                    "reason_zh": "同一孤立作业区内接近" if local else "源地形无普通通路，须真实技术撤离"}
    return None


def wheeled_navigation_goal(goal_xy, buildings, roads, length_m=None):
    """Project an interior goal to the access-facing exterior, using map geometry.

    buildings contains (id, footprint, entrance road ids, entrance, heading,
    approach start, waypoints); roads maps ids to lines. The generator publishes
    a complete stand pose when waypoints are present, including body clearance.
    Both truth and local-map callers supply their own geometry, never a truth
    lookup from an agent. Older maps without entrance ids use the nearest road.
    """
    goal = list(goal_xy[:2])
    buildings = [(row[0], _prepare_polygon(row[1]), *row[2:]) for row in buildings]
    for _, polygon, entrances, entrance, heading, _, waypoints in buildings:
        if not _inside(goal, polygon):
            continue
        if entrance is not None:
            if waypoints:
                goal = list(entrance)
                continue
            if length_m is None:
                return None
            angle = math.radians(heading)
            stand_off = 1.5 + length_m/2
            goal = [entrance[0]-math.cos(angle)*stand_off,
                    entrance[1]-math.sin(angle)*stand_off]
            continue
        center = _center({"coords": polygon})
        lines = [roads[rid] for rid in entrances if rid in roads] or list(roads.values())
        approaches = [_projection(center, a, b)[0] for line in lines for a, b in zip(line, line[1:])]
        if not approaches:
            return None
        approach = min(approaches, key=lambda point: _distance(center, point))
        boundary = footprint_entry(approach, center, polygon)
        if boundary is None or _inside(approach, polygon):
            return None
        span = _distance(boundary, approach)
        # The existing motion model also stops 3 m short of physical obstacles.
        stand_off = min(3., span)
        if span > _EPS:
            goal = [boundary[k] + (approach[k]-boundary[k])*stand_off/span for k in (0, 1)]
    return None if any(_inside(goal, row[1]) for row in buildings) else goal


def building_approach_attachment(point, buildings):
    """Return the published route from a connector point back to its access."""
    for _, _, entrances, entrance, _, start, waypoints in buildings:
        if entrance is not None and start is not None:
            route = waypoints or [start, entrance]
            for index, (a, b) in enumerate(zip(route, route[1:])):
                projected, _ = _projection(point, a, b)
                if _distance(point, projected) < 1e-5:
                    tail = [list(point)]
                    for waypoint in reversed(route[:index+1]):
                        if _distance(tail[-1], waypoint) > _EPS:
                            tail.append(list(waypoint))
                    return entrances, start, tail
    return None


def building_approach_leg(a, b, buildings):
    """Whether both endpoints lie on one published, straight approach leg."""
    for _, _, _, entrance, _, start, waypoints in buildings:
        if entrance is not None and start is not None:
            route = waypoints or [start, entrance]
            if any(_distance(a, _projection(a, c, d)[0]) < 1e-5
                   and _distance(b, _projection(b, c, d)[0]) < 1e-5
                   for c, d in zip(route, route[1:])):
                return True
    return False


def interior_approach_buildings(buildings, points, enabled_ids):
    """Extend an explicitly mapped abstract entrance to this query's interior.

    Only that building's own entry legs admit its footprint. Other buildings,
    parked assets, road constraints and hazard exclusions remain obstacles.
    """
    out = []
    for row in buildings:
        eid, polygon, _, entrance, _, start, waypoints = row
        if eid not in enabled_ids or entrance is None or start is None:
            continue
        for point in points:
            if _inside(point, polygon):
                out.append((*row[:-1], [*(waypoints or [start, entrance]), list(point)]))
    return out


def _embodiment(spec):
    # Cross-lane callers supply §6.1 embodiment dictionaries, not private classes.
    if isinstance(spec, dict):
        return spec
    raise TypeError("embodiment must be the SPEC §6.1 dictionary / 需提供实体规格字典")


class WorldTruth:
    """Mutable physical truth. All field writes go through set() and history."""

    def __init__(self, profile=None, seed=1, semantics_source="authored", missing_inputs=None, bounds=(0, 0, 2000, 2000)):
        self.entities = {}
        self.objectives = []
        self.graph = {"nodes": {}, "edges": {}}
        self.roads = self.graph
        self.profile = deepcopy(profile or {})
        self.seed = seed
        self.semantics_source = semantics_source
        self.missing_inputs = deepcopy(missing_inputs or [])
        self.bounds = tuple(bounds)
        self.history = []
        self.warnings = []
        self.overlays = []
        self.overlay_effects = {k: {} for k in ("world", "network", "compute", "energy", "agents")}
        self.secondary_events = []
        self.t = 0.0
        self._deriving = False
        self._catalog = load()
        self._embodiments = {}
        self._field_revisions = {}
        self._navigation_revision = 0
        self._navigation_cache = {}
        self._contact_cache = {}
        self._navigation_inputs_token = None
        self._navigation_inputs_cache = {}
        self._bounds_key_source = None
        self._bounds_key = None

    def _invalidate_navigation(self):
        self._navigation_revision += 1
        self._navigation_cache.clear()
        self._contact_cache.clear()
        self._navigation_inputs_token = None
        self._navigation_inputs_cache.clear()

    def _navigation_token(self):
        bounds = self.bounds
        if bounds is self._bounds_key_source and self._bounds_key is not None:
            bounds_key = self._bounds_key
        elif type(bounds) is tuple and all(type(v) in (int, float, bool, str, type(None)) for v in bounds):
            self._bounds_key_source = bounds
            bounds_key = self._bounds_key = _exact_key(bounds)
        else:
            bounds_key = _exact_key(bounds)
        return (self._navigation_revision, id(self.graph), len(self.entities), bounds_key)

    def _navigation_inputs(self):
        """Private read-only preparations for the current physical navigation state."""
        token = self._navigation_token()
        if token != self._navigation_inputs_token:
            self._navigation_inputs_cache.clear()
            self._navigation_inputs_token = token
        return self._navigation_inputs_cache

    def _navigation_exclusions(self):
        cache = self._navigation_inputs()
        if 'exclusions' not in cache:
            cache['exclusions'] = self._exclusions()
        return cache['exclusions']

    def _navigation_water_polygons(self):
        cache = self._navigation_inputs()
        if 'water_polygons' not in cache:
            cache['water_polygons'] = [e['geometry']['coords'] for e in self.entities_of('water')
                                       if e['geometry']['type'] == 'polygon']
        return cache['water_polygons']

    def _navigation_isolated_regions(self):
        cache = self._navigation_inputs()
        if 'isolated_regions' not in cache:
            cache['isolated_regions'] = [(e['id'], _prepare_polygon(e['geometry']['coords']))
                                         for e in self.entities.values()
                                         if e['fields'].get('attr.ground_access_isolated') is True]
        return cache['isolated_regions']

    def _prepared_navigation_buildings(self, mobility):
        cache = self._navigation_inputs()
        key = ('buildings', mobility)
        if key not in cache:
            cache[key] = self.navigation_buildings(mobility, prepared=True)
        return cache[key]

    def get(self, eid, key, default=None):
        value = self.entities.get(eid, {}).get("fields", {}).get(key, default)
        if type(value) in (str, int, float, bool, type(None)):
            return value
        return deepcopy(value)

    def set(self, eid, key, value, cause):
        """Create/update an entity field, preserve provenance and return old/new.

        Creating an entity uses entity.kind followed by entity.footprint or
        entity.pose. Unknown model-only keys must carry the explicit attr. prefix.
        Cause may be a rule/action id, or {rule|id, t, refs, scripted}.
        D01 and D03 derivatives are recomputed for affected physical writes.
        """
        if not isinstance(eid, str) or not eid:
            raise ValueError("entity id must be nonempty / 实体标识不能为空")
        if key not in self._catalog.fields_by_key and not key.startswith("attr."):
            raise ValueError("非目录字段须标记 attr. / Non-catalog field requires attr.: " + str(key))
        if key in ("item.quantity", "item.mass_kg") and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise ValueError("物资数量/质量须非负 / Item quantity/mass must be nonnegative")
        entity = self.entities.setdefault(eid, {"id": eid, "cls": "unknown",
                                               "geometry": {"type": "point", "coords": [[0, 0]]},
                                               "fields": {}, "provenance": {}})
        old_holder = entity["fields"].get("item.holder_id")
        before = deepcopy(entity["fields"].get(key))
        value = deepcopy(value)
        meta = cause if isinstance(cause, dict) else {"rule": cause}
        rule = str(meta.get("rule", meta.get("id", "authored")))
        entity["fields"][key] = value
        entity["provenance"][key] = rule
        self._field_revisions[eid] = self._field_revisions.get(eid, 0) + 1
        # Pose/battery/medical progress does not change road geometry. Moving
        # physical blockers and every authored navigation field still invalidate.
        if (key.startswith(("road.", "obstacle.", "hazard.", "building.", "attr."))
                or key in ("entity.kind", "entity.footprint", "station.access_region")
                or key in ("entity.pose", "human.pose") and
                (entity["cls"] not in ("human", "platform")
                 or entity["fields"].get("attr.blocks_ground_navigation") is True
                 or entity["fields"].get("attr.ground_access_isolated") is True)):
            self._invalidate_navigation()
        if key == "entity.kind":
            entity["cls"] = value
        elif key in ("entity.pose", "human.pose") and isinstance(value, (list, tuple)) and len(value) >= 2:
            if entity["geometry"]["type"] == "point":
                entity["geometry"] = {"type": "point", "coords": [list(value[:2])]}
        elif key in ("entity.footprint", "road.centerline"):
            if isinstance(value, dict) and "coords" in value:
                entity["geometry"] = deepcopy(value)
            elif isinstance(value, (list, tuple)):
                entity["geometry"] = {"type": "polyline" if key == "road.centerline" else "polygon", "coords": deepcopy(value)}
        refs = list(meta.get("refs", []))
        if rule in self._catalog.rules or rule in self._catalog.actions or rule in self._catalog.overlays:
            refs.append(rule)
        field_record = self._catalog.fields_by_key.get(key)
        if field_record:
            refs.append(field_record["id"])
        self.history.append({"eid": eid, "entity_id": eid, "key": key, "field": key,
                             "before": before, "after": deepcopy(value), "cause": deepcopy(cause),
                             "provenance": rule, "t": float(meta.get("t", self.t)),
                             "refs": list(dict.fromkeys(refs)),
                             "grounding": "catalog" if field_record else "proposed",
                             "scripted": bool(meta.get("scripted", rule in ("authored", "G01", "G02", "G03", "G04", "G05", "G06", "G07")))})
        if not self._deriving:
            self._deriving = True
            try:
                if entity["cls"] == "human" and key in ("entity.pose", "human.pose"):
                    alias = "human.pose" if key == "entity.pose" else "entity.pose"
                    if entity["fields"].get(alias) != value:
                        self.set(eid, alias, value, cause)
                if entity["cls"] == "obstacle" and key in ("obstacle.state", "obstacle.work_remaining", "obstacle.height_m", "obstacle.blocks_ids", "attr.blocked_width_m"):
                    for road_id in self.get(eid, "obstacle.blocks_ids", []):
                        if road_id in self.entities and self.entities[road_id]["cls"] == "road":
                            self._derive_road(road_id)
                if entity["cls"] == "road" and key in ("road.width_m", "road.obstacle_ids"):
                    self._derive_road(eid)
                elif entity["cls"] == "road" and key in ("road.entry_state", "road.water_depth_m", "road.clear_width_m", "road.step_height_m"):
                    if before != value:
                        self.set(eid,"road.topology_revision",int(self.get(eid,"road.topology_revision",0))+1,"D01")
                    self._derive_passability(eid)
                if entity["cls"] == "hazard" and key == "hazard.exclusion_zone":
                    for road in self.entities_of("road"):
                        self._derive_passability(road["id"])
                if entity["cls"] == "item" and key in ("item.holder_id", "item.quantity", "item.state", "item.kind", "item.mass_kg"):
                    for holder in sorted({h for h in (old_holder, self.get(eid, "item.holder_id")) if h}):
                        self._derive_inventory(holder)
            finally:
                self._deriving = False
        return before, deepcopy(value)

    def entities_of(self, cls):
        return [entity for entity in self.entities.values() if entity["cls"] == cls]

    def nearest(self, point, cls=None):
        candidates = self.entities_of(cls) if cls is not None else list(self.entities.values())
        return min(candidates, key=lambda e: (_distance(point, _center(e["geometry"])), e["id"]), default=None)

    def _derive_inventory(self, holder):
        if holder not in self.entities:
            return
        items = [item for item in self.entities_of("item")
                 if self.get(item["id"], "item.holder_id") == holder
                 and self.get(item["id"], "item.state") not in ("lost", "consumed")]
        inventory, mass = {}, 0.0
        for item in items:
            eid = item["id"]
            kind, quantity = self.get(eid, "item.kind"), self.get(eid, "item.quantity", 0)
            if kind is not None:
                inventory[kind] = inventory.get(kind, 0) + quantity
            mass += self.get(eid, "item.mass_kg", 0) * quantity
        if self.entities[holder]["cls"] == "station":
            self.set(holder, "station.inventory", inventory, "D03")
        elif self.entities[holder]["cls"] == "platform":
            # Cargo changes must preserve the mass of people still aboard.
            human_mass = sum(self.get(human["id"], "attr.mass_kg", 75.)
                             for human in self.entities_of("human")
                             if self.get(human["id"], "human.carrier_id") == holder)
            self.set(holder, "platform.payload_item_ids", [i["id"] for i in items], "D03")
            self.set(holder, "platform.payload_used_kg", round(mass + human_mass, 6), "D03")

    def _derive_road(self, road_id):
        width = float(self.get(road_id, "road.width_m", 0))
        clear, step = width, float(self.get(road_id, "attr.base_step_height_m", 0))
        for oid in self.get(road_id, "road.obstacle_ids", []):
            if self.get(oid, "obstacle.state") not in ("present", "partly_cleared"):
                continue
            initial = max(float(self.get(oid, "attr.initial_work", 1)), _EPS)
            fraction = max(0.0, min(1.0, float(self.get(oid, "obstacle.work_remaining", initial))/initial))
            occupied = float(self.get(oid, "attr.blocked_width_m", width)) * fraction
            clear = min(clear, max(0.0, width - occupied))
            step = max(step, float(self.get(oid, "obstacle.height_m", 0))*fraction)
        self.set(road_id, "road.clear_width_m", round(clear, 6), "D01")
        self.set(road_id, "road.step_height_m", round(step, 6), "D01")
        if self.get(road_id, "road.entry_state") not in ("closed", "unsafe"):
            self.set(road_id, "road.entry_state", "open" if clear >= min(2.2, width) and step < _EPS else "blocked", "D01")
        self.set(road_id, "road.topology_revision", int(self.get(road_id, "road.topology_revision", 0))+1, "D01")
        self._derive_passability(road_id)

    def _derive_passability(self, road_id):
        if not self._embodiments:
            return
        values = {}
        for kind, spec in self._embodiments.items():
            ok, reason = self.passable(road_id, spec)
            values[kind] = {"state": "pass" if ok else "block", "reason": reason}
        self.set(road_id, "road.passability", values, "D01")

    def _exclusions(self):
        polygons = []
        for hazard in self.entities_of("hazard"):
            value = self.get(hazard["id"], "hazard.exclusion_zone", False)
            if not value:
                continue
            if isinstance(value, dict):
                coords = value.get("coords", value.get("geometry", {}).get("coords", []))
            elif isinstance(value, list):
                coords = value
            else:
                coords = hazard["geometry"].get("coords", [])
            if len(coords) >= 3:
                polygons.append(coords)
        return polygons

    def passable(self, edge_or_road_id, embodiment):
        spec = _embodiment(embodiment)
        mobility = spec.get('mobility', 'none')
        # Region queries also accept moving entity ids, whose ordinary poses do
        # not invalidate road navigation. Reuse their polygons, not their result.
        if mobility in ('air', 'water'):
            return self._passable(edge_or_road_id, spec)
        edge = self.graph['edges'].get(edge_or_road_id)
        # Ground checks depend on the road, not which split edge was queried.
        target = edge['road_id'] if edge else edge_or_road_id
        key = ('passable', target, _exact_key((mobility, spec.get('professional_access', False),
               spec.get('width_m', 0), spec.get('step_height_m', 0), spec.get('water_tolerance_m', 0))))
        cache = self._navigation_inputs()
        if key not in cache:
            cache[key] = self._passable(edge_or_road_id, spec)
        return cache[key]

    def _passable(self, edge_or_road_id, spec):
        mobility = spec.get("mobility", "none")
        edge = self.graph["edges"].get(edge_or_road_id)
        rid = edge["road_id"] if edge else edge_or_road_id
        if mobility == "air":
            if edge:
                a, b = (self.graph["nodes"][edge[k]] for k in ("a", "b"))
            elif rid in self.entities:
                coords = self.entities[rid]["geometry"]["coords"]
                a, b = coords[0], coords[-1]
            else:
                return False, "unknown_road"
            if any(any(_inside(p, poly) for p in _segment_samples(a, b, [poly])) for poly in self._navigation_exclusions()):
                return False, "hazard.exclusion_zone"
            return True, "air_ignores_roads"
        if mobility == "water":
            if edge:
                a, b = (self.graph["nodes"][edge[k]] for k in ("a", "b"))
            elif rid in self.entities:
                a, b = self.entities[rid]["geometry"]["coords"][0], self.entities[rid]["geometry"]["coords"][-1]
            else:
                return False, "unknown_road"
            polygons = self._navigation_water_polygons()
            ok = all(any(_inside(p, poly) for poly in polygons) for p in _segment_samples(a, b, polygons))
            return ok, "within_water" if ok else "outside_water"
        if mobility == "none":
            return False, "immobile_embodiment"
        if rid not in self.entities or self.entities[rid]["cls"] != "road":
            return False, "unknown_road"
        if any(_hits(self.entities[rid]['geometry'], polygon) for polygon in self._navigation_exclusions()):
            return False, 'hazard.exclusion_zone'
        allowed = self.get(rid, "attr.allowed_mobilities")
        if allowed is not None and mobility not in allowed:
            return False, "attr.allowed_mobilities"
        if self.get(rid, "attr.professional_only") and not spec.get("professional_access", False):
            return False, "attr.professional_only"
        if self.get(rid, "road.entry_state") in ("unsafe", "closed"):
            return False, "road.entry_state=" + self.get(rid, "road.entry_state")
        if self.get(rid, "road.clear_width_m", self.get(rid, "road.width_m", 0)) + _EPS < spec.get("width_m", 0):
            return False, "road.clear_width_m"
        step = self.get(rid, "road.step_height_m", 0)
        if step > spec.get("step_height_m", 0) + _EPS:
            return False, "road.step_height_m"
        rubble = any(self.get(oid,"obstacle.state") in ("present","partly_cleared")
                     and (self.get(oid,"obstacle.kind") == "rubble" or self.get(oid,"obstacle.state") == "partly_cleared")
                     for oid in self.get(rid,"road.obstacle_ids",[]))
        if mobility == "wheeled" and rubble and step > _EPS:
            return False, "rubble_requires_legged_or_foot"
        if self.get(rid, "road.water_depth_m", 0) > spec.get("water_tolerance_m", 0) + _EPS:
            return False, "road.water_depth_m"
        return True, "D01_clearance_satisfied"

    def _build_graph(self):
        """Split road polylines at all annotated/intersecting junctions."""
        nodes, edges, segments = {}, {}, []
        for road in self.entities_of("road"):
            coords = road["geometry"]["coords"]
            for a, b in zip(coords, coords[1:]):
                if _distance(a, b) > _EPS:
                    segments.append((road["id"], tuple(a), tuple(b)))
        cuts = [[a, b] for _, a, b in segments]
        for i, (_, a, b) in enumerate(segments):
            for j in range(i):
                _, c, d = segments[j]
                cross = _intersection(a, b, c, d)
                if cross is not None:
                    cuts[i].append(cross)
                    cuts[j].append(cross)
                else:
                    # Collinear joins and a spur landing on an existing segment.
                    for p in (c, d):
                        q, _ = _projection(p, a, b)
                        if _distance(p, q) < _EPS:
                            cuts[i].append(p)
                    for p in (a, b):
                        q, _ = _projection(p, c, d)
                        if _distance(p, q) < _EPS:
                            cuts[j].append(p)
        coordinate_ids = {}

        def node(p):
            key = (round(p[0], 7), round(p[1], 7))
            if key not in coordinate_ids:
                nid = f"node-{len(nodes)+1}"
                coordinate_ids[key] = nid
                nodes[nid] = tuple(p)
            return coordinate_ids[key]

        counts = {}
        for (rid, a, b), points in zip(segments, cuts):
            points = sorted(set(points), key=lambda p: _projection(p, a, b)[1])
            for p, q in zip(points, points[1:]):
                if _distance(p, q) <= _EPS:
                    continue
                counts[rid] = counts.get(rid, 0) + 1
                eid = f"{rid}:edge-{counts[rid]}"
                edges[eid] = {"a": node(p), "b": node(q), "road_id": rid, "length_m": _distance(p, q)}
        self.graph = {"nodes": nodes, "edges": edges}
        self.roads = self.graph
        self._invalidate_navigation()
        for road in self.entities_of("road"):
            rid = road["id"]
            coords = road["geometry"]["coords"]
            if len(coords) > 1:
                self.set(rid, "road.from_node", coordinate_ids.get((round(coords[0][0], 7), round(coords[0][1], 7))), "D01")
                self.set(rid, "road.to_node", coordinate_ids.get((round(coords[-1][0], 7), round(coords[-1][1], 7))), "D01")
                self.set(rid, "road.travel_cost", sum(e["length_m"] for e in edges.values() if e["road_id"] == rid), "D01")

    def navigation_footprint(self, eid):
        """Physical displayed-state envelope; the larger placement reserve stays intact."""
        return self.get(eid, "attr.navigation_footprint", self.entities[eid]["geometry"])["coords"]

    def navigation_buildings(self, mobility="wheeled", *, prepared=False):
        """Detached navigation rows, optionally with query-local frozen polygons."""
        out = []
        for e in self.entities.values():
            eid = e['id']
            if not ((e['cls'] == 'building' and mobility == 'wheeled') or
                    self.get(eid, 'attr.blocks_ground_navigation') is True):
                continue
            vehicle = self.get(eid, 'attr.vehicle_approach') if mobility == 'wheeled' else None
            # Preparing already snapshots the coordinates; do not first deepcopy
            # the entire geometry only to discard that temporary mutable copy.
            polygon = (_prepare_polygon(e['fields'].get('attr.navigation_footprint', e['geometry'])['coords'])
                       if prepared else self.navigation_footprint(eid))
            out.append((eid, polygon,
                vehicle['entrance_ids'] if vehicle else self.get(eid, 'building.entrance_ids', []),
                vehicle['stand_xy'] if vehicle else self.get(eid, 'attr.entrance_xy'),
                vehicle['heading_deg'] if vehicle else self.get(eid, 'attr.entrance_heading_deg'),
                vehicle['start_xy'] if vehicle else self.get(eid, 'attr.approach_start_xy'),
                vehicle['waypoints_xy'] if vehicle else self.get(eid, 'attr.approach_waypoints_xy', [])))
        return out

    def navigation_goal(self, goal_xy, embodiment):
        """Keep wheeled goals outside buildings; foot/legged goals stay interior."""
        if embodiment.get('mobility') == 'water':
            for station in self.entities_of('station'):
                access = self.get(station['id'], 'station.access_region')
                if access and _distance(goal_xy, _center(station['geometry'])) < _EPS:
                    return list(_center(access))
        if embodiment.get("mobility") != "wheeled":
            return list(goal_xy[:2])
        cache = self._navigation_inputs()
        if 'road_lines' not in cache:
            cache['road_lines'] = {e['id']: e['geometry']['coords'] for e in self.entities_of('road')}
        return wheeled_navigation_goal(
            goal_xy,
            self._prepared_navigation_buildings('wheeled'),
            cache['road_lines'],
            embodiment.get("length_m"))

    def path(self, start_xy, goal_xy, embodiment):
        key = (_exact_key(start_xy[:2]), _exact_key(goal_xy[:2]), _exact_key(embodiment))
        prepared = self._navigation_inputs()
        if 'paths' not in prepared:
            prepared['paths'] = OrderedDict()
        cache = prepared['paths']
        if key not in cache:
            cache[key] = self._path(start_xy, goal_xy, embodiment)
            if len(cache) > 4096:
                cache.popitem(last=False)
        cache.move_to_end(key)
        # Callers may attach execution metadata or edit the returned waypoints.
        return deepcopy(cache[key])

    def _path(self, start_xy, goal_xy, embodiment):
        spec = _embodiment(embodiment)
        start, goal = tuple(start_xy[:2]), tuple(goal_xy[:2])
        if not all(math.isfinite(n) for n in (*start, *goal)):
            raise ValueError("nonfinite route coordinate / 路径坐标非有限数")
        projected = self.navigation_goal(goal, spec)
        if projected is None:
            return {"ok": False, "waypoints": [list(start)], "length_m": 0.0,
                    "blocked_by": [e["id"] for e in self.entities_of("building")
                                   if _inside(goal, self.navigation_footprint(e["id"]))]}
        goal = tuple(projected)
        if not all(self.bounds[0] <= p[0] <= self.bounds[2] and self.bounds[1] <= p[1] <= self.bounds[3] for p in (start, goal)):
            return {"ok": False, "waypoints": [list(start)], "length_m": 0.0, "blocked_by": ["outside_bounds"]}
        mobility = spec.get("mobility", "none")
        if mobility in ("air", "water"):
            return self._region_path(start, goal, mobility)
        isolated = isolated_ground_route(start, goal, self._navigation_isolated_regions())
        if isolated is not None:
            return isolated
        if _distance(start, goal) < _EPS:
            return {"ok": True, "waypoints": [list(start)], "length_m": 0.0, "blocked_by": []}
        if mobility == "none":
            return {"ok": False, "waypoints": [list(start)], "length_m": 0.0, "blocked_by": ["immobile_embodiment"]}
        if not self.graph["edges"]:
            return {"ok": False, "waypoints": [list(start)], "length_m": 0.0, "blocked_by": ["missing_road_network"]}
        positions = ChainMap({}, self.graph['nodes'])
        blocked = set()
        prepared = self._navigation_inputs()
        buildings = self._prepared_navigation_buildings(mobility)
        interior = interior_approach_buildings(buildings, (start, goal),
            {row[0] for row in buildings if mobility in ('foot', 'legged') and
             self.get(row[0], 'attr.interior_access_via_entrance') is True})
        interior_by_id = {}
        for row in interior:
            interior_by_id.setdefault(row[0], []).append(row)
        if 'prepared_exclusions' not in prepared:
            prepared['prepared_exclusions'] = [_prepare_polygon(poly) for poly in self._navigation_exclusions()]
        exclusions = prepared['prepared_exclusions']
        contact_token = self._navigation_token()

        def raw_contacts(a, b):
            key = (contact_token, mobility, _exact_key(a), _exact_key(b))
            cached = self._contact_cache.get(key)
            if cached is None:
                cached = ([row[0] for row in _footprint_candidates(a, b, buildings)
                           if footprint_entry(a, b, row[1]) is not None],
                          any(footprint_entry(a, b, poly) is not None for poly in exclusions))
                if len(self._contact_cache) >= 16384:
                    self._contact_cache.clear()
                self._contact_cache[key] = cached
            return cached

        def contacts(a, b, raw=None):
            cached = raw_contacts(a, b) if raw is None else raw
            # Only raw contacts are shared: interior entrance permissions always
            # belong to these exact start/goal endpoints.
            hits = [eid for eid in cached[0]
                    if not building_approach_leg(a, b, interior_by_id.get(eid, []))]
            if cached[1]:
                hits.append('hazard.exclusion_zone')
            return hits

        def connect(a, b, length):
            hits = contacts(positions[a], positions[b])
            if hits:
                blocked.update(hits)
                return
            # Only endpoint/interior-touched neighbor lists belong to this query.
            for node in (a, b):
                if node not in local_adjacency:
                    local_adjacency[node] = list(adjacency.get(node, ()))
            adjacency[a].append((b, length))
            adjacency[b].append((a, length))

        # Only these exact spec values affect the fixed ground edges. Goal
        # projection and endpoint/interior permissions are still evaluated above.
        base_key = ('ground_edges', _exact_key((mobility, spec.get('professional_access', False),
                    spec.get('width_m', 0), spec.get('step_height_m', 0), spec.get('water_tolerance_m', 0))))
        if base_key not in prepared:
            rows, admitted, base_adjacency, neighbors = [], {}, {}, {}
            clear_edges, base_blocked, interior_edges = set(), {}, {}
            for eid, edge in self.graph['edges'].items():
                index = len(rows)
                ok = admitted[eid] = self.passable(eid, spec)[0]
                a, b, length = edge['a'], edge['b'], edge['length_m']
                raw = raw_contacts(positions[a], positions[b]) if ok else None
                allowed = self.get(edge['road_id'], 'attr.allowed_mobilities')
                compatible = not (allowed is not None and mobility not in allowed
                                  or self.get(edge['road_id'], 'attr.professional_only')
                                  and not spec.get('professional_access', False))
                rows.append((eid, edge, ok, raw, compatible))
                hits = raw[0] + (['hazard.exclusion_zone'] if raw[1] else []) if ok else [edge['road_id']]
                for hit in hits:
                    base_blocked[hit] = base_blocked.get(hit, 0) + 1
                if ok:
                    neighbors.setdefault(a, []).append((index, b, length))
                    neighbors.setdefault(b, []).append((index, a, length))
                    for hit in raw[0]:
                        interior_edges.setdefault(hit, []).append(index)
                    if not hits:
                        clear_edges.add(index)
                        base_adjacency.setdefault(a, []).append((b, length))
                        base_adjacency.setdefault(b, []).append((a, length))
            prepared[base_key] = (rows, admitted, base_adjacency, neighbors, clear_edges,
                                  base_blocked, interior_edges, {}, OrderedDict())
        (edge_rows, admitted, base_adjacency, neighbors, clear_edges, base_blocked,
         interior_edges, projection_indices, attachment_cache) = prepared[base_key]
        local_adjacency = {}
        adjacency = ChainMap(local_adjacency, base_adjacency)
        counts = base_blocked.copy()
        changed = {index for eid in interior_by_id for index in interior_edges.get(eid, ())}
        opened, affected = set(), set()
        for index in changed:
            _, edge, _, raw, _ = edge_rows[index]
            hits = contacts(positions[edge['a']], positions[edge['b']], raw)
            for hit in raw[0]:
                if hit not in hits:
                    counts[hit] -= 1
            if not hits:
                opened.add(index)
                affected.update((edge['a'], edge['b']))
        for node in affected:
            # Retain the original edge insertion order, even for parallel edges.
            local_adjacency[node] = [(other, length) for index, other, length in neighbors[node]
                                     if index in clear_edges or index in opened]
        blocked.update(hit for hit, count in counts.items() if count)
        attachment_queries = []
        interior_key = _exact_key(interior_by_id)

        def collect_connector_blockers():
            # Successful _shortest results always expose blocked_by=[]. Only a
            # failure needs the union over ALL candidate connectors, including
            # remote rejected ones. Replay it exactly, never just nearest hits.
            for attachment, rows, cached in attachment_queries:
                if cached[1] is None:
                    hits = set()
                    for _, edge, _, _, _ in rows:
                        q, _ = _projection(attachment, self.graph['nodes'][edge['a']],
                                           self.graph['nodes'][edge['b']])
                        hits.update(contacts(attachment, q))
                    cached[1] = tuple(sorted(hits))
                blocked.update(cached[1])

        attached = []
        for name, point in (("@start", start), ("@goal", goal)):
            approach = building_approach_attachment(point, interior + buildings)
            attachment = approach[1] if approach else point
            index_key = _exact_key(approach[0]) if approach else None
            if index_key not in projection_indices:
                rows = [row for row in edge_rows if row[4] and
                        (not approach or row[1]['road_id'] in approach[0])]
                projection_indices[index_key] = (rows, _projection_tree(rows, self.graph['nodes']))
            rows, tree = projection_indices[index_key]
            attachment_key = (index_key, _exact_key(attachment), interior_key)
            cached = attachment_cache.get(attachment_key)
            if cached is None:
                cached = [_nearest_projection(attachment, tree, contacts), None]
                attachment_cache[attachment_key] = cached
                if len(attachment_cache) > 4096:
                    attachment_cache.popitem(last=False)
            attachment_cache.move_to_end(attachment_key)
            attachment_queries.append((attachment, rows, cached))
            if cached[0] is None:
                collect_connector_blockers()
                return {'ok':False,'waypoints':[list(start)],'length_m':0.,
                        'blocked_by':sorted(blocked) or ['no_compatible_source_road']}
            _, eid, q, u = cached[0]
            edge = self.graph["edges"][eid]
            positions[name] = point
            projection_id = name + "_road"
            positions[projection_id] = q
            previous = name
            if approach:
                for waypoint in approach[2][1:]:
                    nid = f"@approach:{waypoint[0]:.7f},{waypoint[1]:.7f}"
                    positions[nid] = tuple(waypoint)
                    connect(previous, nid, _distance(positions[previous], waypoint))
                    previous = nid
            connect(previous, projection_id, _distance(positions[previous], q))
            if admitted[eid]:
                connect(projection_id, edge["a"], u*edge["length_m"])
                connect(projection_id, edge["b"], (1-u)*edge["length_m"])
            else:
                # At an endpoint, a blocked outgoing edge must not imprison the node.
                for endpoint in ("a", "b"):
                    if _distance(q, positions[edge[endpoint]]) < _EPS:
                        connect(projection_id, edge[endpoint], 0.0)
            attached.append((eid, projection_id, u))
        if attached[0][0] == attached[1][0] and admitted[attached[0][0]]:
            connect(attached[0][1], attached[1][1], _distance(positions[attached[0][1]], positions[attached[1][1]]))
        if building_approach_leg(start, goal, interior + buildings):
            connect("@start", "@goal", _distance(start, goal))
        result = self._shortest(positions, adjacency, "@start", "@goal", ())
        if not result['ok']:
            collect_connector_blockers()
            result['blocked_by'] = sorted(blocked) or ['disconnected_road_network']
        return result

    def _region_path(self, start, goal, mobility):
        polygons = (self._navigation_exclusions() if mobility == "air" else
                    self._navigation_water_polygons())
        # Prepare lazily, in predicate order, and only for this route. The
        # original polygons remain in use for sampling and vertex generation.
        prepared = [None] * len(polygons)

        def contains(p, index):
            poly = polygons[index]
            if len(poly) < 3:
                return False
            if prepared[index] is None:
                prepared[index] = _prepare_polygon(poly)
            return _inside(p, prepared[index])

        def allowed(a, b):
            samples = _segment_samples(a, b, polygons)
            if mobility == "water":
                return all(any(contains(p, i) for i in range(len(polygons))) for p in samples)
            return not any(contains(p, i) for p in samples for i in range(len(polygons)))

        if not allowed(start, start) or not allowed(goal, goal):
            return {"ok": False, "waypoints": [list(start)], "length_m": 0.0,
                    "blocked_by": ["outside_water" if mobility == "water" else "hazard.exclusion_zone"]}
        positions = {"@start": start, "@goal": goal}
        for poly in polygons:
            center = _center({"coords": poly})
            for p in poly:
                if mobility == "air":
                    dx, dy = p[0]-center[0], p[1]-center[1]
                    norm = math.hypot(dx, dy) or 1.0
                    p = (p[0]+dx/norm*.05, p[1]+dy/norm*.05)
                if self.bounds[0] <= p[0] <= self.bounds[2] and self.bounds[1] <= p[1] <= self.bounds[3]:
                    positions[f"v-{len(positions)}"] = tuple(p)
        adjacency = {nid: [] for nid in positions}
        ids = list(positions)
        for i, a in enumerate(ids):
            for b in ids[:i]:
                if allowed(positions[a], positions[b]):
                    dist = _distance(positions[a], positions[b])
                    adjacency[a].append((b, dist))
                    adjacency[b].append((a, dist))
        return self._shortest(positions, adjacency, "@start", "@goal",
                              ["disconnected_water" if mobility == "water" else "hazard.exclusion_zone"])

    @staticmethod
    def _shortest(positions, adjacency, start, goal, blockers):
        distances, previous, queue = {start: 0.0}, {}, [(0.0, start)]
        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node] + _EPS:
                continue
            if node == goal:
                break
            for other, length in sorted(adjacency.get(node, [])):
                candidate = dist + length
                if candidate + _EPS < distances.get(other, math.inf):
                    distances[other] = candidate
                    previous[other] = node
                    heapq.heappush(queue, (candidate, other))
        ok = goal in distances
        endpoint = goal if ok else min(distances, key=lambda n: (_distance(positions[n], positions[goal]), distances[n], n))
        route, node = [endpoint], endpoint
        while node != start:
            node = previous[node]
            route.append(node)
        waypoints = []
        for nid in reversed(route):
            p = list(positions[nid])
            if not waypoints or _distance(p, waypoints[-1]) > _EPS:
                waypoints.append(p)
        return {"ok": ok, "waypoints": waypoints, "length_m": round(distances[endpoint], 6),
                "blocked_by": [] if ok else sorted(set(blockers))}

    def to_dict(self, view="truth"):
        if view != "truth":
            raise ValueError("WorldTruth only exports truth; belief views belong to agents / 认知视图须由智能体构建")
        return deepcopy({"entities": self.entities, "objectives": self.objectives,
                         "graph": self.graph, "roads": self.graph, "profile": self.profile,
                         "seed": self.seed, "bounds": list(self.bounds),
                         "semantics_source": self.semantics_source, "missing_inputs": self.missing_inputs,
                         "warnings": self.warnings, "history": self.history, "overlays": self.overlays,
                         "overlay_effects": self.overlay_effects, "secondary_events": self.secondary_events})


def _add(truth, eid, cls, geometry, fields, cause):
    if eid in truth.entities:
        raise ValueError("生成实体标识冲突 / Generated entity id collision: " + eid)
    truth.set(eid, "entity.id", eid, cause)
    truth.set(eid, "entity.kind", cls, cause)
    if geometry["type"] == "point":
        truth.set(eid, "entity.pose", list(geometry["coords"][0]), cause)
    else:
        truth.set(eid, "road.centerline" if cls == "road" else "entity.footprint", geometry, cause)
    for key, value in fields.items():
        truth.set(eid, key, value, cause)
    return truth.entities[eid]


def _write(truth, eid, values, cause):
    for key, value in values.items():
        truth.set(eid, key, value, cause)


def _point(point):
    return {"type": "point", "coords": [list(point)]}


def _area(geometry):
    coords = geometry["coords"]
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(coords, coords[1:]+coords[:1])))/2


def _merge_effects(destination, incoming):
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _merge_effects(destination[key], value)
        elif isinstance(value, list) and isinstance(destination.get(key), list):
            destination[key].extend(deepcopy(value))
        else:
            destination[key] = deepcopy(value)


def generate(semantic_doc, profile_id, seed, overlays=None, options=None):
    """Apply G02–G07, D01 and D03 without exposing truth to any agent."""
    if profile_id not in PROFILES:
        raise ValueError("未知情境 / Unknown profile: " + str(profile_id))
    layer = semantic_doc if isinstance(semantic_doc, SemanticLayer) else ingest(semantic_doc)
    truth = WorldTruth(PROFILES[profile_id], seed, layer.source_kind, layer.missing, layer.bounds)
    # The registry is the only cross-lane embodiment interface (SPEC §6.1).
    # Import lazily: package import need not initialize the agent runtime.
    from ..agents.embodiment import EMBODIMENTS
    truth._embodiments = deepcopy(EMBODIMENTS)
    truth.warnings = list(layer.warnings)
    options = deepcopy(options or {})
    if 'public_brief_zh' in options:
        truth.profile['public_brief_zh'] = options['public_brief_zh']
    truth.overlays = list(dict.fromkeys(overlays or []))
    for oid in truth.overlays:
        if oid not in OVERLAYS:
            raise ValueError("未知叠加条件 / Unknown overlay: " + str(oid))
        effects = deepcopy(OVERLAYS[oid]["effects"])
        for event in effects.get("world",{}).get("secondary_events",[]):
            event["refs"] = list(dict.fromkeys(event.get("refs",[])+[oid]))
        _merge_effects(truth.overlay_effects, effects)
    profile, model = truth.profile, truth.profile["damage_model"]
    missing_keys = {(m["entity_id"], m["attribute"]) for m in layer.missing}
    truth._deriving = True
    try:
        for entity in layer.entities:
            eid = entity["id"]
            source = "authored" if entity["label_source"] == "authored" else "upstream"
            _add(truth, eid, entity["class"], entity["geometry"], {}, source)
            for key, value in entity["attributes"].items():
                truth.set(eid, key, value, "authored" if (eid, key) in missing_keys else source)
    finally:
        truth._deriving = False
    truth.secondary_events = deepcopy(profile.get("secondary_events", []))
    truth.secondary_events.extend(deepcopy(truth.overlay_effects["world"].get("secondary_events", [])))
    for index, event in enumerate(truth.secondary_events, 1):
        event.setdefault("id", f"secondary-{index}")
    truth.secondary_events.sort(key=lambda event: (event.get("t",0),event["id"]))
    # G02: the occupancy RNG and inputs are independent of G03/G04 damage.
    prior = profile["occupancy_prior"]
    for building in truth.entities_of("building"):
        bid = building["id"]
        r = rng_for(seed, "G02", bid)
        capacity = max(0, int(truth.get(bid, "building.capacity", 0)))
        rate = float(prior.get("by_use", {}).get(truth.get(bid, "building.use"), prior.get("default", .4)))
        count = sum(r.random() < rate for _ in range(capacity))
        count = min(count, int(prior.get("max_per_building", capacity)))
        if bid in options.get("pinned_occupants", {}):  # compact demo only (authored_demo)
            count = int(options["pinned_occupants"][bid])
        occupants = []
        for i in range(count):
            hid = f"human-{bid}-{i+1}"
            _add(truth, hid, "human", _point(_center(building["geometry"])),
                 {"entity.parent_id": bid, "human.location_context": "interior", "human.needs": []}, "G02")
            occupants.append(hid)
        truth.set(bid, "building.occupant_ids", occupants, "G02")
    # Outdoor populations are authored occupancy priors, not inferred casualties.
    outdoor_count = int(options.get("outdoor_count", prior.get("outdoor_count", 3 if profile["family"] in ("野外", "地下", "水域") else 1)))
    for i in range(max(0, outdoor_count)):
        r = rng_for(seed, "G02-outdoor", str(i))
        water = truth.entities_of("water")
        if profile_id == "SC05" and water:
            geometry = water[i % len(water)]["geometry"]
            coords = geometry["coords"]
            xy = _center(geometry)
            for _ in range(64):
                candidate = (r.uniform(min(p[0] for p in coords),max(p[0] for p in coords)),
                             r.uniform(min(p[1] for p in coords),max(p[1] for p in coords)))
                if _inside(candidate,coords):
                    xy = candidate
                    break
            context = "water"
        else:
            xy = (truth.bounds[2]*r.uniform(.2, .8), truth.bounds[3]*r.uniform(.2, .8))
            context = "outside"
        _add(truth, f"human-outdoor-{i+1}", "human", _point(xy),
             {"human.location_context": context, "human.needs": []}, "G02")
    # Outdoor people placed by an authored layout (e.g. a person in flood water); G06 and
    # pinned_humans then apply exactly as for G02 people. Compact demo only, still hidden truth.
    for hid, spec in options.get("placed_humans", {}).items():
        _add(truth, hid, "human", _point(spec["xy"]),
             {"human.location_context": spec["location_context"], "human.needs": []}, options.get("cause", "authored_demo"))
    # G03: spatial footprints and independently seeded intensities.
    for index, kind in enumerate(profile["hazards"], 1):
        r = rng_for(seed, "G03", kind)
        low, high = model.get("intensity", [.2, .8])
        intensity = r.uniform(low, high)
        matching = truth.entities_of("water" if kind in ("flood", "tsunami") else "slope" if kind in ("landslide", "volcano") else "underground" if profile["family"] == "地下" else "open_area")
        if matching and kind in ("flood", "tsunami", "landslide", "volcano"):
            geometry = deepcopy(matching[0]["geometry"])
        else:
            w, h = truth.bounds[2:]
            geometry = {"type": "polygon", "coords": [[w*.2,h*.2],[w*.8,h*.2],[w*.8,h*.85],[w*.2,h*.85],[w*.2,h*.2]]}
        _add(truth, f"hazard-{index}", "hazard", geometry,
             {"hazard.kind": kind, "hazard.footprint": deepcopy(geometry), "hazard.intensity": intensity,
              "hazard.active": kind != "none", "hazard.secondary_event": deepcopy(truth.secondary_events) if index == 1 else [],
              "hazard.water_depth_m": model.get("water_depth_m", 0.0) if kind in ("flood", "tsunami") else 0.0,
              "hazard.flow_speed_mps": model.get("flow_speed_mps", .2) if kind in ("flood", "tsunami") else 0.0,
              "hazard.fire_intensity": intensity if kind == "fire" else 0.0,
              "hazard.smoke_density": model.get("smoke_density", 0.0),
              "hazard.gas_level": model.get("gas_level", 0.0), "hazard.wind_mps": model.get("wind_mps", 0.0),
              "hazard.visibility_m": truth.overlay_effects["world"].get("visibility_m", 800.0),
              "hazard.illumination": truth.overlay_effects["world"].get("illumination", {"lux": 10000.0, "source": "authored_uncalibrated"}),
              "hazard.exclusion_zone": False}, "G03")
    for hid, spec in options.get('pinned_hazards', {}).items():
        if hid not in truth.entities or truth.entities[hid]['cls'] != 'hazard':
            raise ValueError('Configured hazard is absent from profile: '+hid)
        geometry = spec['geometry']
        truth.set(hid, 'entity.footprint', geometry, options['cause'])
        truth.set(hid, 'hazard.footprint', geometry, options['cause'])
        _write(truth, hid, spec['fields'], options['cause'])
    # G04: hazard × structure, with a spatial attenuation and separate noise stream.
    for building in truth.entities_of("building"):
        bid, xy = building["id"], _center(building["geometry"])
        r = rng_for(seed, "G04", bid)
        exposure = max((truth.get(h["id"], "hazard.intensity", 0) * (1.0 if _inside(xy, h["geometry"]["coords"]) else .25)
                        for h in truth.entities_of("hazard")), default=0.0)
        fragility = model.get("fragility", {}).get(truth.get(bid, "building.structure"), .8)
        score = max(0.0, min(1.0, exposure*fragility*r.uniform(.5, 1.3)))
        grade = sum(score > threshold for threshold in (.18, .43, .72))
        roof = min(1.0, score * r.uniform(.4, 1.2))
        if "wind" in profile["hazards"] and truth.get(bid, "building.structure") == "RC":
            grade = min(grade, 1)  # Strong RC is not authored as uniformly fragmented by wind.
        _write(truth, bid, {"building.visual_damage": ["none", "minor", "major", "destroyed"][grade],
                           "building.structural_condition": ["stable", "compromised", "unstable", "collapsed"][grade],
                           "building.wall_damage": {"crack_ratio": round(score*.7, 3), "breach_ratio": round(max(0.0,score-.35), 3)},
                           "building.roof_loss_ratio": round(roof, 3), "building.interior_mode": "abstract",
                           "building.search_difficulty": round(.2 + score*.7, 3),
                           "building.interior_area_m2": round(_area(building["geometry"])*truth.get(bid,"building.floors",1), 2),
                           "building.search_coverage_truth": 0.0,
                           "building.flood_exposure": "flooded" if any(truth.get(h["id"], "hazard.water_depth_m", 0) > 0 and _inside(xy,h["geometry"]["coords"]) for h in truth.entities_of("hazard")) else "not_flooded"}, "G04")
    for bid, values in options.get("pinned_buildings", {}).items():  # compact demo only
        if bid in truth.entities:
            _write(truth, bid, values, options.get("cause", "authored_demo"))
    # G05: obstacles retain a mother structure/tree and explicit clearance geometry.
    for road in truth.entities_of("road"):
        rid = road["id"]
        xy = _center(road["geometry"])
        water_depth = max((truth.get(h["id"], "hazard.water_depth_m", 0) for h in truth.entities_of("hazard")
                           if _hits(road["geometry"], h["geometry"]["coords"])), default=0.0)
        base_step = truth.get(rid, "road.step_height_m", 0.0)
        # Capture the physical baseline before obstacle-derived writes can run.
        truth.set(rid, "attr.base_step_height_m", base_step, "authored" if "road.step_height_m" not in road["fields"] else road["provenance"].get("road.step_height_m", "upstream"))
        _write(truth, rid, {"road.obstacle_ids": [], "road.entry_state": "open",
                           "road.water_depth_m": max(water_depth,truth.get(rid,"road.water_depth_m",0.0)),
                           "road.clear_height_m": truth.get(rid,"road.clear_height_m",20.0)}, "D01")
    sources = truth.entities_of("building") + truth.entities_of("tree") + truth.entities_of("tree_cluster")
    for source in sources:
        sid = source["id"]
        r = rng_for(seed, "G05", sid)
        tree = source["cls"] in ("tree", "tree_cluster")
        # Trees fall only where the profile lists trees as affected (wind/landslide/fire...), and
        # debris can only reach a road from a source within plausible reach (tree height / collapse
        # spread), so a distant source never blocks its "nearest" road. Main-agent calibration.
        if tree and not ({"tree", "tree_cluster"} & set(profile.get("affected_classes", ()))):
            continue
        damaged = tree or truth.get(sid, "building.visual_damage") != "none"
        if sid in options.get("no_debris_from", ()):
            continue
        if not damaged or r.random() >= model.get("obstacle_rate", .35):
            continue
        c0 = _center(source["geometry"])

        def _road_gap(rd):
            return min((_distance(_projection(c0, a, b)[0], c0)
                        for a, b in zip(rd["geometry"]["coords"], rd["geometry"]["coords"][1:])), default=1e9)
        roads_all = truth.entities_of("road")
        road = min(roads_all, key=_road_gap) if roads_all else None
        if road is None or _road_gap(road) > (20.0 if tree else 45.0):
            continue
        rid = road["id"]
        # Closest projection keeps debris on a road adjacent to its source.
        pairs = list(zip(road["geometry"]["coords"], road["geometry"]["coords"][1:]))
        if not pairs:
            continue
        xy = min((_projection(_center(source["geometry"]), a, b)[0] for a,b in pairs), key=lambda p: _distance(p,_center(source["geometry"])))
        width = truth.get(rid, "road.width_m", 0)
        kind = "tree" if tree else "roof_sheet" if "wind" in profile["hazards"] else "rubble"
        volume = r.uniform(.2, 3.5)
        oid = f"obstacle-{sid}"
        _add(truth, oid, "obstacle", _point(xy),
             {"obstacle.kind": kind, "obstacle.material": "vegetation" if tree else "steel" if kind == "roof_sheet" else "mixed",
              "obstacle.source_id": sid, "obstacle.volume_m3": volume, "obstacle.height_m": r.uniform(.05,.5),
              "obstacle.state": "present", "obstacle.work_remaining": volume*30,
              "obstacle.blocks_ids": [rid], "attr.initial_work": volume*30,
              "attr.blocked_width_m": max(0.0,width-r.uniform(.35,1.4))}, "G05")
        truth.set(rid, "road.obstacle_ids", truth.get(rid,"road.obstacle_ids",[])+[oid], "G05")
    # Scene-config disaster slots are physical initial conditions, not tasks.
    for slot in options.get("obstacles", []):
        rid, oid = slot["road"], slot["id"]
        if rid not in truth.entities or truth.entities[rid]["cls"] != "road":
            raise ValueError("Configured debris requires a semantic road: " + rid)
        if oid in truth.entities:
            raise ValueError("Duplicate configured obstacle: " + oid)
        _add(truth, oid, "obstacle", _point(slot["pose"]),
             {"obstacle.kind": "rubble", "obstacle.material": "mixed",
              "obstacle.source_id": slot["source"], "obstacle.volume_m3": slot["volume_m3"],
              "obstacle.height_m": slot["height_m"], "obstacle.state": "present",
              "obstacle.work_remaining": slot["work"], "attr.initial_work": slot["work"],
              "obstacle.blocks_ids": [rid], "attr.blocked_width_m": slot["blocked_width_m"]},
             options.get("cause", "authored_scene_config"))
        truth.set(rid, "road.obstacle_ids", truth.get(rid, "road.obstacle_ids", []) + [oid], "G05")
    truth._build_graph()
    for road in truth.entities_of("road"):
        truth._deriving = True
        try:
            truth._derive_road(road["id"])
        finally:
            truth._deriving = False
    # G06 changes existing people only; intact buildings keep their G02 occupants.
    for human in truth.entities_of("human"):
        hid = human["id"]
        r = rng_for(seed, "G06", hid)
        parent = truth.get(hid, "entity.parent_id")
        structural = truth.get(parent, "building.structural_condition", "stable")
        grade = {"stable":0,"compromised":1,"unstable":2,"collapsed":3}.get(structural,0)
        context = truth.get(hid,"human.location_context","outside")
        xy = _center(human["geometry"])
        if context == "outside" and any(truth.get(h["id"],"hazard.water_depth_m",0) > 0 and _inside(xy,h["geometry"]["coords"]) for h in truth.entities_of("hazard")):
            context = "water"
        exposure = max((truth.get(h["id"],"hazard.intensity",0) for h in truth.entities_of("hazard")
                        if truth.get(h["id"],"hazard.active",False) and _inside(xy,h["geometry"]["coords"])),default=0.0)
        trapped = grade >= 2 and r.random() < .65
        mobility = "trapped" if trapped else r.choices(["mobile","limited","immobile"],[.55,.3,.15])[0]
        injury = r.choices(["none","minor","serious","critical"],
                           [max(.1,.65-grade*.15-exposure*.2),.25,.08+grade*.1+exposure*.15,.02+grade*.03+exposure*.05])[0]
        needs = ["water"]
        if trapped:
            needs.append("extrication")
        if injury in ("serious","critical"):
            needs.append("medical_supplies")
        if mobility != "mobile" or context == "water":
            needs.append("transport")
        if context == "water":
            mobility = "limited"
        _write(truth, hid, {"human.mobility": mobility, "human.buriedness": r.uniform(5,60) if trapped else 0.0,
                           "human.injury": injury, "human.needs": needs, "human.alive": True,
                           "human.location_context": "rubble" if trapped and grade == 3 else context,
                           "human.pose": list(_center(human["geometry"])),
                           "human.handover_status": "not_started", "human.support_received": [],
                           "human.carrier_id": None, "human.destination_id": None,
                           "human.visibility_context": {"occluded": context == "interior", "exterior_exposed": context != "interior", "source": "authored_uncalibrated"},
                           "human.health_model": {"model_id": "abstract_no_clinical_evolution", "source": "authored_uncalibrated"}, "human.health_value": 1.0}, "G06")
    for hid, values in options.get("pinned_humans", {}).items():  # compact demo only, still hidden truth
        if hid in truth.entities:
            _write(truth, hid, values, options.get("cause", "authored_demo"))
    # Imported/placed people receive their parent and final context after G02.
    # Index those actual indoor people too; pinned_occupants=0 suppresses new
    # sampling, not the occupants already materialized by the upstream scene.
    occupants_by_building = {building["id"]: [] for building in truth.entities_of("building")}
    for human in truth.entities_of("human"):
        hid = human["id"]
        parent = truth.get(hid, "entity.parent_id")
        if parent in occupants_by_building and truth.get(hid, "human.location_context") in ("interior", "rubble"):
            occupants_by_building[parent].append(hid)
    for bid, occupants in occupants_by_building.items():
        if truth.get(bid, "building.occupant_ids") != occupants:
            truth.set(bid, "building.occupant_ids", occupants, options.get("cause", "G02"))
    # G07 includes unrelated and stale evidence, never labels it as a detection.
    humans = truth.entities_of("human")
    for i in range(max(3, len(humans)//2)):
        r = rng_for(seed, "G07", str(i))
        related = humans[i % len(humans)] if humans and i % 3 else None
        xy = _center(related["geometry"]) if related else (truth.bounds[2]*r.uniform(.1,.9), truth.bounds[3]*r.uniform(.1,.9))
        stale = i % 4 == 0
        _add(truth, f"cue-{i+1}", "cue", _point(xy),
             {"cue.kind": r.choice(["Backpack","Bottle","phone","Shoe","helmet"]),
              "cue.related_human_id": related["id"] if related else None, "cue.location": list(xy),
              "cue.age_s": r.uniform(7200,86400) if stale else r.uniform(0,600),
              "cue.signal_state": "inactive" if stale else "intermittent", "cue.detectability": r.uniform(.3,.9),
              "cue.source_local_id": f"{profile_id}:authored:{i+1}"}, "G07")
    # D03: one physical item id per unit, inventories derived from holders, no copies.
    inventory_factor = float(truth.overlay_effects["energy"].get("inventory_factor", 1.0))
    stock = {"water": 6, "medical_pack": 3, "relay": 3, "battery": 3}
    stock.update(profile.get("inventory_prior", {}))
    stock.update(options.get("station_stock", {}))
    masses = {"water":1.0,"medical_pack":.5,"relay":.8,"battery":1.5,"food":.5,"tool":2.0,"stretcher":5.0}
    for station in truth.entities_of("station"):
        sid = station["id"]
        _write(truth, sid, {"station.occupied": 0, "station.inventory": {}, "station.open_state": "open",
                           "station.queue_ids": [], "station.accepted_need_types": ["water","medical_supplies","transport"]}, "D03")
        for kind, quantity in stock.items():
            if kind not in truth._catalog.enum_domain("item.kind") or quantity < 0:
                raise ValueError("无效库存 / Invalid station_stock: " + str(kind))
            for index in range(int(quantity*inventory_factor)):
                _add(truth, f"item-{sid}-{kind}-{index+1}", "item", _point(_center(station["geometry"])),
                     {"item.kind": kind, "item.mass_kg": masses.get(kind,1), "item.quantity": 1,
                      "item.state": "stock", "item.release_state": "locked", "item.holder_id": sid}, "D03")
    truth.objectives = derive_objectives(truth)
    return truth


def derive_objectives(truth):
    """Desired physical/evaluation outcomes, explicitly NOT agent task records."""
    objectives = []

    def add(kind, target, desired, priority, public, rationale, refs):
        objectives.append({"id": f"objective-{len(objectives)+1}", "kind": kind, "target": target,
                           "desired": desired, "priority": priority, "public": public,
                           "rationale_zh": rationale, "refs": refs})

    for building in truth.entities_of("building"):
        add("assess_all_buildings", building["id"], {"evaluation.assessment_coverage": 1.0}, 70, True,
            "所有建筑均在评估范围内；这是世界目标，不是任务。", ["U06","E01","F361"])
    for human in truth.entities_of("human"):
        hid = human["id"]
        if truth.get(hid, "attr.population_role") == "ordinary":
            continue  # Source residents remain sensable world entities, outside casualty objectives.
        # A finding is evaluator evidence; no physical "found/rescued" truth flag is invented.
        add("find_person", hid, {"evaluation.detection_counts": {"unique_confirmed": 1}}, 90, False,
            "人员位置须由观测证据确认；隐藏真值不直接派生任务。", ["G02","G06","E01"])
        if truth.get(hid,"human.mobility") == "trapped":
            add("extricate",hid,{"human.mobility":["mobile","limited","immobile"]},100,False,"解除受困不等于完成移交。",["G06","T03"])
        if truth.get(hid,"human.injury") in ("minor","serious","critical"):
            add("aid",hid,{"human.support_received":["treated"]},95,False,"需要独立登记救护支持。",["T03","U12"])
        add("handover",hid,{"human.handover_status":"acknowledged"},90,False,"到达不是接收确认。",["T03","U12","F085"])
        if truth.get(hid,"human.location_context") == "water":
            add("recover_from_water",hid,{"human.location_context":["on_carrier","safe_point"]},100,False,"水中目标需要具备能力的主体实施回收。",["T03","A023"])
    for road in truth.entities_of("road"):
        if truth.get(road["id"],"road.entry_state") != "open":
            add("restore_access",road["id"],{"road.entry_state":"open"},75,False,"通路需按实体净空条件恢复。",["D01"])
    # Supply objectives bind existing conserved item ids; no delivery is pre-completed.
    needy = [h for h in truth.entities_of("human") if "water" in truth.get(h["id"],"human.needs",[])]
    water_items = [i for i in truth.entities_of("item") if truth.get(i["id"],"item.kind") == "water"]
    for item, human in zip(water_items, needy):
        add("deliver_supplies", item["id"], {"item.state":"received", "item.receiver_id":human["id"]},60,False,
            "物资释放或接触不代表收货。",["D03","T03","U12"])
    for hazard in truth.entities_of("hazard"):
        hid = hazard["id"]
        if truth.get(hid,"hazard.fire_intensity",0) > 0:
            add("suppress_fire",hid,{"hazard.fire_intensity":0.0},95,False,"抑制火势并允许次生复燃。",["A062","G03"])
        if truth.get(hid,"hazard.gas_level",0) > 0 or any(
                truth.get(e['id'], 'attr.leak_hazard_id') == hid for e in truth.entities_of('facility')):
            add("isolate_hazard",hid,{"hazard.exclusion_zone":True},95,False,"建立危险隔离区不代表危险已消失。",["A063","G03"])
    for link in truth.entities_of("link"):
        if truth.get(link["id"],"link.state") != "reachable":
            add("restore_link",link["id"],{"link.state":"reachable"},65,False,"链路恢复仍不代表消息已经送达。",["D05","U12"])
    return objectives


def apply_secondary(truth, event_spec, t):
    """Apply authored aftershock, rise, spread or movement; return every write.

    Event selectors are deterministic; an absent population/network yields no
    invented victim, obstacle, success or task. The kernel owns event emission.
    """
    kind = event_spec.get("kind")
    if kind not in ("aftershock_reblock", "water_rise", "fire_spread", "target_moves", "moving_target"):
        raise ValueError("未知次生事件 / Unknown secondary event: " + str(kind))
    params = deepcopy(event_spec.get("params", {}))
    target = event_spec.get("target", params.get("target"))
    start = len(truth.history)
    truth.t = float(t)
    cause = {"rule":"G03", "t":t, "refs":["F023"]+list(event_spec.get("refs",[])), "scripted":True}

    def select(candidates):
        candidates = sorted(candidates, key=lambda e: e["id"])
        return rng_for(truth.seed,"secondary-target",f"{kind}|{event_spec.get('id','')}|{t}").choice(candidates) if candidates else None

    if kind == "aftershock_reblock":
        candidates = truth.entities_of("road")
        road = truth.entities.get(target) if target else select(candidates)
        if road and road["cls"] == "obstacle":
            road = truth.entities.get(next(iter(truth.get(road["id"],"obstacle.blocks_ids",[])), ""))
        if road and road["cls"] == "road":
            rid = road["id"]
            old = truth.get(rid,"road.obstacle_ids",[])
            oid = f"secondary-obstacle-{len(truth.entities)+1}"
            width = truth.get(rid,"road.width_m",0)
            work = float(params.get("work_remaining",60))
            _add(truth,oid,"obstacle",_point(_center(road["geometry"])),
                 {"obstacle.kind":params.get("obstacle_kind","rubble"),"obstacle.material":"vegetation" if params.get("obstacle_kind") == "tree" else "mixed", "obstacle.source_id":rid,
                  "obstacle.state":"present", "obstacle.volume_m3":float(params.get("volume_m3",2)),
                  "obstacle.height_m":float(params.get("step_height_m",.25)), "obstacle.work_remaining":work,
                  "attr.initial_work":work, "attr.blocked_width_m":max(0.0,width-float(params.get("clear_width_m",.4))),
                  "obstacle.blocks_ids":[rid]},cause)
            truth.set(rid,"road.obstacle_ids",old+[oid],cause)
    elif kind == "water_rise":
        delta = max(0.0,float(params.get("delta_m",params.get("rise_m",.4))))
        hazards = [h for h in truth.entities_of("hazard") if truth.get(h["id"],"hazard.kind") in ("flood","tsunami") and (not target or h["id"] == target)]
        for hazard in hazards:
            hid = hazard["id"]
            truth.set(hid,"hazard.water_depth_m",truth.get(hid,"hazard.water_depth_m",0)+delta,cause)
        footprints = [h["geometry"]["coords"] for h in hazards]
        for road in truth.entities_of("road"):
            if (target == road["id"]) or any(_hits(road["geometry"],poly) for poly in footprints):
                rid = road["id"]
                # Overlapping flood/tsunami footprints must not double-count a rise.
                truth.set(rid,"road.water_depth_m",truth.get(rid,"road.water_depth_m",0)+delta,cause)
        for human in truth.entities_of("human"):
            if any(_inside(_center(human["geometry"]),poly) for poly in footprints) and truth.get(human["id"],"human.location_context") == "outside":
                truth.set(human["id"],"human.location_context","water",cause)
                truth.set(human["id"],"human.mobility","limited",cause)
    elif kind == "fire_spread":
        hazards = [h for h in truth.entities_of("hazard") if truth.get(h["id"],"hazard.kind") == "fire" and (not target or h["id"] == target)]
        for hazard in hazards:
            hid = hazard["id"]
            intensity = max(0.0,min(1.0,truth.get(hid,"hazard.fire_intensity",0)+float(params.get("intensity_delta",params.get("delta_intensity",.25)))))
            smoke = max(0.0,min(1.0,truth.get(hid,"hazard.smoke_density",0)+float(params.get("smoke_delta",0))))
            _write(truth,hid,{"hazard.fire_intensity":intensity,"hazard.intensity":intensity,"hazard.smoke_density":smoke,"hazard.active":True},cause)
            for facility in truth.entities_of('facility'):
                fid = facility['id']; release = truth.get(fid, 'attr.leak_on_fire_spread')
                if (release is None or release['hazard_id'] != hid or truth.get(fid, 'attr.leak_active') is not False
                        or intensity < release['onset_fire_intensity']):
                    continue
                bund = truth.get(fid, 'attr.bund_id'); gas = truth.get(fid, 'attr.leak_hazard_id')
                truth.set(fid, 'attr.leak_active', True, cause)
                truth.set(bund, 'attr.spill_volume_m3', truth.get(bund, 'attr.spill_volume_m3')+release['released_volume_m3'], cause)
                _write(truth, gas, {'hazard.active': True, 'hazard.gas_level': release['gas_level'],
                                   'hazard.intensity': release['gas_level']}, cause)
            radius = float(params.get("spread_m",80))
            coords = hazard["geometry"]["coords"]
            center = _center(hazard["geometry"])
            expanded = []
            for x,y in coords:
                dx,dy = x-center[0], y-center[1]
                norm = math.hypot(dx,dy) or 1
                expanded.append([max(truth.bounds[0],min(truth.bounds[2],x+radius*dx/norm)),
                                 max(truth.bounds[1],min(truth.bounds[3],y+radius*dy/norm))])
            geometry = {"type":"polygon","coords":expanded}
            truth.set(hid,"entity.footprint",geometry,cause)
            truth.set(hid,"hazard.footprint",geometry,cause)
            for road in truth.entities_of("road"):
                if _hits(road["geometry"],expanded):
                    truth.set(road["id"],"road.entry_state","unsafe",cause)
    else:
        humans = [h for h in truth.entities_of("human") if truth.get(h["id"],"human.mobility") in ("mobile","limited")]
        outdoor = [h for h in humans if truth.get(h["id"],"human.location_context") in ("outside","water")]
        human = truth.entities.get(target) if target else select(outdoor or humans)
        if human and human["cls"] == "human" and truth.get(human["id"],"human.mobility") in ("mobile","limited"):
            hid = human["id"]
            xy = params.get("pose",params.get("to",params.get("position")))
            if xy is None:
                old = _center(human["geometry"])
                offset = params.get("offset_m",params.get("offset",[30,15]))
                xy = [old[0]+offset[0],old[1]+offset[1]]
            xy = [max(truth.bounds[0],min(truth.bounds[2],float(xy[0]))),
                  max(truth.bounds[1],min(truth.bounds[3],float(xy[1])))]
            if truth.get(hid,"human.location_context") == "water":
                polygons = [e["geometry"]["coords"] for e in truth.entities_of("water") if e["geometry"]["type"] == "polygon"]
                if not any(_inside(xy,p) for p in polygons):
                    return []  # Drift does not magically recover a person onto dry land.
            truth.set(hid,"entity.pose",xy,cause)
            parent = truth.get(hid,"entity.parent_id")
            if parent in truth.entities and not _inside(xy,truth.entities[parent]["geometry"]["coords"]):
                occupants = truth.get(parent,"building.occupant_ids",[])
                if hid in occupants:
                    truth.set(parent,"building.occupant_ids",[eid for eid in occupants if eid != hid],cause)
                truth.set(hid,"entity.parent_id",None,cause)
                if truth.get(hid,"human.location_context") != "water":
                    truth.set(hid,"human.location_context","outside",cause)
            context = params.get("location_context")
            if context:
                truth.set(hid,"human.location_context",context,cause)
    return [(h["eid"],h["key"],deepcopy(h["before"]),deepcopy(h["after"])) for h in truth.history[start:]]
