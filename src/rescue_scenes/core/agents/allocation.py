"""Catalog methods and belief-only allocation (SPEC §§6.5–6.7).

No function in this module reads ``sim.world`` or another runtime.  Fleet rows
are received reports, not live platform objects.  The caller emits the returned
allocation decision before executing A052 and sending its assignment.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import heapq
import json
from math import hypot, inf, isfinite

from .embodiment import EMBODIMENTS
from .knowledge import BeliefStore
from .taskgen import is_meta_task
from ..world.truth import (isolated_ground_route, _footprint_intersects, wheeled_navigation_goal,
                          building_approach_attachment, building_approach_leg,
                          interior_approach_buildings, _prepare_polygon, _exact_key,
                          _footprint_candidates)


_SPECIALIST = "具备对应设备的专业/外部执行主体"
_MOVEMENT = {"A001": "air", "A002": "ground", "A003": "water"}
# Local sensing/inspection also needs an approach; delegated A059 work does not.
_PHYSICAL = {*(f"A{i:03d}" for i in range(5, 32)), "A062", "A063"} - {"A006", "A007"}
_TRAVEL_CAPABILITY = {"air": "C01", "legged": "C02", "wheeled": "C02", "foot": "C02", "water": "C03"}
_MOVE_POWER_W = {"air": 180.0, "legged": 80.0, "wheeled": 800.0, "water": 350.0, "foot": 30.0}
_TERMINAL = {"completed", "failed", "canceled", "expired"}
_TASK_MANAGEMENT = frozenset(f"A{i:03d}" for i in range(50, 59))
# These methods commit an actual tool operator or carrier. A059 is not a
# substitute for that executor's installed equipment or carrying capacity.
_EXECUTOR_ACTIONS = frozenset({"A012", "A013", "A016", "A020", "A021", "A022", "A023", "A062", "A063"})
# Clinical tasks commit the assessor/treating practitioner, rather than a
# requester that may delegate to an already busy team outside allocation.
_AID_ACTIONS = frozenset({"A018", "A019"})
# Authored point-obstacle model: 2 m longitudinal half-footprint + 1 m work gap.
# This is an approach boundary, never permission to traverse the obstacle.
OBSTACLE_WORK_STANDOFF_M = 3.0

# Process-local, content-only caches: never retain a belief store or runtime.
# Barrier keys include ordered ids and every prepared coordinate (via _exact_key,
# including numeric types/signed zero). Flags/exclusions are resolved before
# keying, so any change to the effective ordered barriers selects another entry.
# Keep at most 8 barrier sets, each with at most 32768 directed edge contacts.
_CLEARANCE_CACHE_LIMIT = 8
_CONTACT_CACHE_LIMIT = 32768
_CLEARANCE_CACHE = OrderedDict()
# Topology keys contain the ordered pre-crossing segment data and blocked ids;
# passability/work sections are still evaluated from current beliefs each round.
# At most 32 topologies survive between rounds. Endpoint cuts are never stored.
_TOPOLOGY_CACHE_LIMIT = 32
_TOPOLOGY_CACHE = OrderedDict()
# Base graphs use content-derived topology/barrier tokens plus the exact ordered
# interior entrance rows. Tokens are renewed after their owning LRU evicts an
# entry, so no runtime/snapshot identity can alias a different graph. Only the
# endpoint candidate segments are rebuilt; at most 32 detached base graphs survive.
# Source-region attachment cuts/permissions stay query-local, not in these LRUs.
_BELIEF_GRAPH_CACHE_LIMIT = 32
_BELIEF_GRAPH_CACHE = OrderedDict()


def _table(catalog, name):
    table = catalog.get(name, {}) if isinstance(catalog, dict) else getattr(catalog, name, {})
    return table if isinstance(table, dict) else {row["id"]: row for row in table}


def _caps(catalog, action_id):
    return set(_table(catalog, "actions").get(action_id, {}).get("capabilities", ()))


def _compatible(action, spec):
    mobility = spec.get("mobility", "none")
    return (action not in _MOVEMENT
            or action == "A001" and mobility == "air"
            or action == "A002" and mobility in {"legged", "wheeled", "foot"}
            or action == "A003" and mobility == "water")


def _steps(catalog, seq_id):
    rows = catalog.get("steps", {}) if isinstance(catalog, dict) else catalog.steps
    if isinstance(rows, dict):
        return rows.get(seq_id, [])
    return [row for row in rows if row.get("sequence") == seq_id]


def resolve_method(catalog, task_type, spec):
    """Select the first main path and resolve each OR exactly once.

    ``details[i].action_id`` is the original selected catalog action.  A missing
    specialist-only capability is represented by A059 in ``resolved_steps``;
    ``details[i].delegated`` tells the runtime to request that original action,
    wait for the actual specialist result, and never pretend the request did it.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    tasks = _table(catalog, "tasks")
    if task_type not in tasks:
        raise ValueError(f"未知任务 / unknown catalog task: {task_type}")
    if hasattr(catalog, "task_methods"):
        methods = catalog.task_methods(task_type)
    else:
        sequences = _table(catalog, "sequences")
        methods = [sid for sid, row in sequences.items()
                   if task_type in row.get("tasks", ()) and row.get("kind") == "参考主路径"]
    if not methods:
        raise ValueError(f"任务没有参考主路径 / task has no main method: {task_type}")
    seq_id = methods[0]
    owned = set(spec.get("capabilities", ()))
    executor_actions = _EXECUTOR_ACTIONS | (_AID_ACTIONS if task_type in {"T029", "T031"} else set())
    resolved, details = [], []
    for index, original in enumerate(_steps(catalog, seq_id)):
        options = list(original.get("options", original.get("actions", ())))
        if not options:
            raise ValueError(f"空序列步骤 / empty sequence step: {seq_id}:{index + 1}")
        # Do not substitute an action that the source does not offer.  In
        # particular an air/ground OR is not permission to make a boat walk.
        if "A001" in options and "A002" in options:
            action = "A001" if spec.get("mobility") == "air" else "A002"
        else:
            viable = [a for a in options if _compatible(a, spec) and _caps(catalog, a) <= owned]
            action = viable[0] if viable else options[0]
        specialist = original.get("actor") == _SPECIALIST
        delegated = specialist and action not in executor_actions and not _caps(catalog, action) <= owned
        executable = "A059" if delegated else action
        resolved.append(executable)
        details.append({
            "step": original.get("step", index + 1), "action_id": action,
            "resolved_action_id": executable, "options": options,
            "choose_one": bool(original.get("choose_one", len(options) > 1)),
            "actor": original.get("actor", ""), "specialist": specialist,
            "delegated": delegated, "required_capabilities": sorted(_caps(catalog, action)),
            "layer": _table(catalog, "actions").get(executable, {}).get("layer", ""),
            # Runtime resolves the actual original/generated task id and rev;
            # the protocol's own task record must not be its management target.
            "manages": "substantive_task" if is_meta_task(task_type)
                       and action in _TASK_MANAGEMENT else "self",
            "rule": original.get("rule", ""), "terminal": original.get("terminal", ""),
            "original_step": deepcopy(original),
        })
    if not resolved:
        raise ValueError(f"序列没有步骤 / sequence has no steps: {seq_id}")
    # Project the requester's A059 out when the assigned executor owns the
    # immediately following work. Keep the omitted role explicit, not a fake
    # successful action or a request from a busy team to itself.
    requester_steps = []
    for index in range(len(resolved) - 2, -1, -1):
        following = details[index + 1]
        if (resolved[index] == "A059" and details[index]["action_id"] == "A059"
                and following["action_id"] in executor_actions
                and _caps(catalog, following["action_id"]) <= owned):
            requester_steps.append(details.pop(index))
            resolved.pop(index)
    return {"seq_id": seq_id, "resolved_steps": resolved, "step_index": 0, "details": details,
            "requester_steps": requester_steps}


def _requester_actions(method):
    """Project a method onto actions performed by its holder, not remote roles."""
    details = method.get("details", [])
    return [action for index, action in enumerate(method["resolved_steps"])
            if index >= len(details) or not str(details[index].get("actor", "")).startswith("远端")]


def required_capabilities(catalog, task_type, spec, *, method=None):
    """Task capabilities intersect candidate-resolved requester action requirements.

    Catalog actors beginning 远端 execute on the receiving node and are guarded
    there, not as installed equipment on the requester.  Delegated specialist
    equipment likewise stays in details for the receiver's independent checks;
    the actual A059 interface (C46/C31) is required of its requester instead.
    """
    method = resolve_method(catalog, task_type, spec) if method is None else method
    action_caps = set()
    for action in _requester_actions(method):
        action_caps.update(_caps(catalog, action))
    required = set(_table(catalog, "tasks")[task_type].get("capabilities", ())) & action_caps
    if any(step["delegated"] and not str(step.get("actor", "")).startswith("远端")
           for step in method["details"]):
        required.update(_caps(catalog, "A059"))
    return sorted(required)


def _value(entry, default=None):
    if entry is None:
        return default
    value = entry.get("value", default) if isinstance(entry, dict) and "value" in entry else entry
    return default if isinstance(value, str) and value in {"unknown", "unobserved", "未知"} else value


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if isfinite(number) else default


def _belief_entities(belief):
    return belief.entities(copy=False) if hasattr(belief, "entities") else belief


def _field(fields, key, default=None):
    return _value(fields.get(key), default)


def _point(value):
    if isinstance(value, dict):
        value = value.get("xy", value.get("position", value.get("coords")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if all(isinstance(v, (int, float)) and isfinite(v) for v in value[:2]):
            return (float(value[0]), float(value[1]))
    return None


def _coords(geometry):
    if not isinstance(geometry, dict):
        return []
    rows = geometry.get("coords", geometry.get("coordinates", []))
    if not isinstance(rows, (list, tuple)):
        return []
    if _point(rows) is not None:
        return [_point(rows)]
    while rows and isinstance(rows[0], (list, tuple)) and rows[0] and isinstance(rows[0][0], (list, tuple)):
        rows = rows[0]
    return [point for row in rows if (point := _point(row)) is not None]


def _position(fields):
    poses = [(fields[key].get("t", 0) if isinstance(fields[key], dict) else 0, point) for key in
             ("entity.pose", "human.pose", "belief.estimated_pose")
             if (point := _point(_field(fields, key))) is not None]
    if poses:
        return max(poses, key=lambda item: item[0])[1]
    points = _coords(_field(fields, "entity.geometry", {}))
    if not points:
        return None
    return tuple(sum(p[i] for p in points) / len(points) for i in (0, 1))


def water_access_goal(goal, entities):
    """Use a receiver's public F150 water berth, preserving its dry station pose."""
    for fields in entities.values():
        if _field(fields, 'entity.kind') != 'station':
            continue
        position = _position(fields)
        access = _coords(_field(fields, 'station.access_region'))
        if position is not None and access and hypot(goal[0]-position[0], goal[1]-position[1]) < 1e-6:
            if access[0] == access[-1]:
                access = access[:-1]
            return tuple(sum(p[k] for p in access)/len(access) for k in (0, 1))
    return goal


def belief_position(belief, target, *, own_ids=(), own_pose=None):
    """Locate a person from delivered carrier/arrival reports before old poses."""
    if target in own_ids:
        return _point(own_pose)
    if isinstance(belief, BeliefStore):
        # A position query needs only these local entries, not a grouped copy of
        # every entity's growing evidence/task fields. Read live entries each time.
        def fields_for(eid):
            return {key: entry for key in (
                "entity.pose", "human.pose", "belief.estimated_pose", "entity.geometry",
                "human.handover_status", "human.destination_id", "human.carrier_id")
                if (entry := belief.get(str(eid), key)) is not None}
    else:
        entities = _belief_entities(belief)
        def fields_for(eid):
            return entities.get(str(eid), {})
    fields = fields_for(target)
    pose = _position(fields)
    pose_time = max((entry.get("t", 0) if isinstance(entry, dict) else 0 for key, entry in fields.items()
                     if key in {"entity.pose", "human.pose", "belief.estimated_pose"}
                     and _point(_value(entry)) is not None), default=-1)
    handover = _field(fields, "human.handover_status")
    destination = _field(fields, "human.destination_id")
    if destination and handover in {"arrived", "acknowledged"}:
        arrived_time = max(fields.get(key, {}).get("t", 0) for key in
                           ("human.handover_status", "human.destination_id"))
        point = _position(fields_for(destination))
        if point is not None and arrived_time >= pose_time:
            return point
    carrier = _field(fields, "human.carrier_id")
    if carrier and handover not in {"arrived", "acknowledged"}:
        if carrier in own_ids:
            return _point(own_pose)
        # Carrier pose ties historically follow entry insertion order, unlike
        # _position's fixed key order. Preserve it (and all report fields) here.
        carrier_fields = _belief_entities(belief).get(str(carrier), {})
        report = carrier_fields.get("belief.team_estimates", {})
        telemetry = _value(report, {})
        point = _point(telemetry.get("pose", telemetry.get("entity.pose")))
        carrier_time = fields["human.carrier_id"].get("t", 0)
        positions = [(entry.get("t", 0), observed) for key, entry in carrier_fields.items()
                     if key in {"entity.pose", "human.pose", "belief.estimated_pose"}
                     and (observed := _point(_value(entry))) is not None]
        if point is not None:
            positions.append((report.get("t", 0), point))
        if positions:
            when, point = max(positions, key=lambda item: item[0])
            if max(carrier_time, when) >= pose_time:
                return point
    return pose


def _inside(point, polygon):
    if len(polygon) < 3:
        return False
    x, y = point
    result = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-7 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            result = not result
    return result


def _road_passable(eid, fields, entities, spec, obstacle_index=None, exclusions=None):
    mobility = spec.get("mobility")
    line = _coords(_field(fields, 'entity.geometry', {}))
    if exclusions is None:
        exclusions = _known_exclusions(entities)
    for polygon in exclusions:
        if any(_footprint_intersects(a,b,polygon) for a,b in zip(line,line[1:])):
            return False, '已知警戒区阻断通路'
    allowed = _field(fields, 'attr.allowed_mobilities')
    if allowed is not None and mobility not in allowed:
        return False, '源道路不支持该机动方式'
    if _field(fields, 'attr.professional_only') and not spec.get('professional_access', False):
        return False, '源路段需要专业接近能力'
    nominal_width = _number(_field(fields, "road.width_m"), inf)
    width = _number(_field(fields, "road.clear_width_m"), nominal_width)
    if width < spec.get("width_m", 0):
        return False, "净宽不足"
    if _number(_field(fields, "road.water_depth_m"), 0) > spec.get("water_tolerance_m", 0):
        return False, "涉水深度超限"
    step = _number(_field(fields, "road.step_height_m"), 0)
    if step > spec.get("step_height_m", 0):
        return False, "台阶超限"
    state = _field(fields, "road.entry_state")
    if state in {"unsafe", "closed"}:
        return False, "已知封闭或危险"
    obstacle_ids = list(_field(fields, "road.obstacle_ids", []) or [])
    blocking = (obstacle_index.get(eid, ()) if obstacle_index is not None else
                (oid for oid, obstacle in entities.items()
                 if eid in (_field(obstacle, "obstacle.blocks_ids", []) or [])))
    for oid in blocking:
        if oid not in obstacle_ids:
            obstacle_ids.append(oid)
    rubble = False
    active_obstacle = False
    for oid in obstacle_ids:
        obstacle = entities.get(oid, {})
        obs_state = _field(obstacle, "obstacle.state", "present")
        if obs_state in {"cleared", "moved"}:
            continue
        active_obstacle = True
        obs_step = _number(_field(obstacle, "obstacle.height_m"), step)
        if obs_state == "partly_cleared" or _field(obstacle, "obstacle.kind") in {"rubble", "debris", "瓦砾"}:
            rubble = True
            if mobility not in {"legged", "foot"} or obs_step > spec.get("step_height_m", 0):
                return False, "底盘不能越过已知瓦砾"
        else:
            return False, "已知障碍尚未清除"
    if state == "blocked" and not (mobility in {"legged", "foot"} and active_obstacle and rubble):
        return False, "已知道路受阻"
    return True, "已知尺寸允许通行；未知障碍待执行发现"


def _projection(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denominator = dx * dx + dy * dy
    u = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denominator)) if denominator else 0.0
    return (a[0] + u * dx, a[1] + u * dy), u


def _crossing(a, b, c, d):
    ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
    determinant = ux * vy - uy * vx
    if abs(determinant) < 1e-9:
        return None
    wx, wy = c[0] - a[0], c[1] - a[1]
    t, u = (wx * vy - wy * vx) / determinant, (wx * uy - wy * ux) / determinant
    if -1e-8 <= t <= 1 + 1e-8 and -1e-8 <= u <= 1 + 1e-8:
        return (a[0] + t * ux, a[1] + t * uy)
    return None


def _segment_points(start, goal, polygons):
    cuts = {0.0, 1.0}
    for polygon in polygons:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            point = _crossing(start, goal, a, b)
            if point is not None:
                cuts.add(_projection(point, start, goal)[1])
    cuts = sorted(cuts)
    return [start, goal] + [(start[0] + (goal[0] - start[0]) * (a + b) / 2,
                           start[1] + (goal[1] - start[1]) * (a + b) / 2)
                          for a, b in zip(cuts, cuts[1:])]


def obstacle_work_roads(entities, target, spec):
    """Split only an obstacle's work roads into open sections and blocked gaps.

    Fields may be local belief entries or physical fields supplied by execution.
    No road state is changed. D01's obstacle-derived width/step apply at the
    gaps; nominal width, base steps, water and closure still constrain approach.
    """
    obstacle = entities.get(str(target), {})
    if (spec.get("mobility") not in {"legged", "wheeled", "foot"}
            or not any(key.startswith("obstacle.") for key in obstacle)
            or _field(obstacle, "obstacle.state") in {"cleared", "moved"}):
        return {}
    roads = set(_field(obstacle, "obstacle.blocks_ids", []) or [])
    roads.update(eid for eid, fields in entities.items()
                 if target in (_field(fields, "road.obstacle_ids", []) or []))
    result = {}
    for road in sorted(roads):
        fields = entities.get(road, {})
        centerline = _field(fields, "road.centerline")
        points = _coords(centerline if isinstance(centerline, dict) else {"coords": centerline})
        if not points:
            points = _coords(_field(fields, "entity.geometry", {}))
        if (len(points) < 2 or _field(fields, "road.entry_state") in {"closed", "unsafe"}
                or _number(_field(fields, "road.width_m"), inf) < spec.get("width_m", 0)
                or _number(_field(fields, "road.water_depth_m"), 0) > spec.get("water_tolerance_m", 0)
                or _number(_field(fields, "attr.base_step_height_m"), 0) > spec.get("step_height_m", 0)):
            continue
        line, length = [], 0.0
        for a, b in zip(points, points[1:]):
            span = hypot(b[0] - a[0], b[1] - a[1])
            if span > 1e-7:
                line.append((a, b, length, span))
                length += span
        if not line:
            continue
        ids = set(_field(fields, "road.obstacle_ids", []) or []) | {target}
        ids.update(oid for oid, entries in entities.items()
                   if road in (_field(entries, "obstacle.blocks_ids", []) or []))
        gaps, work_at = [], None
        for oid in ids:
            entries = entities.get(oid, {})
            if _field(entries, "obstacle.state") in {"cleared", "moved"}:
                continue
            position = _position(entries)
            if position is None:
                break  # An unlocated known blocker cannot define an open section.
            _, along = min((hypot(q[0] - position[0], q[1] - position[1]), offset + u * span)
                           for a, b, offset, span in line
                           for q, u in [_projection(position, a, b)])
            gaps.append((along - OBSTACLE_WORK_STANDOFF_M, along + OBSTACLE_WORK_STANDOFF_M))
            if oid == target:
                work_at = along
        else:
            if work_at is None:
                continue
            segments, goals = [], []
            for a, b, offset, span in line:
                cuts = sorted({offset, offset + span} | {
                    boundary for gap in gaps for boundary in gap if offset < boundary < offset + span})
                def point(at):
                    return tuple(a[i] + (b[i] - a[i]) * (at - offset) / span for i in (0, 1))
                for left, right in zip(cuts, cuts[1:]):
                    allowed = not any(lo < (left + right) / 2 < hi for lo, hi in gaps)
                    segments.append((point(left), point(right), allowed))
                for at in (work_at - OBSTACLE_WORK_STANDOFF_M, work_at + OBSTACLE_WORK_STANDOFF_M):
                    if offset <= at <= offset + span:
                        goals.append(point(at))
            result[road] = {"segments": segments, "goals": goals}
    return result


def _cached_obstacle_work_roads(entities, target, spec, route_cache):
    """Share work sections only within the caller's unchanged belief snapshot."""
    if route_cache is None:
        return obstacle_work_roads(entities, target, spec)
    key = ("work_roads", id(entities), _exact_key(target),
           _exact_key(spec.get("mobility")), _exact_key(spec.get("width_m", 0)),
           _exact_key(spec.get("water_tolerance_m", 0)), _exact_key(spec.get("step_height_m", 0)))
    cached = route_cache.get(key)
    if cached is None:
        cached = (entities, obstacle_work_roads(entities, target, spec))
        route_cache[key] = cached
    return cached[1]


def _known_exclusions(entities):
    polygons = []
    for fields in entities.values():
        zone = _field(fields, 'hazard.exclusion_zone')
        if zone is True:
            zone = _field(fields, 'hazard.footprint')
        polygon = _coords(zone) if isinstance(zone, dict) else zone if isinstance(zone, list) else []
        if polygon:
            polygons.append(_prepare_polygon(polygon))
    return polygons


def _road_segments(entities, spec, work_roads=None):
    """Evaluate current road state, reusing identical ordered crossing topology."""
    segments, blocked, obstacle_index = [], [], {}
    exclusions = _known_exclusions(entities)
    for oid, fields in entities.items():
        for road_id in _field(fields, "obstacle.blocks_ids", []) or ():
            obstacle_index.setdefault(road_id, []).append(oid)
    for eid, fields in entities.items():
        if _field(fields, "entity.kind") != "road":
            continue
        points = _coords(_field(fields, "entity.geometry", {}))
        centerline = _field(fields, "road.centerline")
        if isinstance(centerline, (list, tuple, dict)):
            points = _coords(centerline if isinstance(centerline, dict) else {"coords": centerline}) or points
        allowed, _ = _road_passable(eid, fields, entities, spec, obstacle_index, exclusions)
        mobilities = _field(fields, 'attr.allowed_mobilities')
        attachable = ((mobilities is None or spec['mobility'] in mobilities)
                      and (not _field(fields,'attr.professional_only') or spec.get('professional_access',False)))
        if not allowed:
            blocked.append(eid)
        sections = (work_roads[eid]["segments"] if work_roads and eid in work_roads
                    else [(a, b, allowed) for a, b in zip(points, points[1:])])
        for a, b, admitted in sections:
            segments.append({"a": a, "b": b, "road_id": eid, "allowed": admitted, "attachable":attachable,"cuts": [a, b]})
    # Key the actual inputs to the crossing loop, after all current road,
    # obstacle, exclusion, embodiment and work-target rules have been applied.
    # This avoids snapshot metadata in the key without omitting any dependency.
    topology_key = _exact_key((tuple((s["a"], s["b"], s["road_id"], s["allowed"], s["attachable"])
                                    for s in segments), blocked))
    cached = _TOPOLOGY_CACHE.get(topology_key)
    if cached is not None:
        _TOPOLOGY_CACHE.move_to_end(topology_key)
        return cached
    # Public crossings/T-junctions only, never the truth graph.  This quadratic
    # geometry step is shared by all targets/candidates with matching geometry.
    for index, left in enumerate(segments):
        a, b = left["a"], left["b"]
        for right in segments[index + 1:]:
            c, d = right["a"], right["b"]
            if max(a[0], b[0]) + 1e-6 < min(c[0], d[0]) or max(c[0], d[0]) + 1e-6 < min(a[0], b[0]):
                continue
            if max(a[1], b[1]) + 1e-6 < min(c[1], d[1]) or max(c[1], d[1]) + 1e-6 < min(a[1], b[1]):
                continue
            point = _crossing(a, b, c, d)
            if point is not None:
                left["cuts"].append(point)
                right["cuts"].append(point)
            else:
                for point in (a, b, c, d):
                    on_left, _ = _projection(point, a, b)
                    on_right, _ = _projection(point, c, d)
                    if hypot(on_left[0] - point[0], on_left[1] - point[1]) < 1e-6 and hypot(on_right[0] - point[0], on_right[1] - point[1]) < 1e-6:
                        left["cuts"].append(point)
                        right["cuts"].append(point)
    cached = (segments, blocked, object())
    if len(_TOPOLOGY_CACHE) >= _TOPOLOGY_CACHE_LIMIT:
        _TOPOLOGY_CACHE.popitem(last=False)
    _TOPOLOGY_CACHE[topology_key] = cached
    return cached


def _belief_segment_graph(segment, raw_cuts, clear):
    # Rebuild the original set from the original insertion sequence, not from
    # sorted base cuts: equal projection keys must retain set-iteration ties.
    cuts = sorted({tuple(round(v, 6) for v in p) for p in raw_cuts},
                  key=lambda p: _projection(p, segment["a"], segment["b"])[1])
    nodes = {point: point for point in cuts}
    adjacency, blockers = {}, set()
    if segment["allowed"]:
        for a, b in zip(cuts, cuts[1:]):
            if not clear(a, b, blockers):
                continue
            length = hypot(a[0] - b[0], a[1] - b[1])
            adjacency.setdefault(a, []).append((b, length, segment["road_id"]))
            adjacency.setdefault(b, []).append((a, length, segment["road_id"]))
    return nodes, adjacency, blockers


def _belief_base_graph(segments, clear):
    nodes, adjacency, owners = {}, {}, {}
    parts = []
    for index, segment in enumerate(segments):
        part = _belief_segment_graph(segment, segment["cuts"], clear)
        parts.append(part)
        local_nodes, local_adjacency, _ = part
        nodes.update(local_nodes)
        for point in local_nodes:
            owners.setdefault(point, []).append(index)
        for point, edges in local_adjacency.items():
            adjacency.setdefault(point, []).extend(edges)
    return nodes, adjacency, owners, parts


def _belief_splice_graph(base, segments, endpoint_cuts, clear):
    base_nodes, base_adjacency, owners, parts = base
    changed = {index: _belief_segment_graph(segments[index], cuts, clear)
               for index, cuts in endpoint_cuts.items()}
    nodes, adjacency = base_nodes.copy(), base_adjacency.copy()
    affected = set()
    for index, (local_nodes, _, _) in changed.items():
        affected.update(parts[index][0])
        affected.update(local_nodes)
    for point in affected:
        # A shared junction's neighbors are ordered by segment, then cut, not
        # by when an endpoint splice was applied. Keep even duplicate edges.
        indices = sorted(set(owners.get(point, ())) |
                         {index for index, part in changed.items() if point in part[0]})
        edges = []
        for index in indices:
            local_nodes, local_adjacency, _ = changed.get(index, parts[index])
            if point in local_nodes:
                # nodes.update historically keeps the LAST segment's value,
                # which matters even for numerically equal signed-zero tuples.
                nodes[point] = local_nodes[point]
                edges.extend(local_adjacency.get(point, ()))
        adjacency[point] = edges
    return nodes, adjacency, changed


def _source_frontage_projections(position, projections, segments, entities, frontages, approach):
    """Use declared pad frontages, never proximity to a different road component.

    Source road children use source_network's exact ``source_id#piece`` format.
    Off-road connectors keep the existing collision checks; they are endpoint
    approaches, never edges between the candidate road components.
    """
    incident = [(index, point) for index, point in projections
                if hypot(point[0] - position[0], point[1] - position[1]) < 1e-6]
    if incident:
        return [(index, point) for index, point in incident if segments[index]['allowed']]
    sources = {source for rid, source in frontages.items()
               if _field(entities.get(rid, {}), 'entity.kind') == 'open_area'
               and _inside(position, _coords(_field(entities[rid], 'entity.geometry', {})))}
    candidates = []
    for index, point in projections:
        road = segments[index]['road_id']
        parent, separator, piece = road.rpartition('#')
        source = parent if separator and piece.isdecimal() else road
        if segments[index]['allowed'] and (approach or source in sources):
            candidates.append((index, point))
    if sources or approach:
        return candidates
    # An off-pad endpoint without a declared frontage retains its old rule.
    nearest = min(projections, key=lambda row: hypot(row[1][0] - position[0], row[1][1] - position[1]))
    return [nearest] if segments[nearest[0]]['allowed'] else []


def _reachable_attachment_path(result, choices, nodes, adjacency):
    """Nearest clear attachments per reachable component, without graph bridges.

    Do not minimize road travel by moving the endpoints' off-road connectors far
    along a frontage. Within a component each endpoint keeps its nearest landing.
    """
    components = {}
    for node in adjacency:
        if node in components:
            continue
        component = len(components)
        pending = [node]
        components[node] = component
        while pending:
            current = pending.pop()
            for neighbor, _, _ in adjacency.get(current, ()):
                if neighbor not in components:
                    components[neighbor] = component
                    pending.append(neighbor)
    nearest = []
    for endpoint in choices:
        local = {}
        for node, path in endpoint:
            if node not in components:
                continue
            cost = sum(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
            component = components[node]
            if component not in local or cost < local[component][0]:
                local[component] = (cost, node, path)
        nearest.append(local)
    shared = nearest[0].keys() & nearest[1].keys()
    choices = [[(node, path) for component, (_, node, path) in local.items() if component in shared]
               for local in nearest]
    distances, previous, origins = {}, {}, {}
    queue, serial = [], 0
    for node, path in choices[0]:
        cost = sum(hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
        if cost < distances.get(node, inf):
            distances[node], origins[node] = cost, (node, path)
            serial += 1
            heapq.heappush(queue, (cost, serial, node))
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, length, road in adjacency.get(node, ()):
            candidate = distance + length
            if candidate < distances.get(neighbor, inf):
                distances[neighbor], previous[neighbor] = candidate, (node, road)
                origins[neighbor] = origins[node]
                serial += 1
                heapq.heappush(queue, (candidate, serial, neighbor))
    reachable = [(distances[node] + sum(hypot(a[0] - b[0], a[1] - b[1])
                                      for a, b in zip(path, path[1:])), index, node, path)
                 for index, (node, path) in enumerate(choices[1]) if node in distances]
    if not reachable:
        return None
    length, _, destination, goal_path = min(reachable, key=lambda row: (row[0], row[1]))
    source, start_path = origins[destination]
    route, roads = [destination], []
    while route[-1] != source:
        node, road = previous[route[-1]]
        route.append(node)
        roads.append(road)
    route.reverse()
    return {**result, 'waypoints': start_path[:-1] + [list(nodes[n]) for n in route]
            + list(reversed(goal_path))[1:], 'length_m': length,
            'road_ids': [None] * (len(start_path) - 1) + list(reversed(roads))
            + [None] * (len(goal_path) - 1)}


def belief_path(belief, start, goal, spec, *, route_cache=None, work_target=None):
    """Reuse complete queries only in the caller-owned belief snapshot cache."""
    if route_cache is None or _point(start) is None or _point(goal) is None:
        return _belief_path(belief, start, goal, spec, route_cache=route_cache, work_target=work_target)
    entities = _belief_entities(belief) or {}
    key = ("path", id(entities), _exact_key(start), _exact_key(goal),
           _exact_key(spec), _exact_key(work_target))
    cached = route_cache.get(key)
    if cached is None:
        result = _belief_path(entities, start, goal, spec, route_cache=route_cache, work_target=work_target)
        # Keep the snapshot alive as well as its result: an id must not be reused
        # while the caller's cache still contains entries for the old snapshot.
        cached = (entities, deepcopy(result))
        route_cache[key] = cached
    return deepcopy(cached[1])


def _belief_path(belief, start, goal, spec, *, route_cache=None, work_target=None):
    """A small A066-style route model using only public geometry/local entries.

    Unknown blockage is *not* imported from truth.  Ground movement uses a road
    graph reconstructed from public polylines and known node ids.  An empty map
    is insufficient evidence for access, not a clairvoyantly successful path.
    ``route_cache`` is optional and must be discarded after that belief snapshot;
    topology, contacts and permission-keyed base graphs are shared; endpoint cuts stay local.
    ``work_target`` permits only the open sections leading to an obstacle's
    stand-off points; the blocked gap remains absent from the graph.
    """
    start, goal = _point(start), _point(goal)
    if start is None or goal is None:
        return {"ok": False, "waypoints": [], "length_m": 0.0, "blocked_by": [], "reason_zh": "尚无已知位置"}
    entities = _belief_entities(belief) or {}
    # Only the opt-in public entrance model adds these connectors. It carries
    # geometry, never evaluator population roles or casualty truth.
    # Each preprocessing entry retains its snapshot against id reuse; even an
    # empty ordered result is a hit. Only caller-owned invocation caches hold it.
    buildings_key = ('entrance_buildings', id(entities), spec.get('mobility'))
    cached = route_cache.get(buildings_key) if route_cache is not None else None
    if cached is None:
        buildings = []
        for eid, fields in entities.items():
            if _field(fields, 'attr.interior_access_via_entrance') is not True:
                continue
            vehicle = _field(fields, 'attr.vehicle_approach') if spec.get('mobility') == 'wheeled' else None
            buildings.append((eid, _coords(_field(fields, 'attr.navigation_footprint', _field(fields, 'entity.geometry', {}))),
                vehicle['entrance_ids'] if vehicle else _field(fields, 'building.entrance_ids', []),
                vehicle['stand_xy'] if vehicle else _field(fields, 'attr.entrance_xy'),
                vehicle['heading_deg'] if vehicle else _field(fields, 'attr.entrance_heading_deg'),
                vehicle['start_xy'] if vehicle else _field(fields, 'attr.approach_start_xy'),
                vehicle['waypoints_xy'] if vehicle else _field(fields, 'attr.approach_waypoints_xy', [])))
        buildings = [(row[0], _prepare_polygon(row[1]), *row[2:]) for row in buildings]
        cached = (entities, buildings)
        if route_cache is not None:
            route_cache[buildings_key] = cached
    buildings = cached[1]
    if buildings and spec.get('mobility') == 'wheeled':
        roads_key = ('wheeled_road_lines', id(entities))
        cached = route_cache.get(roads_key) if route_cache is not None else None
        if cached is None:
            roads = {eid: _coords(_field(fs, 'road.centerline', {})) for eid, fs in entities.items()
                     if _field(fs, 'entity.kind') == 'road'}
            cached = (entities, roads)
            if route_cache is not None:
                route_cache[roads_key] = cached
        goal = wheeled_navigation_goal(goal, buildings, cached[1], spec.get('length_m'))
        if goal is None:
            return {'ok': False, 'waypoints': [list(start)], 'length_m': 0., 'blocked_by': [],
                    'reason_zh': '公开入口未提供合法车辆停靠点'}
    if spec.get('mobility') == 'water':
        goal = water_access_goal(goal, entities)
    direct = hypot(goal[0] - start[0], goal[1] - start[1])
    result = {"ok": True, "waypoints": [list(start), list(goal)], "length_m": direct,
              "blocked_by": [], "reason_zh": "基于已知地图可达；未知障碍待执行发现"}
    if buildings:
        result['navigation_goal'] = list(goal)
    if direct < 1e-7:
        return result
    mobility = spec.get("mobility", "none")
    if mobility == "none":
        return {**result, "ok": False, "reason_zh": "固定节点不能物理移动"}
    if mobility == "air":
        polygons_key = ('air_exclusion_polygons', id(entities))
        cached = route_cache.get(polygons_key) if route_cache is not None else None
        if cached is None:
            polygons = []
            for eid, fields in entities.items():
                exclusion = _field(fields, "hazard.exclusion_zone")
                if not exclusion:
                    continue
                polygon = _coords(exclusion if isinstance(exclusion, dict)
                                  else {"coords": exclusion} if isinstance(exclusion, (list, tuple))
                                  else _field(fields, "entity.geometry", {}))
                polygons.append((eid, polygon))
            cached = (entities, polygons)
            if route_cache is not None:
                route_cache[polygons_key] = cached
        for eid, polygon in cached[1]:
            if any(_inside(p, polygon) for p in _segment_points(start, goal, [polygon])):
                return {**result, "ok": False, "blocked_by": [eid], "reason_zh": "已知航线穿越禁入区"}
        return result
    if mobility == "water":
        waters_key = ('water_polygons', id(entities))
        cached = route_cache.get(waters_key) if route_cache is not None else None
        if cached is None:
            waters = [_coords(_field(fields, "entity.geometry", {}))
                      for fields in entities.values() if _field(fields, "entity.kind") == "water"]
            cached = (entities, waters)
            if route_cache is not None:
                route_cache[waters_key] = cached
        waters = cached[1]
        ok = bool(waters) and all(any(_inside(p, polygon) for polygon in waters)
                                 for p in _segment_points(start, goal, waters))
        return {**result, "ok": ok, "reason_zh": "已知水域内航行" if ok else "不能确认全程位于已知水域"}

    regions_key = ('isolated_regions', id(entities))
    cached = route_cache.get(regions_key) if route_cache is not None else None
    if cached is None:
        regions = [(eid, _coords(_field(fields, 'entity.geometry', {}))) for eid, fields in entities.items()
                   if _field(fields, 'attr.ground_access_isolated') is True]
        cached = (entities, regions)
        if route_cache is not None:
            route_cache[regions_key] = cached
    isolated = isolated_ground_route(start, goal, cached[1])
    if isolated is not None:
        return isolated
    clearance_key = ('clearance', id(entities), mobility)
    clearance = route_cache.get(clearance_key) if route_cache is not None else None
    if clearance is None:
        barriers = [(eid, _coords(_field(fields, 'entity.geometry', {}))) for eid, fields in entities.items()
                    if _field(fields, 'attr.blocks_ground_navigation') is True]
        known_barriers = {eid for eid, _ in barriers}
        barriers += [(row[0], row[1]) for row in buildings if row[0] not in known_barriers]
        for eid, fields in entities.items():
            zone = _field(fields, 'hazard.exclusion_zone')
            if zone is True:
                zone = _field(fields, 'hazard.footprint')
            if isinstance(zone, (dict,list)):
                barriers.append((eid, _coords(zone) if isinstance(zone,dict) else zone))
        barriers = [(eid, _prepare_polygon(poly)) for eid, poly in barriers]
        # Cache only raw geometric contacts across rounds/agents. Entrance
        # permissions remain query-local below, even when these barriers match.
        content_key = _exact_key(tuple((eid, tuple(poly)) for eid, poly in barriers))
        shared = _CLEARANCE_CACHE.get(content_key)
        if shared is None:
            shared = (barriers, OrderedDict(), object())
            if len(_CLEARANCE_CACHE) >= _CLEARANCE_CACHE_LIMIT:
                _CLEARANCE_CACHE.popitem(last=False)
            _CLEARANCE_CACHE[content_key] = shared
        else:
            _CLEARANCE_CACHE.move_to_end(content_key)
        # Only the caller's round-local cache retains the snapshot against id
        # reuse; the process cache holds detached prepared geometry and ids.
        clearance = (entities, *shared)
        if route_cache is not None:
            route_cache[clearance_key] = clearance
    _, barriers, contacts, barrier_token = clearance
    interior = interior_approach_buildings(buildings, (start, goal),
        {row[0] for row in buildings} if mobility in ('foot', 'legged') else set())
    interior_by_id = {}
    for row in interior:
        interior_by_id.setdefault(row[0], []).append(row)

    def clear(a, b, blocked_by=None):
        key = (_exact_key(a), _exact_key(b))
        hits = contacts.get(key)
        if hits is None:
            hits = tuple(eid for eid, poly in _footprint_candidates(a, b, barriers)
                         if _footprint_intersects(a, b, poly))
            if len(contacts) >= _CONTACT_CACHE_LIMIT:
                contacts.popitem(last=False)
            contacts[key] = hits
        else:
            contacts.move_to_end(key)
        # Cache raw contacts, not permission: a different interior goal can
        # admit a different entrance leg. Preserve the first blocker in order.
        for eid in hits:
            if not building_approach_leg(a, b, interior_by_id.get(eid, ())):
                if blocked_by is not None:
                    blocked_by.add(eid)
                return False
        return True

    key = (id(entities), spec.get("mobility"), spec.get("width_m", 0),
           spec.get("water_tolerance_m", 0), spec.get("step_height_m", 0),
           spec.get("professional_access", False), work_target)
    cached = route_cache.get(key) if route_cache is not None else None
    if cached is None:
        work_roads = (_cached_obstacle_work_roads(entities, work_target, spec, route_cache)
                      if work_target is not None else None)
        cached = _road_segments(entities, spec, work_roads)
        if route_cache is not None:
            route_cache[key] = cached
    segments, blocked, topology_token = cached
    if not segments:
        return {**result, "ok": False, "reason_zh": "尚无已知道路拓扑"}
    # Full-query keys above contain the exact spec, including source frontages;
    # region geometry belongs to that caller-owned belief snapshot. Base graph,
    # topology and raw-clearance keys deliberately exclude query-local cuts.
    frontages = spec.get('ground_route_frontages', {})
    endpoint_cuts = {}
    endpoints, endpoint_paths, choices = [], [], []
    for position in (start, goal):
        approach = building_approach_attachment(position, interior + buildings)
        attachment = approach[1] if approach else position
        attachments = [(index, s) for index, s in enumerate(segments) if s['attachable'] and
                       (not approach or s['road_id'] in approach[0])]
        if not attachments:
            return {**result,'ok':False,'reason_zh':'没有支持该机动方式的源路段'}
        projections = [(index, _projection(attachment, s["a"], s["b"])[0]) for index, s in attachments]
        projections = [(index, point) for index, point in projections if clear(attachment, point)]
        if not projections:
            return {**result, "ok": False, "blocked_by": [eid for eid, _ in barriers],
                    "reason_zh": "接近线穿越实体设施占地"}
        if frontages:
            projections = _source_frontage_projections(attachment, projections, segments, entities, frontages, approach)
            local = []
            for index, projected in projections:
                node = tuple(round(v, 6) for v in projected)
                path = [*(approach[2] if approach else [list(position)]), list(node)]
                if any(not clear(a, b) for a, b in zip(path, path[1:])):
                    continue
                endpoint_cuts.setdefault(index, list(segments[index]['cuts'])).append(projected)
                local.append((node, path))
            if not local:
                return {**result, 'ok': False, 'blocked_by': sorted(set(blocked)),
                        'reason_zh': '源临路绑定没有可通行且接近线无碰撞的路段'}
            choices.append(local)
        else:
            index, projected = min(projections, key=lambda item: hypot(item[1][0] - attachment[0],
                                                                      item[1][1] - attachment[1]))
            if index not in endpoint_cuts:
                endpoint_cuts[index] = list(segments[index]["cuts"])
            endpoint_cuts[index].append(projected)
            endpoints.append(tuple(round(v, 6) for v in projected))
            endpoint_paths.append([*(approach[2] if approach else [list(position)]), list(projected)])
    graph_key = (topology_token, barrier_token, _exact_key(interior))
    base_graph = _BELIEF_GRAPH_CACHE.get(graph_key)
    if base_graph is None:
        base_graph = _belief_base_graph(segments, clear)
        if len(_BELIEF_GRAPH_CACHE) >= _BELIEF_GRAPH_CACHE_LIMIT:
            _BELIEF_GRAPH_CACHE.popitem(last=False)
        _BELIEF_GRAPH_CACHE[graph_key] = base_graph
    else:
        _BELIEF_GRAPH_CACHE.move_to_end(graph_key)
    nodes, adjacency, changed = _belief_splice_graph(base_graph, segments, endpoint_cuts, clear)
    if frontages:
        path = _reachable_attachment_path(result, choices, nodes, adjacency)
        if path is not None:
            return path
        geometry_blockers = set()
        for index, part in enumerate(base_graph[3]):
            geometry_blockers.update(changed.get(index, part)[2])
        return {**result, 'ok': False, 'blocked_by': sorted(set(blocked) | geometry_blockers),
                'reason_zh': '源临路候选之间没有已知可通行路径'}
    source, destination = endpoints
    blocked_connector = (any(not clear(a, b) for path in endpoint_paths for a, b in zip(path, path[1:]))
                         if buildings else not clear(start, source) or not clear(destination, goal))
    if blocked_connector:
        return {**result, 'ok': False, 'blocked_by': [eid for eid, _ in barriers], 'reason_zh': '接近线穿越实体设施占地'}
    distances, previous = {source: 0.0}, {}
    queue, serial = [(0.0, 0, source)], 0
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == destination:
            break
        for neighbor, length, road in adjacency.get(node, ()):
            candidate = distance + length
            if candidate < distances.get(neighbor, inf):
                distances[neighbor], previous[neighbor] = candidate, (node, road)
                serial += 1
                heapq.heappush(queue, (candidate, serial, neighbor))
    if destination not in distances:
        geometry_blockers = set()
        for index, part in enumerate(base_graph[3]):
            geometry_blockers.update(changed.get(index, part)[2])
        blockers = sorted(set(blocked) | geometry_blockers)
        return {**result, "ok": False, "blocked_by": blockers,
                "reason_zh": "已知路网无可通行路径" + ("；受阻实体/路段：" + ", ".join(blockers) if blockers else "")}
    route, roads = [destination], []
    while route[-1] != source:
        node, road = previous[route[-1]]
        route.append(node)
        roads.append(road)
    route.reverse()
    if not buildings:
        # Preserve the legacy route coordinates, rounding and cost arithmetic.
        connectors = hypot(nodes[source][0] - start[0], nodes[source][1] - start[1])
        connectors += hypot(nodes[destination][0] - goal[0], nodes[destination][1] - goal[1])
        return {**result, "waypoints": [list(start)] + [list(nodes[n]) for n in route] + [list(goal)],
                "length_m": distances[destination] + connectors,
                "road_ids": [None] + list(reversed(roads)) + [None]}
    connectors = sum(hypot(a[0]-b[0], a[1]-b[1]) for path in endpoint_paths for a, b in zip(path, path[1:]))
    return {**result, "waypoints": endpoint_paths[0][:-1] + [list(nodes[n]) for n in route] + list(reversed(endpoint_paths[1]))[1:],
            "length_m": distances[destination] + connectors,
            "road_ids": [None]*(len(endpoint_paths[0])-1) + list(reversed(roads)) + [None]*(len(endpoint_paths[1])-1)}


def obstacle_approach_path(belief, start, target, spec, *, route_cache=None):
    """Choose the reachable work side by the existing local-map route cost."""
    entities = _belief_entities(belief)
    route_cache = {} if route_cache is None else route_cache
    work_roads = _cached_obstacle_work_roads(entities, target, spec, route_cache)
    paths = [belief_path(entities, start, goal, spec, route_cache=route_cache, work_target=target)
             for road in work_roads.values() for goal in road["goals"]]
    reachable = [path for path in paths if path["ok"]]
    if reachable:
        return {**min(reachable, key=lambda path: path["length_m"]), "work_target": target}
    if paths:
        return min(paths, key=lambda path: path["length_m"])
    return belief_path(entities, start, belief_position(entities, target), spec, route_cache=route_cache)


def hazard_approach_path(belief, start, target, spec, reach, *, route_cache=None):
    """Choose a reachable outside stand from delivered footprint and public facilities."""
    entities = _belief_entities(belief)
    if start is None:
        return belief_path(entities, start, belief_position(entities, target), spec, route_cache=route_cache)
    fields = entities.get(target, {})
    polygon = _coords(_field(fields, 'hazard.footprint', _field(fields, 'entity.geometry', {})))
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        return belief_path(entities, start, belief_position(entities, target), spec, route_cache=route_cache)
    barriers = [_coords(_field(fs, 'entity.geometry', {})) for fs in entities.values()
                if _field(fs, 'attr.blocks_ground_navigation') is True]
    focus = belief_position(entities, target)
    envelope = next((poly for poly in barriers if _inside(focus, poly)), polygon)
    radius = hypot(spec.get('length_m', 0), spec.get('width_m', 0))/2+1.
    def in_reach(p):
        distance = 0. if _inside(p, polygon) else min(hypot(p[0]-q[0], p[1]-q[1])
            for a, b in zip(polygon, polygon[1:]+polygon[:1]) for q, _ in [_projection(p, a, b)])
        return distance <= reach and not any(_inside(p, poly) for poly in [polygon, *barriers])
    if start is not None and in_reach(start):
        return {'ok': True, 'waypoints': [list(start)], 'road_ids': [], 'length_m': 0.,
                'blocked_by': [], 'reason_zh': '已处于源实体外实际作用范围内'}
    center = [sum(p[k] for p in envelope)/len(envelope) for k in (0, 1)]
    candidates = []
    for a, b in zip(envelope, envelope[1:]+envelope[:1]):
        span = hypot(b[0]-a[0], b[1]-a[1])
        if span < 1e-7:
            continue
        normal = [(b[1]-a[1])/span, -(b[0]-a[0])/span]
        middle = [(a[k]+b[k])/2 for k in (0, 1)]
        if sum(normal[k]*(middle[k]-center[k]) for k in (0, 1)) < 0:
            normal = [-v for v in normal]
        for face in (middle, _projection(start, a, b)[0]):
            stand = [face[k]+radius*normal[k] for k in (0, 1)]
            if in_reach(stand):
                candidates.append(belief_path(entities, start, stand, spec, route_cache=route_cache))
    reachable = [path for path in candidates if path['ok']]
    if reachable:
        return min(reachable, key=lambda path: path['length_m'])
    return {'ok': False, 'waypoints': [list(start)], 'road_ids': [], 'length_m': 0.,
            'blocked_by': [target], 'reason_zh': '没有实际可达且在装备范围内的实体外作业点'}


def _reported(row, *names, default=None):
    platform = row.get("platform") or {}
    for name in names:
        if name in row:
            return _value(row[name], default)
        if name in platform:
            return _value(platform[name], default)
    return default


def _fleet_rows(fleet_view):
    if isinstance(fleet_view, dict):
        return [{**row, "id": row.get("id", aid)}
                for aid, row in fleet_view.items() if isinstance(row, dict)]
    return [dict(row) for row in fleet_view or () if isinstance(row, dict) and "id" in row]


def _candidate_spec(row):
    kind = _reported(row, "kind", "platform.kind")
    spec = dict(EMBODIMENTS.get(kind, {}))
    if isinstance(row.get("spec"), dict):
        spec.update(row["spec"])
    caps = _reported(row, "capabilities", "platform.capability_ids")
    if caps is not None:
        spec["capabilities"] = list(caps)
    return spec


class Allocator:
    def allocate(self, coordinator_runtime, sim, proposals, fleet_view):
        raise NotImplementedError


class GreedyCapabilityAccessAllocator(Allocator):
    def __init__(self, catalog=None, reserve_fraction=0.15):
        self.catalog = catalog
        self.reserve_fraction = float(reserve_fraction)
        if not isfinite(self.reserve_fraction) or not 0 <= self.reserve_fraction <= 1:
            raise ValueError("预留比例须在0到1之间 / reserve fraction must be in [0, 1]")
        self._method_cache = {}

    def capability_filter(self, catalog, task, spec):
        cache_key = (id(catalog), task.get("type"), spec.get("mobility"),
                     tuple(sorted(spec.get("capabilities", ()))))
        cached = self._method_cache.get(cache_key)
        try:
            if cached is None:
                method = resolve_method(catalog, task["type"], spec)
                required = required_capabilities(catalog, task["type"], spec, method=method)
                if len(self._method_cache) >= 256:
                    self._method_cache.clear()
                self._method_cache[cache_key] = (method, required)
            else:
                method, required = cached
        except (KeyError, ValueError) as exc:
            return False, str(exc), None
        owned = set(spec.get("capabilities", ()))
        missing = set(required) - owned
        # The task-level intersection is the public record's requirement, but
        # each action still has its own guard.  Never knowingly select a method
        # that cannot pass those guards (including A059's C46/C31 interface).
        local_actions = _requester_actions(method)
        action_missing = {action: _caps(catalog, action) - owned for action in local_actions}
        failures = [f"{action}({','.join(sorted(caps))})" for action, caps in action_missing.items() if caps]
        if failures:
            return False, "目录动作不可执行：" + " ".join(failures), method
        if missing:
            return False, "缺少目录能力 " + " ".join(sorted(missing)), method
        if any(not _compatible(action, spec) for action in local_actions):
            return False, "参考方法与机动模式不兼容", method
        if any(action in _PHYSICAL for action in local_actions):
            movement = _TRAVEL_CAPABILITY.get(spec.get("mobility"))
            if movement is None or movement not in spec.get("capabilities", ()):
                return False, "本地物理步骤缺少到场机动能力", method
        return True, "具备候选方法所需目录能力；远端/专业步骤由接收方独立检查", method

    def access_filter(self, coordinator, task, row, spec, method, *, entities=None, route_cache=None):
        actions = _requester_actions(method)
        if not any(a in _MOVEMENT or a in _PHYSICAL for a in actions):
            return True, "本地信息或专业请求步骤无需请求者物理到场", 0.0
        start = _point(_reported(row, "pose", "entity.pose"))
        entities = coordinator.belief.entities(copy=False) if entities is None else entities
        route_cache = {} if route_cache is None else route_cache
        params = task.get("parameters", {}) or {}
        target = params.get("destination_id", params.get("destination", task.get("target")))
        if task.get("type") == "T089" and not any(a in _MOVEMENT for a in actions):
            # Q089 samples in place (A005/A006/A041/A036); it never requests a
            # flight to the hazard centre. Match A005's actual sensing radius,
            # using only the delivered platform pose and known target position.
            goal = belief_position(entities, str(task.get("target")))
            if start is None or goal is None:
                return False, "尚无已知原地监测位置或危险源位置", 0.0
            radius = max(1., min(_number(params.get("range_m"), 250.),
                                 _number(spec.get("sensor_range_m"), 250.)))
            if hypot(goal[0] - start[0], goal[1] - start[1]) > radius:
                return False, "危险源超出本机原地传感范围；Q089不包含接近动作", 0.0
            return True, "Q089在本机传感范围内原地采集，不进入危险源禁入区", 0.0
        pickup = None
        if task.get("type") in {"T030", "T039", "T040"}:
            item_fields = entities.get(str(task.get("target")), {})
            target = params.get("receiver_id", _field(item_fields, "item.receiver_id", target))
            if "A024" in actions:
                pickup = task.get("target")
        if task.get("type") in {"T034", "T035", "T036"}:
            human_fields = entities.get(str(task.get("target")), {})
            target = (params.get("destination_id") or params.get("destination")
                      or _field(human_fields, "human.destination_id"))
            if target is None:
                return False, "尚无已知人员接收点", 0.0
            if any(a in {"A020", "A023"} for a in actions):
                pickup = task.get("target")
        if task.get("type") in {"T043", "T044"}:
            target = params.get("station_id")
            if target is None and start is not None:
                stations_key = ('charge_station_positions', id(entities))
                cached = route_cache.get(stations_key)
                if cached is None:
                    stations = [eid for eid, fields in entities.items() if
                                _field(fields, "station.open_state", "open") != "closed" and
                                (_field(fields, "station.kind") in {"charging", "charge", "supply", "charging_station"}
                                 or _number(_field(fields, "station.power_W"), 0) > 0)]
                    positioned = [(eid, _position(entities[eid])) for eid in stations]
                    positioned = [(eid, point) for eid, point in positioned if point is not None]
                    cached = (entities, positioned)
                    route_cache[stations_key] = cached
                positioned = cached[1]
                if positioned:
                    target = min(positioned, key=lambda item: hypot(item[1][0] - start[0], item[1][1] - start[1]))[0]
            if target is None:
                return False, "尚无已知可用补能站位置", 0.0
        if task.get('type') == 'T094':
            # Missing starts must return before any station position is read.
            if start is None:
                return False, '尚无实际补给站位置', 0.
            stations_key = ('supply_station_positions', id(entities))
            cached = route_cache.get(stations_key)
            if cached is None:
                stations = [eid for eid, fields in entities.items()
                            if _field(fields, 'station.kind') in {'supply', 'logistics', 'mixed'}]
                # Unlike charging, supply selection historically does not skip
                # missing poses; retain those entries and its existing failure.
                cached = (entities, [(eid, _position(entities[eid])) for eid in stations])
                route_cache[stations_key] = cached
            positioned = cached[1]
            if not positioned:
                return False, '尚无实际补给站位置', 0.
            target = min(positioned, key=lambda item: hypot(item[1][0] - start[0], item[1][1] - start[1]))[0]
        distance = 0.0
        for destination in ([pickup, target] if pickup is not None and pickup != target else [target]):
            goal = _point(destination)
            target_id = destination.get("id", destination.get("entity_id")) if isinstance(destination, dict) else destination
            if goal is None and isinstance(target_id, str):
                goal = belief_position(entities, target_id)
            if destination is None or isinstance(target_id, str) and target_id in {row.get("id"), row.get("platform_eid")}:
                goal = start
            if task.get('type') in {'T093', 'T095'}:
                reach = spec.get('suppression_reach_m', 50.) if task['type']=='T093' else spec.get('isolation_reach_m', 80.)
                path = hazard_approach_path(entities, start, target_id, spec, reach, route_cache=route_cache)
            elif task.get("type") in {"T022", "T023", "T026"} and destination == task.get("target"):
                path = obstacle_approach_path(entities, start, target_id, spec, route_cache=route_cache)
            else:
                path = belief_path(entities, start, goal, spec, route_cache=route_cache)
            distance += path["length_m"]
            if not path["ok"]:
                return False, path["reason_zh"], distance
            start = path['waypoints'][-1] if spec.get('mobility') == 'water' else path.get('navigation_goal', goal)
        return True, "已知路线允许到场/交付；未知障碍由执行反馈确认", distance

    def payload_filter(self, task, row, spec, method, entities, identities=()):
        """Reject known/nominal over-capacity loads, never inspect hidden mass."""
        actions = _requester_actions(method)
        fields = entities.get(str(task.get("target")), {})
        if any(action in {"A020", "A021", "A022", "A023"} for action in actions):
            carrier = _field(fields, "human.carrier_id")
            if carrier is not None and str(carrier) in set(identities):
                # Already aboard this carrier: its mass is in payload_used_kg; only incremental load counts.
                return True, "人员已在本载具上，载荷不重复计入"
            mass = _number(_field(fields, "attr.mass_kg"), 75.0)
        elif "A024" in actions:
            mass = _number(_field(fields, "item.mass_kg"))
            if mass is None:
                return True, "物资质量未知，接收者须以本地装载检查确认"
            mass *= max(0.0, _number(_field(fields, "item.quantity"), 1.0))
        else:
            return True, "方法不要求本机承载人员或物资"
        used = max(0.0, _number(_reported(row, "payload_used_kg", "platform.payload_used_kg"), 0.0))
        maximum = max(0.0, _number(spec.get("payload_kg"), 0.0))
        ok = used + max(0.0, mass) <= maximum
        return ok, "已知/额定载荷估计允许" if ok else "已知或人员额定载荷超过剩余承载量"

    def battery_filter(self, task, row, spec, method, distance):
        if spec.get("mains_power"):
            return True, "固定节点使用市电", 0.0
        battery = _number(_reported(row, "battery_Wh", "battery", "battery_remaining_Wh", "platform.battery_remaining_Wh"))
        if battery is None:
            return False, "尚未收到有效电量报告", 0.0
        capacity = _number(_reported(row, "battery_max_Wh", "platform.battery_capacity_Wh"), spec.get("battery_Wh", 0))
        reserve = max(0.0, _number(_reported(row, "reserve_Wh", "platform.reserve_Wh"), capacity * self.reserve_fraction))
        # Authored W / (m/s) / 3600 matches the movement model's units;
        # the previous Wh/m constants could reject a perfectly viable flight.
        # This is a belief-only estimate, never an action plan or ledger charge.
        mobility = spec.get("mobility")
        speed = max(.1, _number(spec.get("speed_mps"), 1.0))
        power = max(0.0, _number(spec.get("move_power_W"), _MOVE_POWER_W.get(mobility, 20.0)))
        estimate = max(0.0, distance) / speed * power / 3600.0
        params = task.get("parameters", {}) or {}
        defaults = {"物理": (15., 120.), "专业服务": (90., 220.),
                    "抽象物理服务": (120., 70.), "传感器": (3., 8.),
                    "计算": (2., 12.), "计算工具": (3., 12.), "通信": (1., 2.)}
        details = method.get("details", [])
        for index, action in enumerate(method["resolved_steps"]):
            detail = details[index] if index < len(details) else {}
            if str(detail.get("actor", "")).startswith("远端") or action in _MOVEMENT or action == "A021":
                continue
            duration, watts = defaults.get(detail.get("layer"), (2., 5.))
            if action == "A008":
                duration, watts = max(.1, _number(params.get("budget_s"), 60.0)), 70.0
            elif action in {"A029", "A030"}:
                watts = 0.0
            elif action == "A004":
                watts = 100.0 if mobility == "air" else 5.0
            estimate += max(0.0, duration * watts / 3600.0)
        budget = task.get("resource_budget", {}) or {}
        if isinstance(budget, dict):
            estimate = max(estimate, _number(budget.get("energy_Wh"), 0.0))
        # An already-low platform can still seek its own charging service.  It
        # must have enough to reach it; reserve may be replenished there.
        charging = task.get("type") in {"T043", "T044"} or any(a in {"A029", "A030"} for a in method["resolved_steps"])
        required = estimate if charging else estimate + reserve
        ok = float(battery) >= required
        return ok, "电量及预留满足估计" if ok else "电量不足或侵占返程预留", estimate

    def commitment_filter(self, task, row, now):
        if _reported(row, "available", default=True) is False:
            return False, "最近报告不可用"
        if _reported(row, "health", "platform.health_state") in {"failed", "disabled"}:
            return False, "最近报告平台失效"
        unknown = object()
        current = _reported(row, "current_task", "current", "platform.current_task", default=unknown)
        platform = row.get("platform") or {}
        explicitly_idle = any(name in container and container[name] is None
                              for container in (row, platform)
                              for name in ("current_task", "current", "platform.current_task"))
        if current is unknown:
            return (True, "最近报告空闲") if explicitly_idle else (False, "尚无有效当前承诺报告")
        if not current:
            return True, "最近报告空闲"
        commitment = current if isinstance(current, dict) else {}
        if commitment.get("status") in _TERMINAL or commitment.get("id", current) == task.get("id"):
            return True, "没有冲突的活动承诺"
        policy = commitment.get("interruptibility", _reported(row, "interruptibility", default="non_interruptible"))
        if policy == "non_interruptible":
            return False, "当前承诺不可抢占"
        if policy == "finish_current" and _reported(row, "pending_action", "current_action_id", "platform.current_action_id"):
            return False, "须先完成当前动作"
        window = commitment.get("commitment_window", _reported(row, "commitment_window"))
        if isinstance(window, (list, tuple)) and len(window) >= 2:
            until = window[1]
        elif isinstance(window, dict):
            until = window.get("end_s", window.get("end", window.get("until_s")))
        else:
            until = None
        until = _number(until)
        if until is not None and until > now and task.get("priority", 0) <= commitment.get("priority", 0):
            return False, "现有承诺窗口优先"
        if task.get("priority", 0) <= commitment.get("priority", 0):
            return False, "新任务优先级不足以抢占"
        return True, "可在安全中断点抢占当前承诺"

    def _candidates(self, coordinator, sim, task, rows, reserved, *, entities=None, route_cache=None):
        catalog = self.catalog or coordinator.catalog
        considered, eligible = [], []
        # Coordination/protocol work belongs exclusively to its proposer.  It
        # must not reserve an agent or start recursive fleet allocation/bidding.
        if is_meta_task(task):
            return considered, eligible
        entities = coordinator.belief.entities(copy=False) if entities is None else entities
        carrier = None
        if task.get("type") in {"T034", "T035"}:
            # The latest delivered holder belief takes precedence over the
            # proposal's older carrier advertisement; neither is a truth read.
            carrier = _field(entities.get(str(task.get("target")), {}), "human.carrier_id",
                             _value((task.get("parameters", {}) or {}).get("carrier_id")))
        excluded = set(task.get("excluded_agents", task.get("rejected_by", ())) or ())
        # Rows are copied and stably sorted once by allocate; candidate results
        # remain task-local because exclusions and reservations change each time.
        for row in rows:
            aid = row["id"]
            report = {"agent": aid, "ok": False, "why_zh": "", "checks": {}}
            considered.append(report)
            if task.get("type") in {"T043", "T044", "T045"} and task.get("creator") != aid:
                report["why_zh"] = "本机补能或撤离任务不得替另一平台执行"
                continue
            if aid in excluded or aid in reserved:
                report["why_zh"] = "已拒绝/排除该候选" if aid in excluded else "本轮已有新承诺"
                continue
            spec = _candidate_spec(row)
            if not spec:
                report["why_zh"] = "未知具身规格"
                continue
            # Shared authored map policy, not a platform capability or telemetry
            # claim. Fleet reports intentionally whitelist only embodiment data.
            if coordinator.spec.get('ground_route_frontages'):
                spec['ground_route_frontages'] = coordinator.spec['ground_route_frontages']
            if spec.get("external"):
                method = resolve_method(catalog, task["type"], spec)
                executor_actions = _EXECUTOR_ACTIONS | (_AID_ACTIONS if task["type"] in {"T029", "T031"} else set())
                # Explicit mission-scoped contact is an actual onsite operator,
                # not a requester substituting A059 for specialist equipment.
                # Keep legacy external-service roles unchanged without that scope.
                if task["type"] == "T027" and "T027" in (spec.get("task_allowlist") or ()):
                    executor_actions = executor_actions | {"A009"}
                if not any(a in executor_actions for a in _requester_actions(method)):
                    report["why_zh"] = "该外部服务仍经A059请求；直接分配仅用于实际工具或承载主体"
                    continue
            if carrier is not None and carrier != "":
                identities = {str(identity) for identity in
                              (aid, _reported(row, "platform_eid", "platform_id"), spec.get("platform_eid"))
                              if identity is not None}
                ok = str(carrier) in identities
                why = "候选是已报告的实际承载者" if ok else "人员已由另一主体承载，不能分配给非承载者运输或移交"
                report["checks"]["carrier"] = {"ok": ok, "why_zh": why}
                if not ok:
                    report["why_zh"] = why
                    continue
            ok, why = self.commitment_filter(task, row, float(sim.t))
            report["checks"]["commitment"] = {"ok": ok, "why_zh": why}
            if not ok:
                report["why_zh"] = why
                continue
            ok, why, method = self.capability_filter(catalog, task, spec)
            report["checks"]["capability"] = {"ok": ok, "why_zh": why}
            if not ok:
                report["why_zh"] = why
                continue
            ok, why = self.payload_filter(task, row, spec, method, entities, identities=(
                {str(i) for i in (aid, _reported(row, "platform_eid", "platform_id"), spec.get("platform_eid")) if i is not None}))
            report["checks"]["payload"] = {"ok": ok, "why_zh": why}
            if not ok:
                report["why_zh"] = why
                continue
            ok, why, distance = self.access_filter(coordinator, task, row, spec, method,
                                                   entities=entities, route_cache=route_cache)
            report["checks"]["access"] = {"ok": ok, "why_zh": why}
            if not ok:
                report["why_zh"] = why
                continue
            ok, why, energy = self.battery_filter(task, row, spec, method, distance)
            report["checks"]["battery"] = {"ok": ok, "why_zh": why}
            if not ok:
                report["why_zh"] = why
                continue
            report.update(ok=True, why_zh="能力、载荷、已知通路、电量及承诺检查通过")
            score = float(task.get("priority", 0)) * 1000 - distance
            report["score"] = score
            eligible.append({"agent": aid, "score": score, "method": method,
                             "required_capabilities": required_capabilities(catalog, task["type"], spec, method=method),
                             "distance_m": distance, "estimated_energy_Wh": energy})
        return considered, eligible

    def allocate(self, coordinator_runtime, sim, proposals, fleet_view):
        # Borrow beliefs read-only and share topology within this synchronous
        # allocation call; nothing survives to misrepresent a later revision.
        decisions, reserved, route_cache = [], set(), {}
        rows = sorted(_fleet_rows(fleet_view), key=lambda r: r["id"])
        entities = coordinator_runtime.belief.entities(copy=False)
        self._method_cache.clear()
        ordered = sorted(proposals, key=lambda t: (-t.get("priority", 0), str(t["id"])))
        searches = {}
        for index, task in enumerate(ordered):
            if task.get("type") == "T014":
                searches.setdefault(task.get("priority", 0), []).append(index)
        for indices in searches.values():
            if len(indices) < 2:
                continue
            ranked = []
            for index in indices:
                task = ordered[index]
                _, eligible = self._candidates(coordinator_runtime, sim, task, rows, reserved,
                                               entities=entities, route_cache=route_cache)
                ranked.append((max((c["score"] for c in eligible), default=-inf), task))
            # Equal-priority interior searches use the existing known-route
            # score, not lexical ids (#101 formerly overtook nearby #55/#78).
            # Only reorder these search slots; other task policies stay intact.
            ranked.sort(key=lambda row: -row[0])
            for index, (_, task) in zip(indices, ranked):
                ordered[index] = task
        for task in ordered:
            if is_meta_task(task):
                continue
            considered, eligible = self._candidates(coordinator_runtime, sim, task, rows, reserved,
                                                    entities=entities, route_cache=route_cache)
            best = max(eligible, key=lambda c: (c["score"], c["agent"]), default=None)
            decision = {"task_id": task["id"], "rev": task.get("rev", 0), "agent_id": None,
                        "reason_zh": "没有满足全部检查的已知候选", "score": None, "considered": considered}
            if best:
                reserved.add(best["agent"])
                decision.update(agent_id=best["agent"], score=best["score"],
                                reason_zh="按优先级与已知距离选择可执行候选",
                                method=deepcopy(best["method"]), required_capabilities=best["required_capabilities"])
            decisions.append(decision)
        return decisions


class BiddingAllocator(GreedyCapabilityAccessAllocator):
    """Simple asynchronous A051 bidding; only delivered bids may be recorded.

    Receivers execute A051 and send kind='bid', op='bid_response', task_id,
    agent_id, score, rev.  The coordinator's on_message calls ``record_bid``.
    An absent/lost bid never becomes an implicit bid from a live agent object.
    """

    def __init__(self, catalog=None, reserve_fraction=0.15, bid_timeout_s=30.0):
        super().__init__(catalog, reserve_fraction)
        self.bid_timeout_s = float(bid_timeout_s)
        if not isfinite(self.bid_timeout_s) or self.bid_timeout_s <= 0:
            raise ValueError("竞价超时须为正有限数 / bid timeout must be positive")
        self.rounds = {}

    def record_bid(self, task_id, agent_id, score, rev=None, *, received_s=None):
        round_ = self.rounds.get(task_id)
        if round_ is None or round_.get("closed") or rev is not None and rev != round_["rev"]:
            return False
        if received_s is not None:
            received_s = _number(received_s)
            if received_s is None or received_s >= round_["sent_s"] + self.bid_timeout_s:
                return False
        score = _number(score)
        if agent_id not in round_["receivers"] or score is None:
            return False
        round_["bids"][agent_id] = score
        return True

    def allocate(self, coordinator_runtime, sim, proposals, fleet_view):
        decisions, reserved, route_cache = [], set(), {}
        rows = sorted(_fleet_rows(fleet_view), key=lambda r: r["id"])
        entities = coordinator_runtime.belief.entities(copy=False)
        self._method_cache.clear()
        for task in sorted(proposals, key=lambda t: (-t.get("priority", 0), str(t["id"]))):
            if is_meta_task(task):
                continue
            tid, rev = task["id"], task.get("rev", 0)
            round_ = self.rounds.get(tid)
            if round_ is not None and rev < round_["rev"]:
                continue
            if round_ is not None and round_["rev"] == rev and round_.get("closed"):
                continue
            considered, eligible = self._candidates(coordinator_runtime, sim, task, rows, reserved,
                                                    entities=entities, route_cache=route_cache)
            if round_ is None or round_["rev"] != rev:
                # A task with no currently available candidate has not begun a
                # decision/bidding round.  Runtime owns its waiting/expiry policy.
                receivers = sorted(candidate["agent"] for candidate in eligible)
                if not receivers:
                    continue
                payload = {"op": "bid_request", "task": deepcopy(task), "task_id": tid, "rev": rev}
                sim.send_message(coordinator_runtime.id, receivers, "bid", payload,
                                 len(json.dumps(payload, ensure_ascii=False).encode("utf-8")),
                                 task_ref=(tid, rev), ttl_s=self.bid_timeout_s)
                round_ = {"rev": rev, "sent_s": float(sim.t), "receivers": receivers,
                          "bids": {}, "closed": False}
                self.rounds[tid] = round_
            candidates = [candidate for candidate in eligible if candidate["agent"] in round_["bids"]]
            best = max(candidates, key=lambda c: (round_["bids"][c["agent"]], c["score"], c["agent"]), default=None)
            timed_out = float(sim.t) - round_["sent_s"] >= self.bid_timeout_s
            # Waiting is not an allocation decision.  In particular it must not
            # consume runtime's once-per-revision decision allowance.
            if best is None and not timed_out:
                continue
            round_["closed"] = True
            decision = {"task_id": tid, "rev": rev, "agent_id": None, "score": None, "considered": considered,
                        "reason_zh": "竞价超时或无有效报价"}
            if best:
                reserved.add(best["agent"])
                decision.update(agent_id=best["agent"], score=round_["bids"][best["agent"]],
                                reason_zh="在通过独立检查的实际报价中选择最高报价",
                                method=deepcopy(best["method"]), required_capabilities=best["required_capabilities"])
            decisions.append(decision)
        return decisions
