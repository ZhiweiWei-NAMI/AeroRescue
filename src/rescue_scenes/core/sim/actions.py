"""Catalog action models; authored, uncalibrated models, never a vision backend.

The kernel owns writes, accounting and result delivery.  Optional lifecycle hooks
``start``, ``progress`` and ``cancel`` return ActionOutcome just like ``effect``;
the kernel applies their writes without treating them as terminal action results.
Private per-instance bookkeeping lives in the kernel-owned params dictionary.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import heapq
import json
import math
from rescue_scenes.core.catalog import load
from rescue_scenes.core.contracts import ActionOutcome, Observation
from rescue_scenes.core.world.profiles import OVERLAYS
from rescue_scenes.core.world.truth import (BUILDING_NAVIGATION_FIELDS, footprint_entry,
                                           wheeled_navigation_goal, building_approach_attachment,
                                           building_approach_leg, interior_approach_buildings,
                                           isolated_ground_route, _prepare_polygon, _exact_key,
                                           _footprint_intersects, _footprint_candidates)
from rescue_scenes.core.agents.allocation import belief_path, obstacle_approach_path, obstacle_work_roads, water_access_goal


def _exogenous_rng(sim, agent, fallback, family, purpose, target, local_seq=0):
    """Optional branch CRN, keyed by physical event context rather than action ids."""
    source = getattr(sim, "exogenous_crn", None)
    if source is None:
        return fallback
    return source.stream(actor_id=agent.id, event_family=family, purpose=purpose,
                         target_anchor=str(target), quantized_t=sim.t, local_seq=local_seq)


MOVEMENT = {"A001", "A002", "A003", "A021"}
NETWORK = {"A032", "A033", "A034", "A035", "A036", "A037", "A065"}
COMPUTE = {f"A{i:03d}" for i in range(42, 50)}
TASKS = {f"A{i:03d}" for i in range(50, 59)}
CLEARING = {"A012", "A013", "A016"}
# F084 human.destination_id: "救援/医疗接收点". Supply/charging stations are never handover receivers.
RECEIVING_STATION_KINDS = frozenset({"medical", "shelter"})
ATOMIC_STEPS = {"A020", "A022", "A024", "A025", "A026", "A031"}
PHYSICAL = MOVEMENT | {f"A{i:03d}" for i in range(8, 32)} | {"A004", "A062", "A063"}
TERMINAL_TASKS = {"completed", "failed", "canceled", "expired"}
SENSOR_CAPS = {"rgb": "C05", "thermal": "C06", "audio": "C07", "RF": "C08",
               "depth": "C09", "lidar": "C09", "gas": "C10"}
SCRIPTED_SENSORS = {"scripted_rgb": "rgb", "scripted_range": "depth", "scripted_thermal": "thermal",
                    "scripted_acoustic": "audio", "scripted_rf": "RF", "scripted_environment": "gas"}
# Seconds / watts / CPU seconds: deliberately authored, not calibrated timings.
LAYER_COSTS = {
    "物理": (15., 120., 0.), "专业服务": (90., 220., 0.),
    "抽象物理服务": (120., 70., 0.), "传感器": (3., 8., 0.),
    "通信": (1., 2., .01), "计算": (2., 12., 1.),
    "计算工具": (3., 12., 2.), "资源": (60., 0., 0.),
}
DEFAULT_COST = (2., 5., .1)
TARGET_CLASSES = {
    "A008": {"building"}, "A009": {"human"}, "A010": {"building", "hazard"}, "A011": {"road", "building"},
    "A012": {"obstacle"}, "A013": {"obstacle"}, "A014": {"building", "road"},
    "A015": {"building", "obstacle"}, "A016": {"obstacle"},
    **{f"A{i:03d}": {"human"} for i in range(17, 24)},
    **{f"A{i:03d}": {"item"} for i in range(24, 28)},
    "A031": {"platform"}, "A062": {"hazard"}, "A063": {"hazard"},
}


def _num(value, default=0.):
    try:
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except (ValueError, TypeError):
        return float(default)


def _clamp(value, lo=0., hi=1.):
    return min(hi, max(lo, _num(value)))


def _list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _target(params):
    for key in ("target", "target_id", "target_ref", "building_id", "human_id",
                "obstacle_id", "item_id", "platform_id", "hazard_id", "road_id"):
        value = params.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return value.get("id") or value.get("entity_id")
    return None


def _entity(sim, eid):
    return sim.world.entities.get(eid) if eid else None


def _part(entity, key, default=None):
    if isinstance(entity, dict):
        return entity.get(key, default)
    return getattr(entity, key, default)


def _fields(sim, eid):
    return _part(_entity(sim, eid), "fields", {}) or {}


def _cls(sim, eid):
    return _part(_entity(sim, eid), "cls", _part(_entity(sim, eid), "class", ""))


def _xy(value):
    if isinstance(value, dict):
        value = value.get("position", value.get("xy", value.get("coords", value)))
        if isinstance(value, dict) and "x" in value and "y" in value:
            value = [value["x"], value["y"]]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
            return [_num(value[0]), _num(value[1])]
    return None


def _center(geometry):
    if not isinstance(geometry, dict):
        return _xy(geometry)
    coords = geometry.get("coords", geometry.get("coordinates", []))
    point = _xy(coords)
    if point is not None:
        return point
    if coords and isinstance(coords[0], (list, tuple)) and coords[0] and isinstance(coords[0][0], (list, tuple)):
        coords = coords[0]
    points = [p for p in (_xy(v) for v in coords) if p is not None]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return [sum(p[i] for p in points) / len(points) for i in (0, 1)] if points else None


def _pose(sim, eid):
    return (_xy(sim.world.get(eid, "entity.pose"))
            or _xy(sim.world.get(eid, "human.pose"))
            or _center(_part(_entity(sim, eid), "geometry", {})))


def _own_pose(sim, agent):
    return _pose(sim, agent.platform_eid) or [0., 0.]


def _carried_pose_writes(sim, agent, point):
    writes = []
    for eid in sorted(sim.world.entities):
        if _cls(sim, eid) == "human" and sim.world.get(eid, "human.carrier_id") in (agent.id, agent.platform_eid):
            writes += [(eid, "entity.pose", point), (eid, "human.pose", point)]
    return writes


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) if a is not None and b is not None else math.inf


def _near(sim, agent, eid, radius=20.):
    return _dist(_own_pose(sim, agent), _pose(sim, eid)) <= radius


def _entry_value(entry, default=None):
    return entry.get("value", default) if isinstance(entry, dict) else default


def _known(agent, eid, key, default=None):
    return _entry_value(agent.belief.get(eid, key), default)


def _known_pose(agent, eid):
    return (_xy(_known(agent, eid, "entity.pose"))
            or _xy(_known(agent, eid, "belief.estimated_pose"))
            or _xy(_known(agent, eid, "human.pose"))
            or _center(_known(agent, eid, "entity.geometry", {}))
            or _center(_known(agent, eid, "entity.footprint", {})))


def _known_navigation_buildings(agent, mobility="wheeled", *, entities=None):
    buildings = []
    if entities is None:
        entities = agent.belief.entities(copy=False)
    for eid, entries in entities.items():
        if (_entry_value(entries.get("entity.kind")) == "building" and mobility == "wheeled"
                or _entry_value(entries.get('attr.blocks_ground_navigation')) is True
                or _entry_value(entries.get('attr.interior_access_via_entrance')) is True):
            geometry = _entry_value(entries.get("attr.navigation_footprint"),
                                   _entry_value(entries.get("entity.geometry"), _entry_value(entries.get("entity.footprint"), {})))
            polygon = geometry.get("coords", []) if isinstance(geometry, dict) else geometry
            vehicle = _entry_value(entries.get('attr.vehicle_approach')) if mobility == 'wheeled' else None
            buildings.append((eid, polygon,
                vehicle['entrance_ids'] if vehicle else _list(_entry_value(entries.get('building.entrance_ids'))),
                vehicle['stand_xy'] if vehicle else _entry_value(entries.get('attr.entrance_xy')),
                vehicle['heading_deg'] if vehicle else _entry_value(entries.get('attr.entrance_heading_deg')),
                vehicle['start_xy'] if vehicle else _entry_value(entries.get('attr.approach_start_xy')),
                vehicle['waypoints_xy'] if vehicle else _entry_value(entries.get('attr.approach_waypoints_xy'), [])))
    return buildings


def known_navigation_goal(agent, point, embodiment=None):
    """Resolve building stands and water berths using only the public map."""
    spec = embodiment or agent.spec
    if point is not None and spec.get('mobility') == 'water':
        return water_access_goal(point, agent.belief.entities(copy=False))
    if point is None or spec.get("mobility") != "wheeled":
        return point
    roads = {}
    entities = agent.belief.entities(copy=False)
    for eid, entries in entities.items():
        line = _entry_value(entries.get("road.centerline"))
        if isinstance(line, dict):
            line = line.get("coords", [])
        if line:
            roads[eid] = line
    return wheeled_navigation_goal(point, _known_navigation_buildings(agent, entities=entities), roads, spec.get("length_m"))


def _goal(sim, agent, params, known=False):
    def navigation_goal(point):
        if point is None:
            return None
        return known_navigation_goal(agent, point) if known else sim.world.navigation_goal(point, agent.spec)

    for key in ("goal", "goal_xy", "destination", "destination_xy", "destination_id", "waypoint", "position"):
        value = params.get(key)
        if isinstance(value, str):
            return navigation_goal(_known_pose(agent, value) if known else _pose(sim, value))
        point = _xy(value)
        if point is not None:
            return navigation_goal(point)
    path = params.get("path", params.get("route"))
    if isinstance(path, dict):
        path = path.get("waypoints")
    if isinstance(path, (list, tuple)) and path:
        endpoint = _xy(path[-1])
        if endpoint is not None:
            return navigation_goal(endpoint)
    target = _target(params)
    return navigation_goal(_known_pose(agent, target) if known else _pose(sim, target))


def _station(params):
    return params.get("station_id") or params.get("station") or params.get("receiver_id") or params.get("destination_id")


def _obs_dict(obs):
    return obs.to_dict() if hasattr(obs, "to_dict") else obs


def _inputs(agent, params):
    """Only locally held evidence or explicit observation objects, never truth."""
    evidence = getattr(agent.belief, "evidence", {})
    values = _list(params.get("observations", params.get("observation")))
    values += _list(params.get("input_refs", params.get("evidence_ids")))
    result = []
    for item in values:
        if isinstance(item, str):
            item = evidence.get(item)
        if item is not None:
            row = _obs_dict(item)
            if isinstance(row, dict) and isinstance(row.get("values"), dict):
                result.append(deepcopy(row))
    return result


def _observation(sim, agent, params, about, values, kind="authored_sensing", confidence=.85,
                 model="scripted_sensing_v1", suffix=""):
    token = f"{params.get('_instance_id', agent.id)}|{sim.t}|{about}|{kind}|{suffix}"
    oid = "obs-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:18]
    return Observation(id=oid, by=agent.id, t=sim.t, about=list(about), kind=kind,
                       values=deepcopy(values), confidence=_clamp(confidence), model=model)


def _out(status="succeeded", failure="none", writes=(), local=(), obs=(), text="", evidence=()):
    observations = list(obs)
    return ActionOutcome(status=status, failure_code=failure, truth_writes=list(writes),
                         local_writes=list(local), observations=observations,
                         feedback_zh=text, evidence_ids=list(evidence) or [o.id for o in observations])


def _fail(code, text, writes=(), obs=(), partial=False):
    return _out("partial" if partial else "failed", code, writes=writes, obs=obs, text=text)


# Platform-facing rescue chain (frontend rescue-feed.ts counts handedOver from
# rescue.handover_acknowledged and extricated from rescue.extricated). Rows are
# derived from the world writes the kernel actually applied, never from plans.
RESCUE_CHAIN_KINDS = ("rescue.extricated", "rescue.loaded", "rescue.transport_started",
                      "rescue.transport_arrived", "rescue.handover_acknowledged")
_RESCUE_CHAIN_TEXT = {
    "rescue.extricated": "人员已实际解困；未自动装载或移交。",
    "rescue.loaded": "人员已实际装载到载具；尚未转运完成或移交。",
    "rescue.transport_started": "载具已带人进入转运；到达不等于移交。",
    "rescue.transport_arrived": "载具已带人实际到达接收点；尚待接收确认。",
    "rescue.handover_acknowledged": "接收站已确认接收；移交完成。",
}


def _rescue_chain_kind(action_id, field, before, after):
    """Map one applied world write to its platform rescue-chain event, if any."""
    if action_id == "A017" and field == "human.support_received":
        if "extricated" in (after or []) and "extricated" not in (before or []):
            return "rescue.extricated"
        return None
    # A023 reuses the A020 loading model for water recovery.
    if action_id in ("A020", "A023") and field == "human.carrier_id" and after:
        return "rescue.loaded"
    if action_id == "A021" and field == "human.handover_status":
        return {"in_transit": "rescue.transport_started", "arrived": "rescue.transport_arrived"}.get(after)
    if action_id == "A022" and field == "human.handover_status" and after == "acknowledged":
        return "rescue.handover_acknowledged"
    return None


def record_rescue_chain(sim, written, rec):
    """Append one compact rescue-chain row per state transition just applied.

    ``written`` are the events the kernel emitted while applying this outcome's
    truth writes, so the row cites committed truth and cannot claim a planned or
    failed step. Duplicates per (kind, person, action instance) are skipped.
    """
    action_id, instance = rec.get("action_id"), rec.get("id")
    if not action_id or not instance:
        return
    for event in written:
        kind = _rescue_chain_kind(action_id, event.field, event.before, event.after)
        person = event.subject
        if kind is None or not isinstance(person, str) or not person:
            continue
        if any(row.kind == kind and row.subject == person and row.action_instance == instance
               for row in sim.events):
            continue
        payload = {"person": person, "action_id": action_id, "source_seq": event.seq}
        carrier = event.actor
        if isinstance(carrier, str) and carrier:
            payload["carrier"] = carrier
        if kind in ("rescue.transport_started", "rescue.transport_arrived", "rescue.handover_acknowledged"):
            receiver = sim.world.get(person, "human.destination_id")
            if isinstance(receiver, str) and receiver:
                payload["receiver"] = receiver
                if kind == "rescue.handover_acknowledged":
                    payload["confirmed"] = (sim.world.get(receiver, "station.open_state") == "open"
                                            and sim.world.get(receiver, "attr.receiver_confirmation") is True)
        task_id = event.task
        if isinstance(task_id, str) and task_id:
            payload["task_id"] = task_id
        payload["write_seqs"] = sorted({row.seq for row in written if row.subject == person and row.field})
        # Carry the originating write's provenance so the platform can classify
        # the milestone exactly as it classifies the state transition itself.
        sim.emit(layer="world", family="field.write", kind=kind, grounding="proposed",
                 refs=[action_id], subject=person, actor=carrier, after=payload,
                 task=event.task, task_rev=event.task_rev,
                 backend_mode=event.backend_mode, scripted=event.scripted,
                 action_instance=instance, visible_to=["*truth*"],
                 text_zh=_RESCUE_CHAIN_TEXT[kind])


def _add_support(sim, human, tag):
    values = list(sim.world.get(human, "human.support_received", []) or [])
    return values if tag in values else values + [tag]


def _duration(params, default):
    return max(.1, _num(params.get("duration_s", params.get("budget_s", default)), default))


def _target_charge(params, maximum):
    return min(maximum, _num(params.get("target_Wh", params.get("target_battery_Wh")), maximum))


def _sensing_rate(sim, key, default):
    rate = default
    for oid in sim.config.get("overlays", []):
        effect = OVERLAYS.get(oid, {}).get("effects", {}).get("world", {})
        if key in effect:
            rate = max(rate, _num(effect[key], default))
    return _clamp(rate)


def _items(sim, agent, kinds=(), near=False):
    for eid in sorted(sim.world.entities):
        if _cls(sim, eid) != "item":
            continue
        if kinds and sim.world.get(eid, "item.kind") not in kinds:
            continue
        holder = sim.world.get(eid, "item.holder_id")
        if holder in (agent.id, agent.platform_eid) or (near and _near(sim, agent, eid)):
            if sim.world.get(eid, "item.condition", "usable") == "usable" and _num(sim.world.get(eid, "item.quantity", 1)) > 0:
                yield eid


def _equipment(sim, agent, params):
    eid = params.get("equipment_id")
    return eid if eid and _entity(sim, eid) else None


def _capacity(sim, agent, params):
    equipment = _equipment(sim, agent, params)
    return max(0., _num(sim.world.get(equipment, "equipment.rated_load_kg"), 250.) if equipment
               else _num(agent.spec.get("tool_capacity", agent.spec.get("rated_load_kg", 250.)), 250.))


def _charge_value(sim, eid):
    for key in ("attr.charge_Wh", "attr.battery_charge_Wh", "attr.battery_remaining_Wh"):
        value = sim.world.get(eid, key)
        if value is not None:
            return max(0., _num(value)), key
    return 0., "attr.charge_Wh"


def _interpolate(points, distance):
    if not points:
        return None
    for a, b in zip(points, points[1:]):
        length = _dist(a, b)
        if distance <= length and length > 0:
            ratio = max(0., distance) / length
            return [a[i] + (b[i] - a[i]) * ratio for i in (0, 1)]
        distance -= length
    return list(points[-1])


def _project(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    fraction = _clamp(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2) if length2 else 0.
    return (a[0] + dx * fraction, a[1] + dy * fraction), fraction


def _intersection(a, b, c, d):
    ux, uy, vx, vy = b[0] - a[0], b[1] - a[1], d[0] - c[0], d[1] - c[1]
    determinant = ux * vy - uy * vx
    if abs(determinant) < 1e-9:
        return None
    wx, wy = c[0] - a[0], c[1] - a[1]
    s, t = (wx * vy - wy * vx) / determinant, (wx * uy - wy * ux) / determinant
    if -1e-9 <= s <= 1 + 1e-9 and -1e-9 <= t <= 1 + 1e-9:
        return (round(a[0] + s * ux, 7), round(a[1] + s * uy, 7))
    return None


# Content-keyed process LRUs; neither cache retains a belief or an agent.
_KNOWN_ROUTE_TOPOLOGY = OrderedDict()
_KNOWN_ROUTE_BASE = OrderedDict()
_KNOWN_ROUTE_CACHE_LIMIT = 16


def _known_route_cache_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _KNOWN_ROUTE_CACHE_LIMIT:
        cache.popitem(last=False)
    return value


def _known_route_cuts(segments, key):
    cached = _KNOWN_ROUTE_TOPOLOGY.get(key)
    if cached is not None:
        _KNOWN_ROUTE_TOPOLOGY.move_to_end(key)
        return cached
    cuts = [[a, b] for a, b, _, _ in segments]
    endpoint_keys = [((a[0].hex(), a[1].hex()), (b[0].hex(), b[1].hex()))
                     for a, b, _, _ in segments]
    projections = {}

    def endpoint_projection(segment, point, point_key):
        projection_key = (segment, point_key)
        if projection_key not in projections:
            if len(projections) >= 8192:
                projections.clear()
            a, b, _, _ = segments[segment]
            projections[projection_key] = _project(point, a, b)[0]
        return projections[projection_key]

    for index, (a, b, _, _) in enumerate(segments):
        for other in range(index):
            c, d, _, _ = segments[other]
            cross = _intersection(a, b, c, d)
            if cross is not None:
                cuts[index].append(cross)
                cuts[other].append(cross)
            for point, point_key in zip((a, b), endpoint_keys[index]):
                projected = endpoint_projection(other, point, point_key)
                if _dist(projected, point) < 1e-6:
                    cuts[other].append(point)
            for point, point_key in zip((c, d), endpoint_keys[other]):
                projected = endpoint_projection(index, point, point_key)
                if _dist(projected, point) < 1e-6:
                    cuts[index].append(point)
    # Cache the insertion stream, not a set: copying a set can change sort ties.
    return _known_route_cache_put(_KNOWN_ROUTE_TOPOLOGY, key,
                                 tuple(tuple(points) for points in cuts))


def known_route(agent, start_xy, goal_xy, embodiment=None, *, work_target=None):
    """Route over local evidence/public geometry only; never accepts WorldTruth."""
    spec = embodiment or agent.spec
    failure = {"ok": False, "waypoints": [start_xy] if start_xy is not None else [],
               "length_m": 0., "blocked_by": [], "uncertain": True, "road_ids": []}
    if start_xy is None or goal_xy is None or spec.get("mobility") == "none":
        return failure
    goal_xy = known_navigation_goal(agent, goal_xy, spec)
    if goal_xy is None:
        return failure
    if work_target is not None and spec.get("mobility") in {"legged", "wheeled", "foot"}:
        return {**obstacle_approach_path(agent.belief, start_xy, work_target, spec), "uncertain": True}
    if spec.get('ground_route_frontages') and spec.get('mobility') in {'legged', 'wheeled', 'foot'}:
        # Same belief graph, source-supported endpoint candidates, permissions,
        # rounding and road labels as allocation/A066. Execution below still
        # checks the fixed candidate against actual obstacles; no truth detour.
        return {**belief_path(agent.belief, start_xy, goal_xy, spec), 'uncertain': True}
    if _dist(start_xy, goal_xy) < 1e-6:
        return {"ok": True, "waypoints": [list(start_xy)], "length_m": 0., "blocked_by": [], "uncertain": False, "road_ids": []}
    # This synchronous query only reads entries; returned routes own their points.
    entities = agent.belief.entities(copy=False)
    mobility = spec.get("mobility")
    if mobility in {'foot', 'legged', 'wheeled'}:
        isolated = isolated_ground_route(start_xy, goal_xy,
            [(eid, _entry_value(fs.get('entity.geometry'))['coords']) for eid, fs in entities.items()
             if _entry_value(fs.get('attr.ground_access_isolated')) is True])
        if isolated is not None:
            return {**isolated, 'uncertain': False}
    if mobility in ("air", "water"):
        water, exclusions = [], []
        for entries in entities.values():
            if _entry_value(entries.get("entity.kind")) == "water":
                water.append(_entry_value(entries.get("entity.geometry"), _entry_value(entries.get("entity.footprint"))))
            zone = _entry_value(entries.get("hazard.exclusion_zone"))
            if zone is True:
                zone = _entry_value(entries.get("hazard.footprint"), _entry_value(entries.get("entity.geometry")))
            if zone:
                exclusions.append(zone)
        count = max(1, int(_dist(start_xy, goal_xy) / 5.) + 1)
        for index in range(count + 1):
            point = [start_xy[k] + (goal_xy[k] - start_xy[k]) * index / count for k in (0, 1)]
            if mobility == "water" and not any(_point_in(point, poly) for poly in water):
                return failure
            if any(_point_in(point, poly) for poly in exclusions):
                return failure
        return {"ok": True, "waypoints": [list(start_xy), list(goal_xy)],
                "length_m": _dist(start_xy, goal_xy), "blocked_by": [], "uncertain": True, "road_ids": [None]}
    segments, uncertain, blocked, incompatible = [], {}, [], set()
    exclusions = []
    for entries in entities.values():
        zone = _entry_value(entries.get('hazard.exclusion_zone'))
        if zone is True:
            zone = _entry_value(entries.get('hazard.footprint'))
        if isinstance(zone,dict):
            exclusions.append(_prepare_polygon(zone['coords']))
        elif isinstance(zone,list):
            exclusions.append(_prepare_polygon(zone))
    for eid, entries in entities.items():
        points = _entry_value(entries.get("road.centerline"))
        if isinstance(points, dict):
            points = points.get("coords", points.get("coordinates", []))
        points = [tuple(p) for p in (_xy(v) for v in (points or [])) if p is not None]
        if len(points) < 2:
            continue
        state = _entry_value(entries.get("road.entry_state"))
        access = _entry_value(entries.get("belief.estimated_access"))
        access_time = entries.get("belief.estimated_access", {}).get("t", -math.inf)
        measurement_time = max((entries.get(key, {}).get("t", -math.inf) for key in
                                ("road.entry_state", "road.clear_width_m", "road.step_height_m")))
        if measurement_time > access_time:
            access = None  # A fresh measured clearance supersedes the old local estimate.
        if isinstance(access, dict):
            state = access.get("entry_state", state)
        width = _entry_value(entries.get("road.clear_width_m"), _entry_value(entries.get("road.width_m"), math.inf))
        if isinstance(access, dict) and isinstance(access.get("clear_width_m"), (int, float)):
            width = access["clear_width_m"]
        step = _num(_entry_value(entries.get("road.step_height_m")))
        depth = _num(_entry_value(entries.get("road.water_depth_m")))
        admitted = (state not in ("closed", "unsafe") and _num(width, math.inf) >= _num(spec.get("width_m")) and
                    step <= _num(spec.get("step_height_m")) and depth <= _num(spec.get("water_tolerance_m")) and
                    not (step > 0 and mobility == "wheeled"))
        allowed = _entry_value(entries.get('attr.allowed_mobilities'))
        if allowed is not None and mobility not in allowed:
            admitted = False; incompatible.add(eid)
        if _entry_value(entries.get('attr.professional_only')) and not spec.get('professional_access', False):
            admitted = False; incompatible.add(eid)
        if any(footprint_entry(a,b,poly) is not None for a,b in zip(points,points[1:]) for poly in exclusions):
            admitted = False
        if isinstance(access, dict) and access.get("passable") is False:
            admitted = False
        if state == "blocked" and "road.clear_width_m" not in entries and not isinstance(access, dict):
            admitted = False
        if not admitted:
            blocked.append(eid)
        uncertain[eid] = state is None or "road.clear_width_m" not in entries
        segments += [(a, b, eid, admitted) for a, b in zip(points, points[1:]) if _dist(a, b) > 1e-8]
    failure["blocked_by"] = blocked
    if not segments:
        return failure
    topology_key = _exact_key(tuple((a, b) for a, b, _, _ in segments))
    cuts = _known_route_cuts(segments, topology_key)
    buildings = [(row[0], _prepare_polygon(row[1]), *row[2:])
                 for row in _known_navigation_buildings(agent, mobility)]
    interior = interior_approach_buildings(buildings, (start_xy, goal_xy),
        {row[0] for row in buildings if mobility in ('foot', 'legged') and
         _known(agent, row[0], 'attr.interior_access_via_entrance') is True})
    interior_by_id = {}
    for row in interior:
        interior_by_id.setdefault(row[0], []).append(row)
    def clear_hits(a, b):
        hits = [row[0] for row in _footprint_candidates(a, b, buildings)
                if _footprint_intersects(a, b, row[1]) and
                not building_approach_leg(a, b, interior_by_id.get(row[0], []))]
        if any(_footprint_intersects(a, b, poly) for poly in exclusions):
            hits.append('hazard.exclusion_zone')
        return hits

    def clear(a, b):
        hits = clear_hits(a, b)
        blocked.extend(eid for eid in hits if eid not in blocked)
        return not hits

    attachments, approaches, added_cuts = [], [], {}
    if not any(eid not in incompatible for _, _, eid, _ in segments):
        return failure
    for point in (start_xy, goal_xy):
        approach = building_approach_attachment(point, interior + buildings)
        attachment = approach[1] if approach else point
        candidates = [(_dist(attachment, projected), i, projected)
                      for i, (a, b, eid, _) in enumerate(segments)
                      if eid not in incompatible and (not approach or eid in approach[0])
                      for projected in [_project(attachment, a, b)[0]] if clear(attachment, projected)]
        if not candidates:
            return failure
        _, index, projected = min(candidates)
        added_cuts.setdefault(index, []).append(projected)
        attachments.append(tuple(round(v, 7) for v in projected))
        approaches.append(approach)

    def segment_graph(index, points):
        a, b, road, admitted = segments[index]
        records, hits = [], []
        if admitted:
            points = sorted(set(points), key=lambda p: _project(p, a, b)[1])
            for c, d in zip(points, points[1:]):
                blockers = clear_hits(c, d)  # Raw coordinates, before rounding.
                hits.extend(blockers)
                if blockers:
                    continue
                c, d = tuple(round(v, 7) for v in c), tuple(round(v, 7) for v in d)
                length = _dist(c, d)
                records.append((index, c, (d, length, road)))
                records.append((index, d, (c, length, road)))
        return tuple(records), tuple(hits)

    # Admission and permissions are derived afresh, including query-local
    # interior legs. Exact keys preserve order, types and signed-zero values.
    base_key = (topology_key, _exact_key(tuple((road, admitted) for _, _, road, admitted in segments)),
                _exact_key(buildings), _exact_key(exclusions), _exact_key(interior_by_id))
    base = _KNOWN_ROUTE_BASE.get(base_key)
    if base is None:
        results = tuple(segment_graph(i, points) for i, points in enumerate(cuts))
        neighbors, base_graph = {}, {}
        for records, _ in results:
            for record in records:
                _, node, edge = record
                neighbors.setdefault(node, []).append(record)
                base_graph.setdefault(node, []).append(edge)
        base = _known_route_cache_put(_KNOWN_ROUTE_BASE, base_key, (results, neighbors, base_graph))
    else:
        _KNOWN_ROUTE_BASE.move_to_end(base_key)
    results, neighbors, base_graph = base
    changed = {i: segment_graph(i, [*cuts[i], *points]) for i, points in added_cuts.items()}
    # Replay only the actual query's blockers, after attachments and in the
    # original segment/cut order. Base-only blockers must never leak through.
    for i, result in enumerate(results):
        blocked.extend(eid for eid in changed.get(i, result)[1] if eid not in blocked)
    graph = base_graph.copy()
    affected = {}
    for i, (records, _) in changed.items():
        for _, node, _ in results[i][0]:
            affected.setdefault(node, [])
        for record in records:
            affected.setdefault(record[1], []).append(record)
    for node, records in affected.items():
        records = [row for row in neighbors.get(node, ()) if row[0] not in changed] + records
        records.sort(key=lambda row: row[0])  # Stable within each segment.
        graph.pop(node, None)
        if records:
            # Equal signed-zero keys keep the first surviving insertion's
            # actual coordinates; neighbor values are never canonicalized.
            graph[records[0][1]] = [row[2] for row in records]
    owned = set(affected)

    def connect(a, b, road):
        if not clear(a, b):
            return
        a, b = tuple(round(v, 7) for v in a), tuple(round(v, 7) for v in b)
        for node in (a, b):
            if node not in owned:
                graph[node] = list(graph.get(node, ()))
                owned.add(node)
        graph.setdefault(a, []).append((b, _dist(a, b), road))
        graph.setdefault(b, []).append((a, _dist(a, b), road))
    start, goal = tuple(start_xy), tuple(goal_xy)
    for point, attachment, approach in zip((start, goal), attachments, approaches):
        tail = approach[2] if approach else [point]
        for a, b in zip(tail, tail[1:]):
            connect(a, b, None)
        connect(tail[-1], attachment, None)
    if building_approach_leg(start, goal, interior + buildings):
        connect(start, goal, None)
    start, goal = tuple(round(v, 7) for v in start), tuple(round(v, 7) for v in goal)
    queue, distances, previous = [(0., start)], {start: 0.}, {}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        if node == goal:
            break
        for nxt, length, road in graph.get(node, ()):
            candidate = distance + length
            if candidate < distances.get(nxt, math.inf):
                distances[nxt], previous[nxt] = candidate, (node, road)
                heapq.heappush(queue, (candidate, nxt))
    if goal not in distances:
        return failure
    nodes, roads = [goal], []
    while nodes[-1] != start:
        node, road = previous[nodes[-1]]
        nodes.append(node)
        roads.append(road)
    return {"ok": True, "waypoints": [list(p) for p in reversed(nodes)], "length_m": distances[goal],
            "road_ids": list(reversed(roads)), "blocked_by": [], "uncertain": any(uncertain[r] for r in roads if r)}


def _point_in(point, polygon):
    if isinstance(polygon, dict):
        polygon = polygon.get("coords", polygon.get("coordinates", []))
    if point is None or not isinstance(polygon, (list, tuple)):
        return False
    if polygon and isinstance(polygon[0], (list, tuple)) and polygon[0] and isinstance(polygon[0][0], (list, tuple)):
        polygon = polygon[0]
    points = [p for p in (_xy(v) for v in (polygon or [])) if p is not None]
    inside = False
    for a, b in zip(points, points[1:] + points[:1]):
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return inside


def _movement_candidate(agent, start, goal, params):
    work_target = params.get("approach_obstacle_id")
    if goal is None:
        return known_route(agent, start, goal)
    supplied = params.get("path", params.get("route"))
    if isinstance(supplied, dict):
        supplied = supplied.get("waypoints")
    if not isinstance(supplied, (list, tuple)) or not supplied:
        return known_route(agent, start, goal, work_target=work_target)
    points = [p for p in (_xy(p) for p in supplied) if p is not None]
    if not points:
        return known_route(agent, start, goal, work_target=work_target)
    if _dist(points[0], start) > 1e-6:
        points.insert(0, list(start))
    if agent.spec.get("mobility") == "wheeled":
        # A supplied route may still end at the old interior casualty pose.
        # Replace that endpoint; appending the entrance would first drive in.
        points[-1] = known_navigation_goal(agent, points[-1])
        if points[-1] is None:
            return known_route(agent, start, None)
    if goal is not None and _dist(points[-1], goal) > 1e-6:
        points.append(list(goal))
    segments = []
    for eid, entries in agent.belief.entities(copy=False).items():
        centerline = _entry_value(entries.get("road.centerline"), [])
        if isinstance(centerline, dict):
            centerline = centerline.get("coords", centerline.get("coordinates", []))
        line = [p for p in (_xy(p) for p in centerline) if p is not None]
        segments += [(eid, a, b) for a, b in zip(line, line[1:])]
    roads = []
    for a, b in zip(points, points[1:]):
        road = next((eid for eid, c, d in segments if _dist(_project(a, c, d)[0], a) < .01 and _dist(_project(b, c, d)[0], b) < .01), None)
        roads.append(road)
    ground = agent.spec.get("mobility") in ("legged", "wheeled", "foot")
    buildings = _known_navigation_buildings(agent, agent.spec.get('mobility')) if ground else []
    interior = interior_approach_buildings(buildings, (points[0], points[-1]),
        {row[0] for row in buildings if agent.spec.get('mobility') in ('foot', 'legged') and
         _known(agent, row[0], 'attr.interior_access_via_entrance') is True})
    approaches = [building_approach_leg(a, b, interior + buildings) for a, b in zip(points, points[1:])]
    invalid = ground and (not any(roads) and not (approaches and all(approaches))
                          or any(road is None and not approach for road, approach in zip(roads[1:-1], approaches[1:-1])))
    return {"ok": not invalid, "waypoints": points, "road_ids": roads,
            "length_m": sum(_dist(a, b) for a, b in zip(points, points[1:])),
            "blocked_by": ["off_known_roads"] if invalid else [], "uncertain": True,
            "work_target": work_target}


def _execution_cache(world, name):
    """Keep execution LRUs separate from truth-path cache churn, per nav token."""
    prepared = world._navigation_inputs()
    if name not in prepared:
        prepared[name] = OrderedDict()
    return prepared[name]


def _execution_route(sim, agent, candidate, measure=False):
    """Reuse execution geometry only while its physical inputs are unchanged."""
    world = sim.world
    # An absent/invalid candidate falls back to the current pose in the model.
    fallback = None if any(_xy(p) is not None for p in candidate.get("waypoints", [])) else _own_pose(sim, agent)
    key = ("execution", _exact_key(candidate), _exact_key(agent.spec), measure,
           _exact_key(fallback), _exact_key(sim.config.get("bounds_m", [0., 0., 2000., 2000.])))
    # _move_progress keeps the original candidate, so a hit skips validation of
    # the whole unchanged remainder even when world.path fills its own cache.
    # The enclosing preparation is cleared whenever the navigation token changes.
    cache = _execution_cache(world, "execution_routes")
    result = cache.get(key)
    if result is None:
        result = _execution_route_uncached(sim, agent, candidate, measure)
        if len(cache) >= 4096:
            cache.popitem(last=False)
        cache[key] = result
    else:
        cache.move_to_end(key)
    return deepcopy(result)


def _execution_segment(sim, agent, a, b, road, open_work_section, measure,
                       endpoint, buildings, interior_by_id, bounds):
    """Check one fixed leg; endpoint permissions remain part of the caller's key."""
    world = sim.world
    mobility = agent.spec.get('mobility')
    blockers = []
    stop = None
    # Retain the physical travel model's lengths; never follow its truth detour.
    physical = world.path(a, b, agent.spec) if measure else {"ok": True, "length_m": _dist(a, b)}
    outside = any(not bounds[k] <= b[k] <= bounds[k + 2] for k in (0, 1))
    if outside:
        fraction = 1.
        for axis in (0, 1):
            delta = b[axis] - a[axis]
            if delta > 0 and b[axis] > bounds[axis + 2]:
                fraction = min(fraction, max(0., (bounds[axis + 2] - a[axis]) / delta))
            elif delta < 0 and b[axis] < bounds[axis]:
                fraction = min(fraction, max(0., (bounds[axis] - a[axis]) / delta))
        stop = [a[k] + (b[k] - a[k]) * fraction for k in (0, 1)]
        blockers = ["outside_bounds"]
    elif road and not world.passable(road, agent.spec)[0] and not open_work_section:
        blockers = [road]
        obstacles = [eid for eid in _list(world.get(road, "road.obstacle_ids"))
                     if world.get(eid, "obstacle.state") in ("present", "partly_cleared")]
        ahead = []
        for eid in obstacles:
            position = _pose(sim, eid)
            if position is not None:
                projected, fraction = _project(position, a, b)
                if _dist(projected, position) <= max(3., _num(world.get(road, "road.width_m"), 3.)):
                    ahead.append((fraction, eid, projected))
        if ahead:
            fraction, obstacle, projected = min(ahead)
            blockers.append(obstacle)
            stop = _interpolate([a, b], max(0., fraction * _dist(a, b) - 3.))
        elif obstacles and world.get(road, "road.entry_state") not in ("closed", "unsafe") and _num(world.get(road, "road.water_depth_m")) <= _num(agent.spec.get("water_tolerance_m")):
            # Debris farther along this same polyline need not stop an approach.
            blockers = []
        else:
            stop = a
    elif mobility in ("air", "water"):
        prepared = world._navigation_inputs()
        if 'execution_region_zones' not in prepared:
            water = [(_part(e, "id"), _part(e, "geometry")) for e in world.entities.values() if _part(e, "cls") == "water"]
            zones = []
            for eid in world.entities:
                zone = world.get(eid, "hazard.exclusion_zone")
                if zone is True:
                    zone = world.get(eid, "hazard.footprint") or _part(_entity(sim, eid), "geometry")
                if zone:
                    zones.append((eid, zone))
            prepared['execution_region_zones'] = (water, zones)
        water, zones = prepared['execution_region_zones']
        count = max(1, int(_dist(a, b) / 3.) + 1)
        previous = a
        for sample in range(count + 1):
            point = [a[k] + (b[k] - a[k]) * sample / count for k in (0, 1)]
            forbidden = next((eid for eid, zone in zones if _point_in(point, zone)), None)
            outside = mobility == "water" and not any(_point_in(point, poly) for _, poly in water)
            if forbidden or outside:
                stop, blockers = previous, [forbidden or "outside_water"]
                break
            previous = point
    elif road is None and not physical.get("ok") and not endpoint:
        blockers = _list(physical.get("blocked_by"))
        stop = a
    if mobility in {'foot', 'legged', 'wheeled'}:
        isolated = isolated_ground_route(a, b, world._navigation_isolated_regions())
        if isolated is not None and not isolated['ok']:
            stop, blockers = a, isolated['blocked_by']
        prepared = world._navigation_inputs()
        if 'execution_ground_zones' not in prepared:
            zones = []
            for hazard in world.entities_of('hazard'):
                zone = hazard['fields'].get('hazard.exclusion_zone')
                if zone is True:
                    zone = hazard['geometry']
                polygon = zone['coords'] if isinstance(zone, dict) else zone if isinstance(zone, list) else []
                if polygon:
                    zones.append((hazard['id'], _prepare_polygon(polygon)))
            prepared['execution_ground_zones'] = zones
        for eid, polygon in prepared['execution_ground_zones']:
            entry = footprint_entry(a, b, polygon)
            if entry is not None and (stop is None or _dist(a, entry) <= _dist(a, stop)):
                stop = _interpolate([a, b], max(0., _dist(a, entry) - 1e-4))
                blockers = [eid]
    # Preserve collision order, including equal-distance hazard/building ties.
    for building in buildings:
        if building_approach_leg(a, b, interior_by_id.get(building[0], [])):
            continue
        entry = footprint_entry(a, b, building[1])
        if entry is not None and (stop is None or _dist(a, entry) <= _dist(a, stop)):
            stop = _interpolate([a, b], max(0., _dist(a, entry) - 1e-4))
            blockers = [building[0]]
    if stop is not None:
        return stop, blockers, 0.
    segment_length = _num(physical.get("length_m"))
    straight = _dist(a, b)
    return None, blockers, segment_length if physical.get("ok") and abs(segment_length - straight) < .01 else straight


def _execution_route_uncached(sim, agent, candidate, measure=False):
    """Execute a fixed known-map candidate, never silently choose a truth detour."""
    points = [_xy(p) for p in candidate.get("waypoints", [])]
    points = [p for p in points if p is not None]
    if not points:
        points = [_own_pose(sim, agent)]
    reached, length = [points[0]], 0.
    roads = candidate.get("road_ids", [])
    mobility = agent.spec.get("mobility")
    buildings = sim.world._prepared_navigation_buildings(mobility) if mobility in {'foot', 'legged', 'wheeled'} else []
    interior = interior_approach_buildings(buildings, (points[0], points[-1]),
        {row[0] for row in buildings if mobility in ('foot', 'legged') and
         sim.world.get(row[0], 'attr.interior_access_via_entrance') is True})
    interior_by_id = {}
    for row in interior:
        interior_by_id.setdefault(row[0], []).append(row)
    if not candidate.get("ok", False):
        return {"ok": False, "waypoints": reached, "length_m": 0., "blocked_by": candidate.get("blocked_by", [])}
    work_target = candidate.get("work_target")
    work_roads = (obstacle_work_roads({eid: _fields(sim, eid) for eid in sim.world.entities},
                                      work_target, agent.spec) if work_target is not None else {})
    bounds = sim.config.get("bounds_m", [0., 0., 2000., 2000.])
    # A changed leading point need not recheck every unchanged suffix leg.
    # Exact permissions/spec/bounds keys retain types, order and signed zero;
    # the enclosing preparation supplies navigation-token invalidation. Routes,
    # truth paths and one-off leading legs cannot clear all validated suffixes.
    segment_context = (_exact_key(agent.spec), measure,
                       _exact_key(bounds), _exact_key(interior_by_id))
    segment_cache = _execution_cache(sim.world, "execution_segments")
    for index, (a, b) in enumerate(zip(points, points[1:])):
        road = roads[index] if index < len(roads) else None
        # Only a work route may use these partial edges. Recompute their gaps
        # from physical obstacles so hidden/new blockers still stop the agent.
        open_work_section = any(allowed and _dist(_project(a, c, d)[0], a) < 1e-6
                                and _dist(_project(b, c, d)[0], b) < 1e-6
                                for c, d, allowed in work_roads.get(road, {}).get("segments", []))
        endpoint = index in (0, len(points) - 2)
        key = ('execution_segment', segment_context, _exact_key(a), _exact_key(b),
               _exact_key(road), open_work_section, endpoint)
        checked = segment_cache.get(key)
        if checked is None:
            checked = _execution_segment(sim, agent, a, b, road, open_work_section,
                                         measure, endpoint, buildings, interior_by_id, bounds)
            if len(segment_cache) >= 4096:
                segment_cache.popitem(last=False)
            segment_cache[key] = checked
        else:
            segment_cache.move_to_end(key)
        stop, blockers, segment_length = checked
        if stop is not None:
            if _dist(reached[-1], stop) > 1e-8:
                length += _dist(reached[-1], stop)
                reached.append(list(stop))
            return {"ok": False, "waypoints": reached, "length_m": length, "blocked_by": blockers}
        length += segment_length
        if _dist(reached[-1], b) > 1e-8:
            reached.append(list(b))
    return {"ok": True, "waypoints": reached, "length_m": length, "blocked_by": []}


class ActionModel:
    """Common guards, lifecycle, and a concrete handler for every catalog id."""

    calibration = "authored_uncalibrated"

    def __init__(self, row):
        self.id = row["id"]
        self.name = row["name"]
        self.layer = row["layer"]
        self.capabilities = list(row["capabilities"])
        self.writes = list(row["writekeys"])
        self.writekeys = self.writes
        self.parameters = row["parameters"]
        self.preconditions = row["preconditions"]
        self.effect_text = row["effect"]
        self.capability_condition = row.get("capability_condition", "")
        self.backend_mode = ("abstract_model" if self.id in NETWORK or self.id == "A059" else
                             "compute_model" if self.id in COMPUTE or self.id in {"A038", "A039", "A064", "A066"} else
                             "stub" if self.id in {"A005", "A006", "A007", "A008", "A009", "A011", "A018", "A040"} else
                             "abstract_service" if "专业" in self.layer else "state_model")
        self.scripted = self.backend_mode in ("stub", "abstract_service")

    def guard(self, sim, agent, params):
        caps = set(agent.spec.get("capabilities", []))
        required = set(self.capabilities)
        if self.id == "A005":
            modality = params.get("modality", SCRIPTED_SENSORS.get(params.get("sensor_id"), "rgb"))
            if modality not in SENSOR_CAPS:
                return False, "incompatible"
            params["_sensing_modality"] = modality
            required = {SENSOR_CAPS[modality]}
        if not required <= caps:
            return False, "no_capability"
        params["_guard_capabilities"] = sorted(required)
        mobility = agent.spec.get("mobility")
        if (self.id == "A001" and mobility != "air" or
                self.id == "A002" and mobility not in ("legged", "wheeled", "foot") or
                self.id == "A003" and mobility != "water"):
            return False, "incompatible"
        if self.id == "A021":
            needed = {"air": "C01", "water": "C03", "legged": "C02", "wheeled": "C02", "foot": "C02"}.get(mobility)
            if not needed or needed not in caps:
                return False, "no_capability"
            params["_guard_capabilities"] = sorted(required | {needed})
        ref = params.get("_task_ref", params.get("task_ref"))
        task_id = ref[0] if isinstance(ref, (list, tuple)) and len(ref) >= 2 else params.get("task_id")
        record = agent.tasks.get(task_id) if task_id else None
        if isinstance(ref, (tuple, list)) and len(ref) >= 2 and record and ref[1] != record.get("rev"):
            return False, "stale"
        if record and params.get("rev", params.get("task_rev", record.get("rev"))) != record.get("rev"):
            return False, "stale"
        # Failure/closure of the task does not invalidate its returned evidence.
        # The revision guards above still reject evidence from a replaced copy.
        if record and record.get("status") in TERMINAL_TASKS and self.id not in TASKS and self.id != "A041":
            return False, "stale"
        if params.get("authorized") is False:
            return False, "unauthorized"
        target = _target(params)
        if self.id in TARGET_CLASSES and (not _entity(sim, target) or _cls(sim, target) not in TARGET_CLASSES[self.id]):
            return False, "incompatible"
        if self.id in PHYSICAL and self.id != "A061":
            if sim.world.get(agent.platform_eid, "platform.health_state", "nominal") == "failed":
                return False, "incompatible"
            active = sim.world.get(agent.platform_eid, "platform.current_action_id")
            if active and active != params.get("_instance_id"):
                return False, "busy"
        equipment = _equipment(sim, agent, params)
        if params.get("equipment_id") and equipment is None:
            return False, "incompatible"
        if equipment:
            ef = _fields(sim, equipment)
            if not ef.get("equipment.enabled", True) or ef.get("equipment.health", "ready") == "failed":
                return False, "incompatible"
            authority = _list(ef.get("equipment.operator_authority"))
            if authority and not ({agent.id, agent.kind, "any"} & set(authority)):
                return False, "unauthorized"
            if equipment != agent.platform_eid and not _near(sim, agent, equipment, ef.get('attr.operator_reach_m', 10.)):
                params['_guard_feedback_zh']='操作员未到达所选实际装备的作业范围。'
                return False, 'incompatible'
            materials = _list(ef.get("equipment.compatible_materials"))
            if materials and target and sim.world.get(target, "obstacle.material") not in materials:
                return False, "incompatible"
            if self.id == 'A017':
                rating, mass = ef.get('equipment.rated_load_kg'), sim.world.get(target, 'attr.mass_kg')
                if rating is None or mass is None or mass > rating:
                    return False, 'insufficient'
        if self.id == 'A017' and sim.world.get(target, 'attr.rope_equipment_id') not in (None, equipment):
            return False, 'incompatible'
        if self.id in NETWORK:
            ok, code = sim.network.action_guard(agent, self.id, params)
            if not ok:
                return ok, code
        if self.id in COMPUTE:
            ok, code = sim.compute.action_guard(agent, self.id, params)
            if not ok:
                return ok, code
        if self.id in {"A006", "A007", "A039", "A040", "A041", "A064"} and not _inputs(agent, params):
            return False, "insufficient"
        if self.id == "A005":
            sensors = agent.spec.get("sensors", [])
            modality = params.get("_sensing_modality", "rgb")
            sensor = params.get("sensor_id")
            configured = [s for s in sensors if (isinstance(s, str) and
                          ((s == sensor and SCRIPTED_SENSORS.get(s, s) == modality) if sensor else SCRIPTED_SENSORS.get(s, s) == modality)) or
                          (isinstance(s, dict) and (s.get("id") == sensor if sensor else s.get("modality") == modality) and s.get("enabled", True))]
            if not configured:
                return False, "incompatible"
        if self.id == "A010" and not (_inputs(agent, params) or params.get("evidence_ids")):
            return False, "insufficient"
        if self.id in {"A012", "A016"} and _num(sim.world.get(target, "obstacle.mass_kg")) > _capacity(sim, agent, params):
            return False, "insufficient"
        if self.id == "A008" and sim.world.get(target, "building.interior_mode", "abstract") != "abstract":
            return False, "incompatible"
        if self.id in {"A020", "A023"} and sim.world.get(target, "human.carrier_id") not in (agent.id, agent.platform_eid):
            used = _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
            maximum = _num(agent.spec.get("payload_kg"), 0.)
            mass = _num(sim.world.get(target, "attr.mass_kg"), 75.)
            if used + mass > maximum:
                params["_guard_feedback_zh"] = f"人员装载容量不足：已用{used:g} kg，人员{mass:g} kg，额定{maximum:g} kg。"
                return False, "insufficient"
            if self.id == "A020" and sim.world.get(target, "human.mobility") == "trapped":
                return False, "incompatible"
        if self.id in {"A021", "A022"} and sim.world.get(target, "human.carrier_id") not in (agent.id, agent.platform_eid):
            return False, "incompatible"
        if self.id == "A024":
            state = sim.world.get(target, "item.state", "stock")
            if state not in ("stock", "reserved"):
                return False, "incompatible"
            holder = sim.world.get(target, "item.holder_id")
            if holder not in (None, agent.id, agent.platform_eid) and _cls(sim, holder) != "station":
                return False, "unauthorized"
            mass = _num(sim.world.get(target, "item.mass_kg")) * _num(sim.world.get(target, "item.quantity", 1), 1.)
            if mass + _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg")) > _num(agent.spec.get("payload_kg")):
                return False, "insufficient"
        if self.id in {"A025", "A026"}:
            released_nearby = self.id == "A026" and sim.world.get(target, "item.state") == "released" and _near(sim, agent, target, 15.)
            if sim.world.get(target, "item.holder_id") not in (agent.id, agent.platform_eid) and not released_nearby:
                return False, "unauthorized"
            if sim.world.get(target, "item.state") not in ("attached", "in_transit", "released"):
                return False, "incompatible"
        if self.id in {"A028", "A029", "A030"}:
            station = _station(params)
            if not station or _cls(sim, station) != "station":
                return False, "incompatible"
            if sim.world.get(station, "station.open_state", "open") != "open":
                return False, "unauthorized"
            if _num(sim.world.get(station, "station.occupied")) >= _num(sim.world.get(station, "station.capacity", 1), 1.):
                return False, "busy"
            if self.id == "A029" and _num(sim.world.get(station, "station.power_W")) <= 0:
                return False, "insufficient"
            if self.id == "A029":
                connectors = _list(sim.world.get(station, "station.connector_types"))
                connector = agent.spec.get("connector_type", params.get("connector_type"))
                if connectors and connector and connector not in connectors:
                    return False, "incompatible"
            if self.id == "A030":
                battery = params.get("battery_id") or params.get("new_battery_id")
                if not battery or sim.world.get(battery, "item.kind") != "battery" or _charge_value(sim, battery)[0] <= 0:
                    return False, "insufficient"
                if sim.world.get(battery, "item.holder_id") != station or sim.world.get(battery, "item.state", "stock") != "stock":
                    return False, "unauthorized"
                old_battery = params.get("old_battery_id") or sim.world.get(agent.platform_eid, "attr.battery_item_id")
                if old_battery == battery:
                    return False, "incompatible"
                if old_battery and (sim.world.get(old_battery, "item.kind") != "battery" or sim.world.get(old_battery, "item.holder_id") not in (agent.id, agent.platform_eid)):
                    return False, "unauthorized"
        if self.id == "A064":
            source = params.get("source_version", params.get("model_version", "v1"))
            dest = params.get("target_version", source)
            converters = agent.spec.get("representation_converters", [])
            if source != dest and [source, dest] not in converters and (source, dest) not in converters:
                return False, "incompatible"
        if self.id in MOVEMENT and _goal(sim, agent, params, known=True) is None:
            return False, "incompatible"
        plan = self.plan(sim, agent, params)
        available = _num(sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh"), math.inf)
        if not agent.spec.get("mains_power", False) and available < max(0., _num(plan.get("energy_Wh"))):
            return False, "insufficient"
        budget = params.get("energy_budget_Wh")
        if budget is not None and _num(plan.get("energy_Wh")) > _num(budget):
            return False, "insufficient"
        return True, "none"

    def plan(self, sim, agent, params):
        if self.id in NETWORK:
            return sim.network.action_plan(agent, self.id, params)
        if self.id in COMPUTE:
            return sim.compute.action_plan(agent, self.id, params)
        duration, watts, cpu = LAYER_COSTS.get(self.layer, DEFAULT_COST)
        duration = _duration(params, duration)
        route = None
        if self.id in MOVEMENT:
            start, goal = _own_pose(sim, agent), _goal(sim, agent, params, known=True)
            candidate = _movement_candidate(agent, start, goal, params)
            route = _execution_route(sim, agent, candidate, measure=True)
            length = max(0., _num(route.get("length_m")))
            duration = max(1., length / max(.1, _num(agent.spec.get("speed_mps"), 1.)))
            watts = _num(agent.spec.get("move_power_W"), {"air": 180., "water": 350., "wheeled": 800., "legged": 80., "foot": 30.}.get(agent.spec.get("mobility"), 20.))
            cpu = .05
        elif self.id == "A031":
            goal = _goal(sim, agent, params, known=True)
            if goal is not None:
                recovery_path = sim.world.path(_own_pose(sim, agent), goal, agent.spec)
                duration = 30. + max(0., _num(recovery_path.get("length_m"))) / max(.1, _num(agent.spec.get("speed_mps"), 1.))
            watts, cpu = 100., .05
        elif self.id == "A004":
            watts = 100. if agent.spec.get("mobility") == "air" else 5.
        elif self.id == "A008":
            duration, watts, cpu = _duration(params, 120.), 70., .2
        elif self.id in {"A029", "A030"}:
            current = _num(sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh"))
            capacity = _num(sim.world.get(agent.platform_eid, "platform.battery_capacity_Wh"), agent.spec.get("battery_Wh", 0))
            target = _target_charge(params, capacity)
            if self.id == "A029":
                watts_in = max(.1, _num(sim.world.get(_station(params), "station.power_W"), 0.) * .9)
                duration = max(1., (max(current, target) - current) * 3600. / watts_in)
                if "duration_s" in params or "budget_s" in params:
                    duration = min(duration, _duration(params, duration))
            else:
                duration = max(10., _duration(params, 60.))
            watts, cpu = 0., 0.
        elif self.id == "A006":
            cpu = max(.1, _num(params.get("cpu_s"), 2.))
            duration = max(.1, cpu / max(.1, _num(agent.spec.get("cpu_capacity"), 1.)))
            watts = 12.
        elif self.id == "A066":
            cpu = max(.1, len(agent.belief.entities(copy=False)) * .01)
            duration, watts = max(.1, cpu), 12.
        if "compute" in self.backend_mode or self.id in {"A007", "A040", "A041"}:
            cpu = max(.01, cpu)
        plan = {"duration_s": duration, "energy_Wh": max(0., watts * duration / 3600.), "bytes": 0, "cpu_s": max(0., cpu)}
        if route is not None:
            plan["_route"] = deepcopy(route)
            plan["_candidate"] = deepcopy(candidate)
            plan["_start"] = _own_pose(sim, agent)
            plan["_goal"] = _goal(sim, agent, params, known=True)
        return plan

    def start(self, sim, agent, params):
        params["_started_s"] = sim.t
        writes = []
        target = _target(params)
        if self.id in {'A017', 'A062'} and _equipment(sim, agent, params):
            writes.append((params['equipment_id'], 'equipment.setup_state', 'in_use'))
        if self.id in CLEARING:
            params["_work"] = {"elapsed": 0., "failure_code": "none", "feedback_zh": ""}
        if self.id in MOVEMENT:
            plan = params.get("_plan") or self.plan(sim, agent, params)
            external = bool(params.get("_external_motion"))
            params["_motion"] = {"start": deepcopy(plan.get("_start", _own_pose(sim, agent))),
                                 "goal": deepcopy(plan.get("_goal", _goal(sim, agent, params))),
                                 "route": deepcopy(plan.get("_route", {})), "candidate": deepcopy(plan.get("_candidate", {})),
                                 "elapsed": 0., "distance": 0., "external": external}
            # A PX4-managed vehicle is flown by its controller: the scripted
            # interpolation below must not also move it.
            if not external:
                writes.append((agent.platform_eid, "platform.motion_state", "moving"))
            if self.id == "A021":
                if sim.world.get(target, "human.handover_status", "not_started") == "not_started":
                    writes.append((target, "human.handover_status", "in_transit"))
                destination = _station(params)
                if destination:
                    writes.append((target, "human.destination_id", destination))
            for item in sim.world.get(agent.platform_eid, "platform.payload_item_ids", []) or []:
                if sim.world.get(item, "item.state") == "attached":
                    writes.append((item, "item.state", "in_transit"))
        elif self.id in {"A029", "A030"}:
            station = _station(params)
            params["_charge"] = {"elapsed": 0., "start_Wh": _num(sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh")),
                                  "station": station, "failed": False, "released": False}
            writes += [(station, "station.occupied", int(_num(sim.world.get(station, "station.occupied"))) + 1)]
            if self.id == "A030":
                battery = params.get("battery_id") or params.get("new_battery_id")
                energy, key = _charge_value(sim, battery)
                old_battery = params.get("old_battery_id") or sim.world.get(agent.platform_eid, "attr.battery_item_id")
                if not old_battery:
                    # Materialize the already existing installed battery, not
                    # new stock: its stored energy is still on the platform.
                    old_battery = "installed-battery-" + agent.platform_eid
                    if not _entity(sim, old_battery):
                        writes += [(old_battery, "entity.kind", "item"), (old_battery, "item.kind", "battery"),
                                   (old_battery, "item.quantity", 1), (old_battery, "item.mass_kg", _num(agent.spec.get("battery_mass_kg"), 1.5)),
                                   (old_battery, "item.state", "attached"), (old_battery, "item.holder_id", agent.id),
                                   (old_battery, "attr.charge_Wh", 0.),
                                   (old_battery, "attr.materialized_from_platform", agent.platform_eid)]
                capacity = _num(sim.world.get(agent.platform_eid, "platform.battery_capacity_Wh"), agent.spec.get("battery_Wh", 0.))
                params["_charge"].update({"battery": battery, "battery_Wh": energy, "battery_key": key,
                                          "old_battery": old_battery, "installed_target_Wh": min(capacity, energy)})
                writes.append((battery, "item.state", "reserved"))
        elif self.id == "A024" and sim.world.get(target, "item.state", "stock") == "stock":
            writes.append((target, "item.state", "reserved"))
        elif self.id == "A025":
            if sim.world.get(target, "item.release_state", "locked") in ("locked", "jammed"):
                writes.append((target, "item.release_state", "unlocking"))
        elif self.id == "A061" and not params.get("service_id"):
            eid = target or agent.platform_eid
            state = sim.world.get(eid, "platform.health_state", "nominal")
            if state in ("failed", "degraded") and sim.world.get(eid, "attr.fault_recoverable", False):
                writes.append((eid, "platform.health_state", "recovering"))
        return _out(writes=writes)

    def progress(self, sim, agent, params, elapsed_s, delta_s):
        if self.id in MOVEMENT:
            return self._move_progress(sim, agent, params, elapsed_s)
        if self.id in CLEARING:
            state = params.setdefault("_work", {"elapsed": 0., "failure_code": "none", "feedback_zh": ""})
            delta = max(0., _num(elapsed_s) - state["elapsed"])
            if delta <= 0:
                return _out()
            state["elapsed"] = _num(elapsed_s)
            result = self._clear(sim, agent, dict(params, _work_duration_s=delta), None)
            state["failure_code"], state["feedback_zh"] = result.failure_code, result.feedback_zh
            return result
        if self.id not in {"A029", "A030"}:
            return _out()
        state = params.get("_charge")
        if not state:
            return _out()
        delta = max(0., _num(elapsed_s) - state["elapsed"])
        if delta <= 0 or state["failed"]:
            return _out()
        state["elapsed"] = _num(elapsed_s)
        station = state["station"]
        if not _near(sim, agent, station) or sim.world.get(station, "station.open_state", "open") != "open":
            state["failed"] = True
            return _out()
        current = _num(sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh"))
        maximum = _num(sim.world.get(agent.platform_eid, "platform.battery_capacity_Wh"), agent.spec.get("battery_Wh", 0.))
        requested = _target_charge(params, maximum)
        writes = []
        if self.id == "A029":
            power = max(0., _num(sim.world.get(station, "station.power_W"))) * .9
            if power <= 0:
                state["failed"] = True
                return _out()
            added = min(max(0., requested - current), power * delta / 3600.)
        else:
            duration = max(.1, _num(params.get("_plan", {}).get("duration_s"), 60.))
            fraction = min(1., state["elapsed"] / duration) - min(1., max(0., state["elapsed"] - delta) / duration)
            remaining, key = _charge_value(sim, state["battery"])
            incoming = min(remaining, state["installed_target_Wh"] * fraction)
            outgoing = min(current, state["start_Wh"] * fraction)
            old_charge, old_key = _charge_value(sim, state["old_battery"])
            added = incoming - outgoing
            writes += [(state["battery"], key, max(0., remaining - incoming)),
                       (state["old_battery"], old_key, old_charge + outgoing)]
        if abs(added) > 1e-12:
            writes.append((agent.platform_eid, "platform.battery_remaining_Wh", max(0., min(maximum, current + added))))
        return _out(writes=writes)

    def _move_progress(self, sim, agent, params, elapsed_s):
        motion = params.get("_motion")
        if not motion:
            return _out()
        if motion.get("external"):
            # Follow the measured carrier position without writing PX4's pose.
            return _out(writes=_carried_pose_writes(sim, agent, _own_pose(sim, agent)))
        motion["elapsed"] = max(motion["elapsed"], _num(elapsed_s))
        # Re-evaluate truth at execution time: an aftershock can invalidate a
        # previously passable route. Plans themselves never write knowledge.
        candidate = motion.get("candidate") or known_route(agent, motion["start"], motion["goal"])
        route = _execution_route(sim, agent, candidate)
        points = [_xy(p) for p in route.get("waypoints", [])]
        points = [p for p in points if p is not None] or [motion["start"]]
        if points[0] != motion["start"]:
            points.insert(0, motion["start"])
        speed = max(.1, _num(agent.spec.get("speed_mps"), 1.))
        distance = min(speed * motion["elapsed"], _num(route.get("length_m")))
        if distance + 1e-6 < motion.get("distance", 0.):
            point = motion.get("reached", _own_pose(sim, agent))
        else:
            point = _interpolate(points, distance)
            motion["distance"] = distance
        motion["route"] = deepcopy(route)
        motion["reached"] = point
        writes = [(agent.platform_eid, "entity.pose", point)]
        writes += _carried_pose_writes(sim, agent, point)
        return _out(writes=writes)

    def cancel_delay(self, sim, agent, params):
        """Continuous work stops next tick; an atomic actuator finishes its step."""
        tick = max(.1, _num(sim.config.get("tick_s"), 5.))
        remaining = max(0., _num(params.get("_remaining_s"), _num(params.get("_plan", {}).get("duration_s")) - (sim.t - _num(params.get("_started_s"), sim.t))))
        return max(tick, remaining) if self.id in ATOMIC_STEPS else tick

    def cancel(self, sim, agent, params):
        clock_elapsed = max(0., sim.t - _num(params.get("_started_s"), sim.t))
        elapsed = min(clock_elapsed, max(0., _num(params.get("_cancel_elapsed_s"), clock_elapsed)))
        duration = _num(params.get("_plan", {}).get("duration_s"))
        if self.id in ATOMIC_STEPS and elapsed + 1e-6 >= duration:
            # The actuator physically finished its already-started safe step.
            # The kernel still reports CANCELED, never this helper's status.
            return self.effect(sim, agent, params, sim.rng("action", params.get("_instance_id", "")))
        if self.id in CLEARING:
            state = params.get("_work", {})
            actual = min(duration, max(elapsed, _num(params.get("_cancel_elapsed_s"))))
            if actual > _num(state.get("elapsed")):
                return self.progress(sim, agent, params, actual, actual - _num(state.get("elapsed")))
        if self.id == "A030" and elapsed + 1e-6 >= duration:
            return self._effect_A030(sim, agent, params, None)
        writes = []
        if self.id in {'A017', 'A062'} and _equipment(sim, agent, params):
            writes.append((params['equipment_id'], 'equipment.setup_state', 'ready' if self.id=='A017' else 'stowed'))
        if self.id in MOVEMENT:
            writes.append((agent.platform_eid, "platform.motion_state", "idle"))
        if self.id in {"A029", "A030"}:
            writes += self._release_charge(sim, params)
        if self.id == "A024":
            target = _target(params)
            if sim.world.get(target, "item.state") == "reserved":
                # No reserved->stock transition is claimed: preserve a truthful
                # reservation for a later retry rather than inventing rollback.
                pass
        return _out(writes=writes, text="动作在安全边界停止，保留已耗资源与已完成范围。")

    def _release_charge(self, sim, params):
        state = params.get("_charge")
        if not state or state["released"]:
            return []
        state["released"] = True
        station = state["station"]
        return [(station, "station.occupied", max(0, int(_num(sim.world.get(station, "station.occupied"))) - 1))]

    def effect(self, sim, agent, params, rng):
        if self.id in TASKS:
            return agent.apply_task_action(sim, self.id, params)
        if self.id in NETWORK:
            if self.id in {"A035", "A036"} and getattr(agent, "comm_budget_gate", None) is not None:
                receivers = _list(params.get("receivers", params.get("receiver_ids", params.get("receiver", params.get("target")))))
                original = sim.network.messages.get(params.get("message_id") if self.id == "A035" else params.get("retry_of"))
                payload = original.payload if original else params.get("payload", {})
                size = original.size_bytes if original else int(params.get("size_bytes", 0))
                if not agent._admit_send(sim, receivers, payload, size):
                    return _fail("model", "本地通信预算拒绝发送，消息未进入网络队列。")
            return sim.network.action_effect(agent, self.id, params, rng)
        if self.id in COMPUTE:
            return sim.compute.action_effect(agent, self.id, params, rng)
        handler = getattr(self, "_effect_" + self.id)
        result = handler(sim, agent, params, rng)
        equipment = _equipment(sim, agent, params)
        if equipment and self.id in {'A017', 'A062'}:
            result.truth_writes.append((equipment, 'equipment.setup_state', 'ready' if self.id=='A017' else 'stowed'))
        if equipment and self.id in {'A017', 'A062', 'A028'}:
            values = {key: value for eid, key, value in result.truth_writes if eid==equipment and key.startswith('equipment.')}
            if values:
                result.observations.append(_observation(sim, agent, params, [equipment], values,
                    'equipment_feedback', 1., 'actual_equipment_accounting_v1'))
        if self.id=='A062':
            target=_target(params)
            values={key:value for eid,key,value in result.truth_writes if eid==target and key.startswith('hazard.')}
            if values:
                result.observations.append(_observation(sim,agent,params,[target],values,
                    'suppression_feedback',1.,'actual_suppression_v1'))
        return result

    def _effect_A001(self, sim, agent, params, rng):
        target = _target(params)
        motion = params.get("_motion")
        if motion and motion.get("external"):
            # PX4 flew this leg and the kernel was told it arrived. Never
            # overwrite the native pose; still sync carried people and settle
            # the task-visible status writes.
            point = _own_pose(sim, agent)
            writes = [(agent.platform_eid, "platform.motion_state", "idle")]
            writes += _carried_pose_writes(sim, agent, point)
            if self.id == "A021":
                writes.append((target, "human.handover_status", "arrived"))
            return _out(writes=writes, text="外部飞控已到达目标位置；到达不等于移交确认。" if self.id == "A021"
                        else "已由外部飞控到达目标位置。")
        if not motion:
            # Compatibility for a kernel without lifecycle hooks: completion can
            # move, but gradual progress/cancellation requires the documented hook.
            plan = params.get("_plan") or self.plan(sim, agent, params)
            motion = {"start": plan.get("_start", _own_pose(sim, agent)), "goal": plan.get("_goal", _goal(sim, agent, params)), "elapsed": plan["duration_s"]}
            params["_motion"] = motion
        progress = self._move_progress(sim, agent, params, motion.get("elapsed", 0.))
        route = motion["route"]
        point = motion["reached"]
        reached = _dist(point, motion["goal"]) <= max(.5, _num(params.get("tolerance_m"), 2.))
        if not route.get("ok"):
            observations = []
            for eid in _list(route.get("blocked_by")):
                if isinstance(eid, dict):
                    eid = eid.get("id", eid.get("road_id"))
                if not isinstance(eid, str) or not _entity(sim, eid):
                    continue
                if _cls(sim, eid) == "building":
                    values = {key: value for key in ("entity.footprint", "building.entrance_ids", *BUILDING_NAVIGATION_FIELDS)
                              if (value := sim.world.get(eid, key)) is not None}
                    values.update({"entity.kind": "building", "entity.geometry": _entity(sim, eid)["geometry"]})
                elif _cls(sim, eid) == "obstacle":
                    keys = ("obstacle.kind", "obstacle.state", "obstacle.volume_m3", "obstacle.height_m", "obstacle.blocks_ids")
                    values = {key: sim.world.get(eid, key, "unknown") for key in keys}
                else:
                    keys = ("road.entry_state", "road.clear_width_m", "road.step_height_m")
                    values = {key: sim.world.get(eid, key, "unknown") for key in keys}
                    # An inundation stop must report the measured cause through
                    # A041, or A066 keeps choosing the same apparently open road.
                    # Other blockage feedback retains its existing evidence.
                    depth = sim.world.get(eid, "road.water_depth_m")
                    if _cls(sim, eid) == "road" and _num(depth) > _num(agent.spec.get("water_tolerance_m")):
                        values["road.water_depth_m"] = depth
                values["entity.pose"] = _pose(sim, eid)
                observations.append(_observation(sim, agent, params, [eid], values,
                                                 model="scripted_motion_blockage_v1", suffix=eid))
            progress.truth_writes.append((agent.platform_eid, "platform.motion_state", "blocked"))
            return _fail("blocked", "行进受阻，停在实际到达点；未知障碍由本次执行发现。", progress.truth_writes, observations)
        if not reached:
            progress.truth_writes.append((agent.platform_eid, "platform.motion_state", "holding"))
            return _fail("timeout", "行进预算不足，尚未到达目标。", progress.truth_writes, partial=True)
        progress.truth_writes.append((agent.platform_eid, "platform.motion_state", "idle"))
        if self.id == "A021":
            progress.truth_writes.append((target, "human.handover_status", "arrived"))
        return _out(writes=progress.truth_writes, text="到达目标位置；到达不等于移交确认。" if self.id == "A021" else "已按实际通行条件到达目标位置。")

    _effect_A002 = _effect_A001
    _effect_A003 = _effect_A001
    _effect_A021 = _effect_A001

    def _effect_A004(self, sim, agent, params, rng):
        for eid in sim.world.entities:
            zone = sim.world.get(eid, "hazard.exclusion_zone")
            if zone is True:
                zone = sim.world.get(eid, "hazard.footprint") or _part(_entity(sim, eid), "geometry")
            if zone and _point_in(_own_pose(sim, agent), zone):
                return _fail("hazard", "当前位置位于警戒区，无法继续安全驻留。")
        motion = sim.world.get(agent.platform_eid, "platform.motion_state", "idle")
        after = "holding" if motion in ("moving", "holding") else motion
        return _out(local=[("platform.motion_state", after)], text="在声明时段驻留并计入能源消耗。")

    def _effect_A005(self, sim, agent, params, rng):
        radius = max(1., min(_num(params.get("range_m"), 250.), _num(agent.spec.get("sensor_range_m"), 250.)))
        miss = _sensing_rate(sim, "sensing_miss_rate", .08)
        observations = []
        target = _target(params)
        camera_replace = agent.spec.get('camera_replace_scripted', False)
        for eid in sorted(sim.world.entities):
            if not _near(sim, agent, eid, radius):
                continue
            visibility = getattr(sim, "phase1_sensor_visibility", None)
            if visibility is not None and not visibility(agent, eid):
                continue
            cls = _cls(sim, eid)
            if camera_replace and cls in {'building', 'obstacle', 'human'}:
                # These external RGB observations arrive separately from the
                # committed camera image. Other modalities keep their model.
                continue
            if cls == "building":
                keys = ("building.visual_damage",)
            elif cls == "obstacle":
                keys = ("obstacle.kind", "obstacle.state", "obstacle.volume_m3", "obstacle.height_m")
            elif cls == "human":
                context = sim.world.get(eid, "human.location_context")
                if context not in ("outside", "outdoors", "roof", "water", "safe_point") or _exogenous_rng(sim, agent, rng, "sensing", "human_visibility", eid).random() < miss:
                    continue
                keys = ("human.location_context", "human.distress_signal")
            elif cls == "cue":
                if _exogenous_rng(sim, agent, rng, "sensing", "cue_detection", eid).random() > _clamp(sim.world.get(eid, "cue.detectability", .8)):
                    continue
                keys = ("cue.kind", "cue.location", "cue.signal_state")
            elif cls == 'hazard' and (params.get('_sensing_modality')=='gas' or agent.spec.get('sense_hazards')):
                keys = ('hazard.kind', 'hazard.active', 'hazard.footprint', 'hazard.fire_intensity',
                        'hazard.smoke_density', 'hazard.wind_mps', 'hazard.exclusion_zone')
                if params.get('_sensing_modality') == 'gas':
                    keys += ('hazard.gas_level',)
            elif cls == 'facility' and sim.world.get(eid, 'equipment.kind') is not None:
                keys = tuple(key for key in _fields(sim, eid) if key.startswith('equipment.'))
            else:
                continue
            values = {key: ("unknown" if _exogenous_rng(sim, agent, rng, "sensing", "field_dropout:" + key, eid).random() < miss else deepcopy(sim.world.get(eid, key, "unknown"))) for key in keys}
            values["entity.pose"] = _pose(sim, eid)
            observations.append(_observation(sim, agent, params, [eid], values, confidence=1. - miss, suffix=eid))
        return _out(obs=observations, text=("外部建筑、人员及障碍感知由相机图像提供；其余模态仍为脚本采集。"
                    if camera_replace else "脚本采集（非视觉）：仅报告可见外部信息；未发现不代表无人。"))

    def _effect_A006(self, sim, agent, params, rng):
        inputs = _inputs(agent, params)
        if not inputs:
            return _fail("model", "脚本推断（非视觉）：没有本地可读输入。")
        confusion = _sensing_rate(sim, "confusion_rate", .08)
        observations = []
        camera_evidence = []
        for i, source in enumerate(inputs):
            if source.get('source') == 'camera_image':
                # The perception plugin has already interpreted these pixels.
                # Keep its local evidence and provenance, without applying the
                # authored confusion model to an actual image detection.
                agent.belief.add_evidence(sim, source)
                camera_evidence.append(source['id'])
                observations.append(source)
                continue
            values = deepcopy(source["values"])
            for key, value in list(values.items()):
                keyed_rng = _exogenous_rng(sim, agent, rng, "inference", "field_confusion:" + key, tuple(source.get("about", [])))
                if key == "building.visual_damage" and keyed_rng.random() < confusion:
                    values[key] = keyed_rng.choice([v for v in ("none", "minor", "major", "destroyed", "unknown") if v != value])
                elif key.startswith("human.") and keyed_rng.random() < confusion:
                    values[key] = "unknown"
            values["observation.parent_evidence_ids"] = [source.get("id")]
            values["observation.detections"] = {k: v for k, v in values.items() if k.startswith(("building.", "human.", "obstacle.", "cue."))}
            observations.append(_observation(sim, agent, params, source.get("about", []), values,
                                             "authored_inference", min(_num(source.get("confidence"), .8), 1. - confusion),
                                             "scripted_inference_v1", str(i)))
        return _out(obs=observations, evidence=[obs["id"] if isinstance(obs, dict) else obs.id for obs in observations],
                    text=("保留图像感知插件的检测证据；仅对其余输入执行脚本推断。" if camera_evidence else
                          "脚本推断（非视觉）：基于本地观测生成可错误、可不确定的检测。"))

    def _effect_A007(self, sim, agent, params, rng):
        target = _target(params)
        sources = _inputs(agent, params)
        point = None
        evidence = []
        for source in sources:
            if not target or target in source.get("about", []):
                point = _xy(source["values"].get("entity.pose")) or _xy(source["values"].get("belief.estimated_pose"))
                evidence.append(source.get("id"))
                if point is not None:
                    target = target or next(iter(source.get("about", [])), None)
                    break
        point = point or _known_pose(agent, target)
        if point is None or not target:
            return _fail("model", "缺少观测位置或标定信息，不能查询目标真值代替定位。")
        error = .5 if params.get("ranging", False) else 3.
        if params.get("ranging") and "C09" not in agent.spec.get("capabilities", []):
            return _fail("sensor", "测距定位缺少实际测距能力。")
        estimate = [v + _exogenous_rng(sim, agent, rng, "localization", "position_error", target, axis).uniform(-error, error) for axis, v in enumerate(point)]
        agent.belief.put(sim, target, "belief.estimated_pose", estimate, .85 if error < 1 else .65, "inference", evidence_ids=evidence)
        obs = _observation(sim, agent, params, [target], {"belief.estimated_pose": estimate, "attr.position_error_m": error}, "localization", .65, "scripted_localization_v1")
        return _out(obs=[obs], text="由本地观测生成位置估计并保留方法误差。")

    def _effect_A008(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, _num(params.get("entry_tolerance_m"), 60.)):
            return _fail("blocked", "未到达建筑合法入口，不能执行抽象室内搜索。")
        entrances = _list(sim.world.get(target, "building.entrance_ids"))
        if entrances and all(sim.world.get(eid, "road.entry_state", "open") != "open" for eid in entrances):
            return _fail("blocked", "建筑入口仍受阻，搜索尚未开始。")
        difficulty = sim.world.get(target, "building.search_difficulty", 1.)
        if isinstance(difficulty, dict):
            difficulty = difficulty.get("multiplier", difficulty.get("difficulty", difficulty.get("work_factor", 1.)))
        difficulty = max(.2, _num(difficulty, 1.))
        area = max(20., _num(sim.world.get(target, "building.interior_area_m2"), 200.))
        budget = _num(params.get("_plan", {}).get("duration_s"), _duration(params, 120.))
        coverage = min(1., budget * _num(agent.spec.get("search_rate_m2_s"), 2.) / (area * difficulty))
        before = _clamp(sim.world.get(target, "building.search_coverage_truth", 0.))
        cumulative = min(1., before + (1. - before) * coverage)
        detect_rate = .98 * (1. - _sensing_rate(sim, "sensing_miss_rate", .08))
        occupants = _list(sim.world.get(target, "building.occupant_ids"))
        detected, missed = [], []
        observations = []
        for eid in occupants:
            if not _entity(sim, eid) or sim.world.get(eid, "human.location_context") not in ("interior", "rubble"):
                continue
            if _exogenous_rng(sim, agent, rng, "search", "occupant_detection", eid).random() < detect_rate * coverage:
                detected.append(eid)
                values = {"entity.pose": _pose(sim, eid), "human.mobility": sim.world.get(eid, "human.mobility", "unknown"),
                          "human.location_context": sim.world.get(eid, "human.location_context", "interior")}
                observations.append(_observation(sim, agent, params, [eid], values, "interior_search_detection", detect_rate, "scripted_search_v1", eid))
            else:
                missed.append(eid)
        previous_local = 0.
        for evidence in getattr(agent.belief, "evidence", {}).values():
            evidence = _obs_dict(evidence)
            if isinstance(evidence, dict) and target in evidence.get("about", []) and evidence.get("kind") == "interior_search_report":
                previous_local = max(previous_local, _clamp(evidence.get("values", {}).get("observation.coverage_estimate")))
        local_coverage = min(1., previous_local + (1. - previous_local) * coverage)
        state = "complete" if local_coverage >= .999 else "partial" if coverage > 0 else "inconclusive"
        report = {"belief.search_status": state, "observation.coverage_estimate": local_coverage,
                  "belief.human_presence": "present" if detected else "unknown", "attr.detected_ids": detected}
        observations.insert(0, _observation(sim, agent, params, [target], report, "interior_search_report", .85, "scripted_search_v1"))
        # Miss identities belong exclusively to truth/audit, never to the report.
        audit = list(sim.world.get(target, "attr.search_audit", []) or [])
        audit.append({"t": sim.t, "by": agent.id, "coverage": coverage, "detected": detected, "missed": missed, "model": "scripted_search_v1"})
        return _out("succeeded" if state == "complete" else "partial", writes=[(target, "building.search_coverage_truth", cumulative), (target, "attr.search_audit", audit)], obs=observations,
                    text="抽象室内搜索产生本地报告并保留漏检；没有改变任何人员救援状态。")

    def _effect_A009(self, sim, agent, params, rng):
        target = _target(params)
        if _cls(sim, target) != "human" or not _near(sim, agent, target, 35.):
            return _fail("sensor", "接触范围内没有可确认回应；未回应不等于无人。")
        if not sim.world.get(target, "human.alive", True) or not sim.world.get(target, "attr.responsive", True):
            return _fail("timeout", "未获得人员回应，不能据此推断无人或完成交付。")
        # Context is reported by this successful, in-range contact, not inferred
        # from the camera candidate. Water recovery requires that local fact.
        # The contact pose is the physical measurement this response delivers;
        # identity association consumes it instead of re-reading world truth.
        values = {"human.needs": sim.world.get(target, "human.needs", []),
                  "human.mobility": sim.world.get(target, "human.mobility", "unknown"),
                  "human.location_context": sim.world.get(target, "human.location_context", "unknown")}
        contact_pose = _pose(sim, target)
        if contact_pose is not None:
            values["entity.pose"] = contact_pose
        obs = _observation(sim, agent, params, [target], values, "human_response", .9)
        return _out(writes=[(target, "human.support_received", _add_support(sim, target, "contacted"))], obs=[obs], text="已确认需求；接触不等于治疗、领取或移交。")

    def _effect_A010(self, sim, agent, params, rng):
        target = _target(params)
        ids = [row.get("id") for row in _inputs(agent, params)]
        ids += [eid for eid in _list(params.get("evidence_ids")) if eid in getattr(agent.belief, "evidence", {})]
        if not ids:
            return _fail("model", "没有本地执行证据，不能登记已完成作业。")
        key = 'attr.worksite_record' if _cls(sim, target)=='hazard' else 'building.worksite_record'
        record = deepcopy(sim.world.get(target, key, []) or [])
        if not isinstance(record, list):
            record = [record]
        record.append({"agent": agent.id, "t": sim.t, "evidence_ids": list(dict.fromkeys(ids)),
                       "scope": params.get("scope", "evidence_only")})
        return _out(writes=[(target, key, record)], evidence=ids, text="仅登记证据覆盖的实际作业范围。")

    def _effect_A011(self, sim, agent, params, rng):
        target = _target(params)
        building = target if _cls(sim, target) == "building" else None
        if building:
            entrances = [eid for eid in _list(sim.world.get(building, "building.entrance_ids")) if _cls(sim, eid) == "road"]
            target = min(entrances, key=lambda eid: _dist(_own_pose(sim, agent), _pose(sim, eid)), default=None)
        if not target or not _near(sim, agent, target, 100.):
            return _fail("sensor", "入口不在局部测量范围内。")
        passed, reason = sim.world.passable(target, agent.spec)
        values = {"passable": bool(passed), "reason": reason, "clear_width_m": sim.world.get(target, "road.clear_width_m", "unknown"), "entry_state": sim.world.get(target, "road.entry_state", "unknown")}
        obs = _observation(sim, agent, params, [target], {"belief.estimated_access": values}, "access_measurement", .85)
        agent.belief.add_evidence(sim, obs)
        agent.belief.put(sim, target, "belief.estimated_access", values, .85, "observation", evidence_ids=[obs.id])
        if building:
            agent.belief.put(sim, building, "belief.estimated_access", {"entry_id": target, **values}, .85, "observation", evidence_ids=[obs.id])
        return _out(obs=[obs], text="生成局部通路测量，不提供全局可达真值。")

    def _clear(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 35.):
            return _fail("blocked", "尚未到达障碍作业位置。")
        if sim.world.get(target, "obstacle.state") in ("cleared", "moved") and _num(sim.world.get(target, "obstacle.work_remaining")) <= 0:
            return _out(obs=self._clearance_observations(sim, agent, params),
                        text="障碍实际作业量已清零，不重复结算；返回本地复测结果。")
        if sim.world.get(target, "obstacle.stability", "stable") == "unstable":
            return _fail("hazard", "障碍支撑不稳定，须先完成支护。")
        remaining = max(0., _num(sim.world.get(target, "obstacle.work_remaining"), _num(sim.world.get(target, "obstacle.volume_m3"), 1.) * 60.))
        if "_work" in params and "_work_duration_s" not in params:
            state = params["_work"]
            if state.get("failure_code", "none") != "none":
                return _fail(state["failure_code"], state.get("feedback_zh", "作业条件未满足。"))
            return _out("succeeded" if remaining <= 1e-9 else "partial",
                        obs=self._clearance_observations(sim, agent, params),
                        text="清障按实际经过时间推进；终结不重复结算作业量。")
        duration = _num(params.get("_work_duration_s"), _num(params.get("_plan", {}).get("duration_s"), _duration(params, 90.)))
        rate = _num(agent.spec.get("clear_work_rate"), {"A012": 1., "A013": 5., "A016": 3.}[self.id])
        after = max(0., remaining - max(0., rate) * duration)
        state = "cleared" if after <= 1e-9 else "partly_cleared"
        writes = [(target, "obstacle.work_remaining", after), (target, "obstacle.state", state)]
        volume = _num(sim.world.get(target, "obstacle.volume_m3"), 1.)
        fraction = after / remaining if remaining > 0 else 0.
        if self.id == "A013":
            writes.append((target, "obstacle.volume_m3", volume * fraction))
        if self.id in {"A012", "A016"}:
            disposal = _xy(params.get("disposal_location", params.get("drop_point")))
            if disposal is not None:
                writes.append((target, "obstacle.disposal_location", disposal))
                if self.id == "A016" and state == "cleared":
                    writes.append((target, "entity.pose", disposal))
        writes.extend(self._road_updates(sim, target, state, after))
        return _out("succeeded" if state == "cleared" else "partial", writes=writes,
                    obs=self._clearance_observations(sim, agent, params, writes),
                    text="按作业量推进清障，并重新派生道路净空和通行条件。")

    def _clearance_observations(self, sim, agent, params, writes=()):
        # A local post-work scripted measurement, returned through A041 and
        # ordinary reports. Never broadcast road truth directly to agents.
        target = _target(params)
        projected = {(eid, key): value for eid, key, value in writes}
        rows = []
        for eid in [target] + list(sim.world.get(target, "obstacle.blocks_ids", [])):
            keys = (("obstacle.state", "obstacle.work_remaining", "obstacle.blocks_ids") if eid == target else
                    ("road.entry_state", "road.clear_width_m", "road.step_height_m"))
            values = {key: projected[(eid, key)] if (eid, key) in projected else sim.world.get(eid, key) for key in keys}
            values["entity.pose"] = _pose(sim, eid)
            rows.append(_observation(sim, agent, params, [eid], values, "clearance_measurement", .95,
                                     "scripted_clearance_measurement_v1", suffix=eid))
        return rows

    def _road_updates(self, sim, target, state, after):
        """D01 derivatives match remaining/initial work, not repeated fractions."""
        writes = []
        for eid in sorted(sim.world.entities):
            if _cls(sim, eid) != "road":
                continue
            ids = _list(sim.world.get(eid, "road.obstacle_ids"))
            if target not in ids and eid not in _list(sim.world.get(target, "obstacle.blocks_ids")):
                continue
            width = _num(sim.world.get(eid, "road.width_m"), 3.)
            clear = width
            step = _num(sim.world.get(eid, "attr.base_step_height_m"))
            for obstacle in ids:
                current_state = state if obstacle == target else sim.world.get(obstacle, "obstacle.state")
                if current_state not in ("present", "partly_cleared"):
                    continue
                previous = _num(sim.world.get(obstacle, "obstacle.work_remaining"), 1.)
                initial = max(1e-9, _num(sim.world.get(obstacle, "attr.initial_work"), previous))
                fraction = _clamp((after if obstacle == target else previous) / initial)
                occupied = _num(sim.world.get(obstacle, "attr.blocked_width_m"), width) * fraction
                clear = min(clear, max(0., width - occupied))
                step = max(step, _num(sim.world.get(obstacle, "obstacle.height_m")) * fraction)
            old = sim.world.get(eid, "road.entry_state", "blocked")
            entry = old if old in ("closed", "unsafe") else "open" if clear >= min(2.2, width) and step < 1e-9 else "blocked"
            writes += [(eid, "road.clear_width_m", round(clear, 6)), (eid, "road.step_height_m", round(step, 6)), (eid, "road.entry_state", entry)]
        return writes

    _effect_A012 = _clear
    _effect_A013 = _clear
    _effect_A016 = _clear

    def _effect_A014(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 40.):
            return _fail("blocked", "未到达可安全破拆的位置。")
        building = target if _cls(sim, target) == "building" else params.get("building_id")
        roads = _list(sim.world.get(building, "building.entrance_ids")) if building else [target]
        if params.get("road_id"):
            roads = [params["road_id"]]
        if not roads:
            return _fail("blocked", "未指定合法入口，不能凭空建立通路。")
        writes = [(road, "road.entry_state", "open") for road in roads if _entity(sim, road)]
        if building:
            damage = deepcopy(sim.world.get(building, "building.wall_damage", {}))
            if not isinstance(damage, dict):
                damage = {"previous": damage}
            damage["breach"] = {"entry_ids": roads, "by": agent.id, "t": sim.t}
            writes.append((building, "building.wall_damage", damage))
        return _out(writes=writes, text="按专业模型建立指定入口并记录墙体破拆。")

    def _effect_A015(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 35.):
            return _fail("blocked", "未到达支护施工位置。")
        writes = []
        if _cls(sim, target) == "obstacle":
            writes.append((target, "obstacle.stability", "secured"))
        else:
            # A local shore does not certify that an entire building is safe.
            record = deepcopy(sim.world.get(target, "building.worksite_record", []) or [])
            if not isinstance(record, list):
                record = [record]
            record.append({"kind": "local_shoring", "scope": params.get("scope", "entry"), "by": agent.id, "t": sim.t})
            writes.append((target, "building.worksite_record", record))
            hazards = list(sim.world.get(target, "building.remaining_hazard", []) or [])
            writes.append((target, "building.remaining_hazard", hazards))
        return _out(writes=writes, text="已完成局部支护；未宣称整栋建筑绝对安全。")

    def _effect_A017(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 25.):
            return _fail("blocked", "未到达受困者作业位置。")
        remaining = max(0., _num(sim.world.get(target, "human.buriedness")))
        capacity = _capacity(sim, agent, params)
        duration = _num(params.get("_plan", {}).get("duration_s"), 90.)
        work = max(0., _num(agent.spec.get("extrication_rate"), 1.)) * duration
        if capacity <= 0 or work <= 0:
            return _fail("actuator", "工具无法执行当前解救作业。")
        after = max(0., remaining - min(capacity, work))
        writes = [(target, "human.buriedness", after)]
        if after > 0:
            return _fail("actuator", "仅完成部分解困，仍有受困作业量。", writes, partial=True)
        mobility = sim.world.get(target, "human.mobility", "trapped")
        if mobility == "trapped":
            injury = sim.world.get(target, "human.injury", "unknown")
            mobility = "immobile" if injury in ("severe", "critical", "major") else "limited"
            writes.append((target, "human.mobility", mobility))
        writes.append((target, "human.support_received", _add_support(sim, target, "extricated")))
        return _out(writes=writes, text="已解除压埋；未自动治疗、装车或宣称获救。")

    def _effect_A018(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 15.):
            return _fail("blocked", "授权评估者尚未接近人员。")
        values = {key: sim.world.get(target, key, "unknown") for key in ("human.injury", "human.mobility", "human.needs", "human.alive")}
        obs = _observation(sim, agent, params, [target], values, "authorized_casualty_assessment", .95, "abstract_assessment_v1")
        return _out(writes=[(target, "human.last_assessed_s", sim.t)], obs=[obs], text="授权专业服务产生抽象伤情评估，非外观诊断。")

    def _effect_A019(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 15.):
            return _fail("blocked", "治疗者尚未接近人员。")
        kit = params.get("medical_item_id") or next(_items(sim, agent, ("medical_pack",), near=True), None)
        if not kit or kit not in set(_items(sim, agent, ("medical_pack",), near=True)):
            return _fail("actuator", "缺少实际可用且能够取得的医疗耗材，不能把投包当治疗。")
        if not sim.world.get(target, "human.alive", True):
            return _fail("model", "声明的援助模型不具备复生效果。")
        health = _clamp(sim.world.get(target, "human.health_value", .5))
        writes = [(kit, "item.quantity", max(0., _num(sim.world.get(kit, "item.quantity")) - 1)),
                  (target, "human.support_received", _add_support(sim, target, "treated")),
                  (target, "human.health_value", min(1., health + .15))]
        if sim.world.get(target, "human.mobility") == "limited" and sim.world.get(target, "human.injury") in ("none", "minor", "mild"):
            writes.append((target, "human.mobility", "mobile"))
        return _out(writes=writes, text="消耗实际医疗耗材并按显式抽象模型记录治疗，不保证痊愈。")

    def _effect_A020(self, sim, agent, params, rng):
        target = _target(params)
        person = _pose(sim, target)
        pickup = sim.world.navigation_goal(person, agent.spec)
        if pickup is None:
            return _fail("blocked", "建筑占地与现有道路几何未提供可到达的外侧装载点。")
        interior_pickup = agent.spec.get("mobility") == "wheeled" and _dist(person, pickup) > 1e-6
        if interior_pickup and _dist(_own_pose(sim, agent), pickup) > 2.:
            return _fail("blocked", "轮式载具须先到达建筑外侧入口装载点。")
        if not _near(sim, agent, target, 15.):
            if interior_pickup:
                return _fail("blocked", f"载具已在建筑外侧，人员仍距装载点{_dist(_own_pose(sim, agent), person):.1f} m，"
                             "超过既有15 m装载范围；须由具备人员承载能力和足够载荷的队伍实际搬运，解困不等于出楼。")
            return _fail("blocked", "载具尚未到达人员装载位置。")
        if sim.world.get(target, "human.mobility") == "trapped" or sim.world.get(target, "human.carrier_id") not in (None, agent.id, agent.platform_eid):
            return _fail("actuator", "人员仍受困或已由其他载具承担。")
        if sim.world.get(target, "human.carrier_id") in (agent.id, agent.platform_eid):
            return _out(text="人员已由本载具装载，不重复增加载荷。")
        if sim.world.get(target, "human.handover_status") == "acknowledged":
            # A stale loading task must not take a person back out of a receiver's custody.
            return _fail("actuator", "人员已被接收点确认接收，不再装载。")
        used =_num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
        mass = _num(sim.world.get(target, "attr.mass_kg"), 75.)
        if used + mass > _num(agent.spec.get("payload_kg")):
            return _fail("actuator", "装载时载荷容量不足。")
        return _out(writes=[(target, "human.carrier_id", agent.id), (target, "human.location_context", "on_carrier"),
                            (agent.platform_eid, "platform.payload_used_kg", used + mass)], text="人员已装载；载荷同步增加，尚未完成移交。")

    def _effect_A022(self, sim, agent, params, rng):
        target, station = _target(params), _station(params)
        if not station or _cls(sim, station) != "station" or sim.world.get(target, "human.handover_status") != "arrived":
            return _fail("actuator", "尚未到达合法接收站，不能登记移交。")
        if sim.world.get(station, "station.kind") not in RECEIVING_STATION_KINDS:
            return _fail("incompatible", "该站点不是人员接收点（如补给/充电站），不能登记救援移交。")
        accepted = {str(need) for need in _list(sim.world.get(station, "station.accepted_need_types"))}
        needs = {str(need) for need in _list(sim.world.get(target, "human.needs"))}
        if accepted:
            # The station's declared intake must cover what this person needs from
            # a receiver; a water-only point cannot take a transport/medical case.
            for need, aliases in (("transport", ("transport", "handover")),
                                  ("medical_supplies", ("medical_supplies", "medical"))):
                if need in needs and not accepted & set(aliases):
                    return _fail("incompatible", f"接收站兼容性不足：人员需求含{need}，"
                                 f"该站仅可接收{','.join(sorted(accepted))}，不能登记移交。")
        if not _near(sim, agent, station) or _dist(_pose(sim, target), _pose(sim, station)) > 20.:
            return _fail("blocked", "人员或载具尚未到达接收点。")
        occupied = int(_num(sim.world.get(station, "station.occupied")))
        confirmed = (sim.world.get(station, "station.open_state") == "open"
                     and sim.world.get(station, "attr.receiver_confirmation") is True)
        if params.get("receiver_confirmed") is False:
            confirmed = False
        capacity = _num(sim.world.get(station, "station.capacity", 0))
        if occupied >= capacity:
            return _fail("actuator", f"接收站容量不足：已接收{occupied:g}人，容量{capacity:g}人；人员仍由原载具承载，未完成移交。")
        if not confirmed:
            return _fail("timeout", "接收方未确认；到达仍不等于移交。")
        used = _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
        writes = [(target, "human.handover_status", "acknowledged"), (target, "human.carrier_id", None),
                  (target, "human.destination_id", station), (target, "human.location_context", "safe_point"),
                  (station, "station.occupied", occupied + 1),
                  (agent.platform_eid, "platform.payload_used_kg", max(0., used - _num(sim.world.get(target, "attr.mass_kg"), 75.)))]
        if agent.spec.get('mobility') == 'water' and sim.world.get(station, 'station.access_region'):
            # The staffed shore receiver takes custody on its dry side. The
            # existing 20 m range/confirmation checks above still govern this
            # abstract transfer; the boat itself stays at the water berth.
            writes.append((target, 'human.pose', list(_pose(sim, station))))
        return _out(writes=writes, text="接收站确认且容量足够，人员移交现已确认。")

    def _effect_A023(self, sim, agent, params, rng):
        target = _target(params)
        if sim.world.get(target, "human.location_context") != "water":
            return _fail("actuator", "目标不在水中，不能虚报水域救助。")
        if not _near(sim, agent, target, 25.):
            return _fail("blocked", "水域救助设备尚未接近目标。")
        if _num(sim.world.get(target, "attr.flow_speed_mps")) > _num(agent.spec.get("max_rescue_flow_mps"), 2.):
            return _fail("hazard", "海况或流速超出声明救助能力。")
        result = self._effect_A020(sim, agent, params, rng)
        if result.status == "succeeded":
            result.truth_writes.append((target, "human.support_received", _add_support(sim, target, "recovered_from_water")))
            result.feedback_zh = "人员已脱离水体并接收上载具，仍需后续移交。"
        return result

    def _effect_A024(self, sim, agent, params, rng):
        target = _target(params)
        holder = sim.world.get(target, "item.holder_id")
        point = _pose(sim, holder) if holder else _pose(sim, target)
        if _dist(_own_pose(sim, agent), point) > 20.:
            return _fail("blocked", "实物不在挂载位置，不能远程取得物资。")
        state = sim.world.get(target, "item.state", "stock")
        if state not in ("stock", "reserved"):
            return _fail("actuator", "物资已被占有或状态发生变化。")
        if sim.world.get(target, "item.condition", "usable") != "usable" or _num(sim.world.get(target, "item.quantity", 1), 1.) <= 0:
            return _fail("actuator", "物资实物已损坏、过期或耗尽。")
        used = _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
        mass = _num(sim.world.get(target, "item.mass_kg")) * _num(sim.world.get(target, "item.quantity", 1), 1.)
        if used + mass > _num(agent.spec.get("payload_kg")):
            return _fail("actuator", "挂载时载荷容量不足，保留原有持有者。")
        payload = list(sim.world.get(agent.platform_eid, "platform.payload_item_ids", []) or [])
        if target in payload:
            return _fail("actuator", "物资已经挂载，不能重复计入载荷。")
        writes = []
        if state == "stock":
            writes.append((target, "item.state", "reserved"))
        writes += [(target, "item.state", "attached"), (target, "item.holder_id", agent.id),
                   (target, "item.attachment_id", params.get("attachment_id", agent.platform_eid)),
                   (agent.platform_eid, "platform.payload_item_ids", payload + [target])]
        used = _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
        mass = _num(sim.world.get(target, "item.mass_kg")) * _num(sim.world.get(target, "item.quantity", 1), 1.)
        writes.append((agent.platform_eid, "platform.payload_used_kg", used + mass))
        writes += self._inventory_remove(sim, holder, target)
        return _out(writes=writes, text="实际物资已挂载，持有者、库存和载荷同步。")

    def _inventory_remove(self, sim, holder, item):
        if _cls(sim, holder) != "station":
            return []
        inventory = deepcopy(sim.world.get(holder, "station.inventory", []))
        if isinstance(inventory, list):
            inventory = [i for i in inventory if i != item]
        elif isinstance(inventory, dict):
            if item in inventory:
                inventory.pop(item)
            else:
                kind = sim.world.get(item, "item.kind")
                value = inventory.get(kind)
                if isinstance(value, list):
                    inventory[kind] = [i for i in value if i != item]
                elif isinstance(value, (int, float)):
                    inventory[kind] = max(0, value - _num(sim.world.get(item, "item.quantity", 1), 1.))
        return [(holder, "station.inventory", inventory)]

    def _detach(self, sim, agent, item):
        payload = list(sim.world.get(agent.platform_eid, "platform.payload_item_ids", []) or [])
        used = _num(sim.world.get(agent.platform_eid, "platform.payload_used_kg"))
        mass = _num(sim.world.get(item, "item.mass_kg")) * _num(sim.world.get(item, "item.quantity", 1), 1.)
        return [(item, "item.holder_id", None),
                (agent.platform_eid, "platform.payload_item_ids", [i for i in payload if i != item]),
                (agent.platform_eid, "platform.payload_used_kg", max(0., used - mass))]

    def _effect_A025(self, sim, agent, params, rng):
        target = _target(params)
        if sim.world.get(target, "item.holder_id") not in (agent.id, agent.platform_eid):
            return _fail("actuator", "待释放实物不再由本平台持有。")
        state = sim.world.get(target, "item.release_state", "locked")
        writes = [] if state == "unlocking" else [(target, "item.release_state", "unlocking")]
        fault = sim.world.get(target, "attr.release_jammed", False)
        fault = fault or (sim.world.get(agent.platform_eid, "attr.fault_kind") == "release" and
                          sim.world.get(agent.platform_eid, "equipment.health") == "failed")
        jam_rate = _clamp(sim.world.get(target, "attr.release_jam_rate", agent.spec.get("release_jam_rate", .08)))
        if fault or _exogenous_rng(sim, agent, rng, "clearance", "equipment_jam", _target(params)).random() < jam_rate:
            writes.append((target, "item.release_state", "jammed"))
            return _fail("actuator", "释放机构卡滞，物资仍挂载；保留重试条件。", writes)
        point = _xy(params.get("drop_point", params.get("release_point"))) or _own_pose(sim, agent)
        if _dist(_own_pose(sim, agent), point) > 25.:
            return _fail("actuator", "释放点超出本机作用范围。", writes)
        drift = _num(sim.world.get(target, "attr.drop_error_m"), 1.)
        point = [v + rng.uniform(-drift, drift) for v in point]
        writes += [(target, "item.release_state", "released"), (target, "item.state", "released"),
                   (target, "item.contact_pose", point), (target, "entity.pose", point)] + self._detach(sim, agent, target)
        return _out(writes=writes, text="物资已经释放并记录实际落点，尚未签收。")

    def _effect_A026(self, sim, agent, params, rng):
        target = _target(params)
        point = _xy(params.get("drop_point", params.get("place_point"))) or _own_pose(sim, agent)
        if _dist(_own_pose(sim, agent), point) > 15.:
            return _fail("blocked", "放置点不在实际卸载可达范围内。")
        state = sim.world.get(target, "item.state")
        if state not in ("attached", "in_transit", "released"):
            return _fail("actuator", "物资不处于可卸载状态。")
        writes = []
        if state == "attached":
            writes.append((target, "item.state", "released"))
        writes += [(target, "item.state", "placed"), (target, "item.contact_pose", point), (target, "entity.pose", point)]
        if sim.world.get(target, "item.holder_id") in (agent.id, agent.platform_eid):
            writes += self._detach(sim, agent, target)
        return _out(writes=writes, text="物资已放置到实际地面位置；放置不等于领取。")

    def _effect_A027(self, sim, agent, params, rng):
        target = _target(params)
        if sim.world.get(target, "item.state") not in ("released", "placed"):
            return _fail("actuator", "尚无已释放或已放置物资可供领取。")
        point = _xy(sim.world.get(target, "item.contact_pose")) or _pose(sim, target)
        if _dist(_own_pose(sim, agent), point) > _num(agent.spec.get("sensor_range_m"), 250.):
            return _fail("sensor", "落点不在本地复核范围；尚未收到可用领取确认。")
        requested = params.get("receiver_id")
        candidates = [requested] if requested else sorted(sim.world.entities)
        receiver = next((eid for eid in candidates if _cls(sim, eid) == "human" and
                         sim.world.get(eid, "human.alive", True) and sim.world.get(eid, "human.mobility") in ("mobile", "limited") and
                         _dist(_pose(sim, eid), point) <= 8.), None)
        if not receiver:
            return _fail("timeout", "落点附近没有具备领取能力的人员；接触或释放不能替代签收。")
        receipt = "receipt-" + str(params.get("_instance_id", f"{target}-{sim.t}"))
        writes = [(target, "item.state", "received"), (target, "item.receipt_id", receipt),
                  (target, "item.receiver_id", receiver), (target, "item.holder_id", receiver),
                  (receiver, "human.support_received", _add_support(sim, receiver, "supplies_received"))]
        obs = _observation(sim, agent, params, [target], {"item.state": "received", "item.receipt_id": receipt, "item.receiver_id": receiver}, "receipt_confirmation", .95)
        support = _observation(sim, agent, params, [receiver], {"human.support_received": _add_support(sim, receiver, "supplies_received")}, "receipt_confirmation", .95)
        return _out(writes=writes, obs=[obs, support], text="实际在场且能够领取的人员已接收物资；中心仍须等待报告送达。")

    def _effect_A028(self, sim, agent, params, rng):
        station = _station(params)
        if not _near(sim, agent, station):
            return _fail("blocked", "尚未到达补给站。")
        kind = params.get("item_kind", params.get("kind", "water"))
        item = params.get("item_id") or next((eid for eid in sorted(sim.world.entities) if _cls(sim, eid) == "item" and sim.world.get(eid, "item.kind") == kind and sim.world.get(eid, "item.holder_id") == station and _num(sim.world.get(eid, "item.quantity", 1)) > 0), None)
        if not item or sim.world.get(item, "item.holder_id") != station:
            return _fail("actuator", "站点没有对应实际库存，不能生成物资。")
        equipment = _equipment(sim, agent, params)
        if equipment:
            remaining = max(0., _num(params.get('quantity'), 1.))
            writes = []; transferred = 0.
            stock = [item] if params.get('item_id') else [eid for eid in sorted(sim.world.entities)
                if _cls(sim,eid)=='item' and sim.world.get(eid,'item.kind')==kind
                and sim.world.get(eid,'item.holder_id')==station and sim.world.get(eid,'item.state')=='stock']
            for eid in stock:
                available = _num(sim.world.get(eid,'item.quantity'))
                quantity = min(available, remaining)
                if quantity > 0:
                    writes.append((eid,'item.quantity',available-quantity))
                    transferred += quantity; remaining -= quantity
                if remaining <= 0:
                    break
            if transferred <= 0:
                return _fail("actuator", "补充数量为零或库存已耗尽。")
            writes.append((equipment,'equipment.consumable_remaining',sim.world.get(equipment,'equipment.consumable_remaining')+transferred))
            return _out(writes=writes, text="消耗实际站点库存补充装备耗材。")
        # Transfer one existing item instance, never fabricate split inventory.
        updated = dict(params, target=item)
        result = self._effect_A024(sim, agent, updated, rng)
        result.feedback_zh = "从站点转移一个实际物资实例并同步库存与载荷。" if result.status == "succeeded" else result.feedback_zh
        return result

    def _effect_A029(self, sim, agent, params, rng):
        state = params.get("_charge")
        writes = self._release_charge(sim, params)
        if not state or state["elapsed"] <= 0:
            return _fail("timeout", "尚未经历充电服务时间，电量没有瞬间增加。", writes)
        if state["failed"]:
            return _fail("energy", "供电或停靠条件中断，保留实际已充电量。", writes, partial=True)
        if self.id == "A030":
            battery = state["battery"]
            old_battery = state["old_battery"]
            old_state = sim.world.get(old_battery, "item.state", "attached")
            if old_state in ("attached", "in_transit"):
                writes.append((old_battery, "item.state", "released"))
            writes += [(old_battery, "item.state", "placed"), (old_battery, "item.holder_id", state["station"]),
                       (old_battery, "entity.pose", _pose(sim, state["station"])),
                       (battery, "item.state", "attached"), (battery, "item.holder_id", agent.id),
                       (agent.platform_eid, "attr.battery_item_id", battery)]
        current = _num(sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh"))
        maximum = _num(sim.world.get(agent.platform_eid, "platform.battery_capacity_Wh"), agent.spec.get("battery_Wh", 0.))
        target = _target_charge(params, maximum)
        return _out("succeeded" if current + 1e-6 >= target else "partial", writes=writes, text="保障服务按实际经过时间补充电量；保留未达目标与已用库存。")

    _effect_A030 = _effect_A029

    def _effect_A031(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, 20.):
            return _fail("blocked", "回收工具尚未接近失效平台。")
        if sim.world.get(target, "platform.health_state", "nominal") not in ("failed", "degraded"):
            return _fail("actuator", "目标平台不处于声明的故障状态。")
        goal = _goal(sim, agent, params)
        if goal is None:
            return _fail("blocked", "未提供可达回收点。")
        route = sim.world.path(_own_pose(sim, agent), goal, agent.spec)
        if not route.get("ok"):
            return _fail("blocked", "回收路径受阻；故障未凭空修复。")
        writes = [(target, "entity.pose", goal), (agent.platform_eid, "entity.pose", goal)]
        if sim.world.get(target, "platform.motion_state") == "moving":
            writes.append((target, "platform.motion_state", "disabled"))
        return _out(writes=writes, text="失效平台已按有时长的牵引服务回收到指定点，原有故障保持不变。")

    def _effect_A038(self, sim, agent, params, rng):
        query = str(params.get("query", ""))
        target = _target(params)
        matches = []
        for eid, item in getattr(agent.belief, "evidence", {}).items():
            row = _obs_dict(item)
            if not isinstance(row, dict):
                continue
            if target and target not in row.get("about", []):
                continue
            if query and query not in json.dumps(row, ensure_ascii=False, default=str):
                continue
            if _num(row.get("t")) < _num(params.get("since_s"), -math.inf):
                continue
            matches.append(eid)
        obs = _observation(sim, agent, params, [target] if target else [], {"observation.parent_evidence_ids": matches, "cache.hit_count": len(matches)}, "history_query", 1., "local_index_v1")
        return _out("succeeded" if matches else "partial", obs=[obs], evidence=matches, text="检索仅限本机证据与已收到副本；未命中如实保留。")

    def _effect_A039(self, sim, agent, params, rng):
        observations = []
        for i, source in enumerate(_inputs(agent, params)):
            raw = json.dumps(source["values"], sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            # Actual deterministic metadata encoding; no pretend image bytes.
            ratio = _clamp(params.get("quality", .8), .1, 1.)
            transformed = {k: v for k, v in source["values"].items() if ratio >= .5 or k in ("entity.pose", "building.visual_damage", "belief.search_status")}
            encoded = json.dumps(transformed, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            values = dict(transformed, **{"observation.payload_bytes": len(encoded), "observation.content_hash": hashlib.sha256(encoded).hexdigest(),
                                        "observation.parent_evidence_ids": [source.get("id")], "attr.original_bytes": len(raw), "attr.quality_retained": ratio})
            observations.append(_observation(sim, agent, params, source.get("about", []), values, "transformed_evidence", _num(source.get("confidence"), .8) * ratio, "metadata_transform_v1", str(i)))
        return _out(obs=observations, text="已转换本地证据元数据，记录真实编码字节、哈希、来源与质量变化。")

    def _integrate(self, sim, agent, params, fuse=False):
        sources = _inputs(agent, params)
        if not sources:
            return _fail("model", "缺少本地可读证据，认知没有被真值替代。")
        conflicts = False
        for source in sources:
            about = source.get("about", [])
            values = source["values"]
            confidence = _clamp(source.get("confidence", .8))
            provenance = 'camera_image' if source.get('source') == 'camera_image' else 'observation'
            oid = source.get("id")
            agent.belief.add_evidence(sim, source)
            nested = values.get("entities")
            groups = nested.items() if isinstance(nested, dict) else [(eid, values) for eid in about]
            for eid, entity_values in groups:
                if not isinstance(entity_values, dict):
                    continue
                for key, value in entity_values.items():
                    if key.startswith(("attr.", "observation.", "cache.")) or value == "unknown":
                        continue
                    if key not in load().fields_by_key:
                        continue
                    old = agent.belief.get(eid, key)
                    if fuse and old and old.get("value") != value and old.get("source") != "prior":
                        conflicts = True
                        state = _known(agent, eid, "belief.conflict_state", "none")
                        if state == "none":
                            agent.belief.put(sim, eid, "belief.conflict_state", "disputed", .5, "inference", evidence_ids=[oid])
                        # Conflicting measurements remain separate evidence. A
                        # larger confidence is not an oracle resolving the truth.
                        continue
                    if key == "belief.search_status" and value in ("partial", "complete", "inconclusive"):
                        before_status = _entry_value(old, "unknown")
                        if before_status in ("unknown", "not_started", "partial"):
                            agent.belief.put(sim, eid, key, "in_progress", confidence, "inference" if fuse else provenance, evidence_ids=[oid])
                    # put owns the detached entry/event snapshots, and checks
                    # unchanged evidence before allocating them.
                    agent.belief.put(sim, eid, key, value, confidence, "inference" if fuse else provenance, evidence_ids=[oid])
                if oid:
                    previous = _list(_known(agent, eid, "belief.evidence_ids", []))
                    agent.belief.put(sim, eid, "belief.evidence_ids", list(dict.fromkeys(previous + [oid])), confidence, "inference" if fuse else provenance, evidence_ids=[oid])
                agent.belief.put(sim, eid, "belief.updated_at_s", sim.t, 1., "inference" if fuse else provenance, evidence_ids=[oid])
        return _out("partial" if conflicts else "succeeded", evidence=[s.get("id") for s in sources if s.get("id")], text="本地证据存在冲突，已保留争议而非查真值裁决。" if conflicts else "仅根据本地可读证据更新本机认知，未修改世界真值。")

    def _effect_A040(self, sim, agent, params, rng):
        return self._integrate(sim, agent, params, fuse=True)

    def _effect_A041(self, sim, agent, params, rng):
        return self._integrate(sim, agent, params)

    def _effect_A059(self, sim, agent, params, rng):
        receiver = params.get("receiver_id", params.get("specialist_id"))
        if not receiver:
            # Public configured identities/kinds are not private availability.
            receiver = next((aid for aid, runtime in sorted(sim.agents.items()) if runtime.kind == "specialist_team"), None)
        if not receiver or receiver not in sim.agents or sim.agents[receiver].kind != "specialist_team":
            return _fail("lost_link", "没有可寻址的专业队伍；没有把请求记为完成。")
        payload = deepcopy(params.get("payload"))
        if not isinstance(payload, dict):
            payload = {"op": "specialist_request", "action_id": params.get("requested_action_id", params.get("action_id")),
                       "target": _target(params), "required_capabilities": params.get("required_capabilities", []),
                       "task_ref": params.get("_task_ref", params.get("task_ref")), "constraints": deepcopy(params.get("constraints", {}))}
        payload.setdefault("op", "specialist_request")
        if getattr(agent, "comm_budget_gate", None) is not None and not agent._admit_send(sim, [receiver], payload):
            return _fail("model", "本地通信预算拒绝专业请求，消息未进入网络队列。")
        mid = sim.send_message(agent.id, [receiver], "task", payload, len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")), task_ref=params.get("_task_ref"))
        obs = _observation(sim, agent, params, [receiver], {"attr.request_message_id": mid, "attr.request_state": "queued"}, "specialist_request", 1., "abstract_message_v1")
        return _out(obs=[obs], text="专业请求已进入通信；对方须实际收到并自主接受或拒绝。")

    def _effect_A060(self, sim, agent, params, rng):
        records = _inputs(agent, params)
        target = _target(params)
        if not records:
            # Only holder-owned task records, never sim.tasks_index or events.
            records = [deepcopy(row) for tid, row in agent.tasks.items() if not target or target == tid]
        data = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        obs = _observation(sim, agent, params, [agent.id], {"cache.content_ids": [r.get("id") for r in records if isinstance(r, dict)], "cache.size_bytes": len(data), "attr.content_hash": digest, "attr.archive_records": records}, "local_archive", 1., "local_trace_v1")
        agent.belief.add_evidence(sim, obs)
        return _out(obs=[obs], text="已归档本机合法数据及版本，不读取评估真值或增加救援得分。")

    def _effect_A061(self, sim, agent, params, rng):
        if params.get("service_id"):
            return sim.compute.action_effect(agent, self.id, params, rng)
        target = _target(params) or agent.platform_eid
        own_equipment = _list(sim.world.get(agent.platform_eid, "platform.equipment_ids"))
        if target not in [agent.platform_eid] + own_equipment:
            return _fail("actuator", "没有授权操作目标设备的恢复接口。")
        recoverable = sim.world.get(target, "attr.fault_recoverable", False)
        if not recoverable:
            return _fail("actuator", "故障没有可恢复声明，保留故障并需要隔离或回收。")
        writes = []
        if _cls(sim, target) == "platform":
            state = sim.world.get(target, "platform.health_state", "nominal")
            if state in ("failed", "degraded"):
                writes.append((target, "platform.health_state", "recovering"))
            writes.append((target, "platform.health_state", "nominal"))
        elif sim.world.get(target, "equipment.health") is not None:
            if sim.world.get(target, "equipment.health") == "failed":
                return _fail("actuator", "物理装备失效不属于软件自动修复范围。")
            writes.append((target, "equipment.health", "ready"))
        else:
            writes.append((target, "sensor.health", "ready"))
        return _out(writes=writes, text="仅恢复声明可恢复的本机故障，未修复未建模的物理损伤。")

    def _effect_A062(self, sim, agent, params, rng):
        target = _target(params)
        equipment = _equipment(sim, agent, params)
        source_pose = _pose(sim, equipment) if equipment else _own_pose(sim, agent)
        reach = sim.world.get(equipment, 'equipment.reach_m') if equipment else agent.spec.get('suppression_reach_m', 50.)
        polygon = _entity(sim, target)['geometry']['coords']
        distance = 0. if _point_in(source_pose, polygon) else min(_dist(source_pose, _project(source_pose, a, b)[0]) for a, b in zip(polygon, polygon[1:]+polygon[:1]))
        if reach is None or distance > reach:
            return _fail("blocked", "火点超出消防装备作用范围。")
        intensity = max(0., _num(sim.world.get(target, "hazard.fire_intensity")))
        water = params.get("water_item_id") or next(_items(sim, agent, ("water",), near=True), None)
        available = _num(sim.world.get(equipment, "equipment.consumable_remaining")) if equipment else _num(sim.world.get(water, "item.quantity")) if water else 0.
        if available <= 0:
            return _fail("actuator", "没有实际灭火介质或供水，火势未被免费降低。")
        duration = _num(params.get("_plan", {}).get("duration_s"), 90.)
        used = min(available, duration * .05)
        reduction = used * _num(agent.spec.get("suppression_effect_per_unit"), .15)
        after = max(0., intensity - reduction)
        key = "equipment.consumable_remaining" if equipment else "item.quantity"
        source = equipment or water
        writes = [(target, "hazard.fire_intensity", after), (target, "hazard.active", after > 0), (source, key, available - used)]
        return _out("succeeded" if after <= 0 else "partial", writes=writes, text="按实际介质消耗降低火势；残余火势及后续复燃仍可能存在。")

    def _effect_A063(self, sim, agent, params, rng):
        target = _target(params)
        if not _near(sim, agent, target, _num(agent.spec.get("isolation_reach_m"), 80.)):
            return _fail("blocked", "尚未到达可布置隔离的位置。")
        zone = params.get("exclusion_zone") or sim.world.get(target, "hazard.footprint") or _part(_entity(sim, target), "geometry")
        if not zone:
            return _fail("actuator", "没有可声明的隔离区域。")
        obs=_observation(sim,agent,params,[target],{'hazard.exclusion_zone':deepcopy(zone)},
                         'isolation_feedback',1.,'actual_isolation_v1')
        return _out(writes=[(target, "hazard.exclusion_zone", deepcopy(zone))],obs=[obs], text="已设置危险源隔离警戒区，未凭空消除未处置的危险源。")

    def _effect_A064(self, sim, agent, params, rng):
        source_version = params.get("source_version", params.get("model_version", "v1"))
        target_version = params.get("target_version", source_version)
        observations = []
        for i, source in enumerate(_inputs(agent, params)):
            values = deepcopy(source["values"])
            encoded = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            values.update({"cache.model_version": target_version, "cache.reuse_compatible": True,
                           "observation.parent_evidence_ids": [source.get("id")], "observation.payload_bytes": len(encoded),
                           "attr.adaptation": {"from": source_version, "to": target_version, "mode": "metadata_only"}})
            observations.append(_observation(sim, agent, params, source.get("about", []), values, "representation_adaptation", _num(source.get("confidence"), .8), "metadata_adapter_v1", str(i)))
        return _out(obs=observations, text="已记录兼容表示适配及字节成本；未声称执行视觉特征转换。")

    def _effect_A066(self, sim, agent, params, rng):
        goal = _goal(sim, agent, params, known=True)
        route = known_route(agent, _own_pose(sim, agent), goal,
                            work_target=params.get("approach_obstacle_id"))
        obs = _observation(sim, agent, params, [_target(params)] if _target(params) else [],
                           {"job.result_ref": route, "action.parameters": {"path": route["waypoints"], "uncertain": route["uncertain"]}},
                           "known_map_route", .7 if route["uncertain"] else .9, "known_map_dijkstra_v1")
        if not route["ok"]:
            return _fail("blocked", "本机已知地图没有可行候选路线，未查询隐藏障碍或全局真值。", obs=[obs])
        return _out(obs=[obs], text="仅用本机已知地图生成候选路线；未知障碍留待实际执行发现。")


# Metadata is loaded verbatim from the read-only original catalog. No invented
# ids/capabilities/writekeys and no identity-based hand-written coverage claim.
MODELS = {aid: ActionModel(row) for aid, row in load().actions.items()}
