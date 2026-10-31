"""Phase-1 pilot collector: leak-free decision sections with real counterfactual labels.

The collector drives existing ``Environment`` episodes with ``ForcedRuleDriver``
(the explicit rule sequence, not a learned policy) and records each substantive
rule action as an expert decision section, plus a few seeded cadence sections.

Three hard contracts (astra round-3 review):

1. **No answer leakage.**  ``intent_text`` never names the reference action, its
   skill name or an unassigned target; ``candidates`` carry only the documented
   twelve fields and never ``source``/``feasible``.
2. **No world peeking.**  ``center_context`` is built exclusively from the
   coordinator's delivered belief/tasks plus the public static scene config.
   Only coordinator-grounded task proposals are registered; the edge reads
   bounded center slots after native delivery. Empty task tables are valid.
3. **Real counterfactual utility.**  Every candidate is rolled out on a
   ``deepcopy(sim)`` clone through the native ``MODELS[action_id]`` guard and
   physical effect; ``delta_total`` combines real milestone advances, the
   net-state potential difference, cost and precondition rejection. Every VoI
   band separately executes its communication treatment and selected action.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import tempfile
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rescue_scenes.paths import expand
from rescue_scenes.backgrounds import materialize
from rescue_scenes.core.api import Environment
from rescue_scenes.core.sim.kernel import Simulation
from rescue_scenes.forced_rules import ForcedRuleDriver
from rescue_scenes.core.catalog import load as load_catalog
from rescue_scenes.core.agents.allocation import resolve_method
from rescue_scenes.core.agents.taskgen import (RULES_BY_TASK, TASK_NAMES,
                                               candidates as task_candidates)

from aerorescue.phase1 import iscc_core
from aerorescue.phase1.schema import (ACCEPT_EPSILON, SCHEMA_ID, make_record_id,
                                      validate_decision_record)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "phase1_pilot"

SCENES = ("SC01", "SC07", "SC09", "GT04")
# Same geometry/asset family stays inside one split; no episode crosses splits.
SPLITS = {"SC01": "train", "SC09": "train", "SC07": "val", "GT04": "test"}
SEEDS = (1, 2, 3, 4, 5, 6)
# GT04's retained upstream product was generated from the allocated catalog;
# configs/GT04.json refuses to reuse it with the non-allocated catalog.
CONFIG_OVERRIDES = {"GT04": "GT04_upstream_allocated.json"}

# Blueprint line 177: the three aerial-ground coupling regimes.
REGIME_BY_SCENE = {"SC01": "A", "SC04": "A", "SC09": "A", "SC13": "A",
                   "SC05": "B", "SC07": "B", "SC10": "B", "SC11": "B",
                   "GT04": "C", "SC08": "C", "SC12": "C", "SC17": "C"}
# Authored regime priors (blueprint §四.2), not measured sensor calibration.
POSE_PRIOR = {
    "A": {"pitch_deg": -75.0, "sigma_theta_deg": 2.25, "sigma_t_m": 1.0,
          "note_zh": "近垂直俯视与倒塌掩体遮挡；空地视锥交集 10%-18%"},
    "B": {"pitch_deg": -45.0, "sigma_theta_deg": 1.0, "sigma_t_m": 0.6,
          "note_zh": "大面积重复纹理/混凝土地面；共视重叠 40%-55% 但秩亏"},
    "C": {"pitch_deg": -30.0, "sigma_theta_deg": 3.0, "sigma_t_m": 1.5,
          "note_zh": "高速巡逻与间歇零共视窗口；重叠率 0%-35% 周期切换"},
}

MAX_RECORDS_PER_EPISODE = 16
MIN_RECORDS_PER_EPISODE = 10
CADENCE_FLOOR = 2
MAX_CANDIDATES = 5
MEMORY_BUDGET_BYTES = 4096
ISCC_BUDGET_BYTES = 2048
ROLLOUT_WINDOW_CAP_S = 300.0

# Task/skill actions that express a next-step choice; local processing
# (RunInference, LocalizeTarget, UpdateBelief) and task management are excluded.
ACTION_TASK = {
    "A001": "T019", "A002": "T020", "A003": "T020", "A004": "T019",
    "A005": "T011", "A008": "T014", "A009": "T027", "A011": "T003",
    "A012": "T022", "A013": "T023", "A017": "T032", "A019": "T031",
    "A020": "T033", "A021": "T034", "A022": "T035", "A024": "T038",
    "A025": "T039", "A026": "T040", "A062": "T093", "A063": "T095",
}
SUBSTANTIVE = frozenset(ACTION_TASK)

MOVEMENT = frozenset({"A001", "A002", "A003"})
ALT_ACTION = {
    "A001": "A005", "A002": "A005", "A003": "A005", "A004": "A005",
    "A008": "A002", "A009": "A005", "A011": "A005", "A012": "A001",
    "A013": "A001", "A017": "A019", "A019": "A017", "A020": "A021",
    "A021": "A022", "A022": "A021", "A024": "A001", "A025": "A026",
    "A026": "A025", "A062": "A005", "A063": "A005",
}

INTRINSICS = {"width": 1920, "height": 1080, "fx": 1371.0, "fy": 1371.0,
              "cx": 960.0, "cy": 540.0, "hfov_deg": 70.0}

# delta_cursor stays a diagnostic field; its weight is exactly zero.
DELTA_WEIGHTS = {"milestone": 0.60, "potential": 0.40, "cursor": 0.0,
                 "cost": 1.0, "fail": 1.0}
FAIL_PENALTY = -1.0
PARTIAL_PENALTY = -0.5

# Real physical milestones observed on the counterfactual clone.
MILESTONE_WEIGHTS = {
    "support:contacted": 0.20, "support:extricated": 0.30, "support:treated": 0.30,
    "support:recovered_from_water": 0.30, "support:supplies_received": 0.20,
    "loaded": 0.25, "handover:in_transit": 0.15, "handover:arrived": 0.20,
    "handover:acknowledged": 0.30, "exclusion_zone": 0.25,
    "payload_gain": 0.15, "new_observation": 0.03, "building_assessed": 0.15,
    "obstacle_progress": 0.25, "fire_suppressed": 0.30,
}

# Real physical extents used to build the projected anchor footprint Gamma_k.
CLASS_EXTENT_M = {"human": 1.7, "building": 8.0, "obstacle": 2.0, "road": 4.0,
                  "hazard": 3.0, "item": 0.5, "station": 4.0, "open_area": 12.0,
                  "facility": 6.0, "cue": 1.0}
STATION_SOURCE = "coordinator_belief_or_initial_static"
POTENTIAL_RANGE_M = 400.0


# --------------------------------------------------------------------------- helpers

def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _jsonable(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _entry_value(entry):
    return entry.get("value") if isinstance(entry, dict) and "value" in entry else entry


def _action_row(action_id):
    return load_catalog().actions.get(action_id, {})


_ACTION_NAMES = None


def _skill_name(action_id):
    global _ACTION_NAMES
    if _ACTION_NAMES is None:
        _ACTION_NAMES = {row["id"]: row.get("name", row["id"])
                         for row in load_catalog().raw.get("actions", [])}
    return _ACTION_NAMES.get(action_id, action_id)


def _task_priority(task_type):
    rule = RULES_BY_TASK.get(task_type)
    return int(rule.priority) if rule else 50


def _task_name(task_type):
    return TASK_NAMES.get(task_type, task_type)


def _action_task(action_id, agent):
    if action_id == "A001":
        return "T019"
    if action_id in ("A002", "A003"):
        return "T020"
    return ACTION_TASK.get(action_id, "T019")


def _movement_action(agent):
    mobility = agent.spec.get("mobility")
    return "A001" if mobility == "air" else "A003" if mobility == "water" else "A002"


def _adapt_movement(action_id, agent):
    """Record the embodiment-valid movement skill when the forced driver used a
    temporary air/water mobility override."""
    mobility = agent.spec.get("mobility")
    if action_id == "A001" and mobility != "air":
        return "A003" if mobility == "water" else "A002"
    if action_id == "A002" and mobility == "air":
        return "A001"
    if action_id == "A003" and mobility != "water":
        return "A002"
    return action_id


def _point(fields):
    if not isinstance(fields, dict):
        return None
    for key in ("entity.pose", "human.pose", "belief.estimated_pose"):
        if key in fields:
            value = _entry_value(fields[key])
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return [float(value[0]), float(value[1])]
    geometry = _entry_value(fields.get("entity.geometry"))
    if isinstance(geometry, dict) and geometry.get("coords"):
        coords = geometry["coords"]
        if isinstance(coords[0], (list, tuple)):
            return [sum(float(p[0]) for p in coords) / len(coords),
                    sum(float(p[1]) for p in coords) / len(coords)]
        if len(coords) >= 2:
            return [float(coords[0]), float(coords[1])]
    return None


def _entity_class(fields):
    if not isinstance(fields, dict):
        return None
    value = _entry_value(fields.get("entity.kind"))
    return value if isinstance(value, str) else None


def _distance(a, b):
    if a is None or b is None:
        return math.inf
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _targets(agent):
    """Agent-local believed entities: id, class, point and raw belief fields.

    The agent's own platform is telemetry, never a task target.
    """
    rows = []
    for eid, fields in agent.belief.entities().items():
        if eid in (agent.id, agent.platform_eid):
            continue
        point = _point(fields)
        if point is None:
            continue
        rows.append({"entity_id": eid, "class": _entity_class(fields), "point": point,
                     "fields": fields})
    return rows


def _target_index(targets):
    return {row["entity_id"]: row for row in targets}


def _novelty(fields, now):
    if not fields:
        return 0.3
    observed = [entry for entry in fields.values()
                if isinstance(entry, dict) and entry.get("source") not in (None, "prior")]
    if not observed:
        return 1.0
    last = max(float(entry.get("t", 0.0) or 0.0) for entry in observed)
    return 0.2 if now - last <= 120.0 else 0.6


def _feasible(agent, action_id, target_id, index, now):
    """Collector-side guard precheck.  Never serialised into ``candidates``."""
    row = _action_row(action_id)
    if not row:
        return False
    capabilities = set(agent.spec.get("capabilities", []))
    if not set(row.get("capabilities", [])) <= capabilities:
        return False
    mobility = agent.spec.get("mobility")
    if action_id == "A001" and mobility != "air":
        return False
    if action_id == "A002" and mobility not in {"legged", "wheeled", "foot"}:
        return False
    if action_id == "A003" and mobility != "water":
        return False
    target = index.get(target_id)
    if target and target["class"]:
        from rescue_scenes.core.sim.actions import TARGET_CLASSES
        allowed = TARGET_CLASSES.get(action_id)
        if allowed and target["class"] not in allowed:
            return False
    if action_id in MOVEMENT and not agent.spec.get("mains_power"):
        platform = agent.platform
        if float(platform.get("battery_Wh", 0)) <= float(platform.get("reserve_Wh", 0)):
            return False
    return True


def _params(action_id, target_id, point, agent=None):
    params = {"target": target_id}
    if action_id in MOVEMENT and point is not None:
        params["goal"] = [round(point[0], 2), round(point[1], 2)]
    elif action_id == "A005":
        params.update(sensor_id="scripted_rgb", modality="rgb")
    elif action_id == "A008":
        params.update(building=target_id, search_budget_s=120)
    elif action_id == "A009":
        params["wait_s"] = 30
    if agent is not None and action_id in {"A021", "A022"}:
        stations = [row for row in _targets(agent) if row["class"] == "station"
                    and _entry_value(row["fields"].get("station.kind"))
                        in {"medical", "safe_point", "shelter", "handover"}]
        if stations:
            station = min(stations, key=lambda row: (_distance(point, row["point"]), row["entity_id"]))
            params["station_id"] = station["entity_id"]
            if action_id == "A021":
                params["goal"] = station["point"]
    if agent is not None and action_id in {"A017", "A062"}:
        equipment = agent.spec.get("action_equipment", {}).get(action_id)
        if equipment is not None:
            params["equipment_id"] = equipment
    return params


def _class_actions(cls, agent):
    movement = _movement_action(agent)
    capabilities = set(agent.spec.get("capabilities", []))
    if cls == "building":
        return ["A005", movement, "A008"]
    if cls == "human":
        return ["A009", "A005", movement]
    if cls == "hazard":
        actions = ["A005", movement]
        if "C47" in capabilities:
            actions.insert(0, "A062")
        if "C48" in capabilities:
            actions.insert(0, "A063")
        return actions
    if cls == "road":
        return ["A011", movement]
    if cls == "item":
        return ["A024", "A005"]
    if cls == "obstacle":
        return ["A012", "A013"]
    return ["A005", movement]


def _infeasible_action(agent, target_id, index):
    if agent.spec.get("mobility") != "water":
        return "A003"
    for action_id in ("A017", "A013", "A021", "A062", "A009"):
        if not _feasible(agent, action_id, target_id, index, 0.0):
            return action_id
    return None


def _agent_view(agent, driver):
    """Restore nominal embodiment rather than expose expert-specific grants."""
    spec = dict(agent.spec)
    capabilities = set(spec.get("capabilities", []))
    for row in driver.report.get("capability_overrides", []):
        if row.get("agent") != agent.id:
            continue
        capabilities -= set(row.get("added", []))
        if row.get("field") == "mobility":
            spec["mobility"] = row.get("before", spec.get("mobility"))
    spec["capabilities"] = sorted(capabilities)
    return SimpleNamespace(id=agent.id, kind=agent.kind, spec=spec, platform=agent.platform,
                           belief=agent.belief, tasks=agent.tasks,
                           received_messages=agent.received_messages,
                           platform_eid=agent.platform_eid, current=agent.current)


def _compact_value(value, limit=240):
    """Keep delivered observation values readable without copying large world
    polygons into every record."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    if isinstance(value, list):
        return {"summary": "truncated_list", "length": len(value), "preview": value[:3]}
    if isinstance(value, dict):
        return {"summary": "truncated_object", "keys": sorted(map(str, value))[:8]}
    return text[:limit]


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()
                if not (isinstance(key, str) and
                        (key.lower() in {"truth", "provenance", "world.entities",
                                         "truth_for_rendering_only"} or "truth" in key.lower()))}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _coordinator(sim):
    return next((a for a in sim.agents.values() if a.kind == "coordinator"), None)


# --------------------------------------------------------------------------- centre side

def _register_center_tasks(sim, coordinator):
    """Register only proposals grounded in the coordinator's existing knowledge.

    A section's expert action and another holder's unreported beliefs are never
    valid sources for center state. An empty task table is a valid observation.
    """
    if coordinator is None:
        return 0
    registered = 0
    for proposal in task_candidates(coordinator, sim, limit=3):
        outcome = coordinator.apply_task_action(sim, "A050", {
            "type": proposal["type"], "task_type": proposal["type"],
            "target": str(proposal["target"]), "origin": proposal["origin"],
            "priority": proposal.get("priority", 50), "discovered_via": "center_belief"})
        registered += getattr(outcome, "status", None) in ("succeeded", "partial")
    return registered


def _center_context(sim, cfg, agent):
    """Delivered centre state only: coordinator belief/tasks + public static config.

    No ``sim.world`` access at all: station stock/occupancy is reported only if
    the coordinator actually received it, otherwise it stays ``not_delivered``.
    """
    coordinator = _coordinator(sim)
    tasks = []
    observations = []
    poses = []
    messages = 0
    known_entities = 0
    objectives = []
    if coordinator is not None:
        messages = len(coordinator.received_messages)
        known_entities = len(coordinator.belief.entities())
        for task in coordinator.tasks.values():
            tasks.append({"task_id": task.get("id"), "task_type": task.get("type"),
                          "target_id": str(task.get("target")), "status": task.get("status"),
                          "priority": task.get("priority"),
                          "assigned_to": task.get("assigned_to"),
                          "revision": task.get("rev")})
        grouped = {}
        for (eid, key), entry in coordinator.belief.entries.items():
            if entry.get("source") == "prior":
                continue
            grouped.setdefault(eid, {})[key] = entry
        for eid in sorted(grouped):
            fields = grouped[eid]
            values = {}
            for key in sorted(fields)[:6]:
                value = _jsonable(_entry_value(fields[key]))
                if key == "belief.team_estimates" and isinstance(value, dict):
                    value = {name: value.get(name) for name in
                             ("id", "kind", "pose", "battery_Wh", "current_task", "available")}
                values[key] = _compact_value(value)
            observations.append({
                "entity_id": eid,
                "observed": values,
                "last_report_s": round(max(float(entry.get("t", 0.0) or 0.0)
                                           for entry in fields.values()), 3),
                "reported_by": sorted({str(entry.get("from_agent")) for entry in fields.values()
                                       if entry.get("from_agent")}),
            })
        for eid, fields in sorted(coordinator.belief.entities().items()):
            point = _point(fields)
            if point is None:
                continue
            poses.append({"entity_id": eid, "pose": point,
                          "last_report_s": round(max(
                              float(entry.get("t", 0.0) or 0.0)
                              for entry in fields.values() if isinstance(entry, dict)), 3),
                          "source": STATION_SOURCE})
        objectives = [{"objective_id": row.get("id"), "kind": row.get("kind"),
                       "target_id": str(row.get("target")), "priority": row.get("priority")}
                      for row in (coordinator.public_objectives or []) if row.get("public", True)]

    stations = []
    if coordinator is not None:
        for eid, fields in sorted(coordinator.belief.entities().items()):
            if _entity_class(fields) != "station":
                continue
            delivered = sorted(key for key in fields
                               if key.startswith("station.") or key.startswith("attr."))
            stations.append({
                "entity_id": eid,
                "kind": _entry_value(fields.get("station.kind")),
                "capacity": _entry_value(fields.get("station.capacity")),
                "occupied": _entry_value(fields.get("station.occupied", None)),
                "inventory": _jsonable(_entry_value(fields.get("station.inventory", None))),
                "open_state": _entry_value(fields.get("station.open_state", None)),
                "delivered_keys": delivered,
                "source": STATION_SOURCE,
            })

    catalog_tasks = {}
    for row in sim.agents.values():
        for task_type in row.spec.get("task_allowlist", []):
            catalog_tasks.setdefault(task_type, {
                "task_type": task_type, "name_zh": _task_name(task_type),
                "priority": _task_priority(task_type)})
    return {
        "known_tasks": tasks,
        "mission_task_catalog": [catalog_tasks[key] for key in sorted(catalog_tasks)],
        "public_objectives": objectives,
        "reported_observations": observations,
        "known_entity_poses": poses,
        "stations": stations,
        "station_source": STATION_SOURCE,
        "delivered_message_count": messages,
        "known_entity_count": known_entities,
        "disaster_label": str((cfg.get("display") or {}).get("mission_label", "")),
        "public_brief_zh": str(cfg.get("public_brief_zh", "")),
    }


def _build_memory(center, now, budget=MEMORY_BUDGET_BYTES - 64):
    """Serialize bounded slots, reserving 64 wire bytes for the report wrapper."""
    slots = []
    for task in center["known_tasks"]:
        slots.append({"slot_id": f"task:{task['task_id']}", "kind": "task",
                      "key": "task_record", "value": task, "t": now})
    for pose in center["known_entity_poses"]:
        slots.append({"slot_id": f"pose:{pose['entity_id']}", "kind": "entity_pose",
                      "key": "entity.pose", "value": pose["pose"], "t": pose["last_report_s"]})
    for station in center["stations"]:
        slots.append({"slot_id": f"station:{station['entity_id']}", "kind": "station",
                      "key": "station.state", "value": station, "t": now})
    for observation in center["reported_observations"]:
        slots.append({"slot_id": f"obs:{observation['entity_id']}", "kind": "observation",
                      "key": "reported_observations", "value": observation["observed"],
                      "t": observation["last_report_s"]})
    for slot in slots:
        slot["source"] = STATION_SOURCE
        slot["bytes"] = len(json.dumps(slot["value"], ensure_ascii=False,
                                       default=str).encode("utf-8"))

    memory = {
        "schema": "aerorescue.memory_envelope/v1", "mode": "center_to_edge_push",
        "budget_bytes": int(budget), "serialized_bytes": 0,
        "source_cut_s": round(float(now), 3),
        "representation": "structured_json_slots/v1", "slots": slots,
    }
    def measure():
        # Include the envelope and its byte-count field in the wire ledger.
        for _ in range(8):
            size = len(json.dumps(memory, ensure_ascii=False).encode("utf-8"))
            if size == memory["serialized_bytes"]:
                return size
            memory["serialized_bytes"] = size
        raise ValueError("memory byte count did not converge")
    while measure() > budget and slots:
        slots.pop()
    if measure() > budget:
        raise ValueError("memory budget cannot hold the empty envelope")
    return memory


def _delivered_memory(agent, now):
    """Only a complete native-network delivery can expose center slots."""
    memories = [message["payload"]["phase1_memory"]
                for message in agent.received_messages
                if isinstance(message.get("payload", {}).get("phase1_memory"), dict)
                and message.get("delivered_s") is not None
                and float(message["delivered_s"]) <= now
                and float(message["payload"]["phase1_memory"]["source_cut_s"]) <= now]
    if memories:
        return deepcopy(memories[-1])
    return _build_memory({"known_tasks": [], "known_entity_poses": [], "stations": [],
                          "reported_observations": []}, now)


def _context_from_memory(public, memory):
    """The model sees public mission scope and bounded, delivered slots only."""
    center = {key: deepcopy(public[key]) for key in (
        "mission_task_catalog", "public_objectives", "station_source",
        "disaster_label", "public_brief_zh")}
    center.update(known_tasks=[], known_entity_poses=[], stations=[],
                  reported_observations=[], delivered_message_count=0, known_entity_count=0)
    for slot in memory["slots"]:
        kind, value = slot["kind"], deepcopy(slot["value"])
        if kind == "task":
            center["known_tasks"].append(value)
        elif kind == "entity_pose":
            center["known_entity_poses"].append({"entity_id": slot["slot_id"].split(":", 1)[1],
                "pose": value, "last_report_s": slot["t"], "source": STATION_SOURCE})
        elif kind == "station":
            center["stations"].append(value)
        elif kind == "observation":
            center["reported_observations"].append({
                "entity_id": slot["slot_id"].split(":", 1)[1], "observed": value,
                "last_report_s": slot["t"], "reported_by": ["delivered_center_memory"]})
    center["known_entity_count"] = len(center["known_entity_poses"])
    return center


def _pose_enu(agent, cfg):
    pose = agent.platform.get("pose") or [0.0, 0.0]
    z = 0.0
    if agent.spec.get("mobility") == "air":
        z = float(cfg.get("display", {}).get("uav_agl_m", 0.0) or 0.0)
    return [round(float(pose[0]), 3), round(float(pose[1]), 3), round(z, 3)]


def _edge_state(agent, cfg, targets, now):
    fields_by_entity = {row["entity_id"]: row["fields"] for row in targets}
    observed = [eid for eid, fields in fields_by_entity.items()
                if any(isinstance(entry, dict) and entry.get("source") not in (None, "prior")
                       for entry in fields.values())]
    reported = [eid for eid, fields in fields_by_entity.items()
                if any(isinstance(entry, dict) and entry.get("source") == "message"
                       for entry in fields.values())]
    last = 0.0
    for eid in observed:
        for entry in fields_by_entity[eid].values():
            if isinstance(entry, dict) and entry.get("source") not in (None, "prior"):
                last = max(last, float(entry.get("t", 0.0) or 0.0))
    humans = []
    for row in targets:
        if row["class"] != "human":
            continue
        keys = sorted(key for key, entry in row["fields"].items()
                      if isinstance(entry, dict) and entry.get("source") not in (None, "prior"))
        if keys:
            humans.append({"entity_id": row["entity_id"], "observed_keys": keys[:6],
                           "last_seen_s": round(max(float(row["fields"][key].get("t", 0.0) or 0.0)
                                                    for key in keys), 3)})
    summary = {
        "entity_count": len(fields_by_entity),
        "observed_entity_count": len(observed),
        "reported_entity_count": len(reported),
        "uncertain_entity_ids": sorted(set(fields_by_entity) - set(observed))[:12],
        "known_humans": humans[:6],
        "last_local_observation_s": round(last, 3),
        "known_entities": [{"entity_id": row["entity_id"], "class": row["class"],
            "pose": row["point"], "fields": {key: _compact_value(_entry_value(entry))
                for key, entry in row["fields"].items()}}
            for row in targets[:24]],
    }
    platform = agent.platform
    return {
        "agent_id": agent.id,
        "kind": agent.kind,
        "mobility": agent.spec.get("mobility"),
        "pose_enu": _pose_enu(agent, cfg),
        "battery_Wh": round(float(platform.get("battery_Wh", 0.0)), 3),
        "payload_kg": round(float(platform.get("payload_used_kg", 0.0)), 3),
        "capabilities": sorted(agent.spec.get("capabilities", [])),
        "health_state": platform.get("health_state", platform.get("health", "unknown")),
        "motion_state": platform.get("motion_state", "unknown"),
        "local_belief_summary": summary,
    }


def _camera_views(agent, cfg, targets, candidate_count, now):
    capabilities = set(agent.spec.get("capabilities", []))
    if "C05" not in capabilities:
        return []
    pose = _pose_enu(agent, cfg)
    # Authored fallback is explicit; never orient a sensor using the answer.
    yaw = float(agent.platform.get("yaw_deg", 0.0) or 0.0)
    pitch = -35.0 if agent.spec.get("mobility") == "air" else -5.0
    visible = []
    for row in targets:
        distance = _distance(pose, row["point"])
        if distance > 250.0:
            continue
        bearing = math.degrees(math.atan2(row["point"][1] - pose[1],
                                          row["point"][0] - pose[0]))
        offset = (bearing - yaw + 180.0) % 360.0 - 180.0
        if abs(offset) > 40.0:
            continue
        visible.append({"entity_id": row["entity_id"], "class": row["class"],
                        "distance_m": round(distance, 2), "bearing_deg": round(bearing, 2),
                        "observed": _novelty(row["fields"], now) < 1.0})
    visible.sort(key=lambda item: item["distance_m"])
    return [{
        "view_id": f"{agent.id}:rgb:front",
        "intrinsics": dict(INTRINSICS, calibration_source="authored_default_1080p_70deg"),
        "extrinsics": {"position_enu": pose, "yaw_deg": round(yaw, 2),
                       "pitch_deg": pitch, "frame": "local_ENU_m",
                       "orientation_source": "platform_yaw_or_authored_default"},
        "visible_targets": visible[:8],
        "perception_summary": {
            "modality": "rgb", "render_status": "metadata_only", "image_ref": None,
            "visible_count": len(visible),
            "unobserved_visible_count": sum(1 for item in visible if not item["observed"]),
            "candidate_action_count": candidate_count,
        },
    }]


# --------------------------------------------------------------------------- candidates

_METHOD_CACHE = {}


def _spec_key(agent):
    return (agent.spec.get("mobility"), tuple(sorted(agent.spec.get("capabilities", []))))


def _method_info(agent, task_type, action_id):
    """Resolve (sequence id, step index, main/recovery scope) from the catalog."""
    key = (task_type, action_id, _spec_key(agent))
    if key not in _METHOD_CACHE:
        catalog = load_catalog()
        try:
            method = resolve_method(catalog, task_type, agent.spec)
        except (ValueError, KeyError):
            _METHOD_CACHE[key] = (None, 0, "main")
        else:
            seq_id = method.get("seq_id")
            steps = list(method.get("resolved_steps") or [])
            index = steps.index(action_id) if action_id in steps else 0
            kind = (catalog.sequences.get(seq_id) or {}).get("kind")
            _METHOD_CACHE[key] = (seq_id, int(index), "recovery" if kind == "条件分支" else "main")
    return _METHOD_CACHE[key]


def _matching_task(agent, task_type, target_id):
    for task in agent.tasks.values():
        if task.get("status") in {"completed", "failed", "canceled", "expired"}:
            continue
        if task.get("type") == task_type and str(task.get("target")) == target_id:
            return task
    return None


def _candidate_row(agent, action_id, task_type, target_id, point):
    seq_id, step_index, scope = _method_info(agent, task_type, action_id)
    task = _matching_task(agent, task_type, target_id)
    return {
        "candidate_id": None,
        "skill_name": _skill_name(action_id),
        "task_id": task.get("id") if task else None,
        "task_rev": int(task.get("rev", 0)) if task else None,
        "task_type": task_type,
        "method_seq": seq_id,
        "step_index": step_index,
        "purpose": "approach" if action_id in MOVEMENT else "method",
        "method_scope": scope,
        "action_id": action_id,
        "target_id": str(target_id),
        "params": _params(action_id, str(target_id), point, agent),
    }


def _candidates(sim, agent, reference, rng, targets, now):
    """Generate a menu using holder-local state only; reference is never read.

    The same state and seed produce the same menu for every possible expert
    action. This prevents target repetition and action pairs from encoding it.
    """
    rows = []
    def add(action_id, task_type, target_id, point):
        if not any(row["action_id"] == action_id and row["target_id"] == str(target_id)
                   for row in rows):
            rows.append(_candidate_row(agent, action_id, task_type, str(target_id), point))
    for target in targets:
        actions = _class_actions(target["class"], agent)
        if target["class"] == "human":
            actions = ["A009", "A017", "A019", "A020", "A021", "A022",
                       "A005", _movement_action(agent)]
        for action in actions:
            add(action, _action_task(action, agent), target["entity_id"], target["point"])
    index = _target_index(targets)
    for proposal in task_candidates(agent, sim, limit=8):
        action = _task_action_for(proposal["type"], agent)
        target = index.get(str(proposal["target"]))
        if action in SUBSTANTIVE and target:
            add(action, proposal["type"], target["entity_id"], target["point"])
    # Prefer nominally feasible options, but retain one adversarial guard test.
    feasible = [row for row in rows if _feasible(
        agent, row["action_id"], row["target_id"], index, now)]
    infeasible = [row for row in rows if row not in feasible]
    rng.shuffle(feasible)
    rng.shuffle(infeasible)
    rows = feasible[:MAX_CANDIDATES - bool(infeasible)] + infeasible[:1]
    pose = agent.platform.get("pose") or [0., 0.]
    if len(rows) < 2:
        add("A004", "T019", agent.platform_eid, pose)
        add(_movement_action(agent), _action_task(_movement_action(agent), agent),
            agent.platform_eid, pose)
    rng.shuffle(rows)
    for position, row in enumerate(rows, 1):
        row["candidate_id"] = f"c{position:02d}"
    return rows


def _task_action_for(task_type, agent):
    try:
        method = resolve_method(load_catalog(), task_type, agent.spec)
    except (ValueError, KeyError):
        return None
    steps = method.get("resolved_steps") or []
    return steps[0] if steps else None


# --------------------------------------------------------------------------- rollout

def _state_snapshot(sim, agent_id):
    """Oracle-side physical snapshot (supervision only, never model input)."""
    agent = sim.agents[agent_id]
    world = sim.world
    humans, obstacles, hazards = {}, {}, {}
    for eid, entity in world.entities.items():
        cls = entity.get("cls")
        if cls == "human":
            humans[eid] = {
                "support": sorted(world.get(eid, "human.support_received", []) or []),
                "handover": world.get(eid, "human.handover_status"),
                "carrier": world.get(eid, "human.carrier_id"),
            }
        elif cls == "obstacle":
            obstacles[eid] = float(world.get(eid, "obstacle.work_remaining", 0.0) or 0.0)
        elif cls == "hazard":
            hazards[eid] = {"fire": float(world.get(eid, "hazard.fire_intensity", 0.0) or 0.0),
                            "exclusion": bool(world.get(eid, "hazard.exclusion_zone"))}
    belief_observed = 0
    building = {}
    for (eid, key), entry in agent.belief.entries.items():
        if isinstance(entry, dict) and entry.get("source") not in (None, "prior"):
            belief_observed += 1
            if key == "building.visual_damage":
                building[eid] = entry.get("value")
    payload = world.get(agent.platform_eid, "platform.payload_item_ids", []) or []
    steps = sum(int((task.get("method") or {}).get("step_index", 0) or 0)
                for task in agent.tasks.values())
    return {
        "pose": [float(value) for value in (agent.platform.get("pose") or [0.0, 0.0])][:2],
        "payload_items": len(payload) if isinstance(payload, (list, tuple)) else 0,
        "humans": humans, "obstacles": obstacles, "hazards": hazards,
        "belief_observed": belief_observed, "evidence_count": len(agent.belief.evidence),
        "building_damage": building,
        "task_steps": steps,
    }


def _milestones(before, after):
    """Real physical milestone advances measured on the counterfactual clone."""
    parts = {}
    for eid, row in after["humans"].items():
        old = before["humans"].get(eid, {})
        for tag in sorted(set(row["support"]) - set(old.get("support", []))):
            key = "support:" + str(tag)
            if key in MILESTONE_WEIGHTS:
                parts[key] = parts.get(key, 0.0) + MILESTONE_WEIGHTS[key]
        handover = row.get("handover")
        if handover and handover != old.get("handover"):
            key = "handover:" + str(handover)
            if key in MILESTONE_WEIGHTS:
                parts[key] = parts.get(key, 0.0) + MILESTONE_WEIGHTS[key]
        if row.get("carrier") and not old.get("carrier"):
            parts["loaded"] = parts.get("loaded", 0.0) + MILESTONE_WEIGHTS["loaded"]
    for eid, work in after["obstacles"].items():
        old = before["obstacles"].get(eid)
        if old is not None and work < old - 1e-9:
            fraction = (old - work) / max(old, 1e-9)
            parts["obstacle_progress"] = (parts.get("obstacle_progress", 0.0)
                + MILESTONE_WEIGHTS["obstacle_progress"] * fraction)
    for eid, row in after["hazards"].items():
        old = before["hazards"].get(eid, {})
        old_fire = float(old.get("fire", 0.0) or 0.0)
        if old_fire > row["fire"] + 1e-9:
            fraction = (old_fire - row["fire"]) / max(old_fire, 1e-9)
            parts["fire_suppressed"] = (parts.get("fire_suppressed", 0.0)
                + MILESTONE_WEIGHTS["fire_suppressed"] * fraction)
        if row.get("exclusion") and not old.get("exclusion"):
            parts["exclusion_zone"] = parts.get("exclusion_zone", 0.0) + MILESTONE_WEIGHTS["exclusion_zone"]
    gained = after["payload_items"] - before["payload_items"]
    if gained > 0:
        parts["payload_gain"] = MILESTONE_WEIGHTS["payload_gain"] * min(1, gained)
    observed = max(after["belief_observed"] - before["belief_observed"],
                   after.get("evidence_count", 0) - before.get("evidence_count", 0))
    if observed > 0:
        parts["new_observation"] = MILESTONE_WEIGHTS["new_observation"] * min(5, observed)
    assessed = set(after["building_damage"]) - set(before["building_damage"])
    if assessed:
        parts["building_assessed"] = MILESTONE_WEIGHTS["building_assessed"] * min(2, len(assessed))
    return parts


def _potential(pose, work_points):
    """Net-state potential: normalised task-work-site proximity in [0, 1]."""
    total = 0.0
    for point, weight in work_points:
        if point is None:
            continue
        distance = _distance(pose, point)
        if not math.isfinite(distance):
            continue
        total += weight * (1.0 - min(1.0, distance / POTENTIAL_RANGE_M))
    return total


def _work_points(agent, index, center):
    """A common mission potential derived without reading the expert action."""
    current = agent.tasks.get(agent.current) if getattr(agent, "current", None) else None
    if current and str(current.get("target")) in index:
        return [(index[str(current["target"])]["point"], 1.0)]
    objectives = list(center.get("public_objectives", [])) + list(center.get("known_tasks", []))
    points = {}
    for objective in objectives:
        target = str(objective.get("target_id"))
        if target in index:
            points[target] = (index[target]["point"], max(0., float(objective.get("priority", 50) or 50)))
    total = sum(weight for _, weight in points.values())
    return [(point, weight / total) for point, weight in points.values()] if total else []


def _clone_sim(sim):
    """Copy mutable simulation state, omitting append-only audit archives.

    Native transitions only consume new event/history tails. Catalog/config are
    read-only; world fields, agents, queues and RNG states remain independent.
    The memo avoids temporarily mutating the live simulation during a clone.
    """
    clone = deepcopy(sim, {
        id(sim.catalog): sim.catalog, id(sim.config): sim.config,
        id(sim.events): [], id(sim.ledger.entries): [], id(sim.world.history): [],
    })
    # A copied Python function closure still points at the live collector.
    clone.submit_action = Simulation.submit_action.__get__(clone, Simulation)
    return clone


def _rollout(sim, agent_id, candidate, work_points, nominal_spec=None):
    """Real counterfactual short-horizon rollout on a ``deepcopy(sim)`` clone.

    The native ``MODELS[action_id]`` guard decides acceptance; the clone then
    advances by the action's own planned duration so the physical effect and
    the resulting state transition are the ones the simulator would produce.
    """
    action_id, params = candidate["action_id"], dict(candidate["params"])
    before = _state_snapshot(sim, agent_id)
    potential_before = _potential(before["pose"], work_points)
    result = {
        "action_succeeded": False, "rollout_status": "clone_failed",
        "delta_milestone": 0.0, "delta_potential": 0.0, "delta_cursor": 0.0,
        "cost_penalty": 0.0, "fail_penalty": FAIL_PENALTY,
        "milestones": {}, "duration_s": 0.0, "energy_Wh": 0.0,
        "rejection_code": "clone_failed", "action_status": "clone_failed",
    }
    try:
        clone = _clone_sim(sim)
        if nominal_spec is not None:
            clone.agents[agent_id].spec = deepcopy(nominal_spec)
    except Exception:  # pragma: no cover - defensive, clone failure is recorded
        return result
    try:
        instance_id = Simulation.submit_action(clone, agent_id, action_id, params)
    except Exception as exc:
        result.update(rollout_status="submit_error:" + type(exc).__name__,
                      rejection_code="submit_error")
        return result
    record = clone.instances[instance_id]
    duration = float(record.get("plan", {}).get("duration_s") or 0.0)
    energy = float(record.get("plan", {}).get("energy_Wh") or 0.0)
    result.update(duration_s=round(duration, 3), energy_Wh=round(energy, 4))
    if record["status"] == "rejected":
        code = str(record.get("rejection_code", "unknown"))
        after = _state_snapshot(clone, agent_id)
        result.update(rollout_status="rejected:" + code, rejection_code=code,
                      action_status="rejected",
                      delta_potential=_potential(after["pose"], work_points) - potential_before)
        return result
    window = min(max(duration, 0.0), ROLLOUT_WINDOW_CAP_S)
    if window <= 0.0:
        # A zero-duration action is scheduled at the current instant; advance one
        # tick so the heap fires it instead of leaving it running forever.
        window = float(clone.config.get("tick_s", 5))
    try:
        clone.run(clone.t + window)
    except Exception as exc:  # pragma: no cover - defensive
        result.update(rollout_status="run_error:" + type(exc).__name__,
                      rejection_code="run_error")
        return result
    record = clone.instances[instance_id]
    status = str(record.get("status", "unknown"))
    after = _state_snapshot(clone, agent_id)
    # Subtract a matched no-new-action continuation at the same horizon. Other
    # agents' in-flight rescues and scheduled world events are not this action's
    # progress; longer actions must not receive their milestones for free.
    control = _clone_sim(sim)
    if nominal_spec is not None:
        control.agents[agent_id].spec = deepcopy(nominal_spec)
    control.run(control.t + window)
    control_after = _state_snapshot(control, agent_id)
    action_parts = _milestones(before, after)
    control_parts = _milestones(before, control_after)
    parts = {key: action_parts.get(key, 0.0) - control_parts.get(key, 0.0)
             for key in action_parts.keys() | control_parts.keys()}
    delta_milestone = float(np.clip(sum(parts.values()), -1.0, 1.0))
    delta_potential = (_potential(after["pose"], work_points)
                       - _potential(control_after["pose"], work_points))
    native_result = record.get("result")
    actual_cost = getattr(native_result, "cost", None)
    if actual_cost is not None:
        cost_duration = float(actual_cost.get("time_s", duration))
        cost_energy = float(actual_cost.get("energy_Wh", energy))
    else:
        cost_duration = min(duration, window)
        cost_energy = energy * min(1.0, window / max(duration, 1e-12))
    cost_penalty = min(0.10, 0.03 * cost_duration / 120.0 + 0.02 * cost_energy / 50.0)
    if status == "succeeded":
        fail_penalty, succeeded, rollout_status = 0.0, True, "completed"
    elif status == "partial":
        fail_penalty, succeeded, rollout_status = PARTIAL_PENALTY, False, "partial"
    elif status == "running":
        fail_penalty, succeeded, rollout_status = PARTIAL_PENALTY, False, "window_truncated"
    else:
        fail_penalty, succeeded, rollout_status = FAIL_PENALTY, False, status
    result.update(
        action_succeeded=succeeded, rollout_status=rollout_status, action_status=status,
        delta_milestone=round(delta_milestone, 6),
        delta_potential=round(delta_potential, 6),
        delta_cursor=float(after["task_steps"] - before["task_steps"]),
        cost_penalty=round(cost_penalty, 6), fail_penalty=fail_penalty,
        milestones={key: round(value, 4) for key, value in sorted(parts.items())},
        rejection_code="none" if succeeded else str(record.get("failure_code", "none")),
    )
    return result


def _delta_total(row):
    return (DELTA_WEIGHTS["milestone"] * row["delta_milestone"]
            + DELTA_WEIGHTS["potential"] * row["delta_potential"]
            + DELTA_WEIGHTS["cursor"] * row["delta_cursor"]
            - DELTA_WEIGHTS["cost"] * row["cost_penalty"]
            + DELTA_WEIGHTS["fail"] * row["fail_penalty"])


# --------------------------------------------------------------------------- paired VoI

def _uplink_packet(agent, rows, guided, budget=ISCC_BUDGET_BYTES):
    """Equal-byte report encoders; only the selection order changes."""
    task_priority = {row["target_id"]: _task_priority(row["task_type"]) for row in rows}
    entries = [{"eid": eid, "key": key, **_jsonable(entry)}
               for (eid, key), entry in agent.belief.entries.items()
               if entry.get("source") not in (None, "prior")
               and eid not in {agent.id, agent.platform_eid}]
    if guided:
        entries.sort(key=lambda e: (-task_priority.get(e["eid"], 0), -float(e.get("t", 0)),
                                    e["eid"], e["key"]))
    else:
        entries.sort(key=lambda e: (float(e.get("t", 0)), e["eid"], e["key"]))
    payload = {"op": "report", "entries": [], "evidence": [], "require_ack": False}
    for entry in entries:
        payload["entries"].append(entry)
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > budget:
            payload["entries"].pop()
    return payload


def _policy_choice(rows, agent, memory, now):
    """One fixed, label-blind policy for all four information treatments."""
    index = _target_index(_targets(agent))
    priorities = {str(task.get("target")): float(task.get("priority", 50))
                  for task in agent.tasks.values()
                  if task.get("status") not in {"completed", "canceled", "failed", "expired"}}
    poses = {}
    for slot in memory["slots"]:
        if slot["kind"] == "task":
            task = slot["value"]
            if task.get("status") not in {"completed", "canceled", "failed", "expired"}:
                priorities[str(task["target_id"])] = float(task.get("priority", 50))
        elif slot["kind"] == "entity_pose":
            poses[slot["slot_id"].split(":", 1)[1]] = slot["value"]
    scores = {}
    for row in rows:
        target = row["target_id"]
        point = poses.get(target, (index.get(target) or {}).get("point"))
        if point is None:
            point = row["params"].get("goal")
        distance = _distance(agent.platform.get("pose"), point)
        proximity = 0.0 if not math.isfinite(distance) else 1 - min(1.0, distance / POTENTIAL_RANGE_M)
        score = (.6 * priorities.get(target, _task_priority(row["task_type"])) / 100.0
                 + .3 * proximity + .1 * _novelty((index.get(target) or {}).get("fields", {}), now))
        if not _feasible(agent, row["action_id"], target, index, now):
            score -= 10.0
        scores[row["candidate_id"]] = score
    # Semantic tie breaking is invariant to candidate order and generated IDs.
    return min(rows, key=lambda row: (-scores[row["candidate_id"]], row["action_id"],
                                     row["target_id"]))


def _paired_voi(sim, cfg, rows, agent, work_points):
    """Execute a paired 2x2 communication intervention from one pre-action state.

    First digit: baseline(0) / task-guided(1) equal-budget uplink selection.
    Second digit: no new external slots(0) / delivered center memory(1).
    Both network phases use identical time windows and the SAME edge policy.
    Each selected action then runs on a fresh clone through native guards and
    physical effects; no oracle utility is consulted by either encoder/policy.
    """
    ablation, choices, rollouts, treatments = {}, {}, {}, {}
    for band in ("B00", "B10", "B01", "B11"):
        clone = _clone_sim(sim)
        edge = clone.agents[agent.id]
        edge.spec = deepcopy(agent.spec)
        coordinator = _coordinator(clone)
        guided, downlink = band[1] == "1", band[2] == "1"
        tick_window = 2.0 * float(clone.config.get("tick_s", 5))
        packet = _uplink_packet(edge, rows, guided)
        up_mid = None
        if coordinator is not None:
            up_mid = clone.send_message(edge.id, [coordinator.id], "report", packet,
                len(json.dumps(packet, ensure_ascii=False).encode("utf-8")), ttl_s=2 * tick_window)
        clone.run(clone.t + tick_window)
        _register_center_tasks(clone, coordinator)
        memory = _build_memory({"known_tasks": [], "known_entity_poses": [],
                                "stations": [], "reported_observations": []}, clone.t)
        down_mid = None
        if downlink and coordinator is not None:
            outgoing = _build_memory(_center_context(clone, cfg, edge), clone.t)
            down_mid = clone.send_message(coordinator.id, [edge.id], "report",
                {"op": "phase1_memory", "phase1_memory": outgoing},
                outgoing["serialized_bytes"], ttl_s=2 * tick_window)
        clone.run(clone.t + tick_window)
        delivered = next((m for m in edge.received_messages if m["id"] == down_mid), None)
        if delivered is not None:
            memory = deepcopy(delivered["payload"]["phase1_memory"])
        selected = _policy_choice(rows, edge, memory, clone.t)
        outcome = _rollout(clone, edge.id, selected, work_points, agent.spec)
        outcome["delta_total"] = round(_delta_total(outcome), 6)
        choices[band] = selected["candidate_id"]
        rollouts[band] = outcome
        ablation[band] = outcome["delta_total"]
        up = clone.network.messages.get(up_mid)
        down = clone.network.messages.get(down_mid)
        treatments[band] = {
            "task_guided_uplink": guided, "center_memory_enabled": downlink,
            "uplink_message_id": up_mid, "downlink_message_id": down_mid,
            "uplink_bytes": up.size_bytes if up else 0,
            "uplink_entry_refs": [entry["eid"] + ":" + entry["key"] for entry in packet["entries"]],
            "downlink_bytes": down.size_bytes if down else 0,
            "uplink_delivered": bool(up and up.status.get(coordinator.id) == "delivered"),
            "downlink_delivered": delivered is not None,
            "delivered_memory_slots": len(memory["slots"]),
            "delivered_memory_slot_ids": [slot["slot_id"] for slot in memory["slots"]],
            "decision_time_s": round(clone.t, 6),
        }
    return {
        "utility_local_only": ablation["B00"],
        "utility_with_center_memory": ablation["B01"],
        "vo_memory_vs_local": round(ablation["B01"] - ablation["B00"], 6),
        "vo_memory_given_task_uplink": round(ablation["B11"] - ablation["B10"], 6),
        "vo_task_uplink_given_memory": round(ablation["B11"] - ablation["B01"], 6),
        "ablation_2x2": ablation, "ablation_choices": choices,
        "local_choice": choices["B00"], "center_choice": choices["B01"],
        "ablation_rollouts": rollouts, "channel_treatments": treatments,
        "bit_convention": "first=uplink_selection,second=downlink_memory",
        "decision_rule": "fixed_mission_priority_proximity_novelty/v1",
        "scope": "one_decision_after_two_equal_native_network_windows",
    }


# --------------------------------------------------------------------------- ISCC budget

def _pose_prior(regime):
    prior = POSE_PRIOR[regime]
    return {"regime": regime, "pitch_deg": prior["pitch_deg"],
            "sigma_theta_deg": prior["sigma_theta_deg"], "sigma_t_m": prior["sigma_t_m"],
            "source": "blueprint_regime_prior", "note_zh": prior["note_zh"]}


def _camera_point(point, pose, yaw, pitch):
    """Real ENU offset of a target expressed in the receiving camera frame."""
    delta = np.array([point[0] - pose[0], point[1] - pose[1], 0.0 - pose[2]], dtype=float)
    forward = np.array([math.cos(pitch) * math.cos(yaw),
                        math.cos(pitch) * math.sin(yaw), math.sin(pitch)], dtype=float)
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0], dtype=float)
    up = np.cross(right, forward)
    return np.array([float(delta @ right), float(delta @ up), float(delta @ forward)])


def _iscc_budget_plan(agent, views, index, regime, candidate_targets):
    """Ternary budget split from real anchor geometry (no learned backbone here)."""
    prior = POSE_PRIOR[regime]
    if not views or not views[0]["visible_targets"]:
        return {"status": "no_visible_anchor_pool", "regime": regime,
                "reason_zh": "本断面没有可见目标锚点，无法执行 Algorithm 1",
                "budget_bytes": ISCC_BUDGET_BYTES}
    intrinsics = views[0]["intrinsics"]
    extrinsics = views[0]["extrinsics"]
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    pose = extrinsics["position_enu"]
    yaw = math.radians(float(extrinsics["yaw_deg"]))
    pitch = math.radians(float(extrinsics["pitch_deg"]))
    points, gammas, features, entity_ids, grads = [], [], [], [], []
    for row in views[0]["visible_targets"]:
        target = index.get(row["entity_id"])
        if target is None:
            continue
        camera = _camera_point(target["point"], pose, yaw, pitch)
        if camera[2] <= 1.0:
            continue
        depth = float(camera[2])
        extent = CLASS_EXTENT_M.get(str(row.get("class")), 2.0)
        projected = np.array([fx * extent / depth, fy * extent / depth], dtype=float)
        points.append(camera)
        gammas.append(np.diag(projected ** 2))
        grads.append(float(math.hypot(projected[0], projected[1])))
        bearing = math.radians(float(row.get("bearing_deg", 0.0)))
        features.append([camera[0] / depth, camera[1] / depth, 1.0 / depth,
                         math.sin(bearing), math.cos(bearing),
                         1.0 if row.get("class") == "human" else 0.0,
                         1.0 if row.get("class") == "building" else 0.0,
                         1.0 if not row.get("observed") else 0.2])
        entity_ids.append(row["entity_id"])
    if not points:
        return {"status": "no_in_front_anchor", "regime": regime,
                "reason_zh": "可见目标均位于接收端视锥后方", "budget_bytes": ISCC_BUDGET_BYTES}
    points = np.asarray(points, dtype=float)
    gammas = np.asarray(gammas, dtype=float)
    grads = np.asarray(grads, dtype=float)
    features = np.asarray(features, dtype=float).T  # (r, m)
    jacobians = iscc_core.se3_projection_jacobian(points, fx, fy)
    rank = features.shape[0]
    selected = [entity_ids.index(target) for target in candidate_targets if target in entity_ids]
    if not selected:
        selected = list(range(min(3, len(entity_ids))))
    x_t = features[:, selected[:4]]
    v_query = np.linalg.qr(x_t)[0][:, :x_t.shape[1]]
    sigma_theta = math.radians(float(prior["sigma_theta_deg"]))
    sigma_xi = np.diag([sigma_theta ** 2] * 3 + [float(prior["sigma_t_m"]) ** 2] * 3)
    try:
        plan = iscc_core.solve_ternary_budget_alg1(
            ISCC_BUDGET_BYTES, sigma_xi, np.eye(rank), v_query,
            {"G": jacobians, "Gamma": gammas},
            {"X": features, "G": jacobians, "grad": grads},
            x_t, 0.05, G_T=jacobians[selected[:4]])
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"status": "solver_error", "regime": regime,
                "reason_zh": type(exc).__name__, "budget_bytes": ISCC_BUDGET_BYTES}
    return {
        "status": "proxy_solved",
        "regime": regime,
        "budget_bytes": int(plan["budget_bytes"]),
        "b_g_star": round(float(plan["b_g_star"]), 3),
        "b_a_star": round(float(plan["b_a_star"]), 3),
        "b_s_star": round(float(plan["b_s_star"]), 3),
        "J_proxy": round(float(plan["J_proxy"]), 6),
        "pose_trace": round(float(plan["pose_trace"]), 6),
        "m_star": int(plan["m_star"]),
        "anchor_pool_size": int(len(entity_ids)),
        "geometric_anchor_count": len(plan["geometric_anchor_ids"]),
        "representation_anchor_count": len(plan["representation_anchor_ids"]),
        "condition_number_mean": None,
        "homography_status": "unmeasured_without_paired_rgbd_and_surface_normals",
        "sigma_theta_deg": prior["sigma_theta_deg"],
        "sigma_t_m": prior["sigma_t_m"],
        "feature_representation": "handcrafted_geometric_descriptor_r8",
        "evidence_scope": "authored_camera_and_belief_geometry_proxy_only",
        "learned_features": False, "cross_view_geometry_measured": False,
        "feature_note_zh": "本试运行无学习骨干，锚点特征为真实几何描述子；不冒充 VLM 特征",
    }


# --------------------------------------------------------------------------- record builder

def _intent(agent, cfg, scene, now, known_tasks):
    """Intent text: scene, time, embodiment, active mission categories only."""
    label = str((cfg.get("display") or {}).get("mission_label")
                or cfg.get("public_brief_zh", "")[:40] or "rescue")
    objectives = []
    for task in known_tasks:
        name = _task_name(task.get("task_type"))
        if name and name not in objectives:
            objectives.append(name)
    capabilities = ", ".join(sorted(agent.spec.get("capabilities", []))[:5])
    return (f"Scene {scene} {label} at t={now:.1f}s. "
            f"Agent {agent.id} ({agent.spec.get('mobility')}, payload "
            f"{float(agent.spec.get('payload_kg', 0.0) or 0.0):.0f}kg, capabilities {capabilities}). "
            f"Active mission objectives: {', '.join(objectives[:3])}. "
            f"Select the next executable candidate action.")


def _build_record(sim, cfg, agent, scene, seed, split, reference, source, driver):
    now = float(sim.t)
    view = _agent_view(agent, driver)
    targets = _targets(view)
    index = _target_index(targets)
    record_id = make_record_id(scene, seed, now, view.id)
    rng = random.Random(f"{scene}|{seed}|{record_id}")
    if reference is not None:
        reference = dict(reference)
        reference["executed_action_id"] = reference["action_id"]
        reference["action_id"] = _adapt_movement(reference["action_id"], view)
        reference["task_type"] = _action_task(reference["action_id"], view)
        reference["priority"] = _task_priority(reference["task_type"])

    memory = _delivered_memory(view, now)
    center = _context_from_memory(_center_context(sim, cfg, view), memory)
    rows = _candidates(sim, view, reference, rng, targets, now)
    if reference is None:
        feasible = [row for row in rows
                    if _feasible(view, row["action_id"], row["target_id"], index, now)]
        best = max(feasible or rows, key=lambda row: (
            _task_priority(row["task_type"]),
            -_distance(view.platform.get("pose"), index.get(row["target_id"], {}).get("point"))))
        reference = {"action_id": best["action_id"], "task_type": best["task_type"],
                     "target_id": best["target_id"], "priority": _task_priority(best["task_type"])}
    reference_id = next((row["candidate_id"] for row in rows
                         if row["action_id"] == reference["action_id"]
                         and row["target_id"] == str(reference["target_id"])), None)
    work_points = _work_points(view, index, center)
    utilities = {}
    for row in rows:
        rollout = _rollout(sim, view.id, row, work_points, view.spec)
        rollout["delta_total"] = round(_delta_total(rollout), 6)
        utilities[row["candidate_id"]] = rollout
    best_id = min(rows, key=lambda row: (-utilities[row["candidate_id"]]["delta_total"],
                                         row["action_id"], row["target_id"]))["candidate_id"]
    best_total = utilities[best_id]["delta_total"]
    acceptable = [row["candidate_id"] for row in rows
                  if best_total - utilities[row["candidate_id"]]["delta_total"]
                  <= ACCEPT_EPSILON + 1e-9]
    voi = _paired_voi(sim, cfg, rows, view, work_points)
    views = _camera_views(view, cfg, targets, len(rows), now)
    regime = REGIME_BY_SCENE.get(scene, "A")
    budget_plan = _iscc_budget_plan(view, views, index, regime,
                                    [row["target_id"] for row in rows])
    record = {
        "record_id": record_id,
        "schema": SCHEMA_ID,
        "metadata": {"record_id": record_id, "scene_id": scene, "seed": int(seed),
                     "split": split, "sim_time_s": round(now, 3), "agent_id": view.id,
                     "source": source},
        "model_input": {
            "intent_text": _intent(view, cfg, scene, now, center["known_tasks"]),
            "center_context": _sanitize(center),
            "center_memory": memory,
            "edge_state": _sanitize(_edge_state(view, cfg, targets, now)),
            "camera_views": _sanitize(views),
            "candidates": rows,
        },
        "supervision": {
            "candidate_utilities": utilities,
            "best_candidate_id": best_id,
            "acceptable_set": acceptable,
            "reference_candidate_id": reference_id,
            "reference_source": source,
            "paired_voi_eval": voi,
            "regime": regime,
            "pose_prior": _pose_prior(regime),
            "budget_plan": budget_plan,
            "delta_weights": dict(DELTA_WEIGHTS),
            "accept_epsilon": ACCEPT_EPSILON,
        },
    }
    return validate_decision_record(_jsonable(record))


# --------------------------------------------------------------------------- episode runner

def _report_actions(sim, rows, seen):
    coordinator = _coordinator(sim)
    if coordinator is None:
        return len(rows)
    for row in rows[seen:]:
        if row.get("actor") == coordinator.id:
            continue
        instance = sim.instances.get(row.get("id")) or {}
        result = instance.get("result")
        observations = [o.to_dict() if hasattr(o, "to_dict") else o
                        for o in (getattr(result, "observations", None) or [])]
        if not observations:
            continue
        payload = {"op": "report", "observations": _jsonable(observations),
                   "entries": [], "evidence": []}
        size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
        sim.send_message(row["actor"], [coordinator.id], "report", payload, size, ttl_s=180)
    if sim.t - getattr(sim, "_phase1_last_memory_s", -math.inf) >= 30.0:
        _register_center_tasks(sim, coordinator)
        memory = _build_memory(_center_context(sim, {}, coordinator), sim.t)
        for edge in sim.agents.values():
            if edge.kind != "coordinator":
                sim.send_message(coordinator.id, [edge.id], "report",
                    {"op": "phase1_memory", "phase1_memory": memory},
                    memory["serialized_bytes"], ttl_s=180)
        sim._phase1_last_memory_s = sim.t
    return len(rows)


def _seed_disruptions(seed):
    # Seeded link outage: the centre misses some delivered reports while the
    # edge keeps its local observations.
    return [{"t": 20 + 12 * seed, "kind": "link_outage", "target": None,
             "params": {"duration_s": 30}}]


def _run(scene, seed, cfg, base_core, on_action=None, cadence=(), on_cadence=None):
    core = deepcopy(base_core)
    core["seed"] = int(seed)
    core["disruptions"] = list(core.get("disruptions", [])) + _seed_disruptions(seed)
    env = Environment.from_config(core)
    sim = env.sim
    original = sim.submit_action
    events = []

    queued = []
    ready = False

    def submit(agent_id, action_id, params, task_ref=None):
        event = {"t": round(float(sim.t), 3), "agent": agent_id, "action": action_id,
                 "params": _jsonable(params), "task_ref": task_ref}
        events.append(event)
        if on_action is not None:
            # The driver submits its first action inside __init__; defer record
            # building until the driver exists.  Sim time has not advanced, so
            # the captured state is still the pre-action decision state.
            if ready:
                on_action(len(events) - 1, event, sim, driver)
            else:
                # The driver submits its first action inside __init__.  Snapshot
                # the pre-action state now; replaying the callback later must not
                # observe the agent already busy with that same action.
                queued.append((len(events) - 1, event, deepcopy(sim)))
        return original(agent_id, action_id, params, task_ref=task_ref)

    sim.submit_action = submit
    _register_center_tasks(sim, _coordinator(sim))
    driver = ForcedRuleDriver(env, cfg)
    ready = True
    for index, event, snapshot in queued:
        on_action(index, event, snapshot, driver)
    queued.clear()
    seen = 0
    pending = list(cadence)
    next_cadence = 0
    try:
        while not driver.done and sim.t < float(cfg.get("until_s", 3600)):
            driver.advance(sim.t + float(cfg.get("tick_s", 1)))
            seen = _report_actions(sim, driver.report["actions"], seen)
            while next_cadence < len(pending) and sim.t >= pending[next_cadence]:
                if on_cadence is not None:
                    on_cadence(next_cadence, sim, driver)
                next_cadence += 1
    finally:
        sim.submit_action = original
    return sim, events, round(float(sim.t), 3)


def _thin(indices, limit):
    if len(indices) <= limit:
        return list(indices)
    step = (len(indices) - 1) / (limit - 1)
    picked = {indices[round(i * step)] for i in range(limit)}
    return sorted(picked)


def _pick_agent(sim, position):
    agents = [agent for agent in sorted(sim.agents.values(), key=lambda a: a.id)
              if agent.kind != "coordinator"]
    preferred = [agent for agent in agents
                 if agent.tasks or task_candidates(agent, sim, limit=1)]
    pool = preferred or agents
    return pool[position % len(pool)] if pool else None


def _reference_from_event(event, agent):
    action_id = event["action"]
    task_type = _action_task(action_id, agent)
    target_id = event["params"].get("target")
    if not target_id:
        goal = event["params"].get("goal")
        if isinstance(goal, (list, tuple)) and len(goal) >= 2:
            target_id = f"point_{float(goal[0]):.1f}_{float(goal[1]):.1f}"
        else:
            target_id = agent.current or agent.platform_eid
    return {"action_id": action_id, "task_type": task_type, "target_id": str(target_id),
            "params": event["params"], "priority": _task_priority(task_type)}


def collect_episode(scene, seed, cfg, base_core):
    started = time.perf_counter()
    split = SPLITS[scene]
    _, events, end_s = _run(scene, seed, cfg, base_core)
    substantive = [index for index, event in enumerate(events)
                   if event["action"] in SUBSTANTIVE]
    selected = set(_thin(substantive, MAX_RECORDS_PER_EPISODE - CADENCE_FLOOR))
    cadence_count = max(CADENCE_FLOOR, MAX_RECORDS_PER_EPISODE - len(selected))
    cadence_count = min(cadence_count, MAX_RECORDS_PER_EPISODE - len(selected))
    rng = random.Random(f"{scene}|{seed}|cadence")
    cadence = sorted({round(rng.uniform(0.05 * end_s, 0.95 * end_s), 1)
                      for _ in range(cadence_count)})
    while len(cadence) < cadence_count:
        cadence.append(round(rng.uniform(0.05 * end_s, 0.95 * end_s), 1))
    cadence = sorted(set(cadence))

    records = []
    deferred = []
    seen_ids = set()

    def keep(record):
        if record["record_id"] in seen_ids:
            return
        seen_ids.add(record["record_id"])
        records.append(record)

    def add_record(record, forced=False):
        if record["record_id"] in seen_ids:
            return
        if forced:
            reference_id = record["supervision"]["reference_candidate_id"]
            status = (record["supervision"]["candidate_utilities"][reference_id]["rollout_status"]
                      if reference_id is not None else "not_in_menu")
            if status.startswith("rejected"):
                # The expert needed a driver capability/mobility override that
                # the nominal embodiment does not own; this is not a valid
                # expert label.  Keep it only if coverage would fall short.
                deferred.append(record)
                return
        keep(record)

    def on_action(index, event, sim, driver):
        if index not in selected:
            return
        agent = sim.agents[event["agent"]]
        reference = _reference_from_event(event, agent)
        add_record(_build_record(sim, cfg, agent, scene, seed, split, reference,
                                 "forced_rule_action", driver), forced=True)

    def on_cadence(index, sim, driver):
        agent = _pick_agent(sim, index + seed)
        if agent is None:
            return
        add_record(_build_record(sim, cfg, agent, scene, seed, split, None,
                                 "cadence_action_menu", driver))

    _run(scene, seed, cfg, base_core, on_action=on_action, cadence=cadence,
         on_cadence=on_cadence)
    dropped = len(deferred)
    for record in sorted(deferred, key=lambda row: (row["metadata"]["sim_time_s"],
                                                    row["record_id"])):
        if len(records) >= MIN_RECORDS_PER_EPISODE:
            break
        # Coverage fallback: keep the section but flag that the expert action was
        # only executable with a driver capability/mobility override.
        record["supervision"]["reference_source"] = "forced_rule_action_nominal_infeasible"
        keep(record)
    records.sort(key=lambda record: (record["metadata"]["sim_time_s"], record["record_id"]))
    return records, {"scene_id": scene, "seed": seed, "split": split,
                     "end_s": end_s, "records": len(records),
                     "forced_actions": len(selected), "cadence": len(cadence),
                     "reference_rejected_dropped": dropped,
                     "wall_s": round(time.perf_counter() - started, 3)}


def _prepare(scene, workdir):
    path = ROOT / "configs" / CONFIG_OVERRIDES.get(scene, f"{scene}.json")
    cfg = expand(json.loads(path.read_text(encoding="utf-8")))
    directory = workdir / scene
    _, bundle = materialize(cfg, directory)
    core = deepcopy(bundle["core"])
    core["semantics"]["path"] = str(directory / core["semantics"]["path"])
    return cfg, core


def _collect_episode_worker(args):
    scene, seed, cfg, core = args
    return scene, seed, *collect_episode(scene, seed, cfg, core)


def collect(out_dir=DATA_DIR, scenes=SCENES, seeds=SEEDS, workers=8):
    from concurrent.futures import ProcessPoolExecutor
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=".phase1-pilot-", dir=out_dir))
    records = []
    episodes = []
    try:
        tasks = []
        for scene in scenes:
            cfg, core = _prepare(scene, workdir)
            for seed in seeds:
                tasks.append((scene, seed, cfg, core))
        if workers and workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
                for scene, seed, episode_records, stats in pool.map(_collect_episode_worker, tasks):
                    records.extend(episode_records)
                    episodes.append(stats)
                    print(f"[phase1] {scene} seed={seed} split={SPLITS[scene]} "
                          f"records={stats['records']} end_s={stats['end_s']:.1f} "
                          f"wall={stats['wall_s']:.2f}s", flush=True)
        else:
            for scene, seed, cfg, core in tasks:
                episode_records, stats = collect_episode(scene, seed, cfg, core)
                records.extend(episode_records)
                episodes.append(stats)
                print(f"[phase1] {scene} seed={seed} split={SPLITS[scene]} "
                      f"records={stats['records']} end_s={stats['end_s']:.1f} "
                      f"wall={stats['wall_s']:.2f}s", flush=True)
    finally:
        resolved = workdir.resolve()
        if resolved.parent != out_dir.resolve() or not resolved.name.startswith(".phase1-pilot-"):
            raise ValueError("unexpected collector scratch directory")
        shutil.rmtree(resolved, ignore_errors=True)

    jsonl = out_dir / "decision_records.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    wall = time.perf_counter() - started
    summary = summarize(records, episodes, wall)
    (out_dir / "pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize(records, episodes, wall_s):
    counts = [len(record["model_input"]["candidates"]) for record in records]
    per_episode = [episode["records"] for episode in episodes]
    by_scene, by_split, by_regime = {}, {}, {}
    task_labels, reference_sources, rollout_status = {}, {}, {}
    positives = negatives = reference_best = 0
    accepted_sizes, voi_values, memory_bytes, budget_splits = [], [], [], []
    for record in records:
        metadata = record["metadata"]
        supervision = record["supervision"]
        model_input = record["model_input"]
        by_scene[metadata["scene_id"]] = by_scene.get(metadata["scene_id"], 0) + 1
        by_split[metadata["split"]] = by_split.get(metadata["split"], 0) + 1
        by_regime[supervision["regime"]] = by_regime.get(supervision["regime"], 0) + 1
        reference_sources[supervision["reference_source"]] = \
            reference_sources.get(supervision["reference_source"], 0) + 1
        reference_best += int(supervision["reference_candidate_id"]
                              == supervision["best_candidate_id"])
        accepted_sizes.append(len(supervision["acceptable_set"]))
        voi_values.append(supervision["paired_voi_eval"]["vo_memory_vs_local"])
        memory_bytes.append(model_input["center_memory"]["serialized_bytes"])
        if supervision["budget_plan"].get("status") == "proxy_solved":
            budget_splits.append((supervision["budget_plan"]["b_g_star"],
                                  supervision["budget_plan"]["b_a_star"],
                                  supervision["budget_plan"]["b_s_star"]))
        for cid, row in supervision["candidate_utilities"].items():
            rollout_status[row["rollout_status"]] = rollout_status.get(row["rollout_status"], 0) + 1
            if cid == supervision["best_candidate_id"]:
                positives += 1
            else:
                negatives += 1
        best = next(row for row in model_input["candidates"]
                    if row["candidate_id"] == supervision["best_candidate_id"])
        task_labels[best["task_type"]] = task_labels.get(best["task_type"], 0) + 1
    distribution = {}
    for count in counts:
        distribution[str(count)] = distribution.get(str(count), 0) + 1
    solved = len(budget_splits)
    return {
        "schema": "aerorescue.phase1.pilot/v2",
        "record_schema": SCHEMA_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_episodes": len({(record["metadata"]["scene_id"], record["metadata"]["seed"])
                               for record in records}),
        "total_records": len(records),
        "scenes": sorted({record["metadata"]["scene_id"] for record in records}),
        "seeds": sorted({record["metadata"]["seed"] for record in records}),
        "splits": by_split,
        "regimes": by_regime,
        "records_by_scene": by_scene,
        "records_by_split": by_split,
        "records_per_episode": {
            "min": min(per_episode) if per_episode else 0,
            "max": max(per_episode) if per_episode else 0,
            "mean": round(sum(per_episode) / len(per_episode), 2) if per_episode else 0.0,
        },
        "candidates_per_record": {
            "min": min(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / len(counts), 2) if counts else 0.0,
            "distribution": distribution,
        },
        "label_balance": {
            "positive_candidates": positives,
            "negative_candidates": negatives,
            "positive_ratio": round(positives / (positives + negatives), 4)
            if positives + negatives else 0.0,
            "reference_is_best": reference_best,
            "reference_not_best": len(records) - reference_best,
        },
        "acceptable_set": {
            "mean_size": round(sum(accepted_sizes) / len(accepted_sizes), 3)
            if accepted_sizes else 0.0,
            "max_size": max(accepted_sizes) if accepted_sizes else 0,
            "epsilon": ACCEPT_EPSILON,
        },
        "paired_voi": {
            "mean_vo_memory_vs_local": round(sum(voi_values) / len(voi_values), 6)
            if voi_values else 0.0,
            "positive_vo_records": sum(1 for value in voi_values if value > 1e-9),
            "nonzero_vo_records": sum(1 for value in voi_values if abs(value) > 1e-9),
        },
        "center_memory": {
            "budget_bytes": MEMORY_BUDGET_BYTES,
            "mean_serialized_bytes": round(sum(memory_bytes) / len(memory_bytes), 1)
            if memory_bytes else 0.0,
            "max_serialized_bytes": max(memory_bytes) if memory_bytes else 0,
        },
        "iscc_budget": {
            "budget_bytes": ISCC_BUDGET_BYTES,
            "solved_records": 0,
            "proxy_solved_records": solved,
            "evidence_scope": "pipeline_diagnostic_not_blueprint_theorem_experiment",
            "mean_b_g_star": round(sum(row[0] for row in budget_splits) / solved, 2) if solved else 0.0,
            "mean_b_a_star": round(sum(row[1] for row in budget_splits) / solved, 2) if solved else 0.0,
            "mean_b_s_star": round(sum(row[2] for row in budget_splits) / solved, 2) if solved else 0.0,
        },
        "rollout_status": dict(sorted(rollout_status.items())),
        "reference_sources": reference_sources,
        "best_task_types": dict(sorted(task_labels.items())),
        "collection_wall_s": round(wall_s, 3),
        "delta_weights": dict(DELTA_WEIGHTS),
        "episodes": episodes,
        "notes": [
            "delta_total = 0.60*delta_milestone + 0.40*delta_potential "
            "- cost_penalty + fail_penalty; delta_cursor weight is exactly 0.0.",
            "Every candidate is rolled out on deepcopy(sim) through the native "
            "MODELS[action_id] guard and physical effect; no stage-alignment formula.",
            "candidates carry no source/feasible hint; intent_text carries no action id, "
            "skill name or target id.",
            "center_context is built only from coordinator belief/tasks plus public static "
            "config; station stock/occupancy is marked not_delivered when unknown.",
            "camera_views use authored intrinsics and platform/default orientation with "
            "render_status=metadata_only; no PNG or paired RGB-D was rendered.",
            "iscc budget_plan is an explicitly marked geometry-proxy pipeline diagnostic; "
            "its extent-based Gamma and handcrafted descriptors are not measured image "
            "gradients, VLM features or cross-view homography evidence.",
            "Menus, camera orientation, mission potential, and center tasks never depend "
            "on the expert action; absent references remain null diagnostics.",
            "Model context contains only public mission scope, local belief/telemetry, "
            "and bounded center memory already delivered by the native network.",
            "Every B-band executes native equal-budget uplink selection, optional "
            "downlink delivery, one fixed policy, and a separate action rollout. Bits "
            "are uplink then downlink; memory VoI is B01-B00.",
            "Action progress subtracts a no-new-action continuation at the same horizon; "
            "this is a one-decision pilot, not a VLM or multi-episode efficacy result.",
        ],
    }
