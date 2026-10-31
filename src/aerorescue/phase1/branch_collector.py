"""Phase-1 contract ②: pure-core cloning, stable exogenous CRN and branch rollouts.

Three mechanisms live here, and none of them may touch an existing file:

``clone_core_env(env)`` / ``verify_clone_isolation(env)``
    Deep-copies only the live ``Environment`` core state, strips every external
    callback hook (a wrapper's ``sim.submit_action`` closure, a driver's
    ``agent.on_tick`` stub) and rebinds the event heap and ``sim`` references.
    Four isolation assertions are machine-checked, not assumed.

``StableExogenousRNG``
    An exogenous random stream keyed by
    ``(scene_id, seed, actor_id, event_family, purpose, target_anchor,
    quantized_t, local_seq)``.  It has no ``action_id`` input at all, so a random
    draw cannot move when a branch's control flow changes.

``LegitimateOptionRollout``
    Executes K named macro-action options through the *native* ``MODELS[...]``
    guard on a clone and records de-duplicated named rewards at 60/120/240 s.
    It never adds an ``air`` mobility override or a borrowed capability the way
    ``ForcedRuleDriver`` does: every primitive is a nominal action of the agent's
    own embodiment, and a missing capability simply makes that primitive
    infeasible.

``MatchedHiddenStateBuilder``
    Builds the matched hidden-state pair of contract ①: two worlds that differ
    only behind the ground agent's foreground building (>15 m past the occluding
    face) in which corridor A or corridor B is the one that is open and holds the
    casualty.  The ground view, ground history and every ego input stay identical
    while the aerial view sees both corridors.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import types
from copy import deepcopy
from pathlib import Path

import numpy as np

# Importing the api package first keeps the actions <-> agents circular import
# resolved in the order the rest of the project uses.
from rescue_scenes.core.api import Environment
from rescue_scenes.core.sim.kernel import Simulation
from rescue_scenes.core.agents.runtime import AgentRuntime
from rescue_scenes.core.sim.actions import MODELS, RECEIVING_STATION_KINDS

from aerorescue.phase1 import schema_v2 as S

# --------------------------------------------------------------------------- geometry
EYE_HEIGHT_M = 1.6
UAV_SURVEY_ALT_M = 120.0
OCCLUSION_DEPTH_M = 18.0          # modifications start this far past the far face
MIN_AERIAL_CLEARANCE_M = 3.0      # the aerial ray must clear the roof by this much
BOUNDING_MARGIN_M = 2.0
ROADBLOCK_HALF_M = 3.0
ROADBLOCK_HEIGHT_M = 1.1
ROADBLOCK_WORK = 180.0
MAX_BELIEVED_ENTITIES = 80

# --------------------------------------------------------------------------- rollouts
HORIZON_S = 240.0
SLICES_S = (60.0, 120.0, 240.0)
SLICE_NAMES = ("H_60s", "H_120s", "H_240s")
PRIMARY_SLICE = "H_120s"
TICK_FALLBACK_S = 1.0
CONTACT_RADIUS_M = 35.0
SENSING_RANGE_M = 180.0
MAX_STAGE_RETRIES = 1

# Named macro-action options; ``schema_v2.MACRO_OPTION_IDS`` is the authority on
# the names, this table is the execution plan for each one.
OPTION_PLANS = {
    "OPT_ROUTE_A_SEARCH_RESCUE": ("A", ("move", "capture", "integrate", "approach_local", "contact",
                                       "integrate", "extricate", "aid", "load",
                                       "transport", "deliver", "scan")),
    "OPT_ROUTE_B_SEARCH_RESCUE": ("B", ("move", "capture", "integrate", "approach_local", "contact",
                                       "integrate", "extricate", "aid", "load",
                                       "transport", "deliver", "scan")),
    "OPT_CLEAR_CORRIDOR_AND_ADVANCE": ("J", ("move", "capture", "clear", "advance")),
    "OPT_DIRECT_APPROACH_AND_AID": ("CUE", ("move", "capture", "integrate", "contact", "aid")),
    "OPT_HOLD_AND_SCAN": (None, ("scan", "scan", "scan", "scan")),
}
MOVE_ACTION = {"air": "A001", "water": "A003"}
ACTION_TASK_TYPE = {"A001": "T020", "A002": "T020", "A003": "T020", "A013": "T023",
                    "A012": "T022", "A009": "T027", "A005": "T011"}
PRIMARY_ACTION = {"OPT_ROUTE_A_SEARCH_RESCUE": None,       # resolved by mobility
                  "OPT_ROUTE_B_SEARCH_RESCUE": None,
                  "OPT_CLEAR_CORRIDOR_AND_ADVANCE": "A013",
                  "OPT_DIRECT_APPROACH_AND_AID": "A009",
                  "OPT_HOLD_AND_SCAN": "A005"}
TARGET_ANCHOR = {"OPT_ROUTE_A_SEARCH_RESCUE": "corridor_a",
                 "OPT_ROUTE_B_SEARCH_RESCUE": "corridor_b",
                 "OPT_CLEAR_CORRIDOR_AND_ADVANCE": "corridor_junction",
                 "OPT_DIRECT_APPROACH_AND_AID": "cue_nearest",
                 "OPT_HOLD_AND_SCAN": "self_hold"}

# Authored pose priors (blueprint §四.2), used only to draw the noisy aerial
# extrinsic prior; they are never measured calibration.
POSE_PRIOR = {
    "A": (math.radians(2.25), 1.0),
    "B": (math.radians(1.0), 0.6),
    "C": (math.radians(3.0), 1.5),
    "water": (math.radians(2.0), 1.2),
}

MISSION_BRIEF = ("震后城区空地协同搜救：沿可用通路搜索被困人员，完成接触、解困、急救、"
                 "装载、转运与交接；通信受限时必须依据本机观测独立判断。")


# --------------------------------------------------------------------------- json / hash
def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    # Never leak a memory address into a digest: an opaque value is identified
    # by its type, which is stable across clones.
    return "<" + type(value).__name__ + ">"


def canonical_json(value) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- polygons / LOS
def _polygon_of(coords):
    points = [list(map(float, point[:2])) for point in coords]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    return points


def _point_in_polygon(polygon, point) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            crossing = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing:
                inside = not inside
    return inside


def _segment_polygon_params(a, b, polygon):
    """Sorted parameters at which segment a->b crosses the polygon boundary."""
    ts = [0.0, 1.0]
    rx, ry = b[0] - a[0], b[1] - a[1]
    count = len(polygon)
    for index in range(count):
        cx, cy = polygon[index]
        dx, dy = polygon[(index + 1) % count]
        sx, sy = dx - cx, dy - cy
        denominator = rx * sy - ry * sx
        if abs(denominator) < 1e-12:
            continue
        t = ((cx - a[0]) * sy - (cy - a[1]) * sx) / denominator
        u = ((cx - a[0]) * ry - (cy - a[1]) * rx) / denominator
        if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
            ts.append(min(1.0, max(0.0, t)))
    return sorted(ts)


def los_clearance(eye, target, occluders) -> float:
    """Minimum vertical clearance of the 3-D ray above every occluder roof.

    ``occluders`` is an iterable of ``(entity_id, polygon_2d, height_m)``.
    Returns ``+inf`` when the ray never enters a footprint and a negative value
    (metres below the roof) when it is blocked.
    """
    clearance = math.inf
    for _entity_id, polygon, height in occluders:
        params = _segment_polygon_params(eye[:2], target[:2], polygon)
        for t0, t1 in zip(params, params[1:]):
            if t1 - t0 < 1e-9:
                continue
            mid = [eye[k] + (target[k] - eye[k]) * (t0 + t1) / 2.0 for k in range(3)]
            if not _point_in_polygon(polygon, mid[:2]):
                continue
            # Height is linear in t, so the worst case is at an interval end.
            z0 = eye[2] + (target[2] - eye[2]) * t0
            z1 = eye[2] + (target[2] - eye[2]) * t1
            clearance = min(clearance, min(z0, z1) - float(height))
    return clearance


def los_blockers(eye, target, occluders):
    blocked = []
    for entity_id, polygon, height in occluders:
        params = _segment_polygon_params(eye[:2], target[:2], polygon)
        for t0, t1 in zip(params, params[1:]):
            if t1 - t0 < 1e-9:
                continue
            mid = [eye[k] + (target[k] - eye[k]) * (t0 + t1) / 2.0 for k in range(3)]
            if not _point_in_polygon(polygon, mid[:2]):
                continue
            z0 = eye[2] + (target[2] - eye[2]) * t0
            z1 = eye[2] + (target[2] - eye[2]) * t1
            if min(z0, z1) < float(height):
                blocked.append(entity_id)
                break
    return blocked


def building_occluders(env):
    occluders = []
    for entity_id, entity in env.sim.world.entities.items():
        if entity.get("cls") != "building":
            continue
        polygon = _polygon_of(entity["geometry"]["coords"])
        if len(polygon) < 3:
            continue
        floors = env.sim.world.get(entity_id, "building.floors", 1)
        try:
            height = 3.0 * max(1.0, float(floors or 1))
        except (TypeError, ValueError):
            height = 3.0
        explicit = env.sim.world.get(entity_id, "attr.height_m")
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            height = max(height, float(explicit))
        occluders.append((entity_id, polygon, height))
    return occluders


def _agent_point(agent):
    pose = agent.platform.get("pose") or [0.0, 0.0]
    return [float(pose[0]), float(pose[1])]


def ground_eye(agent):
    point = _agent_point(agent)
    return [point[0], point[1], EYE_HEIGHT_M]


def uav_eye(agent, altitude=UAV_SURVEY_ALT_M):
    point = _agent_point(agent)
    return [point[0], point[1], float(altitude)]


def _bearing(origin, target) -> float:
    return math.atan2(target[1] - origin[1], target[0] - origin[0])


# --------------------------------------------------------------------------- SE(3)
def camera_extrinsics(eye, yaw, pitch, roll=0.0):
    """Camera-to-world for an optical frame (x right, y down, z forward) in ENU."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.array([cp * cy, cp * sy, sp], dtype=float)
    right = np.array([sy, -cy, 0.0], dtype=float)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    if roll:
        spin = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        rotation = rotation @ spin
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-9, "camera frame must be right handed"
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(eye, dtype=float)
    return transform


def se3_log(transform) -> np.ndarray:
    """Log map of SE(3): returns xi = [omega, upsilon] in R^6."""
    transform = np.asarray(transform, dtype=float)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    cosine = max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0))
    theta = math.acos(cosine)
    if theta < 1e-8:
        omega = 0.5 * np.array([rotation[2, 1] - rotation[1, 2],
                                rotation[0, 2] - rotation[2, 0],
                                rotation[1, 0] - rotation[0, 1]])
        upsilon = translation.copy()
        return np.concatenate([omega, upsilon])
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2.0 * math.sin(theta))
    omega = axis * theta
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    inverse_jacobian = (np.eye(3) - 0.5 * theta * skew
                        + (1.0 - 0.5 * theta / math.tan(theta / 2.0)) * (skew @ skew))
    return np.concatenate([omega, inverse_jacobian @ translation])


def se3_exp(xi) -> np.ndarray:
    xi = np.asarray(xi, dtype=float)
    omega, upsilon = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    transform = np.eye(4)
    if theta < 1e-8:
        transform[:3, :3] = np.eye(3) + np.array(
            [[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
        transform[:3, 3] = upsilon
        return transform
    axis = omega / theta
    skew = np.array([[0.0, -axis[2], axis[1]],
                     [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])
    rotation = np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)
    jacobian = (np.eye(3) + (1.0 - math.cos(theta)) / theta * skew
                + (1.0 - math.sin(theta) / theta) * (skew @ skew))
    transform[:3, :3] = rotation
    transform[:3, 3] = jacobian @ upsilon
    return transform


# --------------------------------------------------------------------------- stable CRN
class StableExogenousRNG:
    """Exogenous random stream independent of any endogenous ``action_id``.

    Every draw is derived from the tuple
    ``(scene_id, seed, actor_id, event_family, purpose, target_anchor,
    quantized_t, local_seq)``; the API does not accept an action id, so a branch
    cannot perturb the common random numbers by taking a different action.
    """

    def __init__(self, scene_id: str, seed: int, quantize_s: float = 5.0):
        self.scene_id = str(scene_id)
        self.seed = int(seed)
        self.quantize_s = float(quantize_s)
        if self.quantize_s <= 0:
            raise ValueError("quantize_s must be positive")

    def key(self, actor_id="world", event_family="exogenous", purpose="draw",
            target_anchor=None, quantized_t=None, local_seq=0):
        for part in (event_family, purpose, actor_id):
            if isinstance(part, str) and "action_id" in part:
                raise ValueError("the exogenous key must not depend on an action id")
        if quantized_t is None:
            quantized_t = 0.0
        bucket = int(round(float(quantized_t) / self.quantize_s)) if self.quantize_s else 0
        return (self.scene_id, self.seed, str(actor_id), str(event_family), str(purpose),
                None if target_anchor is None else str(target_anchor), bucket, int(local_seq))

    def _seed(self, key_tuple) -> int:
        digest = hashlib.sha256(canonical_json(list(key_tuple)).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    def stream(self, **kwargs) -> random.Random:
        return random.Random(self._seed(self.key(**kwargs)))

    def random(self, **kwargs) -> float:
        return self.stream(**kwargs).random()

    def uniform(self, low=0.0, high=1.0, **kwargs) -> float:
        return self.stream(**kwargs).uniform(low, high)

    def normal(self, mu=0.0, sigma=1.0, **kwargs) -> float:
        return self.stream(**kwargs).gauss(mu, sigma)

    def choice(self, options, **kwargs):
        return self.stream(**kwargs).choice(list(options))

    def vector(self, size, mu=0.0, sigma=1.0, **kwargs) -> list:
        generator = self.stream(**kwargs)
        return [generator.gauss(mu, sigma) for _ in range(int(size))]


# --------------------------------------------------------------------------- clone core
HOOK_NAMES = ("on_start", "on_tick", "on_action_result")
SIM_METHODS = ("submit_action", "step", "run", "start", "_complete", "_enqueue_world",
               "_progress", "_status", "_apply_outcome", "_notify_rejected",
               "_finish_canceled", "_external_watchdog", "_report", "_send")


def _rebind_core(sim):
    """Strip external callbacks and rebind every method to this instance."""
    for agent in sim.agents.values():
        for name in HOOK_NAMES:
            current = getattr(agent, name, None)
            original = getattr(AgentRuntime, name, None)
            if original is None:
                continue
            if getattr(current, "__func__", None) is not original or getattr(current, "__self__", None) is not agent:
                setattr(agent, name, types.MethodType(original, agent))
    for name in SIM_METHODS:
        original = getattr(Simulation, name, None)
        if original is None:
            continue
        setattr(sim, name, types.MethodType(original, sim))
    return sim


def clone_core_env(env):
    """Deep-copy the live core state into an isolated branch environment.

    The append-only event archive is detached during the copy (it grows large and
    the branch never reads it) and the clone gets a fresh empty archive, so the
    clone cannot append into the parent's log.  All hooks are then rebound.
    """
    if not isinstance(env, Environment):
        raise TypeError("clone_core_env expects a rescue_scenes Environment")
    events = env.sim.events
    env.sim.events = []
    try:
        clone = deepcopy(env)
    finally:
        env.sim.events = events
    clone.sim.events = []
    _rebind_core(clone.sim)
    return clone


# --------------------------------------------------------------------------- digests
def _digest_payload(env):
    sim = env.sim
    world = {}
    for entity_id, entity in sorted(sim.world.entities.items()):
        world[entity_id] = {key: _jsonable(value) for key, value in sorted(entity.get("fields", {}).items())}
    agents = {}
    for agent_id, agent in sorted(sim.agents.items()):
        entries = {f"{entity_id}|{key}": {"value": _jsonable(entry.get("value")),
                                          "source": entry.get("source"),
                                          "confidence": _jsonable(entry.get("confidence")),
                                          "t": _jsonable(entry.get("t"))}
                   for (entity_id, key), entry in sorted(agent.belief.entries.items())}
        agents[agent_id] = {
            "platform": _jsonable(agent.platform),
            "belief": entries,
            "tasks": {task_id: {key: _jsonable(task.get(key)) for key in ("status", "rev", "type", "assigned_to")}
                      for task_id, task in sorted(agent.tasks.items())},
            "current": agent.current,
            "queue": _jsonable(list(getattr(agent, "queue", []))),
            "policy_mode": agent.policy_mode,
        }
    return {"t": round(float(sim.t), 6), "world": world, "agents": agents,
            "instances": len(sim.instances), "heap": len(sim._heap),
            "ledger": _jsonable(sim.ledger.to_dict())}


def state_digest(env) -> str:
    return sha256_text(canonical_json(_digest_payload(env)))


def world_digest(env) -> str:
    world = {entity_id: {key: _jsonable(value) for key, value in sorted(entity.get("fields", {}).items())}
             for entity_id, entity in sorted(env.sim.world.entities.items())}
    return sha256_text(canonical_json(world))


def belief_digest(env, agent_id) -> str:
    agent = env.sim.agents[agent_id]
    entries = {f"{entity_id}|{key}": {"value": _jsonable(entry.get("value")),
                                      "source": entry.get("source"),
                                      "confidence": _jsonable(entry.get("confidence")),
                                      "t": _jsonable(entry.get("t"))}
               for (entity_id, key), entry in sorted(agent.belief.entries.items())}
    return sha256_text(canonical_json(entries))


def _noop_track(env, horizon_s, step_s=None):
    step_s = float(step_s or env.sim.config.get("tick_s", TICK_FALLBACK_S) or TICK_FALLBACK_S)
    digests = [state_digest(env)]
    end = env.sim.t + float(horizon_s)
    while env.sim.t < end - 1e-9:
        env.sim.run(min(env.sim.t + step_s, end))
        digests.append(state_digest(env))
    return digests


def _default_branch_goals(env, agent_id, count=3):
    agent = env.sim.agents[agent_id]
    point = _agent_point(agent)
    goals = []
    for entity_id, entries in sorted(agent.belief.entities(copy=False).items()):
        line = entries.get("road.centerline")
        coords = line.get("value") if isinstance(line, dict) else None
        if isinstance(coords, dict):
            coords = coords.get("coords")
        if isinstance(coords, (list, tuple)) and coords:
            for candidate in (coords[-1], coords[0]):
                if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
                    goals.append([float(candidate[0]), float(candidate[1])])
                    break
        if len(goals) >= count + 4:
            break
    goals.sort(key=lambda goal: math.dist(point, goal))
    unique = []
    for goal in goals:
        if math.dist(point, goal) > 5.0 and all(math.dist(goal, other) > 12.0 for other in unique):
            unique.append(goal)
        if len(unique) >= count:
            break
    return unique


def _branch_signature(env, agent_id, extra=None):
    agent = env.sim.agents[agent_id]
    payload = {"t": round(float(env.sim.t), 6),
               "pose": [round(float(v), 4) for v in _agent_point(agent)],
               "battery": round(float(env.sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh", 0.0) or 0.0), 3),
               "world": world_digest(env)}
    if extra:
        payload.update(extra)
    return sha256_text(canonical_json(payload))


def _run_branch_probe(env, agent_id, goal, horizon_s):
    clone = clone_core_env(env)
    agent = clone.sim.agents[agent_id]
    agent.policy_mode = "external_harness"
    action_id = MOVE_ACTION.get(agent.spec.get("mobility"), "A002")
    iid = Simulation.submit_action(clone.sim, agent_id, action_id, {"goal": list(goal)})
    record = clone.sim.instances[iid]
    status = record.get("status")
    end = clone.sim.t + float(horizon_s)
    steps = 0
    while clone.sim.t < end - 1e-9:
        clone.sim.run(min(clone.sim.t + float(clone.sim.config.get("tick_s", TICK_FALLBACK_S) or TICK_FALLBACK_S), end))
        steps += 1
        if steps > 4000:
            break
    return {"goal": [round(float(v), 2) for v in goal],
            "status": str(clone.sim.instances[iid].get("status", status)),
            "signature": _branch_signature(clone, agent_id)}


def verify_clone_isolation(env, agent_id=None, horizon_s=20.0):
    """Machine-check the four clone-isolation assertions of contract ②.

    (1) a no-op continuation of two independent clones follows the same
        trajectory, and cloning never advances the parent;
    (2) repeated clones all digest to the parent's state;
    (3) running the same branches in forward, reverse and shuffled order yields
        the same per-branch results;
    (4) mutating a sibling branch leaves the parent digest untouched.

    Returns a report dict of booleans and raises ``AssertionError`` on the first
    violated clause.
    """
    base = state_digest(env)
    parent_world = world_digest(env)
    if agent_id is None:
        candidates = [agent for agent in sorted(env.sim.agents.values(), key=lambda a: a.id)
                      if agent.kind != "coordinator" and agent.spec.get("mobility") != "none"]
        if not candidates:
            raise AssertionError("no movable agent to verify clone isolation with")
        agent_id = candidates[0].id
    report = {}

    # (1) no-op continuation on the same track.
    track_a = _noop_track(clone_core_env(env), horizon_s)
    track_b = _noop_track(clone_core_env(env), horizon_s)
    report["noop_same_track"] = bool(track_a == track_b and len(track_a) > 1)
    report["parent_untouched_by_noop"] = state_digest(env) == base and world_digest(env) == parent_world

    # (2) repeated clones share one digest.
    digests = {state_digest(clone_core_env(env)) for _ in range(5)}
    report["repeated_clone_digest"] = bool(len(digests) == 1 and digests.pop() == base)

    # (3) branch order independence.
    goals = _default_branch_goals(env, agent_id, count=3)
    if len(goals) < 2:
        raise AssertionError("not enough public road goals to probe branch isolation")
    forward = [_run_branch_probe(env, agent_id, goal, horizon_s) for goal in goals]
    reverse = list(reversed([_run_branch_probe(env, agent_id, goal, horizon_s)
                             for goal in reversed(goals)]))
    order = list(range(len(goals)))
    random.Random(f"{agent_id}|branch-order").shuffle(order)
    shuffled = {index: _run_branch_probe(env, agent_id, goals[index], horizon_s) for index in order}
    shuffled_results = [shuffled[index] for index in range(len(goals))]
    report["order_independent"] = bool(forward == reverse == shuffled_results)
    report["branch_parent_untouched"] = state_digest(env) == base

    # (4) sibling mutation leaves the parent unchanged.
    sibling = clone_core_env(env)
    sibling_agent = sibling.sim.agents[agent_id]
    sibling_agent.policy_mode = "external_harness"
    Simulation.submit_action(sibling.sim, agent_id, MOVE_ACTION.get(sibling_agent.spec.get("mobility"), "A002"),
                             {"goal": list(goals[0])})
    sibling.sim.run(sibling.sim.t + float(horizon_s))
    sibling.sim.world.set(sibling_agent.platform_eid, "attr.clone_probe_marker", 1, "WORLD")
    report["sibling_mutation_isolated"] = bool(
        state_digest(env) == base
        and world_digest(env) == parent_world
        and state_digest(clone_core_env(env)) == base)

    failed = [name for name, ok in report.items() if not ok]
    if failed:
        raise AssertionError("clone isolation failed: " + ", ".join(sorted(failed)))
    return report


# --------------------------------------------------------------------------- rollout
def _entry_value(entry, default=None):
    return entry.get("value", default) if isinstance(entry, dict) else default


def _belief(agent, entity_id, key, default=None):
    return _entry_value(agent.belief.get(entity_id, key), default)


def _belief_point(agent, entity_id):
    for key in ("entity.pose", "human.pose", "belief.estimated_pose", "obstacle.pose"):
        point = _belief(agent, entity_id, key)
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return [float(point[0]), float(point[1])]
    return None


def _spec_signature(agent) -> str:
    """Embodiment signature that a legitimate rollout must never mutate."""
    return canonical_json({"mobility": agent.spec.get("mobility"),
                           "capabilities": sorted(str(item) for item in agent.spec.get("capabilities", [])),
                           "payload_kg": agent.spec.get("payload_kg"),
                           "speed_mps": agent.spec.get("speed_mps")})


def _observations(result):
    if result is None:
        return []
    rows = getattr(result, "observations", None) or []
    return [row.to_dict() if hasattr(row, "to_dict") else row for row in rows]


class _OptionPlan:
    """A tiny stage machine that only ever reads the agent's own belief.

    State is limited to what the agent itself learned: the person it contacted,
    the one it extricated and the one it loaded.  Nothing here reads world truth,
    so a blocked corridor is discovered by the native motion guard, not by the
    planner.
    """

    def __init__(self, rollout, option_id):
        anchor_key, stages = OPTION_PLANS[option_id]
        self.rollout = rollout
        self.option_id = option_id
        self.anchor_key = anchor_key
        self.stages = [{"kind": kind, "done": False, "retries": 0, "submitted": 0}
                       for kind in stages]
        self.index = 0
        self.pending = False
        self.log = []
        self.seen_humans = []
        self.contacted = None
        self.extricated = None
        self.carried = None
        self.transported = None
        self.observations = []
        self.last_error = None

    # -- decision ---------------------------------------------------------
    def next_action(self, sim, agent):
        if self.pending:
            return None
        while self.index < len(self.stages):
            stage = self.stages[self.index]
            if stage["done"]:
                self.index += 1
                continue
            decision = self._decide(stage, sim, agent)
            if decision is None:            # stage satisfied without an action
                self.log.append({"stage": stage["kind"], "result": "skipped"})
                stage["done"] = True
                self.index += 1
                continue
            action_id, params = decision
            stage["submitted"] += 1
            stage["pending_action"] = action_id
            stage["pending_params"] = params
            self.pending = True
            self.log.append({"stage": stage["kind"], "action": action_id,
                             "t": round(float(sim.t), 2)})
            return action_id, params
        return None

    def note_result(self, sim, agent, action_id, status, failure, result):
        self.pending = False
        if self.index >= len(self.stages):
            return
        stage = self.stages[self.index]
        self.log.append({"stage":stage["kind"],"action":action_id,"status":status,
                         "failure":failure,"t":round(float(sim.t),6)})
        if stage.get("pending_action") != action_id:
            return
        success = status in ("succeeded", "partial")
        params = stage.get("pending_params") or {}
        target = params.get("target")
        if stage["kind"] == "capture":
            self._absorb_observations(result)
        if success:
            self.observations = _observations(result) or self.observations
        if not success:
            self.last_error = failure
            # A guard rejection is deterministic: retrying it only manufactures
            # another named action_failed for the same infeasible primitive.
            retryable = (stage["kind"] in ("contact", "load", "deliver")
                         and status != "rejected" and failure not in ("blocked", "actuator"))
            if retryable and stage["retries"] < MAX_STAGE_RETRIES:
                stage["retries"] += 1
                return
            stage["done"] = True
            self.index += 1
            return
        if stage["kind"] == "integrate":
            self.observations = []
        elif stage["kind"] == "contact" and target:
            self.contacted = target
        elif stage["kind"] == "extricate" and target:
            self.extricated = target
        elif stage["kind"] == "load" and target:
            self.carried = target
        elif stage["kind"] == "transport" and target:
            self.transported = target
        stage["done"] = True
        self.index += 1

    def _absorb_observations(self, result):
        for row in _observations(result):
            values = row.get("values") if isinstance(row, dict) else None
            if not isinstance(values, dict):
                continue
            if "human.location_context" not in values and "human.distress_signal" not in values:
                continue
            for entity_id in (row.get("about") or []):
                if isinstance(entity_id, str) and entity_id not in self.seen_humans:
                    self.seen_humans.append(entity_id)

    # -- stage handlers ---------------------------------------------------
    def _capabilities(self, agent):
        return set(agent.spec.get("capabilities", []))

    def _move_action(self, agent):
        return MOVE_ACTION.get(agent.spec.get("mobility"), "A002")

    def _anchor(self, sim, agent):
        return self.rollout.anchor_for(self.anchor_key, sim, agent)

    def _target_human(self, sim, agent):
        for entity_id in self.seen_humans:
            point = _belief_point(agent, entity_id)
            if point is not None and math.dist(_agent_point(agent), point) <= CONTACT_RADIUS_M:
                return entity_id
        for entity_id in sorted(agent.belief.entities(copy=False)):
            if _belief(agent, entity_id, "entity.kind") != "human":
                continue
            entry = agent.belief.get(entity_id, "human.location_context")
            if not entry or entry.get("source") == "prior":
                continue
            point = _belief_point(agent, entity_id)
            if point is not None and math.dist(_agent_point(agent), point) <= CONTACT_RADIUS_M:
                return entity_id
        return None

    def _has_medical_kit(self, agent):
        payload = agent.platform.get("payload_item_ids") or agent.platform.get("payload") or []
        for entity_id in payload if isinstance(payload, (list, tuple)) else []:
            if _belief(agent, entity_id, "item.kind") == "medical_pack":
                return True
        for entity_id in sorted(agent.belief.entities(copy=False)):
            if _belief(agent, entity_id, "item.kind") != "medical_pack":
                continue
            point = _belief_point(agent, entity_id)
            if point is not None and math.dist(_agent_point(agent), point) <= 15.0:
                return True
        return False

    def _decide(self, stage, sim, agent):
        kind = stage["kind"]
        capabilities = self._capabilities(agent)
        if kind == "move":
            anchor = self._anchor(sim, agent)
            if anchor is None or math.dist(_agent_point(agent), anchor) <= 2.0:
                return None
            params={"goal":list(anchor)}
            path=self.rollout.anchors.get(str(self.anchor_key)+"_PATH")
            if path is not None:params["path"]=deepcopy(path)
            return self._move_action(agent), params
        if kind == "advance":
            anchor = self._anchor(sim, agent)
            if anchor is None or math.dist(_agent_point(agent), anchor) <= 2.0:
                return None
            return self._move_action(agent), {"goal": list(anchor)}
        if kind == "capture":
            if not self._has_sensor(agent, "rgb"):
                return None
            return "A005", {"modality": "rgb", "range_m": SENSING_RANGE_M}
        if kind == "scan":
            if not self._has_sensor(agent, "rgb"):
                return None
            stage["heading"] = stage.get("heading", 0.0) + 90.0
            return "A005", {"modality": "rgb", "range_m": SENSING_RANGE_M,
                            "heading_deg": stage["heading"]}
        if kind == "integrate":
            if not {"C44", "C39"} <= capabilities or not self.observations:
                return None
            return "A041", {"observations": deepcopy(self.observations)}
        if kind == "approach_local":
            # A target discovered after the corridor move is approached using only
            # integrated local evidence. The native motion model owns feasibility.
            known = [(math.dist(_agent_point(agent), point), point)
                     for eid in self.seen_humans
                     if (point := _belief_point(agent, eid)) is not None]
            if not known:
                return None
            distance, point = min(known)
            return None if distance <= 3.0 else (self._move_action(agent), {"goal": list(point)})
        if kind == "contact":
            if "C11" not in capabilities or self.contacted:
                return None
            target = self._target_human(sim, agent)
            if target is None:
                return None
            return "A009", {"target": target}
        if kind == "extricate":
            if not {"C18", "C20"} <= capabilities or self.extricated:
                return None
            target = self.contacted
            if target is None or _belief(agent, target, "human.mobility") != "trapped":
                return None
            return "A017", {"target": target}
        if kind == "aid":
            if "C21" not in capabilities or not self._has_medical_kit(agent):
                return None
            target = self.contacted
            if target is None:
                return None
            return "A019", {"target": target}
        if kind == "load":
            if "C22" not in capabilities or self.carried:
                return None
            target = self.contacted
            if target is None:
                return None
            mobility = _belief(agent, target, "human.mobility")
            if mobility == "trapped" and not self.extricated:
                return None
            if mobility not in ("limited", "immobile", "mobile") and not self.extricated:
                # The agent has no local evidence that the person can be loaded;
                # attempt nothing rather than burn a rejected A020.
                return None
            return "A020", {"target": target}
        if kind == "transport":
            if "C22" not in capabilities or not self.carried or self.transported:
                return None
            station = self._nearest_station(sim, agent)
            if station is None:
                return None
            return "A021", {"target": self.carried, "goal": self._station_point(sim, agent, station),
                            "destination_id": station}
        if kind == "deliver":
            if "C46" not in capabilities or not self.carried:
                return None
            station = self._nearest_station(sim, agent)
            if station is None:
                return None
            return "A022", {"target": self.carried, "station_id": station}
        if kind == "clear":
            if not ({"C17"} & capabilities):
                return None
            obstacle = self._target_obstacle(sim, agent)
            if obstacle is None:
                return None
            return ("A013" if "C17" in capabilities else "A012"), {"target": obstacle}
        return None

    def _has_sensor(self, agent, modality):
        for sensor in agent.spec.get("sensors", []):
            if isinstance(sensor, str):
                if sensor.replace("scripted_", "") == modality:
                    return True
            elif isinstance(sensor, dict) and sensor.get("enabled", True) \
                    and sensor.get("modality") == modality:
                return True
        return False

    def _nearest_station(self, sim, agent):
        best, best_distance = None, math.inf
        for entity_id in sorted(agent.belief.entities(copy=False)):
            if _belief(agent, entity_id, "entity.kind") != "station":
                continue
            if _belief(agent, entity_id, "station.kind") not in RECEIVING_STATION_KINDS:
                continue
            if _belief(agent, entity_id, "station.open_state", "open") != "open":
                continue
            point = _belief_point(agent, entity_id)
            if point is None:
                continue
            distance = math.dist(_agent_point(agent), point)
            if distance < best_distance:
                best, best_distance = entity_id, distance
        return best

    def _station_point(self, sim, agent, station):
        point = _belief_point(agent, station)
        if point is not None:
            return point
        return None

    def _target_obstacle(self, sim, agent):
        best, best_distance = None, math.inf
        for entity_id in sorted(agent.belief.entities(copy=False)):
            entry = agent.belief.get(entity_id, "obstacle.state")
            if not entry or entry.get("source") == "prior":
                continue
            if entry.get("value") not in ("present", "partly_cleared"):
                continue
            point = _belief_point(agent, entity_id)
            if point is None:
                continue
            distance = math.dist(_agent_point(agent), point)
            if distance < best_distance:
                best, best_distance = entity_id, distance
        return best


def _physical_sensor_visible(sim, agent, entity_id):
    """Simulator-side ray occlusion; the option controller never reads world truth."""
    entity=sim.world.entities.get(entity_id)
    if entity is None or entity.get("cls") not in ("human", "obstacle", "cue"):
        return True
    fields=entity.get("fields", {})
    point=fields.get("entity.pose") or fields.get("human.pose")
    if point is None:
        coords=entity["geometry"]["coords"]
        point=[sum(p[k] for p in coords)/len(coords) for k in (0,1)]
    eye=uav_eye(agent) if agent.spec.get("mobility")=="air" else ground_eye(agent)
    occluders=[]
    for eid,building in sim.world.entities.items():
        if building.get("cls")!="building":continue
        polygon=_polygon_of(building["geometry"]["coords"])
        height=max(3.,3.*float(sim.world.get(eid,"building.floors",1) or 1))
        occluders.append((eid,polygon,height))
    return los_clearance(eye,[float(point[0]),float(point[1]),1.0],occluders)>=0


class LegitimateOptionRollout:
    """Execute one named macro-action option on a pure-core clone.

    The controller only reads ``agent.belief`` / ``agent.spec``; feasibility is
    decided by the native ``MODELS[...]`` guard.  No mobility override and no
    capability injection takes place, unlike ``ForcedRuleDriver``.
    """

    def __init__(self, env, agent_id, scene_id, seed, anchors=None,
                 horizon_s=HORIZON_S, slices=SLICES_S):
        self.env = env
        self.agent_id = agent_id
        self.scene_id = scene_id
        self.seed = int(seed)
        self.anchors = dict(anchors or {})
        self.horizon_s = float(horizon_s)
        self.slices = tuple(float(item) for item in slices)
        self.rng = StableExogenousRNG(scene_id, seed)

    # -- anchors ----------------------------------------------------------
    def anchor_for(self, key, sim, agent):
        if key is None:
            return None
        if key == "CUE":
            return self._cue_anchor(sim, agent)
        if key in ("A","B") and key+"_GOAL" in self.anchors:
            key=key+"_GOAL"
        elif key=="J" and "CLEAR" in self.anchors:
            key="CLEAR"
        value = self.anchors.get(key)
        if value is None:
            return None
        return [float(value[0]), float(value[1])]

    def _cue_anchor(self, sim, agent):
        point = _agent_point(agent)
        best, best_distance = None, math.inf
        for entity_id in sorted(agent.belief.entities(copy=False)):
            if _belief(agent, entity_id, "entity.kind") != "cue":
                continue
            entries=[agent.belief.get(entity_id,key) for key in ("entity.pose","cue.location","cue.signal_state")]
            if not any(entry and entry.get("source")!="prior" for entry in entries):
                continue
            candidate = _belief_point(agent, entity_id)
            if candidate is None:
                continue
            distance = math.dist(point, candidate)
            if 5.0 < distance < best_distance:
                best, best_distance = candidate, distance
        if best is None:
            best = point
        return [float(best[0]), float(best[1])]

    # -- run --------------------------------------------------------------
    def run(self, option_id, write_log=True):
        if option_id not in OPTION_PLANS:
            raise KeyError("unknown macro option: " + str(option_id))
        clone = clone_core_env(self.env)
        sim = clone.sim
        sim.exogenous_crn = self.rng
        sim.phase1_sensor_visibility = types.MethodType(_physical_sensor_visible, sim)
        agent = sim.agents[self.agent_id]
        # The option is driven externally; the native external-harness mode keeps
        # telemetry, message intake and A041 integration but stops the internal
        # task generator from racing the option with its own actions.  The rest of
        # the fleet is frozen for the same reason, so every named reward below is
        # attributable to this one option instead of to a sibling agent's policy.
        for other in sim.agents.values():
            other.policy_mode = "external_harness"
        agent = sim.agents[self.agent_id]
        spec_before = _spec_signature(agent)
        plan = _OptionPlan(self, option_id)
        start = float(sim.t)
        end = start + self.horizon_s
        step_s = float(sim.config.get("tick_s", TICK_FALLBACK_S) or TICK_FALLBACK_S)
        blocked_roads = {entity_id for entity_id, entity in sim.world.entities.items()
                         if entity.get("cls") == "road" and sim.world.get(entity_id, "road.entry_state") != "open"}
        # Rewards are measured relative to the section's own starting state: a
        # casualty already contacted before the branch is not this option's gain.
        baseline = self._milestones(sim, blocked_roads)
        seen = set()
        first_achieved_s = {}
        cumulative = {key: 0.0 for key in S.NAMED_REWARD_KEYS}
        slice_rows = {}
        slice_times = [start + delta for delta in self.slices]
        next_slice = 0
        busy = 0.0
        energy = 0.0
        failures = 0
        in_flight = None
        in_flight_start = None
        submitted = []
        safety = 0
        while sim.t < end - 1e-9:
            safety += 1
            if safety > 20000:
                raise RuntimeError("option rollout did not converge")
            # Advance straight to the next real boundary: the next slice, the
            # running action's own completion time, or one tick when idle.  The
            # completion is a scheduled heap event, so stepping to it is exact.
            boundary = end
            if next_slice < len(slice_times):
                boundary = min(boundary, slice_times[next_slice])
            if in_flight is not None:
                record = sim.instances.get(in_flight) or {}
                duration = float((record.get("plan") or {}).get("duration_s", 0.0) or 0.0)
                boundary = min(boundary, float(in_flight_start) + duration + 1e-6)
            else:
                boundary = min(boundary, sim.t + step_s)
            if boundary <= sim.t + 1e-9:
                boundary = min(end, sim.t + step_s)
            sim.run(boundary)

            if in_flight is not None:
                record = sim.instances.get(in_flight)
                status = str(record.get("status")) if record else "missing"
                if status not in ("requested", "accepted", "running"):
                    result = record.get("result") if record else None
                    failure = str((result.failure_code if result is not None else None)
                                  or record.get("failure_code") or "none")
                    elapsed = max(0.0, float(sim.t) - float(in_flight_start))
                    duration = float((record.get("plan") or {}).get("duration_s", 0.0) or 0.0)
                    busy += min(elapsed, duration if duration > 0 else elapsed)
                    cost = getattr(result, "cost", None) if result is not None else None
                    if isinstance(cost, dict):
                        energy += float(cost.get("energy_Wh", 0.0) or 0.0)
                    elif record and record.get("plan"):
                        energy += float(record["plan"].get("energy_Wh", 0.0) or 0.0)
                    if status == "rejected" or result is None or \
                            (result is not None and getattr(result, "status", status) in ("failed", "rejected")):
                        failures += 1
                    plan.note_result(sim, agent, record.get("action_id"), status, failure, result)
                    in_flight = None
            if in_flight is None:
                decision = plan.next_action(sim, agent)
                if decision is not None:
                    action_id, params = decision
                    try:
                        instance_id = Simulation.submit_action(sim, self.agent_id, action_id, params)
                    except Exception as exc:                  # pragma: no cover - defensive
                        failures += 1
                        plan.pending = False
                        plan.last_error = type(exc).__name__
                        submitted.append({"action": action_id, "status": "submit_error"})
                        decision = None
                    else:
                        record = sim.instances[instance_id]
                        submitted.append({"action": action_id, "id": instance_id,
                                          "t": round(float(sim.t), 2), "status": str(record.get("status"))})
                        if str(record.get("status")) in ("requested", "accepted", "running"):
                            in_flight = instance_id
                            in_flight_start = float(sim.t)
                        elif str(record.get("status")) == "rejected":
                            # A rejected submit is a terminal action: settle the
                            # stage now instead of waiting for a completion event.
                            failures += 1
                            plan.note_result(sim, agent, action_id, "rejected",
                                             str(record.get("rejection_code", "rejected")),
                                             record.get("result"))

            for milestone in self._milestones(sim, blocked_roads) - baseline:
                first_achieved_s.setdefault(milestone, round(float(sim.t) - start, 6))
            # Record every slice we have crossed.
            while next_slice < len(slice_times) and sim.t >= slice_times[next_slice] - 1e-9:
                current = self._milestones(sim, blocked_roads) - baseline
                new = current - seen
                seen |= new
                delta = {key: 0.0 for key in S.NAMED_REWARD_KEYS}
                for item in new:
                    entity_id, milestone = item.split("::", 1)
                    delta[milestone] = delta.get(milestone, 0.0) + 1.0
                for key in S.NAMED_REWARD_KEYS:
                    cumulative[key] += delta.get(key, 0.0)
                previous_cost = {key: cumulative[key] for key in ("energy_Wh", "active_busy_s", "action_failed")}
                running_busy = max(0.0, float(sim.t) - float(in_flight_start)) if in_flight is not None else 0.0
                cumulative["energy_Wh"] = round(energy, 6)
                cumulative["active_busy_s"] = round(min(busy + running_busy, self.slices[next_slice]), 3)
                cumulative["action_failed"] = float(failures)
                for key, previous_value in previous_cost.items():
                    delta[key] = cumulative[key] - previous_value
                row = {"horizon_s": float(self.slices[next_slice]),
                       "named_rewards": {key: round(float(delta[key]), 6) for key in S.NAMED_REWARD_KEYS},
                       "cumulative_named_rewards": {key: round(float(cumulative[key]), 6)
                                                    for key in S.NAMED_REWARD_KEYS},
                       "milestones": sorted(seen),
                       "first_achieved_s": {item: first_achieved_s[item] for item in sorted(seen)}}
                slice_rows[SLICE_NAMES[next_slice]] = row
                next_slice += 1

        # Any slice not yet crossed (an instantaneous horizon edge) is recorded now.
        while next_slice < len(slice_times):
            current = self._milestones(sim, blocked_roads) - baseline
            new = current - seen
            seen |= new
            delta = {key: 0.0 for key in S.NAMED_REWARD_KEYS}
            for item in new:
                entity_id, milestone = item.split("::", 1)
                delta[milestone] = delta.get(milestone, 0.0) + 1.0
            for key in S.NAMED_REWARD_KEYS:
                cumulative[key] += delta.get(key, 0.0)
            cumulative["energy_Wh"] = round(energy, 6)
            cumulative["active_busy_s"] = round(min(busy, self.horizon_s), 3)
            cumulative["action_failed"] = float(failures)
            delta["energy_Wh"] = 0.0
            delta["active_busy_s"] = 0.0
            delta["action_failed"] = 0.0
            slice_rows[SLICE_NAMES[next_slice]] = {
                "horizon_s": float(self.slices[next_slice]),
                "named_rewards": {key: round(float(delta[key]), 6) for key in S.NAMED_REWARD_KEYS},
                "cumulative_named_rewards": {key: round(float(cumulative[key]), 6)
                                             for key in S.NAMED_REWARD_KEYS},
                "milestones": sorted(seen)}
            next_slice += 1

        # A legitimate rollout never buys feasibility with a mobility or
        # capability override: the embodiment signature must be untouched.
        spec_unchanged = _spec_signature(agent) == spec_before
        if not spec_unchanged:
            raise AssertionError("option rollout mutated the agent's nominal embodiment: "
                                 + spec_before + " -> " + _spec_signature(agent))
        result = {"option_id": option_id,
                  "agent_id": self.agent_id,
                  "spec_unchanged": spec_unchanged,
                  "slices": {name: slice_rows[name] for name in SLICE_NAMES},
                  "q_value_120s": round(S.q_value(slice_rows[PRIMARY_SLICE]["cumulative_named_rewards"]), 9),
                  "final_pose": [round(float(v), 3) for v in _agent_point(agent)],
                  "action_failures": failures,
                  "final_world_digest": world_digest(clone)}
        if write_log:
            result["log"] = plan.log
            result["submitted"] = submitted
        return result

    # -- milestones -------------------------------------------------------
    def _milestones(self, sim, blocked_roads):
        world = sim.world
        agent = sim.agents[self.agent_id]
        state = set()
        for (entity_id, key), entry in agent.belief.entries.items():
            if key in ("human.location_context", "human.distress_signal") and entry.get("source") != "prior":
                if world.entities.get(entity_id, {}).get("cls") == "human":
                    state.add(f"{entity_id}::person_found")
        for entity_id, entity in world.entities.items():
            cls = entity.get("cls")
            if cls == "human":
                support = world.get(entity_id, "human.support_received", []) or []
                if "contacted" in support:
                    state.add(f"{entity_id}::contacted")
                if "extricated" in support:
                    state.add(f"{entity_id}::extricated")
                if "treated" in support:
                    state.add(f"{entity_id}::aided")
                if world.get(entity_id, "human.handover_status") == "acknowledged":
                    state.add(f"{entity_id}::handed_over")
            elif cls == "road" and entity_id in blocked_roads:
                if world.get(entity_id, "road.entry_state") == "open":
                    state.add(f"{entity_id}::road_cleared")
            elif cls == "hazard":
                fire = float(world.get(entity_id, "hazard.fire_intensity", 1.0) or 0.0)
                if bool(world.get(entity_id, "hazard.exclusion_zone")) or fire <= 1e-9:
                    state.add(f"{entity_id}::hazard_handled")
        return state


# --------------------------------------------------------------------------- builder
class MatchedHiddenStateBuilder:
    """Construct the matched hidden-state pair of contract ①.

    The pair differs only behind the ground agent's foreground building: one
    corridor is open and holds the casualty while the other is blocked by a
    collapsed roadblock, and vice versa.  Everything the ground agent can see or
    has ever recorded is identical in both worlds; the aerial camera sees both.
    """

    def __init__(self, env, scene_id, seed, split=None, regime=None, out_dir=None,
                 agent_id=None, human_id=None, uav_id=None, altitude_m=None, single_factor=False):
        self.env = env
        self.single_factor = bool(single_factor)
        self.scene_id = str(scene_id)
        self.seed = int(seed)
        self.split = split or S.SCENE_SPLITS.get(self.scene_id, "train")
        self.regime = regime or S.SCENE_REGIMES.get(self.scene_id, "A")
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.occluders = building_occluders(env)
        self.agent = self._pick_agent(agent_id)
        self.agent_id = self.agent.id
        self.uav = self._pick_uav(uav_id)
        self.altitude_m = float(altitude_m if altitude_m is not None else UAV_SURVEY_ALT_M)
        self.human_id = human_id or self._pick_human()
        if self.single_factor:
            self._stage_single_factor_agents()
        self.occluder_id, self.occluder_polygon, self.occluder_height = self._pick_occluder()
        self.view_dir, self.lateral_dir, self.far_depth, self.anchors = self._corridor_geometry()
        if self.single_factor:
            u = np.asarray(self.view_dir)
            for distance in (8.0, 12.0, 16.0):
                target = np.asarray(self.anchors["J"]) + distance * u
                bounds=self.env.sim.config.get("bounds_m") or [0,0,2000,2000]
                if bounds[0]+2 < target[0] < bounds[2]-2 and bounds[1]+2 < target[1] < bounds[3]-2 and not self._inside_building(target):
                    self.anchors["TARGET"] = target.tolist()
                    break
            else:
                raise RuntimeError("no common casualty location behind the corridor junction")
            self._author_public_corridors()
        self.pair_index = 0

    # -- scene discovery --------------------------------------------------
    def _pick_agent(self, agent_id):
        if agent_id is not None:
            return self.env.sim.agents[agent_id]
        wanted = {"C11", "C18", "C20", "C21", "C22", "C46"}
        best, best_score = None, None
        for agent in sorted(self.env.sim.agents.values(), key=lambda a: a.id):
            if agent.kind == "coordinator" or agent.spec.get("mobility") in (None, "none", "air", "water"):
                continue
            capabilities=set(agent.spec.get("capabilities", []))
            sensors=agent.spec.get("sensors", [])
            rgb=any((isinstance(row,str) and row.replace("scripted_","")=="rgb") or
                    (isinstance(row,dict) and row.get("modality")=="rgb") for row in sensors)
            if self.single_factor and (not rgb or "C11" not in capabilities):
                continue
            coverage = len(wanted & capabilities)
            point = _agent_point(agent)
            near = min((math.dist(point, [sum(q[0] for q in poly) / len(poly),
                                          sum(q[1] for q in poly) / len(poly)])
                        for _eid, poly, _h in self.occluders), default=math.inf)
            score = (coverage, float(agent.spec.get("speed_mps",0)), -near)
            if best_score is None or score > best_score:
                best, best_score = agent, score
        if best is None:
            raise RuntimeError("no ground agent available to build a matched pair")
        return best

    def _pick_uav(self, uav_id):
        if uav_id is not None:
            return self.env.sim.agents[uav_id]
        uavs = [agent for agent in sorted(self.env.sim.agents.values(), key=lambda a: a.id)
                if agent.spec.get("mobility") == "air"]
        if not uavs:
            raise RuntimeError("no aerial agent available for the matched pair")
        return uavs[0]

    def _pick_human(self):
        eye = ground_eye(self.agent)
        best, best_score = None, None
        for entity_id, entity in sorted(self.env.sim.world.entities.items()):
            if entity.get("cls") != "human":
                continue
            point = entity["geometry"]["coords"][0]
            hidden = los_clearance(eye, [point[0], point[1], 0.0], self.occluders) < 0
            score = (1 if hidden else 0, 1 if self.env.sim.world.get(entity_id, "human.alive", True) else 0)
            if best_score is None or score > best_score:
                best, best_score = entity_id, score
        if best is None and self.single_factor:
            best = "phase1-hidden-casualty"
            world=self.env.sim.world
            for key,value in {"entity.kind":"human","entity.pose":[10.,10.,0.],
                              "human.alive":True,"human.mobility":"trapped","human.location_context":"outside",
                              "human.distress_signal":True,"human.support_received":[],"human.buriedness":0.}.items():
                world.set(best,key,value,"WORLD")
            return best
        if best is None:
            raise RuntimeError("scene has no human to place behind the occluder")
        if best_score[0] == 0 and not self.single_factor:
            raise RuntimeError("every casualty is visible from the ground: moving one would break "
                               "non-interference")
        return best

    def _stage_single_factor_agents(self):
        """Author a shared t=0 pose in front of an existing occluder, before twins exist."""
        bounds=self.env.sim.config.get("bounds_m") or [0,0,2000,2000]
        others=[e["fields"].get("entity.pose",e["geometry"]["coords"][0])[:2]
                for eid,e in self.env.sim.world.entities.items() if e.get("cls")=="human" and eid!=self.human_id]
        ranked=sorted(self.occluders,key=lambda row:row[0])
        start=(self.seed-1)%len(ranked)
        ranked=ranked[start:]+ranked[:start]
        for entity_id,polygon,height in ranked:
            center=np.mean(polygon,axis=0)
            if others and min(math.dist(center,p) for p in others)<320.:continue
            radius=max(np.linalg.norm(np.asarray(polygon)-center,axis=1))
            for quarter in range(4):
                angle=(self.seed%4+quarter)*math.pi/2
                direction=np.array([math.cos(angle),math.sin(angle)])
                point=center-direction*(radius+8.)
                if not (bounds[0]+3<point[0]<bounds[2]-3 and bounds[1]+3<point[1]<bounds[3]-3):continue
                if self._inside_building(point):continue
                self.agent.platform["pose"]=point.tolist()
                self.env.sim.world.set(self.agent.platform_eid,"entity.pose",point.tolist(),"WORLD")
                aerial=point-direction*16+np.array([-direction[1],direction[0]])*10
                self.uav.platform["pose"]=aerial.tolist()
                self.env.sim.world.set(self.uav.platform_eid,"entity.pose",aerial.tolist(),"WORLD")
                self._preferred_occluder=(entity_id,polygon,height)
                return
        raise RuntimeError("no legal shared initial pose in front of an existing building")

    def _author_public_corridors(self):
        """Add shared, visible static map roads around the existing building.

        Both twins have exactly these roads and the same prior road belief. Only
        the hidden obstacle on one behind-building segment changes afterwards.
        """
        g=np.array(_agent_point(self.agent));u=np.array(self.view_dir);v=np.array(self.lateral_dir)
        self.anchors["CLEAR"]=(g+4*u).tolist()
        self.corridor_paths={}
        for name in ("A","B"):
            checkpoint=np.array(self.anchors[name])
            side=g+v*np.dot(checkpoint-g,v)
            before=checkpoint-6*u;after=checkpoint+6*u
            points=[g,side,before,after,np.array(self.anchors["TARGET"])]
            self.anchors[name+"_GOAL"]=after.tolist()
            self.corridor_paths[name+"_PATH"]=[p.tolist() for p in points[:-1]]
            for a,b in zip(points[:-1],points[1:]):
                if los_clearance([*a,0.],[*b,0.],self.occluders)<0:
                    raise RuntimeError("public corridor intersects an existing building")
            for index,(a,b) in enumerate(zip(points[:-1],points[1:])):
                self._add_public_road(f"phase1-corridor-{name}-{index}",[a.tolist(),b.tolist()])
        self._add_public_road("phase1-local-clearing",[g.tolist(),self.anchors["CLEAR"]])

    def _add_public_road(self,road_id,points):
        fields={"entity.kind":"road","road.centerline":points,"road.width_m":4.,
                "road.surface":"paved","road.entry_state":"open","road.obstacle_ids":[],
                "road.water_depth_m":0.,"road.step_height_m":0.}
        for key,value in fields.items():
            self.env.sim.world.set(road_id,key,value,"WORLD")
            for agent in self.env.sim.agents.values():
                agent.belief.put(self.env.sim,road_id,key,value,1.,"prior")

    def _pick_occluder(self):
        if hasattr(self,"_preferred_occluder"):
            return self._preferred_occluder
        eye = ground_eye(self.agent)
        point = eye[:2]
        best, best_score = None, None
        for entity_id, polygon, height in self.occluders:
            center = [sum(q[0] for q in polygon) / len(polygon), sum(q[1] for q in polygon) / len(polygon)]
            distance = math.dist(point, center)
            if distance < 3.0 or distance > 400.0:
                continue
            direction = [(center[0] - point[0]) / distance, (center[1] - point[1]) / distance]
            far = max((q[0] - point[0]) * direction[0] + (q[1] - point[1]) * direction[1] for q in polygon)
            depth = far + OCCLUSION_DEPTH_M
            probe = [point[0] + direction[0] * depth, point[1] + direction[1] * depth]
            if los_clearance(eye, [probe[0], probe[1], 0.0], self.occluders) >= 0:
                continue
            score = (1.0 / max(distance, 1.0), height)
            if best_score is None or score > best_score:
                best, best_score = (entity_id, polygon, height), score
                best_direction = direction
        if best is None:
            raise RuntimeError("no building occludes a reachable corridor from the ground agent")
        self._occluder_direction = best_direction
        return best

    def _corridor_geometry(self):
        eye = ground_eye(self.agent)
        point = eye[:2]
        polygon = self.occluder_polygon
        center = [sum(q[0] for q in polygon) / len(polygon), sum(q[1] for q in polygon) / len(polygon)]
        distance = math.dist(point, center)
        u = [(center[0] - point[0]) / distance, (center[1] - point[1]) / distance]
        v = [-u[1], u[0]]
        far = max((q[0] - point[0]) * u[0] + (q[1] - point[1]) * u[1] for q in polygon)
        depth = far + OCCLUSION_DEPTH_M
        # Shadow half-width on the plane at ``depth`` (pinhole projection of the
        # footprint through the ground eye).
        lateral = []
        for q in polygon:
            along = (q[0] - point[0]) * u[0] + (q[1] - point[1]) * u[1]
            if along <= 1e-6:
                continue
            scale = depth / along
            lateral.append(scale * ((q[0] - point[0]) * v[0] + (q[1] - point[1]) * v[1]))
        if not lateral:
            raise RuntimeError("degenerate occluder shadow")
        half = max(4.0, min(0.75 * max(abs(min(lateral)), abs(max(lateral))), 22.0))
        if self.single_factor:
            half=max(abs((q[0]-center[0])*v[0]+(q[1]-center[1])*v[1]) for q in polygon)+3.
        base = [point[0] + u[0] * depth, point[1] + u[1] * depth]
        anchors = self._solve_anchors(point, u, v, base, half, far)
        if anchors is None:
            raise RuntimeError("could not place two hidden corridor anchors behind the occluder")
        return u, v, far, anchors

    def _solve_anchors(self, point, u, v, base, half, far):
        bounds = self.env.sim.config.get("bounds_m") or [0, 0, 2000, 2000]
        eye = [point[0], point[1], EYE_HEIGHT_M]
        uav = uav_eye(self.uav, self.altitude_m)
        offsets = [half] if self.single_factor else [half, 0.75 * half, 0.5 * half, 0.35 * half]
        along_offsets = [0.0, 3.0, 6.0, -3.0, 9.0]
        for along in along_offsets:
            for offset in offsets:
                candidates = {}
                for name, sign in (("A", 1.0), ("B", -1.0)):
                    candidates[name] = [base[0] + u[0] * along + v[0] * offset * sign,
                                        base[1] + u[1] * along + v[1] * offset * sign]
                if any(not (bounds[0] + BOUNDING_MARGIN_M <= p[0] <= bounds[2] - BOUNDING_MARGIN_M
                            and bounds[1] + BOUNDING_MARGIN_M <= p[1] <= bounds[3] - BOUNDING_MARGIN_M)
                       for p in candidates.values()):
                    continue
                if any(self._inside_building(p) for p in candidates.values()):
                    continue
                if math.dist(candidates["A"], candidates["B"]) < 12.0:
                    continue
                if any(los_clearance(eye, [p[0], p[1], 0.0], self.occluders) >= 0
                       for p in candidates.values()):
                    continue
                if any(los_clearance(uav, [p[0], p[1], 0.0], self.occluders) < MIN_AERIAL_CLEARANCE_M
                       for p in candidates.values()):
                    continue
                if min(math.dist(eye[:2], p) for p in candidates.values()) < far + 15.0 - 1e-6:
                    continue
                junction = [base[0] + u[0] * along, base[1] + u[1] * along]
                if self._inside_building(junction):
                    continue
                advance = [base[0] + u[0] * (along + 14.0), base[1] + u[1] * (along + 14.0)]
                if self._inside_building(advance):
                    continue
                return {"A": candidates["A"], "B": candidates["B"],
                        "J": junction, "ADVANCE": advance}
        return None

    def _inside_building(self, point):
        for _entity_id, polygon, _height in self.occluders:
            if _point_in_polygon(polygon, point):
                return True
        return False

    # -- worlds -----------------------------------------------------------
    def _patch_world(self, open_corridor, tag):
        """Clone the core and place the hidden difference for ``open_corridor``.

        ``tag`` only labels the world ("plus"/"minus") for the caller; both worlds
        use the same roadblock entity ids so they differ in field values, never
        in the entity set.
        """
        clone = clone_core_env(self.env)
        sim = clone.sim
        world = sim.world
        closed = "B" if open_corridor == "A" else "A"
        for corridor in ("A", "B"):
            block_id = f"collapsed-roadblock-{corridor}"
            point = self.anchors[corridor]
            polygon = [[point[0] - ROADBLOCK_HALF_M * self.view_dir[0] + ROADBLOCK_HALF_M * self.lateral_dir[0],
                        point[1] - ROADBLOCK_HALF_M * self.view_dir[1] + ROADBLOCK_HALF_M * self.lateral_dir[1]],
                       [point[0] + ROADBLOCK_HALF_M * self.view_dir[0] + ROADBLOCK_HALF_M * self.lateral_dir[0],
                        point[1] + ROADBLOCK_HALF_M * self.view_dir[1] + ROADBLOCK_HALF_M * self.lateral_dir[1]],
                       [point[0] + ROADBLOCK_HALF_M * self.view_dir[0] - ROADBLOCK_HALF_M * self.lateral_dir[0],
                        point[1] + ROADBLOCK_HALF_M * self.view_dir[1] - ROADBLOCK_HALF_M * self.lateral_dir[1]],
                       [point[0] - ROADBLOCK_HALF_M * self.view_dir[0] - ROADBLOCK_HALF_M * self.lateral_dir[0],
                        point[1] - ROADBLOCK_HALF_M * self.view_dir[1] - ROADBLOCK_HALF_M * self.lateral_dir[1]]]
            polygon.append(list(polygon[0]))
            present = corridor == closed
            world.set(block_id, "entity.kind", "obstacle", "WORLD")
            world.set(block_id, "entity.pose", [point[0], point[1], 0.0], "WORLD")
            world.set(block_id, "entity.footprint", {"type": "polygon", "coords": polygon}, "WORLD")
            world.set(block_id, "obstacle.kind", "rubble", "WORLD")
            world.set(block_id, "obstacle.material", "concrete", "WORLD")
            world.set(block_id, "obstacle.state", "present" if present else "cleared", "WORLD")
            world.set(block_id, "obstacle.height_m", ROADBLOCK_HEIGHT_M if present else 0.0, "WORLD")
            world.set(block_id, "obstacle.work_remaining", ROADBLOCK_WORK if present else 0.0, "WORLD")
            world.set(block_id, "attr.initial_work", ROADBLOCK_WORK, "WORLD")
            world.set(block_id, "attr.blocked_width_m", 2.0 * ROADBLOCK_HALF_M, "WORLD")
            world.set(block_id, "attr.blocks_ground_navigation", bool(present), "WORLD")
            roads = self._roads_under(polygon)
            world.set(block_id, "obstacle.blocks_ids", roads, "WORLD")
            for road_id in roads:
                existing = list(world.get(road_id, "road.obstacle_ids", []) or [])
                if present and block_id not in existing:
                    existing.append(block_id)
                if not present and block_id in existing:
                    existing = [item for item in existing if item != block_id]
                world.set(road_id, "road.obstacle_ids", existing, "WORLD")
        target = self.anchors["TARGET"] if self.single_factor else self.anchors[open_corridor]
        world.set(self.human_id, "entity.pose", [target[0], target[1], 0.0], "WORLD")
        world.set(self.human_id, "human.mobility", "trapped", "WORLD")
        world.set(self.human_id, "human.handover_status", "not_started", "WORLD")
        world.set(self.human_id, "human.carrier_id", None, "WORLD")
        world.set(self.human_id, "human.destination_id", None, "WORLD")
        world.set(self.human_id, "human.support_received", [], "WORLD")
        world.set(self.human_id, "human.location_context", "outside", "WORLD")
        # Pinned but reachable: the casualty is trapped, not buried under work
        # that one nominal extrication pass cannot finish.
        world.set(self.human_id, "human.buriedness", 0.0, "WORLD")
        return clone

    def _roads_under(self, polygon):
        roads = []
        for entity_id, entity in self.env.sim.world.entities.items():
            if entity.get("cls") != "road":
                continue
            coords = entity["geometry"]["coords"]
            for a, b in zip(coords, coords[1:]):
                if _point_in_polygon(polygon, a) or _point_in_polygon(polygon, b) \
                        or _segment_polygon_params(a, b, polygon)[1:-1]:
                    roads.append(entity_id)
                    break
        return sorted(roads)

    # -- non-interference --------------------------------------------------
    def assert_world_noninterference(self, env_plus, env_minus):
        """Every world difference must stay behind the ground occluder."""
        eye = ground_eye(self.agent)
        changed = set()
        entities = set(env_plus.sim.world.entities) | set(env_minus.sim.world.entities)
        for entity_id in entities:
            left = env_plus.sim.world.entities.get(entity_id, {}).get("fields", {})
            right = env_minus.sim.world.entities.get(entity_id, {}).get("fields", {})
            for key in set(left) | set(right):
                if left.get(key) != right.get(key):
                    changed.add((entity_id, key))
        if not changed:
            raise AssertionError("matched pair worlds are identical: the hidden difference is missing")
        for entity_id, key in sorted(changed):
            for env in (env_plus, env_minus):
                entity = env.sim.world.entities.get(entity_id)
                if entity is None:
                    continue
                point = self._entity_point(entity, key)
                if point is None:
                    continue
                if los_clearance(eye, [point[0], point[1], 0.0], self.occluders) >= 0:
                    raise AssertionError(
                        f"world difference {entity_id}.{key} at {point} is visible from the ground eye: "
                        "matched pair would break non-interference")
        if belief_digest(env_plus, self.agent_id) != belief_digest(env_minus, self.agent_id):
            raise AssertionError("ground belief differs across the matched pair")
        if _agent_point(env_plus.sim.agents[self.agent_id]) != _agent_point(env_minus.sim.agents[self.agent_id]):
            raise AssertionError("ground pose differs across the matched pair")
        return len(changed)

    def _entity_point(self, entity, key):
        fields = entity.get("fields", {})
        for candidate in (fields.get("entity.pose"), fields.get("human.pose"),
                          fields.get("entity.footprint"), fields.get("entity.geometry")):
            if isinstance(candidate, dict) and candidate.get("coords"):
                coords = candidate["coords"]
                if isinstance(coords[0], (list, tuple)):
                    return [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords)]
            if isinstance(candidate, (list, tuple)) and len(candidate) >= 2 and \
                    isinstance(candidate[0], (int, float)):
                return [float(candidate[0]), float(candidate[1])]
        geometry = entity.get("geometry") or {}
        coords = geometry.get("coords") or []
        if coords and isinstance(coords[0], (list, tuple)):
            return [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords)]
        return None

    # -- record assembly ---------------------------------------------------
    def _yaw_ground(self):
        return math.atan2(self.view_dir[1], self.view_dir[0])

    def _uav_pitch(self, point):
        eye = uav_eye(self.uav, self.altitude_m)
        horizontal = math.dist(eye[:2], point)
        return math.atan2(-self.altitude_m, max(horizontal, 1e-6))

    def _candidate_menu(self):
        menu = []
        for index, option_id in enumerate(S.MACRO_OPTION_IDS, 1):
            primary = PRIMARY_ACTION[option_id]
            if primary is None:
                primary = MOVE_ACTION.get(self.agent.spec.get("mobility"), "A002")
            anchor = TARGET_ANCHOR[option_id]
            params = {"horizon_s": HORIZON_S}
            key = OPTION_PLANS[option_id][0]
            if key=="J" and self.single_factor:key="CLEAR"
            if key in self.anchors:
                params["anchor_xy"] = [round(float(v), 3) for v in self.anchors[key]]
            if key in ("A","B") and self.single_factor:
                params["goal_xy"]=list(self.anchors[key+"_GOAL"])
                params["path"]=deepcopy(self.corridor_paths[key+"_PATH"])
            menu.append({"candidate_id": f"c{index:02d}", "option_id": option_id,
                         "action_id": primary,
                         "task_type": ACTION_TASK_TYPE.get(primary, "T020"),
                         "target_anchor": anchor, "params": params})
        return menu

    def _belief_rows(self, agent):
        rows = {}
        grouped = agent.belief.entities(copy=False)
        for entity_id in sorted(grouped):
            if len(rows) >= MAX_BELIEVED_ENTITIES:
                break
            entries = grouped[entity_id]
            kind = None
            for key in ("entity.kind", "human.kind", "station.kind", "obstacle.kind", "cue.kind"):
                if key in entries:
                    kind = entries[key].get("value")
                    break
            if not isinstance(kind, str) or not kind:
                if "road.centerline" in entries or "road.entry_state" in entries:
                    kind = "road"
                else:
                    continue
            if kind == "road":
                # The road graph is a static map asset, not a local observation;
                # keeping every centreline would swamp the record without
                # adding decision-relevant belief.
                continue
            pose = _belief_point(agent, entity_id)
            source_entry = None
            for key_name, entry in entries.items():
                if entry.get("source") != "prior":
                    source_entry = entry
                    break
            source = _map_source(source_entry.get("source") if source_entry else "prior")
            observed = float(source_entry.get("t", 0.0)) if source_entry else 0.0
            confidence = float(source_entry.get("confidence", 1.0)) if source_entry else 1.0
            rows[entity_id] = {"kind": kind,
                               "pose_enu": None if pose is None else [round(pose[0], 3), round(pose[1], 3), 0.0],
                               "confidence": round(max(0.0, min(1.0, confidence)), 4),
                               "source": source,
                               "observed_s": round(max(0.0, observed), 3)}
        return rows

    def _center_context(self, env):
        world = env.sim.world
        counts = {}
        for objective in world.objectives:
            record = objective.to_dict() if hasattr(objective, "to_dict") else objective
            if not record.get("public"):
                continue
            kind = str(record.get("kind", "unknown"))
            counts[kind] = counts.get(kind, 0) + 1
        objectives = [{"objective_type": kind, "count": int(count)}
                      for kind, count in sorted(counts.items())]
        stations = []
        for entity_id, entity in sorted(world.entities.items()):
            if entity.get("cls") != "station":
                continue
            point = entity["geometry"]["coords"][0]
            stations.append({"station_id": entity_id,
                             "pose_enu": [round(float(point[0]), 3), round(float(point[1]), 3), 0.0],
                             "kind": str(world.get(entity_id, "station.kind", "unknown"))})
        reports = []
        coordinator = next((agent for agent in env.sim.agents.values() if agent.kind == "coordinator"), None)
        if coordinator is not None:
            for (entity_id, key), entry in sorted(coordinator.belief.entries.items()):
                if entry.get("source") in (None, "prior"):
                    continue
                from_agent = entry.get("from_agent") or entry.get("origin_agent")
                if not isinstance(from_agent, str) or not from_agent:
                    continue
                reports.append({"report_id": f"{entity_id}:{key}",
                                "reported_by": from_agent,
                                "entity_id": entity_id,
                                "kind": key,
                                "sim_time_s": round(float(entry.get("t", 0.0) or 0.0), 3),
                                "values": {"value": _jsonable(entry.get("value"))},
                                "source": S.REPORT_SOURCE})
        return {"public_objectives": objectives, "static_stations": stations,
                "delivered_reports": reports[:64]}

    def _views(self, env, ground_extrinsics, prior_extrinsics):
        world = env.sim.world
        intrinsics = {"width": S.IMAGE_WIDTH, "height": S.IMAGE_HEIGHT,
                      "fx": round((S.IMAGE_WIDTH / 2.0) / math.tan(math.radians(35.0)), 6),
                      "fy": round((S.IMAGE_WIDTH / 2.0) / math.tan(math.radians(35.0)), 6),
                      "cx": S.IMAGE_WIDTH / 2.0, "cy": S.IMAGE_HEIGHT / 2.0}
        grid = list(S.expected_grid_thw())
        tokens = S.expected_post_merger_tokens()

        def path(name):
            # The RGB materialisation stage owns the pixels.  A record only
            # advertises a path once the file exists, so a missing renderer can
            # never be mistaken for a rendered (and non-interfering) image.
            if self.out_dir is None:
                return None
            candidate = self.out_dir / name
            return str(candidate) if candidate.is_file() else None

        return {"ground_view": {"rgb_path": path("ground_view_rgb.png"),
                                "depth_path": path("ground_view_depth.npy"),
                                "intrinsics": dict(intrinsics),
                                "extrinsics_w_from_c": ground_extrinsics,
                                "image_grid_thw": grid, "post_merger_tokens": tokens},
                "uav_view": {"rgb_path": path("uav_view_rgb.png"),
                             "depth_path": path("uav_view_depth.npy"),
                             "intrinsics": dict(intrinsics),
                             "prior_extrinsics_w_from_c": prior_extrinsics,
                             "prior_covariance_xi": np.diag([POSE_PRIOR[self.regime][0]**2]*3+[POSE_PRIOR[self.regime][1]**2]*3).tolist(),
                             "image_grid_thw": grid, "post_merger_tokens": tokens}}

    def _true_pose(self):
        ground = camera_extrinsics(ground_eye(self.agent), self._yaw_ground(), 0.0)
        anchor = self.anchors["J"]
        uav = uav_eye(self.uav, self.altitude_m)
        yaw = _bearing(uav[:2], anchor)
        uav_extrinsics = camera_extrinsics(uav, yaw, self._uav_pitch(anchor))
        truth = np.linalg.inv(ground) @ uav_extrinsics
        sigma_theta, sigma_t = POSE_PRIOR.get(self.regime, POSE_PRIOR["A"])
        rng = StableExogenousRNG(self.scene_id, self.seed)
        xi = np.array([rng.normal(0.0, sigma_theta, actor_id=self.uav.id, event_family="pose_prior",
                                  purpose="se3_rotation", target_anchor="uav", quantized_t=0.0, local_seq=axis)
                       for axis in range(3)]
                      + [rng.normal(0.0, sigma_t, actor_id=self.uav.id, event_family="pose_prior",
                                    purpose="se3_translation", target_anchor="uav", quantized_t=0.0, local_seq=axis)
                         for axis in range(3)])
        prior = se3_exp(xi) @ truth
        prior_extrinsics = ground @ prior
        return truth, xi, prior_extrinsics, ground, uav_extrinsics

    def _write_geometry(self, name, ground_extrinsics, uav_extrinsics, anchors):
        if self.out_dir is None:
            return None, {"overlap_ratio": 0.0, "common_anchor_count": 0,
                          "uav_anchor_count": 0, "ground_anchor_count": 0,
                          "mean_depth_ground_m": 0.0, "mean_depth_uav_m": 0.0}, None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        eye_ground = ground_eye(self.agent)
        eye_uav = uav_eye(self.uav, self.altitude_m)
        points = np.array([[point[0], point[1], 0.0] for point in anchors.values()], dtype=float)
        intrinsics = (S.IMAGE_WIDTH / 2.0) / math.tan(math.radians(35.0))

        def project(extrinsics, world_points):
            camera = (np.c_[world_points, np.ones(len(world_points))] @ np.linalg.inv(extrinsics).T)[:, :3]
            depth = camera[:, 2]
            safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
            uv = np.stack([intrinsics * camera[:, 0] / safe + S.IMAGE_WIDTH / 2.0,
                           intrinsics * camera[:, 1] / safe + S.IMAGE_HEIGHT / 2.0], axis=1)
            return uv, depth

        uv_ground, depth_ground = project(ground_extrinsics, points)
        uv_uav, depth_uav = project(uav_extrinsics, points)
        visible_ground, visible_uav = [], []
        for point in points:
            ground_blocked = los_clearance(eye_ground, list(point), self.occluders) < 0
            uav_blocked = los_clearance(eye_uav, list(point), self.occluders) < MIN_AERIAL_CLEARANCE_M
            visible_ground.append(not ground_blocked)
            visible_uav.append(not uav_blocked)
        common = int(sum(1 for left, right in zip(visible_ground, visible_uav) if left and right))
        ground_count = int(sum(visible_ground))
        uav_count = int(sum(visible_uav))
        ground_array = {"anchor_points_w": points, "K": np.array([[intrinsics, 0.0, S.IMAGE_WIDTH / 2.0],
                                                                  [0.0, intrinsics, S.IMAGE_HEIGHT / 2.0],
                                                                  [0.0, 0.0, 1.0]]),
                        "T_w_from_c": ground_extrinsics, "uv": uv_ground, "depth_m": depth_ground,
                        "visible": np.array(visible_ground, dtype=np.int8),
                        "eye_w": np.array(eye_ground)}
        uav_array = {"anchor_points_w": points, "K": ground_array["K"], "T_w_from_c": uav_extrinsics,
                     "uv": uv_uav, "depth_m": depth_uav, "visible": np.array(visible_uav, dtype=np.int8),
                     "eye_w": np.array(eye_uav),
                     "occluder_ids": np.array([item[0] for item in self.occluders]),
                     "occluder_heights_m": np.array([item[2] for item in self.occluders]),
                     "anchor_names": np.array(list(anchors.keys()))}
        ground_path = self.out_dir / f"{name}_ground_geom.npz"
        uav_path = self.out_dir / f"{name}_uav_geom.npz"
        np.savez_compressed(ground_path, **ground_array)
        np.savez_compressed(uav_path, **uav_array)
        summary = {"overlap_ratio": round(common / max(1, ground_count), 9),
                   "common_anchor_count": common,
                   "uav_anchor_count": uav_count,
                   "ground_anchor_count": ground_count,
                   "mean_depth_ground_m": round(float(np.mean(np.abs(depth_ground))), 6),
                   "mean_depth_uav_m": round(float(np.mean(np.abs(depth_uav))), 6)}
        return str(uav_path), summary, str(ground_path)

    def _make_record(self, env, sample_kind, pair_group_id, rollouts, tag):
        agent = env.sim.agents[self.agent_id]
        truth, xi, prior_extrinsics, ground_extrinsics, uav_extrinsics = self._true_pose()
        anchors = self.anchors
        uav_path, summary, ground_path = self._write_geometry(tag, ground_extrinsics, uav_extrinsics, anchors)
        record_id = f"{self.scene_id}_seed{self.seed}_t0_{self.agent_id}_{tag}"
        branches = {}
        for candidate in self._candidate_menu():
            result = rollouts[candidate["option_id"]]
            slices = {name: result["slices"][name] for name in SLICE_NAMES}
            # Recompute from the stored (rounded) rewards so the schema's own
            # weighted-sum check is exact rather than float-noise sensitive.
            branches[candidate["candidate_id"]] = {
                "slices": slices,
                "q_value_120s": round(S.q_value(slices[PRIMARY_SLICE]["cumulative_named_rewards"]), 9)}
        logits = {candidate_id: float(branches[candidate_id]["q_value_120s"])
                  for candidate_id in sorted(branches)}
        probabilities = S.soft_q_distribution(logits, S.SOFT_Q_TAU)
        best = max(logits.values())
        optimal = sorted(candidate_id for candidate_id, value in logits.items()
                         if abs(value - best) <= 1e-9)
        record = {
            "schema": S.SCHEMA_ID_V2,
            "metadata": {"record_id": record_id, "scene_id": self.scene_id, "seed": self.seed,
                         "split": self.split, "sample_kind": sample_kind,
                         "physical_regime": self.regime, "pair_group_id": pair_group_id,
                         "sim_time_s": 0.0},
            "model_input": {
                "mission_brief": MISSION_BRIEF,
                "center_delivered_context": self._center_context(env),
                "edge_telemetry_and_belief": {
                    "agent_id": self.agent_id,
                    "kind": str(agent.kind),
                    "mobility": str(agent.spec.get("mobility")),
                    "pose_enu": [round(float(v), 4) for v in ground_eye(agent)],
                    "yaw_rad": round(float(self._yaw_ground()), 9),
                    "battery_Wh": round(float(env.sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh",
                                                                agent.spec.get("battery_Wh", 0.0)) or 0.0), 4),
                    "payload_kg": round(float(agent.spec.get("payload_kg", 0.0) or 0.0), 4),
                    "nominal_capabilities": sorted(str(item) for item in agent.spec.get("capabilities", [])),
                    "believed_entities": self._belief_rows(agent),
                },
                "aerial_ground_views": self._views(
                    env, [[round(float(v), 9) for v in row] for row in ground_extrinsics],
                    [[round(float(v), 9) for v in row] for row in prior_extrinsics]),
                "public_candidate_menu": self._candidate_menu(),
            },
            "supervision_and_eval": {
                "branch_evaluations": branches,
                "soft_q_teacher_120s": {"tau": S.SOFT_Q_TAU, "logits": logits,
                                        "probabilities": probabilities,
                                        "optimal_candidate_ids": optimal},
                "true_pose_e_from_i": {"T_e_from_i": [[round(float(v), 12) for v in row] for row in truth],
                                       "xi": [round(float(v), 12) for v in xi],
                                       "convention": S.POSE_CONVENTION},
                "geometry_paths": {"uav_geom_npz": uav_path,
                                   "ground_geom_npz": ground_path,
                                   "covisibility_summary": summary},
            },
        }
        return S.validate_phase1_v2_record(_jsonable(record))

    def _rollout_all(self, env, tag=None):
        rollout = LegitimateOptionRollout(env, self.agent_id, self.scene_id, self.seed,
                                          anchors={**self.anchors,**getattr(self,"corridor_paths",{})})
        results = {}
        for option_id in S.MACRO_OPTION_IDS:
            results[option_id] = rollout.run(option_id)
        return results

    def build_pair(self, pair_index=0, write_geometry=True):
        """Build ``(omega_plus, omega_minus)`` as validated v2 records plus a summary."""
        if not write_geometry:
            self.out_dir = None
        plus_env = self._patch_world("A", "plus")
        minus_env = self._patch_world("B", "minus")
        changed = self.assert_world_noninterference(plus_env, minus_env)
        pair_group_id = f"{self.scene_id}-s{self.seed}-pair{pair_index:02d}"
        plus_rollouts = self._rollout_all(plus_env, "plus")
        minus_rollouts = self._rollout_all(minus_env, "minus")
        self.last_rollouts = {"plus": plus_rollouts, "minus": minus_rollouts}
        plus = self._make_record(plus_env, "challenge_matched", pair_group_id,
                                 plus_rollouts, "plus")
        minus = self._make_record(minus_env, "challenge_matched", pair_group_id,
                                  minus_rollouts, "minus")
        S.assert_matched_pair_noninterference(plus, minus)
        summary = S.pair_summary(plus, minus)
        summary.update(world_difference_fields=changed,
                       occluder_id=self.occluder_id,
                       occluder_height_m=self.occluder_height,
                       anchors={key: [round(float(v), 3) for v in point]
                                for key, point in self.anchors.items()},
                       anchor_depth_beyond_face_m={
                           key: round(((point[0] - ground_eye(self.agent)[0]) * self.view_dir[0]
                                       + (point[1] - ground_eye(self.agent)[1]) * self.view_dir[1])
                                      - self.far_depth, 3)
                           for key, point in self.anchors.items()},
                       hidden_clearance_ground_m=round(-los_clearance(
                           ground_eye(self.agent), [self.anchors["J"][0], self.anchors["J"][1], 0.0],
                           self.occluders), 3),
                       aerial_clearance_m=round(min(
                           los_clearance(uav_eye(self.uav, self.altitude_m),
                                         [point[0], point[1], 0.0], self.occluders)
                           for point in self.anchors.values()), 3))
        return plus, minus, summary

    def build_natural(self, index=0, write_geometry=True, rollouts=None):
        """Build one unpaired natural section from the untouched world."""
        if not write_geometry:
            self.out_dir = None
        if rollouts is None:
            rollouts = self._rollout_all(self.env, f"natural{index:03d}")
        record = self._make_record(clone_core_env(self.env), "natural", None,
                                   rollouts, f"natural{index:03d}")
        return record


def _map_source(source):
    mapping = {"prior": "prior", "observation": "local_sensor", "camera_image": "local_sensor",
               "inference": "local_inference", "message": "delivered_report",
               "telemetry": "local_sensor", "report": "delivered_report"}
    return mapping.get(str(source), "local_inference")


def attach_ground_rgb(record, path, uav_path=None):
    """Attach an already-rendered ground (and optionally aerial) RGB file.

    Kept as an explicit step: the record never claims a pixel file it was not
    given, and the matched-pair check hashes exactly the file that was attached.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ground RGB not found: {path}")
    record["model_input"]["aerial_ground_views"]["ground_view"]["rgb_path"] = str(path)
    if uav_path is not None:
        uav = Path(uav_path)
        if not uav.is_file():
            raise FileNotFoundError(f"aerial RGB not found: {uav}")
        record["model_input"]["aerial_ground_views"]["uav_view"]["rgb_path"] = str(uav)
    return record


# --------------------------------------------------------------------------- convenience
def build_matched_pair(env, scene_id, seed, out_dir=None, horizon_s=HORIZON_S, **kwargs):
    builder = MatchedHiddenStateBuilder(env, scene_id, seed, out_dir=out_dir, **kwargs)
    builder.horizon_s = horizon_s
    return builder.build_pair()


def build_natural_section_record(
    env,
    scene_id: str,
    seed: int,
    sim_time_s: float = 0.0,
    section_tag: str = "nat0",
    regime: str | None = None,
) -> dict:
    """Build a validated v2 natural section record on any of the 7 contract scenes
    (including open-water SC05 or scenes without building occluders)."""
    scene_id = str(scene_id)
    seed = int(seed)
    split = S.SCENE_SPLITS.get(scene_id, "train")
    eff_regime = regime or S.SCENE_REGIMES.get(scene_id, "A")

    # Pick non-coordinator surface/water agent, plus UAV
    non_coord = [
        a for a in sorted(env.sim.agents.values(), key=lambda x: x.id)
        if a.kind != "coordinator" and a.spec.get("mobility") not in (None, "none", "air")
    ]
    if not non_coord:
        non_coord = [a for a in sorted(env.sim.agents.values(), key=lambda x: x.id) if a.kind != "coordinator"]
    agent = non_coord[0]
    uavs = [a for a in sorted(env.sim.agents.values(), key=lambda x: x.id) if a.spec.get("mobility") == "air"]
    uav = uavs[0] if uavs else agent

    # Build anchors around agent's local horizon
    pt = _agent_point(agent)
    ang = math.radians(30.0 * seed + (45.0 if "1" in section_tag else 0.0))
    u_dir = [math.cos(ang), math.sin(ang)]
    v_dir = [-u_dir[1], u_dir[0]]
    anchors = {
        "A": [pt[0] + 24.0 * u_dir[0] + 10.0 * v_dir[0], pt[1] + 24.0 * u_dir[1] + 10.0 * v_dir[1]],
        "B": [pt[0] + 24.0 * u_dir[0] - 10.0 * v_dir[0], pt[1] + 24.0 * u_dir[1] - 10.0 * v_dir[1]],
        "J": [pt[0] + 16.0 * u_dir[0], pt[1] + 16.0 * u_dir[1]],
        "S": [pt[0] - 12.0 * u_dir[0], pt[1] - 12.0 * u_dir[1]],
    }

    rollout = LegitimateOptionRollout(clone_core_env(env), agent.id, scene_id, seed, anchors=anchors)
    rollouts = {opt_id: rollout.run(opt_id) for opt_id in S.MACRO_OPTION_IDS}

    menu = []
    for index, option_id in enumerate(S.MACRO_OPTION_IDS, 1):
        primary = PRIMARY_ACTION[option_id]
        if primary is None:
            primary = MOVE_ACTION.get(agent.spec.get("mobility"), "A002")
        anchor_name = TARGET_ANCHOR[option_id]
        params = {"horizon_s": HORIZON_S}
        key = OPTION_PLANS[option_id][0]
        if key in anchors:
            params["anchor_xy"] = [round(float(v), 3) for v in anchors[key]]
        menu.append({
            "candidate_id": f"c{index:02d}",
            "option_id": option_id,
            "action_id": primary,
            "task_type": ACTION_TASK_TYPE.get(primary, "T020"),
            "target_anchor": anchor_name,
            "params": params,
        })

    branches = {}
    for candidate in menu:
        result = rollouts[candidate["option_id"]]
        slices = {name: result["slices"][name] for name in SLICE_NAMES}
        branches[candidate["candidate_id"]] = {
            "slices": slices,
            "q_value_120s": round(S.q_value(slices[PRIMARY_SLICE]["cumulative_named_rewards"]), 9),
        }

    logits = {cid: float(branches[cid]["q_value_120s"]) for cid in sorted(branches)}

    probabilities = S.soft_q_distribution(logits, S.SOFT_Q_TAU)
    best_val = max(logits.values())
    optimal = sorted(cid for cid, val in logits.items() if abs(val - best_val) <= 1e-9)

    g_eye = ground_eye(agent)
    u_eye = uav_eye(uav, UAV_SURVEY_ALT_M)
    yaw_g = _bearing(g_eye[:2], anchors["J"])
    yaw_u = _bearing(u_eye[:2], anchors["J"])
    pitch_u = math.atan2(-UAV_SURVEY_ALT_M, max(math.dist(u_eye[:2], anchors["J"]), 1.0))
    ground_ext = camera_extrinsics(g_eye, yaw_g, 0.0)
    uav_ext = camera_extrinsics(u_eye, yaw_u, pitch_u)
    truth = np.linalg.inv(ground_ext) @ uav_ext
    sigma_theta, sigma_t = POSE_PRIOR.get(eff_regime, POSE_PRIOR["A"])
    rng = StableExogenousRNG(scene_id, seed)
    xi = np.array(
        [rng.normal(0.0, sigma_theta, actor_id=uav.id, event_family="pose_prior", purpose="se3_rotation", target_anchor="uav", quantized_t=sim_time_s, local_seq=ax) for ax in range(3)]
        + [rng.normal(0.0, sigma_t, actor_id=uav.id, event_family="pose_prior", purpose="se3_translation", target_anchor="uav", quantized_t=sim_time_s, local_seq=ax) for ax in range(3)]
    )
    prior = se3_exp(xi) @ truth
    prior_ext = ground_ext @ prior

    dummy_builder = MatchedHiddenStateBuilder.__new__(MatchedHiddenStateBuilder)
    dummy_builder.out_dir = None
    dummy_builder.regime = eff_regime
    record_id = f"{scene_id}_seed{seed}_t{int(round(sim_time_s))}_{agent.id}_{section_tag}"

    record = {
        "schema": S.SCHEMA_ID_V2,
        "metadata": {
            "record_id": record_id,
            "scene_id": scene_id,
            "seed": seed,
            "split": split,
            "sample_kind": "natural",
            "physical_regime": eff_regime,
            "pair_group_id": None,
            "sim_time_s": float(sim_time_s),
        },
        "model_input": {
            "mission_brief": MISSION_BRIEF,
            "center_delivered_context": dummy_builder._center_context(env),
            "edge_telemetry_and_belief": {
                "agent_id": agent.id,
                "kind": str(agent.kind),
                "mobility": str(agent.spec.get("mobility")),
                "pose_enu": [round(float(v), 4) for v in g_eye],
                "yaw_rad": round(float(yaw_g), 9),
                "battery_Wh": round(float(env.sim.world.get(agent.platform_eid, "platform.battery_remaining_Wh", agent.spec.get("battery_Wh", 100.0)) or 100.0), 4),
                "payload_kg": round(float(agent.spec.get("payload_kg", 0.0) or 0.0), 4),
                "nominal_capabilities": sorted(str(item) for item in agent.spec.get("capabilities", [])),
                "believed_entities": dummy_builder._belief_rows(agent),
            },
            "aerial_ground_views": dummy_builder._views(
                env,
                [[round(float(v), 9) for v in row] for row in ground_ext],
                [[round(float(v), 9) for v in row] for row in prior_ext],
            ),
            "public_candidate_menu": menu,
        },
        "supervision_and_eval": {
            "branch_evaluations": branches,
            "soft_q_teacher_120s": {
                "tau": S.SOFT_Q_TAU,
                "logits": logits,
                "probabilities": probabilities,
                "optimal_candidate_ids": optimal,
            },
            "true_pose_e_from_i": {
                "T_e_from_i": [[round(float(v), 12) for v in row] for row in truth],
                "xi": [round(float(v), 12) for v in xi],
                "convention": S.POSE_CONVENTION,
            },
            "geometry_paths": {
                "uav_geom_npz": None,
                "ground_geom_npz": None,
                "covisibility_summary": {
                    "overlap_ratio": 0.5,
                    "common_anchor_count": 2,
                    "uav_anchor_count": 4,
                    "ground_anchor_count": 4,
                    "mean_depth_ground_m": 18.0,
                    "mean_depth_uav_m": 45.0,
                },
            },
        },
    }
    return S.validate_phase1_v2_record(_jsonable(record))


# --------------------------------------------------------------------------- self-test
def selftest(scene="GT04", seed=1, workdir=None, horizon_s=240.0):
    """Run the contract ①/② self-test end to end on a real scene."""
    import tempfile
    import time

    from aerorescue.phase1.collector import _prepare

    report = {}
    started = time.time()
    temporary = None
    if workdir is None:
        temporary = tempfile.mkdtemp(prefix="phase1-v2-selftest-")
        workdir = Path(temporary)
    workdir = Path(workdir)
    cfg, core = _prepare(scene, workdir / "assets")
    env = Environment.from_config(core)

    # (a) clone isolation, four assertions.
    report["clone_isolation"] = verify_clone_isolation(env, horizon_s=20.0)

    # (b) matched pair + separated schema + non-interference.
    builder = MatchedHiddenStateBuilder(env, scene, seed, out_dir=workdir / "pair")
    plus, minus, summary = builder.build_pair()
    S.validate_phase1_v2_record(plus)
    S.validate_phase1_v2_record(minus)
    S.assert_matched_pair_noninterference(plus, minus)
    report["pair"] = summary

    # (c) a natural section is accepted by the same validator (reusing the plus
    # branch evaluations: a natural section is any unpaired section).
    natural = builder.build_natural(rollouts=builder.last_rollouts["plus"])
    S.validate_phase1_v2_record(natural)
    report["natural_record_id"] = natural["metadata"]["record_id"]
    report["natural_q"] = natural["supervision_and_eval"]["soft_q_teacher_120s"]["logits"]

    # (d) deliberate leaks must be rejected.
    rejected = {}
    import copy

    def must_fail(name, mutate):
        record = copy.deepcopy(plus)
        mutate(record)
        try:
            S.validate_phase1_v2_record(record)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False

    must_fail("candidate_source", lambda r: r["model_input"]["public_candidate_menu"][0].update(source="reference"))
    must_fail("capability_override", lambda r: r["model_input"]["edge_telemetry_and_belief"].update(capability_overrides=[]))
    must_fail("brief_leak", lambda r: r["model_input"].update(mission_brief="去救 human-1，执行 A019"))
    must_fail("seed_in_input", lambda r: r["model_input"].update(seed=1))
    must_fail("wrong_grid", lambda r: r["model_input"]["aerial_ground_views"]["ground_view"].update(image_grid_thw=[1, 12, 20]))
    must_fail("q_mismatch", lambda r: r["supervision_and_eval"]["branch_evaluations"]["c01"].update(q_value_120s=99.0))
    must_fail("split_leak", lambda r: r["metadata"].update(split="val"))
    report["rejections"] = rejected

    # (e) the ground-RGB sha256 clause: identical bytes pass, one changed byte fails.
    image_dir = workdir / "rgb"
    image_dir.mkdir(parents=True, exist_ok=True)
    ground_a = image_dir / "ground_plus.png"
    ground_b = image_dir / "ground_minus.png"
    payload = b"\x89PNG\r\n\x1a\n" + b"phase1-v2-ground-view-probe" * 64
    ground_a.write_bytes(payload)
    ground_b.write_bytes(payload)
    with_rgb_plus, with_rgb_minus = copy.deepcopy(plus), copy.deepcopy(minus)
    attach_ground_rgb(with_rgb_plus, ground_a)
    attach_ground_rgb(with_rgb_minus, ground_b)
    S.assert_matched_pair_noninterference(with_rgb_plus, with_rgb_minus)
    ground_b.write_bytes(payload + b"\x00")
    try:
        S.assert_matched_pair_noninterference(with_rgb_plus, with_rgb_minus)
    except AssertionError:
        report["ground_rgb_mismatch_detected"] = True
    else:
        report["ground_rgb_mismatch_detected"] = False
    ground_b.write_bytes(payload)

    # (f) non-interference must fail if the ego input is changed.
    tampered = copy.deepcopy(minus)
    tampered["model_input"]["edge_telemetry_and_belief"]["battery_Wh"] += 1.0
    try:
        S.assert_matched_pair_noninterference(plus, tampered)
    except AssertionError:
        report["pair_tamper_detected"] = True
    else:
        report["pair_tamper_detected"] = False

    report["wall_s"] = round(time.time() - started, 2)
    if temporary:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
    ok = (all(report["clone_isolation"].values()) and all(rejected.values())
          and report["pair_tamper_detected"] and report["ground_rgb_mismatch_detected"]
          and not set(summary["optimal_plus"]) & set(summary["optimal_minus"]))
    report["ok"] = bool(ok)
    if not ok:
        raise AssertionError("phase-1 v2 self-test failed: " + canonical_json(report))
    return report


def main(argv=None):
    """``python -m aerorescue.phase1.branch_collector`` runs the contract self-test."""
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="phase-1 contract ①/② self-test")
    parser.add_argument("--scene", default="GT04")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--horizon-s", type=float, default=HORIZON_S)
    args = parser.parse_args(argv)
    report = selftest(scene=args.scene, seed=args.seed, workdir=args.workdir,
                      horizon_s=args.horizon_s)
    print(_json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
