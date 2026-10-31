"""Phase-1 v2 record contract: physical separation of model input and supervision.

A v2 record is one decision section of one *branch world*.  Contract ① splits it
into two top-level layers that never mix:

``metadata``
    Dataset bookkeeping only (``record_id``/``scene_id``/``seed``/``split``/
    ``sample_kind``/``physical_regime``/``pair_group_id``/``sim_time_s``).
    Never handed to the model.
``model_input``
    Exactly what the edge agent may see at section time: the static scene brief,
    the centre's *delivered* context, its own telemetry/belief, the two sensor
    views and the public candidate menu.  No seed, no split, no reference
    action, no feasibility/oracle flag, no capability override.
``supervision_and_eval``
    Offline only: real counterfactual branch rewards on three horizons, the
    soft-Q teacher distribution, the true relative pose and the geometry paths.
    ``model_input`` never contains any of it.

Two hard assertions are exported:

``validate_phase1_v2_record(record)``
    Fails loudly instead of coercing, so a collector bug cannot ship a leaked or
    mislabelled dataset.

``assert_matched_pair_noninterference(rec_plus, rec_minus)``
    The matched-hidden-state pair check of contract ①: identical ego input
    (telemetry/belief, delivered centre context, mission brief, candidate menu,
    ground RGB bytes) while the optimal macro-action sets are disjoint, which is
    what caps any ground-only policy at 50% on the pair.

Nothing here touches a simulator: the module is pure data validation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA_ID_V2 = "aerorescue.phase1.decision_record/v4"

# --------------------------------------------------------------------------- axes
SPLITS = frozenset({"train", "val", "test"})
SAMPLE_KINDS = frozenset({"natural", "challenge_matched"})
REGIMES = frozenset({"A", "B", "C", "water"})

# Frozen three-axis partition (contract ①.3).  A scene never crosses a split and
# keeps its measured/authored regime, so a seed, twin or branch cannot leak
# across the train/val/test boundary.
SCENE_SPLITS = {
    "SC01": "train", "SC09": "train",
    "SC07": "val", "SC10": "val",
    "SC08": "test", "GT04": "test", "SC05": "test",
}
SCENE_REGIMES = {
    "SC01": "A", "SC04": "A", "SC09": "A", "SC13": "A",
    "SC05": "water",
    "SC07": "B", "SC10": "B", "SC11": "B",
    "SC08": "C", "GT04": "C", "SC12": "C", "SC17": "C",
}

# --------------------------------------------------------------------------- options
# The K in [4, 6] named macro-action options of contract ②.  The menu may use a
# subset, but never an invented name: the option semantics live in
# ``branch_collector.LegitimateOptionRollout``.
MACRO_OPTION_IDS = (
    "OPT_ROUTE_A_SEARCH_RESCUE",
    "OPT_ROUTE_B_SEARCH_RESCUE",
    "OPT_CLEAR_CORRIDOR_AND_ADVANCE",
    "OPT_DIRECT_APPROACH_AND_AID",
    "OPT_HOLD_AND_SCAN",
)
MIN_CANDIDATES = 4
MAX_CANDIDATES = 6

# --------------------------------------------------------------------------- rewards
# Ten named rewards of contract ②.3; energy and busy time are costs, so their
# weights are negative and every reward is a non-negative measured quantity.
NAMED_REWARD_KEYS = (
    "person_found", "contacted", "extricated", "aided", "handed_over",
    "road_cleared", "hazard_handled", "action_failed", "energy_Wh", "active_busy_s",
)
REWARD_WEIGHTS = {
    "person_found": 0.25, "contacted": 0.20, "extricated": 0.30, "aided": 0.30,
    "handed_over": 0.30, "road_cleared": 0.25, "hazard_handled": 0.25,
    "action_failed": -0.50, "energy_Wh": -0.02, "active_busy_s": -0.001,
}
SLICE_KEYS = ("H_60s", "H_120s", "H_240s")
SLICE_HORIZONS = {"H_60s": 60.0, "H_120s": 120.0, "H_240s": 240.0}
PRIMARY_SLICE = "H_120s"
SOFT_Q_TAU = 0.1

# --------------------------------------------------------------------------- keys
METADATA_KEYS = ("record_id", "scene_id", "seed", "split", "sample_kind",
                 "physical_regime", "pair_group_id", "sim_time_s")
MODEL_INPUT_KEYS = ("mission_brief", "center_delivered_context",
                    "edge_telemetry_and_belief", "aerial_ground_views",
                    "public_candidate_menu")
CENTER_KEYS = ("public_objectives", "static_stations", "delivered_reports")
OBJECTIVE_KEYS = ("objective_type", "count")
STATION_KEYS = ("station_id", "pose_enu", "kind")
REPORT_KEYS = ("report_id", "reported_by", "entity_id", "kind", "sim_time_s",
               "values", "source")
REPORT_SOURCE = "network_delivered"
EDGE_KEYS = ("agent_id", "kind", "mobility", "pose_enu", "yaw_rad", "battery_Wh",
             "payload_kg", "nominal_capabilities", "believed_entities")
BELIEF_ENTRY_KEYS = ("kind", "pose_enu", "confidence", "source", "observed_s")
BELIEF_SOURCES = frozenset({"prior", "local_sensor", "local_inference", "delivered_report"})
VIEWS_KEYS = ("uav_view", "ground_view")
GROUND_VIEW_KEYS = ("rgb_path", "depth_path", "intrinsics", "extrinsics_w_from_c",
                    "image_grid_thw", "post_merger_tokens")
UAV_VIEW_KEYS = ("rgb_path", "depth_path", "intrinsics", "prior_extrinsics_w_from_c", "prior_covariance_xi",
                 "image_grid_thw", "post_merger_tokens")
INTRINSICS_KEYS = ("width", "height", "fx", "fy", "cx", "cy")
CANDIDATE_KEYS = ("candidate_id", "option_id", "action_id", "task_type",
                  "target_anchor", "params")
SUPERVISION_KEYS = ("branch_evaluations", "soft_q_teacher_120s",
                    "true_pose_e_from_i", "geometry_paths")
SLICE_ROW_KEYS = ("horizon_s", "named_rewards", "cumulative_named_rewards", "milestones", "first_achieved_s")
BRANCH_KEYS = ("slices", "q_value_120s")
TEACHER_KEYS = ("tau", "logits", "probabilities", "optimal_candidate_ids")
POSE_KEYS = ("T_e_from_i", "xi", "convention")
GEOMETRY_KEYS = ("uav_geom_npz", "ground_geom_npz", "covisibility_summary")
COVISIBILITY_KEYS = ("overlap_ratio", "common_anchor_count", "uav_anchor_count",
                     "ground_anchor_count", "mean_depth_ground_m", "mean_depth_uav_m")
POSE_CONVENTION = "camera_optical_x_right_y_down_z_forward"

# Contract ③: 640x384 sensor images, patch 16, merge 2 -> patch grid 24x40 and
# 12*20 = 240 post-merger visual tokens.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 384
PATCH_SIZE = 16
MERGE_SIZE = 2
IMAGE_GRID_THW = [1, 24, 40]
POST_MERGER_TOKENS = 240

# Answer hints that must never appear anywhere in ``model_input``.  ``source``
# is legitimate provenance on delivered reports, static stations and local
# belief rows, so it is only banned inside the candidate menu (where the v1
# contract forbids it explicitly).
GLOBAL_LEAK_KEYS = frozenset({"is_reference", "reference", "oracle", "oracle_feasible",
                              "utility", "feasible", "priority", "best_candidate",
                              "is_best", "label", "optimal", "q_value", "reward",
                              "teacher"})
CANDIDATE_LEAK_KEYS = GLOBAL_LEAK_KEYS | {"source"}
# Capability accounting that would tell the model what the branch added.
OVERRIDE_KEY_TOKENS = ("override", "effective_capabilit", "granted_capabilit")
# Dataset bookkeeping that must never reach the model.
MODEL_INPUT_BANNED_KEYS = frozenset({"seed", "split", "record_id", "pair_group_id",
                                     "sample_kind", "physical_regime", "episode_index"})

_ACTION_ID = re.compile(r"^A\d{3}$")
_ACTION_IN_TEXT = re.compile(r"\bA\d{3}\b")
_TASK_TYPE = re.compile(r"^T\d{3}$")
# Entity identifiers of the hidden scene: ``human-1``, ``building-2``,
# ``obstacle-1`` ... plus the generic ``<class>-<n>`` forms used by the scenes.
_ENTITY_ID = re.compile(
    r"\b(?:human|building|obstacle|road|hazard|item|station|platform|uav|ground_vehicle|"
    r"specialist_team|specialist-team|clearance_loader|coordinator|missing-person|"
    r"rural-casualty|ash-casualty|landslide-casualty|water-casualty)[-_]?\d+\b",
    re.IGNORECASE)
_REFERENCE_WORD = re.compile(r"\breference\b", re.IGNORECASE)


def _fail(message: str):
    raise ValueError("invalid phase1 v2 record: " + message)


def _number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def canonical_json(value) -> str:
    """Deterministic JSON used by every digest in the v2 pipeline."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_grid_thw(width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> list:
    return [1, height // PATCH_SIZE, width // PATCH_SIZE]


def expected_post_merger_tokens(width: int = IMAGE_WIDTH, height: int = IMAGE_HEIGHT) -> int:
    return (height // PATCH_SIZE // MERGE_SIZE) * (width // PATCH_SIZE // MERGE_SIZE)


def q_value(named_rewards: dict) -> float:
    """Scalar branch utility of contract ②.3, identical in collector and schema."""
    return float(sum(REWARD_WEIGHTS[key] * float(named_rewards.get(key, 0.0))
                     for key in NAMED_REWARD_KEYS))


def soft_q_distribution(logits: dict, tau: float = SOFT_Q_TAU) -> dict:
    """Numerically stable Softmax(Q/tau); shared by the builder and the validator."""
    if not logits:
        return {}
    peak = max(float(value) for value in logits.values())
    weights = {key: math.exp((float(value) - peak) / float(tau)) for key, value in logits.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


# --------------------------------------------------------------------------- scanning
def _scan_keys(value, path: str, banned_keys, banned_tokens=()):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                lowered = key.lower()
                if lowered in banned_keys:
                    _fail(f"forbidden key {key} at {path}.{key}")
                if any(token in lowered for token in banned_tokens):
                    _fail(f"capability-override key {key} at {path}.{key}")
            _scan_keys(item, f"{path}.{key}", banned_keys, banned_tokens)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_keys(item, f"{path}[{index}]", banned_keys, banned_tokens)


def _scan_leaks(value, path: str, banned_keys=GLOBAL_LEAK_KEYS):
    _scan_keys(value, path, banned_keys)


def _se3_matrix(value, path: str):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        _fail(f"{path} must be a 4x4 matrix")
    rows = []
    for index, row in enumerate(value):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            _fail(f"{path}[{index}] must have four entries")
        if not all(_number(item) for item in row):
            _fail(f"{path}[{index}] entries must be finite numbers")
        rows.append([float(item) for item in row])
    if any(abs(rows[3][k] - (1.0 if k == 3 else 0.0)) > 1e-9 for k in range(4)):
        _fail(f"{path} last row must be [0, 0, 0, 1]")
    rotation = [row[:3] for row in rows[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[i][k] * rotation[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-6:
                _fail(f"{path} rotation block is not orthonormal")
    det = (rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
           - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
           + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]))
    if abs(det - 1.0) > 1e-6:
        _fail(f"{path} rotation determinant must be +1")
    return rows


# --------------------------------------------------------------------------- metadata
def _validate_metadata(record: dict):
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        _fail("metadata must be an object")
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        _fail("metadata missing keys: " + ", ".join(missing))
    extra = sorted(set(metadata) - set(METADATA_KEYS))
    if extra:
        _fail("metadata has unknown keys: " + ", ".join(extra))
    record_id = metadata["record_id"]
    if not isinstance(record_id, str) or not record_id:
        _fail("metadata.record_id must be a non-empty string")
    scene = metadata["scene_id"]
    if not isinstance(scene, str) or not scene:
        _fail("metadata.scene_id must be a non-empty string")
    if not _integer(metadata["seed"]) or metadata["seed"] < 0:
        _fail("metadata.seed must be a non-negative integer")
    if metadata["split"] not in SPLITS:
        _fail("metadata.split must be one of train/val/test")
    if scene in SCENE_SPLITS and metadata["split"] != SCENE_SPLITS[scene]:
        _fail(f"scene {scene} belongs to split {SCENE_SPLITS[scene]}, not {metadata['split']}")
    if metadata["sample_kind"] not in SAMPLE_KINDS:
        _fail("metadata.sample_kind must be natural or challenge_matched")
    if metadata["physical_regime"] not in REGIMES:
        _fail("metadata.physical_regime must be A, B, C or water")
    pair_group_id = metadata["pair_group_id"]
    if metadata["sample_kind"] == "challenge_matched":
        if not isinstance(pair_group_id, str) or not pair_group_id:
            _fail("challenge_matched records must carry a non-empty pair_group_id")
    elif pair_group_id is not None:
        _fail("natural records must carry pair_group_id = None")
    if not _number(metadata["sim_time_s"]) or float(metadata["sim_time_s"]) < 0:
        _fail("metadata.sim_time_s must be a non-negative finite number")
    if scene not in record_id:
        _fail("record_id must contain scene_id")
    if f"seed{int(metadata['seed'])}" not in record_id:
        _fail("record_id must contain seed<seed>")


# --------------------------------------------------------------------------- model input
def _validate_brief(brief: str, menu: list):
    if not isinstance(brief, str) or not brief.strip():
        _fail("mission_brief must be a non-empty string")
    leaked = _ACTION_IN_TEXT.findall(brief)
    if leaked:
        _fail("mission_brief leaks atomic action ids: " + ", ".join(sorted(set(leaked))))
    entity = _ENTITY_ID.search(brief)
    if entity:
        _fail("mission_brief leaks a target entity id: " + entity.group(0))
    if _REFERENCE_WORD.search(brief):
        _fail("mission_brief must not mention a reference answer")
    for candidate in menu:
        for key in ("candidate_id", "option_id", "target_anchor"):
            value = str(candidate.get(key, ""))
            if value and value in brief:
                _fail(f"mission_brief leaks the candidate field {key}: {value}")


def _validate_center(center: dict):
    if not isinstance(center, dict):
        _fail("center_delivered_context must be an object")
    if set(center) != set(CENTER_KEYS):
        _fail("center_delivered_context keys must be exactly " + ", ".join(CENTER_KEYS))
    objectives = center["public_objectives"]
    if not isinstance(objectives, list):
        _fail("public_objectives must be a list")
    for row in objectives:
        if not isinstance(row, dict) or set(row) != set(OBJECTIVE_KEYS):
            _fail("public_objectives entries must be exactly " + ", ".join(OBJECTIVE_KEYS))
        if not isinstance(row["objective_type"], str) or not row["objective_type"]:
            _fail("public objective_type must be a non-empty string")
        if _ENTITY_ID.search(row["objective_type"]):
            _fail("public_objectives must stay scene-level, not entity-level")
        if not _integer(row["count"]) or row["count"] < 0:
            _fail("public objective count must be a non-negative integer")
    stations = center["static_stations"]
    if not isinstance(stations, list):
        _fail("static_stations must be a list")
    seen = set()
    for row in stations:
        if not isinstance(row, dict) or set(row) != set(STATION_KEYS):
            _fail("static_stations entries must be exactly " + ", ".join(STATION_KEYS))
        if not isinstance(row["station_id"], str) or not row["station_id"]:
            _fail("station_id must be a non-empty string")
        if row["station_id"] in seen:
            _fail("static_stations repeats " + row["station_id"])
        seen.add(row["station_id"])
        pose = row["pose_enu"]
        if not isinstance(pose, (list, tuple)) or len(pose) != 3 or not all(_number(v) for v in pose):
            _fail("station pose_enu must be three finite numbers")
        if not isinstance(row["kind"], str) or not row["kind"]:
            _fail("station kind must be a non-empty string")
    reports = center["delivered_reports"]
    if not isinstance(reports, list):
        _fail("delivered_reports must be a list")
    for row in reports:
        if not isinstance(row, dict) or set(row) != set(REPORT_KEYS):
            _fail("delivered_reports entries must be exactly " + ", ".join(REPORT_KEYS))
        if row["source"] != REPORT_SOURCE:
            _fail("delivered_reports may only carry reports delivered over the network")
        if not isinstance(row["reported_by"], str) or not row["reported_by"]:
            _fail("delivered report reported_by must name the delivering agent")
        if row["reported_by"] in ("prior", "authored", "world"):
            _fail("delivered_reports must not be prior/authored world state")
        if row["entity_id"] is not None and not isinstance(row["entity_id"], str):
            _fail("delivered report entity_id must be a string or null")
        if not _number(row["sim_time_s"]) or float(row["sim_time_s"]) < 0:
            _fail("delivered report sim_time_s must be a non-negative finite number")
        if not isinstance(row["values"], dict):
            _fail("delivered report values must be an object")
        if not isinstance(row["kind"], str) or not row["kind"]:
            _fail("delivered report kind must be a non-empty string")


def _validate_edge(edge: dict, record_id: str):
    if not isinstance(edge, dict):
        _fail("edge_telemetry_and_belief must be an object")
    if set(edge) != set(EDGE_KEYS):
        missing = sorted(set(EDGE_KEYS) - set(edge))
        extra = sorted(set(edge) - set(EDGE_KEYS))
        _fail("edge_telemetry_and_belief keys must be exactly " + ", ".join(EDGE_KEYS)
              + (f"; missing {missing}" if missing else "")
              + (f"; extra {extra}" if extra else ""))
    agent_id = edge["agent_id"]
    if not isinstance(agent_id, str) or not agent_id:
        _fail("edge agent_id must be a non-empty string")
    if agent_id not in record_id:
        _fail("edge agent_id must be part of record_id")
    if not isinstance(edge["kind"], str) or not edge["kind"]:
        _fail("edge kind must be a non-empty string")
    if edge["mobility"] not in ("air", "ground", "water", "legged", "wheeled", "foot", "none"):
        _fail("edge mobility is not a physical embodiment mode")
    pose = edge["pose_enu"]
    if not isinstance(pose, (list, tuple)) or len(pose) != 3 or not all(_number(v) for v in pose):
        _fail("edge pose_enu must be three finite numbers")
    if not _number(edge["yaw_rad"]):
        _fail("edge yaw_rad must be finite")
    if not _number(edge["battery_Wh"]) or float(edge["battery_Wh"]) < 0:
        _fail("edge battery_Wh must be a non-negative finite number")
    if not _number(edge["payload_kg"]) or float(edge["payload_kg"]) < 0:
        _fail("edge payload_kg must be a non-negative finite number")
    capabilities = edge["nominal_capabilities"]
    if not isinstance(capabilities, list) or not capabilities:
        _fail("nominal_capabilities must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in capabilities):
        _fail("nominal_capabilities entries must be non-empty strings")
    if len(set(capabilities)) != len(capabilities):
        _fail("nominal_capabilities must not repeat")
    belief = edge["believed_entities"]
    if not isinstance(belief, dict):
        _fail("believed_entities must be an object keyed by entity id")
    for entity_id, entry in belief.items():
        if not isinstance(entity_id, str) or not entity_id:
            _fail("believed_entities keys must be entity ids")
        if not isinstance(entry, dict) or set(entry) != set(BELIEF_ENTRY_KEYS):
            _fail(f"believed entity {entity_id} must carry exactly "
                  + ", ".join(BELIEF_ENTRY_KEYS))
        if not isinstance(entry["kind"], str) or not entry["kind"]:
            _fail(f"believed entity {entity_id} kind must be a non-empty string")
        point = entry["pose_enu"]
        if point is not None and (not isinstance(point, (list, tuple)) or len(point) != 3
                                  or not all(_number(v) for v in point)):
            _fail(f"believed entity {entity_id} pose_enu must be three finite numbers or null")
        if not _number(entry["confidence"]) or not 0.0 <= float(entry["confidence"]) <= 1.0:
            _fail(f"believed entity {entity_id} confidence must lie in [0, 1]")
        if entry["source"] not in BELIEF_SOURCES:
            _fail(f"believed entity {entity_id} source must be one of "
                  + ", ".join(sorted(BELIEF_SOURCES)))
        if not _number(entry["observed_s"]) or float(entry["observed_s"]) < 0:
            _fail(f"believed entity {entity_id} observed_s must be non-negative")


def _validate_intrinsics(intrinsics: dict, path: str):
    if not isinstance(intrinsics, dict) or set(intrinsics) != set(INTRINSICS_KEYS):
        _fail(f"{path} must carry exactly " + ", ".join(INTRINSICS_KEYS))
    if intrinsics["width"] != IMAGE_WIDTH or intrinsics["height"] != IMAGE_HEIGHT:
        _fail(f"{path} must describe a {IMAGE_WIDTH}x{IMAGE_HEIGHT} sensor image")
    for key in ("fx", "fy"):
        if not _number(intrinsics[key]) or float(intrinsics[key]) <= 0:
            _fail(f"{path}.{key} must be a positive focal length")
    for key, limit in (("cx", IMAGE_WIDTH), ("cy", IMAGE_HEIGHT)):
        if not _number(intrinsics[key]) or not 0.0 < float(intrinsics[key]) < float(limit):
            _fail(f"{path}.{key} must lie inside the image")


def _validate_view(view: dict, keys, path: str):
    if not isinstance(view, dict) or set(view) != set(keys):
        _fail(f"{path} must carry exactly " + ", ".join(keys))
    rgb = view["rgb_path"]
    if rgb is not None and (not isinstance(rgb, str) or not rgb):
        _fail(f"{path}.rgb_path must be a path string or null")
    depth = view["depth_path"]
    if depth is not None and (not isinstance(depth, str) or not depth.endswith(".npy")):
        _fail(f"{path}.depth_path must be a single-channel optical-Z .npy sensor array")
    _validate_intrinsics(view["intrinsics"], path + ".intrinsics")
    matrix_key = "extrinsics_w_from_c" if "extrinsics_w_from_c" in keys else "prior_extrinsics_w_from_c"
    _se3_matrix(view[matrix_key], path + "." + matrix_key)
    if list(view["image_grid_thw"]) != expected_grid_thw():
        _fail(f"{path}.image_grid_thw must be {expected_grid_thw()} for a "
              f"{IMAGE_WIDTH}x{IMAGE_HEIGHT} image at patch {PATCH_SIZE}/merge {MERGE_SIZE}")
    if view["post_merger_tokens"] != expected_post_merger_tokens():
        _fail(f"{path}.post_merger_tokens must be {expected_post_merger_tokens()}")


def _validate_views(views: dict):
    if not isinstance(views, dict) or set(views) != set(VIEWS_KEYS):
        _fail("aerial_ground_views must carry exactly uav_view and ground_view")
    _validate_view(views["ground_view"], GROUND_VIEW_KEYS, "aerial_ground_views.ground_view")
    _validate_view(views["uav_view"], UAV_VIEW_KEYS, "aerial_ground_views.uav_view")


def _validate_menu(menu) -> list:
    if not isinstance(menu, list) or not MIN_CANDIDATES <= len(menu) <= MAX_CANDIDATES:
        _fail(f"public_candidate_menu must contain {MIN_CANDIDATES}..{MAX_CANDIDATES} options")
    ids, options = [], []
    for candidate in menu:
        if not isinstance(candidate, dict):
            _fail("candidate entries must be objects")
        if set(candidate) != set(CANDIDATE_KEYS):
            missing = sorted(set(CANDIDATE_KEYS) - set(candidate))
            extra = sorted(set(candidate) - set(CANDIDATE_KEYS))
            _fail("candidate keys must be exactly " + ", ".join(CANDIDATE_KEYS)
                  + (f"; missing {missing}" if missing else "")
                  + (f"; extra {extra}" if extra else ""))
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id:
            _fail("candidate_id must be a non-empty string")
        if candidate_id in ids:
            _fail("duplicate candidate_id " + candidate_id)
        ids.append(candidate_id)
        option_id = candidate["option_id"]
        if option_id not in MACRO_OPTION_IDS:
            _fail(f"option_id {option_id!r} is not one of the named macro options")
        if option_id in options:
            _fail("duplicate macro option in menu: " + option_id)
        options.append(option_id)
        if not _ACTION_ID.match(str(candidate["action_id"])):
            _fail("candidate action_id must look like A###")
        if not _TASK_TYPE.match(str(candidate["task_type"])):
            _fail("candidate task_type must look like T###")
        anchor = candidate["target_anchor"]
        if not isinstance(anchor, str) or not anchor:
            _fail("candidate target_anchor must be a non-empty string")
        if not isinstance(candidate["params"], dict):
            _fail("candidate params must be an object")
    # No candidate may carry an answer hint under any nested key, including the
    # v1-forbidden ``source``/``priority``/``feasible`` fields.
    _scan_leaks(menu, "public_candidate_menu", CANDIDATE_LEAK_KEYS)
    return ids


# --------------------------------------------------------------------------- supervision
def _validate_rewards(rewards: dict, path: str):
    if not isinstance(rewards, dict) or set(rewards) != set(NAMED_REWARD_KEYS):
        _fail(f"{path} must carry exactly the ten named rewards")
    for key in NAMED_REWARD_KEYS:
        if not _number(rewards[key]) or float(rewards[key]) < 0:
            _fail(f"{path}.{key} must be a non-negative finite number")


def _validate_branch(branches: dict, ids: list):
    if not isinstance(branches, dict) or set(branches) != set(ids):
        _fail("branch_evaluations must score every candidate exactly once")
    for candidate_id in ids:
        branch = branches[candidate_id]
        if not isinstance(branch, dict) or set(branch) != set(BRANCH_KEYS):
            _fail(f"branch {candidate_id} must carry exactly " + ", ".join(BRANCH_KEYS))
        slices = branch["slices"]
        if not isinstance(slices, dict) or set(slices) != set(SLICE_KEYS):
            _fail(f"branch {candidate_id} must carry slices " + ", ".join(SLICE_KEYS))
        previous = None
        for name in SLICE_KEYS:
            row = slices[name]
            if not isinstance(row, dict) or set(row) != set(SLICE_ROW_KEYS):
                _fail(f"branch {candidate_id}.{name} must carry exactly "
                      + ", ".join(SLICE_ROW_KEYS))
            if not _number(row["horizon_s"]) or abs(float(row["horizon_s"]) - SLICE_HORIZONS[name]) > 1e-9:
                _fail(f"branch {candidate_id}.{name}.horizon_s must be {SLICE_HORIZONS[name]}")
            _validate_rewards(row["named_rewards"], f"branch {candidate_id}.{name}.named_rewards")
            _validate_rewards(row["cumulative_named_rewards"],
                              f"branch {candidate_id}.{name}.cumulative_named_rewards")
            for key in NAMED_REWARD_KEYS:
                increment = float(row["named_rewards"][key])
                total = float(row["cumulative_named_rewards"][key])
                if total + 1e-9 < increment:
                    _fail(f"branch {candidate_id}.{name} cumulative {key} is below its slice")
                before = float(previous[key]) if previous is not None else 0.0
                if abs(total - before - increment) > 2e-6:
                    _fail(f"branch {candidate_id}.{name} increment {key} does not match cumulative difference")
            previous = row["cumulative_named_rewards"]
            milestones = row["milestones"]
            if not isinstance(milestones, list):
                _fail(f"branch {candidate_id}.{name}.milestones must be a list")
            if len(set(milestones)) != len(milestones):
                _fail(f"branch {candidate_id}.{name}.milestones must be de-duplicated")
            first_times = row["first_achieved_s"]
            if not isinstance(first_times, dict) or set(first_times) != set(milestones):
                _fail(f"branch {candidate_id}.{name} must timestamp every milestone")
            if any(not _number(t) or t < 0 or t > row["horizon_s"] + 1e-6 for t in first_times.values()):
                _fail(f"branch {candidate_id}.{name} milestone time is outside its prefix")
            for item in milestones:
                if not isinstance(item, str) or "::" not in item:
                    _fail(f"branch {candidate_id}.{name}.milestones entries must be "
                          "'entity_id::milestone'")
        expected = q_value(slices[PRIMARY_SLICE]["cumulative_named_rewards"])
        if not _number(branch["q_value_120s"]) or abs(float(branch["q_value_120s"]) - expected) > 1e-9:
            _fail(f"branch {candidate_id}.q_value_120s is not the weighted 120 s utility")


def _validate_teacher(teacher: dict, branches: dict, ids: list):
    if not isinstance(teacher, dict) or set(teacher) != set(TEACHER_KEYS):
        _fail("soft_q_teacher_120s must carry exactly " + ", ".join(TEACHER_KEYS))
    tau = teacher["tau"]
    if not _number(tau) or float(tau) <= 0:
        _fail("soft_q_teacher_120s.tau must be positive")
    logits = teacher["logits"]
    if not isinstance(logits, dict) or set(logits) != set(ids):
        _fail("soft_q_teacher_120s.logits must cover every candidate")
    for candidate_id in ids:
        if not _number(logits[candidate_id]):
            _fail(f"soft teacher logit {candidate_id} must be finite")
        if abs(float(logits[candidate_id]) - float(branches[candidate_id]["q_value_120s"])) > 1e-9:
            _fail(f"soft teacher logit {candidate_id} must equal q_value_120s")
    expected = soft_q_distribution({key: float(logits[key]) for key in ids}, float(tau))
    probabilities = teacher["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(ids):
        _fail("soft_q_teacher_120s.probabilities must cover every candidate")
    for candidate_id in ids:
        if not _number(probabilities[candidate_id]):
            _fail(f"soft teacher probability {candidate_id} must be finite")
        if abs(float(probabilities[candidate_id]) - expected[candidate_id]) > 1e-9:
            _fail(f"soft teacher probability {candidate_id} is not Softmax(Q/tau)")
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-9:
        _fail("soft teacher probabilities must sum to one")
    optimal = teacher["optimal_candidate_ids"]
    if not isinstance(optimal, list) or not optimal:
        _fail("optimal_candidate_ids must be a non-empty list")
    if len(set(optimal)) != len(optimal):
        _fail("optimal_candidate_ids must not repeat")
    if any(candidate_id not in ids for candidate_id in optimal):
        _fail("optimal_candidate_ids must reference existing candidates")
    best = max(float(branches[candidate_id]["q_value_120s"]) for candidate_id in ids)
    argmax = sorted(candidate_id for candidate_id in ids
                    if abs(float(branches[candidate_id]["q_value_120s"]) - best) <= 1e-9)
    if sorted(optimal) != argmax:
        _fail("optimal_candidate_ids must be exactly the arg-max candidates")


def _validate_pose(pose: dict):
    if not isinstance(pose, dict) or set(pose) != set(POSE_KEYS):
        _fail("true_pose_e_from_i must carry exactly " + ", ".join(POSE_KEYS))
    _se3_matrix(pose["T_e_from_i"], "true_pose_e_from_i.T_e_from_i")
    xi = pose["xi"]
    if not isinstance(xi, (list, tuple)) or len(xi) != 6 or not all(_number(v) for v in xi):
        _fail("true_pose_e_from_i.xi must be a six-vector of finite numbers")
    if pose["convention"] != POSE_CONVENTION:
        _fail("true_pose_e_from_i.convention must be " + POSE_CONVENTION)


def _validate_geometry(geometry: dict):
    if not isinstance(geometry, dict) or set(geometry) != set(GEOMETRY_KEYS):
        _fail("geometry_paths must carry exactly " + ", ".join(GEOMETRY_KEYS))
    for key in ("uav_geom_npz", "ground_geom_npz"):
        value = geometry[key]
        if value is not None and (not isinstance(value, str) or not value):
            _fail(f"geometry_paths.{key} must be a path string or null")
    summary = geometry["covisibility_summary"]
    if not isinstance(summary, dict) or set(summary) != set(COVISIBILITY_KEYS):
        _fail("covisibility_summary must carry exactly " + ", ".join(COVISIBILITY_KEYS))
    for key in ("common_anchor_count", "uav_anchor_count", "ground_anchor_count"):
        if not _integer(summary[key]) or summary[key] < 0:
            _fail(f"covisibility_summary.{key} must be a non-negative integer")
    for key in ("overlap_ratio", "mean_depth_ground_m", "mean_depth_uav_m"):
        if not _number(summary[key]) or float(summary[key]) < 0:
            _fail(f"covisibility_summary.{key} must be a non-negative finite number")
    if float(summary["overlap_ratio"]) > 1.0 + 1e-9:
        _fail("covisibility_summary.overlap_ratio must lie in [0, 1]")
    if summary["common_anchor_count"] > min(summary["uav_anchor_count"], summary["ground_anchor_count"]):
        _fail("covisibility_summary common anchors cannot exceed either view")
    expected = (float(summary["common_anchor_count"])
                / max(1, summary["ground_anchor_count"]))
    if abs(float(summary["overlap_ratio"]) - expected) > 1e-9:
        _fail("covisibility_summary.overlap_ratio must equal common / ground anchors")


# --------------------------------------------------------------------------- public API
def validate_phase1_v2_record(record: dict) -> dict:
    """Validate one separated v2 record and return it unchanged."""
    if not isinstance(record, dict):
        _fail("record must be an object")
    if set(record) != {"schema", "metadata", "model_input", "supervision_and_eval"}:
        _fail("record must carry exactly schema, metadata, model_input and supervision_and_eval")
    if record["schema"] != SCHEMA_ID_V2:
        _fail("schema must be " + SCHEMA_ID_V2)

    _validate_metadata(record)

    model_input = record["model_input"]
    if not isinstance(model_input, dict) or set(model_input) != set(MODEL_INPUT_KEYS):
        _fail("model_input keys must be exactly " + ", ".join(MODEL_INPUT_KEYS))
    _scan_keys(model_input, "model_input", MODEL_INPUT_BANNED_KEYS, OVERRIDE_KEY_TOKENS)
    _scan_leaks({key: value for key, value in model_input.items()
                 if key != "center_delivered_context"}, "model_input")
    # Delivered reports legitimately carry a network provenance row, so the
    # centre layer is scanned for truth/override leaks but not for ``source``.
    _scan_keys(model_input["center_delivered_context"],
               "model_input.center_delivered_context", frozenset(), OVERRIDE_KEY_TOKENS)

    menu = model_input["public_candidate_menu"]
    ids = _validate_menu(menu)
    _validate_brief(model_input["mission_brief"], menu)
    _validate_center(model_input["center_delivered_context"])
    _validate_edge(model_input["edge_telemetry_and_belief"], record["metadata"]["record_id"])
    _validate_views(model_input["aerial_ground_views"])

    supervision = record["supervision_and_eval"]
    if not isinstance(supervision, dict) or set(supervision) != set(SUPERVISION_KEYS):
        _fail("supervision_and_eval keys must be exactly " + ", ".join(SUPERVISION_KEYS))
    _validate_branch(supervision["branch_evaluations"], ids)
    _validate_teacher(supervision["soft_q_teacher_120s"],
                      supervision["branch_evaluations"], ids)
    _validate_pose(supervision["true_pose_e_from_i"])
    _validate_geometry(supervision["geometry_paths"])
    return record


def optimal_candidate_ids(record: dict) -> list:
    return list(record["supervision_and_eval"]["soft_q_teacher_120s"]["optimal_candidate_ids"])


def canonical_ego_input(record: dict) -> dict:
    """Complete observable ego input, identifying the image by bytes, never pathname."""
    from copy import deepcopy
    ego = deepcopy(record["model_input"])
    ego["aerial_ground_views"].pop("uav_view")
    view = ego["aerial_ground_views"]["ground_view"]
    path = view.pop("rgb_path")
    view["rgb_sha256"] = sha256_file(path) if path is not None else None
    depth = view.pop("depth_path")
    view["depth_sha256"] = sha256_file(depth) if depth is not None else None
    return ego


def assert_matched_pair_noninterference(rec_plus: dict, rec_minus: dict) -> None:
    """Assert the matched-pair contract ① on two branch records.

    Raises ``AssertionError`` with the first violated clause.  The ego input must
    be byte-identical while the optimal macro-action sets must be disjoint: that
    disjointness is exactly what makes any ground-only policy's hard accuracy
    upper bound <= 50% on the pair.
    """
    validate_phase1_v2_record(rec_plus)
    validate_phase1_v2_record(rec_minus)
    meta_plus, meta_minus = rec_plus["metadata"], rec_minus["metadata"]

    assert meta_plus["sample_kind"] == "challenge_matched" \
        and meta_minus["sample_kind"] == "challenge_matched", \
        "both records must be challenge_matched samples"
    assert meta_plus["pair_group_id"] is not None \
        and meta_plus["pair_group_id"] == meta_minus["pair_group_id"], \
        "the two records must belong to the same pair_group_id"
    assert meta_plus["record_id"] != meta_minus["record_id"], \
        "the two records must be distinct decision sections"
    for key in ("scene_id", "seed", "split", "physical_regime", "sim_time_s"):
        assert meta_plus[key] == meta_minus[key], \
            f"matched pair must agree on metadata.{key}"

    input_plus, input_minus = rec_plus["model_input"], rec_minus["model_input"]

    assert input_plus["edge_telemetry_and_belief"] == input_minus["edge_telemetry_and_belief"], \
        "matched pair must deliver the identical edge telemetry and belief"
    assert input_plus["public_candidate_menu"] == input_minus["public_candidate_menu"], \
        "matched pair must offer the identical public candidate menu"
    assert input_plus["mission_brief"] == input_minus["mission_brief"], \
        "matched pair must share the static mission brief"
    assert input_plus["center_delivered_context"] == input_minus["center_delivered_context"], \
        "matched pair must deliver the identical centre context"

    views_plus, views_minus = input_plus["aerial_ground_views"], input_minus["aerial_ground_views"]
    ground_plus, ground_minus = views_plus["ground_view"], views_minus["ground_view"]
    # The two worlds may materialise their RGB under different directories, so
    # the ground view is compared on everything but the path; the bytes are then
    # compared by sha256 below.
    assert {key: value for key, value in ground_plus.items() if key not in ("rgb_path", "depth_path")} == \
        {key: value for key, value in ground_minus.items() if key not in ("rgb_path", "depth_path")}, \
        "matched pair must expose the identical ground view"
    uav_plus, uav_minus = views_plus["uav_view"], views_minus["uav_view"]
    for key in ("intrinsics", "image_grid_thw", "post_merger_tokens"):
        assert uav_plus[key] == uav_minus[key], \
            f"matched pair must agree on the aerial view {key}"
    assert uav_plus["prior_extrinsics_w_from_c"] == uav_minus["prior_extrinsics_w_from_c"], \
        "matched pair must share the same noisy aerial pose prior"

    # Ground RGB bytes must be identical whenever they are on disk.  A half
    # materialised pair is a bug, not a pass.
    ground_path_plus, ground_path_minus = ground_plus["rgb_path"], ground_minus["rgb_path"]
    if ground_path_plus is not None or ground_path_minus is not None:
        from pathlib import Path

        assert ground_path_plus is not None and ground_path_minus is not None, \
            "matched pair must materialise the ground view in both worlds"
        path_plus, path_minus = Path(ground_path_plus), Path(ground_path_minus)
        assert path_plus.is_file() and path_minus.is_file(), \
            "declared ground RGB paths must exist on disk"
        digest_plus, digest_minus = sha256_file(path_plus), sha256_file(path_minus)
        assert digest_plus == digest_minus, \
            "ground RGB bytes differ across the matched pair (sha256 mismatch)"

    assert canonical_json(canonical_ego_input(rec_plus)) == canonical_json(canonical_ego_input(rec_minus)), "complete serialized ego input differs across twins"
    optimal_plus = set(optimal_candidate_ids(rec_plus))
    optimal_minus = set(optimal_candidate_ids(rec_minus))
    assert optimal_plus.isdisjoint(optimal_minus), (
        "optimal macro-action sets must be disjoint, got "
        f"{sorted(optimal_plus)} vs {sorted(optimal_minus)}")
    assert optimal_plus and optimal_minus, "each branch must declare an optimal option"


def pair_summary(rec_plus: dict, rec_minus: dict) -> dict:
    """Compact, JSON-safe summary of one matched pair (reporting only)."""
    return {
        "pair_group_id": rec_plus["metadata"]["pair_group_id"],
        "scene_id": rec_plus["metadata"]["scene_id"],
        "seed": rec_plus["metadata"]["seed"],
        "split": rec_plus["metadata"]["split"],
        "physical_regime": rec_plus["metadata"]["physical_regime"],
        "ego_input_sha256": sha256_text(canonical_json(canonical_ego_input(rec_plus))),
        "menu_sha256": sha256_text(canonical_json(rec_plus["model_input"]["public_candidate_menu"])),
        "optimal_plus": optimal_candidate_ids(rec_plus),
        "optimal_minus": optimal_candidate_ids(rec_minus),
        "q_plus": {key: round(float(value), 6) for key, value
                   in rec_plus["supervision_and_eval"]["soft_q_teacher_120s"]["logits"].items()},
        "q_minus": {key: round(float(value), 6) for key, value
                    in rec_minus["supervision_and_eval"]["soft_q_teacher_120s"]["logits"].items()},
    }
