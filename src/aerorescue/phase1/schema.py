"""Phase-1 decision-record contract (v2) with hard anti-leakage checks.

A record is one *decision section*.  It is split into three isolated layers:

``metadata``
    Dataset bookkeeping (``record_id``/``scene_id``/``seed``/``split``/time).
    Never handed to the model.
``model_input``
    Exactly what the edge agent may see: intent text, delivered centre context,
    the centre memory envelope, local telemetry/belief, camera metadata and the
    candidate list.  Candidate entries carry *no* ``source``/``feasible`` hint.
``supervision``
    Offline labels only: real counterfactual rollout utilities, the best and
    epsilon-acceptable sets, paired VoI, regime/pose prior and the ISCC budget
    split.  Never part of ``model_input``.

``validate_decision_record`` fails loudly instead of coercing, so a collector
bug cannot silently ship a leaked or mislabelled dataset.
"""
from __future__ import annotations

import json
import math
import re

SCHEMA_ID = "aerorescue.phase1.decision_record/v2"

SPLITS = frozenset({"train", "val", "test"})
REGIMES = frozenset({"A", "B", "C"})

REQUIRED_KEYS = ("record_id", "schema", "metadata", "model_input", "supervision")
METADATA_KEYS = ("record_id", "scene_id", "seed", "split", "sim_time_s", "agent_id", "source")
MODEL_INPUT_KEYS = ("intent_text", "center_context", "center_memory", "edge_state",
                    "camera_views", "candidates")
EDGE_KEYS = ("agent_id", "kind", "mobility", "pose_enu", "battery_Wh", "payload_kg",
             "capabilities", "local_belief_summary")
CANDIDATE_KEYS = ("candidate_id", "skill_name", "task_id", "task_rev", "task_type",
                  "method_seq", "step_index", "purpose", "method_scope", "action_id",
                  "target_id", "params")
PURPOSES = frozenset({"approach", "method"})
METHOD_SCOPES = frozenset({"main", "recovery"})
MEMORY_KEYS = ("schema", "mode", "budget_bytes", "serialized_bytes", "source_cut_s",
               "representation", "slots")
MEMORY_SCHEMA = "aerorescue.memory_envelope/v1"
UTILITY_KEYS = ("delta_total", "delta_milestone", "delta_potential", "delta_cursor",
                "cost_penalty", "fail_penalty", "action_succeeded", "rollout_status")
VOI_KEYS = ("utility_local_only", "utility_with_center_memory", "vo_memory_vs_local",
            "ablation_2x2")
ABLATION_KEYS = ("B00", "B10", "B01", "B11")
SUPERVISION_KEYS = ("candidate_utilities", "best_candidate_id", "acceptable_set",
                    "reference_candidate_id", "reference_source", "paired_voi_eval",
                    "regime", "pose_prior", "budget_plan", "delta_weights")
ACCEPT_EPSILON = 0.02

# Any key carrying simulation-only truth is forbidden in model inputs.
FORBIDDEN_KEYS = frozenset({"truth_for_rendering_only", "world.entities", "world", "world_entities",
                            "truth", "provenance"})
# Labels or hints that must never appear inside a candidate (or anywhere in
# ``model_input`` under a name that reveals the reference answer).
LEAK_KEYS = frozenset({"source", "is_reference", "oracle", "utility", "feasible",
                       "best_candidate", "label", "is_best", "reference"})
MODEL_INPUT_BANNED_KEYS = frozenset({
    "seed", "split", "record_id", "episode_index", "supervision", "candidate_utilities",
    "best_candidate_id", "reference_candidate_id", "reference_source", "is_reference",
    "oracle", "utility", "feasible", "label", "is_best", "reference",
    "execution_overrides", "effective_capabilities"})

_RECORD_ID = re.compile(r"^(?P<scene>[A-Za-z0-9_]+)_seed(?P<seed>\d+)_t(?P<ms>\d+)_(?P<agent>.+)$")
_ACTION_ID = re.compile(r"^A\d{3}$")
_ACTION_IN_TEXT = re.compile(r"\bA\d{3}\b")
_TASK_TYPE = re.compile(r"^T\d{3}$")


def make_record_id(scene_id: str, seed: int, sim_time_s: float, agent_id: str) -> str:
    return f"{scene_id}_seed{int(seed)}_t{int(round(float(sim_time_s) * 1000))}_{agent_id}"


def _fail(message: str):
    raise ValueError("invalid decision record: " + message)


def _forbidden(key) -> bool:
    return isinstance(key, str) and (key.lower() in FORBIDDEN_KEYS or "truth" in key.lower())


def _leaky(key) -> bool:
    return isinstance(key, str) and key.lower() in LEAK_KEYS


def _scan_forbidden(value, path: str):
    if isinstance(value, dict):
        for key, item in value.items():
            if _forbidden(key):
                _fail(f"forbidden truth field at {path}.{key}")
            _scan_forbidden(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden(item, f"{path}[{index}]")


def _scan_candidate_hints(value, path):
    if isinstance(value, dict):
        for key, item in value.items():
            if _leaky(key):
                _fail("candidate must not hide answer hints at " + path + "." + key)
            _scan_candidate_hints(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_candidate_hints(item, f"{path}[{index}]")


def _scan_model_input_keys(value, path: str):
    """No dataset bookkeeping may hide inside model_input.

    Delivered provenance markers such as ``station.source`` are legitimate.
    Bookkeeping and answer labels are banned recursively; candidates additionally
    reject provenance keys because those can identify their generation source.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and (key.lower() in MODEL_INPUT_BANNED_KEYS
                    or key.lower() in LEAK_KEYS - {"source"}):
                _fail(f"model_input must not expose {key} (at {path}.{key})")
            _scan_model_input_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_model_input_keys(item, f"{path}[{index}]")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_metadata(record: dict):
    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        _fail("metadata must be an object")
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        _fail("metadata missing keys: " + ", ".join(missing))
    if metadata["record_id"] != record["record_id"]:
        _fail("metadata.record_id must equal the top-level record_id")
    match = _RECORD_ID.match(str(record["record_id"]))
    if not match:
        _fail("record_id must look like {scene_id}_seed{seed}_t{sim_time_ms}_{agent_id}")
    if match.group("scene") != metadata["scene_id"]:
        _fail("record_id scene prefix does not match metadata.scene_id")
    if int(match.group("seed")) != metadata["seed"]:
        _fail("record_id seed does not match metadata.seed")
    if not isinstance(metadata["scene_id"], str) or not metadata["scene_id"]:
        _fail("metadata.scene_id must be a non-empty string")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        _fail("metadata.seed must be an integer")
    if metadata["split"] not in SPLITS:
        _fail("metadata.split must be one of train/val/test")
    if not _number(metadata["sim_time_s"]) or metadata["sim_time_s"] < 0:
        _fail("metadata.sim_time_s must be a non-negative finite number")
    if not isinstance(metadata["agent_id"], str) or not metadata["agent_id"]:
        _fail("metadata.agent_id must be a non-empty string")
    if match.group("agent") != metadata["agent_id"]:
        _fail("record_id agent suffix does not match metadata.agent_id")
    if int(match.group("ms")) != int(round(metadata["sim_time_s"] * 1000)):
        _fail("record_id time does not match metadata.sim_time_s")


def _validate_intent(intent: str, candidates: list, best_id: str):
    if not isinstance(intent, str) or not intent.strip():
        _fail("model_input.intent_text must be a non-empty string")
    leaked = _ACTION_IN_TEXT.findall(intent)
    if leaked:
        _fail("intent_text leaks atomic action ids: " + ", ".join(sorted(set(leaked))))
    for candidate in candidates:
        skill = candidate.get("skill_name")
        if isinstance(skill, str) and skill and skill in intent:
            _fail("intent_text leaks candidate skill name: " + skill)
        if candidate.get("candidate_id") == best_id:
            target = str(candidate.get("target_id", ""))
            if target and target in intent:
                _fail("intent_text leaks the best candidate target: " + target)


def _validate_candidates(candidates) -> list:
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
        _fail("candidates must contain 2..8 entries")
    ids = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            _fail("candidate must be an object")
        _scan_candidate_hints(candidate, "candidate")
        keys = set(candidate)
        leaked = sorted(key for key in keys if _leaky(key))
        if leaked:
            _fail("candidate must not carry answer hints: " + ", ".join(leaked))
        if keys != set(CANDIDATE_KEYS):
            missing = sorted(set(CANDIDATE_KEYS) - keys)
            extra = sorted(keys - set(CANDIDATE_KEYS))
            _fail("candidate keys must be exactly " + ", ".join(CANDIDATE_KEYS)
                  + (f"; missing {missing}" if missing else "")
                  + (f"; extra {extra}" if extra else ""))
        cid = candidate["candidate_id"]
        if not isinstance(cid, str) or not cid:
            _fail("candidate_id must be a non-empty string")
        if cid in ids:
            _fail("duplicate candidate_id " + cid)
        ids.append(cid)
        if not isinstance(candidate["skill_name"], str) or not candidate["skill_name"]:
            _fail("candidate skill_name must be a non-empty string")
        if candidate["task_id"] is not None and not isinstance(candidate["task_id"], str):
            _fail("candidate task_id must be a string or null")
        if candidate["task_rev"] is not None and (
                isinstance(candidate["task_rev"], bool) or not isinstance(candidate["task_rev"], int)):
            _fail("candidate task_rev must be an integer or null")
        if not _TASK_TYPE.match(str(candidate["task_type"])):
            _fail("candidate task_type must look like T###")
        if candidate["method_seq"] is not None and not isinstance(candidate["method_seq"], str):
            _fail("candidate method_seq must be a string or null")
        if isinstance(candidate["step_index"], bool) or not isinstance(candidate["step_index"], int):
            _fail("candidate step_index must be an integer")
        if candidate["purpose"] not in PURPOSES:
            _fail("candidate purpose must be approach or method")
        if candidate["method_scope"] not in METHOD_SCOPES:
            _fail("candidate method_scope must be main or recovery")
        if not _ACTION_ID.match(str(candidate["action_id"])):
            _fail("candidate action_id must look like A###")
        if not isinstance(candidate["target_id"], str) or not candidate["target_id"]:
            _fail("candidate target_id must be a non-empty string")
        if not isinstance(candidate["params"], dict):
            _fail("candidate params must be an object")
    return ids


def _validate_center(center: dict):
    for key, kind in (("known_tasks", list), ("reported_observations", list),
                      ("stations", list), ("mission_task_catalog", list)):
        if not isinstance(center.get(key), kind):
            _fail("center_context." + key + " must be a list")
    tasks = center["known_tasks"]
    for task in tasks:
        for key in ("task_id", "task_type", "target_id", "status", "assigned_to", "priority"):
            if key not in task:
                _fail("center_context.known_tasks entry missing " + key)
        if not isinstance(task["task_id"], str) or not task["task_id"]:
            _fail("known task task_id must be a non-empty string")
        if not _TASK_TYPE.match(str(task["task_type"])):
            _fail("known task task_type must look like T###")
    if center.get("station_source") != "coordinator_belief_or_initial_static":
        _fail("center_context.station_source must be coordinator_belief_or_initial_static")
    for station in center["stations"]:
        if not isinstance(station, dict) or "entity_id" not in station:
            _fail("center_context.stations entries must carry entity_id")
        if station.get("source") != "coordinator_belief_or_initial_static":
            _fail("station rows may only come from coordinator belief or initial static config")
    if not isinstance(center.get("public_brief_zh", ""), str):
        _fail("center_context.public_brief_zh must be a string")


def _validate_memory(memory: dict):
    if not isinstance(memory, dict):
        _fail("center_memory must be an object")
    missing = [key for key in MEMORY_KEYS if key not in memory]
    if missing:
        _fail("center_memory missing keys: " + ", ".join(missing))
    if memory["schema"] != MEMORY_SCHEMA:
        _fail("center_memory.schema must be " + MEMORY_SCHEMA)
    if not isinstance(memory["mode"], str) or not memory["mode"]:
        _fail("center_memory.mode must be a non-empty string")
    if isinstance(memory["budget_bytes"], bool) or not isinstance(memory["budget_bytes"], int) \
            or memory["budget_bytes"] <= 0:
        _fail("center_memory.budget_bytes must be a positive integer")
    if isinstance(memory["serialized_bytes"], bool) or not isinstance(memory["serialized_bytes"], int):
        _fail("center_memory.serialized_bytes must be an integer")
    if memory["serialized_bytes"] <= 0 or memory["serialized_bytes"] > memory["budget_bytes"]:
        _fail("center_memory.serialized_bytes must fit inside budget_bytes")
    if not _number(memory["source_cut_s"]) or memory["source_cut_s"] < 0:
        _fail("center_memory.source_cut_s must be finite and non-negative")
    if not isinstance(memory["representation"], str) or not memory["representation"]:
        _fail("center_memory.representation must be a non-empty string")
    if not isinstance(memory["slots"], list):
        _fail("center_memory.slots must be a list (empty before delivery)")
    actual_bytes = len(json.dumps(memory, ensure_ascii=False).encode("utf-8"))
    if actual_bytes != memory["serialized_bytes"]:
        _fail("center_memory.serialized_bytes must count the entire envelope")
    for slot in memory["slots"]:
        for key in ("slot_id", "kind", "key", "value", "source", "t", "bytes"):
            if key not in slot:
                _fail("center_memory slot missing " + key)
        if slot["source"] != "coordinator_belief_or_initial_static":
            _fail("center_memory slots may only carry delivered/static centre state")
        actual = len(json.dumps(slot["value"], ensure_ascii=False, default=str).encode("utf-8"))
        if slot["bytes"] != actual:
            _fail("center_memory slot bytes must count the serialized value")
        if not _number(slot["t"]) or slot["t"] > memory["source_cut_s"]:
            _fail("memory slot must not postdate the source cut")


def _validate_supervision(supervision: dict, ids: list):
    if not isinstance(supervision, dict):
        _fail("supervision must be an object")
    missing = [key for key in SUPERVISION_KEYS if key not in supervision]
    if missing:
        _fail("supervision missing keys: " + ", ".join(missing))
    weights = supervision["delta_weights"]
    if not isinstance(weights, dict):
        _fail("delta_weights must be an object")
    for key in ("milestone", "potential", "cursor", "cost", "fail"):
        if not _number(weights.get(key)):
            _fail("delta_weights." + key + " must be a finite number")
    if float(weights["cursor"]) != 0.0:
        _fail("delta_cursor weight must be 0.0 (diagnostic only, never scored)")

    utilities = supervision["candidate_utilities"]
    if not isinstance(utilities, dict) or set(utilities) != set(ids):
        _fail("candidate_utilities must score every candidate exactly once")
    totals = {}
    for cid, row in utilities.items():
        if not isinstance(row, dict):
            _fail("candidate utility rows must be objects")
        missing = [key for key in UTILITY_KEYS if key not in row]
        if missing:
            _fail(f"candidate {cid} utility missing: " + ", ".join(missing))
        for key in ("delta_total", "delta_milestone", "delta_potential", "delta_cursor",
                    "cost_penalty", "fail_penalty"):
            if not _number(row[key]):
                _fail(f"candidate {cid} {key} must be a finite number")
        if not isinstance(row["action_succeeded"], bool):
            _fail(f"candidate {cid} action_succeeded must be a boolean")
        if not isinstance(row["rollout_status"], str) or not row["rollout_status"]:
            _fail(f"candidate {cid} rollout_status must be a non-empty string")
        if row["cost_penalty"] < 0:
            _fail(f"candidate {cid} cost_penalty must be non-negative")
        if row["fail_penalty"] > 0:
            _fail(f"candidate {cid} fail_penalty must be non-positive")
        expected = (weights["milestone"] * row["delta_milestone"]
                    + weights["potential"] * row["delta_potential"]
                    + weights["cursor"] * row["delta_cursor"]
                    - weights["cost"] * row["cost_penalty"]
                    + weights["fail"] * row["fail_penalty"])
        if abs(expected - row["delta_total"]) > 1e-6:
            _fail(f"candidate {cid} delta_total is not the declared weighted sum")
        totals[cid] = row["delta_total"]

    best = supervision["best_candidate_id"]
    if best not in ids:
        _fail("best_candidate_id must reference an existing candidate")
    best_total = max(totals.values())
    if abs(totals[best] - best_total) > 1e-9:
        _fail("best_candidate_id is not the highest delta_total candidate")

    acceptable = supervision["acceptable_set"]
    if not isinstance(acceptable, list) or not acceptable:
        _fail("acceptable_set must be a non-empty list")
    if len(set(acceptable)) != len(acceptable):
        _fail("acceptable_set must not repeat candidates")
    if any(cid not in ids for cid in acceptable):
        _fail("acceptable_set must reference existing candidates")
    if best not in acceptable:
        _fail("acceptable_set must contain best_candidate_id")
    for cid in ids:
        within = best_total - totals[cid] <= ACCEPT_EPSILON + 1e-9
        if within and cid not in acceptable:
            _fail("acceptable_set must contain every candidate within epsilon of the best")
        if not within and cid in acceptable:
            _fail("acceptable_set contains a candidate outside the epsilon band")

    reference = supervision["reference_candidate_id"]
    if reference is not None and reference not in ids:
        _fail("reference_candidate_id must reference an existing candidate")

    if supervision["regime"] not in REGIMES:
        _fail("supervision.regime must be A, B or C")
    prior = supervision["pose_prior"]
    if not isinstance(prior, dict) or not _number(prior.get("sigma_theta_deg")) \
            or not _number(prior.get("sigma_t_m")):
        _fail("pose_prior must carry finite sigma_theta_deg and sigma_t_m")
    plan = supervision["budget_plan"]
    if not isinstance(plan, dict) or not isinstance(plan.get("status"), str):
        _fail("budget_plan must be an object with a status string")

    voi = supervision["paired_voi_eval"]
    if not isinstance(voi, dict):
        _fail("paired_voi_eval must be an object")
    missing = [key for key in VOI_KEYS if key not in voi]
    if missing:
        _fail("paired_voi_eval missing: " + ", ".join(missing))
    for key in ("utility_local_only", "utility_with_center_memory", "vo_memory_vs_local"):
        if not _number(voi[key]):
            _fail("paired_voi_eval." + key + " must be finite")
    if abs((voi["utility_with_center_memory"] - voi["utility_local_only"])
           - voi["vo_memory_vs_local"]) > 1e-9:
        _fail("vo_memory_vs_local must equal with_memory - local_only")
    ablation = voi["ablation_2x2"]
    if not isinstance(ablation, dict) or set(ablation) != set(ABLATION_KEYS):
        _fail("ablation_2x2 must define exactly B00, B10, B01, B11")
    for key, value in ablation.items():
        if not _number(value):
            _fail("ablation_2x2." + key + " must be finite")
    if voi.get("bit_convention") != "first=uplink_selection,second=downlink_memory":
        _fail("VoI bits must be uplink selection then downlink memory")
    if (voi["utility_local_only"] != ablation["B00"]
            or voi["utility_with_center_memory"] != ablation["B01"]):
        _fail("memory VoI must compare B01 against B00 at fixed baseline uplink")
    for field in ("ablation_rollouts", "ablation_choices", "channel_treatments"):
        if not isinstance(voi.get(field), dict) or set(voi[field]) != set(ABLATION_KEYS):
            _fail("paired_voi_eval." + field + " must define every band")
    for band in ABLATION_KEYS:
        if voi["ablation_choices"][band] not in ids:
            _fail("ablation choice must reference a candidate")
        row = voi["ablation_rollouts"][band]
        if not isinstance(row, dict) or any(key not in row for key in UTILITY_KEYS):
            _fail("each ablation must carry its actual rollout outcome")
        if any(not _number(row[key]) for key in (
                "delta_total", "delta_milestone", "delta_potential", "delta_cursor",
                "cost_penalty", "fail_penalty")):
            _fail("ablation rollout utility fields must be finite")
        total = (weights["milestone"] * row["delta_milestone"]
                 + weights["potential"] * row["delta_potential"]
                 + weights["cursor"] * row["delta_cursor"]
                 - weights["cost"] * row["cost_penalty"]
                 + weights["fail"] * row["fail_penalty"])
        if abs(total - ablation[band]) > 1e-6 or row["delta_total"] != ablation[band]:
            _fail("ablation utility must equal its native rollout weighted sum")
        treatment = voi["channel_treatments"][band]
        if (treatment.get("task_guided_uplink") is not (band[1] == "1")
                or treatment.get("center_memory_enabled") is not (band[2] == "1")):
            _fail("ablation channel treatment contradicts its bits")
        if band[2] == "0" and (treatment.get("downlink_bytes") != 0
                               or treatment.get("delivered_memory_slots") != 0):
            _fail("downlink-off bands must not consume new external memory")


def validate_decision_record(record: dict) -> dict:
    """Validate one v2 decision record and return it unchanged."""
    if not isinstance(record, dict):
        _fail("record must be an object")
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        _fail("missing keys: " + ", ".join(missing))
    if record["schema"] != SCHEMA_ID:
        _fail("schema must be " + SCHEMA_ID)
    if not isinstance(record["record_id"], str) or not record["record_id"]:
        _fail("record_id must be a non-empty string")

    _validate_metadata(record)

    model_input = record["model_input"]
    if not isinstance(model_input, dict):
        _fail("model_input must be an object")
    if set(model_input) != set(MODEL_INPUT_KEYS):
        _fail("model_input keys must be exactly " + ", ".join(MODEL_INPUT_KEYS))
    _scan_model_input_keys(model_input, "model_input")
    _scan_forbidden(model_input, "model_input")

    edge = model_input["edge_state"]
    if not isinstance(edge, dict):
        _fail("edge_state must be an object")
    missing_edge = [key for key in EDGE_KEYS if key not in edge]
    if missing_edge:
        _fail("edge_state missing keys: " + ", ".join(missing_edge))
    if edge["agent_id"] != record["metadata"]["agent_id"]:
        _fail("edge_state.agent_id must match metadata.agent_id")
    if not isinstance(edge["pose_enu"], (list, tuple)) or len(edge["pose_enu"]) != 3:
        _fail("edge_state.pose_enu must be a three-element ENU position")
    if not all(_number(value) for value in edge["pose_enu"]):
        _fail("edge_state.pose_enu values must be finite numbers")
    if not _number(edge["battery_Wh"]) or not _number(edge["payload_kg"]):
        _fail("edge_state battery_Wh/payload_kg must be finite numbers")
    if not isinstance(edge["capabilities"], list) or not isinstance(edge["local_belief_summary"], dict):
        _fail("edge_state capabilities/local_belief_summary have the wrong type")

    _validate_center(model_input["center_context"])
    _validate_memory(model_input["center_memory"])
    memory = model_input["center_memory"]
    if memory["source_cut_s"] > record["metadata"]["sim_time_s"] + 1e-6:
        _fail("memory source cut must not postdate the decision")
    task_slots = [slot["value"] for slot in memory["slots"] if slot["kind"] == "task"]
    if model_input["center_context"]["known_tasks"] != task_slots:
        _fail("center_context tasks must come from the delivered bounded envelope")

    views = model_input["camera_views"]
    if not isinstance(views, list):
        _fail("camera_views must be a list")
    for view in views:
        if not isinstance(view, dict):
            _fail("camera_views entries must be objects")
        for key in ("view_id", "intrinsics", "extrinsics", "visible_targets", "perception_summary"):
            if key not in view:
                _fail("camera view missing " + key)

    candidates = model_input["candidates"]
    ids = _validate_candidates(candidates)

    supervision = record["supervision"]
    _validate_supervision(supervision, ids)
    _validate_intent(model_input["intent_text"], candidates, supervision["best_candidate_id"])
    return record
