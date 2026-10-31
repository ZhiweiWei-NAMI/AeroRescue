"""Acceptance tests for the phase-1 pilot dataset, collector and ISCC core.

The dataset tests are deliberately adversarial: they re-check every anti-leak
rule independently of ``schema.validate_decision_record`` so a schema change
cannot silently weaken the contract.
"""
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from aerorescue.phase1 import iscc_core
from aerorescue.phase1.collector import (REGIME_BY_SCENE, SCENES, SEEDS, SPLITS,
                                         _prepare, collect_episode)
from aerorescue.phase1.schema import (ACCEPT_EPSILON, CANDIDATE_KEYS, SCHEMA_ID,
                                      validate_decision_record)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase1_pilot"
RECORDS_PATH = DATA / "decision_records.jsonl"
SUMMARY_PATH = DATA / "pilot_summary.json"

FORBIDDEN_KEYS = {"truth", "truth_for_rendering_only", "world.entities", "provenance"}
LEAK_KEYS = {"source", "is_reference", "oracle", "utility", "feasible", "label"}
ACTION_IN_TEXT = re.compile(r"\bA\d{3}\b")


@pytest.fixture(scope="module")
def records():
    assert RECORDS_PATH.is_file(), "run tools/collect_phase1_pilot.py first"
    return [json.loads(line) for line in RECORDS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _walk(value, path, problems):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and (key.lower() in FORBIDDEN_KEYS or "truth" in key.lower()):
                problems.append(f"{path}.{key}")
            _walk(item, f"{path}.{key}", problems)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk(item, f"{path}[{index}]", problems)


def test_pilot_files_exist():
    assert RECORDS_PATH.is_file()
    assert SUMMARY_PATH.is_file()
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["schema"] == "aerorescue.phase1.pilot/v2"
    assert summary["record_schema"] == SCHEMA_ID
    assert summary["delta_weights"]["cursor"] == 0.0


def test_episode_coverage_and_splits(records):
    episodes = {(record["metadata"]["scene_id"], record["metadata"]["seed"])
                for record in records}
    assert episodes == {(scene, seed) for scene in SCENES for seed in SEEDS}
    assert len(episodes) == 24
    for record in records:
        metadata = record["metadata"]
        assert metadata["split"] == SPLITS[metadata["scene_id"]]
        assert record["record_id"] == metadata["record_id"]
    assert 240 <= len(records) <= 400

    per_episode = {}
    for record in records:
        key = (record["metadata"]["scene_id"], record["metadata"]["seed"])
        per_episode[key] = per_episode.get(key, 0) + 1
    assert set(per_episode) == episodes
    assert all(10 <= count <= 20 for count in per_episode.values())


def test_every_record_passes_validation(records):
    for record in records:
        validate_decision_record(record)


def test_model_input_is_isolated_from_metadata(records):
    for record in records:
        model_input = record["model_input"]
        assert set(model_input) == {"intent_text", "center_context", "center_memory",
                                    "edge_state", "camera_views", "candidates"}
        for banned in ("seed", "split", "record_id"):
            assert banned not in json.dumps(model_input), record["record_id"]


def test_model_input_carries_no_hidden_truth(records):
    for record in records:
        problems = []
        _walk(record["model_input"], "model_input", problems)
        assert not problems, (record["record_id"], problems)
        center = record["model_input"]["center_context"]
        assert "world" not in json.dumps(center)
        assert center["station_source"] == "coordinator_belief_or_initial_static"
        for station in center["stations"]:
            assert station["source"] == "coordinator_belief_or_initial_static"
        for observation in center["reported_observations"]:
            assert observation["reported_by"], record["record_id"]
            assert observation["last_report_s"] > 0, record["record_id"]
        assert record["model_input"]["edge_state"]["agent_id"] in record["record_id"]


def test_center_context_has_real_tasks_and_memory_envelope(records):
    for record in records:
        center = record["model_input"]["center_context"]
        tasks = center["known_tasks"]
        assert isinstance(tasks, list)
        for task in tasks:
            assert task["task_id"]
            assert re.match(r"^T\d{3}$", task["task_type"])
            assert task["status"]
            assert task["target_id"]
            assert isinstance(task["priority"], int)
        memory = record["model_input"]["center_memory"]
        assert memory["schema"] == "aerorescue.memory_envelope/v1"
        assert memory["mode"]
        assert memory["representation"]
        assert 0 < memory["serialized_bytes"] <= memory["budget_bytes"]
        assert isinstance(memory["slots"], list)
        assert len(json.dumps(memory, ensure_ascii=False).encode("utf-8")) == memory["serialized_bytes"]
        assert 0 <= memory["source_cut_s"] <= record["metadata"]["sim_time_s"]
        assert tasks == [slot["value"] for slot in memory["slots"] if slot["kind"] == "task"]
        for slot in memory["slots"]:
            assert slot["source"] == "coordinator_belief_or_initial_static"
            assert isinstance(slot["bytes"], int)


def test_candidates_carry_no_answer_hints(records):
    for record in records:
        candidates = record["model_input"]["candidates"]
        assert 2 <= len(candidates) <= 8
        for candidate in candidates:
            assert set(candidate) == set(CANDIDATE_KEYS), candidate.keys()
            assert not (set(candidate) & LEAK_KEYS)
            assert candidate["purpose"] in {"approach", "method"}
            assert candidate["method_scope"] in {"main", "recovery"}
            assert re.match(r"^A\d{3}$", candidate["action_id"])
            assert re.match(r"^T\d{3}$", candidate["task_type"])
            assert isinstance(candidate["params"], dict)
            assert isinstance(candidate["step_index"], int)
            assert candidate["task_rev"] is None or isinstance(candidate["task_rev"], int)


def test_intent_text_has_zero_action_or_target_leak(records):
    for record in records:
        intent = record["model_input"]["intent_text"]
        assert intent.strip()
        assert not ACTION_IN_TEXT.search(intent), intent
        for candidate in record["model_input"]["candidates"]:
            assert candidate["skill_name"] not in intent
        best = next(candidate for candidate in record["model_input"]["candidates"]
                    if candidate["candidate_id"]
                    == record["supervision"]["best_candidate_id"])
        own = {record["model_input"]["edge_state"]["agent_id"]}
        if best["target_id"] not in own:
            assert best["target_id"] not in intent


def test_counterfactual_utilities_are_real_rollouts(records):
    for record in records:
        supervision = record["supervision"]
        utilities = supervision["candidate_utilities"]
        ids = [candidate["candidate_id"] for candidate in record["model_input"]["candidates"]]
        assert set(utilities) == set(ids)
        weights = supervision["delta_weights"]
        assert weights["cursor"] == 0.0
        for cid, row in utilities.items():
            assert row["rollout_status"]
            assert isinstance(row["action_succeeded"], bool)
            assert row["cost_penalty"] >= 0
            assert row["fail_penalty"] <= 0
            if row["rollout_status"].startswith("rejected") or row["rollout_status"] == "failed":
                assert row["action_succeeded"] is False
                assert row["fail_penalty"] <= -1.0 + 1e-9
            expected = (weights["milestone"] * row["delta_milestone"]
                        + weights["potential"] * row["delta_potential"]
                        + weights["cursor"] * row["delta_cursor"]
                        - weights["cost"] * row["cost_penalty"]
                        + weights["fail"] * row["fail_penalty"])
            assert abs(expected - row["delta_total"]) <= 1e-6
        best = supervision["best_candidate_id"]
        assert utilities[best]["delta_total"] == max(
            row["delta_total"] for row in utilities.values())
        acceptable = supervision["acceptable_set"]
        assert best in acceptable
        for cid, row in utilities.items():
            within = utilities[best]["delta_total"] - row["delta_total"] <= ACCEPT_EPSILON + 1e-9
            assert within == (cid in acceptable)
        reference = supervision["reference_candidate_id"]
        # The menu is independent of the expert action. An off-menu reference
        # is a null diagnostic, never a fabricated candidate or training label.
        assert reference is None or reference in ids
        if reference is not None and supervision["reference_source"] == "forced_rule_action":
            assert not utilities[reference]["rollout_status"].startswith("rejected")
        # Cadence references are label-blind heuristic diagnostics. Native busy
        # rejection is retained and scored above; it is not an expert success.


def test_paired_voi_is_consistent(records):
    for record in records:
        voi = record["supervision"]["paired_voi_eval"]
        assert abs(voi["utility_with_center_memory"] - voi["utility_local_only"]
                   - voi["vo_memory_vs_local"]) <= 1e-9
        ablation = voi["ablation_2x2"]
        assert set(ablation) == {"B00", "B10", "B01", "B11"}
        assert ablation["B00"] == voi["utility_local_only"]
        assert ablation["B01"] == voi["utility_with_center_memory"]
        assert voi["bit_convention"] == "first=uplink_selection,second=downlink_memory"
        utilities = record["supervision"]["candidate_utilities"]
        for band, cid in voi["ablation_choices"].items():
            rollout = voi["ablation_rollouts"][band]
            assert abs(ablation[band] - rollout["delta_total"]) <= 1e-6
            treatment = voi["channel_treatments"][band]
            assert treatment["task_guided_uplink"] == (band[1] == "1")
            assert treatment["center_memory_enabled"] == (band[2] == "1")
            if band[2] == "0":
                assert treatment["downlink_bytes"] == 0
                assert treatment["delivered_memory_slots"] == 0
            elif not treatment["downlink_delivered"]:
                assert treatment["delivered_memory_slots"] == 0
        assert voi["local_choice"] in utilities
        assert voi["center_choice"] in utilities


def test_regime_pose_prior_and_iscc_budget(records):
    for record in records:
        supervision = record["supervision"]
        scene = record["metadata"]["scene_id"]
        assert supervision["regime"] == REGIME_BY_SCENE[scene]
        prior = supervision["pose_prior"]
        assert prior["regime"] == supervision["regime"]
        assert prior["sigma_theta_deg"] > 0 and prior["sigma_t_m"] > 0
        plan = supervision["budget_plan"]
        assert plan["status"]
        if plan["status"] == "proxy_solved":
            total = plan["b_g_star"] + plan["b_a_star"] + plan["b_s_star"]
            assert abs(total - plan["budget_bytes"]) <= 1e-6
            assert plan["b_g_star"] >= 0 and plan["b_a_star"] >= 0 and plan["b_s_star"] >= 0
            assert math.isfinite(plan["J_proxy"])


def test_seeds_change_situation_or_candidates(records):
    def signature(rows):
        return {(row["metadata"]["sim_time_s"], row["model_input"]["edge_state"]["agent_id"],
                 tuple(sorted(c["action_id"] + "|" + c["target_id"]
                              for c in row["model_input"]["candidates"])))
                for row in rows}

    for scene in SCENES:
        first = signature([r for r in records if r["metadata"]["scene_id"] == scene
                           and r["metadata"]["seed"] == 1])
        second = signature([r for r in records if r["metadata"]["scene_id"] == scene
                            and r["metadata"]["seed"] == 2])
        assert first and second
        assert first != second, scene
        assert len(first & second) < len(first), scene


def test_summary_matches_records(records):
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["total_episodes"] == 24
    assert summary["total_records"] == len(records)
    assert sum(summary["splits"].values()) == len(records)
    assert sum(summary["records_by_scene"].values()) == len(records)
    assert summary["candidates_per_record"]["min"] >= 2
    assert summary["label_balance"]["positive_candidates"] == len(records)
    assert summary["label_balance"]["negative_candidates"] > 0
    assert summary["collection_wall_s"] > 0
    assert summary["records_per_episode"]["min"] >= 10
    assert summary["records_per_episode"]["max"] <= 20
    assert summary["iscc_budget"]["proxy_solved_records"] > 0
    assert summary["iscc_budget"]["solved_records"] == 0
    assert summary["center_memory"]["mean_serialized_bytes"] > 0


def test_schema_rejects_leaks_and_bad_labels(records):
    record = deepcopy(records[0])
    record["model_input"]["center_context"]["truth_for_rendering_only"] = {"human.alive": True}
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["edge_state"]["world.entities"] = {"hidden": True}
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["candidates"][0]["source"] = "reference"
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["candidates"][0]["feasible"] = True
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["intent_text"] += " Execute A019 now."
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["center_context"]["known_tasks"] = [{"task_id": "invented"}]
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["center_memory"]["budget_bytes"] = 1
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["supervision"]["delta_weights"]["cursor"] = 1.0
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["supervision"]["acceptable_set"] = []
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["model_input"]["seed"] = record["metadata"]["seed"]
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["supervision"]["best_candidate_id"] = "c99"
    with pytest.raises(ValueError):
        validate_decision_record(record)

    record = deepcopy(records[0])
    record["metadata"]["split"] = "holdout"
    with pytest.raises(ValueError):
        validate_decision_record(record)


def test_collector_roundtrip_on_one_episode(tmp_path):
    cfg, core = _prepare("SC09", tmp_path)
    collected, stats = collect_episode("SC09", 1, cfg, core)
    assert stats["scene_id"] == "SC09" and stats["split"] == "train"
    assert 10 <= len(collected) <= 20
    for record in collected:
        validate_decision_record(record)
        assert record["metadata"]["seed"] == 1
        assert record["metadata"]["scene_id"] == "SC09"
        problems = []
        _walk(record["model_input"]["center_context"], "center_context", problems)
        _walk(record["model_input"]["edge_state"], "edge_state", problems)
        assert not problems


# --------------------------------------------------------------------------- iscc_core


def _se3_exp(xi):
    """Left-perturbation exponential used by the finite-difference check."""
    omega, upsilon = np.asarray(xi[:3], dtype=float), np.asarray(xi[3:], dtype=float)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        rotation, jacobian = np.eye(3), np.eye(3)
    else:
        axis = omega / theta
        skew = np.array([[0.0, -axis[2], axis[1]],
                         [axis[2], 0.0, -axis[0]],
                         [-axis[1], axis[0], 0.0]])
        rotation = np.eye(3) + math.sin(theta) * skew + (1 - math.cos(theta)) * (skew @ skew)
        jacobian = (np.eye(3) + (1 - math.cos(theta)) / theta * skew
                    + (1 - math.sin(theta) / theta) * (skew @ skew))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = jacobian @ upsilon
    return transform


def test_se3_projection_jacobian_matches_numeric_derivative():
    points = np.array([[1.0, 0.5, 4.0], [-2.0, 1.0, 6.0], [0.3, -0.7, 3.0]])
    fx, fy = 800.0, 820.0
    jacobian = iscc_core.se3_projection_jacobian(points, fx, fy)
    assert jacobian.shape == (3, 2, 6)
    assert np.isfinite(jacobian).all()

    xi = np.array([1e-4, -2e-4, 1.5e-4, 1e-3, -1e-3, 2e-3])
    transform = _se3_exp(xi)
    moved = (np.c_[points, np.ones(len(points))] @ transform.T)[:, :3]
    project = lambda p: np.stack([fx * p[:, 0] / p[:, 2], fy * p[:, 1] / p[:, 2]], axis=1)
    numeric = project(moved) - project(points)
    analytic = np.einsum("mij,j->mi", jacobian, xi)
    assert np.allclose(numeric, analytic, atol=1e-3)


def test_homography_condition_number_diverges_at_grazing_view():
    intrinsics = np.array([[800.0, 0.0, 320.0], [0.0, 820.0, 240.0], [0.0, 0.0, 1.0]])
    normal = np.array([[0.0, 0.0, 1.0]])
    angle = math.radians(35.0)
    rotation = np.array([[math.cos(angle), 0.0, math.sin(angle)],
                         [0.0, 1.0, 0.0],
                         [-math.sin(angle), 0.0, math.cos(angle)]])
    frontal = np.array([[1.0, 0.0, 5.0]])
    grazing = np.array([[1.0, 0.0, 0.01]])  # n^T p -> 0: the view is grazing
    cond_frontal = iscc_core.homography_condition_numbers(
        frontal, frontal, normal, rotation, intrinsics, frontal[:, 2], 800.0, 820.0)
    cond_grazing = iscc_core.homography_condition_numbers(
        grazing, grazing, normal, rotation, intrinsics, grazing[:, 2], 800.0, 820.0)
    homography = iscc_core.homography_jacobian(
        grazing, grazing, normal, rotation, intrinsics, grazing[:, 2], 800.0, 820.0)
    assert homography.shape == (1, 2, 2)
    assert cond_grazing[0] > 10.0 * cond_frontal[0]


def test_woodbury_posterior_matches_direct_formula_and_downdate_is_psd():
    rank, anchors = 6, 8
    lambda_0 = np.diag(np.linspace(2.0, 3.0, rank))
    x_a = np.array([[math.sin(0.4 * i + 0.3 * j) for j in range(anchors)]
                    for i in range(rank)])
    m_u = np.array([[math.cos(0.7 * k + 0.2 * j) for k in range(6)]
                    for j in range(anchors)])
    sigma_xi = np.diag([0.01] * 3 + [0.2] * 3)
    sigma_obs = 0.1
    post, ideal, delta = iscc_core.woodbury_eiv_posterior_precision(
        lambda_0, x_a, m_u, sigma_xi, sigma_obs)
    direct = lambda_0 + x_a @ np.linalg.inv(
        sigma_obs ** 2 * np.eye(anchors) + m_u @ sigma_xi @ m_u.T) @ x_a.T
    assert np.allclose(post, direct, atol=1e-8)
    assert np.allclose(ideal - delta, post, atol=1e-10)
    assert np.linalg.eigvalsh(delta).min() >= -1e-10
    assert np.linalg.eigvalsh(post).min() > 0


def test_algorithm1_splits_the_whole_budget():
    rank, anchors_g, anchors_a = 8, 5, 6
    jacobians = np.array([[[math.sin(0.3 * i + j), math.cos(0.2 * i - j),
                            math.sin(0.1 * (i + 1) * (j + 1)),
                            math.cos(0.15 * i + 0.3 * j), math.sin(0.25 * i - 0.2 * j),
                            math.cos(0.35 * i + 0.1 * j)] for j in range(2)]
                          for i in range(anchors_g)])
    gamma = np.array([np.diag([1.0 + 0.1 * i, 0.8 + 0.1 * i]) for i in range(anchors_g)])
    features = np.array([[math.sin(0.5 * i + 0.3 * j) for j in range(anchors_a)]
                         for i in range(rank)])
    anchor_jac = np.array([[[math.cos(0.4 * i + j), math.sin(0.6 * i - j),
                             math.cos(0.2 * i + 0.5 * j),
                             math.sin(0.3 * i + 0.2 * j), math.cos(0.5 * i - 0.4 * j),
                             math.sin(0.1 * i + 0.6 * j)] for j in range(2)]
                           for i in range(anchors_a)])
    x_t = features[:, :3]
    query = np.linalg.qr(x_t)[0][:, :3]
    plan = iscc_core.solve_ternary_budget_alg1(
        2048, np.eye(6) * 0.01, np.eye(rank), query,
        {"G": jacobians, "Gamma": gamma},
        {"X": features, "G": anchor_jac, "grad": np.ones(anchors_a)},
        x_t, 0.05, G_T=anchor_jac[:3])
    total = plan["b_g_star"] + plan["b_a_star"] + plan["b_s_star"]
    assert abs(total - 2048) <= 1e-6
    assert plan["b_g_star"] >= 0 and plan["b_a_star"] >= 0 and plan["b_s_star"] >= 0
    assert plan["b_a_star"] % plan["anchor_bytes"] == 0
    assert math.isfinite(plan["J_proxy"])
    assert plan["grid"]
    assert plan["J_proxy"] <= min(row["J_proxy"] for row in plan["grid"]) + 1e-9


def test_wiener_gate_never_increases_risk():
    d_e, rank, rank_act, anchors = 12, 8, 4, 3
    u_act = np.linalg.qr(np.array([[math.sin(0.3 * i + 0.7 * j) for j in range(rank_act)]
                                   for i in range(d_e)]))[0]
    a_hat = np.array([[math.cos(0.4 * i + 0.2 * j) for j in range(rank)]
                      for i in range(d_e)])
    x_t = np.array([[math.sin(0.6 * i + 0.5 * j) for j in range(anchors)]
                    for i in range(rank)])
    lambda_act = np.array([1.0, 0.8, 0.6, 0.4])
    for scale in (1e-6, 1.0, 1e3):
        posts = [np.eye(rank) * scale * (1 + 0.2 * channel)
                 for channel in range(rank_act)]
        gated, weights, reduction = iscc_core.matrix_wiener_shrinkage_gate(
            u_act, lambda_act, a_hat, x_t, posts, 0.05)
        assert gated.shape == (d_e, anchors)
        assert weights.shape == (rank_act, anchors)
        assert np.all(weights >= 0.0) and np.all(weights < 1.0)
        assert reduction >= 0.0
        signal = u_act.T @ a_hat @ x_t
        noise = np.array([np.sum(x_t * np.linalg.solve(post, x_t), axis=0)
                          + np.sum((u_act[:, channel] @ a_hat) ** 2) * .05
                          for channel, post in enumerate(posts)])
        ego_risk = .5 * np.sum(lambda_act[:, None] * signal ** 2)
        gated_risk = .5 * np.sum(lambda_act[:, None] * (
            (1 - weights) ** 2 * signal ** 2 + weights ** 2 * noise))
        assert np.isclose(ego_risk - gated_risk, reduction)
        assert 0 <= reduction <= ego_risk + 1e-12
        assert gated_risk <= ego_risk + 1e-12

    empty = np.zeros((rank, 0))
    gated, weights, reduction = iscc_core.matrix_wiener_shrinkage_gate(
        u_act, lambda_act, a_hat, empty, [np.eye(rank)] * rank_act, 0.05)
    assert gated.shape == (d_e, 0)
    assert weights.shape == (rank_act, 0)
    assert reduction == 0.0



def test_homography_matches_plane_intersection_finite_difference():
    intrinsics = np.array([[800., 0., 320.], [0., 820., 240.], [0., 0., 1.]])
    source = np.array([[.5, -.3, 4.]])
    normal = np.array([[.2, -.1, 1.]])
    angle = .3
    rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                         [-math.sin(angle), 0, math.cos(angle)]])
    translation = np.array([.4, -.2, 1.])
    receiver = source @ rotation.T + translation
    jac = iscc_core.homography_jacobian(receiver, source, normal, rotation,
                                         intrinsics, source[:, 2], 800., 820.)[0]
    uv = (intrinsics @ source[0])[:2] / source[0, 2]
    def warp(pixel):
        ray = np.linalg.solve(intrinsics, np.r_[pixel, 1.])
        point = ray * float(normal[0] @ source[0]) / float(normal[0] @ ray)
        transformed = rotation @ point + translation
        return np.array([800 * transformed[0], 820 * transformed[1]]) / transformed[2]
    step = 1e-4
    numeric = np.column_stack([(warp(uv + step * np.eye(2)[axis])
                               - warp(uv - step * np.eye(2)[axis])) / (2 * step)
                              for axis in range(2)])
    assert np.allclose(jac, numeric, atol=1e-8)


def test_woodbury_handles_zero_and_rank_deficient_pose_covariance():
    x = np.array([[1., 2., -1.], [.3, -.5, 2.]])
    m = np.arange(18, dtype=float).reshape(3, 6) / 10
    for sigma in (np.zeros((6, 6)), np.diag([.2, 0, 0, 0, 0, .01])):
        post, ideal, delta = iscc_core.woodbury_eiv_posterior_precision(
            np.eye(2), x, m, sigma, .3)
        direct = np.eye(2) + x @ np.linalg.solve(.09 * np.eye(3) + m @ sigma @ m.T, x.T)
        assert np.allclose(post, direct)
        assert np.allclose(post + delta, ideal)


def test_algorithm1_uses_task_geometry_and_never_buys_partial_anchors():
    g = np.zeros((1, 2, 6)); g[0, 0, 0] = 1
    x = np.array([[2.]])
    for budget in (0., 7., 12., 24.):
        plan = iscc_core.solve_ternary_budget_alg1(
            budget, np.eye(6), np.eye(1), np.zeros((1, 0)),
            {"G": g, "Gamma": np.eye(2)[None]},
            {"X": np.zeros((1, 0)), "G": np.zeros((0, 2, 6)), "grad": np.zeros(0)},
            x, 1., G_T=g, alpha_quant=0.)
        for row in plan["grid"]:
            assert row["b_g_star"] == 12 * len(row["geometric_anchor_ids"])
            assert row["b_g_star"] <= budget
            assert row["m_star"] == len(row["representation_anchor_ids"])
        assert plan["sigma_x2"] == 4.
        assert plan["pose_trace"] > 0
        if budget < 12:
            assert plan["b_g_star"] == 0 and plan["pose_trace"] == 1.
        else:
            assert plan["b_g_star"] == 12 and plan["pose_trace"] == .5


def test_wiener_rejects_invalid_theorem_inputs():
    for eigenvalue, quant in ((-1., 0.), (1., -1.)):
        with pytest.raises(ValueError):
            iscc_core.matrix_wiener_shrinkage_gate(
                np.eye(1), [eigenvalue], np.eye(1), np.ones((1, 1)), [np.eye(1)], quant)


def test_schema_rejects_nested_memory_and_candidate_param_leaks(records):
    for branch in ("center_memory", "edge_state"):
        record = deepcopy(records[0])
        record["model_input"][branch]["best_candidate_id"] = "c01"
        with pytest.raises(ValueError):
            validate_decision_record(record)
    for key in ("truth", "is_reference"):
        record = deepcopy(records[0])
        record["model_input"]["candidates"][0]["params"][key] = True
        with pytest.raises(ValueError):
            validate_decision_record(record)


def test_camera_and_method_metadata_are_independent_of_reference():
    from types import SimpleNamespace
    from aerorescue.phase1.collector import _camera_views, _method_info, _METHOD_CACHE
    agent = SimpleNamespace(id="edge", spec={"capabilities": ["C05"], "mobility": "air"},
                            platform={"pose": [0., 0.], "yaw_deg": 13.})
    views = _camera_views(agent, {}, [], 3, 0.)
    assert views[0]["extrinsics"]["yaw_deg"] == 13.
    _METHOD_CACHE.clear()
    # The cache key includes the requested action, not just its task sequence.
    _method_info(agent, "T019", "A001")
    _method_info(agent, "T019", "A004")
    assert len(_METHOD_CACHE) == 2


def test_paired_voi_executes_four_native_rollouts_with_fixed_policy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from rescue_scenes.core.api import Environment
    from aerorescue.phase1 import collector
    cfg, core = _prepare("SC09", tmp_path)
    sim = Environment.from_config(core).sim
    sim.start()
    for holder in sim.agents.values():
        holder.on_tick = lambda sim: None
    agent = next(a for a in sim.agents.values() if a.kind != "coordinator")
    agent.effective_capabilities = agent.spec.get("capabilities", [])
    rows = collector._candidates(sim, agent, None, __import__("random").Random(1),
                                 collector._targets(agent), sim.t)
    original_rollout = collector._rollout
    called = []
    def recording_rollout(*args, **kwargs):
        called.append((id(args[0]), args[0].t))
        return original_rollout(*args, **kwargs)
    monkeypatch.setattr(collector, "_rollout", recording_rollout)
    result = collector._paired_voi(sim, cfg, rows, agent, [])
    assert len(called) == 4
    assert len({when for _, when in called}) == 1
    assert sim.t == 0
    for band, treatment in result["channel_treatments"].items():
        assert treatment["task_guided_uplink"] == (band[1] == "1")
        assert treatment["center_memory_enabled"] == (band[2] == "1")
        assert result["ablation_2x2"][band] == result["ablation_rollouts"][band]["delta_total"]



def test_candidate_menu_does_not_encode_expert_action(tmp_path):
    import random
    from rescue_scenes.core.api import Environment
    from aerorescue.phase1 import collector
    cfg, core = _prepare("SC09", tmp_path)
    sim = Environment.from_config(core).sim
    sim.start()
    agent = next(a for a in sim.agents.values() if a.kind != "coordinator")
    targets = collector._targets(agent)
    menus = [collector._candidates(sim, agent, reference, random.Random(37), targets, sim.t)
             for reference in (None, {"action_id": "A019", "target_id": "SECRET_A"},
                               {"action_id": "A001", "target_id": "SECRET_B"})]
    assert menus[0] == menus[1] == menus[2]



def test_algorithm1_greedy_steps_match_direct_matrix_objective():
    x = np.array([[1., .2, -1.], [.4, 2., .5]])
    task = np.array([[1., -.2], [.5, .8]])
    query = np.eye(2)
    g = np.array([[[1., 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0]],
                  [[0, 0, 1., 0, 0, 0], [0, 0, 0, 1., 0, 0]]])
    a_g = np.concatenate([g, 2 * g[:1]], axis=0)
    grad = np.array([.1, .4, .2])
    prior = np.diag([2., 3.])
    variance = .25
    w = task @ task.T + .3 * query @ query.T
    plan = iscc_core.solve_ternary_budget_alg1(
        24, np.eye(6), prior, query, {"G": g, "Gamma": np.tile(np.eye(2), (2, 1, 1))},
        {"X": x, "G": a_g, "grad": grad}, task, .5, G_T=g,
        anchor_bytes=4, gamma_q=.3, g_levels=3)
    for row in plan["grid"]:
        pose_info = np.eye(6) + sum((g[i].T @ g[i] for i in row["geometric_anchor_ids"]),
                                   np.zeros((6, 6)))
        sigma = np.linalg.inv(pose_info)
        beta = 1 / (variance + grad * np.einsum("mij,jk,mik->m", a_g, sigma, a_g))
        precision = prior.copy()
        selected = []
        for chosen in row["representation_anchor_ids"]:
            cov = np.linalg.inv(precision)
            gains = {i: np.trace(w @ cov) - np.trace(w @ np.linalg.inv(
                precision + beta[i] * np.outer(x[:, i], x[:, i])))
                for i in range(3) if i not in selected}
            assert gains[chosen] == pytest.approx(max(gains.values()))
            precision += beta[chosen] * np.outer(x[:, chosen], x[:, chosen])
            selected.append(chosen)
        expected = (np.trace(w @ np.linalg.inv(precision))
                    + np.einsum("mij,jk,mik->", g, sigma, g)
                    + np.sum(task ** 2) * 2 ** (-16 * row["b_s_star"] / task.size))
        assert row["J_proxy"] == pytest.approx(expected)


def test_rollout_does_not_claim_another_actions_scheduled_milestone(tmp_path):
    from rescue_scenes.core.api import Environment
    from aerorescue.phase1 import collector
    cfg, core = _prepare("SC09", tmp_path)
    sim = Environment.from_config(core).sim
    sim.start()
    for holder in sim.agents.values():
        holder.on_tick = lambda sim: None
    required = set(collector._action_row("A004").get("capabilities", []))
    agent = next(a for a in sim.agents.values() if a.kind != "coordinator"
                 and required <= set(a.spec.get("capabilities", [])))
    person = next(eid for eid, entity in sim.world.entities.items() if entity.get("cls") == "human")
    old_support = list(sim.world.get(person, "human.support_received", []))
    sim.schedule(.01, sim.write_truth, person, "human.support_received",
                 old_support + ["contacted"], "WORLD")
    candidate = collector._candidate_row(agent, "A004", "T019", agent.platform_eid, agent.platform["pose"])
    result = collector._rollout(sim, agent.id, candidate, [], agent.spec)
    assert result["rollout_status"] == "completed"
    assert result["delta_milestone"] == 0
    assert sim.t == 0
    assert sim.world.get(person, "human.support_received", []) == old_support


def test_payload_milestone_uses_the_native_world_field(tmp_path):
    from rescue_scenes.core.api import Environment
    from aerorescue.phase1 import collector
    _, core = _prepare("SC09", tmp_path)
    sim = Environment.from_config(core).sim
    agent = next(a for a in sim.agents.values() if a.kind != "coordinator")
    sim.write_truth(agent.platform_eid, "platform.payload_item_ids", ["physical-item"], "WORLD")
    assert collector._state_snapshot(sim, agent.id)["payload_items"] == 1



def test_wiener_gate_is_exact_for_small_signal_scales():
    _, weights, reduction = iscc_core.matrix_wiener_shrinkage_gate(
        np.eye(1), [1.], np.eye(1), np.array([[1e-8]]), [np.eye(1)], 0.)
    assert weights[0, 0] == pytest.approx(.5)
    assert reduction == pytest.approx(2.5e-17, rel=1e-12, abs=0.)



def test_wiener_uses_unquantized_features_for_posterior_variance_when_supplied():
    gated, weights, reduction = iscc_core.matrix_wiener_shrinkage_gate(
        np.eye(1), [2.], np.array([[3.]]), np.array([[.5]]), [np.array([[4.]])],
        .1, X_T=np.array([[2.]]))
    expected = 1.5 ** 2 / (1.5 ** 2 + 2. ** 2 / 4. + 3. ** 2 * .1)
    assert weights[0, 0] == pytest.approx(expected)
    assert gated[0, 0] == pytest.approx(expected * 1.5)
    assert reduction == pytest.approx(expected * 1.5 ** 2)



def test_archive_free_clone_matches_full_deepcopy_physical_transition(tmp_path):
    from rescue_scenes.core.api import Environment
    from rescue_scenes.core.sim.kernel import Simulation
    from aerorescue.phase1 import collector
    _, core = _prepare("SC09", tmp_path)
    sim = Environment.from_config(core).sim
    sim.start()
    for holder in sim.agents.values():
        holder.on_tick = lambda sim: None
    required = set(collector._action_row("A004").get("capabilities", []))
    agent = next(a for a in sim.agents.values() if a.kind != "coordinator"
                 and required <= set(a.spec.get("capabilities", [])))
    live_events, live_history = sim.events, sim.world.history
    full, fast = deepcopy(sim), collector._clone_sim(sim)
    full.submit_action = Simulation.submit_action.__get__(full, Simulation)
    assert sim.events is live_events and sim.world.history is live_history
    assert fast.events == [] and fast.world.history == [] and fast.ledger.entries == []
    for clone in (full, fast):
        iid = clone.submit_action(agent.id, "A004", {"target": agent.platform_eid})
        clone.run(clone.t + float(clone.instances[iid]["plan"]["duration_s"]))
        assert clone.instances[iid]["status"] == "succeeded"
    assert full.world.entities == fast.world.entities
    assert sim.t == 0
