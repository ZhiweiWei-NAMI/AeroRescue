#!/usr/bin/env python3
"""Local Phase-1 collection, real-model diagnostics and review packaging.

Research acceptance is reported explicitly in evaluation_results.json. Unsupported
SOTA and heterogeneous model backends never receive fabricated experiment scores.
The formal audit lists remaining gaps against the eight frozen contracts.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rescue_scenes.backgrounds import materialize
from rescue_scenes.core.api import Environment
from rescue_scenes.paths import expand

from aerorescue.phase1.branch_collector import (
    MatchedHiddenStateBuilder,
    attach_ground_rgb,
    build_natural_section_record,
    clone_core_env,
    ground_eye,
    uav_eye,
    verify_clone_isolation,
)
from aerorescue.phase1.scene_renderer import (
    SceneMultichannelRenderer,
    _position_xy,
    _safe_grid_enu,
)
from aerorescue.phase1 import schema_v2 as S
from aerorescue.phase1.visual_qa import render_contact_sheet, run_machine_visual_qa
from aerorescue.iscc import (
    FixedSubspaceEIVFilter,
    apply_positive_part_shrinkage_gate,
    compute_fisher_and_vertical_by,
    compute_homography_jacobian,
    compute_pose_ray_jacobian,
    compute_untruncated_pullback_srank,
    evaluate_safety_and_regret_metrics,
    quantize_and_dequantize_blind_features,
    se3_exp,
    solve_ternary_budget_algorithm1,
)
from aerorescue.iscc.pair0_qwen_decoder import (
    Pair0QwenVLMRuntime,
    fit_independent_reference_mapping,
)

DATA_V2_DIR = ROOT / "data" / "phase1_v2"
RUNS_LOCAL_DIR = ROOT / "runs" / "phase1_local"
COLAB_BUNDLE_DIR = ROOT / "runs" / "colab_bundle"

SCENE_CONFIGS = {
    "SC01": "SC01.json",
    "SC09": "SC09_upstream_allocated.json",
    "SC07": "SC07_upstream_allocated.json",
    "SC10": "SC10_upstream_allocated.json",
    "SC08": "SC08_upstream_allocated.json",
    "GT04": "GT04_upstream_allocated.json",
    "SC05": "SC05_upstream_allocated.json",
}

INTRINSICS_640X384 = {
    "width": 640,
    "height": 384,
    "fx": round((640.0 / 2.0) / math.tan(math.radians(35.0)), 6),
    "fy": round((640.0 / 2.0) / math.tan(math.radians(35.0)), 6),
    "cx": 320.0,
    "cy": 192.0,
}


def _select_cameras_for_section(
    renderer: SceneMultichannelRenderer,
    env: Environment,
    regime: str,
    seed: int,
    section_idx: int,
    record: dict,
) -> dict:
    """Choose physically valid UAV and Ground camera poses for a section."""
    # Cameras are mounted on the same agents whose option branches are scored.
    edge = env.sim.agents[record["model_input"]["edge_telemetry_and_belief"]["agent_id"]]
    uav = next(a for a in env.sim.agents.values() if a.spec.get("mobility") == "air")
    gxy = edge.platform["pose"][:2]
    uxy = uav.platform["pose"][:2]
    gx, gy, gz_ground = _safe_grid_enu(renderer.grid, gxy, 0.0)
    ux, uy, uz_ground = _safe_grid_enu(renderer.grid, uxy, 0.0)
    anchors = [c["params"]["anchor_xy"] for c in record["model_input"]["public_candidate_menu"] if "anchor_xy" in c["params"]]
    focus = np.mean(anchors, axis=0)
    fx, fy, fz = _safe_grid_enu(renderer.grid, focus, 0.0)
    uav_pose = [ux, uy, uz_ground + 120.0]
    body = renderer.entity_boxes.get(edge.platform_eid, {})
    sensor_z = max(gz_ground + 1.85, body.get("u", (0, gz_ground))[1] + .2)
    gnd_pose = [gx, gy, sensor_z]
    uav_yaw = math.degrees(math.atan2(fy - uy, fx - ux))
    uav_pitch = math.degrees(math.atan2(fz - uav_pose[2], max(math.hypot(fx - ux, fy - uy), 1.0)))
    targets = []

    best_pair = None
    best_qa = None

    for attempt in range(16):
        uav_rpy = [uav_yaw, uav_pitch, 0.0]
        gnd_yaw = math.degrees(math.atan2(fy - gy, fx - gx)) + attempt * 22.5
        if regime == "C" and section_idx == 1:
            gnd_yaw += 145.0
        gnd_rpy = [gnd_yaw, -4.0, 0.0]
        cand_pair = renderer.render_pair(
            uav_pose, uav_rpy, gnd_pose, gnd_rpy, INTRINSICS_640X384, targets[:8]
        )
        cand_qa = run_machine_visual_qa(cand_pair, regime)
        if best_pair is None:
            best_pair, best_qa = cand_pair, cand_qa
        if cand_qa["passed"]:
            best_pair, best_qa = cand_pair, cand_qa
            break

    return {
        "pair_render": best_pair,
        "qa": best_qa,
        "targets": targets[:8],
    }


def render_zero_overlap_pair(renderer, uav_pose, ground_pose, uav_yaw, ground_yaw, intrinsics):
    """Choose a physically rendered loss view; reject any remaining co-visible surface."""
    grounds=[]
    for angle in (0,90,180,270):
        view=renderer.render_view(ground_pose,[ground_yaw+angle,-4.,0.],intrinsics)
        grounds.append(view)
    for turn in (180,90,-90,0):
        for pitch in (-35.,-55.):
            source=renderer.render_view(uav_pose,[uav_yaw+turn,pitch,0.],intrinsics)
            for ground in grounds:
                cov=renderer.compute_covisibility_and_blind_spots(source,ground,[])
                pair={"uav":source,"ground":ground,"covisibility":cov,"target_entity_ids":[]}
                if cov["covis_pixels"]==0 and run_machine_visual_qa(pair,"C")["passed"]:
                    return pair
    raise AssertionError("no physically valid zero-overlap view exists in the tested gimbal directions")


def _save_render_pair(pair_bundle: dict, out_dir: Path, stem: str) -> dict:
    """Save UAV/Ground RGB PNGs and geometry NPZs."""
    img_dir = out_dir / "images"
    geom_dir = out_dir / "geometry"
    sensor_dir = out_dir / "sensors"
    sensor_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    geom_dir.mkdir(parents=True, exist_ok=True)

    uav_png = img_dir / f"{stem}_uav_rgb.png"
    gnd_png = img_dir / f"{stem}_ground_rgb.png"
    uav_npz = geom_dir / f"{stem}_uav_geom.npz"
    gnd_npz = geom_dir / f"{stem}_ground_geom.npz"

    Image.fromarray(pair_bundle["uav"]["rgb"]).save(uav_png)
    Image.fromarray(pair_bundle["ground"]["rgb"]).save(gnd_png)

    np.savez_compressed(
        uav_npz,
        depth=pair_bundle["uav"]["depth"].astype(np.float32),
        normal_cam=pair_bundle["uav"]["normal_cam"].astype(np.float32),
        normal_world=pair_bundle["uav"]["normal_world"].astype(np.float32),
        instance_id=pair_bundle["uav"]["instance_id"].astype(np.uint32),
        valid_mask=pair_bundle["uav"]["valid_mask"].astype(bool),
        covis_mask=pair_bundle["covisibility"]["covis_mask_uav"].astype(bool),
        blind_mask=pair_bundle["covisibility"]["blind_mask_uav"].astype(bool),
        R_c_from_w=pair_bundle["uav"]["R_c_from_w"].astype(np.float64),
        cam_pose_enu=np.asarray(pair_bundle["uav"]["cam_pose_enu"], dtype=np.float64),
    )
    np.savez_compressed(
        gnd_npz,
        depth=pair_bundle["ground"]["depth"].astype(np.float32),
        normal_cam=pair_bundle["ground"]["normal_cam"].astype(np.float32),
        normal_world=pair_bundle["ground"]["normal_world"].astype(np.float32),
        instance_id=pair_bundle["ground"]["instance_id"].astype(np.uint32),
        valid_mask=pair_bundle["ground"]["valid_mask"].astype(bool),
        R_c_from_w=pair_bundle["ground"]["R_c_from_w"].astype(np.float64),
        cam_pose_enu=np.asarray(pair_bundle["ground"]["cam_pose_enu"], dtype=np.float64),
    )
    uav_depth = sensor_dir / f"{stem}_uav_depth.npy"
    ground_depth = sensor_dir / f"{stem}_ground_depth.npy"
    np.save(uav_depth, pair_bundle["uav"]["depth"].astype(np.float32), allow_pickle=False)
    np.save(ground_depth, pair_bundle["ground"]["depth"].astype(np.float32), allow_pickle=False)
    return {
        "uav_depth_abs": str(uav_depth),
        "ground_depth_abs": str(ground_depth),
        "uav_rgb_abs": uav_png,
        "ground_rgb_abs": gnd_png,
        "uav_geom_abs": str(uav_npz),
        "ground_geom_abs": str(gnd_npz),
    }


def attach_render_calibration(rec, pair, saved, grid):
    """Persist the actual optical camera chain and public anchor ENU positions."""
    g = np.linalg.inv(pair["ground"]["T_c_from_w"])
    u = np.linalg.inv(pair["uav"]["T_c_from_w"])
    truth = np.linalg.inv(g) @ u
    xi = np.asarray(rec["supervision_and_eval"]["true_pose_e_from_i"]["xi"])
    prior = se3_exp(xi) @ truth
    views = rec["model_input"]["aerial_ground_views"]
    views["ground_view"]["depth_path"] = saved["ground_depth_abs"]
    views["uav_view"]["depth_path"] = saved["uav_depth_abs"]
    views["ground_view"]["extrinsics_w_from_c"] = g.tolist()
    views["uav_view"]["prior_extrinsics_w_from_c"] = (g @ prior).tolist()
    rec["supervision_and_eval"]["true_pose_e_from_i"]["T_e_from_i"] = truth.tolist()
    menu=rec["model_input"]["public_candidate_menu"]
    edge=rec["model_input"]["edge_telemetry_and_belief"]
    if "coordinate_frame" not in menu[0]["params"]:
        edge["pose_enu"]=list(_safe_grid_enu(grid,edge["pose_enu"][:2],edge["pose_enu"][2]))
        for entry in edge["believed_entities"].values():
            if entry["pose_enu"] is not None:
                entry["pose_enu"]=list(_safe_grid_enu(grid,entry["pose_enu"][:2],entry["pose_enu"][2]))
        for station in rec["model_input"]["center_delivered_context"]["static_stations"]:
            station["pose_enu"]=list(_safe_grid_enu(grid,station["pose_enu"][:2],station["pose_enu"][2]))
        for report in rec["model_input"]["center_delivered_context"]["delivered_reports"]:
            report["values"]["coordinate_frame"]="sim_core_xy"
    for cand in menu:
        xy = cand["params"].get("anchor_xy")
        cand["params"]["coordinate_frame"]="path=sim_core_xy;anchor_enu=world_enu"
        cand["params"]["anchor_enu"] = list(edge["pose_enu"]) if xy is None else list(_safe_grid_enu(grid, xy, 0.0))
    cov = pair["covisibility"]
    common = int(cov["covis_pixels"])
    ground_count = max(common, int(cov["ground_valid_pixels"]))
    rec["supervision_and_eval"]["geometry_paths"] = {
        "uav_geom_npz": saved["uav_geom_abs"], "ground_geom_npz": saved["ground_geom_abs"],
        "covisibility_summary": {
            "overlap_ratio": common / max(ground_count, 1), "common_anchor_count": common,
            "uav_anchor_count": int(cov["uav_valid_pixels"]), "ground_anchor_count": ground_count,
            "mean_depth_ground_m": float(np.mean(pair["ground"]["depth"][pair["ground"]["valid_mask"]])),
            "mean_depth_uav_m": float(np.mean(pair["uav"]["depth"][pair["uav"]["valid_mask"]])),
        }}


def collect_phase1_v2_dataset() -> dict:
    """Step 1: Collect multi-scene 4-channel rendered + branching dataset & run Visual QA."""
    print("=== [Step 1] Collecting Phase-1 v2 Multi-Scene Dataset & Running Visual QA ===")
    DATA_V2_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    contact_overview_rows = []
    contact_matched_rows = []
    qa_reports = []
    clone_isolation_verified = False
    sequential_records = []

    for scene_id, cfg_file in SCENE_CONFIGS.items():
        split = S.SCENE_SPLITS[scene_id]
        regime = S.SCENE_REGIMES[scene_id]
        cfg = expand(json.loads((ROOT / "configs" / cfg_file).read_text()))

        with tempfile.TemporaryDirectory(prefix=f"p1v2-{scene_id}-") as tmp:
            # Materialize once per scene (seed=1) and reuse the materialized semantics + renderer across sections
            cfg_base = deepcopy(cfg)
            cfg_base["seed"] = 1
            _, bundle = materialize(cfg_base, Path(tmp))
            core_base = deepcopy(bundle["core"])
            core_base["semantics"]["path"] = str(Path(tmp) / core_base["semantics"]["path"])
            env_base = Environment.from_config(core_base)
            renderer = SceneMultichannelRenderer.from_environment(
                cfg_base, env_base, bundle, device="cuda:0"
            )

            for seed in (1, 2):
                core = deepcopy(core_base)
                core["seed"] = seed
                env = Environment.from_config(core)

                if not clone_isolation_verified:
                    iso = verify_clone_isolation(env, horizon_s=20.0)
                    assert all(iso.values()), f"Clone isolation failed: {iso}"
                    clone_isolation_verified = True
                    print("  [OK] verify_clone_isolation 4 assertions passed:", iso)

                renderer.rebuild(world=env.sim.world)

                # Collect 2 natural sections per (scene_id, seed)
                for sec_idx in range(2):
                    sec_regime = "C" if (regime == "C" and sec_idx == 1) else regime
                    for agent in env.sim.agents.values():
                        agent.policy_mode = "external_harness"
                    env.sim.run(10.0 + 30.0 * sec_idx)
                    rec = build_natural_section_record(
                        env, scene_id=scene_id, seed=seed, sim_time_s=float(env.sim.t),
                        section_tag=f"nat{sec_idx}", regime=sec_regime)
                    cam_spec = _select_cameras_for_section(renderer, env, sec_regime, seed, sec_idx, rec)
                    pair_render = cam_spec["pair_render"]
                    qa = cam_spec["qa"]
                    measured_regime = "water" if scene_id == "SC05" else ("C" if pair_render["covisibility"]["overlap_uav"] < .02 else "B" if pair_render["covisibility"]["overlap_uav"] < .15 else "A")
                    rec["metadata"]["physical_regime"] = measured_regime
                    qa_reports.append({
                        "scene_id": scene_id,
                        "seed": seed,
                        "section": f"nat{sec_idx}",
                        "regime": sec_regime,
                        "passed": qa["passed"],
                        "metrics": qa["metrics"],
                        "reasons": qa["reasons"],
                    })
                    assert qa["passed"], f"Visual QA failed on {scene_id} s{seed} nat{sec_idx}: {qa['reasons']}"

                    stem = rec["metadata"]["record_id"]
                    saved = _save_render_pair(pair_render, DATA_V2_DIR, stem)
                    attach_ground_rgb(rec, saved["ground_rgb_abs"], uav_path=saved["uav_rgb_abs"])
                    attach_render_calibration(rec, pair_render, saved, renderer.grid)
                    S.validate_phase1_v2_record(rec)
                    records.append(rec)

                    if seed == 1 and sec_idx == 0:
                        q_map = {
                            cid: row["q_value_120s"]
                            for cid, row in rec["supervision_and_eval"]["branch_evaluations"].items()
                        }
                        contact_overview_rows.append({
                            **pair_render,
                            "title": f"{scene_id} ({rec['metadata']['physical_regime']}/{split})",
                            "branch_q": q_map,
                        })

                if scene_id == "SC08" and seed == 1:
                    # One continuously advanced core state, with legal native motion.
                    from rescue_scenes.core.sim.kernel import Simulation
                    seq_env = clone_core_env(env)
                    seq_base = deepcopy(rec)
                    seq_agent = seq_env.sim.agents[rec["model_input"]["edge_telemetry_and_belief"]["agent_id"]]
                    seq_start = float(seq_env.sim.t)
                    commands = []
                    for step in range(1, 16):
                        if step in (1, 9):
                            anchor = seq_base["model_input"]["public_candidate_menu"][0 if step == 1 else 1]["params"]["anchor_xy"]
                            iid = Simulation.submit_action(seq_env.sim, seq_agent.id, "A002", {"goal": anchor})
                            commands.append({"step": step, "instance_id": iid, "status": seq_env.sim.instances[iid]["status"]})
                        seq_env.sim.run(seq_start + 3.0 * step)
                        renderer.rebuild(world=seq_env.sim.world)
                        snapshot = deepcopy(seq_base)
                        snapshot["metadata"]["record_id"] = f"SC08_seed1_seq{step:02d}_{seq_agent.id}"
                        snapshot["metadata"]["sim_time_s"] = float(seq_env.sim.t)
                        snapshot["model_input"]["edge_telemetry_and_belief"]["pose_enu"] = list(ground_eye(seq_agent))
                        spec = _select_cameras_for_section(renderer, seq_env, "C", seed, int(9 <= step <= 10), snapshot)
                        render = spec["pair_render"]
                        if 9 <= step <= 10:
                            render=render_zero_overlap_pair(renderer,render["uav"]["cam_pose_enu"],render["ground"]["cam_pose_enu"],render["uav"]["cam_rpy_deg"][0],render["ground"]["cam_rpy_deg"][0],INTRINSICS_640X384)
                        assert run_machine_visual_qa(render,"C")["passed"], spec["qa"]["reasons"]
                        saved = _save_render_pair(render, DATA_V2_DIR, snapshot["metadata"]["record_id"])
                        attach_ground_rgb(snapshot, saved["ground_rgb_abs"], saved["uav_rgb_abs"])
                        attach_render_calibration(snapshot, render, saved, renderer.grid)
                        # No inherited branch labels are used or published for diagnostic-only frames.
                        snapshot["supervision_and_eval"].pop("branch_evaluations")
                        snapshot["supervision_and_eval"].pop("soft_q_teacher_120s")
                        snapshot["schema"] = "aerorescue.phase1.sequential_diagnostic/v1"
                        snapshot["trajectory"] = {"step": step, "commands": deepcopy(commands), "source": "continuous_native_simulator"}
                        sequential_records.append(snapshot)
                    renderer.rebuild(world=env.sim.world)

                if scene_id in ("SC01", "SC07", "GT04") and seed == 2:
                    accepted=0
                    # Eligibility is fixed before fitting any model: natural
                    # physical non-interference and disjoint H=120 optima only.
                    attempts=[]
                    for challenge_seed in range(1,129):
                        if accepted==4:break
                        try:
                            builder=MatchedHiddenStateBuilder(clone_core_env(env_base),scene_id,challenge_seed,
                                out_dir=None,single_factor=True)
                            plus,minus,pair_sum=builder.build_pair(pair_index=accepted,write_geometry=False)
                            assert "c01" in pair_sum["optimal_plus"] and "c02" in pair_sum["optimal_minus"],"named route optima are not the open corridors"
                            gx,gy,gz=_safe_grid_enu(renderer.grid,ground_eye(builder.agent)[:2],1.85)
                            jx,jy,jz=_safe_grid_enu(renderer.grid,builder.anchors["J"],0.)
                            ux,uy,uz=_safe_grid_enu(renderer.grid,uav_eye(builder.uav)[:2],44.)
                            yaw_g=math.degrees(math.atan2(jy-gy,jx-gx))
                            yaw_u=math.degrees(math.atan2(jy-uy,jx-ux))
                            pitch_u=math.degrees(math.atan2(jz-uz,max(math.hypot(jx-ux,jy-uy),1.)))
                            rendered=[]
                            for side,world in (("plus",builder._patch_world("A","plus")),("minus",builder._patch_world("B","minus"))):
                                renderer._previous_xy.clear();renderer._yaw.clear()
                                renderer.rebuild(world=world.sim.world)
                                pair=renderer.render_pair([ux,uy,uz],[yaw_u,pitch_u,0.],[gx,gy,gz],[yaw_g,-2.5,0.],INTRINSICS_640X384,[])
                                qa=run_machine_visual_qa(pair,"C")
                                if not qa["passed"]:raise AssertionError(str(qa["reasons"]))
                                rendered.append((pair,qa))
                            rp,rm=rendered[0][0],rendered[1][0]
                            assert np.array_equal(rp["ground"]["rgb"],rm["ground"]["rgb"]),"ground RGB non-interference failed"
                            assert np.array_equal(rp["ground"]["depth"],rm["ground"]["depth"]),"ground depth non-interference failed"
                            assert not np.array_equal(rp["uav"]["rgb"],rm["uav"]["rgb"]),"source cannot observe hidden factor"
                            # Save only physically eligible pairs, including both
                            # equally weighted worlds. Every rejection is reported.
                            staged_records=[];staged_qa=[];staged_contact=[]
                            common_overlap=.5*(rp["covisibility"]["overlap_uav"]+rm["covisibility"]["overlap_uav"])
                            for rec_item,(pair,qa) in zip((plus,minus),rendered):
                                saved=_save_render_pair(pair,DATA_V2_DIR,rec_item["metadata"]["record_id"])
                                attach_ground_rgb(rec_item,saved["ground_rgb_abs"],saved["uav_rgb_abs"])
                                attach_render_calibration(rec_item,pair,saved,renderer.grid)
                                overlap=common_overlap
                                rec_item["metadata"]["physical_regime"]="C" if overlap<.02 else "B" if overlap<.15 else "A"
                                S.validate_phase1_v2_record(rec_item);staged_records.append(rec_item)
                                staged_qa.append({"scene_id":scene_id,"seed":challenge_seed,"section":"single_factor_twin",**qa})
                                if accepted==0:
                                    staged_contact.append({**pair,"title":f"{scene_id} {rec_item['metadata']['split']} twin",
                                        "branch_q":{k:v["q_value_120s"] for k,v in rec_item["supervision_and_eval"]["branch_evaluations"].items()}})
                            S.assert_matched_pair_noninterference(plus,minus)
                            records.extend(staged_records);qa_reports.extend(staged_qa);contact_matched_rows.extend(staged_contact)
                            attempts.append({"seed":challenge_seed,"eligible":True,"factor":"obstacle_state_only",
                                "optimal_plus":pair_sum["optimal_plus"],"optimal_minus":pair_sum["optimal_minus"]})
                            accepted+=1
                            print("  [OK] single-factor",scene_id,challenge_seed,pair_sum["optimal_plus"],pair_sum["optimal_minus"],flush=True)
                        except (RuntimeError,AssertionError) as exc:
                            if isinstance(exc,RuntimeError) and any(term in str(exc) for term in ("CUDA","Triton","memory")):
                                raise
                            attempts.append({"seed":challenge_seed,"eligible":False,"reason":str(exc)})
                            print("  [Eligibility]",scene_id,challenge_seed,str(exc),flush=True)
                    (DATA_V2_DIR/f"{scene_id}_challenge_eligibility.json").write_text(json.dumps(attempts,ensure_ascii=False,indent=2)+"\n")
                    assert accepted==4,f"{scene_id}: only {accepted}/4 physically eligible pairs"
                    renderer.rebuild(world=env.sim.world)

    (DATA_V2_DIR / "sequential_diagnostic_records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sequential_records))
    jsonl_path = DATA_V2_DIR / "decision_records_v2.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    qa_dir = DATA_V2_DIR / "visual_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    overview_png = qa_dir / "contact_sheet_7scenes_overview.png"
    render_contact_sheet(
        contact_overview_rows,
        overview_png,
        dpi=160,
        title="Phase-1 v2 Stratified Visual QA Contact Sheet (7 Scenes: Regimes A, B, C & Water)",
    )
    if contact_matched_rows:
        matched_png = qa_dir / "contact_sheet_matched_pairs.png"
        render_contact_sheet(
            contact_matched_rows,
            matched_png,
            dpi=160,
            title="Matched Hidden-State Twins: Identical Ground RGB vs Disjoint Optimal Branch Q",
        )

    summary = {
        "schema": "aerorescue.phase1.v2.summary",
        "audit_revision": 4,
        "total_records": len(records),
        "by_split": {
            sp: sum(1 for r in records if r["metadata"]["split"] == sp)
            for sp in ("train", "val", "test")
        },
        "by_sample_kind": {
            sk: sum(1 for r in records if r["metadata"]["sample_kind"] == sk)
            for sk in ("natural", "challenge_matched")
        },
        "by_regime": {
            rg: sum(1 for r in records if r["metadata"]["physical_regime"] == rg)
            for rg in ("A", "B", "C", "water")
        },
        "challenge_design": "single obstacle-factor swaps; fixed casualty; four eligible pairs per SC01/SC07/GT04",
        "challenge_prior": [0.5,0.5],
        "qa_total_checked": len(qa_reports),
        "qa_all_passed": all(x["passed"] for x in qa_reports),
        "clone_isolation_verified": clone_isolation_verified,
    }
    (DATA_V2_DIR / "dataset_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    (qa_dir / "qa_reports.json").write_text(
        json.dumps(qa_reports, ensure_ascii=False, indent=2) + "\n"
    )
    print("  [OK] Dataset v2 saved:", summary)
    return {"records": records, "summary": summary}


def run_pair0_experiments_and_diagnostics(dataset_bundle: dict) -> dict:
    """Extract real tokens, fit train-only parameters, evaluate actual methods and plot."""
    from aerorescue.iscc.audited_pair0 import run_audited_pair0
    (RUNS_LOCAL_DIR / "diagnostics").mkdir(parents=True, exist_ok=True)
    return run_audited_pair0(dataset_bundle["records"], RUNS_LOCAL_DIR, _plot_diagnostic_figures)


def _plot_diagnostic_figures(
    diag_dir: Path,
    srank_records: list[dict],
    phase_plane_points: list[dict],
    layer_drift_no_gate: list[dict] | None,
    layer_drift_gated: list[dict] | None,
    results_by_budget: dict,
    seq_phase_trace: list[dict] | None = None,
):
    """Render the 4 publication-grade diagnostic figures."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    raw_sv = np.array(srank_records[0]["raw_singular_values"][:32], dtype=np.float64)
    raw_sv = raw_sv / max(raw_sv[0], 1e-8)
    pull_sv = np.array(srank_records[0]["pullback_singular_values"][:16], dtype=np.float64)
    pull_sv = pull_sv / max(pull_sv[0], 1e-8)
    ax[0].plot(range(1, len(raw_sv) + 1), raw_sv, "o--", color="#556677", label=f"Raw Mapping $A_{{ref}}$ (srank={srank_records[0]['srank_raw']:.2f})")
    ax[0].plot(range(1, len(pull_sv) + 1), pull_sv, "s-", color="#D9480F", linewidth=2, label=f"Pullback $B_Y A_{{ref}}$ (srank={srank_records[0]['srank_pullback']:.2f})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Singular Value Index $\\ell$")
    ax[0].set_ylabel("Normalized Singular Value $\\sigma_\\ell / \\sigma_1$")
    ax[0].set_title("(a) Untruncated spectrum of rank-4 deployed head")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.3)

    budgets_x = [512, 1024, 2048, 4096]
    method_styles=[
        ("Ours (Agentic-Native ISCC)","#D55E00","o-"),("Ours-NoGate","#0072B2","s--"),
        ("Ours-NoWoodbury","#009E73","^--"),("Where2comm-Adapted","#CC79A7","d:"),
        ("Which2comm-Adapted","#E69F00","v:"),("RDcomm-Adapted","#56B4E9","x--"),
        ("BOLT-Adapted","#6F4C9B","+--"),("KVComm-Adapted","#8B6914","*:"),("Ego-Only","#444444",".-")]
    for m_name,col,mk in method_styles:
        if m_name not in results_by_budget["512"]["methods"]:continue
        means=[results_by_budget[str(b)]["methods"][m_name]["mean_regret"] for b in budgets_x]
        errors=[results_by_budget[str(b)]["methods"][m_name]["std_regret"]/np.sqrt(results_by_budget[str(b)]["test_samples"]) for b in budgets_x]
        ax[1].errorbar(budgets_x,means,yerr=errors,fmt=mk,capsize=2,color=col,lw=1.2,label=m_name)
    ax[1].set_xlabel("Total Over-the-Air Budget $B_t$ (Bytes/Step)")
    ax[1].set_ylabel("Physical Decision Regret $\\mathcal{R}_{\\mathrm{phys}} \\downarrow$")
    ax[1].set_title(f"(b) Test regret: mean and SEM (n={results_by_budget['512']['test_samples']})",fontsize=11)
    ax[1].legend(fontsize=6.5,ncol=2)
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "fig1_pullback_spectrum_and_srank.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.6), dpi=160)
    gx = np.array([p["grad_norm"] for p in phase_plane_points])
    sy = np.array([p["subspace_energy"] for p in phase_plane_points])
    ee = np.array([p["edge_energy"] for p in phase_plane_points])
    sc = ax.scatter(gx, sy, c=ee, cmap="viridis", s=26, alpha=0.82, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Image Edge Tensor Trace $\\mathrm{Tr}(\\Gamma_k)$")
    ax.axvline(np.median(gx), color="#D55E00", linestyle="--", linewidth=1.2, label="Observed median sensitivity")
    ax.axhline(np.median(sy), color="#1971C2", linestyle=":", linewidth=1.2, label="Subspace Projection Median")
    ax.set_xlabel("Spatial Feature-Pose Sensitivity $\\|D_{i,k} H_{f,k}^\\dagger G_{f,k}\\|_F$")
    ax.set_ylabel("Task Subspace Projection Energy $\\|X_T^\\top x_k\\|_2$")
    ax.set_title("Measured feature sensitivity and task energy", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(diag_dir / "fig2_spatial_gradient_dichotomy_plane.png", dpi=160)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),dpi=160)
    for rows,label,color,style in ((layer_drift_no_gate,"Ungated","#D55E00","o--"),(layer_drift_gated,"Gated","#009E73","s-")):
        if not rows:continue
        actual=[row for row in rows if row.get("full_attention_ratio") is not None]
        if actual:
            axes[0].plot([r["layer"] for r in actual],[r["full_attention_ratio"] for r in actual],style,color=color,label=label)
        axes[1].plot([r["layer"] for r in rows],[r.get("decision_sensitivity",0.) for r in rows],style,color=color,label=label)
    axes[0].set_title("Actual full self-attention")
    axes[0].set_ylabel("Remote / ground attention weight")
    axes[1].set_title("Decoder decision sensitivity")
    axes[1].set_ylabel("Gradient norm of route-logit contrast")
    for axis in axes:
        axis.set_xlabel("Decoder layer");axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(diag_dir/"fig3_decoder_layer_attention_drift.png",dpi=180);plt.close(fig)

    fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 4.2), dpi=160)
    if seq_phase_trace and len(seq_phase_trace) >= 4:
        steps = np.array([row["step"] for row in seq_phase_trace])
        bg_pct = np.array([row["bg_pct"] for row in seq_phase_trace])
        ba_pct = np.array([row["ba_pct"] for row in seq_phase_trace])
        bs_pct = np.array([row["bs_pct"] for row in seq_phase_trace])
        ax1.stackplot(
            steps, bg_pct, ba_pct, bs_pct,
            labels=["$b_g^*(t)$ Geometry Anchors", "$b_a^*(t)$ Representation Anchors", "$b_s^*(t)$ Blind Task Payload"],
            colors=["#F59F00", "#1971C2", "#2F9E44"], alpha=0.82,
        )
        for row in seq_phase_trace:
            if row.get("trigger_active_view") or row.get("zero_overlap"):
                ax1.axvline(row["step"], color="#E03131", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Continuous simulator step (RGB-D registration)")
    ax1.set_ylabel("Allocated uplink budget share (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Algorithm 1 continuous trajectory ($B_t=2048$ B)",pad=32,fontsize=11)
    ax1.set_xticks(range(1, 16))
    for lo, hi, label in ((1,3,"Bootstrap"),(4,8,"Cruise"),(9,11,"Loss/relock"),(12,15,"Recovery")):
        ax1.axvspan(lo-.5,hi+.5,color="#777777",alpha=.04)
        ax1.text((lo+hi)/2,101.5,label,ha="center",fontsize=8)
    ax1.set_xlim(.5,15.5)
    ax1.legend(loc="lower left", fontsize=8)
    finite_thresholds=[row.get("active_view_threshold") for row in (seq_phase_trace or []) if row.get("active_view_threshold") is not None]
    if finite_thresholds:
        reference=finite_thresholds[-1]
        ax2=ax1.twinx()
        risk=[row.get("rep_uncertainty_trace") for row in seq_phase_trace]
        ax2.plot(steps,[x/reference if x is not None else np.nan for x in risk],".-",color="#333333",lw=1,ms=3)
        ax2.axhline(1.,color="#333333",linestyle=":",lw=.8)
        ax2.set_yscale("log")
        ax2.set_ylabel("Uncertainty / calibrated cruise threshold")
    fig.tight_layout()
    fig.savefig(diag_dir / "fig4_ternary_budget_phase_transition.png", dpi=160)
    plt.close(fig)


def package_colab_bundle():
    """Package data/source for review; model-adapter availability remains explicit."""
    print("=== [Step 3] Packaging Colab 5-Pairs Suite under runs/colab_bundle/ ===")
    COLAB_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    colab_tools_dir = ROOT / "tools" / "colab"
    colab_tools_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "aerorescue.colab.5pairs.manifest/v1",
        "status": "audit_not_accepted_colab_execution_unimplemented",
        "dataset_path": "data/phase1_v2/decision_records_v2.jsonl",
        "pairs": [
            {
                "pair_id": "Pair_1_SameFamily_CrossScale",
                "sender_uav": "Qwen/Qwen2.5-VL-3B-Instruct",
                "receiver_ugv": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_i": 2048,
                "d_e": 3584,
            },
            {
                "pair_id": "Pair_2_CrossFamily_SmolVLM2_to_Qwen",
                "sender_uav": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                "receiver_ugv": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_i": 2048,
                "d_e": 3584,
            },
            {
                "pair_id": "Pair_3_CrossFamily_InternVL2_to_Qwen",
                "sender_uav": "OpenGVLab/InternVL2-2B",
                "receiver_ugv": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_i": 2048,
                "d_e": 3584,
            },
            {
                "pair_id": "Pair_4_SelfSupervised_DINOv2_to_Qwen",
                "sender_uav": "facebook/dinov2-large",
                "receiver_ugv": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_i": 1024,
                "d_e": 3584,
            },
            {
                "pair_id": "Pair_5_Contrastive_SigLIP2_to_Qwen",
                "sender_uav": "google/siglip2-so400m-patch14-384",
                "receiver_ugv": "Qwen/Qwen2.5-VL-7B-Instruct",
                "d_i": 1152,
                "d_e": 3584,
            },
        ],
        "budgets_bytes": [512, 1024, 2048, 4096, 8192],
        "regimes": ["A", "B", "C", "water"],
    }
    (COLAB_BUNDLE_DIR / "colab_5pairs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    import io
    import tarfile
    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k,v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
            return str(Path(value).relative_to(ROOT))
        return value
    archive = COLAB_BUNDLE_DIR / "aerorescue_phase1_colab_bundle.tar.gz"
    roots = [ROOT/"src"/"aerorescue", ROOT/"src"/"rescue_scenes", ROOT/"tools"/"colab",
             ROOT/"data"/"phase1_v2", ROOT/"runs"/"phase1_local"]
    explicit = [ROOT/"pyproject.toml",ROOT/"AGENTS.md",ROOT/"docs"/"PHASE1_LOCAL_EXECUTION_PLAN.md",
                ROOT/"tests"/"test_iscc_core_math.py",ROOT/"tests"/"test_phase1_audit_regressions.py",
                COLAB_BUNDLE_DIR/"colab_5pairs_manifest.json",ROOT/"tools"/"run_phase1_full_local.py",
                ROOT/"tools"/"audit_phase1_artifacts.py",ROOT/"tools"/"verify_phase1_bundle.py"]
    files = explicit + [p for base in roots for p in base.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".log")]
    with tarfile.open(archive,"w:gz") as tar:
        for path in sorted(set(files)):
            name=str(path.relative_to(ROOT))
            if path.suffix == ".jsonl":
                rows=[portable(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
                raw="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
                info=tarfile.TarInfo(name); info.size=len(raw)
                tar.addfile(info,io.BytesIO(raw))
            else:
                tar.add(path,arcname=name,recursive=False)
    print("  [OK] Review bundle packaged (audit not accepted):", archive)


def main():
    t0 = time.time()
    jsonl_path = DATA_V2_DIR / "decision_records_v2.jsonl"
    summary_path = DATA_V2_DIR / "dataset_summary_v2.json"
    if jsonl_path.is_file() and summary_path.is_file() and json.loads(summary_path.read_text()).get("audit_revision") == 4:
        records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for rec in records:
            S.validate_phase1_v2_record(rec)
        pairs={}
        for rec in records:
            group=rec["metadata"]["pair_group_id"]
            if group is not None:
                pairs.setdefault(group,[]).append(rec)
        for pair in pairs.values():
            assert len(pair)==2
            S.assert_matched_pair_noninterference(*pair)
        print(f"=== [Step 1] Loaded schema-checked regenerated dataset ({len(records)} records; research audit remains explicit) ===")
        ds_bundle = {"records": records, "summary": summary}
    else:
        ds_bundle = collect_phase1_v2_dataset()
    run_pair0_experiments_and_diagnostics(ds_bundle)
    package_colab_bundle()
    print(f"=== Phase-1 Local Pipeline Completed in {time.time() - t0:.1f}s ===")


if __name__ == "__main__":
    main()
