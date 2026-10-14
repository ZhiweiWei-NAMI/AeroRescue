"""Paired bounded-planner experiments, explicitly separate from UAV execution."""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict

from .adapter import ParticipationFilter, eligible


def compare(episodes: list[dict], model: ParticipationFilter, solve) -> dict:
    rows, totals = [], defaultdict(lambda: defaultdict(float))
    for episode in episodes:
        if episode["split"] != "evaluation":
            continue
        problem = episode["problem"]
        start = time.perf_counter()
        learned = [{**model.propose(r), "vehicle": r["self"]["id"]} for r in episode["requests"]]
        inference_s = time.perf_counter() - start
        start = time.perf_counter()
        rules = [{"vehicle": r["self"]["id"], "roles": [], "available_after_s": max(r["now_s"], (r["self"].get("commitment") or {}).get("busy_until_s", r["now_s"])), "reason": "Distance rule at the same per-role candidate budget"} for r in episode["requests"]]
        for role in ("inspect", "relay", "deliver"):
            count = sum(role in bid["roles"] for bid in learned)
            ranked = sorted((i for i, request in enumerate(episode["requests"]) if eligible(request, role)),
                            key=lambda i: math.dist(episode["requests"][i]["self"]["position"], episode["requests"][i].get("locations", {}).get(role, episode["requests"][i]["self"]["position"])))
            for i in ranked[:count]:
                rules[i]["roles"].append(role)
        rule_s = time.perf_counter() - start
        start = time.perf_counter()
        complete = [{**bid, "roles": [role for role in ("inspect", "relay", "deliver") if eligible(request, role)],
                     "reason": "All physically eligible capabilities"} for bid, request in zip(rules, episode["requests"])]
        candidate_s = time.perf_counter() - start
        methods = {"all_candidates": complete, "learned_filter": learned, "distance_same_count": rules}
        filter_times = {"all_candidates": candidate_s, "learned_filter": inference_s, "distance_same_count": rule_s}
        results = {}
        for name, bids in methods.items():
            start = time.perf_counter()
            plan = solve(**{**problem, "bids": bids})
            solve_s = time.perf_counter() - start
            candidates = sum(len(bid["roles"]) for bid in bids)
            results[name] = {"status": plan["status"], "objective_s": plan["objective_s"], "evaluated": plan["evaluated"],
                             "candidates": candidates, "solver_wall_s": solve_s,
                             "inference_wall_s": inference_s if name == "learned_filter" else 0.0,
                             "filter_wall_s": filter_times[name],
                             "serialized_bid_bytes": len(json.dumps(bids, ensure_ascii=False).encode())}
            total = totals[name]
            total["episodes"] += 1
            total["feasible"] += plan["status"] == "feasible"
            total["candidates"] += candidates
            total["solver_wall_s"] += solve_s
            total["inference_wall_s"] += results[name]["inference_wall_s"]
            total["filter_wall_s"] += filter_times[name]
            total["evaluated"] += plan["evaluated"]
            if results["all_candidates"]["status"] == "feasible":
                total["full_feasible"] += 1
                total["feasible_retained"] += plan["status"] == "feasible"
                if plan["status"] == "feasible":
                    total["objective_ratio_sum"] += plan["objective_s"] / max(results["all_candidates"]["objective_s"], 1e-9)
        rows.append({"episode_id": episode["episode_id"], "methods": results})
    summary = {}
    for name, total in totals.items():
        summary[name] = {**total,
                         "feasible_coalition_retention": total["feasible_retained"] / total["full_feasible"] if total["full_feasible"] else None,
                         "total_decision_wall_s": total["filter_wall_s"] + total["solver_wall_s"],
                         "mean_makespan_ratio_on_retained": total["objective_ratio_sum"] / total["feasible_retained"] if total["feasible_retained"] else None}
    return {"scope": "Paired bounded-planner runs on held-out procedural episodes. No actual flight, visual perception, ns-3, LLM or mission-success claim.",
            "artifact": {k: model.artifact[k] for k in ("format", "threshold", "training_source_sha256")}, "summary": summary, "episodes": rows}
