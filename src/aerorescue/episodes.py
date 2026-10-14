"""Task problems and counterfactual role labels, with explicit provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from .data import write_jsonl
from .planner import all_bids

ROLES = ("inspect", "relay", "deliver")
MISSIONS = {
    "train": ["Suspected stranded person; inspect, relay the report, then deliver supplies if confirmed.",
              "Confirmed person needs supplies; damaged communication requires a continuous relay.",
              "Search an unconfirmed location and relay findings; no supply delivery requested.",
              "Confirmed recipient needs supplies; direct communication is available."],
    "validation": ["Check a possible survivor and forward the observation over a relay; supply delivery depends on confirmation.",
                   "Positive survivor report received. Provide a communication relay while bringing supplies.",
                   "Observe the search location and maintain relay communication. This is a search-only request.",
                   "Bring supplies to the confirmed recipient using the working direct link."],
    "evaluation": ["A person may need help. Verify first; bridge the broken radio link and deliver only after a positive report.",
                   "The person was verified. Keep a relay active until the supply drop is acknowledged.",
                   "Collect and relay evidence at the suspected location without a delivery operation.",
                   "A verified person awaits a supply drop. The direct radio link is operational."]}


def generate(root: Path, count: int = 128, seed: int = 7) -> list[dict]:
    """Synthetic rescue extensions grounded only in published map coordinates/priors.

    The source heatmap does not assert a real survivor, flood, payload need or outcome.
    Whole maps, language templates and capability configurations are split apart.
    """
    episodes = []
    for split, map_id, n in [("train", "1", count), ("validation", "4", max(16, count // 4)), ("evaluation", "6", max(16, count // 4))]:
        directory = root / "sarenv_dataset" / map_id
        geo = json.loads((directory / "features.geojson").read_text())
        heatmap = np.load(directory / "heatmap.npy", allow_pickle=False)
        rng = np.random.default_rng(seed + int(map_id))
        probabilities = heatmap.ravel() / heatmap.sum()
        for index in range(n):
            family = index % len(MISSIONS[split])
            cell = int(rng.choice(heatmap.size, p=probabilities))
            row, column = np.unravel_index(cell, heatmap.shape)
            origin = [(geo["bounds"][0] + geo["bounds"][2]) / 2, (geo["bounds"][1] + geo["bounds"][3]) / 2]
            target = [geo["bounds"][0] + (column + 0.5) * geo["meter_per_bin"] - origin[0],
                      geo["bounds"][1] + (row + 0.5) * geo["meter_per_bin"] - origin[1], 40.0]
            while True:
                capabilities = [sorted(role for role in ROLES if rng.random() < 0.62) for _ in range(4)]
                signature = hashlib.sha256(json.dumps(sorted(capabilities)).encode()).hexdigest()
                partition = int(signature[:8], 16) % 5
                if (split == "train" and partition >= 2) or (split == "validation" and partition == 0) or (split == "evaluation" and partition == 1):
                    break
            deadline = float(rng.uniform(100, 350))
            vehicles = []
            for i, caps in enumerate(capabilities):
                busy = float(rng.choice([0, 0, 0, rng.uniform(30, 400)]))
                vehicles.append({"id": f"drone-{i}", "position": [target[0] + float(rng.uniform(-700, 700)), target[1] + float(rng.uniform(-700, 700)), 40.0],
                                 "capabilities": caps, "soc": float(rng.uniform(0.1, 1.0)),
                                 "payload_kg": float(rng.choice([0, 1, 2])), "payload_capacity_kg": 2.0,
                                 "speed_m_s": float(rng.uniform(8, 20)),
                                 "commitment": {"task": "prior-supply-task", "protected": True, "busy_until_s": busy} if busy else None})
            needed = [("relay", "inspect", "deliver"), ("relay", "deliver"), ("relay", "inspect"), ("deliver",)][family]
            locations = {"inspect": target, "deliver": target, "relay": [target[0] - 80, target[1] - 80, 60.0]}
            durations = {"inspect": 5.0, "relay": 0.0, "deliver": 3.0}
            actions = [{"id": role, "role": role, "target": locations[role], "duration_s": durations[role],
                        "after": (["inspect"] if role == "deliver" and "inspect" in needed else []) + (["relay"] if role in {"inspect", "deliver"} and "relay" in needed else [])} for role in needed]
            graph = {"actions": actions, "deadline_s": deadline, "payload_kg": 1.0}
            episode_id = f"sarenv-{map_id}-{split}-{index:04d}"
            common = {"event": "local-incident", "mission": MISSIONS[split][family], "evidence": "positive" if family in {1, 3} else "suspected",
                      "now_s": 0.0, "deadline_s": deadline, "payload_kg": 1.0, "locations": locations, "duration_s": durations}
            episodes.append({"episode_id": episode_id, "map_id": f"sarenv:{map_id}", "template_id": f"{split}:{family}",
                             "role_combination_id": signature, "split": split,
                             "source_kind": "procedural_task_extension_on_published_sarenv_map", "complete_role_horizon": True,
                             "source_cell": [int(row), int(column)], "origin_utm_m": origin,
                             "requests": [{**common, "self": vehicle} for vehicle in vehicles],
                             "problem": {"vehicles": vehicles, "bids": all_bids(vehicles, 0.0), "graph": graph, "now_s": 0.0}})
    return episodes


def label_episode(episode: dict, solve, planner_source: dict, slack: float = 0.2) -> list[dict]:
    """A positive candidate participates in some near-optimal bounded plan.

    Force-in and remove-one solves are separate from the chosen winner. Infeasible
    base problems and repeated-role horizons remain unknown, never false negatives.
    """
    problem = deepcopy(episode["problem"])
    problem["bids"] = deepcopy(episode.get("candidate_universe", problem["bids"])) if episode["source_kind"] == "actual_platform_run" else all_bids(problem["vehicles"], problem["now_s"])
    full = solve(**problem)
    role_counts = {role: sum(a["role"] == role for a in problem["graph"]["actions"]) for role in ROLES}
    outputs = []
    for request in episode["requests"]:
        identity = request["self"]["id"]
        aligned = request["now_s"] == problem["now_s"]
        for role in ROLES:
            forced, removed, label, critical = None, None, None, None
            if not aligned:
                pass  # A later physical/evidence snapshot is not the old local observation.
            elif role_counts[role] == 0:
                if episode.get("complete_role_horizon"):
                    label = 0
            elif full["status"] == "feasible" and role_counts[role] == 1:
                keep = deepcopy(problem)
                drop = deepcopy(problem)
                for bid in keep["bids"]:
                    if bid["vehicle"] != identity:
                        bid["roles"] = [r for r in bid["roles"] if r != role]
                for bid in drop["bids"]:
                    if bid["vehicle"] == identity:
                        bid["roles"] = [r for r in bid["roles"] if r != role]
                forced, removed = solve(**keep), solve(**drop)
                limit = full["objective_s"] * (1 + slack) + 1e-8
                label = int(forced["status"] == "feasible" and forced["objective_s"] <= limit)
                critical = removed["status"] != "feasible"
            outputs.append({**{k: episode[k] for k in ("episode_id", "map_id", "template_id", "role_combination_id", "split", "source_kind")},
                            "input": request, "role": role, "label": label, "critical": critical,
                            "label_source": "bounded_planner_counterfactual", "near_optimal_slack": slack,
                            "label_status": "aligned" if aligned else "stale_local_observation_unknown",
                            "planner": planner_source, "base_status": full["status"], "base_objective_s": full["objective_s"],
                            "forced_objective_s": None if forced is None else forced["objective_s"],
                            "removed_status": None if removed is None else removed["status"],
                            "label_scope": "role-candidate usefulness, not mission-success probability"})
    return outputs


def export_events(events: list[dict], run_id: str, split: str, map_id: str) -> list[dict]:
    """Export normalized actual platform events in order, retaining local inputs.

    Each row is {schema, payload:{data_json:...}}. The platform's plan_computed
    problem snapshot is mandatory; reconstructing missing physical state is unsafe.
    """
    requests, episodes = {}, []
    for row in events:
        schema = row.get("schema", row.get("schema_id"))
        payload = row.get("payload", {})
        data = json.loads(payload["data_json"]) if "data_json" in payload else payload
        if schema == "rescue.local.request":
            requests[(data["event"], data["self"]["id"])] = data
        elif schema == "rescue.event" and data.get("kind") == "plan_computed":
            if "problem" not in data:
                raise ValueError("plan_computed lacks actual problem snapshot; use updated platform exporter")
            event = data["event"]
            local = [request for (event_id, _), request in requests.items() if event_id == event]
            if not local:
                raise ValueError("No recorded local observation for plan_computed")
            episodes.append({"episode_id": f"{run_id}:{event}:{len(episodes)}", "map_id": map_id,
                             "template_id": "platform:" + event, "role_combination_id": "platform-recorded",
                             "split": split, "source_kind": "actual_platform_run",
                             "complete_role_horizon": False, "requests": local, "problem": data["problem"],
                             "candidate_universe": data.get("candidate_universe", data["problem"]["bids"]),
                             "observed_plan": data["plan"]})
    if not episodes:
        raise ValueError("No recorded platform decision episode was exportable")
    return episodes


def read_platform_events(directory: Path, platform: Path) -> list[dict]:
    """Read committed indexed WAL messages with the platform's own codecs."""
    sys.path[:0] = [str(platform.resolve() / "src"), str(platform.resolve() / "aerokernel")]
    from aerokernel.codec import decode_record
    from aerokernel.values import thaw
    from aeroagentsim.services.storage import RunStorage, committed_prefix
    storage, seen, events = RunStorage(directory), set(), []
    prefix = committed_prefix(directory)
    final = prefix["index"] if prefix else storage.entries[-1]["index"]
    cursor = 1
    while cursor <= final:
        records = storage.records(cursor, min(1000, final - cursor + 1))
        if not records:
            break
        for record in records:
            for item in record["items"]:
                if "message" not in item:
                    continue
                message = decode_record(item["message"])
                if message.schema_id.startswith("rescue.") and message.id not in seen:
                    seen.add(message.id)
                    events.append({"schema": message.schema_id, "payload": thaw(message.payload),
                                   "message_id": message.id, "at_ns": message.at.ns, "commit_index": record["index"]})
        cursor = records[-1]["index"] + 1
    return events
