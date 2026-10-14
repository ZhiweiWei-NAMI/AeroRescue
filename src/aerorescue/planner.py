"""Use the actual AeroAgentSim bounded planner without copying its implementation."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def load_planner(platform: str | Path):
    path = Path(platform) / "src/aeroagentsim/packs/lowalt/rescue_planner.py"
    spec = importlib.util.spec_from_file_location("aerorescue_platform_planner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_plan, {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def all_bids(vehicles: list[dict], now_s: float) -> list[dict]:
    return [{"vehicle": v["id"], "roles": list(v["capabilities"]),
             "available_after_s": max(now_s, (v.get("commitment") or {}).get("busy_until_s", now_s)),
             "reason": "Unfiltered capability candidates"} for v in vehicles]
