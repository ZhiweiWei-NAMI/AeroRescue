"""Small learned filter: local observations only; physical gates stay explicit."""
from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROLES = ("inspect", "relay", "deliver")
HASH_DIM = 256
NUMERIC_DIM = 17


@lru_cache(maxsize=1)
def _qwen_scorer(path: str):
    from .peft_train import QwenRoleScorer
    return QwenRoleScorer(path)


def eligible(request: dict, role: str) -> bool:
    vehicle = request["self"]
    if role not in vehicle["capabilities"] or vehicle["soc"] < 0.2 or vehicle["speed_m_s"] <= 0:
        return False
    if role == "deliver" and not request.get("payload_kg", 1) <= vehicle["payload_kg"] <= vehicle["payload_capacity_kg"]:
        return False
    ready = max(request["now_s"], (vehicle.get("commitment") or {}).get("busy_until_s", request["now_s"]))
    return math.isfinite(ready) and ready <= request.get("deadline_s", math.inf)


def features(request: dict, role: str) -> np.ndarray:
    vehicle, now = request["self"], request["now_s"]
    commitment = vehicle.get("commitment") or {}
    busy = max(0.0, commitment.get("busy_until_s", now) - now)
    target = request.get("locations", {}).get(role)
    travel = math.dist(vehicle["position"], target) / max(vehicle["speed_m_s"], 0.01) if target else 0.0
    deadline = max(0.0, request.get("deadline_s", now + 300) - now)
    x = np.zeros(NUMERIC_DIM + HASH_DIM, dtype=np.float64)
    x[:NUMERIC_DIM] = [1.0, *[float(role == r) for r in ROLES], vehicle["soc"],
                       min(vehicle["payload_kg"] / 2, 2), min(busy / 300, 4), min(travel / 100, 4),
                       min(deadline / 300, 4), max(-4, min(4, (deadline - busy - travel) / 100)),
                       float(bool(commitment.get("protected"))), float(target is not None),
                       *[float(r in vehicle["capabilities"]) for r in ROLES],
                       float(request.get("evidence") == "positive"), float(request.get("evidence") == "suspected")]
    words = re.findall(r"[a-z]+|[\u4e00-\u9fff]", str(request.get("mission", "")).lower())
    tokens = set(words + [" ".join(words[i:i + 2]) for i in range(len(words) - 1)])
    for token in tokens:
        value = int.from_bytes(hashlib.blake2b(f"{role}:{token}".encode(), digest_size=8).digest(), "big")
        x[NUMERIC_DIM + value % HASH_DIM] += 1.0 / max(1.0, math.sqrt(len(tokens)))
    return x


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))


class ParticipationFilter:
    def __init__(self, artifact: dict):
        if artifact.get("format") != "aerorescue.linear-role-filter.v1":
            raise ValueError("Unsupported participation artifact format")
        self.artifact = artifact
        self.weights = np.asarray(artifact["weights"], dtype=np.float64)
        if self.weights.shape != (NUMERIC_DIM + HASH_DIM,) or not np.isfinite(self.weights).all():
            raise ValueError("Invalid learned filter dimensions/weights")
        self.threshold = float(artifact["threshold"])

    @classmethod
    def load(cls, path: str | Path) -> "ParticipationFilter":
        if Path(path).is_dir():
            instance = cls.__new__(cls)
            instance._qwen = _qwen_scorer(str(Path(path).resolve()))
            instance.threshold = instance._qwen.threshold
            return instance
        return cls(json.loads(Path(path).read_text()))

    def score(self, request: dict, role: str) -> float:
        if not eligible(request, role):
            return 0.0
        if hasattr(self, "_qwen"):
            return self._qwen.predict(request, role)
        return float(sigmoid(features(request, role) @ self.weights))

    def propose(self, request: dict) -> dict:
        scores = {role: self.score(request, role) for role in ROLES}
        roles = [role for role in ROLES if eligible(request, role) and scores[role] >= self.threshold]
        available = max(request["now_s"], (request["self"].get("commitment") or {}).get("busy_until_s", request["now_s"]))
        return {"roles": roles, "available_after_s": available,
                "reason": "Learned bounded-planner participation filter; delivery remains conditional on received positive evidence. " + json.dumps(scores, sort_keys=True)}
