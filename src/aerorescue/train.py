"""Actual CPU optimization for a compact reference filter; not LLM fine-tuning."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .adapter import features, eligible, sigmoid
from .data import read_jsonl, write_json


def train(records_path: Path, output: Path, epochs: int = 300) -> dict:
    records = read_jsonl(records_path)
    groups = {}
    for record in records:
        previous = groups.setdefault(record["episode_id"], record["split"])
        if previous != record["split"]:
            raise ValueError("Episode appears in multiple splits")
    subsets = {split: [r for r in records if r["split"] == split and r["label"] is not None and eligible(r["input"], r["role"])]
               for split in ("train", "validation", "evaluation")}
    if not subsets["train"] or not subsets["validation"]:
        raise ValueError("Separate labeled training and validation episodes are required")
    xs = {split: np.stack([features(r["input"], r["role"]) for r in rows]) for split, rows in subsets.items() if rows}
    ys = {split: np.array([r["label"] for r in rows], dtype=float) for split, rows in subsets.items() if rows}
    if len(np.unique(ys["train"])) != 2:
        raise ValueError("Training requires positive and negative candidate examples")
    weights = np.zeros(xs["train"].shape[1])
    costs = np.where(ys["train"] == 1, 3.0, 1.0)
    losses = []
    start = time.perf_counter()
    for epoch in range(epochs):
        probabilities = sigmoid(xs["train"] @ weights)
        residual = (probabilities - ys["train"]) * costs
        regularization = 0.002 * weights
        regularization[0] = 0
        gradient = xs["train"].T @ residual / costs.sum() + regularization
        weights -= 0.2 * gradient
        if epoch == 0 or epoch == epochs - 1:
            loss = -np.average(ys["train"] * np.log(np.maximum(probabilities, 1e-12)) + (1 - ys["train"]) * np.log(np.maximum(1 - probabilities, 1e-12)), weights=costs)
            losses.append({"epoch": epoch, "weighted_log_loss": float(loss)})
    validation_scores = sigmoid(xs["validation"] @ weights)
    positives = ys["validation"] == 1
    if not positives.any():
        raise ValueError("Validation needs useful-role examples for a recall-oriented threshold")
    threshold = 0.01
    for candidate in np.arange(0.01, 0.96, 0.01):
        recall = float(np.mean(validation_scores[positives] >= candidate))
        if recall >= 0.99:
            threshold = float(candidate)
    result = {"format": "aerorescue.linear-role-filter.v1", "weights": weights.tolist(), "threshold": threshold,
              "model_kind": "hashed-text-and-structured-features logistic regression, CPU-trained from scratch",
              "training_source_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
              "epochs": epochs, "training_wall_s": time.perf_counter() - start, "loss": losses,
              "threshold_policy": "Largest fixed-grid threshold retaining >=99% useful candidates on validation only",
              "label_sources": sorted(set(r["label_source"] for r in records)),
              "counts": {split: len(rows) for split, rows in subsets.items()},
              "evaluation": {}}
    for split in xs:
        predicted = sigmoid(xs[split] @ weights) >= threshold
        positive = ys[split] == 1
        result["evaluation"][split] = {"accuracy": float(np.mean(predicted == ys[split])),
                                          "useful_role_recall": float(np.mean(predicted[positive])) if positive.any() else None,
                                          "retained_fraction": float(predicted.mean())}
    write_json(output, result)
    return {k: v for k, v in result.items() if k != "weights"}
