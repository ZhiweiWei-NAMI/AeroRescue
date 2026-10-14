"""Small, pinned official SAREnv subset; no whole archive or model downloads."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

SOURCE = "https://github.com/namurproject/SAREnv"
REVISION = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
RAW = f"https://raw.githubusercontent.com/namurproject/SAREnv/{REVISION}/"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _download(relative: str, destination: Path) -> dict:
    url, expected = RAW + relative, None
    with urlopen(url, timeout=90) as response:
        first = response.read(1024)
        if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
            pointer = dict(line.split(" ", 1) for line in first.decode().splitlines())
            expected = pointer["oid"].removeprefix("sha256:")
            if int(pointer["size"]) > 64 * 1024 * 1024:
                raise ValueError("Selected file exceeds the small-subset 64 MiB limit")
            url = f"https://media.githubusercontent.com/media/namurproject/SAREnv/{REVISION}/{relative}"
        else:
            content = first + response.read(64 * 1024 * 1024)
    if expected:
        with urlopen(url, timeout=90) as response:
            content = response.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024 or content.startswith(b"version https://git-lfs"):
        raise ValueError("Oversized file or unresolved Git LFS pointer")
    sha = hashlib.sha256(content).hexdigest()
    if expected and sha != expected:
        raise ValueError("Downloaded bytes do not match the official LFS SHA256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {"path": relative, "url": url, "bytes": len(content), "sha256": sha}


def acquire(root: Path, ids: list[str]) -> dict:
    if not ids or any(not value.isdigit() or not 1 <= int(value) <= 60 for value in ids):
        raise ValueError("Select SAREnv environment IDs 1 through 60")
    manifest = {"dataset": "SAREnv", "source": SOURCE, "revision": REVISION,
                "acquired_utc": datetime.now(timezone.utc).isoformat(),
                "license": "Repository MIT; underlying OpenStreetMap data retain ODbL attribution",
                "osm_attribution": "© OpenStreetMap contributors, https://www.openstreetmap.org/copyright",
                "semantics": "Published real-world map features and generated lost-person probability priors; not flood imagery or rescue participation labels.",
                "files": []}
    for name in ["LICENSE", "README.md", *[f"sarenv_dataset/{i}/{f}" for i in ids for f in ("features.geojson", "heatmap.npy")]]:
        manifest["files"].append(_download(name, root / name))
    write_json(root / "source.json", manifest)
    return manifest


def prepare(root: Path, output: Path) -> dict:
    source = json.loads((root / "source.json").read_text())
    environments = []
    for directory in sorted((root / "sarenv_dataset").iterdir()):
        if not directory.is_dir():
            continue
        geo = json.loads((directory / "features.geojson").read_text())
        heatmap = np.load(directory / "heatmap.npy", allow_pickle=False)
        if heatmap.ndim != 2 or not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.sum() <= 0:
            raise ValueError(f"Invalid probability map: {directory}")
        environment = {"environment_id": directory.name, "center_point": geo["center_point"],
                       "bounds_utm_m": geo["bounds"], "meter_per_bin": geo["meter_per_bin"],
                       "climate": geo.get("climate"), "environment_type": geo.get("environment_type"),
                       "feature_count": len(geo["features"]),
                       "feature_types": dict(Counter(str(f["properties"].get("feature_type")) for f in geo["features"])),
                       "heatmap_shape": list(heatmap.shape), "probability_mass": float(heatmap.sum()),
                       "nonzero_cells": int(np.count_nonzero(heatmap))}
        environments.append(environment)
    result = {"source": source, "environments": environments,
              "usage": "Map/observation context only. UAV capabilities, commitments, events, and planner labels are separately generated extensions."}
    write_json(output, result)
    return result
