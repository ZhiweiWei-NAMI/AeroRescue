"""Exact run identification for saved evidence: source revision + timestamp."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_revision() -> str:
    """Short git commit of env-v0.5, suffixed '+dirty' if rescue_env/ has uncommitted edits."""
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "rescue_env"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head or "uncommitted") + ("+dirty" if dirty else "")


def stamp() -> dict:
    return {"source_revision": source_revision(), "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
