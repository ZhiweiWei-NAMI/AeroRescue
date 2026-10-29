"""Deterministic abstract rescue simulation components (no perception backend)."""

# Keep imports light: the public Environment constructs the cross-lane components.
__all__ = ["kernel", "ledger", "network", "compute", "actions", "evaluation"]
