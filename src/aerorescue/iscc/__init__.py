"""Agentic-Native ISCC core mathematical, filtering, and budget allocation package."""
from .geometry import (
    compute_homography_jacobian,
    compute_pose_ray_jacobian,
    finite_difference_check_jacobians,
    j_proj,
    j_se3_left,
    project_point,
    se3_adjoint,
    se3_exp,
    so3_exp,
    transport_pose_covariance,
)
from .eiv_filter import FixedSubspaceEIVFilter, woodbury_correlated_precision
from .budget_solver import (
     compute_downlink_bytes,
    quantize_and_dequantize_blind_features,
    select_d_optimal_geometry_anchors,
    solve_ternary_budget_algorithm1,
)
from .gating_and_baselines import (
    apply_positive_part_shrinkage_gate,
    compute_fisher_and_vertical_by,
    compute_untruncated_pullback_srank,
    evaluate_safety_and_regret_metrics,
)

__all__ = [
    "compute_homography_jacobian",
    "compute_pose_ray_jacobian",
    "finite_difference_check_jacobians",
    "j_proj",
    "j_se3_left",
    "project_point",
    "se3_adjoint",
    "se3_exp",
    "so3_exp",
    "transport_pose_covariance",
    "FixedSubspaceEIVFilter",
    "woodbury_correlated_precision",
    "compute_downlink_bytes",
    "quantize_and_dequantize_blind_features",
    "select_d_optimal_geometry_anchors",
    "solve_ternary_budget_algorithm1",
    "apply_positive_part_shrinkage_gate",
    "compute_fisher_and_vertical_by",
    "compute_untruncated_pullback_srank",
    "evaluate_safety_and_regret_metrics",
]
