"""Pullback thin-SVD spectrum, empirically calibrated positive-part shrinkage gating,
and physical-regret evaluation operators.

Implements Contract ④ & Contract ⑥ from docs/PHASE1_LOCAL_EXECUTION_PLAN.md:
- Vertical stacking B_Y = [B_1; B_2; ...; B_k] in R^{(k K) x d_e} with B_j = F_theta^{1/2} J_{Y,j},
  satisfying B_Y^T B_Y = S_ch in R^{d_e x d_e}.
- Un-truncated stable rank srank(B_Y A_ref) = ||B_Y A_ref||_F^2 / ||B_Y A_ref||_2^2.
- Empirically calibrated positive-part shrinkage gate w_{ell, j}^{cal} with strict
  all-off short-circuit to z_ego.
- Same-root physical regret, negative transfer and communication coverage metrics.
"""
from __future__ import annotations

import numpy as np


def compute_fisher_and_vertical_by(
    probs_0: np.ndarray,
    J_Y_slots: np.ndarray,
    r_act: int = 4,
) -> dict:
    """Compute vertical-stacked pullback operator B_Y and active subspace U_act.

    Parameters
    ----------
    probs_0 : (K,) array
        Current receiver policy distribution pi_0 = pi_theta(. | h_t, v=0).
    J_Y_slots : (k, K, d_e) array
        Shadow Probe Jacobians d z_dec / d v_j at v=0 for k blind slots.
    r_act : int
        Number of query singular directions to retain for downlink query.
    """
    pi_0 = np.asarray(probs_0, dtype=np.float64).reshape(-1)
    pi_0 = np.clip(pi_0, 1e-8, 1.0)
    pi_0 = pi_0 / np.sum(pi_0)
    K = len(pi_0)

    # F_theta = diag(pi_0) - pi_0 pi_0^T; symmetric square root via eigendecomposition
    F_theta = np.diag(pi_0) - np.outer(pi_0, pi_0)
    evals, evecs = np.linalg.eigh(0.5 * (F_theta + F_theta.T))
    evals = np.maximum(evals, 0.0)
    F_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T  # (K, K)

    J_arr = np.asarray(J_Y_slots, dtype=np.float64)  # (k, K, d_e)
    k_slots, _, d_e = J_arr.shape

    # Vertically stack B_j = F_theta^{1/2} J_{Y, j} in R^{K x d_e} -> B_Y in R^{(k K) x d_e}
    B_blocks = [F_sqrt @ J_arr[j] for j in range(k_slots)]
    B_Y = np.vstack(B_blocks) if k_slots > 0 else np.zeros((K, d_e), dtype=np.float64)

    # Thin SVD: B_Y = U_left diag(s) V_right^T => S_ch = B_Y^T B_Y = V_right diag(s^2) V_right^T
    _, s_vals, Vt = np.linalg.svd(B_Y, full_matrices=False)
    eigenvalues_sch = s_vals ** 2
    r_eff = min(int(r_act), Vt.shape[0])
    U_act = Vt[:r_eff].T  # (d_e, r_act)
    lambda_act = eigenvalues_sch[:r_eff]

    return {
        "F_theta": F_theta,
        "B_Y": B_Y,
        "singular_values": s_vals,
        "eigenvalues_sch": eigenvalues_sch,
        "U_act": U_act,
        "lambda_act": lambda_act,
        "discarded_fisher_energy_fraction": float(np.sum(s_vals[r_eff:] ** 2) / max(np.sum(s_vals ** 2), 1e-30)),
    }


def compute_untruncated_pullback_srank(
    B_Y: np.ndarray,
    A_full: np.ndarray,
) -> dict:
    """Compute un-truncated stable rank of B_Y A_full and raw A_full (Diagnostic 1)."""
    s_raw = np.linalg.svd(np.asarray(A_full, dtype=np.float64), compute_uv=False)
    srank_raw = float(np.sum(s_raw ** 2) / max(float(s_raw[0] ** 2), 1e-12))

    BA = np.asarray(B_Y, dtype=np.float64) @ np.asarray(A_full, dtype=np.float64)
    s_pull = np.linalg.svd(BA, compute_uv=False)
    srank_pullback = float(np.sum(s_pull ** 2) / max(float(s_pull[0] ** 2), 1e-12))
    return {
        "srank_raw": srank_raw,
        "srank_pullback": srank_pullback,
        "raw_singular_values": s_raw.tolist(),
        "pullback_singular_values": s_pull.tolist(),
    }


def apply_positive_part_shrinkage_gate(
    Y_hat_full: np.ndarray,
    U_act: np.ndarray,
    mu_pred: np.ndarray,
    var_epistemic: np.ndarray,
    a_norm_sq: np.ndarray,
    sigma_q_sq: np.ndarray,
    kappa_gate: float = 1.25,
    max_total_uncertainty: float = float("inf"),
) -> dict:
    """Empirically calibrated positive-part shrinkage gate (Contract ⑥).

    w_{ell, j}^{cal} = max(0, 1 - kappa_gate * sigma_{ell, j}^2 / (mu_{ell, j}^2 + eps))
    When all w == 0, returns exact zeros so the caller short-circuits to z_ego.
    """
    d_e, k = Y_hat_full.shape
    r_act = U_act.shape[1]
    if k == 0 or r_act == 0:
        return {
            "Y_gated": np.zeros_like(Y_hat_full),
            "weights": np.zeros((r_act, k), dtype=np.float64),
            "all_off": True,
        }

    # Total noise variance along active direction ell for slot j
    noise_var = np.asarray(var_epistemic, dtype=np.float64) + np.outer(
        np.asarray(a_norm_sq, dtype=np.float64).reshape(r_act),
        np.asarray(sigma_q_sq, dtype=np.float64).reshape(k),
    )
    sig_sq = np.asarray(mu_pred, dtype=np.float64) ** 2

    # Positive-part empirical Bayes shrinkage
    raw_ratio = 1.0 - (float(kappa_gate) * noise_var) / (sig_sq + 1e-8)
    weights = np.clip(raw_ratio, 0.0, 1.0)
    # Zero out slots whose epistemic variance exceeds safety ceiling
    weights[var_epistemic > float(max_total_uncertainty)] = 0.0

    all_off = bool(np.all(weights == 0.0))
    if all_off:
        return {
            "Y_gated": np.zeros_like(Y_hat_full),
            "weights": weights,
            "all_off": True,
        }

    # Project onto U_act, scale by w_{ell, j}, and lift back to R^{d_e}
    proj_coeffs = U_act.T @ np.asarray(Y_hat_full, dtype=np.float64)  # (r_act, k)
    Y_gated = U_act @ (weights * proj_coeffs)
    return {
        "Y_gated": Y_gated,
        "weights": weights,
        "all_off": False,
    }


def evaluate_safety_and_regret_metrics(
    q_values_matrix: np.ndarray,
    ego_Selected_idx: np.ndarray,
    method_selected_idx: np.ndarray,
    active_comm_flags: np.ndarray,
    bytes_used: np.ndarray,
    ntr_tolerance: float = 1e-4,
) -> dict:
    """Compute Physical Regret, CVaR-10% Regret, NTR (%), and Coverage (%) (Contract ⑥)."""
    n_samples = q_values_matrix.shape[0]
    q_max = np.max(q_values_matrix, axis=1)
    q_ego = q_values_matrix[np.arange(n_samples), ego_Selected_idx]
    q_method = q_values_matrix[np.arange(n_samples), method_selected_idx]

    regrets = q_max - q_method
    ego_regrets = q_max - q_ego

    # Worst 10% CVaR of physical regret
    k_tail = max(1, int(np.ceil(0.10 * n_samples)))
    sorted_regrets = np.sort(regrets)[::-1]
    cvar_10 = float(np.mean(sorted_regrets[:k_tail]))

    # Paired negative transfer rate relative to Ego on the exact same root state
    ntr = float(np.mean((q_method < q_ego - float(ntr_tolerance)).astype(np.float64)) * 100.0)
    coverage = float(np.mean(np.asarray(active_comm_flags, dtype=np.float64)) * 100.0)

    return {
        "mean_regret": float(np.mean(regrets)),
        "std_regret": float(np.std(regrets)),
        "ego_mean_regret": float(np.mean(ego_regrets)),
        "cvar_10_regret": cvar_10,
        "negative_transfer_rate_pct": ntr,
        "active_coverage_pct": coverage,
        "mean_bytes": float(np.mean(bytes_used)),
        "top1_optimal_acc_pct": float(np.mean((regrets <= 1e-6).astype(np.float64)) * 100.0),
    }
