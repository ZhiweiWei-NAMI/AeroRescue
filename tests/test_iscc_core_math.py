"""Unit tests for Agentic-Native ISCC mathematical operators (Contracts ③, ④, ⑤, ⑥, ⑦)."""
from __future__ import annotations

import numpy as np
from aerorescue.iscc import (
    FixedSubspaceEIVFilter,
    apply_positive_part_shrinkage_gate,
    compute_fisher_and_vertical_by,
    compute_homography_jacobian,
    compute_pose_ray_jacobian,
    compute_untruncated_pullback_srank,
    evaluate_safety_and_regret_metrics,
    finite_difference_check_jacobians,
    se3_exp,
    solve_ternary_budget_algorithm1,
    woodbury_correlated_precision,
)


def test_se3_left_jacobian_and_homography_finite_differences():
    K_i = np.array([[450.0, 0.0, 320.0], [0.0, 450.0, 192.0], [0.0, 0.0, 1.0]])
    K_e = np.array([[450.0, 0.0, 320.0], [0.0, 450.0, 192.0], [0.0, 0.0, 1.0]])
    T_e_from_i = se3_exp(np.array([0.15, -0.1, 0.08, 1.5, -0.8, 12.0]))
    p_i = np.array([2.0, -1.5, 25.0])
    n_i = np.array([0.1, -0.2, -0.97])

    res = finite_difference_check_jacobians(p_i, n_i, T_e_from_i, K_i, K_e)
    assert res["G_max_abs_err"] < 1e-5
    assert res["H_max_abs_err"] < 1e-4
    assert res["G_px"].shape == (2, 6)
    assert res["H_px"].shape == (2, 2)


def test_woodbury_identity_and_fixed_subspace_eiv_filter():
    rng = np.random.default_rng(7)
    d_i, d_e, r, p, m = 64, 96, 16, 16, 10
    U_src, _ = np.linalg.qr(rng.standard_normal((d_i, r)))
    U_rec, _ = np.linalg.qr(rng.standard_normal((d_e, p)))

    sigma_obs = 0.2
    M_dir = rng.standard_normal((m, 6)) * 0.5
    sigma_xi = np.diag([0.01, 0.01, 0.01, 0.25, 0.25, 0.25])

    Omega_direct_inv = np.linalg.inv((sigma_obs ** 2) * np.eye(m) + M_dir @ sigma_xi @ M_dir.T)
    Omega_woodbury_inv = woodbury_correlated_precision(sigma_obs ** 2, M_dir, sigma_xi)
    assert np.allclose(Omega_direct_inv, Omega_woodbury_inv, atol=1e-8)

    flt = FixedSubspaceEIVFilter(U_src=U_src, U_rec=U_rec, prior_precision=0.5, sigma_obs=sigma_obs)
    A_true = rng.standard_normal((p, r)) * 0.3
    X_A = rng.standard_normal((r, m))
    D_e_sub = rng.standard_normal((p, m, 2)) * 0.05
    G_f_list = rng.standard_normal((m, 2, 6)) * 0.2
    Y_A_sub = A_true @ X_A + rng.standard_normal((p, m)) * 0.02

    info = flt.step_b_woodbury_rls_update(X_A, Y_A_sub, D_e_sub, G_f_list, sigma_xi, M_pred_list=None)
    assert info["num_anchors"] == m
    assert info["woodbury_loss_trace"] >= 0.0


def test_vertical_by_srank_and_schur_algorithm1_and_gate():
    rng = np.random.default_rng(19)
    K, k_slots, d_e, r = 5, 3, 48, 16
    probs_0 = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
    # Low-rank Jacobian structure
    U_low, _ = np.linalg.qr(rng.standard_normal((d_e, 6)))
    J_Y_slots = np.einsum("kca,da->kcd", rng.standard_normal((k_slots, K, 6)), U_low)

    f_res = compute_fisher_and_vertical_by(probs_0, J_Y_slots, r_act=4)
    B_Y = f_res["B_Y"]
    assert B_Y.shape == (k_slots * K, d_e)
    S_ch = B_Y.T @ B_Y
    assert S_ch.shape == (d_e, d_e)

    A_ref = rng.standard_normal((d_e, 32))
    srank_res = compute_untruncated_pullback_srank(B_Y, A_ref)
    assert srank_res["srank_pullback"] < srank_res["srank_raw"]

    # Algorithm 1 Schur-complement solver test
    sigma_xi = np.eye(6) * 0.05
    P_pred = np.eye(r) * 1.0
    X_T = rng.standard_normal((r, k_slots))
    G_T_list = rng.standard_normal((k_slots, 2, 6)) * 0.1
    geom_cands = [
        {"G_f": rng.standard_normal((2, 6)), "Gamma_k": np.eye(2) * (i + 1), "p_i": np.array([1.0, 2.0, 20.0])}
        for i in range(8)
    ]
    rep_cands = [
        {
            "x_k": rng.standard_normal(r),
            "D_i_k": rng.standard_normal((r, 2)) * (0.02 if i < 6 else 2.0),
            "H_f_pinv": np.eye(2),
            "G_f": rng.standard_normal((2, 6)) * 0.2,
            "reliability_weight": 0.95,
        }
        for i in range(10)
    ]
    V_query = rng.standard_normal((r, 4))
    sol = solve_ternary_budget_algorithm1(
        total_budget_bytes=1024,
        sigma_xi_prior=sigma_xi,
        P_pred_prior=P_pred,
        X_T=X_T,
        G_T_list=G_T_list,
        geom_candidates=geom_cands,
        rep_candidates=rep_cands,
        mode="query",
        V_query=V_query,
        lambda_act=f_res["lambda_act"],
    )
    assert sol["byte_ledger"]["total_used_bytes"] <= 1024
    assert not sol["zero_overlap"]

    # Positive-part gate all-off test when noise is huge
    gate_off = apply_positive_part_shrinkage_gate(
        Y_hat_full=rng.standard_normal((d_e, k_slots)),
        U_act=f_res["U_act"],
        mu_pred=np.zeros((4, k_slots)),
        var_epistemic=np.ones((4, k_slots)) * 50.0,
        a_norm_sq=np.ones(4),
        sigma_q_sq=np.ones(k_slots),
        kappa_gate=1.25,
    )
    assert gate_off["all_off"] is True
    assert np.all(gate_off["Y_gated"] == 0.0)
