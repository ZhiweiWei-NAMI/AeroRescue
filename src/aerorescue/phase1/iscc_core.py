"""Agentic-Native ISCC core operators (pure NumPy, vectorised).

Four operators from ``docs/MAINLINE_AGENTIC_ISCC_BLUEPRINT.md``:

* :func:`se3_projection_jacobian` -- eq. 24-28, ``G_k = J_proj(p) J_se3(p)``.
* :func:`homography_jacobian` / :func:`homography_condition_numbers` -- eq. 30,
  the cross-view patch homography Jacobian and its condition number.
* :func:`woodbury_eiv_posterior_precision` -- theorem 1, eq. 62, the exact
  posterior precision with the cross-anchor rank-6 coherent pose downdate.
* :func:`solve_ternary_budget_alg1` -- Algorithm 1, eq. 111-125, outer
  geometric-budget scan + D-optimal ``S_g`` + Sherman-Morrison greedy ``S_a``.
* :func:`matrix_wiener_shrinkage_gate` -- theorem 3, eq. 88-92, closed-form
  per-channel Wiener weights with the strict zero-negative-transfer property.

Anti-bloat: no classes, no abstraction layers, no SciPy.  Every function is a
short matrix expression that a reviewer can check against the blueprint.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------- geometry

def _points(points):
    array = np.asarray(points, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must be (m, 3)")
    return array


def _skew(points):
    """Batched skew matrices ``[p]_x`` of shape (m, 3, 3)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    out = np.zeros((points.shape[0], 3, 3), dtype=float)
    out[:, 0, 1], out[:, 0, 2] = -z, y
    out[:, 1, 0], out[:, 1, 2] = z, -x
    out[:, 2, 0], out[:, 2, 1] = -y, x
    return out


def se3_projection_jacobian(points_e, fx, fy):
    """Pixel displacement Jacobian ``G_k = J_proj(p_k^e) J_se3(p_k^e)``.

    ``points_e`` are the anchor coordinates in the receiving (ground) camera
    frame, ``Z > 0``.  Returns ``(m, 2, 6)``; column order is
    ``[omega_x, omega_y, omega_z, upsilon_x, upsilon_y, upsilon_z]``.
    Eq. 24/28 expresses perturbations in the receiving frame (left action).
    A source-frame right perturbation requires the corresponding SE(3) adjoint.

    The companion cross-view patch homography Jacobian and its condition number
    are :func:`homography_jacobian` / :func:`homography_condition_numbers`.
    """
    points = _points(points_e)
    if np.any(points[:, 2] <= _EPS):
        raise ValueError("anchor depth Z must be positive")
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    count = points.shape[0]
    jac_proj = np.zeros((count, 2, 3), dtype=float)
    jac_proj[:, 0, 0] = fx / z
    jac_proj[:, 0, 2] = -fx * x / z ** 2
    jac_proj[:, 1, 1] = fy / z
    jac_proj[:, 1, 2] = -fy * y / z ** 2
    jac_se3 = np.concatenate(
        [-_skew(points), np.broadcast_to(np.eye(3), (count, 3, 3))], axis=2)
    return np.einsum("mij,mjk->mik", jac_proj, jac_se3)


def homography_jacobian(points_e, points_i, normals, rotation, k_i, depths, fx, fy):
    """Cross-view patch homography Jacobian ``H_{e<-i}(k)`` of shape (m, 2, 2).

    ``H = J_proj(p_e) R (I - p_i n^T / (n^T p_i)) K_i^{-1} d``.  ``depths`` is
    the per-anchor source depth ``d_k``.
    """
    p_e = _points(points_e)
    p_i = _points(points_i)
    n = _points(normals)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    k_i = np.asarray(k_i, dtype=float).reshape(3, 3)
    d = np.asarray(depths, dtype=float).reshape(-1)
    if not (p_e.shape == p_i.shape == n.shape) or d.shape[0] != p_e.shape[0]:
        raise ValueError("anchor arrays must share the same (m, 3) shape")
    x, y, z = p_e[:, 0], p_e[:, 1], p_e[:, 2]
    if np.any(z <= _EPS):
        raise ValueError("anchor depth Z must be positive")
    count = p_e.shape[0]
    jac_proj = np.zeros((count, 2, 3), dtype=float)
    jac_proj[:, 0, 0] = fx / z
    jac_proj[:, 0, 2] = -fx * x / z ** 2
    jac_proj[:, 1, 1] = fy / z
    jac_proj[:, 1, 2] = -fy * y / z ** 2
    dot = np.einsum("mk,mk->m", n, p_i)
    if np.any(np.abs(dot) <= _EPS):
        raise ValueError("n^T p_i must be non-zero (grazing view is singular)")
    rank_one = np.einsum("mi,mj->mij", p_i, n) / dot[:, None, None]
    shear = np.broadcast_to(np.eye(3), (count, 3, 3)) - rank_one
    left = np.einsum("mij,jk->mik", jac_proj, rotation)
    right = np.einsum("mij,jk->mik", shear, np.linalg.inv(k_i))
    # A pixel displacement is [du, dv, 0]; the homogeneous scale column is not
    # part of the 2x2 patch Jacobian.
    full = np.einsum("mij,mjk->mik", left, right) * d[:, None, None]
    return full[:, :, :2]


def homography_condition_numbers(points_e, points_i, normals, rotation, k_i, depths, fx, fy):
    """2-norm condition number of each patch homography Jacobian (m,)."""
    jac = homography_jacobian(points_e, points_i, normals, rotation, k_i, depths, fx, fy)
    return np.linalg.cond(jac)


# --------------------------------------------------------------------------- theorem 1

def _sym(matrix):
    return 0.5 * (matrix + matrix.T)


def woodbury_eiv_posterior_precision(Lambda_0, X_A, M_u_A, Sigma_xi, sigma_obs):
    """Theorem 1 eq. 62: posterior precision with the rank-6 EIV downdate.

    ``Lambda_post = Lambda_ideal - Delta_Lambda_pose`` with

    ``Lambda_ideal = Lambda_0 + sigma^-2 X_A X_A^T``
    ``Delta = sigma^-4 X_A M (Sigma_xi^-1 + sigma^-2 M^T M)^-1 M^T X_A^T``.

    Returns ``(Lambda_post, Lambda_ideal, Delta_Lambda_pose)``; the downdate is
    symmetric PSD by construction (``Delta = B^T C^-1 B`` with ``C`` PD).
    """
    lambda_0 = _sym(np.asarray(Lambda_0, dtype=float))
    x_a = np.asarray(X_A, dtype=float)
    m_u = np.asarray(M_u_A, dtype=float)
    sigma_xi = _sym(np.asarray(Sigma_xi, dtype=float))
    if x_a.ndim != 2 or m_u.ndim != 2 or m_u.shape[1] != 6:
        raise ValueError("X_A must be (r, m) and M_u_A must be (m, 6)")
    if x_a.shape[1] != m_u.shape[0]:
        raise ValueError("X_A and M_u_A disagree on the anchor count m")
    if lambda_0.shape != (x_a.shape[0], x_a.shape[0]):
        raise ValueError("Lambda_0 must be (r, r)")
    if sigma_xi.shape != (6, 6):
        raise ValueError("Sigma_xi must be (6, 6)")
    observation_std = float(sigma_obs)
    variance = observation_std ** 2
    if not np.isfinite(observation_std) or observation_std <= 0:
        raise ValueError("sigma_obs must be positive and finite")
    # Whitening is Eq. 62 with Sigma=L L^T; unlike Sigma^-1 this also
    # handles the exact zero-pose-noise limit without a pseudoinverse cutoff.
    values, vectors = np.linalg.eigh(sigma_xi)
    if np.any(values < -_EPS):
        raise ValueError("Sigma_xi must be positive semidefinite")
    whitened = m_u @ (vectors * np.sqrt(np.maximum(values, 0.0)))
    ideal = _sym(lambda_0 + (x_a @ x_a.T) / variance)
    cross = x_a @ whitened
    core = np.eye(6) + (whitened.T @ whitened) / variance
    delta = _sym((cross @ np.linalg.solve(core, cross.T)) / variance ** 2)
    return _sym(ideal - delta), ideal, delta


def posterior_risk_trace(X_T, Lambda_post):
    """Trace term ``Tr(X_T^T Lambda_post^-1 X_T)`` of the theorem-1 lower bound."""
    x_t = np.asarray(X_T, dtype=float)
    post = _sym(np.asarray(Lambda_post, dtype=float))
    solved = np.linalg.solve(post, x_t)
    return float(np.trace(x_t.T @ solved))


# --------------------------------------------------------------------------- algorithm 1

def _psd_sqrt(matrix):
    values, vectors = np.linalg.eigh(_sym(matrix))
    return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.T


def solve_ternary_budget_alg1(B_total_bytes, Sigma_xi_prior, Lambda_prior, V_query,
                              V_g_pool, V_a_pool, X_T, sigma_obs, *,
                              delta_g_bytes=12.0, anchor_bytes=None, gamma_q=0.5,
                              alpha_pose=1.0, alpha_quant=1.0, g_levels=5, G_T=None):
    """Algorithm 1: exact ternary split ``(b_g*, b_a*, b_s*)``.

    Outer scan over the geometric budget ``b_g`` with a D-optimal greedy
    ``S_g``; inner Sherman-Morrison greedy for the spatially stationary
    representation anchors ``S_a``; the residual budget ``b_s`` funds the
    blind-spot task payload.  Everything is NumPy and deterministic.

    Pool format: ``V_g_pool = {"G": (n_g, 2, 6), "Gamma": (n_g, 2, 2)}`` and
    ``V_a_pool = {"X": (r, n_a), "G": (n_a, 2, 6), "grad": (n_a,)}``, where
    ``grad`` is the non-negative coefficient c_grad in Eq. 118. ``G_T`` holds
    the actual task-point Jacobians. Byte budgets become bits in the rate term.
    """
    budget = float(B_total_bytes)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("B_total_bytes must be non-negative")
    lambda_prior = _sym(np.asarray(Lambda_prior, dtype=float))
    sigma_prior = _sym(np.asarray(Sigma_xi_prior, dtype=float))
    if sigma_prior.shape != (6, 6):
        raise ValueError("Sigma_xi_prior must be (6, 6)")
    v_query = np.asarray(V_query, dtype=float)
    x_t = np.asarray(X_T, dtype=float)
    if x_t.ndim != 2:
        raise ValueError("X_T must be (r, k)")
    rank = lambda_prior.shape[0]
    if v_query.ndim != 2 or v_query.shape[0] != rank:
        raise ValueError("V_query must be (r, r_act)")
    k_task = x_t.shape[1]
    if not np.isfinite(sigma_obs) or sigma_obs <= 0:
        raise ValueError("sigma_obs must be positive and finite")
    variance = float(sigma_obs) ** 2
    if lambda_prior.shape != (rank, rank) or x_t.shape[0] != rank:
        raise ValueError("Lambda_prior and X_T must share source dimension r")

    g = np.asarray(V_g_pool.get("G", np.zeros((0, 2, 6))), dtype=float)
    gamma = np.asarray(V_g_pool.get("Gamma", np.zeros((0, 2, 2))), dtype=float)
    if g.ndim != 3 or g.shape[1:] != (2, 6) or gamma.shape != (g.shape[0], 2, 2):
        raise ValueError("V_g_pool must provide G (n_g,2,6) and Gamma (n_g,2,2)")
    a_x = np.asarray(V_a_pool.get("X", np.zeros((rank, 0))), dtype=float)
    a_g = np.asarray(V_a_pool.get("G", np.zeros((0, 2, 6))), dtype=float)
    a_grad = np.asarray(V_a_pool.get("grad", np.ones(a_x.shape[1])), dtype=float)
    if a_x.ndim != 2 or a_x.shape[0] != rank:
        raise ValueError("V_a_pool['X'] must be (r, n_a)")
    n_a = a_x.shape[1]
    if a_g.shape != (n_a, 2, 6) or a_grad.shape != (n_a,):
        raise ValueError("V_a_pool must provide G (n_a,2,6) and grad (n_a,)")
    if anchor_bytes is None:
        anchor_bytes = float(2 + rank)  # 2 B grid index + r x FP8 feature
    anchor_bytes = float(anchor_bytes)
    delta_g = float(delta_g_bytes)
    if not np.isfinite(anchor_bytes) or anchor_bytes <= 0 or not np.isfinite(delta_g) or delta_g <= 0:
        raise ValueError("anchor byte costs must be positive and finite")
    if np.any(a_grad < 0):
        raise ValueError("grad must hold non-negative c_grad coefficients")

    w_t = _sym(x_t @ x_t.T + float(gamma_q) * (v_query @ v_query.T))
    w_half = _psd_sqrt(w_t)
    sigma_inv = np.linalg.solve(sigma_prior, np.eye(6))
    lambda_inv = np.linalg.solve(lambda_prior, np.eye(rank))
    # sigma_x^2 is per component; k*r already appears in Eq. 124.
    sigma_x2 = float(np.mean(x_t ** 2)) if x_t.size else 0.0
    if G_T is None and k_task:
        raise ValueError("G_T must provide the actual blind-spot task Jacobians")
    g_task_cov = np.asarray(G_T if G_T is not None else np.zeros((0, 6)), dtype=float)
    if g_task_cov.shape not in ((k_task, 2, 6), (2 * k_task, 6)):
        raise ValueError("G_T must be (k, 2, 6) or (2*k, 6)")
    g_task_cov = g_task_cov.reshape(-1, 6)

    best = None
    grid = []
    for level in range(max(1, int(g_levels))):
        # Only full, available 12-byte (or configured cost) anchors are feasible.
        n_g = level
        b_g = delta_g * n_g
        if b_g > budget or n_g > g.shape[0]:
            break
        # ---- outer: D-optimal greedy geometric anchors
        info = sigma_inv.copy()
        chosen_g = []
        remaining = list(range(g.shape[0]))
        for _ in range(n_g):
            best_gain, best_index = None, None
            for index in remaining:
                contribution = g[index].T @ gamma[index] @ g[index]
                gain = float(np.linalg.slogdet(info + contribution)[1])
                if best_gain is None or gain > best_gain:
                    best_gain, best_index = gain, index
            if best_index is None:
                break
            info = info + g[best_index].T @ gamma[best_index] @ g[best_index]
            chosen_g.append(best_index)
            remaining.remove(best_index)
        sigma_post = _sym(np.linalg.solve(_sym(info), np.eye(6)))
        pose_trace = float(np.trace(g_task_cov @ sigma_post @ g_task_cov.T)) if g_task_cov.size else 0.0
        # ---- inner: Sherman-Morrison greedy representation anchors
        beta = 1.0 / (variance + a_grad * np.einsum(
            "mij,jk,mik->m", a_g, sigma_post, a_g)) if n_a else np.zeros(0)
        p_matrix = lambda_inv.copy()
        chosen_a = []
        pool = list(range(n_a))
        m_max = min(n_a, int((budget - b_g) // anchor_bytes))
        for m in range(max(0, m_max) + 1):
            b_a = m * anchor_bytes
            b_s = max(0.0, budget - b_g - b_a)
            trace_term = float(np.trace(w_t @ p_matrix))
            quant = (float(alpha_quant) * k_task * rank * sigma_x2
                     * 2.0 ** (-2.0 * 8.0 * b_s / max(1.0, rank * max(1, k_task))))
            j_proxy = trace_term + float(alpha_pose) * pose_trace + quant
            candidate = {"b_g_star": b_g, "b_a_star": b_a, "b_s_star": b_s,
                         "J_proxy": float(j_proxy), "m_star": m,
                         "geometric_anchor_ids": list(chosen_g),
                         "representation_anchor_ids": list(chosen_a),
                         "pose_trace": pose_trace}
            grid.append(dict(candidate))
            if best is None or j_proxy < best["J_proxy"] - 1e-12:
                best = candidate
            if m == m_max or not pool:
                break
            pool_idx = np.asarray(pool, dtype=int)
            x_rem = a_x[:, pool_idx]
            px_rem = p_matrix @ x_rem
            beta_rem = beta[pool_idx]
            numerators = beta_rem * np.sum((w_half @ px_rem) ** 2, axis=0)
            denominators = 1.0 + beta_rem * np.sum(x_rem * px_rem, axis=0)
            valid = denominators > _EPS
            if not np.any(valid):
                break
            gains = np.where(valid, numerators / np.maximum(denominators, _EPS), -np.inf)
            best_pos = int(np.argmax(gains))
            best_index = int(pool_idx[best_pos])
            px = px_rem[:, best_pos]
            denominator = float(denominators[best_pos])
            p_matrix = _sym(p_matrix - float(beta[best_index]) * np.outer(px, px) / denominator)
            chosen_a.append(best_index)
            pool.remove(best_index)

    best = dict(best or {"b_g_star": 0.0, "b_a_star": 0.0, "b_s_star": budget,
                         "J_proxy": float(np.trace(w_t @ lambda_inv)), "m_star": 0,
                         "geometric_anchor_ids": [], "representation_anchor_ids": [],
                         "pose_trace": 0.0})
    best.update(budget_bytes=budget, k_task=int(k_task), rank=int(rank),
                anchor_bytes=anchor_bytes, delta_g_bytes=delta_g,
                grid=grid, sigma_x2=sigma_x2)
    return best


# --------------------------------------------------------------------------- theorem 3

def matrix_wiener_shrinkage_gate(U_act, Lambda_act, A_hat, X_T_hat, Lambda_post_list,
                                 sigma_q_sq, *, X_T=None):
    """Theorem 3 eq. 88-92: closed-form matrix Wiener shrinkage gate.

    ``w[l, j] = mu^2 / (mu^2 + x_j^T Lambda_post_l^-1 x_j + ||a_l||^2 sigma_q_j^2)``
    with ``mu = a_l^T x_hat_j``. ``X_T`` supplies the unquantized task features
    in the posterior variance term of Eq. 88; omitted, ``X_T_hat`` is the
    explicitly plug-in estimate. Returns ``(Y_T_gated, weights_w, risk_reduction)``
    where ``E[R_gated] = R_ego - risk_reduction`` and ``risk_reduction >= 0``
    under the theorem's local quadratic signal/noise model, for every pose
    covariance and every (possibly empty) anchor set. This is a model risk
    identity; it is not an empirical guarantee for arbitrary biased features.
    """
    u_act = np.asarray(U_act, dtype=float)
    a_hat = np.asarray(A_hat, dtype=float)
    x_t = np.asarray(X_T_hat, dtype=float)
    lam = np.asarray(Lambda_act, dtype=float).reshape(-1)
    x_variance = x_t if X_T is None else np.asarray(X_T, dtype=float)
    if x_variance.shape != x_t.shape:
        raise ValueError("X_T must have the same (r, k) shape as X_T_hat")
    if u_act.ndim != 2 or a_hat.ndim != 2 or x_t.ndim != 2:
        raise ValueError("U_act, A_hat, X_T_hat must be matrices")
    rank_act = u_act.shape[1]
    if lam.shape[0] != rank_act:
        raise ValueError("Lambda_act must have one entry per action-sensitive channel")
    if a_hat.shape[0] != u_act.shape[0]:
        raise ValueError("A_hat rows must match the receiver feature width d_e")
    if a_hat.shape[1] != x_t.shape[0]:
        raise ValueError("A_hat columns must match the source feature width r")
    posts = Lambda_post_list if isinstance(Lambda_post_list, (list, tuple)) else [Lambda_post_list]
    posts = [_sym(np.asarray(post, dtype=float)) for post in posts]
    if len(posts) == 1:
        posts = posts * rank_act
    if len(posts) != rank_act:
        raise ValueError("Lambda_post_list must hold one (r, r) matrix per active channel")
    for post in posts:
        if post.shape != (x_t.shape[0], x_t.shape[0]):
            raise ValueError("each Lambda_post must be (r, r)")
    anchors = x_t.shape[1]
    quant = np.asarray(sigma_q_sq, dtype=float).reshape(-1)
    if quant.size == 1:
        quant = np.repeat(quant, anchors)
    if quant.shape[0] != anchors:
        raise ValueError("sigma_q_sq must be scalar or one value per anchor")

    if np.any(lam < 0) or not np.isfinite(lam).all():
        raise ValueError("Lambda_act must be finite and non-negative")
    if np.any(quant < 0) or not np.isfinite(quant).all():
        raise ValueError("sigma_q_sq must be finite and non-negative")
    if not np.allclose(u_act.T @ u_act, np.eye(rank_act), atol=1e-8):
        raise ValueError("U_act columns must be orthonormal")
    signal = u_act.T @ a_hat @ x_t  # (r_act, k) == mu_{l,j}
    weights = np.zeros((rank_act, anchors), dtype=float)
    for channel in range(rank_act):
        post = posts[channel]
        np.linalg.cholesky(post)  # The theorem assumes a proper, positive prior.
        solved = np.linalg.solve(post, x_variance)       # (r, k)
        variance = np.einsum("jk,jk->k", x_variance, solved)
        a_norm2 = float(np.sum((u_act[:, channel] @ a_hat) ** 2))
        mu2 = signal[channel] ** 2
        denominator = mu2 + variance + a_norm2 * quant
        weights[channel] = np.divide(mu2, denominator,
                                     out=np.zeros_like(mu2), where=denominator > 0.0)
    gated = u_act @ (weights * signal)
    risk_reduction = 0.5 * float(np.sum(lam[:, None] * weights * signal ** 2))
    return gated, weights, risk_reduction
