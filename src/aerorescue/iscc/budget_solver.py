"""Ternary budget splitting and Schur-complement conditional rank-1 greedy selector (Algorithm 1).

Implements Contract ④ & Contract ⑤ from docs/PHASE1_LOCAL_EXECUTION_PLAN.md:
- Supports cold-start `mode='bootstrap'` (with zero-overlap active view exit and
  guaranteed blind task floor b_s >= 0.25 B_t) and warm `mode='query'`.
- Outer 1D discrete grid search over geometry calibration budget b_g via D-optimal
  greedy log-det maximization.
- Inner representation anchor selection via exact Schur-complement conditional
  innovation variance nu_k = Omega_{kk} - c_k^T Omega_S^{-1} c_k and orthogonalized
  feature x_tilde_k = x_k - X_S Omega_S^{-1} c_k, using 1x6 projected spatial-gradient
  proxy rows m_hat_{ell, k}^{(i)} = v_ell^T D_{i, k} H_{f, k}^dagger G_{f, k}.
- Exact over-the-air ByteLedger accounting for downlink request, uplink geometry/
  representation/blind payloads, and ACK/basis sync metadata.
"""
from __future__ import annotations

import numpy as np


# Exact byte costs per protocol element (Contract ⑤)
DOWNLINK_HEADER_BYTES = 12
DOWNLINK_FRUSTUM_POSE_BYTES = 24       # 12 x FP16
DOWNLINK_SIGMA_XI_BYTES = 42           # 21 upper-triangular FP16 of 6x6
DOWNLINK_QUERY_VEC_BYTES = 256         # r=32 x r_act=4 x FP16
DOWNLINK_LAMBDA_WEIGHTS_BYTES = 8      # 4 x FP16
GEOM_ANCHOR_BYTES = 14                 # 6 FP16 point/normal components + 2-byte id
REP_ANCHOR_BYTES = 42                  # 3 FP16 coordinates + FP16 scale + 32 int8 + id
MAX_BLIND_BITS = 16
BLIND_COORD_HEADER_BYTES = 14           # 3 FP16 coordinates + 2 FP32 quantizer bounds


def compute_downlink_bytes(mode: str, include_basis_sync: bool = False) -> int:
    """Exact downlink request byte count."""
    base = DOWNLINK_HEADER_BYTES + DOWNLINK_FRUSTUM_POSE_BYTES + DOWNLINK_SIGMA_XI_BYTES + 4  # ACK header
    if mode == "query":
        base += DOWNLINK_QUERY_VEC_BYTES + DOWNLINK_LAMBDA_WEIGHTS_BYTES
    if include_basis_sync:
        base += 4  # basis_version tag
    return int(base)


def select_d_optimal_geometry_anchors(
    sigma_xi_prior: np.ndarray,
    geom_candidates: list[dict],
    num_select: int,
) -> tuple[list[int], np.ndarray]:
    """D-optimal greedy selection maximizing log det(Sigma_xi^{-1} + sum G_k^T Gamma_k G_k)."""
    info_mat = np.linalg.inv(np.asarray(sigma_xi_prior, dtype=np.float64) + 1e-10 * np.eye(6))
    selected: list[int] = []
    n_cand = len(geom_candidates)
    if num_select <= 0 or n_cand == 0:
        return selected, np.linalg.inv(info_mat)

    info_Contribs = []
    for c in geom_candidates:
        G_f = np.asarray(c["G_f"], dtype=np.float64).reshape(2, 6)
        Gamma_k = np.asarray(c["Gamma_k"], dtype=np.float64).reshape(2, 2)
        info_Contribs.append(G_f.T @ Gamma_k @ G_f)

    remaining = set(range(n_cand))
    for _ in range(min(num_select, n_cand)):
        best_idx = -1
        best_logdet = -np.inf
        for idx in remaining:
            cand_info = info_mat + info_Contribs[idx]
            _, logdet = np.linalg.slogdet(cand_info)
            if logdet > best_logdet:
                best_logdet = float(logdet)
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        info_mat = info_mat + info_Contribs[best_idx]

    sigma_post = np.linalg.inv(info_mat)
    sigma_post = 0.5 * (sigma_post + sigma_post.T)
    return selected, sigma_post


def solve_ternary_budget_algorithm1(
    total_budget_bytes: int,
    sigma_xi_prior: np.ndarray,
    P_pred_prior: np.ndarray,
    X_T: np.ndarray,
    G_T_list: np.ndarray,
    geom_candidates: list[dict],
    rep_candidates: list[dict],
    mode: str = "query",
    V_query: np.ndarray | None = None,
    lambda_act: np.ndarray | None = None,
    sigma_obs: float = 0.25,
    gamma_q: float = 0.5,
    alpha_pose: float = 1.0,
    alpha_quant: float = 1.0,
    geom_counts_grid: tuple[int, ...] = (0, 3, 6, 12),
    max_rep_anchors: int = 16,
    active_view_threshold: float = float("inf"),
    wire_costs: dict | None = None,
    codec_bits: tuple[int, ...] | None = None,
) -> dict:
    """Execute Algorithm 1 with exact Schur-complement conditional rank-1 greedy updates.

    Parameters
    ----------
    total_budget_bytes : int
        Total per-step byte budget B_t (including downlink + uplink).
    sigma_xi_prior : (6, 6) array
        Prior pose error covariance.
    P_pred_prior : (r, r) or (r_act, r, r) array
        Sender proxy posterior covariance P_0 = Lambda_{post, t-1}^{-1}.
    X_T : (r, k) array
        Blind-spot target features in sender subspace (r=32).
    G_T_list : (k, 2, 6) array
        Feature-grid ray Jacobians at k blind targets.
    geom_candidates : list of dict
        Candidate pool V_g, each with 'G_f' (2,6), 'Gamma_k' (2,2), 'p_i' (3,).
    rep_candidates : list of dict
        Candidate pool V_a, each with:
          - 'x_k': (r,) sender subspace feature
          - 'D_i_k': (r, 2) spatial feature gradient in sender PCA subspace
          - 'H_f_pinv': (2, 2) damped homography inverse
          - 'G_f': (2, 6) pose ray Jacobian
          - 'reliability_weight': float in [0, 1]
    """
    r, k_blind = X_T.shape
    sigma_obs_sq = float(sigma_obs) ** 2
    g_cost=GEOM_ANCHOR_BYTES if wire_costs is None else int(wire_costs["geometry"])
    a_cost=REP_ANCHOR_BYTES if wire_costs is None else int(wire_costs["representation"])
    s_header=BLIND_COORD_HEADER_BYTES if wire_costs is None else int(wire_costs["slot_header"])

    # Check if warm query is usable; fall back to bootstrap if V_query is absent or zero
    effective_mode = str(mode)
    if effective_mode == "query":
        if V_query is None or np.linalg.norm(V_query) < 1e-8:
            effective_mode = "bootstrap"

    downlink_bytes = compute_downlink_bytes(effective_mode) if wire_costs is None else int(wire_costs["fixed"])
    if total_budget_bytes < downlink_bytes + k_blind * s_header:
        raise ValueError("budget cannot hold control packet and blind coordinate/quantizer headers")
    uplink_budget = int(total_budget_bytes) - downlink_bytes

    # Zero-overlap exit: if both V_g and V_a are empty, allocate 0 to calibration and trigger active view
    if len(geom_candidates) == 0 and len(rep_candidates) == 0:
        b_s_bytes = uplink_budget
        if codec_bits is not None:
            feasible=[bits for bits in codec_bits if k_blind*s_header+int(np.ceil(r*k_blind*bits/8))<=b_s_bytes]
            if k_blind and not feasible:raise ValueError("budget cannot encode blind payload")
            b_s_bytes=k_blind*s_header+int(np.ceil(r*k_blind*max(feasible)/8)) if k_blind else 0
        b_s_payload_bits = max(0, (b_s_bytes - k_blind * s_header) * 8)
        return {
            "mode": effective_mode,
            "zero_overlap": True,
            "trigger_active_view": True,
            "selected_geom_indices": [],
            "selected_rep_indices": [],
            "sigma_xi_post": sigma_xi_prior.copy(),
            "byte_ledger": {
                "total_budget_bytes": int(total_budget_bytes),
                "downlink_bytes": downlink_bytes,
                "b_g_bytes": 0,
                "b_a_bytes": 0,
                "b_s_bytes": b_s_bytes,
                "b_s_bits_per_component": float(b_s_payload_bits / max(r * k_blind, 1)),
                "total_used_bytes": downlink_bytes + b_s_bytes,
            },
            "proxy_objective": float("inf"),
        }

    # Construct query directions V_dirs (r, L) and directional weights lam_dirs (L,)
    if effective_mode == "query" and V_query is not None:
        V_dirs = np.asarray(V_query, dtype=np.float64)
        if V_dirs.ndim == 1:
            V_dirs = V_dirs[:, None]
        L_dirs = V_dirs.shape[1]
        lam_dirs = (
            np.asarray(lambda_act, dtype=np.float64).reshape(L_dirs)
            if lambda_act is not None
            else np.ones(L_dirs, dtype=np.float64) / L_dirs
        )
        W_T = (X_T @ X_T.T) + float(gamma_q) * (V_dirs @ np.diag(lam_dirs) @ V_dirs.T)
    else:
        # Bootstrap mode: use top principal directions of [X_T, X_pool] for broad subspace exploration
        pool_mat = (
            np.column_stack([c["x_k"] for c in rep_candidates])
            if len(rep_candidates) > 0
            else np.eye(r, dtype=np.float64)
        )
        joint_mat = np.hstack([X_T, pool_mat]) if k_blind > 0 else pool_mat
        U_boot, s_boot, _ = np.linalg.svd(joint_mat, full_matrices=False)
        L_dirs = min(4, U_boot.shape[1])
        V_dirs = U_boot[:, :L_dirs]
        lam_dirs = (s_boot[:L_dirs] ** 2) / max(float(np.sum(s_boot[:L_dirs] ** 2)), 1e-8)
        W_T = (X_T @ X_T.T) + float(gamma_q) * (V_dirs @ np.diag(lam_dirs) @ V_dirs.T)

    W_T = 0.5 * (W_T + W_T.T) + 1e-6 * np.eye(r, dtype=np.float64)
    W_T_sqrt = np.linalg.cholesky(W_T).T  # W_T = W_T_sqrt.T @ W_T_sqrt

    # Normalize P_pred_prior into (L_dirs, r, r)
    P_prior_arr = np.asarray(P_pred_prior, dtype=np.float64)
    if P_prior_arr.ndim == 2:
        P_init_dirs = np.repeat(P_prior_arr[None, :, :], L_dirs, axis=0)
    else:
        if P_prior_arr.shape[0] >= L_dirs:
            P_init_dirs = P_prior_arr[:L_dirs].copy()
        else:
            P_init_dirs = np.repeat(np.mean(P_prior_arr, axis=0, keepdims=True), L_dirs, axis=0)

    # Precompute 1x6 projected spatial-gradient proxy rows m_hat_{ell, k} = v_ell^T D_{i,k} H_f^dagger G_f
    n_rep = len(rep_candidates)
    M_proxy_all = np.zeros((L_dirs, n_rep, 6), dtype=np.float64)
    X_pool = np.zeros((r, n_rep), dtype=np.float64)
    rel_weights = np.ones(n_rep, dtype=np.float64)
    for k_idx, cand in enumerate(rep_candidates):
        X_pool[:, k_idx] = np.asarray(cand["x_k"], dtype=np.float64).reshape(r)
        D_i_k = np.asarray(cand["D_i_k"], dtype=np.float64).reshape(r, 2)
        H_pinv = np.asarray(cand["H_f_pinv"], dtype=np.float64).reshape(2, 2)
        G_f = np.asarray(cand["G_f"], dtype=np.float64).reshape(2, 6)
        HG = H_pinv @ G_f  # (2, 6)
        for ell in range(L_dirs):
            M_proxy_all[ell, k_idx] = (V_dirs[:, ell].T @ D_i_k) @ HG
        rel_weights[k_idx] = float(cand.get("reliability_weight", 1.0))

    # Reserve at least 25% of uplink budget for blind task payload during bootstrap
    min_bs_bytes = max(int(np.ceil(0.25 * total_budget_bytes)), k_blind * s_header) if (effective_mode == "bootstrap" and k_blind > 0) else (
        k_blind * s_header if k_blind > 0 else 0
    )
    max_calib_bytes = max(0, uplink_budget - min_bs_bytes)

    best_solution = None
    best_J = float("inf")
    sigma_x_sq = float(np.mean(X_T ** 2)) if k_blind > 0 else 1.0

    for n_g in geom_counts_grid:
        b_g_bytes = n_g * g_cost
        if b_g_bytes > max_calib_bytes or n_g > len(geom_candidates):
            continue

        sel_g, sigma_xi_g = select_d_optimal_geometry_anchors(sigma_xi_prior, geom_candidates, n_g)

        # Pose risk term on blind targets
        pose_term = 0.0
        if k_blind > 0 and G_T_list is not None and len(G_T_list) > 0:
            for j in range(min(k_blind, len(G_T_list))):
                G_Tj = np.asarray(G_T_list[j], dtype=np.float64).reshape(2, 6)
                pose_term += float(np.trace(G_Tj @ sigma_xi_g @ G_Tj.T))

        # Inner Schur-complement conditional rank-1 greedy loop over m = 0 .. max_rep_anchors
        P_curr = P_init_dirs.copy()  # (L_dirs, r, r)
        sel_a: list[int] = []
        # Maintain Omega_S^{-1} per direction ell: list of (m x m) arrays
        Omega_S_inv = [np.zeros((0, 0), dtype=np.float64) for _ in range(L_dirs)]

        max_m = min(
            max_rep_anchors,
            n_rep,
            (max_calib_bytes - b_g_bytes) // a_cost,
        )

        for m_step in range(max_m + 1):
            b_a_bytes = m_step * a_cost
            b_s_bytes = uplink_budget - b_g_bytes - b_a_bytes
            if codec_bits is not None:
                if k_blind:
                    feasible=[bits for bits in codec_bits if k_blind*s_header+int(np.ceil(r*k_blind*bits/8))<=b_s_bytes
                              and k_blind*s_header+int(np.ceil(r*k_blind*bits/8))>=min_bs_bytes]
                    if not feasible:break
                    bits_selected=max(feasible)
                    b_s_bytes=k_blind*s_header+int(np.ceil(r*k_blind*bits_selected/8))
                else:
                    bits_selected=0;b_s_bytes=0
            b_s_payload_bits = max(0, (b_s_bytes - k_blind * s_header) * 8)

            # Evaluate current proxy objective J_proxy(b_g, m)
            rep_trace = sum(
                float(lam_dirs[ell]) * float(np.trace(W_T @ P_curr[ell]))
                for ell in range(L_dirs)
            )
            quant_term = (
                float(k_blind * r * sigma_x_sq) * (2.0 ** (-2.0 * min(MAX_BLIND_BITS, b_s_payload_bits / max(r * k_blind, 1))))
                if k_blind > 0
                else 0.0
            )
            J_val = rep_trace + float(alpha_pose) * pose_term + float(alpha_quant) * quant_term

            if J_val < best_J:
                best_J = J_val
                best_solution = {
                    "mode": effective_mode,
                    "zero_overlap": False,
                    "selected_geom_indices": list(sel_g),
                    "selected_rep_indices": list(sel_a),
                    "sigma_xi_post": sigma_xi_g.copy(),
                    "P_post_pred": P_curr.copy(),
                    "M_proxy_selected": M_proxy_all[:, sel_a, :].copy() if len(sel_a) > 0 else np.zeros((L_dirs, 0, 6)),
                    "byte_ledger": {
                        "total_budget_bytes": int(total_budget_bytes),
                        "downlink_bytes": downlink_bytes,
                        "b_g_bytes": b_g_bytes,
                        "b_a_bytes": b_a_bytes,
                        "b_s_bytes": b_s_bytes,
                        "b_s_bits_per_component": float(b_s_payload_bits / max(r * k_blind, 1)),
                        "total_used_bytes": downlink_bytes + b_g_bytes + b_a_bytes + b_s_bytes,
                    },
                    "proxy_objective": float(J_val),
                    "rep_uncertainty_trace": float(rep_trace),
                }

            if m_step == max_m:
                break

            # Select next representation anchor k* via Schur-complement conditional rank-1 gain
            best_k = -1
            best_gain = -1.0
            best_step_cache = None

            X_S = X_pool[:, sel_a] if len(sel_a) > 0 else np.zeros((r, 0), dtype=np.float64)

            for k_cand in range(n_rep):
                if k_cand in sel_a or rel_weights[k_cand] <= 1e-4:
                    continue
                x_k = X_pool[:, k_cand]
                total_gain = 0.0
                per_dir_updates = []

                for ell in range(L_dirs):
                    m_k = M_proxy_all[ell, k_cand]  # (6,)
                    omega_kk = sigma_obs_sq / max(rel_weights[k_cand], 0.1) + float(m_k @ sigma_xi_g @ m_k)
                    if len(sel_a) == 0:
                        nu_k = max(omega_kk, 1e-8)
                        x_tilde = x_k
                        c_k = np.zeros(0, dtype=np.float64)
                    else:
                        M_S = M_proxy_all[ell, sel_a]  # (m, 6)
                        c_k = M_S @ (sigma_xi_g @ m_k)  # (m,)
                        q_vec = Omega_S_inv[ell] @ c_k  # (m,)
                        nu_k = max(omega_kk - float(c_k @ q_vec), 1e-6)
                        x_tilde = x_k - X_S @ q_vec

                    Px = P_curr[ell] @ x_tilde
                    num = float(np.sum((W_T_sqrt @ Px) ** 2))
                    den = nu_k + float(x_tilde @ Px)
                    gain_ell = num / max(den, 1e-12)
                    total_gain += float(lam_dirs[ell]) * gain_ell
                    per_dir_updates.append((nu_k, x_tilde, Px, den, c_k))

                if total_gain > best_gain:
                    best_gain = total_gain
                    best_k = k_cand
                    best_step_cache = per_dir_updates

            if best_k < 0 or best_step_cache is None:
                break

            # Commit rank-1 update for P_curr and block-matrix inverse for Omega_S_inv
            for ell in range(L_dirs):
                nu_k, x_tilde, Px, den, c_k = best_step_cache[ell]
                P_curr[ell] = P_curr[ell] - np.outer(Px, Px) / max(den, 1e-12)
                P_curr[ell] = 0.5 * (P_curr[ell] + P_curr[ell].T)
                m_prev = len(sel_a)
                if m_prev == 0:
                    Omega_S_inv[ell] = np.array([[1.0 / nu_k]], dtype=np.float64)
                else:
                    q_vec = Omega_S_inv[ell] @ c_k
                    inv_nu = 1.0 / nu_k
                    new_inv = np.zeros((m_prev + 1, m_prev + 1), dtype=np.float64)
                    new_inv[:m_prev, :m_prev] = Omega_S_inv[ell] + inv_nu * np.outer(q_vec, q_vec)
                    new_inv[:m_prev, m_prev] = -inv_nu * q_vec
                    new_inv[m_prev, :m_prev] = -inv_nu * q_vec
                    new_inv[m_prev, m_prev] = inv_nu
                    Omega_S_inv[ell] = 0.5 * (new_inv + new_inv.T)

            sel_a.append(best_k)

    assert best_solution is not None
    best_solution["trigger_active_view"] = bool(
        best_solution["rep_uncertainty_trace"] > float(active_view_threshold)
    )
    return best_solution


def quantize_and_dequantize_blind_features(
    X_T: np.ndarray,
    bits_per_component: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Uniform mid-rise quantization of blind subspace features X_T (r, k).

    Returns (X_T_hat, sigma_q_sq_per_token, actual_payload_bytes).
    """
    r, k = X_T.shape
    if k == 0 or bits_per_component < 1.0:
        return np.zeros_like(X_T), np.var(X_T, axis=0) + 1e-4, 0

    b_int = int(np.clip(np.floor(bits_per_component), 1, MAX_BLIND_BITS))
    levels = (1 << b_int) - 1
    X_hat = np.zeros_like(X_T, dtype=np.float64)
    sigma_q_sq = np.zeros(k, dtype=np.float64)

    for j in range(k):
        col = X_T[:, j]
        c_min, c_max = float(np.float32(np.min(col))), float(np.float32(np.max(col)))
        span = max(c_max - c_min, 1e-6)
        step = span / levels
        q_idx = np.round((col - c_min) / step)
        X_hat[:, j] = c_min + q_idx * step
        sigma_q_sq[j] = float( np.mean((col - X_hat[:, j]) ** 2) + (step ** 2) / 12.0 )

    payload_bytes = k * BLIND_COORD_HEADER_BYTES + int(np.ceil(r * k * b_int / 8.0))
    return X_hat, sigma_q_sq, payload_bytes


def encode_blind_packet(X_T: np.ndarray, coordinates: np.ndarray, bits_per_component: float):
    """Serialize real coordinate/quantizer headers and packed scalar codes.

    Headers per slot: three FP16 optical coordinates and two FP32 range values.
    Returns the actual bytes, decoded features, and measured quantization variance.
    """
    import struct
    X=np.asarray(X_T,dtype=float); r,k=X.shape
    bits=int(np.clip(np.floor(bits_per_component),1,MAX_BLIND_BITS))
    if bits_per_component < 1 or k == 0:
        return b"",np.zeros_like(X),np.var(X,axis=0)+1e-4
    payload=bytearray(); decoded=np.empty_like(X); variance=np.empty(k)
    for j in range(k):
        lo=float(np.float32(X[:,j].min())); hi=float(np.float32(X[:,j].max()))
        payload.extend(struct.pack("<3e2f",*coordinates[:,j],lo,hi))
        step=max(hi-lo,1e-6)/((1<<bits)-1)
        codes=np.round((X[:,j]-lo)/step).astype(np.uint16)
        packed=np.packbits(((codes[:,None]>>np.arange(bits))&1).astype(np.uint8).ravel(),bitorder="little")
        payload.extend(packed.tobytes())
        restored=np.unpackbits(packed,bitorder="little")[:r*bits].reshape(r,bits)@(1<<np.arange(bits))
        decoded[:,j]=lo+restored*step
        variance[j]=np.mean((decoded[:,j]-X[:,j])**2)+step**2/12
    return bytes(payload),decoded,variance
