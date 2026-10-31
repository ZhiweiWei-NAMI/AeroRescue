"""Cascaded SE(3) registration and Woodbury rank-6 EIV recursive filter.

Implements Contract ⑤ from docs/PHASE1_LOCAL_EXECUTION_PLAN.md:
- Persistent cross-variant mapping A_hat in R^{p x r} and row precision matrices
  {Lambda_m}_{m=1}^p reside in fixed receiver subspace U_rec in R^{d_e x p}
  and fixed sender subspace U_src in R^{d_i x r} (p = r = 32).
- Step A: Point-to-plane / edge-tensor weighted Gauss-Newton pose refinement.
- Step B: Exact Woodbury rank-6 covariance inverse Omega_m^{-1} for correlated
  spatial-gradient pose noise M_{m,A} xi + eps_m.
- Tracks both sender predicted covariance Omega_pred and receiver measured
  covariance Omega_meas to quantify distributed calibration gap.
"""
from __future__ import annotations

import numpy as np
from .geometry import compute_pose_ray_jacobian, se3_exp, transport_pose_covariance


def woodbury_correlated_precision(
    sigma_obs_sq: float,
    M_dir: np.ndarray,
    sigma_xi: np.ndarray,
) -> np.ndarray:
    """Compute Omega^{-1} = (sigma_obs^2 I_m + M_dir Sigma_xi M_dir^T)^{-1} via Woodbury.

    Parameters
    ----------
    sigma_obs_sq : float
        Base observation variance sigma_obs^2 > 0.
    M_dir : (m, 6) array
        Directional spatial-pose Jacobian rows for m anchors.
    sigma_xi : (6, 6) array
        Current 6-DoF pose covariance matrix.
    """
    m = M_dir.shape[0]
    inv_s2 = 1.0 / max(float(sigma_obs_sq), 1e-8)
    if m == 0:
        return np.zeros((0, 0), dtype=np.float64)
    # Regularize sigma_xi inverse
    sigma_xi_reg = np.asarray(sigma_xi, dtype=np.float64) + 1e-10 * np.eye(6, dtype=np.float64)
    sigma_xi_inv = np.linalg.inv(sigma_xi_reg)
    inner = sigma_xi_inv + inv_s2 * (M_dir.T @ M_dir)
    inner_inv = np.linalg.inv(inner)
    return inv_s2 * np.eye(m, dtype=np.float64) - (inv_s2 * inv_s2) * (M_dir @ inner_inv @ M_dir.T)


class FixedSubspaceEIVFilter:
    """Persistent EIV filter in fixed receiver/sender orthonormal bases (p x r)."""

    def __init__(
        self,
        U_src: np.ndarray,
        U_rec: np.ndarray,
        prior_precision: float = 1.0,
        sigma_obs: float = 0.25,
        forgetting_factor: float = 0.995,
        basis_version: str = "v1.0",
    ):
        self.U_src = np.asarray(U_src, dtype=np.float64)  # (d_i, r)
        self.U_rec = np.asarray(U_rec, dtype=np.float64)  # (d_e, p)
        self.d_i, self.r = self.U_src.shape
        self.d_e, self.p = self.U_rec.shape
        self.prior_precision = float(prior_precision)
        self.sigma_obs_sq = float(sigma_obs) ** 2
        self.forgetting = float(forgetting_factor)
        self.basis_version = str(basis_version)
        self.reset()

    def reset(self, A_init_subspace: np.ndarray | None = None, prior_information: np.ndarray | None = None):
        """Reset per-episode persistent state."""
        if A_init_subspace is not None:
            self.A_hat = np.asarray(A_init_subspace, dtype=np.float64).copy()
        else:
            self.A_hat = np.zeros((self.p, self.r), dtype=np.float64)
        information=self.prior_precision*np.eye(self.r) if prior_information is None else np.asarray(prior_information,float)
        self.prior_information=information.copy()
        self.Lambda_post = np.repeat(
            information[None, :, :],
            self.p,
            axis=0,
        )  # (p, r, r)
        self.P_post = np.repeat(
            np.linalg.inv(information)[None, :, :],
            self.p,
            axis=0,
        )  # (p, r, r)
        self.rhs = np.einsum("mij,mj->mi", self.Lambda_post, self.A_hat)  # (p, r)
        self.step_count = 0

    def project_sender_features(self, F_i: np.ndarray) -> np.ndarray:
        """Project raw post-merger sender features (d_i, N) -> (r, N)."""
        return self.U_src.T @ np.asarray(F_i, dtype=np.float64)

    def project_receiver_features(self, F_e: np.ndarray) -> np.ndarray:
        """Project raw post-merger receiver features (d_e, N) -> (p, N)."""
        return self.U_rec.T @ np.asarray(F_e, dtype=np.float64)

    def reconstruct_receiver_features(self, X_sub: np.ndarray) -> np.ndarray:
        """Map sender subspace features X_sub (r, k) -> full receiver space (d_e, k)."""
        Y_sub = self.A_hat @ np.asarray(X_sub, dtype=np.float64)
        return self.U_rec @ Y_sub

    def full_space_mapping(self) -> np.ndarray:
        """Return full d_e x d_i mapping U_rec A_hat U_src^T."""
        return self.U_rec @ self.A_hat @ self.U_src.T

    def step_a_geometric_registration(
        self,
        T_prior_e_from_i: np.ndarray,
        sigma_xi_prior: np.ndarray,
        geom_anchors: list[dict],
        K_e: np.ndarray,
        grid_scale_e: float = 1.0 / 32.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step A: Update pose estimate T_hat and posterior covariance tilde_Sigma_xi.

        Each item in geom_anchors contains:
          - 'p_i': (3,) 3D point in UAV frame
          - 'Gamma_k': (2, 2) edge gradient structure tensor in feature-grid coordinates
          - 'residual_f': (2,) optional 2D edge alignment residual in feature-grid units
        """
        T_curr = np.asarray(T_prior_e_from_i, dtype=np.float64).copy()
        sigma_inv = np.linalg.inv(np.asarray(sigma_xi_prior, dtype=np.float64) + 1e-10 * np.eye(6))
        info_accum = sigma_inv.copy()
        grad_accum = np.zeros(6, dtype=np.float64)

        for anchor in geom_anchors:
            p_i = np.asarray(anchor["p_i"], dtype=np.float64).reshape(3)
            p_e = T_curr[:3, :3] @ p_i + T_curr[:3, 3]
            _, G_f = compute_pose_ray_jacobian(p_e, K_e, grid_scale_e=grid_scale_e)
            Gamma_k = np.asarray(anchor["Gamma_k"], dtype=np.float64).reshape(2, 2)
            info_accum += G_f.T @ Gamma_k @ G_f
            if "residual_f" in anchor and anchor["residual_f"] is not None:
                res = np.asarray(anchor["residual_f"], dtype=np.float64).reshape(2)
                grad_accum += G_f.T @ (Gamma_k @ res)

        sigma_post = np.linalg.inv(info_accum)
        sigma_post = 0.5 * (sigma_post + sigma_post.T)
        xi_update = sigma_post @ grad_accum
        delta_T = se3_exp(xi_update)
        T_post = delta_T @ T_curr
        if np.linalg.norm(xi_update) > 1e-8:
            sigma_post = transport_pose_covariance(sigma_post, delta_T)
        return T_post, sigma_post, xi_update

    def step_b_woodbury_rls_update(
        self,
        X_A: np.ndarray,
        Y_A_sub: np.ndarray,
        D_e_sub: np.ndarray,
        G_f_list: np.ndarray,
        sigma_xi_post: np.ndarray,
        M_pred_list: np.ndarray | None = None,
        query_receiver_directions: np.ndarray | None = None,
        diagonal_pose_noise: bool = False,
    ) -> dict:
        """Step B: Update A_hat and Lambda_post with Woodbury rank-6 correction.

        Parameters
        ----------
        X_A : (r, m) array
            Sender subspace features at m representation anchors.
        Y_A_sub : (p, m) array
            Receiver subspace features sampled at registered anchor locations.
        D_e_sub : (p, m, 2) array
            Receiver subspace spatial feature gradients [d y / d u_f, d y / d v_f].
        G_f_list : (m, 2, 6) array
            Feature-grid ray Jacobians at the m anchors.
        sigma_xi_post : (6, 6) array
            Posterior pose covariance from Step A.
        """
        m = X_A.shape[1]
        if m == 0:
            self.step_count += 1
            return {"num_anchors": 0, "woodbury_loss_trace": 0.0, "omega_calibration_rel_err": 0.0}

        # Apply mild exponential forgetting only to accumulated evidence above prior
        if self.forgetting < 1.0 and self.step_count > 0:
            prior_I = self.prior_precision * np.eye(self.r, dtype=np.float64)[None, :, :]
            self.Lambda_post = prior_I + self.forgetting * (self.Lambda_post - prior_I)
            self.rhs = np.einsum("mij,mj->mi", self.Lambda_post, self.A_hat)

        woodbury_losses = []
        omega_rel_errs = []
        inv_s2 = 1.0 / max(self.sigma_obs_sq, 1e-8)
        ideal_delta = inv_s2 * (X_A @ X_A.T)

        for row_idx in range(self.p):
            # M_row: (m, 6), each anchor k has D_e_sub[row_idx, k] (1x2) @ G_f_list[k] (2x6)
            M_row = np.einsum("ka,kab->kb", D_e_sub[row_idx], G_f_list)
            if diagonal_pose_noise:
                variances = self.sigma_obs_sq + np.einsum("ki,ij,kj->k", M_row, sigma_xi_post, M_row)
                Omega_inv = np.diag(1.0 / variances)
            else:
                Omega_inv = woodbury_correlated_precision(self.sigma_obs_sq, M_row, sigma_xi_post)
            delta_Lambda = X_A @ Omega_inv @ X_A.T
            woodbury_losses.append(float(np.trace(ideal_delta - delta_Lambda)))

            self.Lambda_post[row_idx] = 0.5 * (
                self.Lambda_post[row_idx] + delta_Lambda + (self.Lambda_post[row_idx] + delta_Lambda).T
            )
            self.P_post[row_idx] = np.linalg.inv(self.Lambda_post[row_idx])
            y_row = Y_A_sub[row_idx]  # (m,)
            self.rhs[row_idx] = self.rhs[row_idx] + X_A @ (Omega_inv @ y_row)
            self.A_hat[row_idx] = self.P_post[row_idx] @ self.rhs[row_idx]

        # Compare sender and receiver in the SAME query directions, rather than
        # comparing the first four receiver PCA rows to unrelated query axes.
        if M_pred_list is not None and query_receiver_directions is not None:
            D_query = np.einsum("pl,pka->lka", query_receiver_directions, D_e_sub)
            M_meas = np.einsum("lka,kab->lkb", D_query, G_f_list)
            for M_pred, M_row in zip(M_pred_list, M_meas):
                Omega_meas = self.sigma_obs_sq * np.eye(m) + M_row @ sigma_xi_post @ M_row.T
                Omega_pred = self.sigma_obs_sq * np.eye(m) + M_pred @ sigma_xi_post @ M_pred.T
                omega_rel_errs.append(float(np.linalg.norm(Omega_meas - Omega_pred) /
                                           max(np.linalg.norm(Omega_meas), 1e-8)))

        self.step_count += 1
        return {
            "num_anchors": int(m),
            "woodbury_loss_trace": float(np.mean(woodbury_losses)),
            "omega_calibration_rel_err": float(np.mean(omega_rel_errs)) if omega_rel_errs else 0.0,
        }

    def query_direction_uncertainty(
        self,
        U_act_full: np.ndarray,
        X_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project action-sensitive directions U_act_full (d_e, r_act) into fixed subspace.

        Returns
        -------
        mu_pred : (r_act, k) array
            Predicted signal amplitude along each active direction for k blind tokens.
        var_pred : (r_act, k) array
            Posterior epistemic variance x_{T,j}^T P_ell x_{T,j} along each active direction.
        """
        U_act_sub = self.U_rec.T @ np.asarray(U_act_full, dtype=np.float64)  # (p, r_act)
        r_act = U_act_sub.shape[1]
        k = X_T.shape[1]
        Y_T_sub = self.A_hat @ X_T  # (p, k)
        mu_pred = U_act_sub.T @ Y_T_sub  # (r_act, k)

        # Approximate directional covariance P_ell = sum_m u_{m, ell}^2 P_post[m]
        var_pred = np.zeros((r_act, k), dtype=np.float64)
        for ell in range(r_act):
            weights = U_act_sub[:, ell] ** 2  # (p,)
            P_ell = np.einsum("m,mij->ij", weights, self.P_post)  # (r, r)
            var_pred[ell] = np.einsum("ik,ij,jk->k", X_T, P_ell, X_T)
        return mu_pred, var_pred
