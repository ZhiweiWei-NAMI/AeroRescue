"""Core differential geometry and SE(3) projective operators for Agentic-Native ISCC.

Implements Contract ③ from docs/PHASE1_LOCAL_EXECUTION_PLAN.md:
- Left-perturbation SE(3) convention: T_hat_{e<-i} = exp(xi^) T*_{e<-i}
- Exact projective & Lie-algebra Jacobians: G_{px,k}, G_{f,k} in R^{2x6}
- Homogeneous pixel tangent embedding E_{3x2} and 2x2 homography Jacobian H_{f,k}
- Damped pseudoinverse H_{f,k}^dagger with grazing-angle / condition-number reliability
- Adjoint covariance transport Ad(Delta T)
"""
from __future__ import annotations

import numpy as np

E_3X2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)


def skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix [v]_x for v in R^3."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map exp([omega]_x) -> SO(3)."""
    omega = np.asarray(omega, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64) + skew(omega)
    W = skew(omega / theta)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) exponential map exp(xi^) for xi = [omega(3), upsilon(3)]^T in R^6."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    omega, upsilon = xi[:3], xi[3:]
    theta = float(np.linalg.norm(omega))
    R = so3_exp(omega)
    if theta < 1e-12:
        V = np.eye(3, dtype=np.float64) + 0.5 * skew(omega)
    else:
        W = skew(omega / theta)
        V = (
            np.eye(3, dtype=np.float64)
            + ((1.0 - np.cos(theta)) / theta) * W
            + ((theta - np.sin(theta)) / theta) * (W @ W)
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = V @ upsilon
    return T


def se3_log(T):
    """Logarithm for rotation-first left SE(3) coordinates."""
    T=np.asarray(T,float);R=T[:3,:3]
    theta=float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
    vee=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
    omega=.5*vee if theta<1e-8 else theta/(2*np.sin(theta))*vee
    W=skew(omega)
    coefficient=1/12 if theta<1e-6 else (1-.5*theta/np.tan(theta/2))/(theta*theta)
    return np.r_[omega,(np.eye(3)-.5*W+coefficient*(W@W))@T[:3,3]]


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    """6x6 Adjoint matrix Ad(T) for T = [[R, t], [0, 1]] in SE(3)."""
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float64)
    Ad = np.zeros((6, 6), dtype=np.float64)
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(t) @ R
    Ad[3:, 3:] = R
    return Ad


def transport_pose_covariance(
    sigma_xi: np.ndarray,
    delta_T: np.ndarray,
    process_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Transport left-perturbation covariance across reference frame update delta_T."""
    Ad = se3_adjoint(delta_T)
    transported = Ad @ np.asarray(sigma_xi, dtype=np.float64) @ Ad.T
    if process_noise is not None:
        transported = transported + np.asarray(process_noise, dtype=np.float64)
    return 0.5 * (transported + transported.T)


def project_point(p_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project 3D point p_cam = [X, Y, Z]^T (Z > 0) to pixel coordinates [u, v]^T."""
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return np.array([fx * (X / Z) + cx, fy * (Y / Z) + cy], dtype=np.float64)


def j_proj(p_e: np.ndarray, K_e: np.ndarray) -> np.ndarray:
    """2x3 perspective projection Jacobian J_proj(p_e) at receiver camera point p_e."""
    X, Y, Z = float(p_e[0]), float(p_e[1]), max(float(p_e[2]), 1e-6)
    fx, fy = float(K_e[0, 0]), float(K_e[1, 1])
    inv_z = 1.0 / Z
    return np.array([
        [fx * inv_z, 0.0, -fx * X * (inv_z * inv_z)],
        [0.0, fy * inv_z, -fy * Y * (inv_z * inv_z)],
    ], dtype=np.float64)


def j_se3_left(p_e: np.ndarray) -> np.ndarray:
    """3x6 left-perturbation SE(3) Jacobian J_se3(p_e) = [-[p_e]_x, I_3]."""
    J = np.zeros((3, 6), dtype=np.float64)
    J[:, :3] = -skew(p_e)
    J[:, 3:] = np.eye(3, dtype=np.float64)
    return J


def compute_pose_ray_jacobian(
    p_e: np.ndarray,
    K_e: np.ndarray,
    grid_scale_e: float = 1.0 / 32.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (G_px, G_f) in R^{2x6} for pixel and post-merger feature grid coordinates."""
    G_px = j_proj(p_e, K_e) @ j_se3_left(p_e)
    G_f = float(grid_scale_e) * G_px
    return G_px, G_f


def compute_homography_jacobian(
    p_i: np.ndarray,
    n_i: np.ndarray,
    T_e_from_i: np.ndarray,
    K_i: np.ndarray,
    K_e: np.ndarray,
    grid_scale_i: float = 1.0 / 32.0,
    grid_scale_e: float = 1.0 / 32.0,
    damping_lambda: float = 1e-2,
    min_cos_incidence: float = 0.15,
    max_cond: float = 50.0,
) -> dict:
    """Compute 2x2 homography Jacobian H_{e<-i}(k) and its damped pseudoinverse.

    Parameters
    ----------
    p_i : (3,) array
        3D point in sender (UAV) optical camera frame, with depth d_k = p_i[2] > 0.
    n_i : (3,) array
        Unit surface normal in sender (UAV) optical camera frame.
    """
    p_i = np.asarray(p_i, dtype=np.float64).reshape(3)
    n_i = np.asarray(n_i, dtype=np.float64).reshape(3)
    n_norm = float(np.linalg.norm(n_i))
    n_i = n_i / max(n_norm, 1e-12)
    d_k = max(float(p_i[2]), 1e-4)

    R_e_from_i = np.asarray(T_e_from_i[:3, :3], dtype=np.float64)
    t_e_from_i = np.asarray(T_e_from_i[:3, 3], dtype=np.float64)
    p_e = R_e_from_i @ p_i + t_e_from_i

    dot_np = float(np.dot(n_i, p_i))
    cos_inc = abs(dot_np) / max(float(np.linalg.norm(p_i)), 1e-8)
    # Regularize grazing denominator while flagging low reliability
    safe_denom = np.copysign(max(abs(dot_np), 1e-10), dot_np if dot_np != 0 else -1.0)
    plane_proj = np.eye(3, dtype=np.float64) - np.outer(p_i, n_i) / safe_denom

    K_i_inv = np.linalg.inv(np.asarray(K_i, dtype=np.float64))
    H_px = j_proj(p_e, K_e) @ R_e_from_i @ plane_proj @ (K_i_inv @ E_3X2) * d_k
    H_f = (float(grid_scale_e) / float(grid_scale_i)) * H_px

    svals = np.linalg.svd(H_f, compute_uv=False)
    cond = float(svals[0] / max(svals[-1], 1e-12))
    reliable = bool(cos_inc >= min_cos_incidence and cond <= max_cond and p_e[2] > 0.2)

    # Always provide a well-conditioned damped inverse; use stronger damping if ill-conditioned
    lam = float(damping_lambda) if reliable else max(float(damping_lambda), 0.1 * float(svals[0] ** 2 + 1e-4))
    H_f_pinv = np.linalg.inv(H_f.T @ H_f + lam * np.eye(2, dtype=np.float64)) @ H_f.T
    reliability_weight = float(np.clip(cos_inc / max(min_cos_incidence, 1e-3), 0.0, 1.0) * (1.0 / (1.0 + (cond / max_cond) ** 2)))

    return {
        "H_px": H_px,
        "H_f": H_f,
        "H_f_pinv": H_f_pinv,
        "cond": cond,
        "cos_incidence": cos_inc,
        "reliable": reliable,
        "reliability_weight": reliability_weight,
        "p_e": p_e,
    }


def finite_difference_check_jacobians(
    p_i: np.ndarray,
    n_i: np.ndarray,
    T_e_from_i: np.ndarray,
    K_i: np.ndarray,
    K_e: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """Verify analytical G_px and H_px against central finite differences."""
    p_i = np.asarray(p_i, dtype=np.float64).reshape(3)
    n_i = np.asarray(n_i, dtype=np.float64).reshape(3)
    n_i = n_i / np.linalg.norm(n_i)
    p_e = T_e_from_i[:3, :3] @ p_i + T_e_from_i[:3, 3]

    G_px, _ = compute_pose_ray_jacobian(p_e, K_e)
    G_num = np.zeros((2, 6), dtype=np.float64)
    for d in range(6):
        e_vec = np.zeros(6, dtype=np.float64)
        e_vec[d] = eps
        p_plus = (se3_exp(e_vec) @ np.append(p_e, 1.0))[:3]
        p_minus = (se3_exp(-e_vec) @ np.append(p_e, 1.0))[:3]
        G_num[:, d] = (project_point(p_plus, K_e) - project_point(p_minus, K_e)) / (2.0 * eps)

    hom = compute_homography_jacobian(p_i, n_i, T_e_from_i, K_i, K_e)
    H_px = hom["H_px"]
    u_i0 = project_point(p_i, K_i)
    K_i_inv = np.linalg.inv(K_i)
    plane_d = float(np.dot(n_i, p_i))

    H_num = np.zeros((2, 2), dtype=np.float64)
    for d in range(2):
        du = np.zeros(2, dtype=np.float64)
        du[d] = eps
        ray_plus = K_i_inv @ np.array([u_i0[0] + du[0], u_i0[1] + du[1], 1.0])
        p_i_plus = (plane_d / np.dot(n_i, ray_plus)) * ray_plus
        p_e_plus = T_e_from_i[:3, :3] @ p_i_plus + T_e_from_i[:3, 3]

        ray_minus = K_i_inv @ np.array([u_i0[0] - du[0], u_i0[1] - du[1], 1.0])
        p_i_minus = (plane_d / np.dot(n_i, ray_minus)) * ray_minus
        p_e_minus = T_e_from_i[:3, :3] @ p_i_minus + T_e_from_i[:3, 3]

        H_num[:, d] = (project_point(p_e_minus + (p_e_plus - p_e_minus), K_e) - project_point(p_e_minus, K_e)) / (2.0 * eps)

    return {
        "G_max_abs_err": float(np.max(np.abs(G_px - G_num))),
        "H_max_abs_err": float(np.max(np.abs(H_px - H_num))),
        "G_px": G_px,
        "H_px": H_px,
    }
