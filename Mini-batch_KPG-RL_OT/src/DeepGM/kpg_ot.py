"""
KPG-RL OT solver for Deep Generative Models (unsupervised setting).

In DeepGM there are no class labels, so keypoints are selected via a
two-pass strategy:
  Pass 1 — solve standard OT to get an initial transport plan pi_init.
  Pass 2 — from pi_init, pick the top-U most confident matched pairs
            (highest pi_init[i,j] entries) as keypoints, then solve
            the KPG-RL-KP blended problem with the mask and guiding matrix.

This provides structural guidance without requiring labels.
When use_kpg=False the module falls back to standard OT solvers.
"""

import numpy as np
import ot


# -----------------------------------------------------------------------
# Numerics
# -----------------------------------------------------------------------

def _softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-20)


def _js_matrix(P, Q, eps=1e-10):
    P_e = P[:, np.newaxis, :]
    Q_e = Q[np.newaxis, :, :]
    M = 0.5 * (P_e + Q_e)
    kl1 = np.sum(P_e * (np.log(P_e + eps) - np.log(M + eps)), axis=-1)
    kl2 = np.sum(Q_e * (np.log(Q_e + eps) - np.log(M + eps)), axis=-1)
    return 0.5 * (kl1 + kl2)


def _sq_dist(A, B):
    return (
        np.sum(A ** 2, axis=1, keepdims=True)
        + np.sum(B ** 2, axis=1, keepdims=True).T
        - 2.0 * A @ B.T
    )


# -----------------------------------------------------------------------
# KPG helpers
# -----------------------------------------------------------------------

def _select_keypoints_from_plan(pi_init, n_kp):
    """Pick the top-n_kp entries of pi_init as keypoint index pairs.

    Each selected (i, j) is the location of a high-confidence match.
    We ensure each row and column appears at most once (greedy).
    """
    flat = pi_init.flatten()
    order = np.argsort(-flat)
    used_rows, used_cols = set(), set()
    I_kp, J_kp = [], []
    for idx in order:
        if len(I_kp) >= n_kp:
            break
        i, j = divmod(int(idx), pi_init.shape[1])
        if i in used_rows or j in used_cols:
            continue
        I_kp.append(i)
        J_kp.append(j)
        used_rows.add(i)
        used_cols.add(j)
    return I_kp, J_kp


def _build_mask(m, n, I_kp, J_kp):
    Mask = np.ones((m, n), dtype=np.float64)
    for idx in I_kp:
        Mask[idx, :] = 0.0
    for jdx in J_kp:
        Mask[:, jdx] = 0.0
    for idx, jdx in zip(I_kp, J_kp):
        Mask[idx, jdx] = 1.0
    return Mask


def _guiding_matrix(feat_real, feat_fake, I_kp, J_kp, tau_s=0.1, tau_t=0.1):
    C_rr = _sq_dist(feat_real, feat_real)
    C_ff = _sq_dist(feat_fake, feat_fake)
    C_rr = C_rr / (C_rr.max() + 1e-10)
    C_ff = C_ff / (C_ff.max() + 1e-10)
    R_r = _softmax_rows(-2.0 * C_rr[:, I_kp] / tau_s)
    R_f = _softmax_rows(-2.0 * C_ff[:, J_kp] / tau_t)
    return _js_matrix(R_r, R_f)


def _sinkhorn_kpg_log(p, q, C, Mask, reg=0.01, niter=500, thresh=1e-9):
    def log_kernel(u, v):
        lK = (-C + u[:, None] + v[None, :]) / reg
        lK[Mask == 0] = -1e20
        return lK

    def lse(A, axis):
        mx = np.max(A, axis=axis, keepdims=True)
        return np.log(np.exp(A - mx).sum(axis=axis, keepdims=True) + 1e-20) + mx

    u = np.zeros(len(p))
    v = np.zeros(len(q))
    for _ in range(niter):
        u_prev = u.copy()
        lk = log_kernel(u, v)
        u = reg * (np.log(p + 1e-20) - lse(lk, axis=1).squeeze()) + u
        lk = log_kernel(u, v)
        v = reg * (np.log(q + 1e-20) - lse(lk, axis=0).squeeze()) + v
        if np.linalg.norm(u - u_prev) < thresh:
            break
    return np.exp(log_kernel(u, v))


# -----------------------------------------------------------------------
# Standard OT solver (identical to baseline)
# -----------------------------------------------------------------------

def solve_ot(a, b, C_np, method, reg, tau, mass):
    if method == "OT":
        if reg == 0:
            return ot.emd(a, b, C_np)
        else:
            return ot.sinkhorn(a, b, C_np, reg=reg)
    elif method == "UOT":
        return ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_np, reg=reg, reg_m=tau)
    elif method == "POT":
        if reg == 0:
            return ot.partial.partial_wasserstein(a, b, C_np, m=mass)
        else:
            return ot.partial.entropic_partial_wasserstein(a, b, C_np, m=mass, reg=reg)


# -----------------------------------------------------------------------
# KPG-guided OT solver
# -----------------------------------------------------------------------

def solve_ot_kpg(a, b, C_np, method, reg, tau, mass,
                 feat_real, feat_fake, n_kp=5, alpha=0.5,
                 tau_s=0.1, tau_t=0.1):
    """Two-pass KPG-RL-KP solver for unsupervised generative models.

    Pass 1: solve standard OT -> pi_init
    Pass 2: select top-n_kp confident pairs from pi_init as keypoints,
             build mask + guiding matrix, solve blended problem.

    Parameters
    ----------
    feat_real  : ndarray (m, d)  real sample features (CPU numpy)
    feat_fake  : ndarray (n, d)  fake sample features
    n_kp       : int             number of keypoint pairs to select
    alpha      : float           blending coefficient
    """
    # Pass 1: initial plan
    pi_init = solve_ot(a, b, C_np, method, reg, tau, mass)
    I_kp, J_kp = _select_keypoints_from_plan(pi_init, n_kp)

    if len(I_kp) == 0:
        return pi_init

    Mask = _build_mask(len(a), len(b), I_kp, J_kp)
    G = _guiding_matrix(feat_real, feat_fake, I_kp, J_kp, tau_s, tau_t)

    C_norm = C_np / (C_np.max() + 1e-10)
    M_kpg = alpha * C_norm + (1.0 - alpha) * G

    if method == "OT" and reg > 0:
        return _sinkhorn_kpg_log(a, b, M_kpg, Mask, reg=reg)
    elif method == "OT" and reg == 0:
        M_masked = M_kpg.copy()
        M_masked[Mask == 0] = M_masked.max() * 1e3 + 1.0
        return ot.emd(a, b, M_masked)
    else:
        M_masked = M_kpg.copy()
        M_masked[Mask == 0] = M_masked.max() * 1e3 + 1.0
        return solve_ot(a, b, M_masked, method, reg, tau, mass)
