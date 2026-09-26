"""SVD, clusters and retained quantities of ``fit_combinations`` (private).

With ``a = a_off + J x``, the metric ``M_x = J^T M J = R_x^T R_x`` and the
Euclidean factorisation ``T J R_x^-1 = L Q``, the fitted model is
``R H T a = R H T a_off + R (H L) Q (R_x x)``. The whitened design
``G_w = L_Sigma^-1 R H L`` on the kept rows is decomposed by SVD; clusters of
repeated singular values are kept or dropped whole and receive a canonical
basis that depends on the observation only (pivoted Gram-Schmidt on the
cluster projector in the metric coordinates ``x~ = R_x x``); signs make the
largest component of each direction in ``x`` positive. Formulas: design
Section 5.4 (``docs/DESIGN.md`` Section 10.1). Nothing here is public API.

The estimator uses the SVD's own left vectors: with ``V_R = V_svd O``
(``O`` block-orthogonal, ``+-1`` outside clusters), ``K_beta = O^T D^-1
U_R^T L_Sigma^-1``, so ``K = W K_beta`` is the truncated pseudo-inverse and
``K C = Pi`` holds to ``O(u cond)``. Rebuilding ``U_R = G_w V_R / D`` would
square ``cond``. Inside a cluster only the subspace is determined: the
``beta`` of a cluster are correlated by at most ``(rho^2 - 1)/2``, ``rho``
the cluster's ``s_max/s_min``, and each such correlation is noted.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._identifiability import whitened_design
from ._relations import lq_factor, null_basis

TIE_RTOL = 1e-12  # sign rule: ties of the largest component
PIVOT_TIE_RTOL = 1e-8  # floor of the pivot tie tolerance of a cluster basis
SUBSPACE_ROUNDING = 1e3  # SVD subspace rounding, in units of u s_max / separation
RETAINED, WEAK, NULL = "retained", "weak", "numerical_null"


class Solution(NamedTuple):
    """Every array of the fit, NumPy float64."""

    R_x: np.ndarray
    M_x: np.ndarray
    L: np.ndarray
    Q: np.ndarray
    lq_method: str
    s: np.ndarray
    classes: tuple
    clusters: tuple
    W: np.ndarray
    B: np.ndarray
    K_beta: np.ndarray
    K: np.ndarray
    beta_hat: np.ndarray
    beta_sigma: np.ndarray
    x_hat: np.ndarray
    a_hat: np.ndarray
    Pi: np.ndarray
    Cov: np.ndarray
    chi2: float
    dof: int
    kept_rows: tuple
    directions: tuple
    stokes_hat: np.ndarray
    stokes_sigma: np.ndarray
    notes: tuple


def metric_factor(J, M):
    """``(M_x, R_x)`` with ``M_x = J^T M J = R_x^T R_x`` (``R_x`` upper triangular)."""
    M_x = J.T @ M @ J
    M_x = 0.5 * (M_x + M_x.T)
    try:
        lower = np.linalg.cholesky(M_x)
    except np.linalg.LinAlgError:
        lower = None
    if lower is None or not np.all(np.isfinite(lower)):
        raise ValueError("the parameter map has redundant parameters under this metric")
    diag = np.abs(np.diag(lower))
    if diag.size and diag.min() <= 1e-12 * diag.max():
        raise ValueError("the parameter map has redundant parameters under this metric")
    return M_x, lower.T


def find_clusters(s, cluster_rtol):
    """Maximal runs (length >= 2) of nonzero ``s`` with gaps ``<= cluster_rtol s_max``."""
    gap = cluster_rtol * s[0]
    out, run = [], [0]
    for i in range(1, len(s)):
        if s[i] > 0 and s[i - 1] - s[i] <= gap:
            run.append(i)
        else:
            if len(run) > 1:
                out.append(tuple(run))
            run = [i]
    if len(run) > 1 and s[run[-1]] > 0:
        out.append(tuple(run))
    return tuple(out)


def classify(s, clusters, *, retain_at, null_below, notes):
    """Class of every singular value; a cluster is decided by its minimum."""
    members = {i: (i,) for i in range(len(s))}
    for c in clusters:
        for i in c:
            members[i] = c
    classes = []
    for i in range(len(s)):
        group = members[i]
        low, high = min(s[j] for j in group), max(s[j] for j in group)
        if low > null_below and low >= retain_at:
            classes.append(RETAINED)
        elif low > null_below:
            classes.append(WEAK)
        else:
            classes.append(NULL)
        straddles = {
            "1/max_sigma": low < retain_at <= high,  # retained when s >= threshold
            "rank_tol s_max": low <= null_below < high,  # nonzero when s > threshold
        }
        if i == group[0] and len(group) > 1:
            for name, hit in straddles.items():
                if hit:
                    notes.append(
                        f"cluster {list(group)} straddles {name}; decided as a whole "
                        "by its minimum singular value"
                    )
    return tuple(classes)


def pivoted_basis(P, k, tie_rtol=PIVOT_TIE_RTOL):
    """``k`` orthonormal columns spanning ``range(P)``, pivoting on the largest column.

    Columns within ``tie_rtol`` (relative) of the largest norm are tied and
    the lowest index wins, so rounding below ``tie_rtol`` (for example two
    LAPACK builds on a symmetric pair) does not change the pivot. Only the
    subspace of a (near-)degenerate set is meaningful; this fixes a basis.
    """
    cols = np.array(P, dtype=float)
    out = []
    for _ in range(k):
        norms = np.linalg.norm(cols, axis=0)
        top = norms.max()
        j = int(np.flatnonzero(norms >= (1.0 - tie_rtol) * top)[0])
        v = cols[:, j] / norms[j]
        for u in out:  # re-orthogonalise against earlier picks
            v = v - (u @ v) * u
        v = v / np.linalg.norm(v)
        out.append(v)
        cols = cols - np.outer(v, v @ cols)
    return np.column_stack(out) if out else np.zeros((P.shape[0], 0))


def _sign_fixed(Wx):
    """Flip each column so its largest-magnitude component (lowest index on ties) is positive."""
    signs = np.ones(Wx.shape[1])
    for i in range(Wx.shape[1]):
        col = np.abs(Wx[:, i])
        j = (
            int(np.flatnonzero(col >= (1.0 - TIE_RTOL) * col.max())[0])
            if col.size
            else 0
        )
        if col.size and Wx[j, i] < 0:
            signs[i] = -1.0
    return signs


def pivot_tie_rtol(s, cluster):
    """``max(PIVOT_TIE_RTOL, SUBSPACE_ROUNDING u s_max / sep)``, at most 0.5.

    ``sep`` is the distance of the cluster to the rest of the spectrum; by
    Davis-Kahan the computed cluster subspace, hence its projector's column
    norms, is determined only to about ``u s_max / sep``.
    """
    lo, hi = cluster[0], cluster[-1]
    seps = [s[lo - 1] - s[lo]] if lo > 0 else []
    seps += [s[hi] - s[hi + 1]] if hi + 1 < len(s) else []
    sep = min(seps) if seps else np.inf
    if sep <= 0.0:
        return 0.5
    u = 0.5 * np.finfo(float).eps
    return float(min(0.5, max(PIVOT_TIE_RTOL, SUBSPACE_ROUNDING * u * s[0] / sep)))


def canonical(V, s, clusters, Q, G_w, null_below):
    """Canonical cluster bases (observation only) and ``s_i := ||G_w v_i||`` inside clusters.

    Clusters whose minimum is at or below ``null_below`` are numerical null
    (they carry no fitted number) and keep the SVD basis, which bounds the
    cost for large null clusters.
    """
    s_svd = s
    V, s = V.copy(), s.copy()
    for c in clusters:
        idx = list(c)
        if min(s[i] for i in idx) <= null_below:
            continue
        Wt = Q.T @ V[:, idx]  # x~ coordinates
        basis = pivoted_basis(Wt @ Wt.T, len(idx), pivot_tie_rtol(s_svd, idx))
        V[:, idx] = Q @ basis
        s[idx] = np.linalg.norm(G_w @ V[:, idx], axis=0)
    return V, s


def _whitened(fit_data, H, L, HT, a_off):
    import jax.numpy as jnp

    G_w = np.asarray(whitened_design(fit_data, jnp.asarray(H), jnp.asarray(L)))
    target, Linv = (np.asarray(v) for v in fit_data.whitened())
    offset = np.asarray(fit_data.design(jnp.asarray((HT @ a_off)[:, None])))[:, 0]
    return G_w, Linv, target - Linv @ offset


def rotation(V_svd, V, retained, clusters):
    """``O`` with ``V[:, retained] = V_svd[:, retained] O``: +-1 outside clusters, a block inside."""
    pos = {i: k for k, i in enumerate(retained)}
    O = np.zeros((len(retained), len(retained)))
    for i in retained:
        O[pos[i], pos[i]] = np.sign(V_svd[:, i] @ V[:, i])
    for c in clusters:
        if c[0] not in pos:
            continue
        k = [pos[i] for i in c]  # classify keeps or drops a cluster whole
        O[np.ix_(k, k)] = V_svd[:, list(c)].T @ V[:, list(c)]
    return O


def cluster_notes(s_svd, clusters, retained, cov_beta):
    """Largest ``beta`` correlation of every retained cluster and its bound ``(rho^2 - 1)/2``."""
    pos = {i: k for k, i in enumerate(retained)}
    notes = []
    for c in clusters:
        if c[0] not in pos:
            continue
        k = [pos[i] for i in c]
        block = cov_beta[np.ix_(k, k)]
        sd = np.sqrt(np.diag(block))
        corr = np.abs(block / np.outer(sd, sd) - np.eye(len(k))).max()
        rho = float(s_svd[c[0]] / s_svd[c[-1]])
        notes.append(
            f"cluster {list(c)}: beta correlation up to {corr:.2g}, against "
            f"(rho^2 - 1)/2 = {0.5 * (rho**2 - 1.0):.2g} in exact arithmetic "
            f"(rho = s_max/s_min = {rho:.12g} inside the cluster); K, Pi and "
            "Cov do not depend on the cluster basis"
        )
    return notes


def solve(reduction, fit_data, J, a_off, M, *, max_sigma, rank_tol, cluster_rtol):
    """The retained, weak and null structure and every estimator array (module docstring)."""
    T, H = np.asarray(reduction.T), np.asarray(reduction.H)
    notes = []
    M_x, R_x = metric_factor(J, M)
    Rx_inv = np.linalg.solve(R_x, np.eye(R_x.shape[0]))
    L, Q, method = lq_factor(T @ J @ Rx_inv)
    if method == "svd":
        notes.append("T J R_x^-1 is row-rank deficient: thin-SVD factorisation")
    r_T, n_x = Q.shape
    HT = H @ T
    G_w, Linv, y = _whitened(fit_data, H, L, HT, a_off)
    if r_T == 0:
        raise ValueError("the whitened design has no finite nonzero singular value")
    U, s_raw, Vt = np.linalg.svd(G_w, full_matrices=True)
    s_svd = np.zeros(r_T)
    s_svd[: s_raw.size] = s_raw
    if not np.all(np.isfinite(s_svd)) or s_svd[0] <= 0.0:
        raise ValueError("the whitened design has no finite nonzero singular value")
    clusters = find_clusters(s_svd, cluster_rtol)
    V, s = canonical(Vt.T, s_svd, clusters, Q, G_w, rank_tol * s_svd[0])
    classes = classify(
        s,
        clusters,
        retain_at=(1.0 / max_sigma if max_sigma > 0 else np.inf),
        null_below=rank_tol * s_svd[0],
        notes=notes,
    )
    if clusters:
        notes.append(
            f"clusters {[list(c) for c in clusters]}: canonical bases from the "
            "observation; only the subspace of a cluster is determined"
        )
    V = V * _sign_fixed(Rx_inv @ Q.T @ V)[None, :]
    sel = {
        k: [i for i, c in enumerate(classes) if c == k] for k in (RETAINED, WEAK, NULL)
    }
    ret = sel[RETAINED]
    if not ret:
        notes.append("no mode meets max_sigma: beta is empty, K = 0, Pi = 0")
    V_R, U_R = V[:, ret], U[:, ret]
    F = rotation(Vt.T, V, ret, clusters).T / s_svd[ret][None, :]  # O^T D^-1
    W = Rx_inv @ Q.T @ V_R
    WF = W @ F  # = R_x^-1 Q^T V_svd D^-1
    uy = U_R.T @ y
    K_beta = F @ U_R.T @ Linv
    beta_hat = F @ uy
    x_hat = W @ beta_hat
    residual = y - U_R @ uy
    notes += cluster_notes(s_svd, clusters, ret, F @ F.T)
    null_an = null_basis(Q) if n_x > r_T else np.zeros((n_x, 0))  # SVD complement
    an = Rx_inv @ null_an
    an = an * _sign_fixed(an)[None, :]
    return Solution(
        R_x=R_x,
        M_x=M_x,
        L=L,
        Q=Q,
        lq_method=method,
        s=s,
        classes=classes,
        clusters=clusters,
        W=W,
        B=V_R.T @ Q @ R_x,
        K_beta=K_beta,
        K=W @ K_beta,
        beta_hat=beta_hat,
        beta_sigma=np.sqrt(np.sum(F**2, axis=1)),
        x_hat=x_hat,
        a_hat=a_off + J @ x_hat,
        Pi=W @ (V_R.T @ Q @ R_x),
        Cov=WF @ WF.T,
        chi2=float(residual @ residual),
        dof=G_w.shape[0] - len(ret),
        kept_rows=tuple(fit_data.kept_rows()),
        directions=(
            W,
            Rx_inv @ Q.T @ V[:, sel[WEAK]],
            Rx_inv @ Q.T @ V[:, sel[NULL]],
            an,
        ),
        stokes_hat=(HT @ (a_off + J @ x_hat)).reshape(-1, 4),
        stokes_sigma=np.sqrt(np.sum((HT @ J @ WF) ** 2, axis=1)).reshape(-1, 4),
        notes=tuple(notes),
    )


__all__ = [
    "Solution",
    "solve",
    "metric_factor",
    "find_clusters",
    "classify",
    "pivoted_basis",
    "pivot_tie_rtol",
    "rotation",
]
