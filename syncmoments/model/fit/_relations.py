"""Relation generation and ``C = H T`` assembly of ``reduce_response`` (private).

Continuum scaling identity
--------------------------
``ContinuumKernel`` evaluates ``K_I = A_theta F(x)``, ``K_Q = -A_theta G(x)``
with ``A_theta`` proportional to ``B_perp``, ``x = nu / (a_B gamma^2)``,
``a_B`` proportional to ``B_perp`` and ``B_perp = B sqrt(1 - eta^2)``. Hence
``K = B Phi_{nu,eta}(B gamma^2)`` and, for every ``lambda > 0``,

    K(gamma / sqrt(lambda), lambda B) = lambda K(gamma, B).            (H)

The clamp at ``x = 1e4``, the ``F``/``G`` quadrature and the ``x_min`` check
act on ``x`` alone, so (H) holds for the implemented function. Differentiating
at ``lambda = 1`` gives the Euler relation

    B dK/dB - (gamma / 2) dK/dgamma - K = 0.                           (E)

Its coefficients depend on ``(gamma, B)`` only. The eta projection on fixed
nodes, the channel average against a fixed response, the phase weights
``w_b(tau; depth_ref, s_depth)`` (the ``PhaseWeights`` call signature has no
``gamma`` or ``B``) and the real layout are linear and free of ``gamma`` and
``B``, so (E) holds in every block ``(h, part, l, k, b)``. With
``z_g = (gamma - gamma0)/s_gamma``, ``z_B = (B - B0)/s_B``, ``eps_g =
s_gamma/gamma0``, ``eps_B = s_B/B0`` and the columns ``c_rs`` of ``eq:
channel derivative coefficients`` (the ``1/(r! s!)`` convention), the
``z_g^r z_B^s`` coefficient of (E) is

    (s+1)/eps_B c[r,s+1] - (r+1)/(2 eps_g) c[r+1,s] - (1 + r/2 - s) c[r,s] = 0.   (R)

One relation is generated per block and ``(r, s)`` when ``(r, s)``,
``(r+1, s)`` and ``(r, s+1)`` are all retained; each owns its ``(r, s+1)``
slot, so the relations are independent. For an uncapped block of degree
``D`` there are ``D (D+1)/2`` relations and ``D + 1`` groups; the ``D``-jet of
``B Phi(B gamma^2)`` is fixed by ``Phi^(0..D)``, so no further linear relation
holds for a generic ``Phi`` (the set is complete). Capped blocks are checked
on listed cases only (``tests/model/test_reduction.py``).

Other helpers: structural zero relations, residual checks, redundancy
pruning, deterministic representative selection, ``T``/``H`` assembly,
the Euclidean factorisation ``T = L Q`` (``lq_factor``, ``lq_qr``,
``lq_svd``) and numerical column-relation candidates. Nothing here is public
API; the contracts are stated in ``syncmoments.model.fit.reduction``.
"""

from __future__ import annotations

import math

import numpy as np

EPS = np.finfo(float).eps
RANK_RTOL = 1e-12
CLEAN_RTOL = 1e-15
SOURCES = ("structural", "analytic", "declared", "approximate")


def kernel_name(basis):
    """The ``("name", ...)`` entry of ``basis.provenance.kernel`` or ``None``."""
    record = getattr(getattr(basis, "provenance", None), "kernel", None) or ()
    try:
        return dict(record).get("name")
    except (TypeError, ValueError):
        return None


def structural_relations(layout):
    """Unit rows ``C_j = 0`` for every structural slot: ``[(weights, source, scope)]``."""
    return [
        (((slot, 1.0),), "structural", reason)
        for slot, reason in zip(layout.structural_zero, layout.structural_reasons)
    ]


def fractional_scales(layout):
    """``(eps_g, eps_B) = (s_gamma/gamma0, s_B/B0)``; ``ValueError`` unless positive finite."""
    eps_g = layout.value("s_gamma") / layout.value("gamma0")
    eps_B = layout.value("s_B") / layout.value("B0")
    for name, value in (("eps_g", eps_g), ("eps_B", eps_B)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} = {value} must be positive and finite")
    return eps_g, eps_B


def continuum_relations(layout):
    """Relation (R) in every block of the non-structural slots (module docstring)."""
    eps_g, eps_B = fractional_scales(layout)
    skip = set(layout.structural_zero)
    blocks = {}
    for e in layout.entries:
        if e.slot not in skip:
            blocks.setdefault((e.h, e.part, e.l, e.k, e.b), {})[(e.r, e.s)] = e.slot
    scope = (
        "continuum scaling identity (E) in coefficient form (R), exact for "
        "K = B Phi(B gamma^2) at eps_g = s_gamma/gamma0 = "
        f"{eps_g:.6g}, eps_B = s_B/B0 = {eps_B:.6g}"
    )
    out = []
    for block in blocks.values():
        for (r, s), slot in sorted(block.items()):
            up_s, up_r = block.get((r, s + 1)), block.get((r + 1, s))
            if up_s is None or up_r is None:
                continue
            weights = [
                (up_s, (s + 1) / eps_B),
                (up_r, -(r + 1) / (2.0 * eps_g)),
                (slot, -(1.0 + r / 2.0 - s)),
            ]
            out.append(
                (tuple((j, w) for j, w in weights if w != 0.0), "analytic", scope)
            )
    return out


def residual(C, norms, weights) -> float:
    """``||C w||_inf / sum_j |w_j| ||C_j||_inf`` with ``0/0 := 0``."""
    num = float(np.max(np.abs(sum(w * C[:, j] for j, w in weights)), initial=0.0))
    den = float(sum(abs(w) * norms[j] for j, w in weights))
    if num == 0.0:
        return 0.0
    return num / den if den > 0.0 else math.inf


def check_relations(C, relations, check_rtol, context):
    """Residual of each relation, raising ``ValueError`` per the source rules."""
    norms = np.max(np.abs(C), axis=0, initial=0.0)
    out = []
    for weights, source, scope in relations:
        rho = residual(C, norms, weights)
        slots = [j for j, _ in weights]
        if source == "structural" and rho != 0.0:
            raise ValueError(
                f"structural slot {slots[0]} has a nonzero response column ({scope})"
            )
        if source in ("analytic", "declared") and not rho <= check_rtol:
            extra = (
                "; declare it approximate or correct it"
                if source == "declared"
                else f"; kernel {context}"
            )
            raise ValueError(
                f"{source} relation on slots {slots} fails its residual check: "
                f"rho = {rho:.3g} > check_rtol = {check_rtol:.3g}{extra}"
            )
        out.append(rho)
    return tuple(out)


def relation_matrix(relations, n_full):
    """``(n_rel, n_full)`` rows scaled to unit Euclidean norm."""
    R = np.zeros((len(relations), n_full))
    for i, (weights, _, _) in enumerate(relations):
        for j, w in weights:
            R[i, j] += w
    norms = np.linalg.norm(R, axis=1)
    return R / np.where(norms > 0, norms, 1.0)[:, None]


def _orthogonalise(v, basis):
    for _ in range(2):  # twice is enough (Kahan)
        for b in basis:
            v = v - (b @ v) * b
    return v


def prune(R):
    """Row Gram-Schmidt in stacking order: ``(kept, dropped)`` row indices."""
    basis, kept, dropped = [], [], []
    for i, row in enumerate(R):
        v = _orthogonalise(row.copy(), basis)
        norm = float(np.linalg.norm(v))
        if norm > RANK_RTOL * max(float(np.linalg.norm(row)), 1e-300):
            basis.append(v / norm)
            kept.append(i)
        else:
            dropped.append(i)
    return kept, dropped


def preference_key(layout, representatives):
    """Sort key of a slot, most preferred first; structural zeros forced last."""
    listed = (
        {}
        if representatives == "gamma"
        else {t: i for i, t in enumerate(representatives)}
    )
    structural = set(layout.structural_zero)
    entries = layout.entries

    def key(slot):
        e = entries[slot]
        rank = listed.get((e.r, e.s, e.b), len(listed))
        return (slot in structural, rank, e.s, e.r + e.s + e.b, slot)

    return key


def select(R, layout, representatives):
    """``(rep, eliminated)``: greedy walk from the least preferred slot (module docstring)."""
    n_full = R.shape[1]
    target = R.shape[0]
    order = sorted(range(n_full), key=preference_key(layout, representatives))
    basis, eliminated = [], []
    for j in reversed(order):
        if len(eliminated) == target:
            break
        column = R[:, j]
        norm = float(np.linalg.norm(column))
        if norm == 0.0:
            continue
        v = _orthogonalise(column.copy(), basis)
        if float(np.linalg.norm(v)) > RANK_RTOL * norm:
            basis.append(v / np.linalg.norm(v))
            eliminated.append(j)
    if len(eliminated) != target:
        raise ValueError("the relation set does not determine an elimination set")
    rep = tuple(sorted(set(range(n_full)) - set(eliminated)))
    return rep, tuple(sorted(eliminated))


def assemble(C, R, rep, eliminated):
    """``(T, H, amplification, n_cleaned)`` with ``T[:, rep] = I`` and ``C ~= H T``.

    ``amplification = max_d sum_i ||R_i||_1 |R_D^-1|_{d i}`` bounds how the
    relation residuals enter ``C - H T`` (``delta_C_D = E R_D^-T``).
    """
    n_full = R.shape[1]
    T = np.zeros((len(rep), n_full))
    T[np.arange(len(rep)), list(rep)] = 1.0
    amplification = 1.0
    if eliminated:
        R_D, R_rep = R[:, list(eliminated)], R[:, list(rep)]
        inverse = np.linalg.solve(R_D, np.eye(R_D.shape[0]))
        T[:, list(eliminated)] = (-(inverse @ R_rep)).T
        row_l1 = np.sum(np.abs(R), axis=1)
        amplification = max(1.0, float(np.max(np.abs(inverse) @ row_l1)))
    small = (np.abs(T) <= CLEAN_RTOL * np.max(np.abs(T), initial=0.0)) & (T != 0.0)
    T[small] = 0.0
    return T, C[:, list(rep)], amplification, int(np.sum(small))


def lq_qr(T):
    """``(L, Q)`` from ``numpy.linalg.qr(T.T)``, ``diag(L) > 0``; full row rank only."""
    T = np.asarray(T, dtype=float)
    if T.shape[0] == 0:
        return np.zeros((0, 0)), np.zeros((0, T.shape[1]))
    Q0, R0 = np.linalg.qr(T.T)
    signs = np.where(np.diag(R0) < 0, -1.0, 1.0)
    return R0.T * signs[None, :], signs[:, None] * Q0.T


def lq_svd(T, rank=None):
    """``(L, Q) = (U_r S_r, V_r^T)`` from the thin SVD; any row rank."""
    T = np.asarray(T, dtype=float)
    if T.shape[0] == 0:
        return np.zeros((0, 0)), np.zeros((0, T.shape[1]))
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    if rank is None:
        rank = int(np.sum(S > RANK_RTOL * S[0])) if S.size and S[0] > 0 else 0
    return U[:, :rank] * S[:rank], Vt[:rank]


def row_rank(T):
    """Rank of ``T`` from its singular values at ``RANK_RTOL`` times the largest."""
    T = np.asarray(T, dtype=float)
    if T.size == 0:
        return 0
    S = np.linalg.svd(T, compute_uv=False)
    return int(np.sum(S > RANK_RTOL * S[0])) if S[0] > 0 else 0


def lq_factor(T):
    """``(L, Q, method)``: ``lq_qr`` at full row rank, else ``lq_svd`` (``"svd"``)."""
    rank = row_rank(T)
    if rank == np.shape(T)[0]:
        return (*lq_qr(T), "qr")
    return (*lq_svd(T, rank), "svd")


def null_basis(T):
    """Orthonormal ``(n, n - rank T)`` basis of ``null(T)``."""
    T = np.asarray(T, dtype=float)
    n = T.shape[1]
    if T.shape[0] == 0:
        return np.eye(n)
    _, S, Vt = np.linalg.svd(T, full_matrices=True)
    rank = int(np.sum(S > RANK_RTOL * S[0])) if S.size and S[0] > 0 else 0
    return Vt[rank:].T


def candidates(C, skip, rtol):
    """Zero, identical and proportional columns: ``[(kind, i, j, ratio)]``."""
    norms = np.max(np.abs(C), axis=0, initial=0.0)
    scale = float(np.max(norms, initial=0.0))
    active = [j for j in range(C.shape[1]) if j not in skip]
    zero = [j for j in active if norms[j] <= rtol * scale]
    out = [("zero", j, None, None) for j in zero]
    live = [j for j in active if j not in zero]
    for a, i in enumerate(live):
        for j in live[:a]:
            cj = C[:, j]
            ratio = float(C[:, i] @ cj / (cj @ cj))
            if np.max(np.abs(C[:, i] - ratio * cj)) <= rtol * norms[i]:
                kind = "identical" if abs(ratio - 1.0) <= rtol else "proportional"
                out.append((kind, i, j, ratio))
    return out


__all__ = [
    "kernel_name",
    "structural_relations",
    "fractional_scales",
    "continuum_relations",
    "residual",
    "check_relations",
    "relation_matrix",
    "prune",
    "select",
    "assemble",
    "lq_qr",
    "lq_svd",
    "lq_factor",
    "row_rank",
    "null_basis",
    "candidates",
    "SOURCES",
]
