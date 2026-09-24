"""Moment algebra behind ``feasibility_checks`` (private helper).

Split out of :mod:`synchro.model.fit.diagnostics` to keep that file short.
Everything here is eager NumPy on concrete moment values: a moment vector is
looked up row by row (``None`` for rows outside the retained index), products
of the basis monomials ``P_l(mu) P_k(eta) z_gamma^r z_B^s z_depth^b`` are
expanded back onto the retained rows with the Legendre linearisation
``P_a P_b = sum_c g_abc P_c`` (``numpy.polynomial.legendre.legmul``), and the
moment matrices ``<conj(f_i) f_j>`` of a candidate list of functions are
assembled over the largest prefix-greedy subset whose products are all
retained. Nothing here certifies feasibility; the inequalities it evaluates
are necessary conditions on ``eq: joint moment feasible set``.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import legendre as npleg

Row = tuple[int, int, int, int, int]
Expansion = dict[Row, float]

_UNIT: Expansion = {(0, 0, 0, 0, 0): 1.0}
_Z_GAMMA: Expansion = {(0, 0, 1, 0, 0): 1.0}
_Z_B: Expansion = {(0, 0, 0, 1, 0): 1.0}
_Z_DEPTH: Expansion = {(0, 0, 0, 0, 1): 1.0}
_P1_MU: Expansion = {(1, 0, 0, 0, 0): 1.0}
_P1_ETA: Expansion = {(0, 1, 0, 0, 0): 1.0}
_P1_P1: Expansion = {(1, 1, 0, 0, 0): 1.0}

REAL_CANDIDATES = (
    ("1", _UNIT, 0),
    ("z_gamma", _Z_GAMMA, 0),
    ("z_B", _Z_B, 0),
    ("P_1(mu)", _P1_MU, 0),
    ("P_1(eta)", _P1_ETA, 0),
    ("z_depth", _Z_DEPTH, 0),
)
HERMITIAN_CANDIDATES = (
    ("1", _UNIT, 0),
    ("z_gamma", _Z_GAMMA, 0),
    ("z_B", _Z_B, 0),
    ("z_depth", _Z_DEPTH, 0),
    ("P_1(mu) P_1(eta)", _P1_P1, 0),
    ("e^{2i phi}", _UNIT, 2),
    ("z_gamma e^{2i phi}", _Z_GAMMA, 2),
    ("z_B e^{2i phi}", _Z_B, 2),
    ("z_depth e^{2i phi}", _Z_DEPTH, 2),
    ("P_1(mu) P_1(eta) e^{2i phi}", _P1_P1, 2),
)
NAMED_F = ("1", "z_gamma", "z_B", "z_depth", "P_1(mu)", "P_1(eta)", "P_1(mu) P_1(eta)")


class MomentLookup:
    """Concrete retained moments with ``None`` for rows outside the index.

    ``get(h, row)`` returns ``M0`` rows (``b = 0`` from ``m0``, ``b >= 1``
    from ``m0_ext`` when present) for ``h = 0`` and complex ``M2`` rows for
    ``h = 2``.
    """

    def __init__(self, moments, index):
        self.index = index
        self.m0 = np.asarray(moments.m0, dtype=float)
        self.m2 = np.asarray(moments.m2, dtype=complex)
        ext = moments.m0_ext
        self.ext = None if ext is None else np.asarray(ext, dtype=float)
        self.pos0 = {row: i for i, row in enumerate(index.h0)}
        self.pos_ext = {row: i for i, row in enumerate(index.h0_ext)}
        self.pos2 = {row: i for i, row in enumerate(index.h2)}

    def has_ext(self) -> bool:
        return self.ext is not None

    def get(self, h, row):
        row = tuple(int(v) for v in row)
        if h == 0 and row[4] == 0:
            i = self.pos0.get(row)
            return None if i is None else float(self.m0[i])
        if h == 0:
            i = self.pos_ext.get(row)
            if i is None or self.ext is None:
                return None
            return float(self.ext[i])
        i = self.pos2.get(row)
        return None if i is None else complex(self.m2[i])

    def evaluate(self, expansion: Expansion, h: int):
        """``sum coeff * M^(h)[row]``; ``None`` when any needed row is missing."""
        total = 0.0
        for row, coeff in expansion.items():
            if coeff == 0.0:
                continue
            value = self.get(h, row)
            if value is None:
                return None
            total = total + coeff * value
        return total


def _legendre_product(a: int, b: int) -> dict[int, float]:
    """Coefficients ``g_c`` of ``P_a P_b = sum_c g_c P_c``."""
    ea = np.zeros(a + 1)
    ea[a] = 1.0
    eb = np.zeros(b + 1)
    eb[b] = 1.0
    coeffs = npleg.legmul(ea, eb)
    return {c: float(g) for c, g in enumerate(coeffs) if abs(g) > 1e-15}


def product(e1: Expansion, e2: Expansion) -> Expansion:
    """Expansion of the product of two expansions on the row monomials."""
    out: Expansion = {}
    for (l1, k1, r1, s1, b1), c1 in e1.items():
        for (l2, k2, r2, s2, b2), c2 in e2.items():
            for l, gl in _legendre_product(l1, l2).items():
                for k, gk in _legendre_product(k1, k2).items():
                    row = (l, k, r1 + r2, s1 + s2, b1 + b2)
                    out[row] = out.get(row, 0.0) + c1 * c2 * gl * gk
    return out


def monomial(row: Row) -> Expansion:
    return {tuple(int(v) for v in row): 1.0}


def f_label(row: Row) -> str:
    """Printed name of ``P_l(mu) P_k(eta) z_gamma^r z_B^s z_depth^b`` (``"1"`` for the unit)."""
    l, k, r, s, b = row
    parts = []
    if l:
        parts.append(f"P_{l}(mu)")
    if k:
        parts.append(f"P_{k}(eta)")
    for name, e in (("z_gamma", r), ("z_B", s), ("z_depth", b)):
        if e:
            parts.append(name + (f"^{e}" if e > 1 else ""))
    return " ".join(parts) if parts else "1"


def f_rows(index) -> tuple[Row, ...]:
    """All ``(l, k, r, s, b)`` monomials of the truncation, both parities.

    Rows follow the ``P`` cutoff (``r+s+b <= N`` or the ``app: depth moments`` layout)
    over every ``(l, k)`` pair with ``l <= L_mu, k <= L_eta``.
    """
    t = index.truncation
    rows = []
    for l in range(t.L_mu + 1):
        for k in range(t.L_eta + 1):
            for r in range(t.N + 1):
                for s in range(t.N + 1 - r):
                    if t.depth_degree is None:
                        b_max = t.N - r - s
                    else:
                        b_max = t.depth_degree
                    rows.extend((l, k, r, s, b) for b in range(b_max + 1))
    return tuple(rows)


def moment_matrix(lookup: MomentLookup, candidates):
    """Greedy maximal moment matrix ``H_ij = <conj(f_i) f_j>``.

    ``candidates`` is a sequence of ``(label, expansion, h)`` with ``h`` in
    ``{0, 2}``; a candidate is kept when its products with itself and every
    kept candidate are retained moments. Returns ``(labels, H)`` with ``H``
    Hermitian (real symmetric when all ``h = 0``), or ``(labels, None)`` when
    fewer than two candidates survive.
    """
    kept: list[tuple[str, Expansion, int]] = []
    for label, expansion, h in candidates:
        trial = kept + [(label, expansion, h)]
        if all(
            _entry(lookup, a, b) is not None
            for a in trial
            for b in trial
            if a is trial[-1] or b is trial[-1]
        ):
            kept = trial
    labels = tuple(label for label, _, _ in kept)
    if len(kept) < 2:
        return labels, None
    n = len(kept)
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            H[i, j] = _entry(lookup, kept[i], kept[j])
    if all(h == 0 for _, _, h in kept):
        H = H.real
    return labels, H


def _entry(lookup, fi, fj):
    """``<conj(f_i) f_j>`` for ``f = g e^{i h phi}`` with real ``g``."""
    _, gi, hi = fi
    _, gj, hj = fj
    expansion = product(gi, gj)
    delta = hj - hi
    if delta == 0:
        return lookup.evaluate(expansion, 0)
    value = lookup.evaluate(expansion, 2)
    if value is None:
        return None
    return value if delta > 0 else np.conj(value)


def min_eigenvalue(H) -> float:
    return float(np.linalg.eigvalsh((H + H.conj().T) / 2)[0])


def legendre_range(l: int) -> tuple[float, float]:
    """``(min, max)`` of ``P_l`` on ``[-1, 1]``: interior critical points
    from the roots of ``P_l'`` plus the endpoints; the maximum is ``1``.

    Dimensionless; a floating-point NumPy root computation, compared with a
    dense grid in the tests for ``l <= 8`` (a finite check, not a
    certificate); no physical assumption. It states the range of ``P_l``
    only, nothing about a population.
    """
    l = int(l)
    if l < 0:
        raise ValueError("l must be a nonnegative integer")
    if l == 0:
        return 1.0, 1.0
    unit = np.zeros(l + 1)
    unit[l] = 1.0
    critical = npleg.legroots(npleg.legder(unit)) if l >= 2 else np.zeros(0)
    points = np.concatenate([[-1.0, 1.0], np.real(critical)])
    values = npleg.legval(points, unit)
    return float(values.min()), 1.0


__all__ = [
    "MomentLookup",
    "REAL_CANDIDATES",
    "HERMITIAN_CANDIDATES",
    "NAMED_F",
    "product",
    "monomial",
    "f_label",
    "f_rows",
    "moment_matrix",
    "min_eigenvalue",
    "legendre_range",
]
