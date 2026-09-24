"""Rules of ``feasibility_checks`` (private helper of :mod:`.diagnostics`).

Split out of :mod:`synchro.model.fit.diagnostics` to keep that file short.
Each rule adds ``(name, passed, margin)`` triples to a ``_Collector`` or
records why a check is not computable from the retained moment rows; the
inequalities are the necessary conditions listed in the diagnostics module
docstring (``eq: joint moment feasible set``). Nothing here is public API.
"""

from __future__ import annotations

import numpy as np

from . import _feasibility as _fz
from ._feasibility import legendre_range


class _Collector:
    """Accumulates ``(name, passed, margin)`` triples and not-computable notes."""

    def __init__(self, tol):
        self.tol = tol
        self.checks = []
        self.missing = []

    def add(self, name, margin, scale=1.0):
        margin = float(margin)
        passed = bool(margin >= -self.tol * max(1.0, abs(float(scale))))
        self.checks.append((name, passed, margin))

    def skip(self, name, reason):
        self.missing.append(f"{name} ({reason})")


def _support_z(support, reference):
    """``(z_lo, z_hi)`` per variable of the declared support in ``z`` units."""
    gamma0, B0, depth_ref = (
        float(reference.gamma0),
        float(reference.B0),
        float(reference.depth_ref),
    )
    scales = tuple(float(s) for s in reference.scales)
    out = []
    for (lo, hi), centre, scale in zip(
        (support.gamma, support.B, support.depth), (gamma0, B0, depth_ref), scales
    ):
        out.append(((float(lo) - centre) / scale, (float(hi) - centre) / scale))
    return tuple(out)


def _support_checks(col, lookup, support, reference):
    if support is None:
        col.skip("support_mean/support_bound", "no Support supplied")
        return
    z = _support_z(support, reference)
    absmax = tuple(max(abs(lo), abs(hi)) for lo, hi in z)
    means = (("z_gamma", (0, 0, 1, 0, 0)), ("z_B", (0, 0, 0, 1, 0)))
    means += (("z_depth", (0, 0, 0, 0, 1)),)
    for (name, row), (lo, hi) in zip(means, z):
        value = lookup.get(0, row)
        if value is None:
            reason = "needs N >= 1"
            if row[4]:
                reason = "needs M0 with b = 1: not in the fitted vector"
            col.skip(f"support_mean[{name}]", reason)
            continue
        col.add(f"support_mean[{name}]", min(value - lo, hi - value), max(1, hi - lo))
    for h, rows, label in ((0, lookup.index.h0, "M0"), (2, lookup.index.h2, "M2")):
        margin, scale = np.inf, 1.0
        for row in rows:
            bound = absmax[0] ** row[2] * absmax[1] ** row[3] * absmax[2] ** row[4]
            margin = min(margin, bound - abs(lookup.get(h, row)))
            scale = max(scale, bound)
        if rows:
            col.add(f"support_bound[{label}]", margin, scale)


def _legendre_checks(col, lookup):
    t = lookup.index.truncation
    for var, L, position in (("mu", t.L_mu, 0), ("eta", t.L_eta, 1)):
        for l in range(1, L + 1):
            row = [0, 0, 0, 0, 0]
            row[position] = l
            value = lookup.get(0, tuple(row))
            if value is None:
                continue
            lo, hi = legendre_range(l)
            col.add(f"legendre_range[P_{l}({var})]", min(value - lo, hi - value))


def _matrix_checks(col, lookup):
    t = lookup.index.truncation
    explicit = _fz.REAL_CANDIDATES[:3]
    labels, H = _fz.moment_matrix(lookup, explicit)
    name = "psd[1,z_gamma,z_B]"
    if H is None or len(labels) < 3:
        col.skip(name, "needs N >= 2")
    else:
        col.add(name, _fz.min_eigenvalue(H), np.max(np.abs(np.diag(H))))
    labels, H = _fz.moment_matrix(lookup, _fz.REAL_CANDIDATES)
    if H is not None and len(labels) > 3:
        col.add(
            f"psd[{','.join(labels)}]",
            _fz.min_eigenvalue(H),
            np.max(np.abs(np.diag(H))),
        )
    elif t.N >= 2 and not lookup.has_ext():
        col.skip("psd[...,z_depth]", "needs M0 with b > 0: not in the fitted vector")
    labels, H = _fz.moment_matrix(lookup, _fz.HERMITIAN_CANDIDATES)
    if H is None:
        col.skip("hermitian_psd", "needs M2 rows with l = k = 0")
    else:
        col.add(
            f"hermitian_psd[{','.join(labels)}]",
            _fz.min_eigenvalue(H),
            np.max(np.abs(np.diag(H))),
        )


def _square_checks(col, lookup):
    """``abs_bound``: ``|<f^2 e^{2i phi}>| <= <f^2>``; ``cauchy_schwarz``:
    ``|<f e^{2i phi}>|^2 <= <f^2>`` for the monomials ``f`` of the truncation.

    Monomials in ``NAMED_F`` are listed individually when not computable;
    the others are counted in one summary entry.
    """
    skipped = 0
    for row in _fz.f_rows(lookup.index):
        label = _fz.f_label(row)
        square = _fz.product(_fz.monomial(row), _fz.monomial(row))
        second = lookup.evaluate(square, 0)
        m2_square = None if second is None else lookup.evaluate(square, 2)
        m2_row = lookup.get(2, row)
        named = label in _fz.NAMED_F
        if second is not None and m2_square is not None:
            col.add(f"abs_bound[f={label}]", second - abs(m2_square), second)
        elif named:
            reason = _missing_reason(lookup, row, second, "the M2 rows of f^2")
            col.skip(f"abs_bound[f={label}]", reason)
        else:
            skipped += 1
        if label == "1":
            continue
        if second is not None and m2_row is not None:
            col.add(f"cauchy_schwarz[f={label}]", second - abs(m2_row) ** 2, second)
        elif named:
            reason = _missing_reason(lookup, row, second, "the M2 row of f")
            col.skip(f"cauchy_schwarz[f={label}]", reason)
        else:
            skipped += 1
    if skipped:
        col.missing.append(
            f"abs_bound/cauchy_schwarz for {skipped} further monomials (need "
            "moment rows outside the retained index)"
        )


def _missing_reason(lookup, row, second, what):
    l, k, r, s, b = row
    if second is not None:
        return f"{what} not retained (parity or cutoff)"
    if b and not lookup.has_ext():
        return f"needs M0 with b = {2 * b}: not in the fitted vector"
    return f"needs M0 rows with (l,k) = ({2 * l},{2 * k}) and r+s+b = {2 * (r + s + b)}"


__all__ = [
    "_Collector",
    "_support_checks",
    "_legendre_checks",
    "_matrix_checks",
    "_square_checks",
]
