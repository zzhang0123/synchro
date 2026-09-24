"""Identifiability and feasibility diagnostics for spectral fits.

LABEL: ``extra eq: finite fit model`` (the whitened design ``L^-1 R C [c | P]``
whose null space makes admissible moments indistinguishable, extended
discussion: "moments separated by ``v`` with ``C v = 0`` are
indistinguishable; if other parameters are fitted, use the full response
Jacobian") and ``eq: joint moment feasible set`` (necessary conditions on
``m in conv{psi(x): x in D}``).

``identifiability`` takes the SVD of the whitened design restricted to the
free directions of a ``ParameterMap``: for an affine map ``m = P theta + c``
the columns are ``[c | P]`` (unknowns ``u = (A, A theta)``); for a nonlinear
map the columns are the Jacobian of ``m`` at a supplied ``theta`` (a local
statement only). The columns are equilibrated to unit norm first (``G D``,
``D = diag(1/||G_i||)``), so the rank does not depend on the units of the
unknowns; singular values of ``G D`` below ``tol`` times the largest define
the null space; the resolution of parameter ``a`` is ``sum_{i <= rank}
V_ai^2`` (``V`` of ``G D``), in ``[0, 1]`` and scale invariant.

``feasibility_checks`` evaluates inequalities that every moment vector of a
nonnegative normalised population on the declared support satisfies:
normalisation, positive semidefinite moment matrices (real and Hermitian),
support bounds, Legendre ranges, ``|<f^2 e^{2i phi}>| <= <f^2>`` and
Cauchy-Schwarz ``|<f e^{2i phi}>|^2 <= <f^2>`` for the monomials ``f`` whose
squares expand onto retained rows. Checks that need ``M0`` rows with
``b > 0`` are computed only when ``m0_ext`` is present (``from_samples``) and
are otherwise listed as not computable from the fitted vector.

Shapes: the design is ``(n_kept, n_u)``; checks are ``(name, passed, margin)``.
Units: moments are dimensionless in the displacement coordinates ``z``;
data and design rows carry the data units. Both functions are eager: the
rank, the pass/fail flags and the margins are concrete Python values, so
they are called outside ``jit`` (the SVD itself is a ``jax.numpy`` call).
Not certified: passing every check does not establish feasibility (the
conditions are necessary only); a full-rank design does not make a fit
unbiased (discrepancy ``delta`` and the noise model are separate inputs).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import _feasibility as _fz
from ._feasibility import legendre_range
from ._feasibility_rules import (
    _Collector,
    _legendre_checks,
    _matrix_checks,
    _square_checks,
    _support_checks,
)
from ._identifiability import equilibrate, parameter_columns, whitened_design
from .._provenance import strict_json

LABEL_IDENTIFIABILITY = "extra eq: finite fit model"
LABEL_FEASIBILITY = "eq: joint moment feasible set"


# -- identifiability ---------------------------------------------------------


class IdentifiabilityReport(eqx.Module):
    """SVD summary of the column-equilibrated whitened design ``G D``.

    ``column_scales`` ``(n_u,)`` is ``diag(D)`` (``1/||G_i||``, 1 for a zero
    column); ``singular_values`` ``(min(n_data, n_u),)`` descending, of
    ``G D``; ``modes`` ``(n_u, n_u)`` right singular vectors of ``G D`` as
    columns (mode ``i`` belongs to ``singular_values[i]``); ``null_basis``
    ``(n_u, null_dim)`` orthonormal columns spanning the modes with
    ``sigma <= tol sigma_max``; ``resolution`` ``(n_u,)`` in ``[0, 1]``.
    Modes, null basis and resolution are in the equilibrated coordinates
    ``D^-1 u`` (scale invariant): ``D @ null_basis`` spans the null
    directions of the unknowns ``u`` (``G D w = 0``). ``labels`` names the unknowns
    (``"amplitude"`` first when the amplitude column is included, then
    ``ParameterMap.labels()``). ``note`` says whether the design is exact
    (affine map) or linearised at a supplied ``theta``. Singular values are
    dimensionless (whitened design). Assumes the noise model of the data;
    a null direction is a statement about the linear design at ``tol``,
    not certified against nonlinear or prior information.
    """

    singular_values: jax.Array
    modes: jax.Array
    null_dim: int = eqx.field(static=True)
    null_basis: jax.Array
    resolution: jax.Array
    labels: tuple[str, ...] = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    tol: float = eqx.field(static=True)
    note: str = eqx.field(static=True, default="")
    LABEL: str = eqx.field(static=True, default=LABEL_IDENTIFIABILITY)
    column_scales: jax.Array | None = None

    def weak(self, threshold) -> tuple[str, ...]:
        """Labels whose resolution is below ``threshold``."""
        res = np.asarray(self.resolution)
        return tuple(name for name, r in zip(self.labels, res) if r < float(threshold))

    def null_components(self, j, *, cutoff=1e-3) -> tuple[tuple[str, float], ...]:
        """``(label, coefficient)`` pairs of null vector ``j`` with ``|coef| > cutoff``."""
        if not 0 <= int(j) < self.null_dim:
            raise ValueError(f"null vector index must lie in [0, {self.null_dim})")
        column = np.asarray(self.null_basis)[:, int(j)]
        return tuple(
            (name, float(c)) for name, c in zip(self.labels, column) if abs(c) > cutoff
        )

    def to_dict(self) -> dict:
        return strict_json(
            {
                "label": self.LABEL,
                "note": self.note,
                "tol": self.tol,
                "rank": self.rank,
                "null_dim": self.null_dim,
                "labels": list(self.labels),
                "singular_values": np.asarray(self.singular_values).tolist(),
                "resolution": dict(
                    zip(self.labels, np.asarray(self.resolution, dtype=float).tolist())
                ),
                "null_basis": np.asarray(self.null_basis).tolist(),
                "column_scales": (
                    None
                    if self.column_scales is None
                    else np.asarray(self.column_scales).tolist()
                ),
                "null_components": [
                    dict(self.null_components(j)) for j in range(self.null_dim)
                ],
            }
        )


def identifiability(
    basis,
    data,
    parameter_map,
    *,
    tol=1e-8,
    theta=None,
    reference=None,
    amplitude=True,
) -> IdentifiabilityReport:
    """SVD of the whitened design ``L^-1 R C [c | P]`` (see the module docstring).

    ``basis`` supplies ``response_matrix()`` ``(4 n_ch, n_real)``, ``index``
    and ``reference``; ``data`` supplies ``whitened()``, ``response`` and
    ``mask``; ``parameter_map`` must use the basis index. ``tol`` is the
    relative singular-value cutoff on the column-equilibrated design
    (``sigma <= tol sigma_max`` is null; the same rule as ``fit_linear``).
    ``theta`` is required for a nonlinear map; ``reference`` defaults to the
    basis reference; ``amplitude=False`` drops the amplitude column (fixed
    amplitude). Eager: the rank is concrete. Raises ``ValueError`` for
    mismatched indices or shapes. Dimensionless output. Assumes the data's
    noise model; not certified: identifiability under priors or nonlinear
    constraints, or beyond the linearisation point ``theta``.
    """
    if parameter_map.index != basis.index:
        raise ValueError("parameter_map must use the basis MomentIndex")
    tol = float(tol)
    if not tol >= 0.0:
        raise ValueError("tol must be nonnegative")
    C = jnp.asarray(basis.response_matrix())
    if C.ndim != 2 or C.shape[1] != basis.index.n_real:
        raise ValueError("basis.response_matrix() must be (4 n_ch, n_real)")
    reference = basis.reference if reference is None else reference
    columns, labels, note = parameter_columns(
        parameter_map, reference, theta, amplitude
    )
    G, scales = equilibrate(whitened_design(data, C, columns))
    _, s, Vt = jnp.linalg.svd(G, full_matrices=True)
    s_np = np.asarray(s)
    s_max = float(s_np[0]) if s_np.size else 0.0
    rank = int(np.sum(s_np > tol * s_max)) if s_max > 0 else 0
    n_u = int(G.shape[1])
    return IdentifiabilityReport(
        singular_values=s,
        modes=Vt.T,
        null_dim=n_u - rank,
        null_basis=Vt[rank:].T,
        resolution=jnp.sum(Vt[:rank] ** 2, axis=0),
        labels=labels,
        rank=rank,
        tol=tol,
        note=note + "; column-equilibrated design G D, D = diag(1/||G_i||)",
        column_scales=scales,
    )


# -- feasibility -------------------------------------------------------------


class FeasibilityReport(eqx.Module):
    """Necessary conditions on a moment vector, each as ``(name, passed, margin)``.

    ``margin`` is the slack of the inequality written as ``g >= 0`` (the
    negative absolute deviation for the normalisation equality); a check
    passes when ``margin >= -tol * scale`` with ``scale`` the natural size of
    its bound. ``not_computable`` lists checks the retained rows cannot
    supply. ``statement`` says that the conditions are necessary only.
    Margins are dimensionless (moments in ``z`` units). Assumes the declared
    ``support``; passing every check is not certified as membership of the
    feasible set ``eq: joint moment feasible set``.
    """

    checks: tuple[tuple[str, bool, float], ...] = eqx.field(static=True)
    not_computable: tuple[str, ...] = eqx.field(static=True)
    statement: str = eqx.field(static=True)
    tol: float = eqx.field(static=True, default=1e-9)
    LABEL: str = eqx.field(static=True, default=LABEL_FEASIBILITY)

    def failed(self) -> tuple[str, ...]:
        return tuple(name for name, passed, _ in self.checks if not passed)

    def passed(self) -> tuple[str, ...]:
        return tuple(name for name, passed, _ in self.checks if passed)

    def margins(self) -> dict[str, float]:
        return {name: margin for name, _, margin in self.checks}

    def to_dict(self) -> dict:
        return strict_json(
            {
                "label": self.LABEL,
                "checks": [list(check) for check in self.checks],
                "not_computable": list(self.not_computable),
                "statement": self.statement,
                "tol": self.tol,
            }
        )


def feasibility_checks(moments, index, support, *, tol=1e-9) -> FeasibilityReport:
    """Necessary conditions for ``moments`` to lie in the feasible set.

    ``moments`` is a ``JointMoments`` on ``index`` (``ValueError`` otherwise);
    ``support`` is a ``Support`` in raw units (converted to ``z`` with the
    moments' reference) or ``None`` (support checks listed as not
    computable). ``tol`` is the absolute slack allowed, scaled by the size
    of each bound. Eager: needs concrete moment values. Dimensionless
    margins in ``z`` units. Assumes the declared support; a pass is
    necessary, not sufficient, for a nonnegative population.
    """
    if moments.index != index:
        raise ValueError("moments.index must equal the supplied index")
    tol = float(tol)
    if not tol >= 0.0:
        raise ValueError("tol must be nonnegative")
    lookup = _fz.MomentLookup(moments, index)
    col = _Collector(tol)
    col.add("normalisation", -abs(lookup.m0[0] - 1.0))
    _matrix_checks(col, lookup)
    _support_checks(col, lookup, support, moments.reference)
    _legendre_checks(col, lookup)
    _square_checks(col, lookup)
    checks = tuple(col.checks)
    failed = [name for name, passed, _ in checks if not passed]
    statement = (
        f"Necessary conditions only ({LABEL_FEASIBILITY}): "
        f"{len(checks) - len(failed)} passed, {len(failed)} failed"
        + (": " + ", ".join(failed) if failed else "")
        + f"; {len(col.missing)} entries not computable from the supplied moments. "
        "Passing does not certify membership of conv{psi(x): x in D}."
    )
    return FeasibilityReport(checks, tuple(col.missing), statement, tol)


__all__ = [
    "IdentifiabilityReport",
    "FeasibilityReport",
    "identifiability",
    "feasibility_checks",
    "legendre_range",
]
