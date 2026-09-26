"""Pre-fit response reduction ``C = H T`` (``syncmoments.model.fit.reduction``).

LABEL: ``extra eq: finite fit model`` (the padded response ``C`` of the full
coordinates ``a``), ``eq: channel derivative coefficients`` (the columns the
continuum relations act on) and ``[extension]`` (the reduction itself).

:func:`reduce_response` groups the columns of the padded response ``C``
``(4 n_ch, n_full)`` (:class:`~syncmoments.model.fit.layout.CoefficientLayout`)
into ``H = C[:, representatives]`` ``(4 n_ch, n_q)`` and ``T`` ``(n_q, n_full)``
with ``q = T a``, before any data is seen: it takes the basis only (no data,
noise or moments). Relations ``sum_j w_j C_j = 0`` come from four sources,
kept apart in :class:`LinearRelation`:

* ``"structural"``: identically zero columns (``CoefficientLayout.structural_zero``),
  checked to be exactly zero.
* ``"analytic"``: the continuum scaling identity (``_relations`` docstring),
  generated only when ``basis.provenance.kernel`` names ``continuum``, for any
  ``Truncation`` accepted by ``build_basis``, any channels, any
  ``PhaseWeights`` and any positive reference scales; checked on the grid to
  ``check_rtol``. ``HarmonicKernel`` (the Doppler factor and harmonic lines
  depend on ``gamma`` separately from ``B gamma^2``), ``PolynomialTestKernel``
  and unnamed kernels are outside this scope: ``relations="auto"`` falls back
  to structural zeros with a note, ``relations="continuum"`` raises.
* ``"declared"``: a caller statement with a non-empty scope, checked to
  ``check_rtol`` and then treated as exact ("declared by caller", not derived).
* ``"approximate"``: kept with ``delta_C = C - H T`` and propagated by the
  fit; :func:`find_column_relations` proposes such candidates from a sampled
  grid and never promotes them to analytic or declared.

The Euclidean factorisation ``T = L Q`` (``Q Q^T = I``, ``L L^T = T T^T``)
keeps the coefficient metric: ``C = (H L) Q``. Units: ``C`` and ``H`` in the
units of the basis response (per unit ``z``-moment), ``T``, ``L``, ``Q``
dimensionless. Eager NumPy on concrete values (not ``jax.jit`` safe); array
fields are ``jnp`` float64 leaves. Not certified: the relation residual
checks are finite float64 checks on the supplied grid (``check_rtol``), a
declared relation is the caller's claim, and a numerical near-dependence is
not a physical identity. Completeness of the relation set is proven for
uncapped continuum blocks only.
"""

from __future__ import annotations

import math
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .._provenance import strict_json
from . import _reduction_checks as chk
from . import _relations as rel
from .layout import CoefficientLayout

LABEL = ("extra eq: finite fit model", "eq: channel derivative coefficients")
RELATIONS = chk.RELATIONS
CHECK_RTOL = 1e-10


class LinearRelation(NamedTuple):
    """A claim ``sum_j w_j C[:, j] = 0`` on full slots (``[extension]``).

    ``weights``: ``((slot, w), ...)``; ``source``: ``"structural"``,
    ``"analytic"``, ``"declared"`` or ``"approximate"``; ``scope``: the
    derivation scope (generated) or the caller statement. Dimensionless.
    Validated by :func:`reduce_response`; a declared relation is not
    certified by the package beyond its residual check (module docstring).
    """

    weights: tuple[tuple[int, float], ...]
    source: str
    scope: str

    @classmethod
    def proportional(cls, i, j, ratio, *, scope, approximate=False) -> "LinearRelation":
        """``C_i = ratio C_j`` (``"declared"``, or ``"approximate"``)."""
        source = "approximate" if approximate else "declared"
        return cls(((int(i), 1.0), (int(j), -float(ratio))), source, scope)

    @classmethod
    def identical(cls, i, j, *, scope, approximate=False) -> "LinearRelation":
        """``C_i = C_j``."""
        return cls.proportional(i, j, 1.0, scope=scope, approximate=approximate)


class ResponseReduction(eqx.Module):
    """``C = H T`` before any data (module docstring for sources and scope).

    Arrays (``jnp`` float64): ``C`` ``(4 n_ch, n_full)``, ``T`` ``(n_q,
    n_full)``, ``H`` ``(4 n_ch, n_q)``, ``L`` ``(n_q, r)`` and ``Q`` ``(r,
    n_full)`` with ``T = L Q``, ``delta_C = C - H T``, ``null_basis``
    ``(n_full, n_full - rank T)``. Static: ``layout``, ``lq_method``,
    ``representatives`` (full slot of each ``q``), ``group_labels``,
    ``group_formulas``, ``relations`` (the independent relations applied),
    ``relation_residuals``, ``exact`` (no approximate relation),
    ``max_delta_C`` (``max|delta_C| / max|C|``), ``kernel``, ``scope`` and
    ``notes``. Units and what is not certified: see the module docstring.
    """

    LABEL: ClassVar[tuple[str, ...]] = LABEL
    layout: CoefficientLayout = eqx.field(static=True)
    C: jax.Array
    T: jax.Array
    H: jax.Array
    L: jax.Array
    Q: jax.Array
    delta_C: jax.Array
    null_basis: jax.Array
    lq_method: str = eqx.field(static=True)
    representatives: tuple[int, ...] = eqx.field(static=True)
    group_labels: tuple[str, ...] = eqx.field(static=True)
    group_formulas: tuple[str, ...] = eqx.field(static=True)
    relations: tuple[LinearRelation, ...] = eqx.field(static=True)
    relation_residuals: tuple[float, ...] = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    max_delta_C: float = eqx.field(static=True)
    kernel: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    notes: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        n_rows, n_full = self.C.shape
        n_q = len(self.representatives)
        if n_full != self.layout.n_full or n_q > n_full:
            raise ValueError("C must be (4 n_ch, n_full) with n_q <= n_full")
        for name, shape in (
            ("T", (n_q, n_full)),
            ("H", (n_rows, n_q)),
            ("delta_C", (n_rows, n_full)),
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.L.shape[0] != n_q or self.Q.shape != (self.L.shape[1], n_full):
            raise ValueError("L must be (n_q, r) and Q (r, n_full)")

    @property
    def n_q(self) -> int:
        """Number of grouped coefficients."""
        return len(self.representatives)

    @property
    def n_full(self) -> int:
        """Number of full coordinates."""
        return self.layout.n_full

    def grouped(self, a) -> np.ndarray:
        """``q = T a`` for ``a`` ``(n_full,)`` or ``(n_full, k)``."""
        return np.asarray(self.T) @ np.asarray(a, dtype=float)

    def predict_full(self, a) -> np.ndarray:
        """``(H T a).reshape(n_ch, 4)``: the reduced-model channel Stokes."""
        stokes = np.asarray(self.H) @ self.grouped(a)
        return stokes.reshape(-1, 4)

    def to_dict(self, arrays=False) -> dict:
        """Strict JSON; the arrays ``C``, ``T``, ``H``, ``L``, ``Q``, ``delta_C`` only with ``arrays=True``."""
        out = {
            "layout": self.layout.to_dict(),
            "n_full": self.n_full,
            "n_active": self.layout.n_active,
            "n_q": self.n_q,
            "representatives": list(self.representatives),
            "group_labels": list(self.group_labels),
            "group_formulas": list(self.group_formulas),
            "relations": [
                {
                    "weights": [list(p) for p in r.weights],
                    "source": r.source,
                    "scope": r.scope,
                    "record": chk.record(r.source),
                    "residual": rho,
                }
                for r, rho in zip(self.relations, self.relation_residuals)
            ],
            "exact": self.exact,
            "max_delta_C": self.max_delta_C,
            "lq_method": self.lq_method,
            "kernel": self.kernel,
            "scope": self.scope,
            "notes": list(self.notes),
        }
        if arrays:
            for name in ("C", "T", "H", "L", "Q", "delta_C", "null_basis"):
                out[name] = np.asarray(getattr(self, name))
        return strict_json(out)


def reduce_response(
    basis,
    *,
    relations="auto",
    declared=(),
    representatives="gamma",
    include_ext=True,
    check_rtol=CHECK_RTOL,
) -> ResponseReduction:
    """Group the response columns before any data (module docstring).

    ``basis``: a ``SpectralBasis`` or a duck type with ``response_matrix()``,
    ``index``, ``reference`` and ``provenance``. ``relations`` in
    ``RELATIONS``; ``declared``: ``LinearRelation`` s with source
    ``"declared"`` or ``"approximate"``; ``representatives``: ``"gamma"``
    (keep the ``s = 0`` columns) or a tuple of distinct ``(r, s, b)`` most
    preferred first (changes only ``H``, ``T`` and labels);
    ``include_ext`` adds the ``h0_ext`` slots; ``check_rtol`` the relative
    residual accepted for analytic and declared relations. Raises
    ``ValueError`` for invalid options or relations, a non-finite ``C``, a
    failed residual check or an inconsistent exact reduction. Not certified:
    see the module docstring.
    """
    check_rtol = chk.check_options(relations, representatives, include_ext, check_rtol)
    declared = tuple(declared)
    layout = CoefficientLayout.build(
        basis.index, basis.reference, include_ext=include_ext
    )
    C = layout.pad(np.asarray(basis.response_matrix(), dtype=float))
    if not np.all(np.isfinite(C)):
        raise ValueError("the response matrix must be finite")
    kernel = rel.kernel_name(basis)
    notes = []
    stack = chk.generated(layout, relations, kernel, notes)
    for relation in declared:
        weights, source = chk.validate_relation(
            relation, layout.n_full, caller=True, relation_type=LinearRelation
        )
        stack.append((weights, source, relation.scope))
    residuals = rel.check_relations(
        C, stack, check_rtol, f"{kernel!r} at reference {dict(layout.reference)}"
    )
    R = rel.relation_matrix(stack, layout.n_full)
    kept, dropped = rel.prune(R)
    for i in dropped:
        notes.append(
            f"dropped redundant {stack[i][1]} relation on slots "
            f"{[j for j, _ in stack[i][0]]} ({stack[i][2]})"
        )
    R = R[kept]
    rep, eliminated = rel.select(R, layout, representatives)
    T, H, amplification, cleaned = rel.assemble(C, R, rep, eliminated)
    if cleaned:
        notes.append(f"{cleaned} entries of T below 1e-15 max|T| set to 0")
    applied = tuple(LinearRelation(*stack[i]) for i in kept)
    exact = all(r.source != "approximate" for r in applied)
    delta_C = C - H @ T
    scale = float(np.max(np.abs(C), initial=0.0))
    max_delta = float(np.max(np.abs(delta_C), initial=0.0)) / scale if scale else 0.0
    limit = (check_rtol + 16 * rel.EPS) * amplification
    if exact and max_delta > limit:
        raise ValueError(
            f"exact reduction inconsistent: max|C - H T| / max|C| = {max_delta:.3g} "
            f"exceeds the propagated residual limit {limit:.3g}"
        )
    L, Q, method = rel.lq_factor(T)
    labels = layout.labels()
    scope = chk.scope_text(layout, kernel, applied)
    return ResponseReduction(
        layout=layout,
        C=jnp.asarray(C),
        T=jnp.asarray(T),
        H=jnp.asarray(H),
        L=jnp.asarray(L),
        Q=jnp.asarray(Q),
        delta_C=jnp.asarray(delta_C),
        null_basis=jnp.asarray(rel.null_basis(T)),
        lq_method=method,
        representatives=rep,
        group_labels=tuple(f"q[{labels[j]}]" for j in rep),
        group_formulas=tuple(chk.formula(row, labels) for row in T),
        relations=applied,
        relation_residuals=tuple(residuals[i] for i in kept),
        exact=exact,
        max_delta_C=max_delta,
        kernel=str(kernel),
        scope=scope,
        notes=tuple(notes),
    )


def find_column_relations(
    reduction_or_basis, *, rtol=1e-10
) -> tuple[LinearRelation, ...]:
    """Candidate zero, identical and proportional non-structural columns (advisory).

    ``||C_i - ratio C_j||_inf <= rtol ||C_i||_inf`` with the least-squares
    ratio (a zero column: ``||C_i||_inf <= rtol max|C|``). Every candidate
    has source ``"approximate"`` and a scope naming the grid; it enters a
    reduction only when passed in ``declared``. Not certified: a sampled
    near-dependence is not a physical identity.
    """
    try:
        rtol = float(rtol)
    except (TypeError, ValueError):
        raise ValueError("rtol must be a float") from None
    if not (math.isfinite(rtol) and rtol >= 0.0):
        raise ValueError("rtol must be finite and nonnegative")
    if isinstance(reduction_or_basis, ResponseReduction):
        C, layout = np.asarray(reduction_or_basis.C), reduction_or_basis.layout
        channels = None
    else:
        basis = reduction_or_basis
        layout = CoefficientLayout.build(basis.index, basis.reference)
        C = layout.pad(np.asarray(basis.response_matrix(), dtype=float))
        channels = getattr(basis, "channels", None)
    grid = f"{C.shape[0] // 4} channels"
    centres = getattr(channels, "centres_hz", None)
    if centres is not None:
        c = np.asarray(centres)
        grid += f", {c.min():.4g}-{c.max():.4g} Hz"
    scope = f"numerical candidate on {grid}, rtol={rtol:g}; not a physical identity"
    out = []
    for kind, i, j, ratio in rel.candidates(C, set(layout.structural_zero), rtol):
        if kind == "zero":
            out.append(
                LinearRelation(((i, 1.0),), "approximate", f"zero column; {scope}")
            )
        else:
            out.append(
                LinearRelation.proportional(
                    i, j, ratio, scope=f"{kind} columns; {scope}", approximate=True
                )
            )
    return tuple(out)


__all__ = [
    "LinearRelation",
    "ResponseReduction",
    "reduce_response",
    "find_column_relations",
    "RELATIONS",
    "CHECK_RTOL",
    "LABEL",
]
