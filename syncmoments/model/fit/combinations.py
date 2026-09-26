"""Identifiable-combination fits (``syncmoments.model.fit.combinations``).

LABEL: ``extra eq: finite fit model`` (``d = R C a + delta + n`` with the
full coordinates ``a`` of a ``ResponseReduction``), ``extra eq: data error
propagation`` (``|K| delta``) and ``[extension]``.

:func:`fit_combinations` fits the reduced model ``R H T a`` in the unknowns
``x`` (``a = a_off + J x``): the full coordinates themselves by default
(``J = I``, the source-column weighted coefficients ``a = (N_src/N_*) (m,
m0_ext)``, no slot fixed), ``a[1:]`` with a fixed amplitude, or the
``u = (A, A theta)`` / ``v = A theta`` of an affine ``ParameterMap``. The
coefficient metric ``||a||^2 = a^T M a`` (Euclidean by default; a positive
diagonal or an SPD matrix) is pulled back to ``M_x = J^T M J = R_x^T R_x``,
and ``T J R_x^-1 = L Q`` (``Q Q^T = I``) keeps it: the whitened design
``G_w = L_Sigma^-1 R H L`` is decomposed by SVD. An SVD of ``H`` alone
would change the metric and the meaning of the cutoff.

Three claims are kept apart: analytic redundancy (the reduction, before any
data), the numerical rank (singular values ``> rank_tol s_max``, strict)
and practical recoverability (``1/s <= max_sigma``, inclusive; ``max_sigma``
is required). Repeated singular values (``cluster_rtol``) are kept or dropped
as a whole. Returns a
:class:`~syncmoments.model.fit.combination_result.CombinationFit`.

Relation to ``fit_linear`` (whose defaults are unchanged): with a full-rank
affine map, ``max_sigma = inf`` and the same ``rank_tol``, ``x_hat`` equals
its ``u``; when rank-deficient the predictions agree but ``fit_linear``
returns the minimum equilibrated-norm solution and this path the minimum
metric-norm one. Unlike ``fit_linear``, a reduced rank does not make
``bias`` unbounded: it bounds the error of the identified projection
``Pi x``; the unidentified part is ``unresolved``.

Shapes: ``J`` ``(n_full, n_x)``, estimator ``(n_x, n_kept)``. Units: data in
the units of the basis response, ``x`` dimensionless. Eager NumPy on
concrete values; not ``jax.jit`` safe. Not certified: that ``data.noise``
is the noise, the declared discrepancy and coefficient bound, and anything
about the unresolved directions beyond a declared bound.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ..assumptions import ParameterMap
from ..errors import ErrorTerm
from . import _combination_core as core_svd
from . import _combination_errors as err
from . import _result_core as core
from ._linear_core import inflated
from ._result_core import MAX_FISHER_PARAMS, TOL
from .combination_result import CombinationFit, Subspaces
from .linear import _check_amplitude, _declared_chi2_note
from .reduction import ResponseReduction, reduce_response

LABEL = ("extra eq: finite fit model", "extra eq: data error propagation")
POLICIES = ("bias_bound", "inflate")
CLUSTER_RTOL = 1e-10
SYMMETRY_RTOL = 1e-12


def _positive(value, name, *, allow_inf):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive float") from None
    if not value > 0.0 or math.isnan(value) or (math.isinf(value) and not allow_inf):
        raise ValueError(
            f"{name} must be positive" + (" (or math.inf)" if allow_inf else "")
        )
    return value


def _check_reduction(basis, reduction):
    if reduction is None:
        return reduce_response(basis)
    if not isinstance(reduction, ResponseReduction):
        raise ValueError("reduction must be a ResponseReduction")
    if reduction.layout.index != basis.index:
        raise ValueError("reduction.layout.index must equal basis.index")
    C = reduction.layout.pad(np.asarray(basis.response_matrix(), dtype=float))
    ref = np.asarray(reduction.C)
    scale = float(np.max(np.abs(C), initial=0.0))
    if C.shape != ref.shape or np.max(np.abs(C - ref), initial=0.0) > 1e-12 * scale:
        raise ValueError(
            "reduction.C does not match the basis response (foreign reduction)"
        )
    return reduction


def _coordinates(basis, layout, parameter_map, fitted, fixed):
    """``(J, a_off, labels, description)`` of design Section 5.2."""
    n_full, n_real = layout.n_full, layout.n_active
    if parameter_map is None:
        J = np.eye(n_full)
        labels = layout.labels()
        if fitted:
            return J, np.zeros(n_full), labels, "a (full coordinates, amplitude as a_0)"
        a_off = np.zeros(n_full)
        a_off[0] = fixed
        return (
            J[:, 1:],
            a_off,
            labels[1:],
            f"a[1:] with a_0 = A = {fixed:.6g} fixed (declared, not inferred)",
        )
    if not isinstance(parameter_map, ParameterMap):
        raise ValueError("parameter_map must be a ParameterMap or None")
    if parameter_map.index != basis.index:
        raise ValueError("parameter_map.index must equal basis.index")
    if not parameter_map.is_affine():
        raise ValueError(
            f"parameter map '{parameter_map.name}' is not affine; use fit_bfgs"
        )
    P, c = (
        np.asarray(v, dtype=float)
        for v in parameter_map.affine_pieces(reference=basis.reference)
    )
    pad = n_full - n_real
    if fitted:
        J = np.vstack([np.column_stack([c, P]), np.zeros((pad, P.shape[1] + 1))])
        labels = ("amplitude",) + tuple(parameter_map.labels())
        return J, np.zeros(n_full), labels, "u = (A, A theta) of the affine map"
    J = np.vstack([P, np.zeros((pad, P.shape[1]))])
    a_off = fixed * np.concatenate([c, np.zeros(pad)])
    return (
        J,
        a_off,
        tuple(parameter_map.labels()),
        f"v = A theta, A = {fixed:.6g} fixed",
    )


def _metric(metric, n_full):
    if metric is None:
        return np.eye(n_full), "Euclidean in the full coordinates a (declared default)"
    M = np.asarray(metric, dtype=float)
    if not np.all(np.isfinite(M)):
        raise ValueError("metric must be finite")
    if M.shape == (n_full,):
        if not np.all(M > 0.0):
            raise ValueError("diagonal metric weights must be positive")
        return np.diag(M), "diagonal metric supplied by the caller"
    if M.shape != (n_full, n_full):
        raise ValueError(f"metric must be ({n_full},) or ({n_full}, {n_full})")
    scale = float(np.max(np.abs(M)))
    if np.max(np.abs(M - M.T)) > SYMMETRY_RTOL * scale:
        raise ValueError("metric must be symmetric")
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError("metric must be positive definite") from None
    return 0.5 * (M + M.T), "dense SPD metric supplied by the caller"


def _check_bound(bound, n_full):
    if bound is None:
        return None
    if not isinstance(bound, ErrorTerm):
        raise ValueError("coefficient_bound must be an ErrorTerm or None")
    if bound.kind not in ("bound", "estimate", "measured"):
        raise ValueError(
            "coefficient_bound must be of kind bound, estimate or measured"
        )
    if tuple(bound.value.shape) != (n_full,):
        raise ValueError(f"coefficient_bound must have shape ({n_full},)")
    return bound


def _check_data(data, reduction):
    n_rows = reduction.C.shape[0]
    if not all(hasattr(data, n) for n in ("n_ch", "whitened", "design", "kept_rows")):
        raise ValueError("data must be a StokesData")
    if 4 * int(data.n_ch) != int(n_rows):
        raise ValueError(
            f"data has {data.n_ch} channels but the basis has {n_rows // 4}"
        )


def _policy(data, policy):
    if policy not in POLICIES:
        raise ValueError(f"discrepancy_policy must be one of {POLICIES}")
    if policy == "bias_bound":
        return data, ["discrepancy_policy=bias_bound: bias = |K| delta over x"]
    delta = data.discrepancy_vector()
    if delta is None:
        raise ValueError("discrepancy_policy='inflate' needs a valued data.discrepancy")
    return inflated(data, delta), [
        "discrepancy_policy=inflate (heuristic): diag(delta^2) added to the noise; the "
        "singular values, the retained set and the covariance describe the inflated "
        "weights, not the noise"
    ]


def fit_combinations(
    basis,
    data,
    *,
    max_sigma,
    reduction=None,
    parameter_map=None,
    amplitude="fit",
    metric=None,
    rank_tol=TOL,
    cluster_rtol=CLUSTER_RTOL,
    discrepancy_policy="bias_bound",
    coefficient_bound=None,
) -> CombinationFit:
    """Fit the identifiable combinations of the reduced model (module docstring).

    ``max_sigma`` (required, positive, finite or ``math.inf``): keep a mode
    when ``1/s <= max_sigma``. ``reduction``: ``None`` (``reduce_response(basis)``)
    or a ``ResponseReduction`` of this basis. ``parameter_map``: ``None`` or
    an affine ``ParameterMap``. ``amplitude``: ``"fit"`` or a positive float.
    ``metric``: ``None`` (Euclidean in ``a``), ``(n_full,)`` positive weights
    or an ``(n_full, n_full)`` SPD matrix. ``rank_tol``: numerical-rank
    cutoff relative to ``s_max``; ``cluster_rtol``: repeated-value gap
    relative to ``s_max`` (an absolute gap, since a singular vector is
    determined only to about ``u s_max / gap``; the ``beta`` of a cluster may
    be correlated, see the notes); ``discrepancy_policy`` in ``POLICIES``;
    ``coefficient_bound``: an ``ErrorTerm`` ``(n_full,)`` bounding ``|a|``.
    Raises ``ValueError`` for invalid inputs, a non-affine map, a redundant
    map under the metric, more than ``MAX_FISHER_PARAMS`` unknowns or a zero
    design. Not certified: see the module docstring.
    """
    max_sigma = _positive(max_sigma, "max_sigma", allow_inf=True)
    rank_tol = core.check_tol(rank_tol)
    try:
        cluster_rtol = float(cluster_rtol)
    except (TypeError, ValueError):
        raise ValueError("cluster_rtol must be a float") from None
    if not (math.isfinite(cluster_rtol) and cluster_rtol >= 0.0):
        raise ValueError("cluster_rtol must be finite and >= 0")
    fitted, fixed = _check_amplitude(amplitude)
    reduction = _check_reduction(basis, reduction)
    layout = reduction.layout
    _check_data(data, reduction)
    J, a_off, labels, description = _coordinates(
        basis, layout, parameter_map, fitted, fixed
    )
    if J.shape[1] > MAX_FISHER_PARAMS:
        raise ValueError(
            f"{J.shape[1]} unknowns exceed MAX_FISHER_PARAMS = {MAX_FISHER_PARAMS}"
        )
    M, metric_note = _metric(metric, layout.n_full)
    bound = _check_bound(coefficient_bound, layout.n_full)
    fit_data, notes = _policy(data, discrepancy_policy)
    sol = core_svd.solve(
        reduction,
        fit_data,
        J,
        a_off,
        M,
        max_sigma=max_sigma,
        rank_tol=rank_tol,
        cluster_rtol=cluster_rtol,
    )
    notes += list(sol.notes)
    if discrepancy_policy == "inflate":
        notes.append(_declared_chi2_note(data, jnp.asarray(sol.stokes_hat.reshape(-1))))
    delta = err.discrepancy_term(data, jnp.asarray(sol.stokes_hat.reshape(-1)))
    envelope = err.reduction_envelope(reduction, fit_data, bound)
    n_x = J.shape[1]
    comp = np.eye(n_x) - sol.Pi
    D = sol.s[[i for i, c in enumerate(sol.classes) if c == core_svd.RETAINED]]
    notes.append(
        f"fit_combinations: {n_x} unknowns ({description}), rank(T J) = {sol.Q.shape[0]}, "
        f"rank_tol {rank_tol:g}, max_sigma {max_sigma:g}, cluster_rtol {cluster_rtol:g}"
    )
    return CombinationFit(
        labels=tuple(labels),
        coordinates=description,
        reduction=reduction,
        jacobian=jnp.asarray(J),
        offset=jnp.asarray(a_off),
        metric=jnp.asarray(sol.M_x),
        metric_factor=jnp.asarray(sol.R_x),
        metric_note=metric_note,
        factor_L=jnp.asarray(sol.L),
        factor_Q=jnp.asarray(sol.Q),
        lq_method=sol.lq_method,
        singular_values=jnp.asarray(sol.s),
        classes=sol.classes,
        numerical_rank=sum(c != core_svd.NULL for c in sol.classes),
        n_retained=int(D.size),
        rank_tol=rank_tol,
        max_sigma=max_sigma,
        cluster_rtol=cluster_rtol,
        clusters=sol.clusters,
        combination_rows=jnp.asarray(sol.B),
        beta_hat=jnp.asarray(sol.beta_hat),
        beta_sigma=jnp.asarray(sol.beta_sigma),
        directions=Subspaces(*(jnp.asarray(d) for d in sol.directions)),
        estimator=jnp.asarray(sol.K),
        beta_estimator=jnp.asarray(sol.K_beta),
        kept_rows=sol.kept_rows,
        x_hat=jnp.asarray(sol.x_hat),
        representative=jnp.asarray(sol.a_hat),
        covariance=jnp.asarray(sol.Cov),
        projector=jnp.asarray(sol.Pi),
        chi2=jnp.asarray(sol.chi2),
        dof=sol.dof,
        stokes_hat=jnp.asarray(sol.stokes_hat),
        stokes_sigma=jnp.asarray(sol.stokes_sigma),
        bias=err.propagated(sol.K, delta, "bias of x_hat relative to Pi x", "delta"),
        beta_bias=err.propagated(sol.K_beta, delta, "bias of beta_hat", "delta"),
        reduction_bias=err.propagated(
            sol.K, envelope, "approximate-reduction bias of x_hat", "R delta_C a"
        ),
        unresolved=err.unresolved_term(
            np.eye(n_x), comp, J, bound, "unresolved part (I - Pi) x", check_scale=1.0
        ),
        data_discrepancy=delta,
        reduction_envelope=envelope,
        coefficient_bound=bound,
        response=None if fit_data.response is None else jnp.asarray(fit_data.response),
        noise_model="variances" if np.ndim(fit_data.noise) == 1 else "covariance",
        provenance=core.provenance_of(basis, (), notes),
    )


__all__ = ["fit_combinations", "POLICIES", "CLUSTER_RTOL", "LABEL", "TOL"]
