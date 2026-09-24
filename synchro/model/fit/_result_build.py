"""Builders of :class:`~synchro.model.fit.result.FitResult` (private helper).

``from_linear`` wraps the closed-form solution of ``fit_linear``;
``from_bfgs`` and ``from_nodal`` wrap the private outcomes of ``fit_bfgs``
and ``fit_nodal`` with a Gauss-Newton Fisher matrix in the fit coordinates
(``J^T J`` of the whitened residual: ``jax.jacfwd`` in ``z`` for BFGS, the
closed-form softmax Jacobian for the nodal fit), the identifiability and
feasibility reports and the prediction. ``from_bfgs`` also forms the
linearised bias ``|K| |delta|`` over ``z`` (``_linear_core.linearised_bias``). See ``result.py`` for the meaning of
every field and for what is not certified.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import _project
from ..assumptions import Parameters, nodal
from . import _result_core as core
from ._linear_core import linearised_bias
from ._result_core import MAX_FISHER_PARAMS, TOL
from .result import FitResult


def _assemble(
    *,
    basis,
    data,
    parameter_map,
    theta,
    moments,
    amplitude,
    summary,
    chi2,
    converged,
    n_iter,
    labels,
    coordinates,
    method,
    notes,
    bias_bound=None,
    bias_note=None,
    m_of=None,
    x=None,
    diagnostics=True,
    amplitude_fitted=True,
    z=None,
    fun=None,
    weights=None,
    discretisation=None,
    tol=TOL,
) -> FitResult:
    notes = list(notes)
    hyper_note = _project.fitted_hyper_note(parameter_map, theta)
    if hyper_note is not None:
        notes.append(hyper_note)
    n_params = len(labels)
    if summary.fisher is None:
        dof = data.dof(n_params)
    else:
        dof = data.dof(summary.rank)
        if summary.covariance is None:
            notes.append(
                f"covariance None: Fisher rank {summary.rank} < {n_params} at tol {tol:g}"
            )
    report = feasibility = prediction = None
    if diagnostics and core.finite_moments(moments, amplitude):
        report, feasibility = core.diagnostics_of(
            basis,
            data,
            parameter_map,
            theta,
            amplitude=amplitude_fitted,
            moments=moments,
            notes=notes,
            tol=tol,
        )
        statistical = None
        if summary.covariance is not None and m_of is not None:
            statistical = core.moment_uncertainty(
                m_of, x, summary.covariance, bias_bound=bias_bound, bias_note=bias_note
            )
        else:
            notes.append(
                "statistical_input of the prediction unbounded: no covariance "
                "available for the moment uncertainty"
            )
        prediction = core.prediction_of(
            basis,
            moments,
            amplitude,
            statistical=statistical,
            discretisation=discretisation,
            notes=notes,
            hyper_note=hyper_note,
        )
    elif diagnostics:
        notes.append(
            "diagnostics and prediction not computed: fitted moments or amplitude "
            "are not finite"
        )
    else:
        notes.append("diagnostics=False: no identifiability, feasibility or prediction")
    return FitResult(
        theta=theta,
        moments=moments,
        amplitude=jnp.asarray(amplitude, dtype=float),
        fisher=summary.fisher,
        covariance=summary.covariance,
        rank=summary.rank,
        chi2=jnp.asarray(chi2, dtype=float),
        dof=int(dof),
        converged=bool(converged),
        n_iter=int(n_iter),
        bias_bound=bias_bound,
        identifiability=report,
        feasibility=feasibility,
        prediction=prediction,
        provenance=core.provenance_of(basis, moments.assumptions, notes),
        method=method,
        labels=tuple(labels),
        coordinates=coordinates,
        parameter_map=parameter_map,
        z=z,
        fun=None if fun is None else jnp.asarray(fun, dtype=float),
        weights=weights,
        discretisation=discretisation,
    )


def from_linear(
    *,
    basis,
    data,
    parameter_map,
    theta,
    amplitude,
    u,
    summary,
    chi2,
    bias_bound,
    labels,
    amplitude_fitted,
    notes,
    diagnostics,
    tol=TOL,
) -> FitResult:
    """Wrap the closed-form solution of ``fit_linear`` (see that function).

    ``data`` is the data the fit was weighted with (the inflated noise under
    ``discrepancy_policy="inflate"``), so the identifiability report and the
    Fisher summary share weights; ``tol`` is the fit's singular-value cutoff.
    """
    moments = parameter_map(theta, basis.reference)
    reference = basis.reference

    if amplitude_fitted:

        def m_of(v):
            return parameter_map(
                parameter_map.unflatten(v[1:] / v[0]), reference
            ).to_vector()

        coordinates = "u = (A, A theta): amplitude and amplitude-scaled parameters"
    else:
        A = jnp.asarray(amplitude, dtype=float)

        def m_of(v):
            return parameter_map(parameter_map.unflatten(v), reference).to_vector()

        coordinates = f"theta with the amplitude fixed at {float(A):.6g}"
    return _assemble(
        basis=basis,
        data=data,
        parameter_map=parameter_map,
        theta=theta,
        moments=moments,
        amplitude=amplitude,
        summary=summary,
        chi2=chi2,
        converged=True,
        n_iter=0,
        labels=labels,
        coordinates=coordinates,
        method="linear",
        notes=notes,
        bias_bound=bias_bound,
        m_of=m_of,
        x=u,
        diagnostics=diagnostics,
        amplitude_fitted=amplitude_fitted,
        tol=tol,
    )


def from_bfgs(outcome, logdensity, *, diagnostics=True) -> FitResult:
    """Wrap a ``fit_bfgs`` outcome: Gauss-Newton Fisher ``J^T J`` in ``z``."""
    pm, basis, data = logdensity.parameter_map, logdensity.basis, logdensity.data
    transform = logdensity.transform
    theta, z = outcome.theta, jnp.asarray(outcome.z)
    if logdensity.amplitude is None:
        amplitude = jnp.exp(theta.log_amplitude)
    else:
        amplitude = logdensity.amplitude
        theta = Parameters(
            log_amplitude=jnp.log(amplitude),
            tables=theta.tables,
            hyper=theta.hyper,
            logits=theta.logits,
        )
    moments = pm(outcome.theta, basis.reference)
    J = jax.jacfwd(logdensity.whitened_residual)(z)
    summary = core.fisher_from_jacobian(J)
    stokes_hat = amplitude * (
        jnp.asarray(basis.response_matrix()) @ moments.to_vector()
    )
    bias_bound = linearised_bias(J, summary, data, stokes_hat)
    notes = [
        f"fit: bfgs in the transform coordinates z ({transform.n_params} parameters), "
        f"converged={outcome.converged}, n_iter={outcome.n_iter}",
        "fisher: Gauss-Newton J^T J of the whitened residual in z at the fitted point "
        "(prior and log-Jacobian terms excluded)",
    ]

    def m_of(v):
        return pm(transform.inverse(v), basis.reference).to_vector()

    return _assemble(
        basis=basis,
        data=data,
        parameter_map=pm,
        theta=theta,
        moments=moments,
        amplitude=amplitude,
        summary=summary,
        chi2=logdensity.chi2(z),
        converged=outcome.converged,
        n_iter=outcome.n_iter,
        labels=transform.labels,
        coordinates="z: unconstrained Transform coordinates (log A first when fitted)",
        method="bfgs",
        notes=notes,
        bias_bound=bias_bound,
        bias_note="|dm/dz| bias_bound (linearised at the optimum)",
        m_of=m_of,
        x=z,
        diagnostics=diagnostics,
        amplitude_fitted=logdensity.amplitude is None,
        z=z,
        fun=outcome.fun,
    )


def from_nodal(outcome, basis, data, nodes, *, diagnostics=True) -> FitResult:
    """Wrap a ``fit_nodal`` outcome: closed-form Fisher in ``(log A, logits)``."""
    pm = nodal(basis.index, nodes)
    theta, w = outcome.theta, jnp.asarray(outcome.weights)
    amplitude = jnp.exp(theta.log_amplitude)
    moments = pm(theta, basis.reference)
    labels = ("log_amplitude",) + pm.labels()
    notes = [
        f"fit: nodal exponentiated-gradient on {nodes.size} nodes, "
        f"converged={outcome.converged}, n_iter={outcome.n_iter}",
    ]
    if nodes.size + 1 <= MAX_FISHER_PARAMS:
        target, Linv = data.whitened()
        design = Linv @ data.design(jnp.asarray(basis.response_matrix()))
        J = core.nodal_jacobian(basis, data, nodes, w, amplitude, design)
        summary = core.fisher_from_jacobian(J)
        notes.append(
            "fisher: Gauss-Newton J^T J in (log A, logits); the softmax gauge "
            "direction is a null vector"
        )
    else:
        summary = core.FisherSummary(None, None, 0, None)
        notes.append(
            f"fisher not formed: {nodes.size + 1} parameters exceed "
            f"MAX_FISHER_PARAMS={MAX_FISHER_PARAMS}"
        )
    return _assemble(
        basis=basis,
        data=data,
        parameter_map=pm,
        theta=theta,
        moments=moments,
        amplitude=amplitude,
        summary=summary,
        chi2=2.0 * jnp.asarray(outcome.fun),
        converged=outcome.converged,
        n_iter=outcome.n_iter,
        labels=labels,
        coordinates="(log A, logits): amplitude and softmax logits of the node weights",
        method="nodal",
        notes=notes,
        diagnostics=diagnostics,
        z=jnp.asarray(outcome.z),
        fun=outcome.fun,
        weights=w,
        discretisation=outcome.discretisation,
    )


__all__ = ["from_linear", "from_bfgs", "from_nodal"]
