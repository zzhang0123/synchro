"""Solver and bias bound of :func:`~synchro.model.fit.linear.fit_linear` (private).

``solve`` is the rank-revealing least-squares solve on the column-equilibrated
whitened design; ``bias_bound`` is ``|K| |delta|`` with the two terms of
``extra eq: data error propagation``; ``linearised_bias`` is the same
quantity for ``fit_bfgs`` with ``K = J^+ L^-1`` from the whitened Jacobian at
the optimum (an ``estimate``: the model is linearised there);
``inflated`` builds the noise model of ``discrepancy_policy="inflate"``.
Nothing here is public API; the contracts are stated in the ``fit_linear``
and ``fit/result.py`` module docstrings.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..errors import ErrorTerm
from . import _result_core as core
from ._identifiability import equilibrate
from .observation import N_STOKES, StokesData

TARGET = "bias of u"
TARGET_Z = "bias of z"


def inflated(data, delta):
    """``StokesData`` with ``diag(delta^2)`` added to the noise (kept rows)."""
    if not isinstance(data, StokesData):
        raise ValueError("discrepancy_policy='inflate' needs a StokesData")
    full = jnp.zeros(data.n_data()).at[jnp.asarray(data.kept_rows())].set(delta)
    noise = data.noise + (full**2 if data.noise.ndim == 1 else jnp.diag(full**2))
    return StokesData(
        data.stokes,
        noise,
        mask=data.mask,
        response=data.response,
        discrepancy=data.discrepancy,
        response_uncertainty=data.response_uncertainty,
    )


def solve(G, target, tol):
    """Least squares on ``G D`` (``D = diag(1/||G_i||)``): ``(u, chi2, summary, pinv)``.

    The rank counts singular values of ``G D`` above ``tol`` times the
    largest, so it does not depend on the units of the unknowns.
    ``pinv = D (G D)^+`` (equal to ``G^+`` at full rank; the solution of
    minimum equilibrated norm ``||D^-1 u||`` otherwise); the covariance is
    ``D (D G^T G D)^-1 D = (G^T G)^-1`` at full rank and ``None`` otherwise;
    the Fisher matrix is ``G^T G``.
    """
    Ge, scales = equilibrate(G)
    U, s, Vt = jnp.linalg.svd(Ge, full_matrices=False)
    s_np = np.asarray(s)
    n = int(G.shape[1])
    if s_np.size == 0 or not np.all(np.isfinite(s_np)) or s_np[0] <= 0:
        raise ValueError("the whitened design has no finite nonzero singular value")
    rank = int(np.sum(s_np > tol * s_np[0]))
    Ur, sr = U[:, :rank], s[:rank]
    Vr = scales[:, None] * Vt[:rank].T  # D V_r
    pinv = (Vr / sr) @ Ur.T  # (n, n_kept) = D (G D)^+
    u = pinv @ target
    residual = target - G @ u
    covariance = (Vr / sr**2) @ Vr.T if rank == n else None
    summary = core.FisherSummary(G.T @ G, covariance, rank, s_np)
    return u, residual @ residual, summary, pinv


def _stokes_envelope(data):
    """Stokes-space discrepancy ``E`` as ``(4 n_ch,)``, zero for a data-space term."""
    value = data.discrepancy.value
    if value.shape == (data.n_data(),):
        return jnp.zeros(N_STOKES * data.n_ch)
    return jnp.broadcast_to(value, (data.n_ch, N_STOKES)).reshape(-1)


def _response_part(data, stokes_hat):
    """``(|delta R| (|S_hat| + E) on the kept rows | None, note)``; ``None`` if exact."""
    dR = jnp.abs(jnp.asarray(data.response_uncertainty))
    if not float(jnp.max(dR)) > 0.0:
        return None, "; response_uncertainty declared zero (R exact)"
    envelope = jnp.abs(jnp.asarray(stokes_hat)) + _stokes_envelope(data)
    note = (
        "; plus |delta R| (|S_hat| + E), the delta R S_hat term of extra eq: data "
        "error propagation with the fitted channel Stokes S_hat (a plug-in value, "
        "so the result is an estimate, not a bound)"
    )
    return data.select(dR @ envelope), note


def bias_bound(pinv, Linv, data, stokes_hat, *, rank, unknowns="u"):
    """``|K| |delta|`` over ``unknowns`` with ``K = pinv L^-1`` (``u_hat = K d``).

    ``delta`` is the declared discrepancy on the kept rows plus, with an
    observing response, ``|delta R| (|S_hat| + E)``. ``unbounded`` when no
    valued discrepancy is declared, when a response is set without
    ``response_uncertainty``, or when the design is rank deficient.
    """
    target = f"bias of {unknowns}"
    term = getattr(data, "discrepancy", None)
    if term is None or term.value is None:
        note = (
            "no data.discrepancy declared; the bias of the fitted vector is not bounded"
            if term is None
            else f"data.discrepancy is {term.kind}: the bias is not bounded"
        )
        return ErrorTerm.unbounded(note, target)
    kind, extra = term.kind, ""
    delta = data.discrepancy_vector()
    if getattr(data, "response", None) is not None:
        if getattr(data, "response_uncertainty", None) is None:
            return ErrorTerm.unbounded(
                "data.response is set without response_uncertainty (delta R): the "
                "|delta R S_hat| term of extra eq: data error propagation is not "
                "bounded (pass zeros to declare the response exact)",
                target,
            )
        part, extra = _response_part(data, stokes_hat)
        if part is not None:
            delta, kind = delta + part, "estimate"
    value = jnp.abs(pinv @ Linv) @ delta
    n = int(pinv.shape[0])
    if rank < n:
        return ErrorTerm.unbounded(
            f"design rank {rank} < {n}: the component of {unknowns} in the "
            f"{n - rank}-dimensional null space is not identified, so the bias of "
            f"{unknowns} is not bounded; |K| |delta| (max {float(jnp.max(value)):.6g}) "
            "bounds only the bias of the identified projection",
            target,
        )
    return ErrorTerm(
        value,
        kind,
        "|K| |delta| with K = (G^T Sigma^-1 G)^+ G^T Sigma^-1 the estimator matrix "
        f"({unknowns}_hat = K d); declared discrepancy: {term.note}{extra}",
        target,
    )


def linearised_bias(J, summary, data, stokes_hat):
    """``|K| |delta|`` over ``z`` at the optimum of ``fit_bfgs`` (an ``estimate``).

    ``J`` is the Jacobian ``(n_kept, n)`` of the whitened residual in ``z``,
    so the whitened model Jacobian is ``-J`` and ``K = (-J)^+ L^-1``, the
    Gauss-Newton response of the optimum to a data perturbation (prior and
    log-Jacobian curvature excluded). ``delta``, the response term and the
    ``unbounded`` cases follow :func:`bias_bound` with ``G = -J``; a valued
    result is downgraded to ``estimate`` because the model is linearised.
    """
    _, Linv = data.whitened()
    if summary.covariance is not None:
        pinv = -(summary.covariance @ J.T)
    else:
        Je, scales = equilibrate(J)
        pinv = -(scales[:, None] * jnp.linalg.pinv(Je))
    term = bias_bound(pinv, Linv, data, stokes_hat, rank=summary.rank, unknowns="z")
    if term.value is None:
        return term
    return ErrorTerm(
        term.value,
        "estimate",
        "linearised at the optimum (Gauss-Newton K = J^+ L^-1 of the whitened model "
        f"in z; prior and log-Jacobian curvature excluded): {term.note}",
        TARGET_Z,
    )


__all__ = ["inflated", "solve", "bias_bound", "linearised_bias", "TARGET", "TARGET_Z"]
