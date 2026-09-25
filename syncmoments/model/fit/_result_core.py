"""Private helpers of :mod:`syncmoments.model.fit.result` and :mod:`.linear`.

Fisher matrices from whitened Jacobians (rank on the column-equilibrated
Jacobian, pseudo-inverse), the moment uncertainty that fills
``statistical_input`` of a fit's prediction (propagated 1-sigma noise plus
the discrepancy-induced moment bias, ``unbounded`` without one), the closed-form
nodal Jacobian, the prediction assembly and the JSON rendering of
``Parameters``. Nothing here is public API and nothing here certifies a
fit: the rank cutoff is a numerical convention (``tol`` relative to the
largest singular value) and the moment uncertainty is a linearised
propagation, not an envelope.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from .._provenance import strict_json
from ..errors import ErrorBudget, ErrorTerm, Provenance
from ..predict import Prediction, predict
from . import _nodal
from ._identifiability import equilibrate
from .diagnostics import feasibility_checks, identifiability

TOL = 1e-8
MAX_FISHER_PARAMS = 2000


@dataclasses.dataclass(frozen=True)
class FisherSummary:
    """``fisher`` ``(n, n)``, ``covariance`` (``None`` when rank-deficient), static rank."""

    fisher: jax.Array | None
    covariance: jax.Array | None
    rank: int
    singular_values: np.ndarray | None


def fisher_from_jacobian(J, *, tol=TOL) -> FisherSummary:
    """Gauss-Newton Fisher ``J^T J`` of a whitened Jacobian ``(n_kept, n)``.

    The rank counts singular values of the column-equilibrated ``J D``
    (``D = diag(1/||J_i||)``, the rule of ``fit_linear`` and
    ``identifiability``) above ``tol`` times the largest (concrete, eager);
    the covariance is ``D (D J^T J D)^-1 D = (J^T J)^-1`` through that SVD
    when the rank is full and ``None`` otherwise. ``singular_values`` are
    those of ``J D``. A non-finite Jacobian gives rank 0 and no covariance.
    """
    J = jnp.asarray(J)
    if J.ndim != 2:
        raise ValueError("the Jacobian must be two-dimensional (n_kept, n_params)")
    n = int(J.shape[1])
    fisher = J.T @ J
    if not np.all(np.isfinite(np.asarray(J))):
        return FisherSummary(fisher, None, 0, None)
    Je, scales = equilibrate(J)
    _, s, Vt = jnp.linalg.svd(Je, full_matrices=False)
    s_np = np.asarray(s)
    if not np.all(np.isfinite(s_np)) or s_np.size == 0 or s_np[0] <= 0:
        return FisherSummary(fisher, None, 0, s_np)
    rank = int(np.sum(s_np > tol * s_np[0]))
    covariance = None
    if rank == n:
        V = scales[:, None] * Vt.T
        covariance = (V / (s**2)) @ V.T
    return FisherSummary(fisher, covariance, rank, s_np)


def check_tol(tol) -> float:
    tol = float(tol)
    if not tol >= 0.0:
        raise ValueError("tol must be nonnegative")
    return tol


STATISTICAL_TERM = "N_src sum_a |S_a| Delta_a"


def moment_uncertainty(m_of, x, covariance, *, bias_bound, bias_note=None) -> ErrorTerm:
    """``Delta_a`` over ``m`` for ``statistical_input``: noise plus discrepancy bias.

    The noise part is the one-sigma ``sqrt(diag(J cov J^T))``, ``J = dm/dx``
    (linearised). ``bias_bound`` (a term over ``x``: ``u`` for ``fit_linear``,
    ``z`` for ``fit_bfgs``) adds ``|J| bias_bound`` (``eq: channel error
    budget`` defines ``Delta_a`` by ``|m_a - m_hat_a| <= Delta_a``, so the
    moment bias from the declared discrepancy belongs here). ``bias_note``
    says how exact that propagation is (default: the ``fit_linear`` text,
    exact with a fixed amplitude, linearised through ``theta = u[1:]/u[0]``
    otherwise). An ``unbounded`` or missing (``None``) bias makes the term
    ``unbounded`` (the noise part is quoted in the note). Returns an
    ``estimate`` term ``(n_real,)`` that ``predict`` contracts as ``A |C| Delta``.
    """
    J = jax.jacfwd(m_of)(x)
    var = jnp.sum((J @ covariance) * J, axis=1)
    delta = jnp.sqrt(jnp.maximum(var, 0.0))
    noise_note = (
        "one-sigma statistical uncertainty of the fitted moments, linearised "
        "propagation of the Fisher covariance (a standard deviation, not an envelope)"
    )
    if bias_bound is None or bias_bound.value is None:
        reason = "none computed" if bias_bound is None else bias_bound.note
        return ErrorTerm.unbounded(
            "discrepancy-induced bias of the fitted moments not bounded "
            f"(bias_bound: {reason}); the {noise_note} alone has max "
            f"{float(jnp.max(delta)):.6g}",
            STATISTICAL_TERM,
        )
    if bias_note is None:
        bias_note = (
            f"|dm/du| bias_bound ({bias_bound.kind} in u; linear in u only with a "
            "fixed amplitude)"
        )
    bias = jnp.abs(J) @ jnp.asarray(bias_bound.value)
    return ErrorTerm(
        delta + bias,
        "estimate",
        f"{noise_note}, plus the discrepancy-induced bias of the moments "
        f"{bias_note}, max {float(jnp.max(bias)):.6g}",
        STATISTICAL_TERM,
    )


def nodal_jacobian(basis, data, nodes, w, amplitude, design) -> jax.Array:
    """Whitened Jacobian of ``A L^-1 R C X^T softmax(logits)`` in ``(log A, logits)``.

    ``design`` is ``L^-1 R C`` ``(n_kept, n_real)``; ``B = design X^T`` is
    formed row by row with the chunked node operator (never ``X`` for large
    ``S``); ``d softmax / d logits = diag(w) - w w^T``. Shape ``(n_kept, S + 1)``.
    """
    operator = _nodal.NodeOperator(nodes, basis.index, basis.reference)
    w = jnp.asarray(w, dtype=float)
    B = jax.vmap(operator.adjoint)(design)  # (n_kept, S)
    Bw = B @ w
    J_logits = amplitude * (B * w[None, :] - Bw[:, None] * w[None, :])
    J_logA = (amplitude * Bw)[:, None]
    return jnp.concatenate([J_logA, J_logits], axis=1)


def finite_moments(moments, amplitude) -> bool:
    values = np.concatenate(
        [np.asarray(moments.to_vector(), dtype=float), [float(amplitude)]]
    )
    return bool(np.all(np.isfinite(values)))


def diagnostics_of(
    basis, data, parameter_map, theta, *, amplitude, moments, notes, tol=TOL
):
    """``(identifiability, feasibility)`` or ``None`` entries with a note each.

    ``tol`` is the fit's singular-value cutoff, so the identifiability rank
    follows the same rule as the fit.
    """
    report = None
    if parameter_map.nodes is not None:
        notes.append(
            "identifiability not computed for a nodal map (the Fisher rank in "
            "(log A, logits) carries the same information)"
        )
    else:
        report = identifiability(
            basis,
            data,
            parameter_map,
            theta=theta,
            reference=basis.reference,
            amplitude=amplitude,
            tol=tol,
        )
    support = getattr(basis, "support", None)
    feasibility = feasibility_checks(moments, basis.index, support)
    return report, feasibility


def _rebuilt(prediction, *, budget=None, provenance=None) -> Prediction:
    return Prediction(
        stokes=prediction.stokes,
        amplitude=prediction.amplitude,
        moments=prediction.moments,
        budget=prediction.budget if budget is None else budget,
        provenance=prediction.provenance if provenance is None else provenance,
        channels=prediction.channels,
    )


def _replace_assumption(prediction, name, term) -> Prediction:
    budget = prediction.budget
    pairs = tuple((n, term if n == name else t) for n, t in budget.assumption)
    new_budget = ErrorBudget(
        **{slot: getattr(budget, slot) for slot in ErrorBudget.SLOTS}, assumption=pairs
    )
    return _rebuilt(prediction, budget=new_budget)


def prediction_of(
    basis, moments, amplitude, *, statistical, discretisation, notes, hyper_note=None
):
    """``predict(basis, moments, amplitude=)`` with the fit's statistical input.

    ``discretisation`` (nodal fits) replaces the ``unbounded`` assumption term
    of the nodal record by the supplied ``extra eq: nodal bound`` term.
    ``hyper_note`` (the fitted hyper-parameter values, whose record holds NaN
    placeholders) is appended to the prediction's provenance notes.
    """
    needed = ("kernel_terms", "provenance", "channels", "support")
    if any(not hasattr(basis, name) for name in needed):
        notes.append("prediction not computed: basis lacks the SpectralBasis fields")
        return None
    if not float(amplitude) > 0.0:
        notes.append("prediction not computed: fitted amplitude is not positive")
        return None
    prediction = predict(
        basis, moments, amplitude=amplitude, statistical_input=statistical
    )
    if discretisation is not None and moments.assumptions:
        name = moments.assumptions[0].name
        term = ErrorTerm(
            discretisation.value,
            discretisation.kind,
            "extra eq: nodal bound (supplied discretisation term): "
            + discretisation.note,
            "E_phys",
        )
        prediction = _replace_assumption(prediction, name, term)
    if hyper_note is not None:
        provenance = prediction.provenance.with_notes(hyper_note)
        prediction = _rebuilt(prediction, provenance=provenance)
    return prediction


def provenance_of(basis, records, notes) -> Provenance:
    """Basis provenance with the fit's assumption records and notes appended."""
    provenance = getattr(basis, "provenance", None)
    if provenance is None:
        provenance = Provenance(
            "unknown", (), (), (), (), (), (), (), (), "unknown", (), (), (), ()
        )
        notes = ["basis carries no provenance record"] + list(notes)
    present = {record.name for record in provenance.assumptions}
    new = tuple(r for r in records if r.name not in present)
    return provenance.with_assumptions(*new).with_notes(*notes)


def _array_json(value):
    if value is None:
        return None
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return {"real": array.real.tolist(), "imag": array.imag.tolist()}
    return array.tolist()


def jsonable(value):
    """Strict-JSON rendering shared with every ``to_dict`` (``strict_json``).

    Tuples become lists, NumPy scalars Python ones, and non-finite floats
    (the NaN placeholders of fitted hyper-parameters in an
    ``AssumptionRecord``, a non-finite ``chi2``) ``None``.
    """
    return strict_json(value)


def theta_json(theta) -> dict:
    """JSON-ready ``Parameters`` (complex tables as ``{"real", "imag"}``)."""
    return {
        "log_amplitude": (
            None if theta.log_amplitude is None else float(theta.log_amplitude)
        ),
        "tables": [_array_json(t) for t in theta.tables],
        "hyper": [_array_json(h) for h in theta.hyper],
        "logits": _array_json(theta.logits),
    }


__all__ = [
    "TOL",
    "MAX_FISHER_PARAMS",
    "FisherSummary",
    "fisher_from_jacobian",
    "check_tol",
    "moment_uncertainty",
    "nodal_jacobian",
    "finite_moments",
    "diagnostics_of",
    "prediction_of",
    "provenance_of",
    "jsonable",
    "theta_json",
]
