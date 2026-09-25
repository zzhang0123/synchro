"""Linear least-squares spectral fits and Fisher matrices (``syncmoments.model.fit.linear``).

LABEL: ``extra eq: finite fit model`` (``d = A C m + delta + eta`` with the
observing response ``R`` inside ``C``) and ``extra eq: data error
propagation`` (the declared discrepancy ``delta`` that the bias bound uses).

:func:`fit_linear` needs an affine ``ParameterMap`` (``m = P theta + c``,
``ParameterMap.is_affine()``); any other map raises ``ValueError`` naming
``fit_bfgs``. With the amplitude fitted the unknowns are ``u = (A, A theta)``
and the design is ``G = R C [c | P]``; with ``amplitude=<float>`` the
unknowns are ``theta`` and the design ``A R C P`` with the offset
``A R C c`` removed from the data. The solver minimises
``||L^-1 (d - G u)||`` on the kept rows by a rank-revealing SVD of the
column-equilibrated design ``G D``, ``D = diag(1/||G_i||)`` (singular values
below ``tol`` times the largest are dropped, so the rank does not depend on
the ``Reference`` scales; minimum equilibrated-norm solution when
rank-deficient). It reports ``A = u[0]``, ``theta = u[1:] / u[0]``, the
Fisher matrix ``G^T Sigma^-1 G``, its inverse as ``covariance`` (``None``
when rank-deficient), ``chi2``, ``dof = n_kept - rank`` and
``bias_bound = |K| |delta|`` with ``K = (G^T Sigma^-1 G)^+ G^T Sigma^-1`` the
estimator matrix (``u_hat = K d``). ``delta`` follows ``extra eq: data error
propagation``: the declared ``data.discrepancy`` on the kept rows (``|R| E``
for a Stokes-space term) plus, with an observing response,
``|delta R| (|S_hat| + E)`` from ``data.response_uncertainty`` and the fitted
channel Stokes ``S_hat``. The kind follows the discrepancy's kind, and is
``estimate`` when the plug-in ``S_hat`` term is nonzero. ``bias_bound`` is
``unbounded`` without a valued discrepancy, with a response but no
``response_uncertainty`` (pass zeros to declare ``R`` exact), and when the
design is rank deficient (the null-space component of ``u`` is not
identified). The prediction's ``statistical_input`` adds the moment bias
``|dm/du| bias_bound`` to the one-sigma noise term (``unbounded`` when the
bias is). ``discrepancy_policy="inflate"`` adds ``diag(delta^2)`` (declared
discrepancy only) to the noise covariance before solving; this is a
heuristic (noted in the provenance): the Fisher matrix, covariance,
``chi2`` and the identifiability report then describe the inflated weights,
not the noise (the ``chi2`` under the declared noise is noted). The bias
bound is computed under both policies.

:func:`fisher` is ``J^T Sigma^-1 J`` with ``J`` the ``jax.jacfwd`` Jacobian
of the whitened prediction ``L^-1 R C A m(theta)``; in the ``"natural"``
coordinates ``x = (A, flatten(theta))`` (or ``flatten(theta)`` with a fixed
``amplitude``), or in the ``"linear"`` coordinates ``u = (A, A theta)`` of
``fit_linear``. It works for any map (affine or not), is ``jax.jit`` safe
in ``theta`` and depends on ``theta`` only through the map (exact for
affine maps in the linear coordinates).

Shapes: ``G`` is ``(n_kept, 1 + n_free)`` (``(n_kept, n_free)`` with a fixed
amplitude); ``fisher`` returns ``(n, n)`` in the chosen coordinates.
Units: data, amplitude and ``chi2`` in the units of the basis prediction;
``theta`` and the moments in the displacement coordinates of the map.
Shape and configuration errors raise ``ValueError`` at trace time.
``fit_linear`` is eager (the rank and the reports are concrete). Not
certified: that ``data.noise`` is the measurement covariance, the truth of a
declared discrepancy, feasibility of the fitted moments (the attached
report lists necessary conditions only) and any statement about the
optimum beyond the closed-form least-squares solution.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..assumptions import ParameterMap, Parameters
from . import _result_core as core
from ._identifiability import whitened_design
from ._linear_core import bias_bound as _bias_bound
from ._linear_core import inflated as _inflated
from ._linear_core import solve as _solve
from ._result_core import TOL
from ._transform import _flatten
from ._result_build import from_linear
from .result import FitResult

LABEL = "extra eq: finite fit model"
POLICIES = ("bias_bound", "inflate")
COORDINATES = ("natural", "linear")


def _check_map(basis, parameter_map):
    if not isinstance(parameter_map, ParameterMap):
        raise ValueError("parameter_map must be a ParameterMap")
    if parameter_map.index != basis.index:
        raise ValueError("parameter_map.index must equal basis.index")
    C = jnp.asarray(basis.response_matrix())
    if C.ndim != 2 or C.shape[1] != basis.index.n_real:
        raise ValueError("basis.response_matrix() must be (4 n_ch, n_real)")
    return C


def _check_amplitude(amplitude):
    """``(fitted: bool, value: float | None)`` from ``"fit"`` or a positive number."""
    if isinstance(amplitude, str):
        if amplitude != "fit":
            raise ValueError("amplitude must be 'fit' or a positive number")
        return True, None
    value = np.asarray(amplitude, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or not value > 0.0:
        raise ValueError("amplitude must be 'fit' or a positive finite scalar")
    return False, float(value)


def fit_linear(
    basis,
    data,
    parameter_map,
    *,
    amplitude="fit",
    discrepancy_policy="bias_bound",
    tol=TOL,
    diagnostics=True,
) -> FitResult:
    """Closed-form least-squares fit of an affine map (module docstring).

    ``basis``: ``SpectralBasis``; ``data``: ``StokesData``; ``parameter_map``:
    an affine ``ParameterMap`` on ``basis.index``. ``amplitude="fit"`` or a
    positive float; ``discrepancy_policy`` in ``POLICIES``; ``tol`` the
    relative singular-value cutoff on the column-equilibrated design (passed
    to ``identifiability``, so both report the same rank);
    ``diagnostics=False`` skips the identifiability report, the feasibility
    checks and the prediction (keyword addition, for repeated fits). Raises
    ``ValueError`` for a non-affine map (use ``fit_bfgs``), an index or
    channel mismatch, an unknown policy, ``"inflate"`` without a valued
    discrepancy, or a non-positive fixed amplitude. Returns a ``FitResult``;
    shapes, units and what is not certified: module docstring.
    """
    C = _check_map(basis, parameter_map)
    if not parameter_map.is_affine():
        raise ValueError(
            f"parameter map '{parameter_map.name}' is not affine (several free "
            "groups, fitted hyper-parameters or nodal); use fit_bfgs or fit_nodal"
        )
    if discrepancy_policy not in POLICIES:
        raise ValueError(f"discrepancy_policy must be one of {POLICIES}")
    fitted, fixed = _check_amplitude(amplitude)
    tol = core.check_tol(tol)
    if 4 * int(data.n_ch) != int(C.shape[0]):
        raise ValueError(
            f"data has {data.n_ch} channels but the basis has {C.shape[0] // 4}"
        )
    delta = data.discrepancy_vector() if hasattr(data, "discrepancy_vector") else None
    if discrepancy_policy == "inflate":
        if delta is None:
            raise ValueError(
                "discrepancy_policy='inflate' needs a valued data.discrepancy"
            )
        fit_data = _inflated(data, delta)
    else:
        fit_data = data
    P, c = parameter_map.affine_pieces(reference=basis.reference)
    target, Linv = fit_data.whitened()
    if fitted:
        G = whitened_design(fit_data, C, jnp.concatenate([c[:, None], P], axis=1))
        labels = ("amplitude",) + tuple(parameter_map.labels())
    else:
        G = fixed * whitened_design(fit_data, C, P)
        target = target - fixed * whitened_design(fit_data, C, c[:, None])[:, 0]
        labels = tuple(parameter_map.labels())
    u, chi2, summary, pinv = _solve(G, target, tol)
    if fitted:
        stokes_hat = C @ (c * u[0] + P @ u[1:])
    else:
        stokes_hat = fixed * (C @ (c + P @ u))
    bias_bound = _bias_bound(pinv, Linv, data, stokes_hat, rank=summary.rank)
    notes = [
        f"fit: linear least squares by rank-revealing SVD of the column-equilibrated "
        f"design, {len(labels)} unknowns "
        + ("u = (A, A theta)" if fitted else f"theta (amplitude fixed at {fixed:.6g})")
        + f", rank {summary.rank}, tol {tol:g}",
        f"discrepancy_policy={discrepancy_policy}: " + _policy_note(discrepancy_policy),
    ]
    if discrepancy_policy == "inflate":
        notes.append(_declared_chi2_note(data, stokes_hat))
    if fitted:
        A = u[0]
        A_np = float(A)
        safe = jnp.where(A != 0.0, A, 1.0)
        flat = jnp.where(A != 0.0, u[1:] / safe, 0.0)
        if A_np <= 0.0:
            notes.append(
                f"fitted amplitude {A_np:.6g} is not positive; theta = u[1:]/u[0] "
                + ("is left zero" if A_np == 0.0 else "has the sign of u[0]")
            )
        theta = parameter_map.unflatten(flat)
        log_amplitude = jnp.log(A) if A_np > 0.0 else None
    else:
        A = jnp.asarray(fixed)
        theta = parameter_map.unflatten(u)
        log_amplitude = jnp.log(A)
    theta = Parameters(
        log_amplitude=log_amplitude,
        tables=theta.tables,
        hyper=theta.hyper,
        logits=theta.logits,
    )
    return from_linear(
        basis=basis,
        data=fit_data,
        parameter_map=parameter_map,
        theta=theta,
        amplitude=A,
        u=u,
        summary=summary,
        chi2=chi2,
        bias_bound=bias_bound,
        labels=labels,
        amplitude_fitted=fitted,
        notes=notes,
        diagnostics=diagnostics,
        tol=tol,
    )


def _policy_note(policy):
    if policy == "inflate":
        return (
            "heuristic; diag(delta^2) of the declared discrepancy added to the noise "
            "covariance, so the Fisher matrix, covariance, chi2 and the "
            "identifiability report describe the inflated weights, not the noise"
        )
    return "bias_bound = |K| |delta| over the unknowns"


def _declared_chi2_note(data, stokes_hat):
    """``chi2`` of the inflated solution under the declared noise ``data.noise``."""
    target, Linv = data.whitened()
    residual = target - Linv @ data.design(stokes_hat[:, None])[:, 0]
    value = float(residual @ residual)
    return (
        f"chi2 under the declared noise (not the inflated weights): {value:.6g} "
        f"on {data.n_kept()} kept rows"
    )


def fisher(
    basis, data, parameter_map, theta, *, amplitude=None, coordinates="natural"
) -> jax.Array:
    """``J^T Sigma^-1 J`` of the whitened prediction (module docstring).

    ``theta`` is a ``Parameters`` of ``parameter_map``; its ``log_amplitude``
    supplies ``A`` unless ``amplitude=`` fixes it (then the amplitude is not
    a coordinate; passing both raises ``ValueError``). ``coordinates``:
    ``"natural"`` for ``x = (A, flatten(theta))`` or ``"linear"`` for
    ``u = (A, A theta)`` (``v = A theta`` with a fixed amplitude). Returns
    ``(n, n)``; ``jax.jit`` safe in ``theta``. A local information matrix;
    not certified as a covariance unless ``data.noise`` is the noise model.
    """
    C = _check_map(basis, parameter_map)
    if coordinates not in COORDINATES:
        raise ValueError(f"coordinates must be one of {COORDINATES}")
    if 4 * int(data.n_ch) != int(C.shape[0]):
        raise ValueError("data channels do not match the basis response matrix")
    flat = _flatten(parameter_map, theta)  # jit-safe layout of ParameterMap.flatten
    if amplitude is None:
        if theta.log_amplitude is None:
            raise ValueError(
                "theta.log_amplitude is required unless amplitude= is given"
            )
        A = jnp.exp(jnp.asarray(theta.log_amplitude, dtype=float))
    else:
        if theta.log_amplitude is not None:
            raise ValueError("pass either theta.log_amplitude or amplitude=, not both")
        A = jnp.asarray(amplitude, dtype=float)
        if A.ndim != 0:
            raise ValueError("amplitude must be a scalar")
    _, Linv = data.whitened()
    reference = basis.reference

    def whitened_model(A_, v):
        m = parameter_map(parameter_map.unflatten(v), reference).to_vector()
        return Linv @ data.design(C @ (A_ * m))

    if amplitude is None:
        x = jnp.concatenate([A[None], flat])
        if coordinates == "natural":

            def model(x_):
                return whitened_model(x_[0], x_[1:])

        else:
            x = jnp.concatenate([A[None], A * flat])

            def model(x_):
                return whitened_model(x_[0], x_[1:] / x_[0])

    else:
        x = flat if coordinates == "natural" else A * flat
        scale = 1.0 if coordinates == "natural" else A

        def model(x_):
            return whitened_model(A, x_ / scale)

    J = jax.jacfwd(model)(x)
    return J.T @ J


__all__ = ["fit_linear", "fisher", "LABEL", "POLICIES", "COORDINATES", "TOL"]
