"""Nonlinear spectral fits: transforms, log density, BFGS and nodal fits.

LABEL: ``extra eq: finite fit model`` (the chi-square of
``d = A C m(theta) + delta + eta`` for a general :class:`ParameterMap`),
``extra eq: nodal distribution`` and ``extra eq: nodal bound``
(:func:`fit_nodal`: nonnegative, normalised weights on fixed nodes with the
discretisation error supplied as an input).

Coordinates. :class:`Transform` maps the ``Parameters`` of a map to one
real unconstrained vector ``z`` and back (class in ``_transform.py``):
``log_amplitude`` is carried as is (``A = exp z_0``; a prior on it is a
prior on ``log A``, no Jacobian), free tables are identity,
hyper-parameters that must be positive (labels ``hyper:...:sigma`` and
``hyper:...:kappa_2``) are carried as their logarithm, nodal logits are
identity (the map applies the softmax), and any slot given explicit
``bounds`` is a tanh box. ``log_det`` is the log Jacobian of the
constrained coordinates ``(log A, tables, positive hyper, boxed slots,
simplex weights)`` with respect to ``z``, so a prior stated on those
coordinates becomes a density in ``z``. For nodal logits it is the
additive-log-ratio Jacobian ``sum log softmax(z)`` with the softmax gauge
direction left flat (improper). With a flat prior the mode of the density
in ``z`` is therefore not the maximum-likelihood point when a log or box
slot is present; a prior of ``-log sigma`` (or ``-log_det``) restores it.

:class:`LogDensity` evaluates ``-chi^2/2 + log prior(theta) + log_det(z)``
with the whitened residual of :class:`StokesData` on the kept rows; it is
``jax.jit``, ``jax.grad`` and ``jax.vmap`` ready in ``z``. The data
``discrepancy`` and ``response_uncertainty`` are not part of the density;
they enter the ``bias_bound`` of the ``FitResult`` (for ``fit_bfgs`` an
estimate linearised at the optimum) and through it ``statistical_input``,
which is ``unbounded`` without a declared discrepancy. :func:`fit_bfgs` minimises
``-LogDensity`` with ``jax.scipy.optimize.minimize(method="BFGS")`` and
refuses more than ``MAX_BFGS_PARAMS`` parameters (dense inverse Hessian).
Both fitters build the private :class:`_FitOutcome` and return it wrapped
as a ``FitResult`` (``fit/result.py``: Gauss-Newton Fisher matrix in the fit
coordinates, diagnostics and prediction; ``diagnostics=False`` skips the
reports). The ``FitResult`` exposes ``z``, ``fun``, ``weights`` and
``discretisation`` with the meanings given for ``_FitOutcome``.

Shapes: ``z`` is ``(n_params,)``; the whitened residual is ``(n_kept,)``.
Units: amplitudes and data in the units of the basis prediction; moments in
the displacement coordinates of the map. Shape and configuration errors
raise ``ValueError`` at trace time; value errors (a nonpositive value for a
log slot, a boxed value outside its interval) use ``equinox.error_if``;
inside :func:`fit_bfgs` a line-search trial point that would trip one is
given objective ``+inf`` instead (``_bfgs.total_objective``).
Not certified: convergence to the global optimum (BFGS reports its own
``success`` or the precision floor; the nodal fit reports the simplex
stationarity residual), feasibility of the fitted moments, and the nodal
discretisation error, which the caller supplies.
"""

from __future__ import annotations

from typing import Any, ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

from ..assumptions import ParameterMap, Parameters, nodal
from ..errors import ErrorTerm
from ..moments import PopulationSamples
from . import _nodal
from ._bfgs import PRECISION_FLOOR  # noqa: F401  (re-exported)
from ._bfgs import accepted_point as _accepted_point
from ._bfgs import at_precision_floor as _at_precision_floor
from ._bfgs import gauss_newton_scale as _gauss_newton_scale
from ._bfgs import total_objective as _total_objective
from ._transform import AMPLITUDE_LABEL, POSITIVE_HYPER, Transform

MAX_BFGS_PARAMS = 2000


class _FitOutcome(NamedTuple):
    """Private fit outcome (wrapped into ``FitResult`` by ``fit/result.py``).

    ``theta``: fitted ``Parameters``; ``z``: the unconstrained vector;
    ``converged``: optimiser flag; ``n_iter``: iterations; ``fun``: the
    minimised objective (``-log density`` for BFGS, half chi-square for the
    nodal fit); ``weights``: simplex weights of a nodal fit (else ``None``);
    ``discretisation``: the ``ErrorTerm`` passed to ``fit_nodal`` (else ``None``).
    """

    theta: Parameters
    z: jax.Array
    converged: bool
    n_iter: int
    fun: float
    weights: jax.Array | None = None
    discretisation: ErrorTerm | None = None


class LogDensity(eqx.Module):
    """``z -> -chi^2/2 + log prior(theta) + log_det(z)`` for a spectral basis.

    ``chi^2 = ||L^-1 (d - R (A C m(theta)))||^2`` on the kept rows of
    ``data`` (``StokesData.whitened``), ``C = basis.response_matrix()``
    (``(4 n_ch, n_real)``, channel-major), ``m(theta)`` the map's moment
    vector at ``basis.reference``, ``A = exp(theta.log_amplitude)`` from
    the transform or the fixed keyword ``amplitude`` when the transform
    carries none. ``prior(theta) -> scalar`` is a log prior on the
    constrained coordinates (``None`` for flat). The whitened design
    ``W = L^-1 R C`` and target ``L^-1 d`` are precomputed leaves.
    Dimensionless value (log density in ``z``); data and basis in the
    prediction's units. Assumes Gaussian noise (``StokesData``) and the
    map's factorisation; not certified: the global optimum, feasibility of
    ``m(theta)``, and the data discrepancy, which is not in the density.
    """

    LABEL: ClassVar[str] = "extra eq: finite fit model"
    basis: Any
    data: Any
    parameter_map: ParameterMap
    prior: Any = eqx.field(static=True)
    transform: Transform
    amplitude: jax.Array | None
    design: jax.Array
    target: jax.Array

    def __init__(
        self, basis, data, parameter_map, prior=None, transform=None, *, amplitude=None
    ):
        if not isinstance(parameter_map, ParameterMap):
            raise ValueError("parameter_map must be a ParameterMap")
        if transform is None:
            transform = Transform(parameter_map, amplitude=amplitude is None)
        if not isinstance(transform, Transform):
            raise ValueError("transform must be a Transform")
        if transform.parameter_map.labels() != parameter_map.labels():
            raise ValueError("transform was built for a different parameter map")
        if parameter_map.index != basis.index:
            raise ValueError("parameter_map.index must equal basis.index")
        if prior is not None and not callable(prior):
            raise ValueError("prior must be callable or None")
        if transform.amplitude and amplitude is not None:
            raise ValueError("amplitude is fitted by the transform; do not fix it")
        if not transform.amplitude and amplitude is None:
            raise ValueError("transform carries no amplitude; pass amplitude=")
        C = jnp.asarray(basis.response_matrix())
        if C.shape != (4 * data.n_ch, parameter_map.index.n_real):
            raise ValueError(
                f"response matrix {C.shape} does not match data (4 n_ch = "
                f"{4 * data.n_ch}) and index (n_real = {parameter_map.index.n_real})"
            )
        target, Linv = data.whitened()
        self.basis = basis
        self.data = data
        self.parameter_map = parameter_map
        self.prior = prior
        self.transform = transform
        self.amplitude = None if amplitude is None else jnp.asarray(amplitude, float)
        self.design = Linv @ data.design(C)
        self.target = target

    @property
    def n_params(self) -> int:
        return self.transform.n_params

    def moments(self, z):
        """``JointMoments`` of the map at ``transform.inverse(z)``."""
        theta = self.transform.inverse(z)
        return theta, self.parameter_map(theta, self.basis.reference)

    def whitened_residual(self, z) -> jax.Array:
        """``L^-1 (d - R A C m)`` on the kept rows, ``(n_kept,)``."""
        theta, moments = self.moments(z)
        amplitude = self.amplitude
        if amplitude is None:
            amplitude = jnp.exp(theta.log_amplitude)
        return self.target - amplitude * (self.design @ moments.to_vector())

    def chi2(self, z) -> jax.Array:
        r = self.whitened_residual(z)
        return r @ r

    def __call__(self, z) -> jax.Array:
        value = -0.5 * self.chi2(z) + self.transform.log_det(z)
        if self.prior is not None:
            theta = self.transform.inverse(z)
            log_prior = jnp.asarray(self.prior(theta))
            if jnp.iscomplexobj(log_prior) or log_prior.shape != ():
                raise ValueError("prior(theta) must return a real scalar")
            value = value + log_prior
        return value


def fit_bfgs(
    logdensity,
    z0,
    *,
    maxiter=500,
    tol=1e-9,
    precondition=True,
    restarts=2,
    line_search_maxiter=30,
    diagnostics=True,
) -> "FitResult":  # noqa: F821  (defined in fit/result.py)
    """Maximise ``logdensity`` from ``z0`` with ``jax.scipy.optimize.minimize`` (BFGS).

    Refuses ``n_params > MAX_BFGS_PARAMS`` (``ValueError``): BFGS stores a
    dense inverse Hessian. ``tol`` is the gradient-norm tolerance (passed as
    ``gtol``; JAX ignores its own ``tol`` argument) and ``maxiter`` the total
    iteration cap. With ``precondition`` (default) the search runs in
    ``y = z / s`` with the Gauss-Newton diagonal scale ``s`` of
    :func:`_gauss_newton_scale` at ``z0`` (the whitened chi-square is badly
    scaled in ``z`` otherwise and the JAX line search then fails). After a
    line-search failure the solver is restarted from the current point with
    a fresh inverse Hessian, at most ``restarts`` times. A trial point the
    density cannot evaluate (non-finite, or a log slot beyond
    ``LOG_SLOT_LIMIT`` of ``_bfgs.py``) has objective ``+inf`` and is
    rejected by the line search; after each run the lowest finite point of
    ``_bfgs.accepted_point`` is kept. Returns a
    ``FitResult`` (``fit/result.py``) with ``fun`` the minimised ``-logdensity``;
    ``converged`` is the optimiser's ``success`` (``gtol`` met) or the
    precision floor of :func:`_at_precision_floor` (a line search that
    stops with a predicted decrease below ``PRECISION_FLOOR`` relative);
    ``n_iter`` is the total iteration count. The optimum is not certified
    beyond that flag. Assumes the density's Gaussian noise model and the
    map's factorisation; ``z`` is dimensionless, ``fun`` in the units of
    the log density.
    """
    if not isinstance(logdensity, LogDensity):
        raise ValueError("logdensity must be a LogDensity")
    z0 = jnp.asarray(z0)
    if z0.ndim != 1 or jnp.iscomplexobj(z0):
        raise ValueError("z0 must be a real one-dimensional vector")
    n = int(z0.shape[0])
    if n > MAX_BFGS_PARAMS:
        raise ValueError(
            f"fit_bfgs refuses n_params={n} > {MAX_BFGS_PARAMS}; use fit_linear on "
            "an affine map or a nodal fit"
        )
    if n != logdensity.n_params:
        raise ValueError(f"z0 must have shape ({logdensity.n_params},), got ({n},)")
    for name, value in (
        ("maxiter", maxiter),
        ("line_search_maxiter", line_search_maxiter),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        isinstance(restarts, bool)
        or not isinstance(restarts, (int, np.integer))
        or restarts < 0
    ):
        raise ValueError("restarts must be a nonnegative integer")
    z0 = z0.astype(jnp.result_type(z0, 1.0))
    scale = _gauss_newton_scale(logdensity, z0) if precondition else jnp.ones(n)

    objective = _total_objective(logdensity, scale)
    y, n_iter, remaining = z0 / scale, 0, int(maxiter)
    for _ in range(int(restarts) + 1):
        result = minimize(
            objective,
            y,
            method="BFGS",
            options={
                "gtol": float(tol),
                "maxiter": remaining,
                "line_search_maxiter": int(line_search_maxiter),
            },
        )
        y, n_iter = _accepted_point(objective, y, result), n_iter + int(result.nit)
        remaining = int(maxiter) - n_iter
        converged = bool(result.success) or _at_precision_floor(result)
        if converged or remaining <= 0 or int(result.nit) == 0:
            break
    z = scale * y
    outcome = _FitOutcome(
        theta=logdensity.transform.inverse(z),
        z=z,
        converged=converged,
        n_iter=n_iter,
        fun=float(-logdensity(z)),
    )
    from ._result_build import from_bfgs  # deferred: keeps the predict chain out

    return from_bfgs(outcome, logdensity, diagnostics=diagnostics)


def fit_nodal(
    basis,
    data,
    nodes,
    *,
    iters=500,
    step=0.1,
    discretisation=None,
    tol=1e-8,
    diagnostics=True,
) -> "FitResult":  # noqa: F821  (defined in fit/result.py)
    """Nonnegative normalised weights on fixed ``nodes`` (``extra eq: nodal distribution``).

    Solves ``min_{w >= 0, sum w = 1, A >= 0} ||L^-1 (d - R A C X^T w)||^2``
    by monotone exponentiated-gradient updates with the amplitude profiled
    (``O(S)`` memory, no ``(n_data, S)`` matrix). ``step`` is the initial
    relative step (fraction of the initial gradient range), ``iters`` the
    cap, ``tol`` the simplex stationarity tolerance
    ``max_i w_i |g_i - <g, w>| / (1 + max |g|)`` (``converged``). Nodes with
    zero weight in ``nodes`` start at zero and stay there (the updates are
    multiplicative); uniform weights are the neutral start. ``discretisation`` is
    the caller's ``ErrorTerm`` for ``extra eq: nodal bound`` (``None`` leaves
    it unbounded downstream); it is validated and passed through. Returns a
    :class:`_FitOutcome` with ``theta = Parameters(log_amplitude, logits=log w)``
    for the map ``nodal(basis.index, nodes)``, ``weights = w`` and ``fun``
    the final half chi-square. The optimum is not certified beyond
    ``converged``. Assumes the population is supported on ``nodes`` (the
    discretisation error is the caller's ``discretisation`` term) and the
    Gaussian noise model; weights are dimensionless, ``fun`` in data units.
    """
    if not isinstance(nodes, PopulationSamples):
        raise ValueError("nodes must be a PopulationSamples")
    if discretisation is not None and not isinstance(discretisation, ErrorTerm):
        raise ValueError("discretisation must be an ErrorTerm or None")
    if not isinstance(iters, (int, np.integer)) or iters < 1:
        raise ValueError("iters must be a positive integer")
    if not float(step) > 0.0:
        raise ValueError("step must be positive")
    parameter_map = nodal(basis.index, nodes)
    C = jnp.asarray(basis.response_matrix())
    if C.shape != (4 * data.n_ch, basis.index.n_real):
        raise ValueError("response matrix does not match data and index")
    target, Linv = data.whitened()
    design = Linv @ data.design(C)
    operator = _nodal.NodeOperator(nodes, basis.index, basis.reference)
    w0 = nodes.normalised_weights()

    def forward(w):
        return design @ operator.moments(w)

    def adjoint(r):
        return operator.adjoint(design.T @ r)

    run = jax.jit(
        lambda t, w: _nodal.simplex_fit(
            forward, adjoint, t, w, iters=int(iters), step=float(step), tol=float(tol)
        )
    )
    w, amplitude, fun, n_iter, converged, _ = run(target, w0)
    logits = jnp.log(jnp.maximum(w, 1e-300))
    log_amplitude = jnp.log(jnp.maximum(amplitude, 1e-300))
    theta = Parameters(log_amplitude=log_amplitude, logits=logits)
    z = Transform(parameter_map, amplitude=True).forward(theta)
    outcome = _FitOutcome(
        theta=theta,
        z=z,
        converged=bool(converged),
        n_iter=int(n_iter),
        fun=float(fun),
        weights=w,
        discretisation=discretisation,
    )
    from ._result_build import from_nodal  # deferred: keeps the predict chain out

    return from_nodal(outcome, basis, data, nodes, diagnostics=diagnostics)


__all__ = [
    "Parameters",
    "Transform",
    "AMPLITUDE_LABEL",
    "LogDensity",
    "fit_bfgs",
    "fit_nodal",
    "MAX_BFGS_PARAMS",
    "POSITIVE_HYPER",
]
