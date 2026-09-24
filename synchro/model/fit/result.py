"""The ``FitResult`` container of every spectral fit (``synchro.model.fit.result``).

LABEL: ``extra eq: finite fit model`` (the fitted ``u = (A, A theta)`` or
``theta`` of ``d = A C m(theta) + delta + eta``), ``eq: joint moment
feasible set`` (the attached :class:`FeasibilityReport`), ``eq: channel error
budget`` (the attached :class:`Prediction` and its budget), ``extra eq:
nodal bound`` (the ``discretisation`` term of a nodal fit).

A :class:`FitResult` holds the fitted parameters, the moments they imply,
the amplitude, the Fisher matrix and its covariance in the fit's own
coordinates (named by ``labels`` and described by ``coordinates``): for
``fit_linear`` the unknowns ``u = (A, A theta)`` (or ``theta`` with a fixed
amplitude); for ``fit_bfgs`` the unconstrained ``z`` of its ``Transform``;
for ``fit_nodal`` ``(log A, logits)``. ``covariance`` is ``None`` whenever
the Fisher matrix is rank-deficient at the relative cutoff ``tol``; ``rank``
is the numerical rank, ``dof = n_kept - rank`` (``n_kept - n_params`` when
no Fisher matrix was formed). ``chi2`` is the whitened residual norm
squared at the fitted point, in the data units.

``bias_bound`` is ``|K| |delta|`` over the fit coordinates, ``K`` the
estimator matrix and ``delta`` the declared ``data.discrepancy`` on the kept
rows plus, with an observing response, ``|delta R| (|S_hat| + E)`` (``extra
eq: data error propagation``). ``fit_linear``: over ``u``, kind of the
discrepancy (``estimate`` when the plug-in ``S_hat`` term is nonzero).
``fit_bfgs``: over ``z`` with ``K = J^+ L^-1`` from the whitened Jacobian,
an ``estimate`` linearised at the optimum (prior and log-Jacobian curvature
excluded). It is ``unbounded`` when no valued discrepancy is declared, when
a response is set without ``response_uncertainty`` (zeros declare ``R``
exact) and when the Fisher or design rank is below the number of unknowns
(only the identified projection is bounded; its max is quoted in the note);
``None`` for ``fit_nodal``. ``identifiability``, ``feasibility`` and
``prediction`` are ``None`` when not computed (``diagnostics=False``, nodal
maps for the identifiability report, non-finite fits); the ``provenance``
notes say why. The prediction's ``statistical_input`` is
``Delta = sigma + |dm/dx| bias_bound`` over ``m``: the one-sigma noise
propagated from the Fisher covariance plus the propagated bias (an
``estimate``, not an envelope). It is ``unbounded`` whenever ``bias_bound``
is (including the default fit with no declared discrepancy; the noise part
is quoted in the note) and whenever no covariance exists (every nodal fit:
the softmax gauge makes its Fisher matrix singular). A map with fitted
hyper-parameters records NaN placeholders in its ``AssumptionRecord``; the
fitted values are in ``theta`` and in a ``provenance`` note.

Shapes: ``fisher`` and ``covariance`` are ``(n, n)`` with ``n = len(labels)``.
Units: data and amplitude in the units of the basis prediction; moments in
the displacement coordinates of the map. Every field is immutable; the
builders are eager (ranks and reports are concrete). Not certified: that
the optimum is global (``converged`` is the optimiser's own flag),
feasibility of the fitted moments (the report lists necessary conditions
only), and the truth of any declared discrepancy.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..assumptions import Parameters
from ..errors import ErrorTerm, Provenance
from . import _result_core as core
from ._result_core import MAX_FISHER_PARAMS, TOL

LABEL = "extra eq: finite fit model"
METHODS = ("linear", "bfgs", "nodal")


def _static_int(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


class FitResult(eqx.Module):
    """Fitted parameters with their Fisher information, diagnostics and prediction.

    Fields in the frozen order: ``theta`` (``Parameters``), ``moments``
    (``JointMoments``), ``amplitude`` (scalar), ``fisher`` ``(n, n)`` or
    ``None``, ``covariance`` ``(n, n)`` or ``None``, static ``rank``,
    ``chi2``, static ``dof``, static ``converged``, static ``n_iter``,
    ``bias_bound`` (``ErrorTerm`` over the fit coordinates, ``None`` for a
    nodal fit), ``identifiability``,
    ``feasibility``, ``prediction`` (each or ``None``), static
    ``provenance``. Trailing additions: ``method``, ``labels`` and
    ``coordinates`` (static; the meaning of the Fisher axes),
    ``parameter_map`` (the map that produced ``moments``), ``z`` (the
    optimiser's coordinates), ``fun`` (the minimised objective), ``weights``
    (nodal simplex weights) and ``discretisation`` (nodal ``ErrorTerm``).
    Units and what is not certified (global optimum, feasibility, declared
    discrepancies): see the module docstring.
    """

    theta: Parameters
    moments: object
    amplitude: jax.Array
    fisher: jax.Array | None
    covariance: jax.Array | None
    rank: int = eqx.field(static=True)
    chi2: jax.Array
    dof: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    n_iter: int = eqx.field(static=True)
    bias_bound: ErrorTerm | None
    identifiability: object | None
    feasibility: object | None
    prediction: object | None
    provenance: Provenance = eqx.field(static=True)
    method: str = eqx.field(static=True, default="")
    labels: tuple[str, ...] = eqx.field(static=True, default=())
    coordinates: str = eqx.field(static=True, default="")
    parameter_map: object | None = None
    z: jax.Array | None = None
    fun: jax.Array | None = None
    weights: jax.Array | None = None
    discretisation: ErrorTerm | None = None
    LABEL: ClassVar[str] = LABEL

    def __check_init__(self):
        if not isinstance(self.theta, Parameters):
            raise ValueError("theta must be a Parameters")
        _static_int(self.rank, "rank")
        if isinstance(self.dof, bool) or not isinstance(self.dof, (int, np.integer)):
            raise ValueError("dof must be an int")
        if not isinstance(self.converged, bool):
            raise ValueError("converged must be a bool")
        _static_int(self.n_iter, "n_iter")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")
        if self.bias_bound is not None and not isinstance(self.bias_bound, ErrorTerm):
            raise ValueError("bias_bound must be an ErrorTerm or None")
        if self.discretisation is not None and not isinstance(
            self.discretisation, ErrorTerm
        ):
            raise ValueError("discretisation must be an ErrorTerm or None")
        for name in ("fisher", "covariance"):
            value = getattr(self, name)
            if value is None:
                continue
            if value.ndim != 2 or value.shape[0] != value.shape[1]:
                raise ValueError(f"{name} must be a square matrix")
            if self.labels and value.shape[0] != len(self.labels):
                raise ValueError(f"{name} must have one row per label")
        if self.method and self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")

    @property
    def n_params(self) -> int:
        return len(self.labels)

    def standard_errors(self) -> jax.Array | None:
        """``sqrt(diag(covariance))`` in the fit coordinates, or ``None``."""
        if self.covariance is None:
            return None
        return jnp.sqrt(jnp.maximum(jnp.diag(self.covariance), 0.0))

    def to_dict(self) -> dict:
        """Strict-JSON summary (eager; tuples become lists, non-finite floats
        such as fitted-hyper placeholders become ``None``)."""
        pm = self.parameter_map
        parameters = None
        if pm is not None:
            values = np.asarray(pm.flatten(self.theta), dtype=float).tolist()
            parameters = dict(zip(pm.labels(), values))
        return core.jsonable(self._raw_dict(parameters))

    def _raw_dict(self, parameters) -> dict:
        return {
            "label": self.LABEL,
            "method": self.method,
            "coordinates": self.coordinates,
            "labels": list(self.labels),
            "theta": core.theta_json(self.theta),
            "parameters": parameters,
            "moments": self.moments.to_dict(),
            "amplitude": float(np.asarray(self.amplitude)),
            "fisher": None if self.fisher is None else np.asarray(self.fisher).tolist(),
            "covariance": (
                None
                if self.covariance is None
                else np.asarray(self.covariance).tolist()
            ),
            "rank": self.rank,
            "chi2": float(np.asarray(self.chi2)),
            "dof": self.dof,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "fun": None if self.fun is None else float(np.asarray(self.fun)),
            "bias_bound": (
                None if self.bias_bound is None else self.bias_bound.to_dict()
            ),
            "identifiability": (
                None if self.identifiability is None else self.identifiability.to_dict()
            ),
            "feasibility": (
                None if self.feasibility is None else self.feasibility.to_dict()
            ),
            "prediction": (
                None if self.prediction is None else self.prediction.to_dict()
            ),
            "weights": (
                None if self.weights is None else np.asarray(self.weights).tolist()
            ),
            "discretisation": (
                None if self.discretisation is None else self.discretisation.to_dict()
            ),
            "provenance": self.provenance.to_dict(),
        }


__all__ = ["FitResult", "LABEL", "METHODS", "MAX_FISHER_PARAMS", "TOL"]
