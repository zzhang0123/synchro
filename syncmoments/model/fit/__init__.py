"""Spectral fits of channel Stokes data (``syncmoments.model.fit``).

Implements the finite fitting model of the extended discussion
(``eq: finite fit model``): data ``d = A R C m + delta + eta`` with a supplied
observing response ``R``, noise covariance and optional discrepancy envelope.

* ``StokesData``: data, noise, mask, observing response and discrepancy.
* ``fit_linear``: closed-form weighted least squares for affine parameter maps.
* ``LogDensity``, ``Transform``, ``fit_bfgs``: nonlinear maps and external samplers.
* ``fit_nodal``: nonnegative weights on fixed nodes (always-feasible moments).
* ``identifiability``, ``feasibility_checks``: SVD null directions and necessary
  moment conditions.
* ``FitResult``: the self-describing result.
* ``reduce_response``, ``fit_combinations`` ``[extension]``: exact pre-fit
  grouping ``C = H T`` and the identifiable combinations of a linear fit in a
  declared coefficient metric (``CombinationFit``; analytic redundancy,
  numerical rank and practical recoverability reported separately).

A small residual is not a remainder certificate, feasibility checks are
necessary conditions only, and an error envelope is not a noise distribution.
"""

from __future__ import annotations

from .observation import StokesData
from .linear import fisher, fit_linear
from .nonlinear import LogDensity, Transform, fit_bfgs, fit_nodal
from .diagnostics import (
    FeasibilityReport,
    IdentifiabilityReport,
    feasibility_checks,
    identifiability,
)
from .result import FitResult
from .layout import CoefficientLayout, coefficient_bounds
from .reduction import (
    LinearRelation,
    ResponseReduction,
    find_column_relations,
    reduce_response,
)
from .combinations import fit_combinations
from .combination_result import CombinationFit, ObservableSummary, TruthComparison

__all__ = [
    "StokesData",
    "fit_linear",
    "fisher",
    "LogDensity",
    "Transform",
    "fit_bfgs",
    "fit_nodal",
    "FeasibilityReport",
    "IdentifiabilityReport",
    "feasibility_checks",
    "identifiability",
    "FitResult",
    "CoefficientLayout",
    "coefficient_bounds",
    "LinearRelation",
    "ResponseReduction",
    "reduce_response",
    "find_column_relations",
    "fit_combinations",
    "CombinationFit",
    "ObservableSummary",
    "TruthComparison",
]
