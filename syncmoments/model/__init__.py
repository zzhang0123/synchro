"""Finite joint response and spectral fits from joint population statistics.

This layer implements the manuscript's Section 4 construction: Legendre
projection in the angular cosines, Taylor expansion in scaled energy, field and
Faraday-depth displacements, smooth-channel evaluation of the Dirac-line kernel
(main.tex ``eq: smooth channel kernel``, ``eq: channel derivative coefficients``),
the finite joint response (``eq: finite joint response``) and its error budget
(``eq: channel error budget``), plus a fitting layer (extended discussion
``eq: finite fit model``).

Objects
-------
Truncation, MomentIndex      index layout (h; l,k; r,s,b), flattening to m
lower_set_margin             remainder margin of a capped (lower-set) truncation
Channels                     smooth compact channel responses R_j(nu)
TaylorPhase, GaussianScreen, CumulantScreen, EmpiricalScreen,
LaplaceScreen, GammaScreen                                      depth phase routes
HarmonicKernel, ContinuumKernel, PolynomialTestKernel           kernel models
Reference, Support, PopulationSamples, JointMoments             statistics
ParameterMap and named constructors                             declared assumptions
build_basis -> SpectralBasis                                    the cacheable response
predict, direct_channel_average -> Prediction                   forward modelling
fit.*                                                           spectral fits

No independence, isotropy or Gaussian closure is assumed unless declared;
every approximation carries an ErrorTerm whose kind is bound, estimate,
measured, unbounded or not_applicable. Nothing here certifies a physical
model discrepancy or an excluded population tail.
"""

from __future__ import annotations

from .index import Entry, MomentIndex, Truncation, lower_set_margin
from .errors import AssumptionRecord, ErrorBudget, ErrorTerm, Provenance
from .channels import Channels
from .phase import (
    CumulantScreen,
    EmpiricalScreen,
    GaussianScreen,
    TaylorPhase,
    phase_coordinate,
)
from .screens import GammaScreen, LaplaceScreen
from .moments import JointMoments, PopulationSamples, Reference, Support
from .bounds import (
    RemainderInputs,
    azimuth_factorisation_bound,
    depth_error_bound,
    screen_factorisation_bound,
)
from .assumptions import (
    Closure,
    Factorisation,
    ParameterMap,
    Parameters,
    azimuth_separable,
    field_independent,
    field_reversal_symmetric,
    fully_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
    pitch_symmetric,
)
from .kernels import (
    ContinuumKernel,
    Modes,
    PolynomialTestKernel,
    ProjectedModes,
    required_m_max,
)
from .harmonic import HarmonicKernel
from .basis import KernelTerms, SpectralBasis, basis_convergence, build_basis
from .predict import Prediction, direct_channel_average, predict
from . import adapters, fit

__all__ = [
    "Entry",
    "MomentIndex",
    "Truncation",
    "lower_set_margin",
    "AssumptionRecord",
    "ErrorBudget",
    "ErrorTerm",
    "Provenance",
    "Channels",
    "CumulantScreen",
    "EmpiricalScreen",
    "GaussianScreen",
    "GammaScreen",
    "LaplaceScreen",
    "TaylorPhase",
    "phase_coordinate",
    "JointMoments",
    "PopulationSamples",
    "Reference",
    "Support",
    "RemainderInputs",
    "azimuth_factorisation_bound",
    "depth_error_bound",
    "screen_factorisation_bound",
    "Closure",
    "Factorisation",
    "ParameterMap",
    "Parameters",
    "azimuth_separable",
    "field_independent",
    "field_reversal_symmetric",
    "fully_independent",
    "gaussian_screen",
    "independent_screen",
    "isotropic_pitch",
    "no_assumption",
    "nodal",
    "pitch_symmetric",
    "ContinuumKernel",
    "Modes",
    "PolynomialTestKernel",
    "ProjectedModes",
    "required_m_max",
    "HarmonicKernel",
    "KernelTerms",
    "SpectralBasis",
    "basis_convergence",
    "build_basis",
    "Prediction",
    "direct_channel_average",
    "predict",
    "adapters",
    "fit",
]
