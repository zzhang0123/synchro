"""Kernel models of the finite joint response (``syncmoments.model.kernels``).

LABEL: ``eq: smooth channel kernel`` and ``eq: channel derivative
coefficients`` (the ``KernelModel`` contract: channel-integrated natural-basis
modes and their Legendre projections, which ``build_basis`` differentiates),
``eq: legendre spectral response`` (the ``(2l+1)(2k+1)/4`` projection weights),
``eq: directional continuum`` and ``extra eq: channel basis integration``
(:class:`ContinuumKernel`, in ``_continuum.py``). :class:`PolynomialTestKernel`
(``_polynomial.py``) is an ``[extension, test oracle]`` with no manuscript
counterpart. :class:`syncmoments.model.harmonic.HarmonicKernel` is the Dirac-line
reference kernel. Shared algebra lives in ``_kernel_helpers.py``.

A kernel returns :class:`Modes` at a scalar ``(gamma, B, mu, eta)`` point:
``I``, ``V`` of shape ``(n_ch,)`` in erg/s/sr per electron (band-integrated
against the dimensionless ``unit_peak`` response; per Hz for
``unit_integral``), and ``P`` of shape ``(n_weights, n_ch)`` complex, the
natural-basis ``Q`` modes multiplied by the phase weights ``w_b(tau_m)`` of
``syncmoments.model.phase`` (``tau = 2 (C_SI_M/nu)^2`` in m^2) and excluding the
sky factor ``exp(2 i phi)``. :class:`ProjectedModes` carries the Legendre
projections ``(2l+1)(2k+1)/4 int int (.) P_l(mu) P_k(eta) dmu deta`` for the
``(l, k)`` pairs of ``MomentIndex.pairs``. Units: Gauss, Hz, rad/m^2.

``phase`` may be ``None`` (one unit weight, no Faraday phase) and
``depth_ref``/``s_depth`` default to ``0``/``1``; these keyword defaults are
additive to the frozen interface.

Not certified here: the physical adequacy of any kernel (``physical_error``
is ``unbounded`` unless the caller declares ``E_phys``), quadrature error of
the angular projections (the refined-rule ``numerical`` estimate lives in
``basis.py``), and Bessel/``F``/``G`` roundoff beyond the finite checks of the
package tests.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from ..constants import C_CGS, E_ESU, M_E
from ._continuum import ContinuumKernel
from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    gyrofrequency_hz,
    legendre_norm,
    legendre_table,
    phase_coordinate,
    phase_weights,
    resolve_index,
)
from ._polynomial import PolynomialTestKernel
from .errors import ErrorTerm


@runtime_checkable
class KernelModel(Protocol):
    """Contract every kernel implements (see the module docstring for units).

    ``channel_modes`` returns :class:`Modes` at one scalar point;
    ``angular_projection`` returns :class:`ProjectedModes` for the pairs of
    the index; ``physical_error`` and ``truncation_error`` return
    ``ErrorTerm`` ``(n_ch, 4)`` per electron. Each kernel states its own
    assumptions in ``required_closures`` and ``LABEL``; what it does not
    certify is in its docstring (``physical_error`` is unbounded unless
    declared).
    """

    name: str
    LABEL: tuple[str, ...]
    components: tuple[str, ...]
    required_closures: tuple[str, ...]

    def channel_modes(
        self, channels, gamma, B, mu, eta, *, phase, depth_ref, s_depth
    ) -> Modes: ...

    def angular_projection(
        self,
        channels,
        gamma,
        B,
        *,
        phase,
        depth_ref,
        s_depth,
        truncation,
        quadrature,
    ) -> ProjectedModes: ...

    def physical_error(self, channels) -> ErrorTerm: ...

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ) -> ErrorTerm: ...

    def describe(self) -> tuple: ...


def _concrete(value, name):
    try:
        return float(np.asarray(value))
    except Exception as exc:  # traced or non-numeric
        raise ValueError(f"{name} must be a concrete number, got {value!r}") from exc


def required_m_max(support, channels) -> int:
    """``ceil((1 + beta_max) nu_hi,max / nu_B,min)``: the largest harmonic that
    can meet any channel on the declared support (the manuscript's
    ``harmonic_cutoff``).

    ``support.gamma = (lo, hi)`` and ``support.B = (lo, hi)`` (Gauss) must be
    concrete; ``nu_B,min = e B_min / (2 pi gamma_max m_e c)``. Eager only.
    Lines with ``m`` above this value have zero channel weight because
    ``nu_m = m nu_B / D`` with ``D >= 1 - beta``; the result is a support
    statement, not a tail estimate. ``ceil`` can be conservative by one: the
    returned harmonic itself need not meet a channel. Units: Hz for
    frequencies, Gauss for ``B``; the result is a dimensionless integer.
    Assumes the population lies inside ``support`` (checked against concrete
    samples by ``HarmonicKernel.truncation_error``, ``_support_check``); not
    certified: the support declaration itself when no samples are supplied.
    """
    gamma_max = _concrete(support.gamma[1], "support.gamma[1]")
    B_min = _concrete(support.B[0], "support.B[0]")
    if not gamma_max > 1.0 or not B_min > 0.0:
        raise ValueError("required_m_max needs gamma_max > 1 and B_min > 0")
    supp = np.asarray(channels.support, dtype=float)
    if supp.ndim != 2 or supp.shape[1] != 2 or not np.all(np.isfinite(supp)):
        raise ValueError("channels.support must be a finite (n_ch, 2) array")
    beta_max = math.sqrt(1.0 - 1.0 / gamma_max**2)
    nu_B_min = E_ESU * B_min / (2.0 * math.pi * gamma_max * M_E * C_CGS)
    return int(math.ceil((1.0 + beta_max) * float(np.max(supp[:, 1])) / nu_B_min))


__all__ = [
    "Modes",
    "ProjectedModes",
    "KernelModel",
    "ContinuumKernel",
    "PolynomialTestKernel",
    "required_m_max",
    "legendre_table",
    "legendre_norm",
    "gather_pairs",
    "phase_coordinate",
    "phase_weights",
    "gyrofrequency_hz",
    "check_scalar_point",
    "resolve_index",
]
