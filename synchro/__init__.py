"""PySync: moment expansion of synchrotron emission (JAX + Equinox).

Modules
-------
bessel           : differentiable J_n, J_n', K_nu (integral representations)
stokes           : exact Schott harmonic Stokes (I, Q, V) with correct
                   normalisation
derivatives      : derivative spectra dS/dp, d^2S/dp^2 via autodiff
moment_expansion : eqx.Module realising the moment/cumulant expansion
ultrarel         : ultra-relativistic F(x), G(x) fast path
rm               : rotation measure (RM) constants + Burn depolarisation
transfer         : Mueller matrix + exact slab/LOS polarised transfer
solutions        : analytic limiting solutions (thin/rotation/self-absorbed)
kirchhoff        : emissivity/absorption moment expansion + Kirchhoff closure
conversion       : cold-plasma Faraday rotation/conversion coefficients + moments
magnus           : Magnus expansion (Omega1, Omega2) of the LOS transfer
los_moments      : moment-driven LOS (coefficients from moments -> transfer)
sed              : frequency-domain SED (spectral index, curvature, abs. emissivity)
"""

from __future__ import annotations

import jax

# Enable float64 for accuracy of Bessel quadrature and likelihood-style sums.
jax.config.update("jax_enable_x64", True)

from . import bessel, conversion, derivatives, kirchhoff, los_moments, magnus, moment_expansion, rm, sed, solutions, stokes, transfer, ultrarel  # noqa: E402,F401
from .stokes import larmor_power, stokes_harmonic  # noqa: E402,F401
from .derivatives import derivative_spectra  # noqa: E402,F401
from .moment_expansion import MomentExpansion, build_expansion  # noqa: E402,F401
from .ultrarel import F, G  # noqa: E402,F401

__all__ = [
    "bessel",
    "stokes",
    "derivatives",
    "moment_expansion",
    "ultrarel",
    "rm",
    "transfer",
    "solutions",
    "kirchhoff",
    "conversion",
    "magnus",
    "los_moments",
    "sed",
    "stokes_harmonic",
    "larmor_power",
    "derivative_spectra",
    "MomentExpansion",
    "build_expansion",
    "F",
    "G",
]
