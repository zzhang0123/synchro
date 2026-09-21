"""synchro: moment expansion of synchrotron emission (JAX + Equinox).

Note: importing this package enables ``jax_enable_x64`` (float64) as a global
side effect, because the Bessel quadrature and the likelihood-style sums need
float64 for accuracy. Do this once at the top of any driver script, before any
array is created.

Modules
-------
bessel           : differentiable J_n, J_n', K_nu (integral representations)
stokes           : exact Schott harmonic Stokes (I, Q, V) with correct
                   normalisation
derivatives      : derivative spectra dS/dp, d^2S/dp^2 via autodiff
expansion         : eqx.Module for quadratic Taylor averaging (mean/covariance)
cumulants        : finite Bell moment contractions from supplied cumulants
ultrarel         : ultra-relativistic F(x), G(x) fast path
rm               : rotation measure (RM) constants + Burn depolarisation
transfer         : Mueller matrix + numerical slab/LOS matrix exponentials
solutions        : analytic limiting solutions (thin/rotation/self-absorbed)
kirchhoff        : emissivity/absorption moment expansion + Kirchhoff closure
conversion       : cold-plasma Faraday rotation/conversion coefficients + moments
magnus           : Magnus expansion (Omega1, Omega2) of the LOS transfer
los_moments      : explicit reduced-units and physical-CGS moment-driven slabs
sed              : frequency-domain SED (spectral index, curvature, abs. emissivity)
"""

from __future__ import annotations

import jax

# Enable float64 for accuracy of Bessel quadrature and likelihood-style sums.
jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402
    bessel,
    conversion,
    cumulants,
    derivatives,
    kirchhoff,
    los_moments,
    magnus,
    expansion,
    rm,
    sed,
    solutions,
    stokes,
    transfer,
    ultrarel,
)  # noqa: E402,F401
from .stokes import larmor_power, stokes_harmonic  # noqa: E402,F401
from .derivatives import derivative_spectra  # noqa: E402,F401
from .expansion import (  # noqa: E402,F401
    CumulantExpansion,
    QuadraticTaylorExpansion,
    build_expansion,
)
from .ultrarel import F, G  # noqa: E402,F401

__all__ = [
    "bessel",
    "stokes",
    "derivatives",
    "expansion",
    "cumulants",
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
    "CumulantExpansion",
    "QuadraticTaylorExpansion",
    "build_expansion",
    "F",
    "G",
]
