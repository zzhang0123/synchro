"""Leading cold-plasma coefficients in CGS and the paper's Stokes convention.

Inputs: frequency Hz, thermal density cm^-3, magnetic field Gauss. The
``mueller_*`` functions return signed Stokes rates cm^-1 for dS/ds=eps-K S,
Q=parallel-minus-perpendicular to projected B, and V=-2 Im(E_parallel E_perp*).
In that natural basis conversion is rQ (U--V mixing); rU=0. At sky azimuth
phi its components are rQ*cos(2phi), rQ*sin(2phi).

``rotation_coefficient`` retains the position-angle rate for compatibility;
``conversion_coefficient`` returns rQ itself. There is no additional factor
of two on the latter. These leading high-frequency cold-dielectric terms
require plasma and cyclotron frequencies small compared with nu. Hot and
non-thermal corrections are distribution dependent and are not bounded here.
"""

from __future__ import annotations

import numpy as np

from .rm import E_ESU, M_E, C_CGS

# rho_V = e^3 n_e B_par / (2 pi m_e^2 c^2 nu^2)
C_ROT = E_ESU**3 / (2.0 * np.pi * M_E**2 * C_CGS**2)
# rho_Q = e^4 n_e B_perp^2 / (4 pi^2 m_e^3 c^3 nu^3)
C_CONV = E_ESU**4 / (4.0 * np.pi**2 * M_E**3 * C_CGS**3)


def rotation_coefficient(nu, n_e, B_par):
    """Physical Faraday rotation rate rho_V [rad/cm] (position angle)."""
    return C_ROT * n_e * B_par / nu**2


def conversion_coefficient(nu, n_e, B_perp):
    """Natural-basis Stokes conversion rate rQ [cm^-1], signed."""
    return -C_CONV * n_e * B_perp**2 / nu**3


def conversion_rotation_ratio(nu, B_perp, B_par):
    """Signed rQ/rV, independent of n_e; undefined when B_par=0."""
    return -(E_ESU * B_perp**2) / (4.0 * np.pi * M_E * C_CGS * nu * B_par)


def mueller_rotation(nu, n_e, B_par):
    """Stokes-space rotation coefficient rV = 2 rho_V for syncmoments.transfer."""
    return 2.0 * rotation_coefficient(nu, n_e, B_par)


def mueller_conversion(nu, n_e, B_perp):
    """Natural-basis Stokes conversion coefficient rQ for syncmoments.transfer."""
    return conversion_coefficient(nu, n_e, B_perp)
