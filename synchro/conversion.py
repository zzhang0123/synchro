"""Faraday rotation + conversion coefficients (cold plasma) and their moments.

Physical (position-angle / conversion-angle) rates, CGS:

    rho_V = e^3 n_e B_par / (2 pi m_e^2 c^2 nu^2)     [rotation, ~ nu^-2]
    rho_Q = e^4 n_e B_perp^2 / (4 pi^2 m_e^3 c^3 nu^3) [conversion, ~ nu^-3]

In the Mueller matrix (synchro.transfer) the Stokes-space rotation rates are
TWICE the physical rates (rV = 2 rho_V, rU = 2 rho_Q), because Stokes Q,U,V
rotate by twice the physical polarisation-ellipse angle.  The rotation measure
RM = int rho_V ds / lam^2 reproduces the standard 0.812 rad/m^2 constant.

The conversion/rotation ratio is convention-independent:

    rho_Q / rho_V = (nu_B / nu) (sin^2 theta / cos theta) / (2 pi) ... see
    conversion_rotation_ratio().

Cold-plasma, quasi-longitudinal limit.  Relativistic (thermal) corrections are
an order-unity function of the electron temperature (cf. Huang & Shcherbakov
2011) and are deferred.
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
    """Physical Faraday conversion rate rho_Q [rad/cm] (conversion angle)."""
    return C_CONV * n_e * B_perp**2 / nu**3


def conversion_rotation_ratio(nu, B_perp, B_par):
    """rho_Q / rho_V  (dimensionless; independent of n_e)."""
    return (E_ESU * B_perp**2) / (2.0 * np.pi * M_E * C_CGS * nu * B_par)


def mueller_rotation(nu, n_e, B_par):
    """Stokes-space rotation coefficient rV = 2 rho_V for synchro.transfer."""
    return 2.0 * rotation_coefficient(nu, n_e, B_par)


def mueller_conversion(nu, n_e, B_perp):
    """Stokes-space conversion coefficient rQ = 2 rho_Q for synchro.transfer."""
    return 2.0 * conversion_coefficient(nu, n_e, B_perp)
