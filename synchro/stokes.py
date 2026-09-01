"""Exact single-particle synchrotron Stokes parameters (Schott decomposition).

These are the ``basic element`` of the model: the power emitted at the n-th
harmonic of the gyrofrequency, per unit solid angle, decomposed into linear
polarisations parallel/perpendicular to the sky-projection of **B**, plus the
circular component.

Expressions (corrected absolute normalisation, cf. Legg & Westfold 1968):

    dP_n^|| / dOmega = C_n * [(cosT - b_par)^2 / sin^2 T]  * J_n^2(x_n)
    dP_n^_| / dOmega = C_n * [b_perp^2]                   * J_n'^2(x_n)

    C_n = e^2 w_B^2 n^2 / (2 pi c) * 1 / D^3,
    D   = 1 - b_par cosT  (Doppler factor),
    x_n = n b_perp sinT / D,
    b_par  = beta cos(alpha),  b_perp = beta sin(alpha).

Stokes:  I = || + _|,  Q = || - _|,  V = 2 Im(E_x E_y^*) (see code for sign).

The frequency of the n-th harmonic is w_n = n w_B / D.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .bessel import bessel_jn_and_prime

# CGS constants
E_ESU = 4.80320427e-10  # elementary charge [esu]
C_CGS = 2.99792458e10  # speed of light [cm/s]
M_E = 9.1093837e-28  # electron mass [g]


def beta_of_gamma(gamma):
    return jnp.sqrt(1.0 - 1.0 / gamma**2)


def doppler(beta_par, theta):
    return 1.0 - beta_par * jnp.cos(theta)


def _harmonic(n, gamma, alpha, theta):
    """Dimensionless Stokes (I, Q, V) at harmonic ``n``.

    Normalised by e^2 w_B^2 / (2 pi c), i.e. the returned values equal the
    physical dP/dOmega divided by that prefactor. ``n`` is static.
    """
    beta = beta_of_gamma(gamma)
    b_par = beta * jnp.cos(alpha)
    b_perp = beta * jnp.sin(alpha)
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    D = doppler(b_par, theta)
    x = n * b_perp * st / D

    jn, jnp_ = bessel_jn_and_prime(n, x)

    g_par = (ct - b_par) ** 2 / st**2
    # dP/dOmega divided by [e^2 w_B^2 / (2 pi c)]
    P_par = n**2 * g_par * jn**2 / D**3
    P_perp = n**2 * b_perp**2 * jnp_**2 / D**3

    I = P_par + P_perp
    Q = P_par - P_perp
    # V is defined up to an overall handedness convention: V = 2 Im(E_x E_y*)
    # (IAU) yields the opposite sign for an electron.  Magnitude and parameter
    # dependence are unambiguous; the full-polarisation identity I^2 = Q^2 + V^2
    # is sign-independent.
    V = 2.0 * n**2 * b_perp * (ct - b_par) / (st * D**3) * jn * jnp_
    return I, Q, V


def stokes_harmonic(n, gamma, alpha, theta, B=None):
    """Stokes (I, Q, V) power per unit solid angle at harmonic ``n``.

    If ``B`` (Gauss) is given, returns physical ``dP_n/dOmega`` [erg/s/sr];
    otherwise returns the value normalised by e^2 w_B^2/(2 pi c) (which is
    B-independent and convenient for derivative spectra).
    """
    I, Q, V = _harmonic(n, gamma, alpha, theta)
    if B is None:
        return I, Q, V
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2.0 * jnp.pi * C_CGS)
    return pref * I, pref * Q, pref * V


def larmor_power(gamma, alpha, B):
    """Total synchrotron power (all harmonics, all solid angles), erg/s."""
    beta = beta_of_gamma(gamma)
    b_perp = beta * jnp.sin(alpha)
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    return (2.0 / 3.0) * (E_ESU**2 / C_CGS) * gamma**4 * wB**2 * b_perp**2
