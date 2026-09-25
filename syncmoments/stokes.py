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

Stokes:  I = || + _|,  Q = || - _|,  V = -2 Im(E_parallel E_perp^*) for the exp(+i omega t) Fourier convention.

The frequency of the n-th harmonic is w_n = n w_B / D.
"""

from __future__ import annotations

import jax.numpy as jnp

from .bessel import bessel_jn, bessel_jn_and_prime

# Imported aliases preserve the public constant names used by older callers.
from .constants import C_CGS, E_ESU, M_E


def beta_of_gamma(gamma):
    return jnp.sqrt(1.0 - 1.0 / gamma**2)


def doppler(beta_par, theta):
    return 1.0 - beta_par * jnp.cos(theta)


def _harmonic(n, gamma, alpha, theta, *, n_nodes=None):
    """Dimensionless Stokes (I, Q, V) at harmonic ``n``.

    Normalised by e^2 w_B^2 / (2 pi c), i.e. the returned values equal the
    physical dP/dOmega divided by that prefactor. ``n`` is static unless a
    static ``n_nodes`` fixes the Bessel resolution (see ``bessel_jn_and_prime``).
    """
    beta = beta_of_gamma(gamma)
    b_par = beta * jnp.cos(alpha)
    b_perp = beta * jnp.sin(alpha)
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    D = doppler(b_par, theta)
    x = n * b_perp * st / D

    _, jnp_ = bessel_jn_and_prime(n, x, n_nodes=n_nodes)

    # J_n(x)/sin(theta) = beta_perp [J_(n-1)(x)+J_(n+1)(x)]/(2D).
    # This exact recurrence removes the removable viewing-axis singularity.
    neighbours = bessel_jn(n - 1, x, n_nodes=n_nodes) + bessel_jn(
        n + 1, x, n_nodes=n_nodes
    )
    amplitude_par = (ct - b_par) * b_perp * neighbours / (2.0 * D)
    amplitude_perp = b_perp * jnp_
    factor = n**2 / D**3
    I = factor * (amplitude_par**2 + amplitude_perp**2)
    Q = factor * (amplitude_par**2 - amplitude_perp**2)
    V = 2.0 * factor * amplitude_par * amplitude_perp
    return I, Q, V


def stokes_harmonic(n, gamma, alpha, theta, B=None, *, n_nodes=None):
    """Stokes (I, Q, V) power per unit solid angle at harmonic ``n``.

    If ``B`` (Gauss) is given, returns physical ``dP_n/dOmega`` [erg/s/sr];
    otherwise returns the value normalised by e^2 w_B^2/(2 pi c). That
    normalisation depends on gamma; derivatives of this dimensionless output
    are not physical fixed-B derivatives. Harmonic n >= 1, gamma >= 1,
    alpha/theta in [0, pi]. Angular endpoints use their analytic limits.
    Numerical quadrature errors remain separate from the vacuum helical-orbit
    reference physics. Differentiation is intended for interior parameters.

    ``n_nodes`` (static int, keyword-only) passes through to the Bessel
    quadrature: ``None`` keeps the automatic count (identical output to the
    historical behaviour); a static count lets ``n`` be a traced float order
    under ``jax.jit``/``vmap`` (unresolved orders return NaN, see
    ``syncmoments.bessel``). The count is a resolution choice, not an error bound.
    """
    I, Q, V = _harmonic(n, gamma, alpha, theta, n_nodes=n_nodes)
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
