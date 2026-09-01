"""Kirchhoff: emissivity and self-absorption from a distribution N(gamma).

In the ultra-relativistic regime the single-particle power per unit frequency
is P(nu, gamma) ~ F(nu / nu_c) with nu_c ~ gamma^2, and the emissivity (per
unit solid angle) and absorption coefficient are

    j_nu   =  int N(gamma) P(nu, gamma) dgamma
    alpha  =  (1/nu^2) int g(gamma) P(nu, gamma) dgamma ,
    g(gamma) = - gamma^2 d/dgamma [ N(gamma) / gamma^2 ]          (Kirchhoff)

up to gamma-independent prefactors (which cancel in the source function
S_nu = j_nu / alpha_nu).  The absorption therefore probes the *derivative-
weighted* distribution g rather than N itself; for a power law N ~ gamma^-p,
g = (p+2) N / gamma, giving the standard alpha ~ nu^-(p+4)/2 and
S_nu ~ nu^(5/2).

Moment expansion: expanding P(nu, gamma) around gamma0 gives j_nu as moments
of N and alpha_nu as moments of g, with the SAME derivative spectra of P
(computed by autodiff).  This is the transfer-side analogue of the emissivity
derivative spectra of the main paper.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .ultrarel import F, G

_F = jax.jit(F)
_G = jax.jit(G)


def single_particle_P(nu, gamma, nu_c_ref):
    """P(nu, gamma) = F(nu / (nu_c_ref gamma^2)), vectorised over ``gamma``."""
    x = nu / (nu_c_ref * gamma**2)
    return _F(x)


def emissivity(N, gamma_grid, nu, nu_c_ref):
    """j_nu = int N(gamma) P(nu, gamma) dgamma (relative units)."""
    P = np.asarray(single_particle_P(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(N * P, gamma_grid)


def derivative_weighted(N, gamma_grid):
    """g(gamma) = - gamma^2 d/dgamma [ N / gamma^2 ]  (Kirchhoff weight)."""
    q = N / gamma_grid**2
    dq = np.gradient(q, gamma_grid)
    return -gamma_grid**2 * dq


def absorption(N, gamma_grid, nu, nu_c_ref):
    """alpha_nu = (1/nu^2) int g(gamma) P(nu, gamma) dgamma (relative)."""
    g = derivative_weighted(N, gamma_grid)
    P = np.asarray(single_particle_P(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(g * P, gamma_grid) / nu**2


def source_function(N, gamma_grid, nu, nu_c_ref):
    """S_nu = j_nu / alpha_nu."""
    return emissivity(N, gamma_grid, nu, nu_c_ref) / absorption(N, gamma_grid, nu, nu_c_ref)


def P_derivatives(nu, gamma0, nu_c_ref):
    """P, dP/dgamma, d^2P/dgamma^2 of F(nu/(nu_c_ref gamma^2)) at gamma0."""
    f = lambda g: _F(nu / (nu_c_ref * g**2))  # noqa: E731
    P0 = f(gamma0)
    P1 = jax.grad(f)(gamma0)
    P2 = jax.grad(jax.grad(f))(gamma0)
    return P0, P1, P2


def moment_expansion_emissivity(N, gamma_grid, gamma0, P0, P1, P2):
    """2nd-order moment expansion of j_nu about gamma0."""
    dg = gamma_grid - gamma0
    M0 = np.trapezoid(N, gamma_grid)
    M1 = np.trapezoid(N * dg, gamma_grid)
    M2 = np.trapezoid(N * dg**2, gamma_grid)
    return P0 * M0 + P1 * M1 + 0.5 * P2 * M2


def moment_expansion_absorption(g, gamma_grid, gamma0, P0, P1, P2, nu):
    """2nd-order moment expansion of alpha_nu about gamma0 (weight g)."""
    dg = gamma_grid - gamma0
    D0 = np.trapezoid(g, gamma_grid)
    D1 = np.trapezoid(g * dg, gamma_grid)
    D2 = np.trapezoid(g * dg**2, gamma_grid)
    return (P0 * D0 + P1 * D1 + 0.5 * P2 * D2) / nu**2


# ---------------------------------------------------------------------------
# Kirchhoff closure in moment form
#
# The absorption weight g = 2N/gamma - N' makes the absorption moments D_k
# expressible through the emissivity moments M_k = int N (gamma-gamma0)^k dgamma
# plus the *inverse* moment <1/gamma> = int N/gamma dgamma:
#
#     D0 = 2<1/g>,  D1 = 3 M0 - 2 g0 <1/g>,
#     D2 = 4 M1 - 2 g0 M0 + 2 g0^2 <1/g>.                        (EXACT)
#
# The inverse moment is not determined by the integer moments; for a narrow
# distribution it is given by the geometric series
#
#     <1/gamma> = M0/g0 - M1/g0^2 + M2/g0^3 + ...                (approx.)
# ---------------------------------------------------------------------------


def inverse_gamma_moment(M0, M1, M2, gamma0):
    """<1/gamma> = int N/gamma dgamma via the geometric-series expansion.

    Valid for |gamma - gamma0| < gamma0; truncated at the second moment.
    """
    return M0 / gamma0 - M1 / gamma0**2 + M2 / gamma0**3


def absorption_moments(M0, M1, M2, gamma0, inv_gamma):
    """Exact Kirchhoff closure: absorption moments D_k from M_k and <1/gamma>."""
    D0 = 2.0 * inv_gamma
    D1 = 3.0 * M0 - 2.0 * gamma0 * inv_gamma
    D2 = 4.0 * M1 - 2.0 * gamma0 * M0 + 2.0 * gamma0**2 * inv_gamma
    return D0, D1, D2


def emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2):
    """j_nu from the emissivity moments M_k (2nd order, no N needed)."""
    P0, P1, P2 = P_derivatives(nu, gamma0, nu_c_ref)
    return P0 * M0 + P1 * M1 + 0.5 * P2 * M2


def absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma=None):
    """alpha_nu from the emissivity moments via Kirchhoff closure.

    ``inv_gamma`` = <1/gamma>; if None it is approximated by the geometric
    series (narrow-distribution limit).
    """
    if inv_gamma is None:
        inv_gamma = inverse_gamma_moment(M0, M1, M2, gamma0)
    D0, D1, D2 = absorption_moments(M0, M1, M2, gamma0, inv_gamma)
    P0, P1, P2 = P_derivatives(nu, gamma0, nu_c_ref)
    return (P0 * D0 + P1 * D1 + 0.5 * P2 * D2) / nu**2


def absorbed_intensity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, L,
                                    inv_gamma=None):
    """I_nu = S_nu (1 - e^-tau), S = j/alpha, from moments only."""
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    a = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    S = j / a
    return S * (1.0 - jnp.exp(-a * L))


# ---------------------------------------------------------------------------
# Polarised absorption (Stokes Q): same Kirchhoff weight g, kernel G = x K_2/3.
# The absorption moments D_k are kernel-INDEPENDENT (moments of g); only the
# derivative spectra change (G vs F).  This gives the self-absorbed linear
# polarisation fraction  Pi_thick = (j_Q/alpha_Q) / (j_I/alpha_I).
# ---------------------------------------------------------------------------


def single_particle_G(nu, gamma, nu_c_ref):
    """P_Q(nu, gamma) = G(nu / (nu_c_ref gamma^2)) (linear polarisation kernel)."""
    x = nu / (nu_c_ref * gamma**2)
    return _G(x)


def emissivity_Q(N, gamma_grid, nu, nu_c_ref):
    """j_Q = int N(gamma) G(nu/nu_c) dgamma (relative units)."""
    P = np.asarray(single_particle_G(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(N * P, gamma_grid)


def absorption_Q(N, gamma_grid, nu, nu_c_ref):
    """alpha_Q = (1/nu^2) int g(gamma) G(nu/nu_c) dgamma (relative)."""
    g = derivative_weighted(N, gamma_grid)
    P = np.asarray(single_particle_G(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(g * P, gamma_grid) / nu**2


def P_derivatives_G(nu, gamma0, nu_c_ref):
    """G, dG/dgamma, d^2G/dgamma^2 of G(nu/(nu_c_ref gamma^2)) at gamma0."""
    f = lambda g: _G(nu / (nu_c_ref * g**2))  # noqa: E731
    P0 = f(gamma0)
    P1 = jax.grad(f)(gamma0)
    P2 = jax.grad(jax.grad(f))(gamma0)
    return P0, P1, P2


def emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2):
    """j_Q from the emissivity moments M_k (2nd order, kernel G)."""
    P0, P1, P2 = P_derivatives_G(nu, gamma0, nu_c_ref)
    return P0 * M0 + P1 * M1 + 0.5 * P2 * M2


def absorption_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma=None):
    """alpha_Q from the emissivity moments (same D_k closure, kernel G)."""
    if inv_gamma is None:
        inv_gamma = inverse_gamma_moment(M0, M1, M2, gamma0)
    D0, D1, D2 = absorption_moments(M0, M1, M2, gamma0, inv_gamma)
    P0, P1, P2 = P_derivatives_G(nu, gamma0, nu_c_ref)
    return (P0 * D0 + P1 * D1 + 0.5 * P2 * D2) / nu**2


def absorbed_polarisation_fraction(nu, gamma0, nu_c_ref, M0, M1, M2, L,
                                   inv_gamma=None):
    """Absorbed linear polarisation fraction Pi = Q/I of a uniform slab.

    The two linear source functions are (perpendicular to the projected B has
    kernel F+G, parallel has F-G):

        S_perp = (jF + jG)/(aF + aG),   S_par = (jF - jG)/(aF - aG)

    and the polarisation degree is Pi = (S_perp - S_par)/(S_perp + S_par),
    with I = S_perp(1-e^-tau_perp) + S_par(1-e^-tau_par) and similarly for Q.
    Optically thick this tends to Pi_thick = (S_perp - S_par)/(S_perp+S_par),
    which for a power law equals -3/(6p+13) (the parallel source function
    dominates, i.e. the polarisation angle flips relative to the thin limit).
    """
    jF = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    aF = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    jG = emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    aG = absorption_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)

    j_perp = jF + jG
    j_par = jF - jG
    a_perp = aF + aG
    a_par = aF - aG
    S_perp = j_perp / a_perp
    S_par = j_par / a_par

    I = S_perp * (1.0 - jnp.exp(-a_perp * L)) + S_par * (1.0 - jnp.exp(-a_par * L))
    Q = S_perp * (1.0 - jnp.exp(-a_perp * L)) - S_par * (1.0 - jnp.exp(-a_par * L))
    return Q / I
