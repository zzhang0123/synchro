"""Kirchhoff: emissivity and self-absorption from a distribution N(gamma).

In the ultra-relativistic regime the single-particle power per unit frequency
is P(nu, gamma) ~ F(nu / nu_c) with nu_c ~ gamma^2, and the emissivity (per
unit solid angle) and absorption coefficient are

    j_nu   =  int N(gamma) P(nu, gamma) dgamma
    alpha  =  (1/nu^2) int g(gamma) P(nu, gamma) dgamma ,
    g(gamma) = - gamma^2 d/dgamma [ N(gamma) / gamma^2 ]          (Kirchhoff)

in reduced units. The distinct physical prefactors do NOT cancel in the
source function; ``cgs_coefficients_from_moments`` restores them. Absorption
probes the *derivative-
weighted* distribution g rather than N itself; for a power law N ~ gamma^-p,
g = (p+2) N / gamma, giving the standard alpha ~ nu^-(p+4)/2 and
S_nu ~ nu^(5/2).

Moment expansion: expanding P(nu, gamma) around gamma0 gives j_nu as moments
of N and alpha_nu as moments of g, with the SAME derivative spectra of P
(computed by autodiff).  This is the transfer-side analogue of the emissivity
derivative spectra of the main paper.

Boundary: the grid-integral functions are *NumPy precomputations*. They use
``np.trapezoid``/``np.gradient`` (not differentiable), but the single-particle
kernel P = F(nu/nu_c) is evaluated by the jitted JAX ``F``. Do NOT wrap these
grid functions in ``jax.jit``/``jax.grad``. The scalar kernel derivatives and
``*_from_moments`` contractions ARE differentiable JAX functions. Their
second-order Taylor truncation needs a remainder or direct-integration error
test; finite moments do not by themselves justify that kernel truncation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .ultrarel import F, G
from .rm import E_ESU, M_E, C_CGS
from .transfer import optical_depth_factor

_F = jax.jit(F)
_G = jax.jit(G)


def _population_grid(N, gamma_grid):
    N, gamma_grid = np.asarray(N, dtype=float), np.asarray(gamma_grid, dtype=float)
    if (
        gamma_grid.ndim != 1
        or gamma_grid.size < 3
        or N.shape != gamma_grid.shape
        or not np.all(np.isfinite(gamma_grid))
        or not np.all(np.isfinite(N))
        or np.any(np.diff(gamma_grid) <= 0)
        or np.any(gamma_grid <= 0)
        or np.any(N < 0)
    ):
        raise ValueError(
            "N must be nonnegative and finite on a matching, positive, increasing gamma grid (>=3 points)"
        )
    return N, gamma_grid


def single_particle_P(nu, gamma, nu_c_ref):
    """P(nu, gamma) = F(nu / (nu_c_ref gamma^2)), vectorised over ``gamma``."""
    x = nu / (nu_c_ref * gamma**2)
    return _F(x)


def emissivity(N, gamma_grid, nu, nu_c_ref):
    """j_nu = int N(gamma) P(nu, gamma) dgamma (relative units)."""
    N, gamma_grid = _population_grid(N, gamma_grid)
    P = np.asarray(single_particle_P(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(N * P, gamma_grid)


def derivative_weighted(N, gamma_grid, *, relativistic=False):
    """Finite-difference -h(N/h)' on a finite grid; endpoint jumps excluded.

    Default h=gamma^2 is the ultra-relativistic approximation. Set
    relativistic=True for h=gamma*sqrt(gamma^2-1), requiring gamma>1.
    No physical gamma<1 population is implied by the reduced default's
    mathematically positive coordinate domain. Refine the grid separately.
    """
    N, gamma_grid = _population_grid(N, gamma_grid)
    if relativistic and np.any(gamma_grid <= 1):
        raise ValueError("exact radial measure requires gamma > 1")
    h = gamma_grid * np.sqrt(gamma_grid**2 - 1) if relativistic else gamma_grid**2
    return -h * np.gradient(N / h, gamma_grid, edge_order=2)


def absorption(N, gamma_grid, nu, nu_c_ref):
    """alpha_nu = (1/nu^2) int g(gamma) P(nu, gamma) dgamma (relative)."""
    g = derivative_weighted(N, gamma_grid)
    P = np.asarray(single_particle_P(nu, gamma_grid, nu_c_ref))
    return np.trapezoid(g * P, gamma_grid) / nu**2


def source_function(N, gamma_grid, nu, nu_c_ref):
    """S_nu = j_nu / alpha_nu."""
    return emissivity(N, gamma_grid, nu, nu_c_ref) / absorption(
        N, gamma_grid, nu, nu_c_ref
    )


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
#     D2 = 4 M1 - 2 g0 M0 + 2 g0^2 <1/g>.                        (UR identity; vanishing endpoints)
#
# The inverse moment is not determined by the integer moments; for a narrow
# distribution it is given by the geometric series
#
#     <1/gamma> = M0/g0 - M1/g0^2 + M2/g0^3 + ...                (approx.)
# ---------------------------------------------------------------------------


def inverse_gamma_moment(M0, M1, M2, gamma0):
    """<1/gamma> = int N/gamma dgamma via the geometric-series expansion.

    Valid for ``|gamma - gamma0| < gamma0``; truncated at the second moment.
    """
    return M0 / gamma0 - M1 / gamma0**2 + M2 / gamma0**3


def absorption_moments(
    M0, M1, M2, gamma0, inv_gamma, *, boundary_terms=(0.0, 0.0, 0.0)
):
    """UR integration-by-parts identity, retaining finite-domain endpoints.

    ``boundary_terms[k]`` is [N(gamma)*(gamma-gamma0)^k]_lower^upper.
    Default zero assumes these boundary terms vanish, or are represented
    separately. Physical sharp cutoffs can also carry distributional jumps.
    This identity for h=gamma^2 is not the exact mildly-relativistic weight.
    M2 is accepted for a consistent interface but does not enter D0..D2.
    """
    if len(boundary_terms) != 3:
        raise ValueError("boundary_terms must have three entries")
    D0 = 2.0 * inv_gamma - boundary_terms[0]
    D1 = 3.0 * M0 - 2.0 * gamma0 * inv_gamma - boundary_terms[1]
    D2 = 4.0 * M1 - 2.0 * gamma0 * M0 + 2.0 * gamma0**2 * inv_gamma - boundary_terms[2]
    return D0, D1, D2


def emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2):
    """j_nu from the emissivity moments M_k (2nd order, no N needed)."""
    P0, P1, P2 = P_derivatives(nu, gamma0, nu_c_ref)
    return P0 * M0 + P1 * M1 + 0.5 * P2 * M2


def absorption_from_moments(
    nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma=None, *, boundary_terms=(0.0, 0.0, 0.0)
):
    """alpha_nu from the emissivity moments via Kirchhoff closure.

    ``inv_gamma`` = <1/gamma>; if None it is approximated by the geometric
    series (narrow-distribution limit).
    """
    if inv_gamma is None:
        inv_gamma = inverse_gamma_moment(M0, M1, M2, gamma0)
    D0, D1, D2 = absorption_moments(
        M0, M1, M2, gamma0, inv_gamma, boundary_terms=boundary_terms
    )
    P0, P1, P2 = P_derivatives(nu, gamma0, nu_c_ref)
    return (P0 * D0 + P1 * D1 + 0.5 * P2 * D2) / nu**2


def absorbed_intensity_from_moments(
    nu, gamma0, nu_c_ref, M0, M1, M2, L, inv_gamma=None
):
    """I_nu = S_nu (1 - e^-tau), S = j/alpha, from moments only."""
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    a = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    return j * L * optical_depth_factor(a * L)


# ---------------------------------------------------------------------------
# Polarised absorption (Stokes Q): same Kirchhoff weight g, kernel G = x K_2/3.
# The absorption moments D_k are kernel-INDEPENDENT (moments of g); only the
# derivative spectra change (-G vs F). The thick polarisation follows the
# two normal-mode source functions, not the ratio of Q and I source functions.
# ---------------------------------------------------------------------------


def single_particle_G(nu, gamma, nu_c_ref):
    """Positive G kernel; the paper's signed Q kernel is MINUS this value."""
    x = nu / (nu_c_ref * gamma**2)
    return _G(x)


def emissivity_Q(N, gamma_grid, nu, nu_c_ref):
    """j_Q = -int N(gamma) G(nu/nu_c) dgamma (relative units)."""
    N, gamma_grid = _population_grid(N, gamma_grid)
    P = np.asarray(single_particle_G(nu, gamma_grid, nu_c_ref))
    return -np.trapezoid(N * P, gamma_grid)


def absorption_Q(N, gamma_grid, nu, nu_c_ref):
    """alpha_Q = -(1/nu^2) int g(gamma) G(nu/nu_c) dgamma (relative)."""
    g = derivative_weighted(N, gamma_grid)
    P = np.asarray(single_particle_G(nu, gamma_grid, nu_c_ref))
    return -np.trapezoid(g * P, gamma_grid) / nu**2


def P_derivatives_G(nu, gamma0, nu_c_ref):
    """G, dG/dgamma, d^2G/dgamma^2 of G(nu/(nu_c_ref gamma^2)) at gamma0."""
    f = lambda g: _G(nu / (nu_c_ref * g**2))  # noqa: E731
    P0 = f(gamma0)
    P1 = jax.grad(f)(gamma0)
    P2 = jax.grad(jax.grad(f))(gamma0)
    return P0, P1, P2


def emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2):
    """Signed j_Q from emissivity moments (2nd order, kernel -G)."""
    P0, P1, P2 = P_derivatives_G(nu, gamma0, nu_c_ref)
    return -(P0 * M0 + P1 * M1 + 0.5 * P2 * M2)


def absorption_Q_from_moments(
    nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma=None, *, boundary_terms=(0.0, 0.0, 0.0)
):
    """Signed alpha_Q from the same UR D_k closure, with kernel -G."""
    if inv_gamma is None:
        inv_gamma = inverse_gamma_moment(M0, M1, M2, gamma0)
    D0, D1, D2 = absorption_moments(
        M0, M1, M2, gamma0, inv_gamma, boundary_terms=boundary_terms
    )
    P0, P1, P2 = P_derivatives_G(nu, gamma0, nu_c_ref)
    return -(P0 * D0 + P1 * D1 + 0.5 * P2 * D2) / nu**2


def absorbed_polarisation_fraction(nu, gamma0, nu_c_ref, M0, M1, M2, L, inv_gamma=None):
    """Signed Q/I of a uniform emitting slab with Q=parallel-perpendicular.

    The normal modes have j_parallel=(jI+jQ)/2, alpha_parallel=aI+aQ,
    and the complementary minus signs for perpendicular. For a broad power
    law the thin ratio is -(p+1)/(p+7/3), and the thick ratio +3/(6p+13).
    A zero-intensity population has no defined polarisation fraction.
    """
    jI = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    aI = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    jQ = emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    aQ = absorption_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    parallel = 0.5 * (jI + jQ) * L * optical_depth_factor((aI + aQ) * L)
    perpendicular = 0.5 * (jI - jQ) * L * optical_depth_factor((aI - aQ) * L)
    return (parallel - perpendicular) / (parallel + perpendicular)


def cgs_coefficients_from_moments(
    nu_hz,
    gamma0,
    B_perp_gauss,
    M0,
    M1,
    M2,
    inv_gamma,
    *,
    boundary_terms=(0.0, 0.0, 0.0),
):
    """Return (jI,jQ,alphaI,alphaQ) in CGS for isotropic UR electrons.

    M_k=int N(gamma)(gamma-gamma0)^k dgamma and inv_gamma=int N/gamma
    are local number-density moments [cm^-3], not normalized PDF moments.
    j is erg/s/cm^3/Hz/sr; alpha is cm^-1. This is a SECOND-ORDER local
    kernel Taylor approximation using the UR Kirchhoff weight; it is not a
    reconstruction of the electron PDF. Supply an independent inverse moment
    and finite-support endpoints. The zero endpoint default is a hypothesis.
    Use only nu>0, gamma0>1 (physically gamma0>>1), B_perp>=0 and M0>=0;
    remaining moment realizability and remainder checks belong to the caller.
    """
    import equinox as eqx

    values = jnp.asarray([nu_hz, gamma0, B_perp_gauss, M0, M1, M2, inv_gamma])
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values))
        | (values[0] <= 0)
        | (values[1] <= 1)
        | (values[2] < 0)
        | (values[3] < 0),
        "cgs inputs require finite nu>0, gamma0>1, B_perp>=0, M0>=0",
    )
    nu_hz, gamma0, B_perp_gauss, M0, M1, M2, inv_gamma = values
    # Safe internal frequency at B_perp=0; its physical amplitude is zero.
    safe_b = jnp.where(B_perp_gauss > 0, B_perp_gauss, 1.0)
    aB = 3 * E_ESU * safe_b / (4 * jnp.pi * M_E * C_CGS)
    amp = jnp.sqrt(3.0) * E_ESU**3 * B_perp_gauss / (4 * jnp.pi * M_E * C_CGS**2)
    args = (nu_hz, gamma0, aB, M0, M1, M2)
    jI = amp * emissivity_from_moments(*args)
    jQ = amp * emissivity_Q_from_moments(*args)
    aI = (
        amp
        / (2 * M_E)
        * absorption_from_moments(*args, inv_gamma, boundary_terms=boundary_terms)
    )
    aQ = (
        amp
        / (2 * M_E)
        * absorption_Q_from_moments(*args, inv_gamma, boundary_terms=boundary_terms)
    )
    # At fixed positive nu and finite energy, radiation and its one-sided
    # field derivatives vanish as B_perp->0. Do not differentiate safe_b=1.
    return tuple(jnp.where(B_perp_gauss > 0, v, 0.0) for v in (jI, jQ, aI, aQ))
