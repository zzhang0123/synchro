"""Frequency-domain SED: spectral index, curvature, and absolute emissivity.

The observable synchrotron SED at fixed frequency is what matters for 21-cm
foregrounds.  For a power-law electron spectrum N(gamma) = C gamma^-p (per
unit volume) the ultra-relativistic emissivity (per unit volume, per unit
frequency, per unit solid angle, isotropically averaged over pitch angle) is
exact:

    j_nu = sqrt(3) e^3 C B / (4 pi m_e c^2 (p+1))
         * Gamma(p/4 + 19/12) Gamma(p/4 - 1/12)
         * (3 e B / (2 pi m_e c nu))^((p-1)/2)
         * sqrt(pi) Gamma((p+5)/4) / Gamma((p+7)/4)

so that j_nu ~ nu^-(p-1)/2 and the spectral index is exactly

    alpha_s = - d ln j_nu / d ln nu = (p-1)/2.

If p (or B) varies along the sightline, the observed spectrum is the average
of j_nu over that distribution, and to leading order the SED-level moments of
Chluba+2017 are determined by the physics-level moments:

    <alpha_s> = (<p> - 1)/2,
    Delta_alpha_s = Var(p)/4            (spectral curvature).

(There is an O(Var(p)^2) correction from the p-dependence of the Gamma-function
prefactor.)  This is the first-principles origin of the phenomenological
spectral-index moment expansion.

Boundary: the closed-form emissivity (power_law_emissivity_abs/_rel) is JAX-
traceable and differentiable in (p, nu) via ``jax.scipy.special.gammaln``; the
numerical gamma-integrals (emissivity_curved, running_spectral_index) remain
NumPy/precompute and should not be used inside ``jax.jit``/``jax.grad``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

E_ESU = 4.80320427e-10
M_E = 9.1093837e-28
C_CGS = 2.99792458e10


def spectral_index(p):
    """alpha_s = (p-1)/2 for a power-law electron spectrum (I ~ nu^-alpha_s)."""
    return (p - 1.0) / 2.0


def spectral_curvature(var_p):
    """Delta_alpha_s = Var(p)/4 (leading order; the Chluba+2017 curvature)."""
    return var_p / 4.0


def power_law_emissivity_abs(nu, p, C, B):
    """Absolute isotropic power-law emissivity j_nu [erg/s/cm^3/Hz/sr].

    N(gamma) = C gamma^-p [cm^-3], B [G], nu [Hz].  Differentiable in (p, nu).
    """
    p = jnp.asarray(p)
    nu = jnp.asarray(nu)
    pref = np.sqrt(3.0) * E_ESU**3 * C * B / (4.0 * np.pi * M_E * C_CGS**2)
    g1 = jnp.exp(jax.scipy.special.gammaln(p / 4.0 + 19.0 / 12.0)
                 + jax.scipy.special.gammaln(p / 4.0 - 1.0 / 12.0))
    x = jnp.asarray(3.0 * E_ESU * B / (2.0 * np.pi * M_E * C_CGS)) / nu
    ang = jnp.exp(0.5 * jnp.log(jnp.pi)
                  + jax.scipy.special.gammaln((p + 5.0) / 4.0)
                  - jax.scipy.special.gammaln((p + 7.0) / 4.0))
    return pref * g1 * x ** ((p - 1.0) / 2.0) * ang / (p + 1.0)


def power_law_emissivity_rel(nu, p):
    """Relative (unit-normalised) power-law emissivity, differentiable in (p, nu)."""
    p = jnp.asarray(p)
    nu = jnp.asarray(nu)
    g1 = jnp.exp(jax.scipy.special.gammaln(p / 4.0 + 19.0 / 12.0)
                 + jax.scipy.special.gammaln(p / 4.0 - 1.0 / 12.0))
    ang = jnp.exp(0.5 * jnp.log(jnp.pi)
                  + jax.scipy.special.gammaln((p + 5.0) / 4.0)
                  - jax.scipy.special.gammaln((p + 7.0) / 4.0))
    return 3.0 ** ((p - 1.0) / 2.0) * nu ** (-(p - 1.0) / 2.0) * g1 * ang / (p + 1.0)


# ---------------------------------------------------------------------------
# Item 1: LOS (or sky) average of a spatially varying spectral index.
#
# The observed total intensity is the emissivity-weighted sum over the
# sightline (and, for the global signal, over the sky):
#     I(nu) = int ds j0(s) nu^-alpha_s(s),   alpha_s(s) = (p(s)-1)/2 .
# ln I(nu) is then the cumulant-generating function of alpha_s, so its Taylor
# coefficients in ln nu are the cumulants of alpha_s:
#     ln I = -<alpha_s> ln nu + (1/2)Var(alpha_s) ln^2 nu
#          - (1/6)Skew(alpha_s) ln^3 nu + ...
# i.e. the Chluba+2017 SED moments are the (emissivity-weighted) cumulants of
# the electron-spectrum index along the sightline/sky.
# ---------------------------------------------------------------------------


def los_intensity(nu, s_grid, j0, p):
    """I(nu) = int j0(s) nu^-((p(s)-1)/2) ds (relative units)."""
    return np.trapezoid(j0 * nu ** (-(p - 1.0) / 2.0), s_grid)


def spectral_index_cumulants(s_grid, j0, p):
    """Emissivity-weighted cumulants (mean, var, skew) of alpha_s = (p-1)/2."""
    alpha = (p - 1.0) / 2.0
    w = j0 / np.trapezoid(j0, s_grid)
    mean = np.trapezoid(w * alpha, s_grid)
    var = np.trapezoid(w * (alpha - mean) ** 2, s_grid)
    skew = np.trapezoid(w * (alpha - mean) ** 3, s_grid)
    return mean, var, skew


# ---------------------------------------------------------------------------
# Item 2: curved (log-parabola) electron spectrum -> running spectral index.
#
# For p(gamma) = p0 + a ln(gamma/gamma_ref) the emissivity is no longer a pure
# power law; the spectral index runs with frequency as
#     alpha_s(nu) = (p0-1)/2 + (a/2) ln(nu/nu_ref) + O(a^2)
# (verified numerically; the monochromatic approx. would give a/4 and is wrong
# by a factor of two).  The general curved spectrum is handled by
# running_spectral_index via a direct log-derivative of the emissivity.
# ---------------------------------------------------------------------------


def log_parabola_running_index(nu, p0, a, nu_ref=1.0):
    """alpha_s(nu) for p(gamma)=p0 + a ln(gamma), leading order in a."""
    return (p0 - 1.0) / 2.0 + (a / 2.0) * np.log(nu / nu_ref)


def emissivity_curved(nu, p_func, nu_c_ref=1.0, gmin=0.05, gmax=5000.0, ng=400):
    """j_nu = int gamma^-p(gamma) F(nu/(nu_c_ref gamma^2)) dgamma (relative)."""
    from .ultrarel import F as _F
    import jax.numpy as jnp
    g = np.geomspace(gmin, gmax, ng)
    dl = np.log(g[1] / g[0])
    tot = 0.0
    for gi in g:
        tot += gi ** (-p_func(gi)) * float(_F(jnp.asarray(nu / (nu_c_ref * gi**2)))) * gi * dl
    return tot


def running_spectral_index(nus, p_func, nu_c_ref=1.0, **kw):
    """alpha_s(nu) = -d ln j_nu / d ln nu for a curved electron spectrum."""
    js = np.array([emissivity_curved(nu, p_func, nu_c_ref, **kw) for nu in nus])
    return -np.gradient(np.log(js), np.log(nus))
