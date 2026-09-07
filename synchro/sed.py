"""Continuum power-law emissivity and empirical-SED mixture identities.

The analytic power-law formula extends the energy integral to (0,infinity),
requires p>1/3, uses the ultra-relativistic vacuum continuum kernel, and assumes
conditional pitch isotropy. Energy support/tails and finite-gamma corrections
are separate discrepancies. The absolute API declares whether a directional
ordered field or an average over viewing directions is wanted; these differ
from an unnormalised angular integral.

The slope a_s=(p-1)/2 holds within that reference model. For spatial mixtures,
cumulants are emissivity-amplitude weighted at a declared frequency pivot,
not electron-number weighted. Curvature Var(p)/4 refers to those weights at
that pivot; mapping unweighted population statistics into it is another step.

Analytic formulas use JAX. Finite energy quadrature remains NumPy precompute,
with caller-controlled bounds/resolution; numerical convergence does not bound
excluded physical tails or certify a PDF closure.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

# Imported aliases preserve the legacy public names.
from .constants import C_CGS, E_ESU, M_E


def spectral_index(p):
    """alpha_s = (p-1)/2 for a power-law electron spectrum (I ~ nu^-alpha_s)."""
    return (p - 1.0) / 2.0


def spectral_curvature(var_p):
    """Delta_alpha_s = Var(p)/4 (leading order; the Chluba+2017 curvature)."""
    return var_p / 4.0


def power_law_emissivity_abs(nu, p, C, B, *, theta=None):
    """Physical emissivity [erg/s/cm^3/Hz/sr] for N(gamma)=C gamma^-p.

    B [G], nu [Hz], p>1/3. ``theta`` selects an ordered-field viewing angle;
    the directional continuum approximation assumes isotropic electron pitch
    angles. ``theta=None`` averages that directional emissivity with the
    normalised measure sin(theta)dtheta/2 (equivalently, random field axes).
    This default corrects the legacy omitted factor1/2. A physical ordered
    field generally emits anisotropically even for isotropic electron pitch.
    The analytic integral extends gamma to (0,infinity); its physical energy
    cutoffs, continuum error and pitch anisotropy remain explicit limitations.
    """
    p = jnp.asarray(p)
    nu = jnp.asarray(nu)
    field = jnp.asarray(B)
    positive_field = jnp.where(field > 0, field, 1.0)
    pref = np.sqrt(3.0) * E_ESU**3 * C * positive_field / (4.0 * np.pi * M_E * C_CGS**2)
    g1 = jnp.exp(
        jax.scipy.special.gammaln(p / 4.0 + 19.0 / 12.0)
        + jax.scipy.special.gammaln(p / 4.0 - 1.0 / 12.0)
    )
    x = 3.0 * E_ESU * positive_field / (2.0 * np.pi * M_E * C_CGS * nu)
    ang = jnp.exp(
        0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln((p + 5.0) / 4.0)
        - jax.scipy.special.gammaln((p + 7.0) / 4.0)
    )
    orientation = (
        0.5 * ang if theta is None else jnp.abs(jnp.sin(theta)) ** ((p + 1.0) / 2.0)
    )
    value = pref * g1 * x ** ((p - 1.0) / 2.0) * orientation / (p + 1.0)
    value = jnp.where(field == 0, 0.0, value)
    return jnp.where((p > 1.0 / 3.0) & (nu > 0) & (field >= 0), value, jnp.nan)


def power_law_emissivity_rel(nu, p):
    """Legacy relative integral, with unnormalised sin(theta)dtheta measure.

    Its angular integral is twice the probability average. For the absolute
    orientation-averaged result multiply by sqrt(3)e^3 C B/(8pi m_e c^2)
    and evaluate at nu/nu_B. p>1/3 and nu>0; the energy integral is extended.
    """
    p = jnp.asarray(p)
    nu = jnp.asarray(nu)
    g1 = jnp.exp(
        jax.scipy.special.gammaln(p / 4.0 + 19.0 / 12.0)
        + jax.scipy.special.gammaln(p / 4.0 - 1.0 / 12.0)
    )
    ang = jnp.exp(
        0.5 * jnp.log(jnp.pi)
        + jax.scipy.special.gammaln((p + 5.0) / 4.0)
        - jax.scipy.special.gammaln((p + 7.0) / 4.0)
    )
    value = 3.0 ** ((p - 1.0) / 2.0) * nu ** (-(p - 1.0) / 2.0) * g1 * ang / (p + 1.0)
    return jnp.where((p > 1.0 / 3.0) & (nu > 0), value, jnp.nan)


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


def los_intensity(nu, s_grid, j0, p, *, nu0=1.0):
    """I(nu)=int j0(s) (nu/nu0)^-((p(s)-1)/2) ds; j0 is at pivot nu0."""
    return np.trapezoid(j0 * (nu / nu0) ** (-(p - 1.0) / 2.0), s_grid)


def spectral_index_cumulants(s_grid, j0, p):
    """Emissivity-weighted cumulants (mean, variance, third cumulant) at j0's pivot.

    The third returned value is not dimensionless normalised skewness."""
    alpha = (p - 1.0) / 2.0
    w = j0 / np.trapezoid(j0, s_grid)
    mean = np.trapezoid(w * alpha, s_grid)
    var = np.trapezoid(w * (alpha - mean) ** 2, s_grid)
    skew = np.trapezoid(w * (alpha - mean) ** 3, s_grid)
    return mean, var, skew


# ---------------------------------------------------------------------------
# Log-parabola: N=C exp[-p0*y-a*y^2], y=ln(gamma/gamma_ref).
# The local electron slope is p0+2*a*y. This is not p0+a*y.
# The finite continuum kernel supplies the Mellin offset m_x.
# ---------------------------------------------------------------------------


def log_parabola_running_index(nu, p0, a, nu_ref=1.0):
    """Leading index for N=C exp(-p0*y-a*y^2), with an extended energy range.

    For a>=0 (zero is the power-law limit),
    alpha=(p0-1)/2 + a/2 [ln(nu/nu_ref)-d ln M(q)/dq] + O(a^2),
    q=(p0-1)/2 > -1/3. nu_ref=a_B*gamma_ref^2. For local electron slope
    p0+b*y use a=b/2. The omitted index term is
    -a/2*(<ln x>_a-<ln x>_0); the running correction is
    -a^2 Var_a(ln x)/4 under the tilted continuum kernel. These remainders
    exclude finite-energy-tail and physical-kernel discrepancies.
    """
    q = (jnp.asarray(p0) - 1.0) / 2.0
    mx = (
        jnp.log(2.0)
        - 1.0 / (q + 1.0)
        + 0.5 * jax.scipy.special.digamma(q / 2 + 11.0 / 6.0)
        + 0.5 * jax.scipy.special.digamma(q / 2 + 1.0 / 6.0)
    )
    value = q + a / 2.0 * (jnp.log(jnp.asarray(nu) / nu_ref) - mx)
    return jnp.where(
        (q > -1.0 / 3.0) & (jnp.asarray(nu) > 0) & (nu_ref > 0) & (jnp.asarray(a) >= 0),
        value,
        jnp.nan,
    )


def emissivity_curved(nu, p_func, nu_c_ref=1.0, gmin=0.05, gmax=5000.0, ng=400):
    """Finite integral gamma^-p_func(gamma) F(nu/(nu_c_ref*gamma^2)).

    p_func is the exponent, not the local logarithmic electron slope. Bounds
    below gamma=1 are a formal mathematical continuation, not physical Lorentz
    factors. The default retains the historical integration interval; callers
    must declare physical support and propagate excluded tails separately.
    """
    from .ultrarel import F

    if not (0 < gmin < gmax) or ng < 3:
        raise ValueError("require 0 < gmin < gmax and ng >= 3")
    g = np.geomspace(gmin, gmax, ng)
    exponents = np.array([p_func(gi) for gi in g])
    kernel = np.asarray(F(jnp.asarray(nu / (nu_c_ref * g**2))))
    return np.trapezoid(g ** (1.0 - exponents) * kernel, np.log(g))


def running_spectral_index(nus, p_func, nu_c_ref=1.0, **kw):
    """alpha_s(nu) = -d ln j_nu / d ln nu for a curved electron spectrum."""
    js = np.array([emissivity_curved(nu, p_func, nu_c_ref, **kw) for nu in nus])
    return -np.gradient(np.log(js), np.log(nus), edge_order=2)
