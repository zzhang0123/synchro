"""Differentiable Bessel functions for JAX.

JAX's ``jax.scipy.special.bessel_jn`` returns NaN for small arguments in
v0.10, so we implement the functions we need directly from their defining
integral representations, which are exact for the orders we use and are
automatically differentiable w.r.t. the argument.

References
----------
* J_n(x) = (1/pi) int_0^pi  cos(n t - x sin t) dt         (integer n >= 0)
* J_n'(x) = (1/pi) int_0^pi  sin(t) sin(n t - x sin t) dt
* K_nu(x) = int_0^inf exp(-x cosh t) cosh(nu t) dt
          = (1/2) int_0^1 exp[-(x/2)(u + 1/u)] (u^(nu-1) + u^(-nu-1)) du

The Gauss--Legendre grids are precomputed once (NumPy, at import) and held as
static arrays; they never enter any jitted graph as leaves.

Note: these functions are *eager* (not jitted) and re-run the fixed quadrature
on every call. For repeated evaluation the fast path is to precompute the
derivative spectra once via ``synchro.moment_expansion.build_expansion`` (or
``synchro.derivatives.derivative_spectra``), not to call ``stokes_harmonic`` in
a loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# Precomputed quadrature grids (static)
# ----------------------------------------------------------------------------


def _gauss_legendre(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(x), jnp.asarray(w)


# Bessel integral lives on t in [0, pi].
_N_J = 512
_xg, _wg = _gauss_legendre(_N_J)
_T_J = jnp.asarray(np.pi / 2.0 * (_xg + 1.0))  # (N,)
_W_J = jnp.asarray(_wg * np.pi / 2.0)  # (N,)

# K_nu integral lives on u in [0, 1].
_N_K = 512
_xgk, _wgk = _gauss_legendre(_N_K)
_U_K = jnp.asarray(0.5 * (_xgk + 1.0))  # (N,)
_W_K = jnp.asarray(_wgk * 0.5)  # (N,)


# ----------------------------------------------------------------------------
# Regular Bessel J_n and its derivative
# ----------------------------------------------------------------------------


def bessel_jn(n, x):
    """Regular Bessel function J_n(x).

    Parameters
    ----------
    n : int or array-like
        Order (integer). Static for autodiff w.r.t. ``x``.
    x : scalar or array-like
        Argument. Broadcast against ``n``.
    """
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    integrand = jnp.cos(n[..., None] * _T_J - x[..., None] * jnp.sin(_T_J))
    return jnp.sum(_W_J * integrand, axis=-1) / jnp.pi


def bessel_jn_prime(n, x):
    """Derivative J_n'(x) = dJ_n/dx via its integral representation."""
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    integrand = jnp.sin(_T_J) * jnp.sin(n[..., None] * _T_J - x[..., None] * jnp.sin(_T_J))
    return jnp.sum(_W_J * integrand, axis=-1) / jnp.pi


def bessel_jn_and_prime(n, x):
    """Return (J_n(x), J_n'(x)) together (single quadrature pass)."""
    n = jnp.asarray(n)
    x = jnp.asarray(x)
    ph = n[..., None] * _T_J - x[..., None] * jnp.sin(_T_J)
    jn = jnp.sum(_W_J * jnp.cos(ph), axis=-1) / jnp.pi
    jnp_ = jnp.sum(_W_J * jnp.sin(_T_J) * jnp.sin(ph), axis=-1) / jnp.pi
    return jn, jnp_


# ----------------------------------------------------------------------------
# Modified Bessel K_nu (for the ultra-relativistic F/G functions)
# ----------------------------------------------------------------------------


def bessel_kn(nu: float, x):
    """Modified Bessel function of the second kind K_nu(x).

    Implemented via the u-integral representation with Gauss--Legendre on
    [0, 1]. Reliable for x >~ 1e-2; for x -> 0 the small-argument
    asymptotics K_nu(x) ~ Gamma(nu)/2 (x/2)^-nu is used.
    """
    nu = float(nu)
    x = jnp.asarray(x)
    u = _U_K
    # integrand of K_nu = 0.5 * exp(-(x/2)(u + 1/u)) * (u^(nu-1) + u^(-nu-1))
    arg = 0.5 * x[..., None] * (u + 1.0 / u)
    poly = u ** (nu - 1.0) + u ** (-nu - 1.0)
    integrand = 0.5 * jnp.exp(-arg) * poly
    k = jnp.sum(_W_K * integrand, axis=-1)

    # small-x guard: K_nu ~ Gamma(nu)/2 * (x/2)^-nu   (nu > 0)
    small = 0.5 * jnp.exp(jax.scipy.special.gammaln(nu)) * (x / 2.0) ** (-nu)
    return jnp.where(x < 1e-2, small, k).reshape(jnp.shape(x))
