"""Differentiable integer J and modified K functions with explicit quadrature.

Integer J uses a periodic trapezoidal rule, with an exactly equivalent complex
contour shift for 0 < |x| < n to resolve exponentially small values without
subtracting order-one oscillations. The contour is held fixed during autodiff:
the integral is invariant under that shift, so this differentiates the same
analytic function rather than the arbitrary contour-selection algorithm.

For a concrete order the default grid resolves |x| <= n (the harmonic domain).
For a traced order the default is 2048 nodes; choose a larger static ``n_nodes``
when needed. Calls outside |n| + |x| <= n_nodes/2 return NaN, rather than silently
aliasing unresolved oscillations. This conservative resolution guard is not a
roundoff or quadrature-error certificate. Compare increased resolution and an
independent reference for a new domain, including the required derivatives.

K uses one continuous cosh integral; there is no asymptotic switch at x=0.01.
Its finite upper limit has the explicit bound returned by ``bessel_kn_tail_bound``.
The quadrature error is separate. Finite independent tests cover orders 0, 2/3,
5/3 over x=1e-12..100 and integer J through n=3200; they do not establish uniform
accuracy outside those cases. Order, node count and tail cutoff are static;
positive arguments are differentiated. SciPy is used only in independent tests.
"""

from __future__ import annotations

from functools import lru_cache
import math
import numbers

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=20)
def _periodic_grid(n_nodes):
    if n_nodes < 16:
        raise ValueError("n_nodes must be at least 16")
    t = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    return np.sin(t), np.cos(t), t


@lru_cache(maxsize=8)
def _legendre_grid(n_nodes):
    if n_nodes < 16:
        raise ValueError("n_nodes must be at least 16")
    return np.polynomial.legendre.leggauss(n_nodes)


def _j_resolution(n, x, n_nodes):
    if n_nodes is not None:
        return int(n_nodes)
    # No host materialisation of traced or device arrays in an online graph.
    if isinstance(n, (numbers.Real, np.ndarray, list, tuple)):
        extent = float(np.max(np.abs(np.asarray(n))))
        if isinstance(x, (numbers.Real, np.ndarray, list, tuple)):
            extent = max(extent, float(np.max(np.abs(np.asarray(x)))))
        count = 2 ** math.ceil(math.log2(max(128, 4 * extent + 64)))
        if count > 65536:
            raise ValueError(
                "automatic J quadrature exceeds 65536 nodes; choose a validated static n_nodes or a suitable reference approximation"
            )
        return count
    return 2048


def bessel_jn_and_prime(n, x, *, n_nodes=None):
    """Return integer-order (J_n(x), J_n'(x)); broadcast n and x.

    ``n >= 0`` must be integral. For JIT with a varying order, ``n_nodes`` is
    a static resolution choice; unresolved inputs are NaN. At x=0 the analytic
    function values are supplied; use a nonzero argument for autodiff.
    """
    count = _j_resolution(n, x, n_nodes)
    sin_t, cos_t, t = _periodic_grid(count)
    n, x = jnp.broadcast_arrays(jnp.asarray(n), jnp.asarray(x))
    ax = jnp.abs(x)
    safe_x = jnp.where(ax > 0, ax, 1.0)
    shifted = (n > safe_x) & (n > 0)
    # Fix the contour while differentiating. Scaled sinh/cosh avoid overflow
    # when the shift is large but x*sinh(a) and x*cosh(a) remain modest.
    d = jax.lax.stop_gradient(jnp.sqrt(jnp.maximum(n**2 - safe_x**2, 0.0)))
    shift = jax.lax.stop_gradient(
        jnp.where(shifted, jnp.log(jnp.maximum(n + d, 1.0)) - jnp.log(safe_x), 0.0)
    )
    reference_x = jax.lax.stop_gradient(safe_x)
    ratio = safe_x / reference_x
    xsinh = jnp.where(shifted, d * ratio, 0.0)
    xcosh = jnp.where(shifted, n * ratio, safe_x)
    phase = n[..., None] * t - xcosh[..., None] * sin_t
    envelope = jnp.exp(xsinh[..., None] * (cos_t - 1.0))
    pref = jnp.exp(xsinh - n * shift)
    value = pref * jnp.mean(envelope * jnp.cos(phase), axis=-1)
    prime = (
        pref
        * jnp.mean(
            envelope
            * (
                xsinh[..., None] * cos_t * jnp.cos(phase)
                + xcosh[..., None] * sin_t * jnp.sin(phase)
            ),
            axis=-1,
        )
        / safe_x
    )
    parity = jnp.where(n % 2 == 0, 1.0, -1.0)
    value = jnp.where(x < 0, parity * value, value)
    prime = jnp.where(x < 0, -parity * prime, prime)
    value = jnp.where(ax == 0, jnp.where(n == 0, 1.0, 0.0), value)
    prime = jnp.where(ax == 0, jnp.where(n == 1, 0.5, 0.0), prime)
    valid = (n >= 0) & (n == jnp.floor(n)) & (n + ax <= count / 2)
    return jnp.where(valid, value, jnp.nan), jnp.where(valid, prime, jnp.nan)


def bessel_jn(n, x, *, n_nodes=None):
    """Integer J_n; see ``bessel_jn_and_prime`` for the resolution contract."""
    return bessel_jn_and_prime(n, x, n_nodes=n_nodes)[0]


def bessel_jn_prime(n, x, *, n_nodes=None):
    """Argument derivative of integer J_n, evaluated on the same contour."""
    return bessel_jn_and_prime(n, x, n_nodes=n_nodes)[1]


def _log_cosh(x):
    magnitude = jnp.abs(x)
    return magnitude + jnp.log1p(jnp.exp(-2.0 * magnitude)) - jnp.log(2.0)


def _modified_tmax(nu, x, tail_cutoff):
    cutoff = tail_cutoff + 2 * abs(nu)
    # arccosh(1+cutoff/x), expressed without overflowing cutoff/x at tiny x.
    return jnp.log(x + cutoff + jnp.sqrt(cutoff * (cutoff + 2 * x))) - jnp.log(x)


def _modified_integral(nu, x, *, divide_cosh=False, n_nodes=128, tail_cutoff=50.0):
    """Integral exp(-x cosh t) cosh(nu t) [ / cosh(t)] dt."""
    nu = float(nu)
    if not np.isfinite(nu) or tail_cutoff <= 0:
        raise ValueError("nu must be finite and tail_cutoff must be positive")
    nodes, weights = _legendre_grid(int(n_nodes))
    x = jnp.asarray(x)
    safe_x = jnp.where(x > 0, x, 1.0)
    tmax = _modified_tmax(nu, safe_x, tail_cutoff)
    t = 0.5 * tmax[..., None] * (nodes + 1.0)
    log_cosh_t = _log_cosh(t)
    log_integrand = -jnp.exp(jnp.log(safe_x)[..., None] + log_cosh_t) + _log_cosh(
        nu * t
    )
    if divide_cosh:
        log_integrand = log_integrand - log_cosh_t
    integrand = jnp.exp(log_integrand)
    result = 0.5 * tmax * jnp.sum(weights * integrand, axis=-1)
    return jnp.where(x > 0, result, jnp.where(x == 0, jnp.inf, jnp.nan))


def _modified_tail_bound(nu, x, *, divide_cosh=False, tail_cutoff=50.0):
    """Analytic upper bound for the omitted positive-t integral, x > 0.

    cosh(t)>=exp(t)/2 and cosh(nu*t)<=exp(|nu|*t) reduce the tail to an
    incomplete Gamma integral. Bounding log(u/z)<= (u-z)/z gives the elementary
    bound below. It excludes quadrature and floating-point error.
    """
    nu = abs(float(nu))
    x = jnp.asarray(x)
    tmax = _modified_tmax(nu, x, tail_cutoff)
    z = 0.5 * jnp.exp(jnp.log(x) + tmax)
    power = nu - int(divide_cosh)
    log_pref = power * jnp.log(2.0 / x) + int(divide_cosh) * jnp.log(2.0)
    slope = max(power - 1.0, 0.0) / z
    bound = jnp.exp(log_pref - z + (power - 1) * jnp.log(z)) / (1.0 - slope)
    return jnp.where((x > 0) & (slope < 1), bound, jnp.inf)


def bessel_kn(nu, x, *, n_nodes=128, tail_cutoff=50.0):
    """K_nu(x), x > 0; nonnegative static nu is sufficient for current uses.

    A continuous cosh integral with a parameter-dependent upper limit is used.
    K_nu(0)=infinity and negative x returns NaN. Node count and tail_cutoff must
    be static under JIT. Derivatives inherit both quadrature and finite-tail
    errors, and must be checked separately from function values.
    """
    return _modified_integral(nu, x, n_nodes=n_nodes, tail_cutoff=tail_cutoff)


def bessel_kn_tail_bound(nu, x, *, tail_cutoff=50.0):
    """Upper bound on K's omitted integral, excluding discretisation error."""
    return _modified_tail_bound(nu, x, tail_cutoff=tail_cutoff)
