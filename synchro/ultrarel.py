"""Ultra-relativistic synchrotron functions F(x), G(x).

In the limit gamma >> 1 the harmonic sum collapses to the standard continuum
result in terms of modified Bessel functions:

    F(x) = x int_x^inf K_{5/3}(xi) dxi,   G(x) = x K_{2/3}(x),

with x = nu / nu_c and nu_c = (3/2) gamma^3 w_B sin(alpha) / (2 pi).

These are used as (i) a fast path for the emissivity in the ultra-relativistic
regime and (ii) a cross-check of the exact harmonic sum in that limit.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .bessel import bessel_kn

_N_F = 256
_xg, _wg = np.polynomial.legendre.leggauss(_N_F)
_XG = jnp.asarray(_xg)
_WG = jnp.asarray(_wg)


def _kn53(x):
    return bessel_kn(5.0 / 3.0, x)


def _kn23(x):
    return bessel_kn(2.0 / 3.0, x)


def G(x):
    """G(x) = x K_{2/3}(x)."""
    return x * _kn23(x)


def F(x):
    """F(x) = x int_x^inf K_{5/3}(xi) dxi.

    Gauss--Legendre in u = ln(xi) over [ln x, ln(x + 40)]; the upper tail is
    ~e^-40 and negligible.
    """
    x = jnp.asarray(x)
    lo = jnp.log(x)
    hi = jnp.log(x + 40.0)
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    u = mid[..., None] + half[..., None] * _XG[None, ...]
    xi = jnp.exp(u)
    integrand = xi * _kn53(xi)  # K_{5/3}(xi) dxi = xi K_{5/3} d(ln xi)
    return (x * jnp.sum(_WG * integrand, axis=-1) * half).reshape(jnp.shape(x))
