"""Ultra-relativistic continuum kernels F and G with explicit tail control.

F(x)=x integral_x^infinity K_(5/3)(u) du and G(x)=x K_(2/3)(x).
Interchanging the two positive integrals gives the single-integral form
F=x integral_0^infinity exp(-x cosh t) cosh(5t/3)/cosh(t) dt. This avoids nested
quadrature and the former switched small-x approximation. These are continuum
kernels; agreement with their integral definitions does not bound the physical
error of replacing finite-gamma, directional harmonic radiation by a continuum.
"""

from __future__ import annotations

import jax.numpy as jnp

from .bessel import _modified_integral, _modified_tail_bound, bessel_kn


def G(x, *, n_nodes=128, tail_cutoff=50.0):
    """G(x)=x K_(2/3)(x); G(0)=0, autodiff is intended for x > 0."""
    x = jnp.asarray(x)
    safe_x = jnp.where(x > 0, x, 1.0)
    value = safe_x * bessel_kn(
        2.0 / 3.0, safe_x, n_nodes=n_nodes, tail_cutoff=tail_cutoff
    )
    return jnp.where(x > 0, value, jnp.where(x == 0, 0.0, jnp.nan))


def F(x, *, n_nodes=128, tail_cutoff=50.0):
    """F(x); node count/cutoff are static under JIT. F(0)=0.

    ``F_tail_bound`` controls the finite upper limit only. Compare node counts
    and independent references to assess quadrature and floating-point error.
    The continuum kernel has a divergent slope at zero, so differentiability
    claims and tests concern positive arguments.
    """
    x = jnp.asarray(x)
    safe_x = jnp.where(x > 0, x, 1.0)
    value = safe_x * _modified_integral(
        5.0 / 3.0, safe_x, divide_cosh=True, n_nodes=n_nodes, tail_cutoff=tail_cutoff
    )
    return jnp.where(x > 0, value, jnp.where(x == 0, 0.0, jnp.nan))


def F_tail_bound(x, *, tail_cutoff=50.0):
    """Bound F's omitted positive-t tail; excludes quadrature/roundoff errors."""
    return jnp.asarray(x) * _modified_tail_bound(
        5.0 / 3.0, x, divide_cosh=True, tail_cutoff=tail_cutoff
    )
