"""Derivative spectra via JAX autodiff.

The building blocks of the moment expansion are the derivatives of the
single-particle Stokes parameters w.r.t. the physical parameters, evaluated at
a reference point.  Because the exact Schott expressions are implemented with
pure JAX primitives, these derivatives are exact (no finite differences) and
automatically differentiable to any order.

Parameter vector (natural polarisation basis, U = 0)::

    p = (gamma, alpha, theta)

``B`` is omitted: it enters S_n only as an overall factor w_B^2 ~ B^2, so
``d S/dB = 2 S/B`` and ``d^2 S/dB^2 = 2 S/B^2`` exactly (see expansion).
The azimuth ``phi`` enters only through the sky-plane rotation
``Q -> Q cos 2phi, U -> Q sin 2phi`` and is handled by its own moments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .stokes import _harmonic

PARAMS = ("gamma", "alpha", "theta")
N_PARAM = 3
STOKES = ("I", "Q", "V")


def _stokes_stack(n, p):
    """p = (gamma, alpha, theta) -> (I, Q, V) array."""
    I, Q, V = _harmonic(n, p[0], p[1], p[2])
    return jnp.stack([I, Q, V])


def derivative_spectra(n: int, gamma0: float, alpha0: float, theta0: float):
    """Value, gradient and Hessian of (I, Q, V)_n at the reference point.

    Returns
    -------
    val : (3,)    S_s(p0)
    grad : (3, 3) grad[s, i] = d S_s / d p_i
    hess : (3, 3, 3) hess[s, i, j] = d^2 S_s / d p_i d p_j
    """
    p0 = jnp.array([gamma0, alpha0, theta0])
    f = lambda p: _stokes_stack(n, p)  # noqa: E731
    val = f(p0)
    grad = jax.jacfwd(f)(p0)
    hess = jnp.stack([jax.hessian(lambda p: f(p)[s])(p0) for s in range(3)])
    return val, grad, hess
