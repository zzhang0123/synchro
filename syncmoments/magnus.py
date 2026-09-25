"""First two deterministic Magnus terms for an ordered piecewise-constant path.

For S'=-K S, Omega1=-sum A_i and Omega2=1/2 sum_{j>i}[A_j,A_i],
A_i=K_i ds_i. Commutators are not statistical cumulants. Ensemble averages
require two-position statistics and generally E[exp(Omega)] != exp(E[Omega]).
Truncation is a small-depth approximation; int ||K||_2 ds < pi is a sufficient
convergence condition for the full series, not an error bound for order two.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def commutator(A, B):
    return A @ B - B @ A


def _weighted_slabs(K_s, ds):
    K_s, ds = jnp.asarray(K_s), jnp.asarray(ds)
    if K_s.ndim != 3 or K_s.shape[1:] != (4, 4):
        raise ValueError("K_s must have shape (N,4,4)")
    if ds.ndim != 0 and ds.shape != (K_s.shape[0],):
        raise ValueError("ds must be scalar or shape (N,)")
    return K_s * (ds if ds.ndim == 0 else ds[:, None, None])


def omega1(K_s, ds):
    """-sum_i K_i ds_i, including nonuniform slabs."""
    return -jnp.sum(_weighted_slabs(K_s, ds), axis=0)


def omega2(K_s, ds):
    """Ordered pair sum in O(N) work using a running prefix matrix."""
    A = _weighted_slabs(K_s, ds)
    zero = jnp.zeros((4, 4), dtype=A.dtype)

    def body(state, slab):
        prefix, correction = state
        return (prefix + slab, correction + commutator(slab, prefix)), None

    (_, correction), _ = jax.lax.scan(body, (zero, zero), A)
    return 0.5 * correction


def magnus_S(S0, K_s, ds, order=2):
    """Homogeneous transfer at static truncation order 1 or 2."""
    if order not in (1, 2):
        raise ValueError(
            "order must be 1 or 2; higher Magnus terms are not implemented"
        )
    Om = omega1(K_s, ds)
    if order == 2:
        Om = Om + omega2(K_s, ds)
    return expm(Om) @ S0
