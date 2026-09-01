"""Magnus expansion of the polarised transfer (path-ordered exponential).

For the homogeneous transfer dS/ds = -K(s) S the solution is the path-ordered
exponential  S(L) = P exp(-int_0^L K ds) S(0) = exp(Omega(L)) S(0), with the
Magnus series

    Omega1 = -int_0^L K ds
    Omega2 = (1/2) int_0^L ds1 int_0^s1 ds2 [K(s1), K(s2)]
    Omega3 = ... (nested commutators)

The moment-expansion content is transparent here:
  * Omega1 ~ K_bar L, i.e. the LOS-AVERAGED coefficients = first moments of
    alpha, rho_V, rho_Q along the sightline;
  * Omega2 is built from the commutator [K(s1), K(s2)], whose LOS average
    involves the cross-cumulants <rho_V rho_Q> - <rho_Q rho_V> etc. (the
    non-commutativity of rotation and conversion).

This module provides the first two terms and validates them against the exact
slab chain (synchro.transfer.transfer_los with eps=0).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import expm


def commutator(A, B):
    return A @ B - B @ A


def omega1(K_s, ds):
    """Omega1 = -int K ds (array of (N,4,4) Mueller matrices, slab length ds)."""
    return -jnp.sum(K_s, axis=0) * ds


def omega2(K_s, ds):
    """Omega2 = (1/2) sum_{j>i} [K_j, K_i] ds^2 (ordered double integral)."""
    N = K_s.shape[0]
    out = jnp.zeros((4, 4), dtype=K_s.dtype)
    for j in range(1, N):
        for i in range(j):
            out = out + commutator(K_s[j], K_s[i])
    return 0.5 * out * ds**2


def magnus_S(S0, K_s, ds, order=2):
    """S(L) = exp(Omega1 [+ Omega2]) S0 for the homogeneous transfer."""
    Om = omega1(K_s, ds)
    if order >= 2:
        Om = Om + omega2(K_s, ds)
    return expm(Om) @ S0
