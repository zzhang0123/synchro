"""Polarised radiative transfer along a line of sight (JAX).

Solves  dS/ds = eps - K S,  S = (I, Q, U, V),  with the Mueller matrix

    K = [ aI  aQ  aU  aV ]
        [ aQ  aI  rV -rU ]
        [ aU -rV  aI  rQ ]
        [ aV  rU -rQ  aI ]

where a* are absorption coefficients and r* are Faraday rotation (rV) and
conversion (rQ, rU) coefficients.

Convention: with the rotation block [[0, rV], [-rV, 0]] the complex
polarisation P = Q + i U rotates as  P -> P exp(i rV ds)  over a length ds.
The physical Faraday position-angle rotation is  chi = RM lam^2, and the
Stokes rotation angle is 2 chi, so  rV ds = 2 RM lam^2  (i.e. rV = 2 dchi/ds).

A constant-coefficient slab is solved exactly with the matrix exponential;
a LOS is a chain of such slabs, which correctly handles the non-commutativity
of K(s) (path-ordered).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def mueller_matrix(aI, aQ, aU, aV, rQ, rU, rV):
    """Build the 4x4 Mueller matrix K from absorption and rotation coefficients."""
    return jnp.array([
        [aI, aQ, aU, aV],
        [aQ, aI, rV, -rU],
        [aU, -rV, aI, rQ],
        [aV, rU, -rQ, aI],
    ])


def transfer_slab(S, eps, K, ds):
    """Exact propagation through a constant-K slab of length ``ds``.

    Uses the augmented 5x5 matrix exponential
        y(ds) = exp( [[-K, eps], [0, 0]] ds ) @ [S; 1],   S = y[:4]
    which is well-defined even when K is singular (pure rotation/conversion).

    The source column is normalised before exponentiating.  The map
    eps -> S is linear, so carrying the scale in the input vector instead of
    in the matrix is exact; without it the augmented matrix mixes entries of
    order ``|K| ds`` with entries of order ``|eps| ds``, and the
    scaling-and-squaring in ``expm`` overflows to NaN when those differ by
    many orders of magnitude (an optically thick slab with a bright source).
    """
    dtype = jnp.result_type(S, K, eps)
    scale = jnp.linalg.norm(eps) * ds
    scale = jnp.where(scale > 0.0, scale, 1.0).astype(dtype)
    M = jnp.zeros((5, 5), dtype=dtype)
    M = M.at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(eps * ds / scale)
    y0 = jnp.concatenate([S, scale[None]])
    return (expm(M) @ y0)[:4]


def transfer_los(S0, eps_s, K_s, ds):
    """Chain of constant-coefficient slabs along the LOS.

    ``eps_s``: (N, 4) emissivity per slab; ``K_s``: (N, 4, 4) Mueller matrices;
    ``ds``: scalar slab length (or per-slab (N,)).  Returns final Stokes (4,).
    """
    if jnp.ndim(ds) == 0:
        def body(S, x):
            eps, K = x
            return transfer_slab(S, eps, K, ds), None
        S, _ = jax.lax.scan(body, S0, (eps_s, K_s))
    else:
        def body(S, x):
            eps, K, d = x
            return transfer_slab(S, eps, K, d), None
        S, _ = jax.lax.scan(body, S0, (eps_s, K_s, ds))
    return S


def faraday_rotation_matrix(theta):
    """Stokes-space rotation of (Q, U) by angle ``theta``: P -> P exp(i theta)."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def conversion_matrix(psi):
    """Faraday conversion: rotation of the (Q, V) plane by angle ``psi``.

    Q' = Q cos(psi) + V sin(psi),  V' = -Q sin(psi) + V cos(psi).
    Converts linear Q into circular V (and back).
    """
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, 0.0, s],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -s, 0.0, c],
    ])
