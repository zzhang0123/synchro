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

The matrix exponential is the analytic constant-coefficient solution;
its implementation has finite-precision error. A LOS is a chain of slabs
in traversal order. Approximating a varying medium by slabs incurs a separate
discretisation error; refine the path and propagate that discrepancy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm


def mueller_matrix(aI, aQ, aU, aV, rQ, rU, rV):
    """Build the 4x4 Mueller matrix K from absorption and rotation coefficients."""
    return jnp.array(
        [
            [aI, aQ, aU, aV],
            [aQ, aI, rV, -rU],
            [aU, -rV, aI, rQ],
            [aV, rU, -rQ, aI],
        ]
    )


def transfer_slab(S, eps, K, ds, *, max_squarings=32):
    """Constant-coefficient propagation of shape-(4,) Stokes through ``ds``.

    Uses the augmented 5x5 matrix exponential::

        y(ds) = exp( [[-K, eps], [0, 0]] ds ) @ [S; 1],   S = y[:4]

    which is well-defined even when K is singular (pure rotation/conversion).

    The source column is normalised before exponentiating.  The map
    eps -> S is linear, so carrying the scale in the input vector instead of
    in the matrix is exact; without it the augmented matrix mixes entries of
    order ``|K| ds`` with entries of order ``|eps| ds``, and the
    scaling-and-squaring in ``expm`` overflows to NaN when those differ by
    many orders of magnitude (an optically thick slab with a bright source).

    ``max_squarings`` is static under JIT. The default permits strongly
    absorbing slabs beyond the library's default limit; a regression checks
    tau=1e6. This does not certify arbitrary large or ill-conditioned K.
    Exceeding the configured exponential budget returns NaN; extreme gain can
    overflow. Check finiteness and refine/compare the propagation as needed.
    """
    S, eps, K, ds = map(jnp.asarray, (S, eps, K, ds))
    if S.shape != (4,) or eps.shape != (4,) or K.shape != (4, 4) or ds.ndim != 0:
        raise ValueError("shape contract: S and eps (4,), K (4,4), scalar ds")
    dtype = jnp.result_type(S, K, eps, ds, 1.0)
    # A coordinate rescaling only: freezing it preserves derivatives of the
    # analytic map, and avoids the undefined derivative of ||eps|| at zero.
    scale = jax.lax.stop_gradient(jnp.maximum(1.0, jnp.max(jnp.abs(eps * ds))))
    scale = scale.astype(dtype)
    M = jnp.zeros((5, 5), dtype=dtype)
    M = M.at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(eps * ds / scale)
    y0 = jnp.concatenate([S, scale[None]])
    return (expm(M, max_squarings=max_squarings) @ y0)[:4]


def transfer_los(S0, eps_s, K_s, ds, *, max_squarings=32):
    """Chain of constant-coefficient slabs along the LOS.

    ``eps_s``: (N, 4) emissivity per slab; ``K_s``: (N, 4, 4) Mueller matrices;
    ``ds``: scalar slab length (or per-slab (N,)).  Returns final Stokes (4,).
    """
    S0, eps_s, K_s, ds = map(jnp.asarray, (S0, eps_s, K_s, ds))
    if (
        S0.shape != (4,)
        or eps_s.ndim != 2
        or eps_s.shape[1:] != (4,)
        or K_s.shape != (eps_s.shape[0], 4, 4)
        or (ds.ndim != 0 and ds.shape != (eps_s.shape[0],))
    ):
        raise ValueError(
            "shape contract: S0 (4,), eps_s (N,4), K_s (N,4,4), ds scalar or (N,)"
        )
    if eps_s.shape[0] == 0:
        return S0
    S0 = S0.astype(jnp.result_type(S0, eps_s, K_s, ds, 1.0))
    if ds.ndim == 0:

        def body(S, x):
            eps, K = x
            return transfer_slab(S, eps, K, ds, max_squarings=max_squarings), None

        S, _ = jax.lax.scan(body, S0, (eps_s, K_s))
    else:

        def body(S, x):
            eps, K, d = x
            return transfer_slab(S, eps, K, d, max_squarings=max_squarings), None

        S, _ = jax.lax.scan(body, S0, (eps_s, K_s, ds))
    return S


def optical_depth_factor(tau):
    """(1-exp(-tau))/tau with value and derivatives continued through zero.

    A fourth-degree series is used for ``|tau| < 1e-4`` (next term ``<= 1.4e-23``).
    Negative optical depths describe gain; overflow for extreme gain remains
    a numerical-domain limit, not a physical saturation model.
    """
    tau = jnp.asarray(tau)
    small = jnp.abs(tau) < 1e-4
    regular = -jnp.expm1(-tau) / jnp.where(small, 1.0, tau)
    series = 1 - tau / 2 + tau**2 / 6 - tau**3 / 24 + tau**4 / 120
    return jnp.where(small, series, regular)


def faraday_rotation_matrix(theta):
    """Stokes-space rotation of (Q, U) by angle ``theta``: P -> P exp(i theta)."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def conversion_matrix(psi):
    """Faraday conversion: rotation of the (Q, V) plane by angle ``psi``.

    Q' = Q cos(psi) + V sin(psi),  V' = -Q sin(psi) + V cos(psi).
    Converts linear Q into circular V (and back).
    """
    c, s = jnp.cos(psi), jnp.sin(psi)
    return jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, 0.0, s],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -s, 0.0, c],
        ]
    )
