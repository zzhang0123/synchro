"""Matrix exponential whose branches survive ``vmap`` under reverse mode.

``jax.scipy.linalg.expm`` picks its Pade degree with ``lax.switch`` and each
squaring with ``lax.cond``, on values of the operand. Under ``vmap`` with a
batched operand JAX turns both into selects that wrap every operand in
``stop_gradient``. When the exponential is differentiated inside a
``custom_jvp`` rule (``transfer._tangent``), those operands include the
rule's tangents, and ``stop_gradient`` of a tangent cannot be transposed, so
``grad`` of a sum over ``vmap`` in ``K`` would raise ``NotImplementedError``.
Here the degree is fixed (Pade 13) and a squaring runs while any member of the
batch needs it, selected per member.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# Higham (2005) Pade-13 coefficients b_0..b_13 and theta_13 (float64). The
# approximant P/Q is unchanged by a common factor, so the coefficients are held
# at the exact scale 2^-40: with b_0 = 6.5e16, V - U ~ b_0 I and the transpose
# of the solve divided cotangents by b_0, which flushed float32 reverse-mode
# derivatives for small sources (tests/test_transfer_scale_and_budget.py).
# Dividing by b_0 fully overflows rev-over-fwd Hessians near 1e306.
_PADE13 = tuple(
    b * 2.0**-40
    for b in (64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
              1187353796428800.0, 129060195264000.0, 10559470521600.0,
              670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
              960960.0, 16380.0, 182.0, 1.0)
)  # fmt: skip
_THETA13 = 5.371920351148152


@partial(jax.jit, static_argnames="max_squarings")
def expm_pade13(A, max_squarings):
    """``exp(A)`` by Pade 13 on ``A 2^-n``, ``n = max(0, ceil(log2(|A|_1 / theta_13)))``
    (Higham 2005: backward error below the unit roundoff).

    ``jax.scipy.linalg.expm`` takes ``floor`` and so evaluates Pade 13 up to
    ``2 theta_13``; for a rotation-dominated ``A`` just below ``2^(m+1) theta_13``
    that costs relative errors from 1e-10 (m = 0) to 1e-1 (m = 31) where
    ``ceil`` stays at the conditioning limit (``tests/test_transfer_expm_boundaries.py``).
    Where ``ceil`` exceeds ``max_squarings`` but ``floor`` does not, ``floor``
    is used, so the result is NaN exactly where ``expm``'s is in float64 (in
    float32 ``expm`` used Pade 7, ``theta_7 = 3.93``, and a lower NaN bound).
    Used for the slab value (5x5) and its propagators (8x8, ``propagators``).
    Compiled, as ``expm`` is: op by op, XLA CPU rounded it an ulp differently."""
    norm = jax.lax.stop_gradient(jnp.max(jnp.sum(jnp.abs(A), axis=0)))
    ratio = jnp.log2(norm / _THETA13)
    low, high = jnp.maximum(0, jnp.floor(ratio)), jnp.maximum(0, jnp.ceil(ratio))
    return expm_squared(A, jnp.where(high > max_squarings, low, high), max_squarings)


@partial(jax.jit, static_argnames="max_squarings")
def expm_squared(A, n, max_squarings):
    """Pade-13 of ``A / 2^n`` squared ``n`` times (NaN if ``n > max_squarings``).
    A squaring step runs while any member needs it (``any_member``) and is
    applied per member (``batched_only``)."""
    dot = partial(jnp.matmul, precision=jax.lax.Precision.HIGHEST)
    A = A / 2 ** n.astype(A.dtype)
    b, ident = _PADE13, jnp.eye(A.shape[0], dtype=A.dtype)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    U = dot(A6, b[13] * A6 + b[11] * A4 + b[9] * A2)
    U = dot(A, U + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = dot(A6, b[12] * A6 + b[10] * A4 + b[8] * A2)
    V = V + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    R = jnp.linalg.solve(V - U, V + U)

    def squaring(R, i):
        def square(R):
            return jnp.where(batched_only(i < n), dot(R, R), R)

        return jax.lax.cond(any_member(i < n), square, lambda R: R, R), None

    steps = jnp.arange(max_squarings, dtype=n.dtype)
    R = jax.lax.scan(squaring, R, steps)[0]
    return jnp.where(n > max_squarings, jnp.nan, R)


def propagators(K, ds, max_squarings):
    """``(Phi, G) = (e^{-K ds}, int_0^ds e^{-K t} dt)`` from one 8x8 exponential.

    The identity block (not ``I ds``) keeps the norm at ``max(|K ds|, 1)``, at
    most that of the slab value's 5x5 exponential (same ``expm_pade13``), so
    they exceed ``max_squarings`` only where the value is refused. There is no
    branch on a member's value, so a ``jax.jvp`` of it inside a ``custom_jvp``
    rule transposes under a batched ``vmap``."""
    A = jnp.zeros((8, 8), dtype=K.dtype)
    A = A.at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4, dtype=K.dtype))
    E = expm_pade13(A, max_squarings)
    return E[:4, :4], E[:4, 4:] * ds


@jax.custom_batching.custom_vmap
def any_member(flag):
    """``flag``; under ``vmap`` its ``any`` over the batch, unbatched, so that a
    ``lax.cond`` on it stays a branch instead of evaluating both sides. The rule
    re-enters ``any_member``, so an enclosing ``vmap`` reduces its axis too (a
    plain ``jnp.any`` stayed batched there, and reverse mode through nested
    ``vmap`` hit ``stop_gradient`` in the select that replaced the ``cond``)."""
    return flag


@any_member.def_vmap
def _any_member_vmap(axis_size, in_batched, flag):
    return any_member(jnp.any(flag, axis=0) if in_batched[0] else flag), False


@jax.custom_batching.custom_vmap
def batched_only(flag):
    """``True``; under ``vmap`` with a batched ``flag``, ``flag``. In a branch
    taken on ``any_member(flag)`` it selects the members per step, and folds
    to nothing when unbatched (a ``where`` on ``flag`` cost 1.4x in reverse).
    Unbatched at one level, it re-enters itself for an enclosing ``vmap``."""
    return jnp.ones_like(flag)


@batched_only.def_vmap
def _batched_only_vmap(axis_size, in_batched, flag):
    return (flag if in_batched[0] else batched_only(flag)), in_batched[0]
