"""Reverse mode through nested ``vmap`` of the transfer functions.

Found by the AD-transform matrix (``tests/test_ad_transform_matrix.py``, cell
``grad_nested_vmap``): ``grad`` of a sum over ``vmap(vmap(transfer_slab))``
in ``K`` or ``ds`` raised ``NotImplementedError: Transpose rule ... for
'stop_gradient'`` in round 5 of 0.3.0, for ``transfer_slab``,
``transfer_los`` and ``moment_driven_slab(_cgs)``; a single ``vmap`` worked.
Cause: the batching rule of ``_expm.any_member`` reduced over its own batch
axis with a plain ``jnp.any``, whose result is still batched along an outer
``vmap``; there the ``lax.cond`` on it became a select, which wraps the tangent
rule's operands in ``stop_gradient``. The rules of ``any_member`` and
``batched_only`` now re-enter themselves, so every enclosing ``vmap`` level
reduces (``any_member``) or selects per member (``batched_only``).

Checks: the ``cond`` stays a ``cond`` under two and three ``vmap`` levels;
the two helpers' values under every batching pattern; ``grad`` and
``hessian`` of sums over nested ``vmap`` against forward-mode autodiff of the
independent reference (``_ad_matrix_cases.slab_ref``) at 1e-12, with members
that need 0 to 9 squarings in one batch, and per member equal to the
unbatched evaluation. ``transfer_los`` mixes a member at ``max|eps ds| ~ 1e306``
(derivatives of the value from ``_companion``) with order-1 members: Stokes
vectors bit-identical to the unbatched ones, the contracted loss within 2 ulp
(batched dot order) and gradients within 1e-13.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401  (enables x64)
from _ad_matrix_cases import K1, K2, K3, slab_ref
from syncmoments._expm import any_member, batched_only
from syncmoments.transfer import _uses_companion, transfer_los, transfer_slab

S = jnp.array([1.0, 0.2, -0.1, 0.05])
EPS = jnp.array([2.0, 0.3, -0.2, 0.1])
C = jnp.array([0.3, -1.0, 0.5, 0.2])
# |K ds|_1 from 0.4 to 1.5e3: 0 to 9 squarings of the Pade-13 exponential
SCALES = jnp.array([0.5, 1.0, 7.0, 40.0, 0.2, 3.0, 15.0, 250.0]).reshape(2, 2, 2)
KB = K1 * SCALES[..., None, None]
DB = 0.7 * jnp.cos(jnp.arange(8.0)).reshape(2, 2, 2) ** 2 + 0.1
WRAPS = {"eager": lambda f: f, "jit": jax.jit}


def _rel(got, want):
    got, want = np.asarray(got), np.asarray(want)
    assert np.all(np.isfinite(got))
    return np.max(np.abs(got - want)) / np.max(np.abs(want))


def _vmaps(f, depth):
    for _ in range(depth):
        f = jax.vmap(f)
    return f


def _branchy(flag, x):
    return jax.lax.cond(any_member(flag), lambda y: 2 * y, lambda y: y, x)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_cond_on_any_member_stays_a_branch_under_nested_vmap(depth):
    shape = (2,) * depth
    flags, xs = jnp.zeros(shape, bool).ravel().at[0].set(True).reshape(shape), jnp.ones(
        shape
    )
    jaxpr = str(jax.make_jaxpr(_vmaps(_branchy, depth))(flags, xs))
    assert "cond[" in jaxpr and "select_n" not in jaxpr
    np.testing.assert_array_equal(_vmaps(_branchy, depth)(flags, xs), 2 * xs)
    np.testing.assert_array_equal(
        _vmaps(_branchy, depth)(jnp.zeros_like(flags), xs), xs
    )


def test_helper_values_under_every_batching_pattern():
    flags = jnp.array([[True, False, False], [False, False, False]])
    both = jax.vmap(jax.vmap(any_member))(flags)
    np.testing.assert_array_equal(both, jnp.ones((2, 3), bool))
    np.testing.assert_array_equal(jax.vmap(jax.vmap(batched_only))(flags), flags)
    outer = jax.vmap(jax.vmap(batched_only, in_axes=None, axis_size=3))(flags[:, 0])
    np.testing.assert_array_equal(outer, jnp.repeat(flags[:, :1], 3, axis=1))
    inner = jax.vmap(jax.vmap(batched_only), in_axes=None, axis_size=2)(flags[0])
    np.testing.assert_array_equal(inner, jnp.stack([flags[0]] * 2))
    assert bool(any_member(jnp.asarray(False))) is False
    assert bool(batched_only(jnp.asarray(False))) is True


def _slab_loss(K, d):
    return C @ transfer_slab(S, EPS, K, d)


def _ref_loss(K, d):
    return C @ slab_ref(S, EPS, K, d)


def _nested_sum(loss):
    return lambda K, d: jnp.sum(_vmaps(loss, 3)(K, d))


def _per_member(fun):
    """``fun`` evaluated member by member (no ``vmap``), stacked as ``(2, 2, 2, ...)``."""
    out = [fun(KB.reshape(8, 4, 4)[i], DB.ravel()[i]) for i in range(8)]
    return jax.tree_util.tree_map(
        lambda *a: jnp.stack(a).reshape(2, 2, 2, *a[0].shape), *out
    )


@pytest.mark.parametrize("wrap", WRAPS)
def test_grad_of_a_sum_over_nested_vmap_of_transfer_slab(wrap):
    got = WRAPS[wrap](jax.grad(_nested_sum(_slab_loss), (0, 1)))(KB, DB)
    want = _per_member(jax.jacfwd(_ref_loss, (0, 1)))
    unbatched = _per_member(jax.grad(_slab_loss, (0, 1)))
    for g, w, u in zip(got, want, unbatched):
        assert _rel(g, w) < 1e-12
        assert _rel(g, u) < 1e-15


@pytest.mark.parametrize("wrap", WRAPS)
def test_hessian_of_a_sum_over_nested_vmap_is_block_diagonal(wrap):
    Kb, db = KB[0], DB[0]
    H = WRAPS[wrap](jax.hessian(lambda K: jnp.sum(_vmaps(_slab_loss, 2)(K, db))))(Kb)
    for i, j in np.ndindex(2, 2):
        block = H[i, j][..., i, j, :, :]
        want = jax.jacfwd(jax.jacfwd(_ref_loss))(Kb[i, j], db[i, j])
        assert _rel(block, want) < 1e-12
        off = np.asarray(H[i, j]).copy()
        off[..., i, j, :, :] = 0.0
        assert not np.any(off)


def test_one_level_batched_inside_or_outside_another_vmap():
    """``K`` batched only in the outer level (inner over ``S``) and vice versa."""
    Ss = jnp.stack([S, -2 * S, 0.5 * S])

    def outer_K(Kb):
        rows = jax.vmap(
            lambda K: jax.vmap(lambda s: C @ transfer_slab(s, EPS, K, 0.7))(Ss)
        )
        return jnp.sum(rows(Kb))

    def inner_K(Kb):
        rows = jax.vmap(
            lambda s: jax.vmap(lambda K: C @ transfer_slab(s, EPS, K, 0.7))(Kb)
        )
        return jnp.sum(rows(Ss))

    Kb = KB[0, 0]
    ref = lambda K: sum(C @ slab_ref(s, EPS, K, 0.7) for s in Ss)  # noqa: E731
    want = jnp.stack([jax.jacfwd(ref)(K) for K in Kb])
    for loss in (outer_K, inner_K):
        assert _rel(jax.grad(loss)(Kb), want) < 1e-12


def test_nested_vmap_of_transfer_los_mixing_the_companion_route():
    """One member at ``max|eps ds| ~ 1e306`` takes ``_companion`` for the batch;
    ``K_s`` and the source scale are batched at both levels.

    The Stokes vectors under two ``vmap`` levels are compared byte for byte
    with the unbatched ones. The scalar ``C @ S`` of ``value_and_grad`` is
    compared within 2 ulp: XLA may evaluate that 4-term dot in a different
    order (or with a fused multiply-add) once it is batched, which is a
    rounding of the contraction, not of ``transfer_los``. JAX 0.10.0 gives 1
    ulp on members (0, 1) and (1, 1); JAX 0.10.2 gives 0.
    """
    eps_s = jnp.stack([EPS, -0.5 * EPS, 0.2 * EPS])
    K_s = jnp.stack([K1, K2, K3])
    K_n = K_s * jnp.array([[1.0, 3.0], [0.5, 20.0]])[..., None, None, None]
    a_n = jnp.array([[1.0, 1e306], [2.0, 0.5]])
    ds = jnp.array([0.7, 0.4, 1.1])

    def stokes(K, a):
        return transfer_los(S, eps_s * a, K, ds)

    def loss(K, a):
        return C @ stokes(K, a)

    batched_stokes = jax.vmap(jax.vmap(stokes))(K_n, a_n)
    nested = jax.vmap(jax.vmap(jax.value_and_grad(loss, (0, 1))))
    value, (gK, ga) = nested(K_n, a_n)
    total = jax.grad(lambda K, a: jnp.sum(jax.vmap(jax.vmap(loss))(K, a)), (0, 1))
    sK, sa = total(K_n, a_n)
    for i, j in np.ndindex(2, 2):
        single = stokes(K_n[i, j], a_n[i, j])
        assert (
            np.asarray(batched_stokes[i, j]).tobytes() == np.asarray(single).tobytes()
        )
        v, (uK, ua) = jax.value_and_grad(loss, (0, 1))(K_n[i, j], a_n[i, j])
        v = float(v)
        assert abs(float(value[i, j]) - v) <= 2 * np.spacing(abs(v))
        assert _rel(gK[i, j], uK) < 1e-13 and _rel(ga[i, j], ua) < 1e-13
        assert _rel(sK[i, j], uK) < 1e-13 and _rel(sa[i, j], ua) < 1e-13
    routes = [
        bool(jnp.any(jax.vmap(_uses_companion)(eps_s * a_n[i, j], K_n[i, j], ds)))
        for i, j in np.ndindex(2, 2)
    ]
    assert routes == [False, True, False, False]
