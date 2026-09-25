"""Reverse-mode derivatives for small sources, and the ``max_squarings`` argument.

The propagators' Pade-13 exponential solves with ``V - U ~ b_0 I``. With the
textbook ``b_0 = 6.5e16`` the transpose of that solve divided cotangents by
``b_0``, so float32 reverse-mode derivatives in ``K`` flushed to zero for
sources below about 1e-15 (``ds = 20``); the coefficients are therefore held
at an exact power-of-two scale. ``max_squarings`` is static: numpy integers
and integral floats are accepted as in 0.2.0.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401  (enables float64)
from syncmoments.transfer import mueller_matrix, transfer_los, transfer_slab

K1 = np.asarray(mueller_matrix(1.2, 0.3, -0.2, 0.1, 0.4, -0.3, 2.0))
S1 = np.array([1.0, 0.3, -0.2, 0.1])
EPS0 = np.zeros(4)


def _reference(ds):
    """d out[1] / dK at unit source scale, forward mode in float64."""

    def f(k):
        return transfer_slab(jnp.asarray(S1), jnp.asarray(EPS0), k, ds)[1]

    return np.asarray(jax.jacfwd(f)(jnp.asarray(K1)))


CASES = (
    [(np.float32, 20.0, s) for s in (1.0, 1e-10, 1e-15, 1e-20)]
    + [(np.float32, 1.0, s) for s in (1e-18, 1e-20, 1e-25)]
    + [(np.float64, 1.0, s) for s in (1e-250, 1e-280, 1e-290)]
)


@pytest.mark.parametrize("dtype, ds, scale", CASES)
@pytest.mark.parametrize("mode", ["grad", "jacfwd"])
def test_small_sources_keep_reverse_and_forward_derivatives(dtype, ds, scale, mode):
    S = jnp.asarray(S1 * scale, dtype)
    K, eps, d = jnp.asarray(K1, dtype), jnp.asarray(EPS0, dtype), dtype(ds)
    f = lambda k: transfer_slab(S, eps, k, d)[1]  # noqa: E731
    got = (jax.grad if mode == "grad" else jax.jacfwd)(f)(K)
    got = np.asarray(got, np.float64) / scale
    ref = _reference(ds)
    tol = 1e-4 if dtype == np.float32 else 1e-12
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - ref)) <= tol * np.max(np.abs(ref))


BUDGETS = [32, np.int64(32), np.int32(40), 32.0, np.float64(40.0)]


@pytest.mark.parametrize("budget", BUDGETS, ids=repr)
def test_max_squarings_accepts_integral_scalars(budget):
    S, eps, K = jnp.ones(4), jnp.ones(4), jnp.asarray(K1)
    expected = transfer_slab(S, eps, K, 1.0)
    np.testing.assert_array_equal(
        transfer_slab(S, eps, K, 1.0, max_squarings=budget), expected
    )
    jitted = jax.jit(lambda k: transfer_slab(S, eps, k, 1.0, max_squarings=budget))(K)
    np.testing.assert_array_equal(jitted, expected)
    grad = jax.grad(lambda k: transfer_slab(S, eps, k, 1.0, max_squarings=budget)[0])(K)
    assert np.all(np.isfinite(np.asarray(grad)))
    los = transfer_los(
        S, jnp.ones((3, 4)), jnp.stack([K] * 3), 1.0, max_squarings=budget
    )
    np.testing.assert_array_equal(
        los, transfer_los(S, jnp.ones((3, 4)), jnp.stack([K] * 3), 1.0)
    )


@pytest.mark.parametrize("budget", [32.5, -1, "32", None])
def test_max_squarings_rejects_non_integral_or_negative(budget):
    S, eps, K = jnp.ones(4), jnp.ones(4), jnp.asarray(K1)
    with pytest.raises((TypeError, ValueError), match="max_squarings"):
        transfer_slab(S, eps, K, 1.0, max_squarings=budget)
    with pytest.raises((TypeError, ValueError), match="max_squarings"):
        transfer_los(S, jnp.ones((2, 4)), jnp.stack([K] * 2), 1.0, max_squarings=budget)


def test_refusal_type_is_unchanged_by_the_budget_coercion():
    K = jnp.asarray(K1) * 1e12
    with pytest.raises(eqx.EquinoxRuntimeError, match="max_squarings"):
        transfer_slab(jnp.ones(4), jnp.ones(4), K, 1.0, max_squarings=np.int64(3))
