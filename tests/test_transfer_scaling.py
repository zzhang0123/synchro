"""Source-column scaling of ``transfer_slab`` near the float64 overflow limit.

``transfer_slab`` exponentiates ``[[-K ds, eps ds / s], [0, 0]]`` and carries
the scale ``s`` in the input vector ``[S; s]``. v0.2.0 used the data-dependent
``s = max(1, max|eps ds|)``. On jax 0.10.2 CPU an eager ``x / s`` is evaluated
as ``x * (1 / s)``; for ``s > 2^1022`` the reciprocal is subnormal and flushed
to zero, so the source column vanished and the slab returned all zeros (a
finite, silently wrong result). The scale is now an exact power of two
``2^k``, ``k = floor(log2 max|eps ds|) + 1`` clipped to ``[0, 1022]``, so that
``2^k`` and ``2^-k`` are both normal; no reciprocal is ever formed. v0.2.0
also lost the forward source derivative at 1.79e308 (``1 / 1e308`` is
subnormal), in DEV too. Since 0.3.0 derivatives do not pass through the scaled
column (``tests/test_transfer_derivatives.py``).

Oracles: with ``K = 0`` the augmented matrix is nilpotent and the exact result
is ``S + eps ds``; with ``K = a I`` it is ``S e^{-a ds} + eps (1 - e^{-a ds}) / a``.
Both are evaluated in NumPy, which does not flush subnormals.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import transfer_slab

# max|eps ds| in {4e307, 1e308, 1.79e308}; the last is within 0.5% of float64 max.
SOURCES = (
    [4e307, 2e307, 0.0, 0.0],
    [1e308, 5e307, 1e307, 0.0],
    [1.79e308, 0.0, 0.0, 0.0],
    [4.6e307, -3e307, 0.0, 7.0],
)
WRAPS = {"eager": lambda f: f, "jit": jax.jit}


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("eps", SOURCES)
def test_transparent_slab_near_overflow_returns_the_source(eps, wrap):
    """``K = 0``: ``S = S0 + eps ds``. The scaling is exact (powers of two); the
    Pade evaluation of the nilpotent exponential adds a few ulp (measured 1.2 ulp)."""
    eps = np.asarray(eps)
    got = WRAPS[wrap](transfer_slab)(jnp.zeros(4), eps, jnp.zeros((4, 4)), 1.0)
    assert_allclose(np.asarray(got), eps, rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("scale", (4e307, 1e308, 1.79e308))
def test_absorbing_slab_near_overflow_matches_the_analytic_solution(scale, wrap):
    """``K = I``, ``ds = 1``: ``S = eps (1 - e^{-1})`` (6.32e307 at eps = 1e308)."""
    eps = np.array([1.0, 0.5, -0.25, 0.0]) * scale
    got = WRAPS[wrap](transfer_slab)(jnp.zeros(4), eps, jnp.eye(4), 1.0)
    assert_allclose(np.asarray(got), eps * -np.expm1(-1.0), rtol=4e-15, atol=0.0)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
def test_source_entries_below_the_scaled_normal_range_are_dropped(wrap):
    """Documented limit: an entry is scaled by ``2^-k`` (``k = 1022`` here), so
    entries below ``2^(k - 1022) = 1`` become subnormal and XLA CPU flushes them.
    The absolute error is below ``2^(k - 1022) <= max|eps ds| 2^-1021``; v0.2.0
    flushed the same entries (``0.5 / 4.6e307`` is subnormal)."""
    eps = np.array([4.6e307, -3e307, 0.5, 7.0])
    got = WRAPS[wrap](transfer_slab)(jnp.zeros(4), eps, jnp.zeros((4, 4)), 1.0)
    assert_allclose(np.asarray(got), eps, rtol=4e-16, atol=1.0)
    assert np.asarray(got)[2] == 0.0 and abs(np.asarray(got)[3] - 7.0) < 1.0


def test_large_source_scale_split_between_eps_and_ds():
    """``max|eps ds|`` near ``2^1023`` reached through ``ds``: same exact result."""
    eps = np.array([1e300, 0.0, 3e299, 0.0])
    ds = 1.5e8  # eps ds = 1.5e308
    for wrap in WRAPS.values():
        got = wrap(transfer_slab)(jnp.zeros(4), eps, jnp.zeros((4, 4)), ds)
        assert_allclose(np.asarray(got), eps * ds, rtol=4e-16, atol=0.0)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("a", (0.0, 2.0, 50.0))
@pytest.mark.parametrize("scale", (1e307, 1e308, 1.79e308))
def test_source_derivatives_near_overflow_are_the_analytic_ones(scale, a, wrap):
    """``dS/d eps = (1 - e^{-a ds}) / a I`` (``I`` at ``a = 0``), forward and
    reverse, to 1.79e308.

    Up to round 2 forward-mode tangents were carried at ``2^-k`` through the
    scaled column and flushed to zero on XLA CPU from ``max|eps ds| ~ 2e307``
    (``a = 2``), ``1e307`` (``a = 50``) and ``4e307`` (``a = 0``), as in v0.2.0. The 0.3.0 derivative
    rule gives ``d out / d eps = G`` without the scaled column."""
    eps = jnp.array([scale, 0.0, 0.0, 0.0])

    def f(e):
        return transfer_slab(jnp.zeros(4), e, a * jnp.eye(4), 1.0)

    expected = (-np.expm1(-a) / a if a else 1.0) * np.eye(4)
    for jac in (jax.jacrev, jax.jacfwd):
        assert_allclose(np.asarray(WRAPS[wrap](jac(f))(eps)), expected, rtol=4e-15)


def test_sources_below_one_keep_the_unit_scale_bit_for_bit():
    """``max|eps ds| < 1`` (0.63 here) gives ``k = 0``, ``s = 1``, as v0.2.0.

    On ``[1, 2)`` the code gives ``k = frexp(max|eps ds|)[1] = 1``, ``s = 2``,
    where v0.2.0 used ``s = max|eps ds|``: a change of the exact scaling, which
    can change the value by rounding only; that range is not bit-for-bit.

    Up to 0.3.0 the comparison was bit for bit with v0.2.0's value. Since 0.4.0
    the exponential is ``expm_pade13`` (not ``jax.scipy.linalg.expm``), so the
    unscaled matrix is exponentiated with it for the bit-for-bit check, and
    v0.2.0's value is only required within ``16 u`` (both are accurate to a
    few ``u`` here, ``|K ds|_1 = 0.99``)."""
    from syncmoments._expm import expm_pade13

    K = jnp.asarray(
        [[0.3, 0.04, 0, 0], [0.04, 0.3, 0.6, 0], [0, -0.6, 0.3, -0.2], [0, 0, 0.2, 0.3]]
    )
    eps, S0 = jnp.array([0.7, -0.2, 0.1, 0.05]), jnp.array([1.0, 0.1, 0.0, 0.0])
    ds = 0.9

    def unit_scale(expm):
        def value(S, e, Kmat, d):
            M = jnp.zeros((5, 5)).at[:4, :4].set(-Kmat * d).at[:4, 4].set(e * d)
            return jnp.sum(expm(M)[:4] * jnp.concatenate([S, jnp.ones(1)]), axis=1)

        return value

    unscaled = unit_scale(lambda M: expm_pade13(M, 32))
    v020 = unit_scale(lambda M: jax.scipy.linalg.expm(M, max_squarings=32))
    for wrap in WRAPS.values():
        got = wrap(transfer_slab)(S0, eps, K, ds)
        assert_array_equal(got, wrap(unscaled)(S0, eps, K, ds))
        size = float(jnp.max(jnp.abs(got)))
        assert_allclose(got, v020(S0, eps, K, ds), rtol=0, atol=16 * 2.0**-53 * size)
