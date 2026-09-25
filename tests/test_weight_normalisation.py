"""Relative-weight normalisation at the float64 extremes (T-004 package A).

Every weighted average in the package normalises relative electron-number
weights ``w`` through ``syncmoments.rm._relative_weights``. On jax 0.10.2 CPU,
``5e307 / 1e308`` evaluates to 0: XLA forms the reciprocal ``1e-308``, which
is subnormal and flushed to zero. XLA CPU also treats subnormal inputs as
zero in arithmetic and comparisons (``5e-324 > 0`` is False on jax 0.10.0 and
0.10.2). The helper therefore reads the float64 bit pattern, rescales by the
power of two ``2^floor(log2 max w)`` exactly, then divides by the rescaled
maximum in ``[1, 2)``; these tests pin ``w / max(w)`` at
``max w in {4e307, 1e308, 1.79e308, 1e-300, 2^-1022}``, eagerly and under
``jax.jit``, through the public entry points that use it. At ``2^-1022`` (the
smallest normal) the other entries of ``[4, 2, 1]`` are subnormal and keep their
ratios. A subnormal largest weight is refused (``test_weight_normalisation_edges``).

Oracles are NumPy (no flush-to-zero): the relative weights ``[4, 2, 1] / 7``
and ``[1, 1, 1] / 3`` are exact dyadic patterns at every scale, so the
normalised weights are compared to 1 ulp-level tolerances (``rtol = 4e-16``).
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.expansion import mixed_moments
from syncmoments.model.moments import PopulationSamples
from syncmoments.rm import _relative_weights, gaussian_rm_cumulants, screen_polarisation

TINY = 5e-324  # smallest positive subnormal float64, 2^-1074
MIN_NORMAL = 2.0**-1022  # smallest normal float64, 2.2e-308
SCALES = (4e307, 1e308, 1.79e308, 1e-300, MIN_NORMAL)
PATTERN = np.array([4.0, 2.0, 1.0])  # exact relative masses, normalised [4, 2, 1]/7
RMS = np.array([0.0, 1.0, 2.0])  # rad/m^2


def _weights(scale, pattern=PATTERN):
    """``pattern`` scaled so that its largest entry is ``scale`` (exact in NumPy;
    at ``MIN_NORMAL`` the smaller entries are the subnormals 2^-1023, 2^-1024)."""
    return pattern / pattern.max() * scale


def _numpy_relative(w):
    """``w / max(w)`` in NumPy, which does not flush subnormals."""
    return w / np.max(w)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", SCALES)
def test_relative_weights_equal_w_over_max_without_flushing(scale, jit):
    w = _weights(scale)
    fn = jax.jit(_relative_weights) if jit else _relative_weights
    got = np.asarray(fn(jnp.asarray(w)))
    # jax 0.10.2 compiles ``x / m`` as ``x * (1/m)``: two roundings, so the
    # dyadic ratios are exact only to one ulp (2^-52 relative) there.
    assert_allclose(got, _numpy_relative(w), rtol=2.0**-52, atol=0)
    assert abs(got.max() - 1.0) <= 2.0**-52 and np.all(got > 0)


def test_relative_weights_bit_identical_to_the_v020_division_in_the_normal_range():
    """Non-dyadic weights well inside the normal range: same bits as ``w / max w``
    computed by XLA (the v0.2.0 expression), eagerly and under ``jax.jit``."""
    w = jnp.asarray(np.random.default_rng(3).uniform(0.2, 3.0, 17) * 1e5)

    def v020(x):
        return x / jax.lax.stop_gradient(jnp.max(x))

    for wrap in (lambda f: f, jax.jit):
        assert_array_equal(np.asarray(wrap(_relative_weights)(w)), wrap(v020)(w))


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", SCALES)
def test_public_averages_at_extreme_weight_scales(scale, jit):
    """RM cumulants, the screen average and mixed moments see ``[4,2,1]/7``."""
    w = _weights(scale)
    p = PATTERN / PATTERN.sum()
    mean = p @ RMS
    var = p @ (RMS - mean) ** 2
    wrap = jax.jit if jit else (lambda f: f)
    assert_allclose(wrap(gaussian_rm_cumulants)(RMS, w), [mean, var], rtol=4e-16)
    lam = np.array([0.1, 0.3])
    expected = p @ np.exp(2j * RMS[:, None] * lam**2)
    got = wrap(screen_polarisation)(1.0 + 0.0j, RMS, lam, w)
    assert_allclose(np.asarray(got), expected, rtol=4e-16)
    offsets = RMS[:, None]
    mass, first, second = wrap(mixed_moments)(offsets, np.ones(3), w)
    assert_allclose([mass, first[0], second[0, 0]], [1.0, mean, p @ RMS**2], rtol=4e-16)


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("scale", SCALES)
def test_equal_extreme_weights_are_uniform(scale, jit):
    w = np.full(3, scale)
    wrap = jax.jit if jit else (lambda f: f)
    assert_allclose(wrap(gaussian_rm_cumulants)(RMS, w), [1.0, 2.0 / 3.0], rtol=4e-16)


@pytest.mark.parametrize("scale", SCALES)
def test_population_samples_normalise_extreme_weights(scale):
    values = jnp.asarray([1.5, 2.0, 2.5])
    samples = PopulationSamples(
        values,
        values,
        0.1 * values,
        0.2 * values,
        values,
        values,
        weights=_weights(scale),
    )
    assert_allclose(
        np.asarray(samples.normalised_weights()), PATTERN / PATTERN.sum(), rtol=4e-16
    )


@pytest.mark.parametrize("scale", (1.79e308, 1e-300, MIN_NORMAL))
def test_product_population_normalises_extreme_marginal_weights(scale):
    grid = [1.0, 2.0]
    marginal = (np.asarray(grid), _weights(scale, np.array([2.0, 1.0])))
    samples = PopulationSamples.product(
        gamma=marginal, B=grid, mu=[0.1], eta=[0.3], phi=[0.0], depth=[0.0]
    )
    w = np.asarray(samples.normalised_weights())
    assert np.all(np.isfinite(w))
    assert_allclose(w.reshape(2, 2), [[1 / 3, 1 / 3], [1 / 6, 1 / 6]], rtol=4e-16)


@pytest.mark.parametrize("scale", (1e-300, 1.0, 1e300))
def test_weight_gradients_are_scale_covariant_and_match_the_oracle(scale):
    """``d mean / d w_i = (x_i - mean) / sum(w)``: with ``w -> s w`` the gradient
    scales as ``1/s`` exactly as before (the scale is a stop-gradient power of
    two). Scales whose true gradient is subnormal (``> 1e308``) are excluded:
    the derivative itself is not representable there."""
    w = PATTERN * scale

    def mean(weights):
        return gaussian_rm_cumulants(RMS, weights)[0]

    p = PATTERN / PATTERN.sum()
    oracle = (RMS - p @ RMS) / w.sum()
    for fn in (jax.grad(mean), jax.jit(jax.grad(mean))):
        assert_allclose(np.asarray(fn(jnp.asarray(w))), oracle, rtol=1e-15)
    fwd = jax.jacfwd(mean)(jnp.asarray(w))
    assert_allclose(np.asarray(fwd), oracle, rtol=1e-15)


def test_second_derivative_in_the_weights_matches_the_oracle():
    """``d^2 mean / dw_i dw_j = -[(x_i - m) + (x_j - m)] / W^2`` (``W = sum w``)."""
    w = jnp.asarray(PATTERN)

    def mean(weights):
        return gaussian_rm_cumulants(RMS, weights)[0]

    m = (PATTERN @ RMS) / PATTERN.sum()
    d = RMS - m
    oracle = -(d[:, None] + d[None, :]) / PATTERN.sum() ** 2
    assert_allclose(np.asarray(jax.hessian(mean)(w)), oracle, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize(
    "bad", [[-TINY, 1.0], [0.0, 0.0], [TINY * 0.0, -0.0], [np.nan, 1.0], [np.inf, 1.0]]
)
def test_invalid_weights_are_refused_including_negative_subnormals(bad):
    """``-5e-324 < 0`` is False under flush-to-zero; the sign bit is read instead."""
    with pytest.raises(Exception, match="weights must be finite"):
        gaussian_rm_cumulants(np.array([0.0, 1.0]), np.array(bad))
