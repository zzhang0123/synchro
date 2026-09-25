"""Weight normalisation at the edges of the float formats (T-004 round 2, G1).

1. A largest weight below the smallest normal float64 (``2^-1022``) is refused
   with a clear error, eagerly, under ``jax.jit`` and under differentiation
   (v0.2.0 refused these through its flushed ``max(w) > 0`` test; round 1
   returned silently NaN forward-mode derivatives, because ``1 / max(w)``
   overflows).
2. At the smallest normal maximum, forward-mode derivatives are finite and
   equal the oracle ``d mean = sum_i t_i (x_i - mean) / W`` (``W = sum w``),
   including non-unit tangents whose intermediate ``sum(dw) / max(w)`` exceeds
   float64 (round 1 returned NaN).
3. float16, bfloat16 and float32 weights are read in their own bit layout and
   normalised in float64, subnormals of their format included (round 1 raised
   in ``bitcast_convert_type``); results are cast to the samples' dtype.
4. The documented flush threshold on XLA CPU: ratios to the largest weight
   below ``2^-1022`` are dropped, also where ``_normalised_weights`` applies its
   tangent headroom factor (``max(w) < 2^-958``).
5. ``[-5e-324, 1]`` is refused under ``jax.jit`` as well (round 1 accepted it
   there through the sign-magnitude mask ``bits & 0x7ff...``).

Oracles are NumPy (no flush-to-zero) on exact dyadic patterns.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.expansion import mixed_moments
from syncmoments.model.moments import PopulationSamples
from syncmoments.rm import (
    _normalised_weights,
    _relative_weights,
    gaussian_rm_cumulants,
    screen_polarisation,
)

TINY = 5e-324
REFUSAL = "largest weight is below the smallest normal float64"
X = np.array([0.0, 1.0, 2.0, 3.0])
SUBNORMAL_MAX = {
    "tiny-pattern": np.array([4.0, 2.0, 1.0, 1.0]) * TINY,
    "half-min-normal": np.full(4, 2.0**-1023),
    "with-zeros": np.array([2.0**-1030, 0.0, 2.0**-1040, 0.0]),
}
WRAPS = {"eager": lambda f: f, "jit": jax.jit}
CPU = jax.default_backend() == "cpu"


def _entry_points():
    def population(w):
        values = jnp.asarray([1.5, 2.0, 2.5, 3.0])
        samples = PopulationSamples(
            values, values, 0.1 * values, 0.2 * values, values, values, weights=w
        )
        return samples.normalised_weights()

    return {
        "rm_cumulants": lambda w: gaussian_rm_cumulants(X, w),
        "screen": lambda w: screen_polarisation(1.0 + 0.0j, X, jnp.array([0.2]), w),
        "mixed_moments": lambda w: mixed_moments(X[:, None], jnp.ones(4), w),
        "population": population,
    }


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("entry", _entry_points())
@pytest.mark.parametrize("case", SUBNORMAL_MAX)
def test_subnormal_largest_weight_is_refused(case, entry, wrap):
    fn = WRAPS[wrap](_entry_points()[entry])
    with pytest.raises(Exception, match=REFUSAL):
        jax.block_until_ready(fn(jnp.asarray(SUBNORMAL_MAX[case])))


def _mean(w):
    return gaussian_rm_cumulants(X, w)[0]


def _var(w):
    return gaussian_rm_cumulants(X, w)[1]


DERIVATIVES = {
    "grad": jax.grad(_mean),
    "jacfwd": jax.jacfwd(_mean),
    "jit-jacfwd": jax.jit(jax.jacfwd(_mean)),
    "hessian": jax.hessian(_var),
    "jvp": lambda w: jax.jvp(_mean, (w,), (jnp.ones_like(w),))[1],
}


@pytest.mark.parametrize("name", DERIVATIVES)
def test_subnormal_largest_weight_is_refused_under_differentiation(name):
    """No silent NaN or inf: every derivative transform raises the same error."""
    w = jnp.asarray(SUBNORMAL_MAX["tiny-pattern"])
    with pytest.raises(Exception, match=REFUSAL):
        jax.block_until_ready(DERIVATIVES[name](w))


@pytest.mark.parametrize(
    "w",
    [np.full(4, 2.0**-1022), np.array([4.0, 2.0, 1.0, 1.0]) * 2.0**-1024],
    ids=["equal", "pattern"],
)
def test_smallest_normal_maximum_has_finite_oracle_derivatives(w):
    """``d mean / d w_i = (x_i - mean) / W``; ``W = 2^-1020`` or ``2^-1021``, so
    derivatives reach 4.8e307 and the jvp with ``t = [1, 2, 3, 4] / 4`` 6.3e307.
    With a ones tangent the unscaled intermediate ``sum(t) / max(w) = 2^1024``
    overflows (round 1: NaN); the ``2^-64`` headroom factor keeps it finite."""
    W = w.sum()
    m = w @ X / W
    grad_oracle = (X - m) / W
    t = np.array([1.0, 2.0, 3.0, 4.0]) / 4
    jvp_oracle = t @ (X - m) / W
    wj = jnp.asarray(w)
    for fn in (jax.grad(_mean), jax.jacfwd(_mean), jax.jit(jax.jacfwd(_mean))):
        assert_allclose(np.asarray(fn(wj)), grad_oracle, rtol=1e-15)
    for wrap in WRAPS.values():
        got = wrap(lambda v: jax.jvp(_mean, (v,), (jnp.asarray(t),))[1])(wj)
        assert np.isfinite(got)
        assert_allclose(float(got), jvp_oracle, rtol=1e-15)
        ones = wrap(lambda v: jax.jvp(_mean, (v,), (jnp.ones(4),))[1])(wj)
        scale = np.abs(grad_oracle).max()
        assert_allclose(float(ones), grad_oracle.sum(), rtol=0, atol=4e-15 * scale)


LOW_PRECISION = {
    "float16": (jnp.float16, 2.0**-24),  # smallest subnormal of each format
    "bfloat16": (jnp.bfloat16, 2.0**-133),
    "float32": (jnp.float32, 2.0**-149),
}
PATTERN = np.array([4.0, 2.0, 1.0])
RMS3 = np.array([0.0, 1.0, 2.0])


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("unit", ["one", "subnormal"])
@pytest.mark.parametrize("dtype", LOW_PRECISION)
def test_low_precision_weights_are_normalised_in_float64(dtype, unit, wrap):
    """``[4, 2, 1]`` in units of 1 or of the format's smallest subnormal is exact
    in every format; float64 samples then give the float64 result to 4e-16."""
    dt, tiny = LOW_PRECISION[dtype]
    w = jnp.asarray(PATTERN * (1.0 if unit == "one" else tiny), dtype=dt)
    assert np.all(
        np.asarray(w, dtype=np.float64) == PATTERN * (tiny if unit != "one" else 1)
    )
    p = PATTERN / PATTERN.sum()
    mean = p @ RMS3
    got = WRAPS[wrap](gaussian_rm_cumulants)(RMS3, w)
    assert_allclose(
        np.asarray(got, dtype=np.float64), [mean, p @ (RMS3 - mean) ** 2], rtol=4e-16
    )


@pytest.mark.parametrize("dtype", LOW_PRECISION)
def test_low_precision_samples_keep_their_dtype(dtype):
    """Samples and weights in the same low-precision format: the result has that
    dtype and matches the float64 oracle to a few ulp of the format."""
    dt = LOW_PRECISION[dtype][0]
    mean, var = gaussian_rm_cumulants(jnp.asarray(RMS3, dt), jnp.asarray(PATTERN, dt))
    assert mean.dtype == dt and var.dtype == dt
    p = PATTERN / PATTERN.sum()
    eps = float(jnp.finfo(dt).eps)
    assert_allclose(float(mean), p @ RMS3, rtol=4 * eps)
    offsets = jnp.asarray(RMS3[:, None], dt)
    mass, first, _ = mixed_moments(offsets, jnp.ones(3, dt), jnp.asarray(PATTERN, dt))
    assert_allclose([float(mass), float(first[0])], [1.0, p @ RMS3], rtol=4 * eps)


def test_unsupported_float_width_is_refused():
    w = jnp.asarray(PATTERN, dtype=jnp.float8_e4m3fn)
    with pytest.raises(ValueError, match="unsupported float dtype"):
        gaussian_rm_cumulants(RMS3, w)


@pytest.mark.skipif(not CPU, reason="flush-to-zero thresholds measured on XLA CPU")
def test_documented_flush_thresholds_on_cpu():
    for wrap in WRAPS.values():
        rel = np.asarray(wrap(_relative_weights)(jnp.asarray([1.0, 2.0**-1022])))
        assert rel[0] == 1.0 and rel[1] == 2.0**-1022
        rel = np.asarray(wrap(_relative_weights)(jnp.asarray([1.0, 2.0**-1030])))
        assert rel[0] == 1.0 and rel[1] == 0.0
    for w in ([1.0, 2.0**-1022], [2.0**-1000, 2.0**-1074]):  # headroom s = 0, 42
        p = np.asarray(_normalised_weights(jnp.asarray(w)))
        assert_allclose(p, np.array(w) / np.sum(w), rtol=2**-52, atol=0)
    p = np.asarray(_normalised_weights(jnp.asarray([1.0, 2.0**-1030])))
    assert p[0] == 1.0 and p[1] == 0.0


@pytest.mark.parametrize("wrap", WRAPS)
def test_negative_subnormal_weight_is_refused_under_jit(wrap):
    def mixed(x, w):
        return mixed_moments(x[:, None], jnp.ones(2), w)

    for fn in (gaussian_rm_cumulants, mixed):
        with pytest.raises(Exception, match="weights must be finite"):
            w = jnp.asarray([-TINY, 1.0])
            jax.block_until_ready(WRAPS[wrap](fn)(jnp.zeros(2), w))
