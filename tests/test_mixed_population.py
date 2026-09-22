"""Independent discrete oracles for correlated fixed-harmonic populations."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from synchro import QuadraticTaylorExpansion, build_expansion, mixed_moments
from synchro.stokes import stokes_harmonic


def polynomial_model():
    # Analytic natural-basis kernels in x,y; no derivative implementation oracle.
    return QuadraticTaylorExpansion(
        harmonics=(1,),
        S0=jnp.array([[4.0, -1.0, 0.2]]),
        dS=jnp.array([[[1.0, 0.2], [0.3, -0.1], [0.1, 0.2]]]),
        ddS=jnp.array(
            [
                [
                    [[1.0, 0.3], [0.3, 0.4]],
                    [[0.2, -0.2], [-0.2, 0.1]],
                    [[0.0, 0.1], [0.1, 0.0]],
                ]
            ]
        ),
    )


def polynomial_values(points):
    x, y = np.asarray(points).T
    return np.array(
        [
            4 + x + 0.2 * y + 0.5 * x * x + 0.3 * x * y + 0.2 * y * y,
            -1 + 0.3 * x - 0.1 * y + 0.1 * x * x - 0.2 * x * y + 0.05 * y * y,
            0.2 + 0.1 * x + 0.2 * y + 0.1 * x * y,
        ]
    ).T


def sky_average(natural, factors, phi, weights):
    # Explicit per-electron sky rotation then sum: independent of moment algebra.
    w = np.asarray(weights) / np.sum(weights)
    I, Q, V = np.asarray(natural).T
    return np.sum(
        w[:, None]
        * factors[:, None]
        * np.stack([I, Q * np.cos(2 * phi), Q * np.sin(2 * phi), V], axis=-1),
        axis=0,
    )


def test_correlated_polynomial_and_failed_marginal_factorisation():
    q = np.array([[-0.5, 0.3], [0.2, -0.4], [0.8, 0.6]])
    f = np.array([0.2, 1.0, 2.5])  # (B/B0)^2
    phi = np.array([0.0, 0.7, 1.2])
    w = np.array([2.0, 3.0, 5.0])
    field = mixed_moments(q, f, w)
    phase = mixed_moments(q, f * np.exp(2j * phi), w)
    expected = sky_average(polynomial_values(q), f, phi, w)
    for fn in (
        polynomial_model().mixed_average,
        eqx.filter_jit(polynomial_model().mixed_average),
    ):
        result, error = fn(field, phase, absolute_error=0.0)
        assert_allclose(result[0], expected, rtol=2e-14, atol=2e-14)
        assert_allclose(error, np.zeros((1, 4)))
    # Same marginals, deliberately wrong independence replacement.
    independent = np.average(polynomial_values(q)[:, 0], weights=w) * np.average(
        f, weights=w
    )
    assert abs(independent - expected[0]) > 0.5


def test_conditional_regrouping_and_equal_marginal_pairings():
    q = np.array([[-0.4, 0.2], [-0.4, 0.2], [0.7, -0.3], [0.7, -0.3]])
    f = np.array([0.1, 0.4, 1.3, 2.2])
    phi = np.array([0.0, 0.5, 1.1, -0.4])
    w = np.array([1.0, 2.0, 4.0, 3.0])
    phase_factor = f * np.exp(2j * phi)
    result = polynomial_model().mixed_average(
        mixed_moments(q, f, w), mixed_moments(q, phase_factor, w), absolute_error=0.0
    )[0][0]
    grouped = np.zeros(4)
    for indices in ([0, 1], [2, 3]):
        mass = w[indices].sum() / w.sum()
        group_f = np.average(f[indices], weights=w[indices])
        group_p = np.average(phase_factor[indices], weights=w[indices])
        I, Q, V = polynomial_values(q[indices[:1]])[0]
        grouped += mass * np.array(
            [I * group_f, Q * group_p.real, Q * group_p.imag, V * group_f]
        )
    assert_allclose(result, grouped, atol=2e-14)
    # Equal weights preserve every marginal under permutation, not the joint law.
    a = polynomial_model().mixed_average(
        mixed_moments(q, f), mixed_moments(q, f), absolute_error=0.0
    )[0]
    b = polynomial_model().mixed_average(
        mixed_moments(q, f[::-1]), mixed_moments(q, f[::-1]), absolute_error=0.0
    )[0]
    assert abs(float(a[0, 0] - b[0, 0])) > 0.5


def test_independent_limit_and_zero_field():
    q0 = np.array([[-0.3, 0.2], [0.4, -0.1]])
    q = np.repeat(q0, 2, axis=0)
    ratios = np.tile([0.8, 1.2], 2)
    model = polynomial_model()
    mean = q0.mean(0)
    cov = (q0 - mean).T @ (q0 - mean) / 2
    expected = model.apply_B(
        model(jnp.asarray(mean), jnp.asarray(cov)), 1.0, var_B=0.04
    )
    moments = mixed_moments(q, ratios**2)
    result, _ = model.mixed_average(moments, moments, absolute_error=0.0)
    assert_allclose(result[:, [0, 1, 3]], expected, atol=2e-14)
    assert_allclose(result[:, 2], 0.0)
    zero = mixed_moments(q, np.zeros(4))
    assert_allclose(model.mixed_average(zero, zero, absolute_error=0.0)[0], 0.0)


def test_mixed_moments_jit_gradient_and_extreme_weights():
    q = jnp.array([[-0.3, 0.4], [0.7, -0.2]])
    f = jnp.array([0.2, 1.3])
    w = jnp.array([1e308, 1e308])
    fn = eqx.filter_jit(mixed_moments)
    for got, expected in zip(fn(q, f, w), mixed_moments(q, f)):
        assert_allclose(got, expected, atol=1e-14)

    def predict(factors):
        moments = mixed_moments(q, factors)
        return polynomial_model().mixed_average(moments, moments, absolute_error=0.0)[
            0
        ][0, 0]

    assert_allclose(jax.grad(predict)(f), polynomial_values(q)[:, 0] / 2, atol=2e-14)


def test_zero_circular_moment_retains_correlated_linear_term():
    q = np.array([[-1.0, 0.0], [1.0, 0.0]])
    phase = np.array([1.0, -1.0], dtype=complex)
    field_m = mixed_moments(q, np.ones(2))
    phase_m = mixed_moments(q, phase)
    assert phase_m[0] == 0
    result, _ = polynomial_model().mixed_average(field_m, phase_m, absolute_error=0.0)
    # E[Q]*E[exp(2i phi)] would be zero; the correlated answer is -0.3.
    assert_allclose(result[0, 1:3], [-0.3, 0.0], atol=2e-14)


def test_reference_field_rescaling_multiple_harmonics_and_phase_gradient():
    base = polynomial_model()
    model = QuadraticTaylorExpansion(
        harmonics=(1, 2),
        S0=jnp.concatenate([base.S0, 2 * base.S0]),
        dS=jnp.concatenate([base.dS, 2 * base.dS]),
        ddS=jnp.concatenate([base.ddS, 2 * base.ddS]),
    )
    q = jnp.array([[-0.3, 0.2], [0.4, -0.1]])
    f = jnp.array([0.6, 1.4])
    phi = jnp.array([0.2, 0.8])

    def predict(angles, response=model, factors=f):
        return response.mixed_average(
            mixed_moments(q, factors),
            mixed_moments(q, factors * jnp.exp(2j * angles)),
            absolute_error=0.0,
        )[0]

    original = predict(phi)
    assert_allclose(original[1], 2 * original[0], atol=2e-14)
    scaled = eqx.tree_at(
        lambda m: (m.S0, m.dS, m.ddS),
        model,
        (9 * model.S0, 9 * model.dS, 9 * model.ddS),
    )
    assert_allclose(predict(phi, scaled, f / 9), original, atol=2e-14)
    # d<Q>/dphi_a = -2*w_a*f_a*Q_a*sin(2phi_a), w_a=1/2.
    derivative = jax.grad(lambda angles: predict(angles)[0, 1])(phi)
    assert_allclose(
        derivative, -f * polynomial_values(q)[:, 1] * np.sin(2 * phi), atol=2e-14
    )


def test_direct_harmonics_and_asymmetric_cubic_remainder():
    base = np.array([3.0, 0.7, 1.1])
    directions = np.array([[-0.8, 0.2, -0.3], [0.3, -0.6, 0.1], [0.9, 0.5, 0.4]])
    B0 = 5e-6
    ratios = np.array([0.6, 1.0, 1.5])
    phi = np.array([0.0, 0.6, -0.8])
    weights = np.array([1.0, 2.0, 4.0])
    model = build_expansion([2], *base, B=B0)
    errors = []
    for width in [0.02, 0.01]:
        offsets = width * directions
        f = ratios**2
        direct = np.array(
            [stokes_harmonic(2, *q, B=B0 * r) for q, r in zip(base + offsets, ratios)]
        )
        reference = sky_average(direct, np.ones(3), phi, weights)
        pred, _ = model.mixed_average(
            mixed_moments(offsets, f, weights),
            mixed_moments(offsets, f * np.exp(2j * phi), weights),
            absolute_error=reference[0],
        )
        errors.append(np.max(np.abs(np.asarray(pred[0]) - reference)) / reference[0])
    assert 6.5 < errors[0] / errors[1] < 9.5
    assert errors[1] < 2e-5
    # Loose test envelope only; the observed cubic scaling is a finite probe,
    # not a derivative/support certificate for other populations.


def test_weighted_remainder_is_not_the_unweighted_remainder():
    x = np.array([-0.2, 0.1, 0.8])
    q = np.c_[x, np.zeros(3)]
    f = np.array([0.1, 0.3, 5.0])
    w = np.array([1.0, 2.0, 4.0])
    w /= w.sum()
    model = polynomial_model()
    field = mixed_moments(q, f, w)
    # Add cubic remainders to the otherwise quadratic I, Q, V kernels.
    true = polynomial_values(q) + x[:, None] ** 3 * np.array([1.0, -0.2, 0.1])
    rho = np.sum(w * f * np.abs(x) ** 3) * np.array([1.0, 0.2, 0.2, 0.1])
    pred, error = model.mixed_average(field, field, absolute_error=rho)
    expected = sky_average(true, f, np.zeros(3), w)
    assert np.all(np.abs(pred[0] - expected) <= np.asarray(error[0]) + 1e-14)
    assert abs(pred[0, 0] - expected[0]) > np.sum(w * np.abs(x) ** 3)


@pytest.mark.parametrize(
    "q,f,w",
    [
        ([], [], None),
        ([[0.0, 1.0]], [1.0, 2.0], None),
        ([[np.nan]], [1.0], None),
        ([[1j]], [1.0], None),
        ([[0.0]], [np.inf], None),
        ([[0.0]], [1.0], [-1.0]),
        ([[0.0]], [1.0], [0.0]),
        ([[0.0]], [1.0], [np.nan]),
        ([[0.0]], [1.0], [1j]),
        ([[0.0]], [1.0], [1.0, 2.0]),
        ([[1e200]], [1e200], None),
    ],
)
def test_mixed_moments_invalid(q, f, w):
    for fn in (mixed_moments, eqx.filter_jit(mixed_moments)):
        with pytest.raises(Exception):
            jax.block_until_ready(fn(q, f, w))


def test_mixed_average_contracts():
    model = polynomial_model()
    good = (jnp.array(1.0), jnp.zeros(2), jnp.eye(2))
    for bad in [
        (1.0, jnp.zeros(3), jnp.eye(2)),
        (1j, jnp.zeros(2), jnp.eye(2)),
        (np.inf, jnp.zeros(2), jnp.eye(2)),
    ]:
        for fn in (model.mixed_average, eqx.filter_jit(model.mixed_average)):
            with pytest.raises(Exception):
                jax.block_until_ready(fn(bad, good, absolute_error=0.0))
    for bad in [
        (np.nan + 1j, jnp.zeros(2), jnp.eye(2)),
        (1.0, jnp.zeros(1), jnp.eye(2)),
        (1.0, jnp.zeros(2)),
    ]:
        for fn in (model.mixed_average, eqx.filter_jit(model.mixed_average)):
            with pytest.raises(Exception):
                jax.block_until_ready(fn(good, bad, absolute_error=0.0))
    huge = (1e308, jnp.zeros(2), jnp.eye(2))
    for fn in (model.mixed_average, eqx.filter_jit(model.mixed_average)):
        with pytest.raises(Exception, match="overflowed"):
            jax.block_until_ready(fn(huge, good, absolute_error=0.0))
    for err in [-1.0, np.inf, 1j, np.zeros(3)]:
        for fn in (model.mixed_average, eqx.filter_jit(model.mixed_average)):
            with pytest.raises(Exception):
                jax.block_until_ready(fn(good, good, absolute_error=err))
    with pytest.raises(TypeError):
        model.mixed_average(good, good)  # the remainder contract is explicit
