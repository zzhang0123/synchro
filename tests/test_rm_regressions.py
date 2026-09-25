"""External-screen statistics, physical input gates and independent CF checks."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.testing import assert_allclose
import pytest

from syncmoments.rm import (
    burn_depolarisation,
    gaussian_rm_cumulants,
    rotation_measure_practical,
)


@pytest.mark.parametrize(
    "weights", ([-1.0, 2.0], [0.0, 0.0], [1.0, np.nan], [1.0, np.inf])
)
def test_invalid_weights_are_rejected_in_eager_and_jit(weights):
    with pytest.raises(Exception, match="weights must"):
        gaussian_rm_cumulants([1.0, 2.0], weights)
    with pytest.raises(Exception, match="weights must"):
        eqx.filter_jit(gaussian_rm_cumulants)(
            jnp.array([1.0, 2.0]), jnp.asarray(weights)
        )


def test_rm_sample_shape_and_finiteness_contracts():
    for values in ([], [[1.0, 2.0]], [1.0, np.nan]):
        with pytest.raises(Exception, match="RM samples"):
            gaussian_rm_cumulants(values)
    with pytest.raises(ValueError, match="match"):
        gaussian_rm_cumulants([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="real"):
        gaussian_rm_cumulants([1.0 + 0.1j, 2.0])


def test_weight_scaling_and_sample_derivatives_preserve_population_statistics():
    samples = np.array([-2.0, 0.5, 3.0])
    weights = np.array([1.0, 2.0, 1.0])
    normalized = weights / weights.sum()
    mean = normalized @ samples
    variance = normalized @ (samples - mean) ** 2
    oracle_jacobian = np.stack([normalized, 2 * normalized * (samples - mean)])
    for scale in (1.0, 1e308):
        # Keep each weight finite while their unscaled total exceeds float64.
        relative_weights = weights * (scale / 2)
        assert_allclose(
            jax.jit(gaussian_rm_cumulants)(samples, relative_weights),
            [mean, variance],
            atol=2e-15,
        )

    def statistic(values):
        return jnp.stack(gaussian_rm_cumulants(values, weights))

    assert_allclose(
        jax.jacfwd(statistic)(jnp.asarray(samples)), oracle_jacobian, atol=2e-15
    )
    assert_allclose(
        jax.jit(jax.jacrev(statistic))(jnp.asarray(samples)),
        oracle_jacobian,
        atol=2e-15,
    )

    def weighted_statistic(value):
        return jnp.stack(gaussian_rm_cumulants(samples, value))

    weight_oracle = np.stack(
        [
            (samples - mean) / weights.sum(),
            ((samples - mean) ** 2 - variance) / weights.sum(),
        ]
    )
    assert_allclose(
        jax.jit(jax.jacrev(weighted_statistic))(jnp.asarray(weights)),
        weight_oracle,
        atol=2e-15,
    )


def test_gaussian_screen_matches_independent_characteristic_integral():
    nodes, weights = hermgauss(80)
    mean, variance, incident = -1.3, 0.16, 0.7 - 0.2j
    rm_values = mean + np.sqrt(2 * variance) * nodes
    wavelengths = np.array([0.0, 0.2, 0.7, 1.1])
    oracle = (
        incident
        * (np.exp(2j * rm_values[:, None] * wavelengths[None, :] ** 2).T @ weights)
        / np.sqrt(np.pi)
    )
    actual = jax.jit(burn_depolarisation)(incident, mean, variance, wavelengths)
    assert_allclose(actual, oracle, rtol=3e-14, atol=3e-15)
    lam = 0.7

    def response(value):
        output = burn_depolarisation(incident, mean, value, lam)
        return jnp.array([output.real, output.imag])

    expected = -2 * lam**4 * complex(burn_depolarisation(incident, mean, variance, lam))
    assert_allclose(
        jax.jacrev(response)(variance), [expected.real, expected.imag], atol=2e-15
    )


def test_two_rm_statistics_do_not_assert_gaussianity():
    # A symmetric two-point screen has the same mean/variance as N(0,1)
    # but characteristic function cos(2 lambda^2), not exp(-2 lambda^4).
    mean, variance = gaussian_rm_cumulants([-1.0, 1.0])
    lam = np.sqrt(np.pi / 4)
    empirical_response = np.mean(np.exp(2j * np.array([-1.0, 1.0]) * lam**2))
    gaussian_response = burn_depolarisation(1.0, mean, variance, lam)
    assert abs(empirical_response) < 1e-14
    assert abs(gaussian_response) > 0.2


def test_external_screen_rejects_negative_variance_and_nonfinite_inputs():
    for variance in (-0.1, np.nan, np.inf):
        with pytest.raises(Exception, match="Gaussian screen"):
            burn_depolarisation(1.0, 0.0, variance, 0.7)
    with pytest.raises(Exception, match="Gaussian screen"):
        eqx.filter_jit(burn_depolarisation)(1.0, 0.0, jnp.asarray(-0.1), 0.7)


def test_practical_path_inputs_and_uniform_integral():
    path = np.linspace(0.0, 100.0, 21)
    assert_allclose(
        rotation_measure_practical(
            np.full_like(path, 0.03), np.full_like(path, 5.0), path
        ),
        0.812 * 0.03 * 5 * 100,
    )
    with pytest.raises(ValueError, match="matching"):
        rotation_measure_practical([1.0], [1.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="increasing"):
        rotation_measure_practical([1.0, 1.0], [1.0, 1.0], [1.0, 0.0])
