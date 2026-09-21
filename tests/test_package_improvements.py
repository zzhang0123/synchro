"""Independent oracles for the T-001 statistical APIs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from synchro import CumulantExpansion, QuadraticTaylorExpansion
from synchro.rm import gaussian_rm_cumulants, rm_moments, screen_polarisation


def response():
    # S(x,y) = x^2 + 3xy + 2y^2, expanded at zero, for each Stokes slot.
    return QuadraticTaylorExpansion(
        harmonics=(1,),
        S0=jnp.zeros((1, 3)),
        dS=jnp.zeros((1, 3, 2)),
        ddS=jnp.tile(jnp.array([[2.0, 3.0], [3.0, 4.0]]), (1, 3, 1, 1)),
    )


def test_compatibility_names():
    assert QuadraticTaylorExpansion is CumulantExpansion
    assert rm_moments is gaussian_rm_cumulants


def test_checked_average_against_discrete_polynomial_expectation():
    samples = np.array([[-1.0, 2.0], [2.0, 0.0], [1.0, -1.0]])
    mean = samples.mean(axis=0)
    cov = (samples - mean).T @ (samples - mean) / len(samples)
    expected = np.mean(
        samples[:, 0] ** 2 + 3 * samples[:, 0] * samples[:, 1] + 2 * samples[:, 1] ** 2
    )
    model = response()
    val, bound = eqx.filter_jit(model.checked_average)(mean, cov, absolute_error=0.0)
    assert_allclose(val, expected, atol=1e-14)
    assert_allclose(bound, np.zeros((1, 3)))
    grad = jax.grad(
        lambda m: model.checked_average(m, cov, absolute_error=0.2)[0][0, 0]
    )(jnp.asarray(mean))
    assert_allclose(grad, np.array([[2.0, 3.0], [3.0, 4.0]]) @ mean)
    assert_allclose(
        model.checked_average([0.0, 0.0], np.zeros((2, 2)), absolute_error=0.2)[1], 0.2
    )


@pytest.mark.parametrize(
    "cov",
    [
        [[1.0, 2.0], [2.0, 1.0]],
        [[-1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [0.0, 1.0]],
        [[1.0, np.nan], [0.0, 1.0]],
        [[-1e-20, 0.0], [0.0, 1e20]],
        [[1e-20, 2.0], [2.0, 1e20]],
        [[0.0, 1e-20], [1e-20, 1.0]],
    ],
)
def test_checked_average_rejects_invalid_covariance(cov):
    for fn in (response().checked_average, eqx.filter_jit(response().checked_average)):
        with pytest.raises(Exception, match="covariance"):
            fn(jnp.zeros(2), jnp.asarray(cov), absolute_error=0.0)


def test_checked_average_rejects_bad_statistics_and_error_envelopes():
    model = response()
    for mean, cov, bound in [
        ([0.0], np.eye(2), 0.0),
        ([0.0, 0.0], np.eye(3), 0.0),
        ([np.inf, 0.0], np.eye(2), 0.0),
        ([0.0, 0.0], np.eye(2), -0.1),
        ([0.0, 0.0], np.eye(2), np.inf),
        ([1j, 0.0], np.eye(2), 0.0),
    ]:
        with pytest.raises(Exception):
            model.checked_average(mean, cov, absolute_error=bound)


def test_checked_average_scale_invariance_and_covariance_gradient():
    model = response()
    cov = jnp.array([[1e-20, 0.5], [0.5, 1e20]])
    value, _ = eqx.filter_jit(model.checked_average)(
        jnp.zeros(2), cov, absolute_error=0.0
    )
    assert_allclose(value, 1e-20 + 1.5 + 2e20)
    # AD through valid SPD statistics is still just the polynomial derivative.
    derivative = jax.grad(
        lambda variance: model.checked_average(
            jnp.zeros(2), jnp.diag(variance), absolute_error=0.0
        )[0][0, 0]
    )(jnp.array([1.0, 2.0]))
    assert_allclose(derivative, [1.0, 2.0])


def test_checked_average_propagates_component_envelope():
    envelope = np.array([[0.1, 0.2, 0.3]])
    _, actual = response().checked_average(
        [0.0, 0.0], np.eye(2), absolute_error=envelope
    )
    assert_allclose(actual, envelope)


def test_covariance_psd_requires_more_than_pairwise_bounds():
    model = QuadraticTaylorExpansion(
        harmonics=(1,),
        S0=jnp.zeros((1, 3)),
        dS=jnp.zeros((1, 3, 3)),
        ddS=jnp.tile(jnp.eye(3), (1, 3, 1, 1)),
    )
    correlation = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    scales = np.array([1e-10, 1.0, 1e10])
    cov = correlation * scales[:, None] * scales[None, :]
    for fn in (model.checked_average, eqx.filter_jit(model.checked_average)):
        with pytest.raises(Exception, match="positive semidefinite"):
            fn(jnp.zeros(3), jnp.asarray(cov), absolute_error=0.0)


@pytest.mark.parametrize("rho", [0.0, 1.0 - 1e-10, 1.0])
def test_covariance_valid_and_singular_boundary(rho):
    prediction, _ = eqx.filter_jit(response().checked_average)(
        jnp.zeros(2), jnp.array([[1.0, rho], [rho, 1.0]]), absolute_error=0.0
    )
    assert_allclose(prediction, 3.0 + 3.0 * rho)


def test_covariance_outside_psd_boundary():
    with pytest.raises(Exception, match="covariance"):
        eqx.filter_jit(response().checked_average)(
            jnp.zeros(2),
            jnp.array([[1.0, 1.0 + 1e-10], [1.0 + 1e-10, 1.0]]),
            absolute_error=0.0,
        )


def test_screen_two_point_and_correlated_polarisation():
    rm = np.array([-1.0, 1.0])
    lam = np.sqrt(np.pi / 4)
    assert_allclose(screen_polarisation(1.0, rm, lam), 0.0, atol=1e-15)
    # Correlation prevents factoring <P0 exp(i phase)> into <P0> times a CF.
    incident = np.array([1.0, -1.0])
    assert_allclose(screen_polarisation(incident, rm, lam), -1j, atol=1e-15)


def test_screen_grid_weights_and_derivatives():
    rm = np.array([-2.0, 0.5, 3.0])
    incident = np.array([1.0 + 0.2j, 0.7 - 0.4j, -0.1 + 0.3j])
    weights = np.array([1.0, 2.0, 1.0])
    lam = np.array([[0.0, 0.2], [0.5, 1.0]])
    phase = np.exp(2j * rm[:, None, None] * lam**2)
    expected = (
        np.sum(weights[:, None, None] * incident[:, None, None] * phase, axis=0) / 4
    )
    for scale in (1.0, 5e307):
        actual = jax.jit(screen_polarisation)(incident, rm, lam, weights * scale)
        assert_allclose(actual, expected, atol=2e-15)
    jac = jax.jacrev(lambda r: screen_polarisation(incident, r, 0.7, weights).real)(
        jnp.asarray(rm)
    )
    assert_allclose(
        jac,
        (weights / 4 * incident * 2j * 0.7**2 * np.exp(2j * rm * 0.7**2)).real,
        atol=2e-15,
    )
    wgrad = jax.grad(lambda w: screen_polarisation(incident, rm, 0.7, w).real)(
        jnp.asarray(weights)
    )
    rays = incident * np.exp(2j * rm * 0.7**2)
    assert_allclose(wgrad, ((rays - np.sum(weights * rays) / 4) / 4).real, atol=2e-15)


def test_screen_gaussian_quadrature_matches_burn():
    from numpy.polynomial.hermite import hermgauss
    from synchro.rm import burn_depolarisation

    nodes, weights = hermgauss(60)
    lam = np.linspace(0, 1, 9)
    assert_allclose(
        screen_polarisation(0.7 + 0.2j, -0.3 + np.sqrt(0.4) * nodes, lam, weights),
        burn_depolarisation(0.7 + 0.2j, -0.3, 0.2, lam),
        atol=3e-15,
    )


@pytest.mark.parametrize(
    "incident,rm,lam,weights",
    [
        (1.0, [], 0.2, None),
        (1.0, [0.0, np.nan], 0.2, None),
        (1.0, [0.0, 1.0], 0.2, [-1.0, 2.0]),
        (1.0, [0.0, 1.0], 0.2, [0.0, 0.0]),
        ([1.0, 2.0, 3.0], [0.0, 1.0], 0.2, None),
        (np.inf, [0.0, 1.0], 0.2, None),
        (1.0, [0.0, 1.0], np.inf, None),
        (1.0, [0.0, 1.0], 1j, None),
        (1.0, [1e308], 1e100, None),
    ],
)
def test_screen_invalid_inputs(incident, rm, lam, weights):
    for fn in (screen_polarisation, eqx.filter_jit(screen_polarisation)):
        with pytest.raises(Exception):
            fn(incident, jnp.asarray(rm), jnp.asarray(lam), weights)
