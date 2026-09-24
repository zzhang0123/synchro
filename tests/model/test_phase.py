"""Phase routes: Taylor weights, screens against synchro.rm/faraday, AD and JIT."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from synchro.constants import C_SI_M
from synchro.faraday import emission_polarisation
from synchro.model.phase import (
    CumulantScreen,
    EmpiricalScreen,
    GaussianScreen,
    PhaseWeights,
    TaylorPhase,
    phase_coordinate,
)
from synchro.rm import burn_depolarisation, screen_polarisation

NU = np.array([[1.0e8, 1.5e8], [4.0e8, 1.4e9], [2.0e9, 1.0e10]])
TAU = 2 * (C_SI_M / NU) ** 2


def test_phase_coordinate_is_two_lambda_squared_in_metres():
    assert C_SI_M == 2.99792458e8
    tau = phase_coordinate(NU)
    assert tau.shape == NU.shape
    assert_allclose(np.asarray(tau), TAU, rtol=1e-15)
    assert_allclose(float(phase_coordinate(1.0e8)), 2 * (2.99792458) ** 2, rtol=1e-15)
    grad = jax.grad(lambda nu: phase_coordinate(nu))(jnp.asarray(1.0e8))
    assert_allclose(float(grad), -4 * C_SI_M**2 / 1.0e8**3, rtol=1e-14)
    for bad in [0.0, -1.0e8, np.nan, np.inf]:
        with pytest.raises(Exception):
            jax.jit(phase_coordinate)(jnp.asarray(bad))
    with pytest.raises(ValueError):
        phase_coordinate(jnp.asarray(1.0e8 + 0j))


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 5])
def test_taylor_weights_match_the_formula(degree):
    phase = TaylorPhase(degree)
    depth_ref, s_depth = 37.0, 12.5
    weights = phase(
        jnp.asarray(TAU), depth_ref=jnp.asarray(depth_ref), s_depth=jnp.asarray(s_depth)
    )
    assert weights.shape == (degree + 1, *NU.shape)
    assert jnp.iscomplexobj(weights)
    expected = np.stack(
        [
            np.exp(1j * TAU * depth_ref) * (1j * TAU * s_depth) ** b / math.factorial(b)
            for b in range(degree + 1)
        ]
    )
    assert_allclose(np.asarray(weights), expected, rtol=1e-13, atol=1e-300)
    assert phase.n_weights == degree + 1 and phase.max_b() == degree
    assert phase.forced_assumptions() == ()
    assert phase.name == "taylor" and isinstance(phase.LABEL, str)
    assert hash(phase.describe()) == hash(TaylorPhase(degree).describe())
    assert isinstance(phase, PhaseWeights)


def test_taylor_scalar_tau_and_degree_validation():
    weights = TaylorPhase(2)(
        jnp.asarray(0.7), depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0)
    )
    assert weights.shape == (3,)
    assert_allclose(np.asarray(weights), [1.0, 0.7j, -0.245], atol=1e-15)
    for bad in [-1, 1.5, True, "2"]:
        with pytest.raises(ValueError):
            TaylorPhase(bad)


def test_gaussian_screen_equals_burn_depolarisation():
    mean, sigma = 43.0, 9.0
    screen = GaussianScreen(jnp.asarray(mean), jnp.asarray(sigma))
    lam = C_SI_M / NU
    expected = np.asarray(burn_depolarisation(1.0, mean, sigma**2, lam))
    weights = screen(
        jnp.asarray(TAU), depth_ref=jnp.asarray(999.0), s_depth=jnp.asarray(3.0)
    )
    assert weights.shape == (1, *NU.shape)
    assert_allclose(np.asarray(weights)[0], expected, rtol=1e-13, atol=1e-300)
    # Reference depth and scale are ignored by construction (documented).
    other = screen(
        jnp.asarray(TAU), depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0)
    )
    assert_allclose(np.asarray(other), np.asarray(weights), rtol=0, atol=0)
    assert screen.n_weights == 1 and screen.max_b() == 0
    names = tuple(record.name for record in screen.forced_assumptions())
    assert names == ("independent_screen", "gaussian_screen")
    for record in screen.forced_assumptions():
        assert ("depth",) in record.groups
        assert record.to_dict()["name"] == record.name
    assert screen.forced_assumptions()[1].hyper == (mean, sigma)
    assert isinstance(screen, PhaseWeights)
    with pytest.raises(Exception):
        GaussianScreen(jnp.asarray(1.0), jnp.asarray(-1.0))(
            jnp.asarray(TAU), depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0)
        )


@pytest.mark.parametrize("sigma", [0.0, 1e-3, 1.0, 1e3, 1e6])
def test_gaussian_screen_extreme_dispersion_stays_finite(sigma):
    screen = GaussianScreen(jnp.asarray(5.0), jnp.asarray(sigma))
    weights = screen(
        jnp.asarray(TAU), depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0)
    )
    assert np.all(np.isfinite(np.asarray(weights)))
    assert np.all(np.abs(np.asarray(weights)) <= 1 + 1e-15)
    grad = jax.grad(
        lambda s: jnp.sum(
            GaussianScreen(jnp.asarray(5.0), s)(
                jnp.asarray(TAU), depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0)
            ).real
        )
    )(jnp.asarray(sigma))
    assert np.isfinite(float(grad))


def test_cumulant_screen_reduces_to_gaussian_and_pins_the_signs():
    mean, sigma = 11.0, 4.0
    gaussian = GaussianScreen(jnp.asarray(mean), jnp.asarray(sigma))
    cumulant = CumulantScreen(jnp.asarray([mean, sigma**2, 0.0, 0.0]))
    kwargs = dict(depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0))
    assert_allclose(
        np.asarray(cumulant(jnp.asarray(TAU), **kwargs)),
        np.asarray(gaussian(jnp.asarray(TAU), **kwargs)),
        rtol=1e-14,
    )
    # Sign pin: for a skewed discrete screen, log <e^{i tau phi}> - g_4(tau) = O(tau^5).
    depths = np.array([-2.0, -0.5, 0.7, 3.5, 6.0])
    weights = np.array([0.1, 0.3, 0.35, 0.2, 0.05])
    m = np.sum(weights * depths)
    central = depths - m
    k2 = np.sum(weights * central**2)
    k3 = np.sum(weights * central**3)
    k4 = np.sum(weights * central**4) - 3 * k2**2
    empirical = EmpiricalScreen(jnp.asarray(depths), jnp.asarray(weights))
    closed = CumulantScreen(jnp.asarray([m, k2, k3, k4]))
    residuals = []
    for tau in [0.2, 0.1, 0.05]:
        exact = np.asarray(empirical(jnp.asarray(tau), **kwargs))[0]
        approx = np.asarray(closed(jnp.asarray(tau), **kwargs))[0]
        residuals.append(abs(np.log(exact) - np.log(approx)))
    exponents = np.diff(np.log(residuals)) / np.log(0.5)
    assert np.all(exponents > 4.8) and np.all(exponents < 5.2)
    # Sign pins on a Bernoulli screen (depth 1 with probability p, else 0),
    # whose cumulants are closed-form: k3 = p(1-p)(1-2p), k4 = p(1-p)(1-6p(1-p)).
    p = 0.25
    bernoulli = EmpiricalScreen(jnp.asarray([0.0, 1.0]), jnp.asarray([1 - p, p]))
    q = p * (1 - p)
    exact_kappa = [p, q, q * (1 - 2 * p), q * (1 - 6 * q)]
    right = CumulantScreen(jnp.asarray(exact_kappa))
    for flip in (2, 3):
        wrong = list(exact_kappa)
        wrong[flip] = -wrong[flip]
        bad = CumulantScreen(jnp.asarray(wrong))
        for tau in [0.1, 0.05]:
            exact = np.log(np.asarray(bernoulli(jnp.asarray(tau), **kwargs))[0])
            good = abs(exact - np.log(np.asarray(right(jnp.asarray(tau), **kwargs))[0]))
            worse = abs(exact - np.log(np.asarray(bad(jnp.asarray(tau), **kwargs))[0]))
            assert worse > 10 * good
    assert cumulant.n_weights == 1 and cumulant.max_b() == 0
    names = tuple(record.name for record in cumulant.forced_assumptions())
    assert names == ("independent_screen", "cumulant_screen")
    assert cumulant.forced_assumptions()[1].discrepancy_kind == "unbounded"
    assert cumulant.exponent_remainder(jnp.asarray(TAU)) is None
    bounded = CumulantScreen(
        jnp.asarray([mean, sigma**2, 0.0, 0.0]), g5_bound=jnp.asarray(2.0)
    )
    assert bounded.forced_assumptions()[1].discrepancy_kind == "bound"
    eps5 = bounded.exponent_remainder(jnp.asarray(TAU))
    assert_allclose(np.asarray(eps5), np.abs(TAU) ** 5 * 2.0 / 120, rtol=1e-15)
    assert isinstance(cumulant, PhaseWeights)
    with pytest.raises(ValueError):
        CumulantScreen(jnp.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(Exception):
        CumulantScreen(jnp.asarray([1.0, -2.0, 0.0, 0.0]))(jnp.asarray(TAU), **kwargs)


def test_empirical_screen_with_a_delta_channel_matches_screen_polarisation():
    depths = np.array([-30.0, 4.0, 18.5, 77.0])
    weights = np.array([1.0, 2.5, 0.5, 4.0])
    screen = EmpiricalScreen(jnp.asarray(depths), jnp.asarray(weights))
    lam = C_SI_M / NU
    expected = np.asarray(screen_polarisation(1.0, depths, lam, weights))
    kwargs = dict(depth_ref=jnp.asarray(0.0), s_depth=jnp.asarray(1.0))
    actual = np.asarray(screen(jnp.asarray(TAU), **kwargs))
    assert actual.shape == (1, *NU.shape)
    assert_allclose(actual[0], expected, atol=1e-14, rtol=0)
    assert_allclose(
        actual[0],
        np.asarray(emission_polarisation(1.0, depths, lam, weights)),
        atol=1e-14,
        rtol=0,
    )
    # Relative weights are normalised (overflow-safe) and default to uniform.
    huge = EmpiricalScreen(jnp.asarray(depths), jnp.asarray(weights) * 1e300)
    assert_allclose(np.asarray(huge(jnp.asarray(TAU), **kwargs)), actual, atol=1e-14)
    uniform = EmpiricalScreen(jnp.asarray(depths))
    assert_allclose(
        np.asarray(uniform(jnp.asarray(TAU), **kwargs))[0],
        np.mean(np.exp(1j * TAU[..., None] * depths), axis=-1),
        atol=1e-14,
    )
    assert screen.n_weights == 1 and screen.max_b() == 0
    names = tuple(record.name for record in screen.forced_assumptions())
    assert names == ("independent_screen",)
    assert screen.forced_assumptions()[0].closure_kind == "fixed_table"
    assert isinstance(screen, PhaseWeights)
    assert hash(screen.describe())
    for bad_depths, bad_weights in [
        (np.zeros((2, 2)), None),
        (np.array([]), None),
        (depths, weights[:-1]),
        (depths, -weights),
        (depths, 0 * weights),
        (np.array([np.nan, 1.0]), None),
    ]:
        with pytest.raises(Exception):
            eqx.filter_jit(
                EmpiricalScreen(
                    jnp.asarray(bad_depths),
                    None if bad_weights is None else jnp.asarray(bad_weights),
                )
            )(jnp.asarray(TAU), **kwargs)


def _routes():
    return {
        "taylor": TaylorPhase(2),
        "gaussian": GaussianScreen(jnp.asarray(3.0), jnp.asarray(2.0)),
        "cumulant": CumulantScreen(jnp.asarray([3.0, 4.0, 0.5, -0.2])),
        "empirical": EmpiricalScreen(
            jnp.asarray([1.0, 2.0, 5.0]), jnp.asarray([1.0, 1.0, 2.0])
        ),
    }


@pytest.mark.parametrize("name", ["taylor", "gaussian", "cumulant", "empirical"])
def test_jit_and_grad_through_every_route(name):
    route = _routes()[name]
    tau = jnp.asarray(TAU)

    def evaluate(route, tau, depth_ref, s_depth):
        return route(tau, depth_ref=depth_ref, s_depth=s_depth)

    compiled = eqx.filter_jit(evaluate)
    eager = evaluate(route, tau, jnp.asarray(1.5), jnp.asarray(0.5))
    assert_allclose(
        np.asarray(compiled(route, tau, jnp.asarray(1.5), jnp.asarray(0.5))),
        np.asarray(eager),
        rtol=1e-13,
        atol=1e-30,
    )

    def loss(route, tau, depth_ref, s_depth):
        w = evaluate(route, tau, depth_ref, s_depth)
        return jnp.sum(w.real * jnp.arange(1, w.size + 1).reshape(w.shape) + w.imag)

    grads = eqx.filter_grad(loss)(route, tau, jnp.asarray(1.5), jnp.asarray(0.5))
    leaves = jax.tree_util.tree_leaves(grads)
    # TaylorPhase has no array leaves; the screens do.
    assert (name == "taylor") == (not leaves)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    tau_grad = jax.grad(lambda t: loss(route, t, jnp.asarray(1.5), jnp.asarray(0.5)))(
        tau
    )
    assert np.all(np.isfinite(np.asarray(tau_grad)))
    # Central differences in tau confirm the tau gradient.
    h = 1e-6
    flat = np.asarray(tau).ravel()
    numeric = np.zeros_like(flat)
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += h
        minus[i] -= h
        numeric[i] = (
            float(
                loss(
                    route,
                    jnp.asarray(plus.reshape(NU.shape)),
                    jnp.asarray(1.5),
                    jnp.asarray(0.5),
                )
            )
            - float(
                loss(
                    route,
                    jnp.asarray(minus.reshape(NU.shape)),
                    jnp.asarray(1.5),
                    jnp.asarray(0.5),
                )
            )
        ) / (2 * h)
    assert_allclose(np.asarray(tau_grad).ravel(), numeric, rtol=1e-5, atol=1e-6)
    ref_grad = jax.grad(lambda d: loss(route, tau, d, jnp.asarray(0.5)))(
        jnp.asarray(1.5)
    )
    if name == "taylor":
        assert abs(float(ref_grad)) > 0
    else:
        assert float(ref_grad) == 0.0


def test_taylor_large_phase_and_extreme_scales():
    phase = TaylorPhase(4)
    tau = phase_coordinate(jnp.asarray([1.0e7, 1.0e8]))
    weights = phase(tau, depth_ref=jnp.asarray(1e4), s_depth=jnp.asarray(1e3))
    assert np.all(np.isfinite(np.asarray(weights)))
    assert_allclose(np.abs(np.asarray(weights)[0]), 1.0, rtol=1e-15)
    magnitudes = np.abs(np.asarray(weights))[:, 0]
    tau0 = float(tau[0])
    assert_allclose(
        magnitudes,
        [(tau0 * 1e3) ** b / math.factorial(b) for b in range(5)],
        rtol=1e-13,
    )
    with pytest.raises(Exception):
        jax.jit(
            lambda t: phase(t, depth_ref=jnp.asarray(1e308), s_depth=jnp.asarray(1.0))
        )(tau)
