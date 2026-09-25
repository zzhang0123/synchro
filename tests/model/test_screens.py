"""Laplace and shifted-Gamma screens (main text ``eq: illustrated screens``).

Oracles are independent of the JAX code: SciPy quadrature of the screen
densities, the closed forms printed in the manuscript, and an explicit
per-atom channel sum for the route through ``direct_channel_average``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import integrate, stats
from scipy.special import eval_legendre

import syncmoments  # noqa: F401
from syncmoments.model.channels import Channels
from syncmoments.model.phase import GaussianScreen
from syncmoments.model.predict import direct_channel_average
from syncmoments.model.screens import GammaScreen, LaplaceScreen

from _predict_helpers import (
    B0,
    DEPTH_REF,
    GAMMA0,
    LINE_NU,
    S_DEPTH,
    TAU,
    nine_atoms,
    polynomial_kernel,
    samples_of,
)


def density_cf(pdf, tau, lo, hi):
    """Characteristic function by direct SciPy quadrature of the density."""
    re = integrate.quad(lambda x: pdf(x) * np.cos(tau * x), lo, hi, limit=400)[0]
    im = integrate.quad(lambda x: pdf(x) * np.sin(tau * x), lo, hi, limit=400)[0]
    return re + 1j * im


TAUS = [0.0, 0.03, 0.4, 1.0, 2.5]


@pytest.mark.parametrize("tau", TAUS)
def test_laplace_matches_density_quadrature(tau):
    mean, sigma = 1.3, 0.8
    law = stats.laplace(loc=mean, scale=sigma / np.sqrt(2))
    assert_allclose(law.std(), sigma, rtol=1e-14)
    expected = density_cf(law.pdf, tau, mean - 60 * sigma, mean + 60 * sigma)
    got = LaplaceScreen(jnp.asarray(mean), jnp.asarray(sigma))(jnp.asarray(tau))
    assert got.shape == (1,)
    assert_allclose(complex(got[0]), expected, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("shape", [0.7, 4.0, 25.0])
@pytest.mark.parametrize("tau", TAUS)
def test_gamma_matches_density_quadrature(tau, shape, sign):
    mean, sigma = -0.4, 0.6
    root = np.sqrt(shape)
    y = stats.gamma(shape)
    # depth = mean + sign sigma (Y - k)/sqrt(k); change of variables in Y.
    re = integrate.quad(
        lambda v: y.pdf(v) * np.cos(tau * (mean + sign * sigma * (v - shape) / root)),
        0,
        np.inf,
        limit=400,
    )[0]
    im = integrate.quad(
        lambda v: y.pdf(v) * np.sin(tau * (mean + sign * sigma * (v - shape) / root)),
        0,
        np.inf,
        limit=400,
    )[0]
    got = GammaScreen(jnp.asarray(mean), jnp.asarray(sigma), shape=shape, sign=sign)
    assert_allclose(
        complex(got(jnp.asarray(tau))[0]), re + 1j * im, rtol=1e-8, atol=1e-11
    )


def test_closed_forms_and_position_angle_of_the_manuscript_figure():
    t = np.linspace(0.0, 6.0, 61)
    tau = jnp.asarray(t)  # sigma = 1, mean = 0: t = tau sigma
    gauss = np.asarray(GaussianScreen(jnp.asarray(0.0), jnp.asarray(1.0))(tau)[0])
    lap = np.asarray(LaplaceScreen(jnp.asarray(0.0), jnp.asarray(1.0))(tau)[0])
    gam = np.asarray(GammaScreen(jnp.asarray(0.0), jnp.asarray(1.0))(tau)[0])
    assert_allclose(gauss, np.exp(-(t**2) / 2), atol=1e-15)
    assert_allclose(lap, 1 / (1 + t**2 / 2), atol=1e-15)
    assert_allclose(gam, np.exp(-2j * t) * (1 - 1j * t / 2) ** (-4), rtol=1e-13)
    # Continuous residual position angle -t + 2 arctan(t/2) (half the phase).
    angle = 0.5 * np.unwrap(np.angle(gam))
    assert_allclose(angle, -t + 2 * np.arctan(t / 2), atol=1e-12)
    # Symmetric Laplace screen: no position-angle shift; depolarisation differs.
    assert np.max(np.abs(np.angle(lap))) < 1e-15
    assert np.all(np.abs(lap[1:]) > np.abs(gauss[1:]))


def test_mean_shift_is_a_pure_phase():
    tau = jnp.linspace(0.0, 3.0, 7)
    for cls in (LaplaceScreen, GammaScreen):
        base = np.asarray(cls(jnp.asarray(0.0), jnp.asarray(0.5))(tau)[0])
        shifted = np.asarray(cls(jnp.asarray(2.0), jnp.asarray(0.5))(tau)[0])
        assert_allclose(shifted, base * np.exp(2j * np.asarray(tau)), rtol=1e-13)


def test_records_validation_and_autodiff():
    lap = LaplaceScreen(jnp.asarray(0.5), jnp.asarray(0.2))
    gam = GammaScreen(jnp.asarray(0.5), jnp.asarray(0.2), shape=2.0, sign=-1)
    assert [r.name for r in lap.forced_assumptions()] == [
        "independent_screen",
        "laplace_screen",
    ]
    assert [r.name for r in gam.forced_assumptions()] == [
        "independent_screen",
        "gamma_screen",
    ]
    for route in (lap, gam):
        assert route.max_b() == 0 and route.n_weights == 1
        assert all(
            r.discrepancy_kind == "unbounded" for r in route.forced_assumptions()
        )
        assert hash(route.describe()) is not None
    for bad in (dict(shape=0.0), dict(shape=np.inf), dict(sign=0)):
        with pytest.raises(ValueError):
            GammaScreen(jnp.asarray(0.0), jnp.asarray(1.0), **bad)
    with pytest.raises(Exception):
        LaplaceScreen(jnp.asarray(0.0), jnp.asarray(-1.0))(jnp.asarray(1.0))
    # d|C|/d sigma at tau: Laplace -tau^2 sigma / (1 + tau^2 sigma^2/2)^2.
    f = jax.grad(
        lambda s: jnp.abs(LaplaceScreen(jnp.asarray(0.0), s)(jnp.asarray(1.5))[0])
    )
    s0 = 0.7
    assert_allclose(
        f(jnp.asarray(s0)), -(1.5**2) * s0 / (1 + 1.5**2 * s0**2 / 2) ** 2, rtol=1e-12
    )
    g = jax.jit(lambda tau: gam(tau))(jnp.asarray([0.1, 1.0]))
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.parametrize("route_name", ["laplace", "gamma"])
def test_direct_channel_average_applies_the_screen_per_line(route_name):
    kernel, c = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    pop = nine_atoms(constant_depth=False)
    samples = samples_of(pop)
    mean, sigma = DEPTH_REF, 0.3 * S_DEPTH
    route = (
        LaplaceScreen(jnp.asarray(mean), jnp.asarray(sigma))
        if route_name == "laplace"
        else GammaScreen(jnp.asarray(mean), jnp.asarray(sigma), shape=4.0)
    )
    pred = direct_channel_average(samples, kernel, channels, amplitude=1.0, phase=route)
    w = pop["w"] / pop["w"].sum()
    emitted = np.zeros(len(TAU), dtype=complex)
    for n in range(len(w)):
        zg, zB = (pop["gamma"][n] - GAMMA0) / GAMMA0, (pop["B"][n] - B0) / B0
        pl = eval_legendre(np.arange(3), pop["mu"][n])
        pk = eval_legendre(np.arange(3), pop["eta"][n])
        KQ = np.einsum(
            "jlkrt,l,k,r,t->j", c[1], pl, pk, zg ** np.arange(3), zB ** np.arange(3)
        )
        emitted += w[n] * np.exp(2j * pop["phi"][n]) * KQ
    t = TAU * sigma
    if route_name == "laplace":
        cf = np.exp(1j * TAU * mean) / (1 + t**2 / 2)
    else:
        cf = np.exp(1j * TAU * mean) * np.exp(-2j * t) * (1 - 1j * t / 2) ** (-4)
    got = np.asarray(pred.stokes[:, 1]) + 1j * np.asarray(pred.stokes[:, 2])
    assert_allclose(got, emitted * cf, rtol=1e-12)
    names = [name for name, _ in pred.budget.assumption]
    assert names == ["independent_screen", f"{route_name}_screen"]
    assert pred.budget.total().kind == "unbounded"


@pytest.mark.parametrize("route_name", ["laplace", "gamma"])
def test_build_basis_and_predict_route_matches_direct_and_oracle(route_name):
    """The documented ``build_basis``/``predict`` route (``depth_degree = 0``).

    The polynomial kernel's finite response is exact, so ``predict`` equals
    ``direct_channel_average`` and the per-atom NumPy oracle to roundoff.
    """
    from syncmoments.model.basis import build_basis
    from syncmoments.model.index import Truncation
    from syncmoments.model.moments import JointMoments
    from syncmoments.model.predict import predict

    from _predict_helpers import reference, support

    kernel, c = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    pop = nine_atoms(constant_depth=False)
    samples = samples_of(pop)
    mean, sigma = DEPTH_REF, 0.3 * S_DEPTH
    route = (
        LaplaceScreen(jnp.asarray(mean), jnp.asarray(sigma))
        if route_name == "laplace"
        else GammaScreen(jnp.asarray(mean), jnp.asarray(sigma), shape=4.0)
    )
    basis = build_basis(
        kernel,
        channels,
        Truncation(2, 2, 2, depth_degree=0),
        reference(),
        support=support(False),
        phase=route,
        convergence=False,
    )
    moments = JointMoments.from_samples(samples, basis.index, reference())
    pred = predict(basis, moments, amplitude=1.0)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, phase=route
    )
    scale = np.max(np.abs(np.asarray(direct.stokes)))
    assert_allclose(
        np.asarray(pred.stokes), np.asarray(direct.stokes), rtol=0, atol=1e-12 * scale
    )
    w = pop["w"] / pop["w"].sum()
    emitted = np.zeros(len(TAU), dtype=complex)
    for n in range(len(w)):
        zg, zB = (pop["gamma"][n] - GAMMA0) / GAMMA0, (pop["B"][n] - B0) / B0
        pl = eval_legendre(np.arange(3), pop["mu"][n])
        pk = eval_legendre(np.arange(3), pop["eta"][n])
        KQ = np.einsum(
            "jlkrt,l,k,r,t->j", c[1], pl, pk, zg ** np.arange(3), zB ** np.arange(3)
        )
        emitted += w[n] * np.exp(2j * pop["phi"][n]) * KQ
    t = TAU * sigma
    if route_name == "laplace":
        cf = np.exp(1j * TAU * mean) / (1 + t**2 / 2)
    else:
        cf = np.exp(1j * TAU * mean) * np.exp(-2j * t) * (1 - 1j * t / 2) ** (-4)
    got = np.asarray(pred.stokes[:, 1]) + 1j * np.asarray(pred.stokes[:, 2])
    assert_allclose(got, emitted * cf, rtol=1e-12)
    names = [name for name, _ in pred.budget.assumption]
    assert names == ["independent_screen", f"{route_name}_screen"]
    assert all(term.kind == "unbounded" for _, term in pred.budget.assumption)
    assert pred.budget.screen_exponent.kind == "bound"
    assert pred.budget.total().kind == "unbounded"
