"""Independent radiation and statistical-interface regressions.

These finite tests distinguish numerical kernel accuracy, a physical fixed-B
response, and finite Taylor/Bell algebra. They do not certify a Galactic PDF
closure or an experiment's total modelling error.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma, jv, jvp, kv

from syncmoments.bessel import bessel_jn_and_prime, bessel_kn
from syncmoments.cumulants import cumulant_expansion, vector_cumulant_expansion
from syncmoments.derivatives import derivative_spectra
from syncmoments.expansion import build_expansion
from syncmoments.sed import log_parabola_running_index
from syncmoments.stokes import C_CGS, E_ESU, M_E, stokes_harmonic
from syncmoments.ultrarel import F, G


@pytest.mark.parametrize("n,x", [(512, 512.0), (1000, 1000.0), (3200, 2500.0)])
def test_high_harmonics_and_exponential_tails_match_scipy(n, x):
    got, prime = bessel_jn_and_prime(n, x)
    np.testing.assert_allclose(got, jv(n, x), rtol=2e-10, atol=0)
    np.testing.assert_allclose(prime, jvp(n, x), rtol=2e-9, atol=0)


@pytest.mark.parametrize(
    "n,x", [(1, 0.01), (5, 3.0), (80, 79.9), (80, 80.0), (80, 80.1)]
)
def test_bessel_derivatives_across_contour_choice(n, x):
    def f(value):
        return bessel_jn_and_prime(n, value)[0]

    np.testing.assert_allclose(jax.grad(f)(x), jvp(n, x), rtol=2e-9, atol=2e-13)
    np.testing.assert_allclose(
        jax.grad(jax.grad(f))(x), jvp(n, x, 2), rtol=2e-8, atol=2e-12
    )


@pytest.mark.parametrize("order", [0.0, 2 / 3, 5 / 3])
def test_modified_bessel_continuous_small_argument_range(order):
    xs = np.unique(np.r_[np.geomspace(1e-12, 100, 40), 0.0099, 0.01, 0.0101])
    np.testing.assert_allclose(bessel_kn(order, xs), kv(order, xs), rtol=2e-10, atol=0)


def test_continuum_kernel_and_gradient_at_old_switch():
    for x in (1e-7, 0.0099, 0.01, 0.0101, 0.1, 1.0, 20.0):
        # Independent transformed integral remains well-conditioned at small x.
        reference = (
            x
            * quad(
                lambda u: np.exp(u) * kv(5 / 3, np.exp(u)),
                np.log(x),
                np.log(x + 60),
                epsabs=1e-11,
                epsrel=1e-11,
            )[0]
        )
        np.testing.assert_allclose(F(x), reference, rtol=3e-10, atol=0)
        np.testing.assert_allclose(G(x), x * kv(2 / 3, x), rtol=2e-10, atol=0)
        fp = reference / x - x * kv(5 / 3, x)
        np.testing.assert_allclose(jax.grad(F)(x), fp, rtol=3e-9, atol=1e-12)
    assert F(0.0) == 0.0 and G(0.0) == 0.0
    assert np.isnan(F(-1.0)) and np.isnan(G(-1.0))


@pytest.mark.parametrize("theta", [0.0, np.pi])
def test_harmonic_axis_limits(theta):
    gamma, alpha = 2.0, 0.7
    beta = np.sqrt(1 - gamma**-2)
    bp, bt = beta * np.cos(alpha), beta * np.sin(alpha)
    ct = np.cos(theta)
    d = 1 - bp * ct
    expected_i = bt**2 / (2 * d**3)
    np.testing.assert_allclose(
        stokes_harmonic(1, gamma, alpha, theta),
        [expected_i, 0.0, ct * expected_i],
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        stokes_harmonic(2, gamma, alpha, theta), [0.0, 0.0, 0.0], atol=1e-28
    )


def _physical_scipy(n, gamma, alpha, theta, field):
    beta = np.sqrt(1 - gamma**-2)
    bp, bt = beta * np.cos(alpha), beta * np.sin(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    d = 1 - bp * ct
    x = n * bt * st / d
    ap, at = (ct - bp) * jv(n, x) / st, bt * jvp(n, x)
    wb = E_ESU * field / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wb**2 / (2 * np.pi * C_CGS) * n**2 / d**3
    return pref * np.array([ap**2 + at**2, ap**2 - at**2, 2 * ap * at])


def test_fixed_field_physical_derivatives_and_expansion():
    n, gamma, alpha, theta, field = 7, 5.0, 0.7, 1.0, 3e-6
    val, grad, hess = derivative_spectra(n, gamma, alpha, theta, B=field)

    def f(g):
        return _physical_scipy(n, g, alpha, theta, field)

    step = 2e-3
    fd1 = (
        f(gamma - 2 * step)
        - 8 * f(gamma - step)
        + 8 * f(gamma + step)
        - f(gamma + 2 * step)
    ) / (12 * step)
    fd2 = (
        -f(gamma + 2 * step)
        + 16 * f(gamma + step)
        - 30 * f(gamma)
        + 16 * f(gamma - step)
        - f(gamma - 2 * step)
    ) / (12 * step**2)
    np.testing.assert_allclose(val, f(gamma), rtol=2e-11)
    np.testing.assert_allclose(grad[:, 0], fd1, rtol=2e-8, atol=0)
    np.testing.assert_allclose(hess[:, 0, 0], fd2, rtol=3e-6, atol=0)
    expansion = build_expansion([n], gamma, alpha, theta, B=field)
    np.testing.assert_allclose(
        expansion(jnp.zeros(3), jnp.zeros((3, 3)))[0], val, rtol=1e-12
    )
    # Normalisation has gamma dependence; multiplying a normalised derivative
    # by the physical value prefactor does not give the physical derivative.
    norm, norm_grad, _ = derivative_spectra(n, gamma, alpha, theta)
    pref = val[0] / norm[0]
    np.testing.assert_allclose(
        grad[:, 0], pref * (norm_grad[:, 0] - 2 * norm / gamma), rtol=2e-11
    )


def test_scalar_bell_sum_is_traceable_and_matches_same_order_moments():
    fn = jax.jit(lambda mu: cumulant_expansion([1.0] * 5, [mu, 0.25]))
    mu = 0.3
    moments = [
        1,
        mu,
        mu**2 + 0.25,
        mu**3 + 3 * mu * 0.25,
        mu**4 + 6 * mu**2 * 0.25 + 3 * 0.25**2,
    ]
    expected = sum(m / math.factorial(k) for k, m in enumerate(moments))
    np.testing.assert_allclose(fn(mu), expected, rtol=1e-13)
    assert np.isfinite(jax.grad(fn)(mu))
    assert abs(float(fn(mu)) - np.exp(mu + 0.25 / 2)) > 1e-4


def test_vector_bell_rejects_unsupported_orders():
    with pytest.raises(ValueError, match="four|4"):
        vector_cumulant_expansion(
            [jnp.ones((1,) * k) for k in range(6)],
            [jnp.ones((1,) * k) for k in range(1, 6)],
        )


def test_log_parabola_mapping_contains_kernel_offset():
    p0, a = 2.5, 0.02
    q = (p0 - 1) / 2
    mx = (
        np.log(2)
        - 1 / (q + 1)
        + 0.5 * digamma(q / 2 + 11 / 6)
        + 0.5 * digamma(q / 2 + 1 / 6)
    )
    for nu in (0.1, 1.0, 10.0):
        expected = q + a / 2 * (np.log(nu) - mx)
        np.testing.assert_allclose(
            log_parabola_running_index(nu, p0, a), expected, rtol=1e-13
        )


def test_absolute_emissivity_uses_normalised_angular_measure():
    from syncmoments.sed import power_law_emissivity_abs

    nu, field, density = 1e8, 3e-6, 2.0
    nu_b = E_ESU * field / (2 * np.pi * M_E * C_CGS)
    pref = np.sqrt(3) * E_ESU**3 * density * field / (4 * np.pi * M_E * C_CGS**2)
    for p in (2.0, 2.5, 3.0):
        q = (p - 1) / 2
        # Swap the positive F and energy integrals, then integrate K directly.
        mellin = quad(
            lambda x: x ** (q + 1) * kv(5 / 3, x), 0, np.inf, epsabs=1e-11, epsrel=1e-11
        )[0] / (q + 1)
        energy = 0.5 * (1.5 * nu_b / nu) ** q * mellin
        angular = quad(
            lambda angle: 0.5 * np.sin(angle) ** (q + 2),
            0,
            np.pi,
            epsabs=1e-12,
            epsrel=1e-12,
        )[0]
        reference = pref * energy * angular
        np.testing.assert_allclose(
            power_law_emissivity_abs(nu, p, density, field), reference, rtol=3e-10
        )
        theta = 0.8
        directional = pref * energy * np.sin(theta) ** (q + 1)
        np.testing.assert_allclose(
            power_law_emissivity_abs(nu, p, density, field, theta=theta),
            directional,
            rtol=3e-10,
        )


def test_jit_resolution_guard_and_explicit_static_resolution():
    default = jax.jit(lambda n, x: bessel_jn_and_prime(n, x)[0])
    assert np.isnan(default(3200, 2500.0))
    resolved = jax.jit(lambda n, x: bessel_jn_and_prime(n, x, n_nodes=16384)[0])
    np.testing.assert_allclose(
        resolved(3200, 2500.0), jv(3200, 2500.0), rtol=2e-10, atol=0
    )


def test_finite_integral_tail_bounds_are_separate_from_quadrature():
    from syncmoments.bessel import bessel_kn_tail_bound
    from syncmoments.ultrarel import F_tail_bound

    # Deliberately short cutoff makes the omitted positive tail measurable;
    # default cutoff50 would be below the floating-point comparison floor.
    for x in (0.01, 1.0):
        for order in (0.0, 2 / 3, 5 / 3):
            cutoff = 5.0
            tmax = np.arccosh(1 + (cutoff + 2 * order) / x)
            omitted = quad(
                lambda t: np.exp(-x * np.cosh(t)) * np.cosh(order * t),
                tmax,
                tmax + 8,
                epsabs=1e-20,
                epsrel=1e-11,
            )[0]
            bound = float(bessel_kn_tail_bound(order, x, tail_cutoff=cutoff))
            assert 0 < omitted <= bound
        order = 5 / 3
        tmax = np.arccosh(1 + (5 + 2 * order) / x)
        omitted = (
            x
            * quad(
                lambda t: np.exp(-x * np.cosh(t)) * np.cosh(order * t) / np.cosh(t),
                tmax,
                tmax + 8,
                epsabs=1e-20,
                epsrel=1e-11,
            )[0]
        )
        assert 0 < omitted <= float(F_tail_bound(x, tail_cutoff=5.0))


def test_absolute_emissivity_zero_field_limit():
    from syncmoments.sed import power_law_emissivity_abs

    for p in (0.5, 1.0, 2.5):
        assert power_law_emissivity_abs(1e8, p, 1.0, 0.0) == 0.0
        assert power_law_emissivity_abs(1e8, p, 1.0, 0.0, theta=0.8) == 0.0


def test_extended_log_parabola_curvature_domain():
    assert np.isnan(log_parabola_running_index(1.0, 2.5, -0.01))
    np.testing.assert_allclose(
        log_parabola_running_index(jnp.array([0.1, 1.0, 10.0]), 2.5, 0.0), 0.75
    )


def test_shared_constants_and_absolute_larmor_normalisation():
    from syncmoments import constants, sed, stokes

    # Independent numerical SI definitions/conversions, not module aliases.
    charge = 1.602176634e-19 * 2.99792458e9
    mass = 9.1093837139e-31 * 1e3
    speed = 299792458.0 * 100
    np.testing.assert_allclose(
        [constants.E_ESU, constants.M_E, constants.C_CGS],
        [charge, mass, speed],
        rtol=2e-16,
    )
    assert stokes.E_ESU == sed.E_ESU == constants.E_ESU
    gamma, alpha, field = 5.0, 0.7, 3e-6
    expected = (
        2
        * charge**4
        * field**2
        * (gamma**2 - 1)
        * np.sin(alpha) ** 2
        / (3 * mass**2 * speed**3)
    )
    np.testing.assert_allclose(
        stokes.larmor_power(gamma, alpha, field), expected, rtol=5e-15
    )
