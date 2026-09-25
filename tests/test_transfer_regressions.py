"""Independent analytic and SciPy checks of transfer conventions and limits."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.linalg import expm

from syncmoments.conversion import mueller_conversion
from syncmoments.rm import C_CGS, E_ESU, M_E
from syncmoments.transfer import mueller_matrix, transfer_slab


def test_cold_plasma_conversion_sign_and_normalisation():
    nu, ne, b = 8.0e7, 0.03, 5.0e-6
    # Leading cold-dielectric Stokes rate in the paper's Q=parallel-perp basis.
    expected = -(E_ESU**4) * ne * b**2 / (4 * np.pi**2 * M_E**3 * C_CGS**3 * nu**3)
    assert_allclose(mueller_conversion(nu, ne, b), expected, rtol=2e-15)


def test_zero_emissivity_reverse_derivative_matches_augmented_exponential():
    K = np.array(
        [[0.3, 0.04, 0, 0], [0.04, 0.3, 0.6, 0], [0, -0.6, 0.3, -0.2], [0, 0, 0.2, 0.3]]
    )
    ds = 0.7
    expected = np.empty((4, 4))
    for i in range(4):
        aug = np.zeros((5, 5))
        aug[:4, :4] = -K * ds
        aug[i, 4] = ds
        expected[:, i] = expm(aug)[:4, 4]

    def f(e):
        return transfer_slab(jnp.zeros(4), e, jnp.asarray(K), ds)

    assert_allclose(jax.jit(jax.jacrev(f))(jnp.zeros(4)), expected, atol=3e-14)
    assert_allclose(jax.jacfwd(f)(jnp.zeros(4)), expected, atol=3e-14)


def test_zero_absorption_scalar_solution_and_derivative():
    from syncmoments.solutions import uniform_source_intensity

    def f(a):
        return uniform_source_intensity(1.0, 3.0, 1.0, 2.0, eps0=3.0, alpha0=a)

    assert_allclose(f(0.0), 6.0, atol=1e-14)
    assert_allclose(jax.grad(f)(0.0), -6.0, atol=1e-13)
    assert_allclose(jax.grad(jax.grad(f))(0.0), 8.0, atol=1e-12)


def test_kirchhoff_endpoint_terms_against_analytic_weight():
    from syncmoments.kirchhoff import absorption_moments

    lo, hi, g0 = 2.0, 5.0, 3.0

    def N(g):
        return 1.0 + 0.2 * g + 0.03 * g * g

    M = [quad(lambda g: N(g) * (g - g0) ** k, lo, hi)[0] for k in range(3)]
    ig = quad(lambda g: N(g) / g, lo, hi)[0]
    boundary = [N(hi) * (hi - g0) ** k - N(lo) * (lo - g0) ** k for k in range(3)]
    expected = [
        quad(lambda g: (2 * N(g) / g - 0.2 - 0.06 * g) * (g - g0) ** k, lo, hi)[0]
        for k in range(3)
    ]
    assert_allclose(
        absorption_moments(*M, g0, ig, boundary_terms=boundary), expected, atol=3e-14
    )


def test_reduced_slab_rejects_cgs_faraday_parameters():
    from syncmoments.los_moments import moment_driven_slab

    with pytest.raises(ValueError, match="cgs"):
        moment_driven_slab(
            100.0, 30.0, 1.0, 1.0, 0.0, 0.01, 1 / 30.0, 1.0, n_e=0.03, B_par=1e-6
        )


def test_weighted_magnus_matches_ordered_pair_sum():
    from syncmoments.magnus import omega1, omega2, magnus_S

    rng = np.random.default_rng(492)
    K = rng.normal(size=(5, 4, 4)) * 0.03
    ds = np.array([0.2, 0.4, 0.1, 0.7, 0.3])
    A = K * ds[:, None, None]
    expected = sum((A[j] @ A[i] - A[i] @ A[j]) for j in range(5) for i in range(j)) / 2
    assert_allclose(omega1(K, ds), -A.sum(axis=0), atol=1e-16)
    assert_allclose(jax.jit(omega2)(K, ds), expected, atol=1e-16)
    assert_allclose(omega2(K[::-1], ds[::-1]), -expected, atol=1e-16)
    assert np.all(
        np.isfinite(jax.grad(lambda d: jnp.sum(omega2(K, d)))(jnp.asarray(ds)))
    )
    with pytest.raises(ValueError, match="order"):
        magnus_S(jnp.ones(4), K, ds, order=3)


def test_cgs_emission_absorption_prefactors_and_signed_q():
    from syncmoments.kirchhoff import cgs_coefficients_from_moments
    from scipy.special import kv

    nu, g, B, number = 1e8, 2500.0, 5e-6, 2e-9
    aB = 3 * E_ESU * B / (4 * np.pi * M_E * C_CGS)
    amp = np.sqrt(3) * E_ESU**3 * B / (4 * np.pi * M_E * C_CGS**2)
    x = nu / (aB * g * g)
    f = x * quad(lambda z: kv(5 / 3, z), x, np.inf, epsabs=1e-12)[0]
    gg = x * kv(2 / 3, x)
    got = np.asarray(
        cgs_coefficients_from_moments(nu, g, B, number, 0.0, 0.0, number / g)
    )
    # The local emissivity is exact for the point population. Absorption below
    # is specifically the second-order derivative-kernel approximation.
    assert_allclose(got[:2], amp * number * np.array([f, -gg]), rtol=2e-9)
    from syncmoments.kirchhoff import absorption_from_moments, absorption_Q_from_moments

    assert_allclose(
        got[2:],
        amp
        / (2 * M_E)
        * np.array(
            [
                absorption_from_moments(nu, g, aB, number, 0.0, 0.0, number / g),
                absorption_Q_from_moments(nu, g, aB, number, 0.0, 0.0, number / g),
            ]
        ),
        rtol=2e-14,
    )


def test_uniform_internal_rotation_has_sinc_depolarisation():
    # Finite depth tests distributed emission, rather than a near-zero-depth
    # comparison to an external screen that would hide the distinction.
    depth = 1.8
    eps = jnp.array([2.0, -0.7, 0.2, 0.0])
    K = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, depth)
    S = np.asarray(transfer_slab(jnp.zeros(4), eps, K, 1.0))
    expected = (
        complex(-0.7, 0.2) * np.exp(1j * depth / 2) * np.sinc(depth / (2 * np.pi))
    )
    assert_allclose(S[1] + 1j * S[2], expected, atol=2e-14)
    assert abs(abs(expected) - abs(complex(-0.7, 0.2))) > 0.08


def test_cgs_slab_conversion_axis_and_rotation_covariance():
    from syncmoments.los_moments import moment_driven_slab_cgs

    nu, ne, b = 1e8, 0.03, 5e-6
    L = 0.7 / abs(mueller_conversion(nu, ne, b))
    initial = jnp.array([2.0, 0.0, 1.0, 0.0])
    S = moment_driven_slab_cgs(
        nu, 2500.0, b, 0.0, 0.0, 0.0, 0.0, L, n_e=ne, B_par=0.0, S0=initial
    )
    assert_allclose(S, [2.0, 0.0, np.cos(0.7), -np.sin(0.7)], atol=3e-14)
    # Rotate the projected-field basis and incident linear Stokes together.
    phi = 0.37
    R = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(2 * phi), -np.sin(2 * phi), 0],
            [0, np.sin(2 * phi), np.cos(2 * phi), 0],
            [0, 0, 0, 1],
        ]
    )
    rotated = moment_driven_slab_cgs(
        nu, 2500.0, b, 0.0, 0.0, 0.0, 0.0, L, n_e=ne, B_par=0.0, phi=phi, S0=R @ initial
    )
    assert_allclose(rotated, R @ S, atol=3e-14)


def test_transfer_shape_contract_and_empty_los():
    from syncmoments.transfer import transfer_los

    with pytest.raises(ValueError, match="shape"):
        transfer_slab(jnp.ones(3), jnp.zeros(4), jnp.eye(4), 1.0)
    initial = jnp.arange(4.0, dtype=jnp.float64)
    assert_allclose(
        transfer_los(initial, jnp.empty((0, 4)), jnp.empty((0, 4, 4)), jnp.empty(0)),
        initial,
    )


def test_zero_perpendicular_field_has_zero_radiation_derivatives():
    from syncmoments.kirchhoff import cgs_coefficients_from_moments

    def f(b):
        return jnp.asarray(
            cgs_coefficients_from_moments(1e8, 2500.0, b, 2e-9, 0.0, 0.0, 2e-9 / 2500.0)
        )

    assert_allclose(f(0.0), np.zeros(4), atol=0.0)
    assert_allclose(jax.jit(jax.jacfwd(f))(0.0), np.zeros(4), atol=0.0)
    assert_allclose(jax.jacrev(f)(0.0), np.zeros(4), atol=0.0)
    assert_allclose(f(1e-10), np.zeros(4), atol=1e-250)


def test_strong_absorption_uses_declared_exponential_budget():
    actual = transfer_slab(
        jnp.zeros(4), jnp.array([1.0, 0.0, 0.0, 0.0]), 1e6 * jnp.eye(4), 1.0
    )
    assert_allclose(actual, [1e-6, 0.0, 0.0, 0.0], rtol=3e-10, atol=0.0)
