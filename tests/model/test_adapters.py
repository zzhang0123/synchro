"""Tests of ``syncmoments.model.adapters`` against the existing package routines:
``faraday.joint_faraday_average`` (``app: depth moments`` contraction), ``expansion``
(fixed-harmonic quadratic average) and ``rm.burn_depolarisation``.

Oracles are NumPy sums over discrete populations and SciPy Bessel functions
(``_harmonic_oracles``); the spectral basis is the ``_basis_stub`` stand-in.
"""

import math

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

import syncmoments  # noqa: F401
from syncmoments.constants import C_SI_M
from syncmoments.expansion import build_expansion
from syncmoments.faraday import joint_faraday_average
from syncmoments.model.adapters import (
    BurnCheck,
    HarmonicComparison,
    burn_screen_check,
    fixed_harmonic_comparison,
    joint_faraday_rows,
    to_joint_faraday,
)
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel, PolynomialTestKernel
from syncmoments.model.moments import (
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)
from syncmoments.model.predict import predict

from _basis_stub import build_stub_basis
from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH, oracle_lines

# -- to_joint_faraday: continuum kernel, delta-like channels, depth_degree = L ---------

G_GAMMA0, G_B0 = 3000.0, 5e-6
G_CENTRES = np.array([3.0e8, 5.0e8, 8.0e8])
G_TAU = 2 * (C_SI_M / G_CENTRES) ** 2
G_DEPTH_REF, G_S_DEPTH = 0.4, 0.5


def galactic_samples(seed=1):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, 7)
    return PopulationSamples(
        G_GAMMA0 * (1 + 0.1 * t),
        G_B0 * (1 + 0.15 * t**2 - 0.1 * t),
        rng.uniform(-0.8, 0.8, 7),
        np.clip(0.2 + 0.5 * t, -0.9, 0.9),
        0.3 + 0.9 * t,
        G_DEPTH_REF + G_S_DEPTH * (0.4 * t - 0.2 * t**2),
        weights=rng.uniform(0.5, 2.0, 7),
    )


@pytest.fixture(scope="module")
def galactic_basis():
    kernel = ContinuumKernel(n_eta=24)
    channels = Channels.bump(
        G_CENTRES, 1e-13 * G_CENTRES, normalisation="unit_integral"
    )
    reference = Reference(
        G_GAMMA0, G_B0, G_DEPTH_REF, scales=(G_GAMMA0, G_B0, G_S_DEPTH)
    )
    support = Support((2000.0, 4000.0), (3e-6, 7e-6), (0.0, 1.0), truncated=False)
    basis = build_stub_basis(
        kernel, channels, Truncation(0, 2, 1, depth_degree=3), reference, support
    )
    return kernel, channels, reference, support, basis


def test_to_joint_faraday_reproduces_predict(galactic_basis):
    kernel, channels, reference, support, basis = galactic_basis
    samples = galactic_samples()
    moments = JointMoments.from_samples(samples, basis.index, reference)
    amplitude = 2.0
    pred = predict(basis, moments, amplitude=amplitude)
    coefficients, M, absolute_next = to_joint_faraday(basis, moments, samples=samples)
    rows = joint_faraday_rows(basis.index)
    n_a = len(rows)
    assert coefficients.shape == (3, n_a) and M.shape == (n_a, 4)
    assert absolute_next.shape == (n_a,)
    assert n_a == basis.index.n2 // 4 and all(len(r) == 4 for r in rows)
    lam = C_SI_M / G_CENTRES
    P, error = joint_faraday_average(
        coefficients,
        M,
        lam,
        absolute_next=absolute_next,
        source_error=0.0,
        reference_depth=reference.depth_ref,
        source_column=amplitude,
    )
    predicted = np.asarray(pred.stokes[:, 1]) + 1j * np.asarray(pred.stokes[:, 2])
    scale = np.max(np.abs(predicted))
    assert_allclose(np.asarray(P), predicted, rtol=0, atol=1e-12 * scale)
    # Residual due to the channel width (tau varies as nu^-2 across the support): recorded.
    residual = float(np.max(np.abs(np.asarray(P) - predicted)) / scale)
    assert residual < 1e-12
    # M[a, b] are the raw-displacement depth moments of the z-basis rows.
    w = np.asarray(samples.normalised_weights())
    zg = (np.asarray(samples.gamma) - G_GAMMA0) / G_GAMMA0
    zB = (np.asarray(samples.B) - G_B0) / G_B0
    dd = np.asarray(samples.depth) - G_DEPTH_REF
    for a, (l, k, r, s) in enumerate(rows):
        chi = (
            eval_legendre(l, np.asarray(samples.mu))
            * eval_legendre(k, np.asarray(samples.eta))
            * zg**r
            * zB**s
            * np.exp(2j * np.asarray(samples.phi))
        )
        for b in range(4):
            assert_allclose(complex(M[a, b]), np.sum(w * chi * dd**b), rtol=1e-12)
        assert_allclose(
            float(absolute_next[a]),
            np.sum(w * np.abs(chi) * np.abs(dd) ** 4),
            rtol=1e-12,
        )
    # The error envelope is the ``app: depth moments`` depth tail of the manuscript at the nominal tau.
    expected = (
        amplitude
        * G_TAU**4
        / math.factorial(4)
        * (np.abs(np.asarray(coefficients)) @ np.asarray(absolute_next))
    )
    assert_allclose(np.asarray(error), expected, rtol=1e-12)
    # The coefficients are the per-Hz continuum Q kernel derivatives at the channel centres.
    K_Q = kernel.angular_projection(channels, G_GAMMA0, G_B0, truncation=basis.index).P[
        0
    ]
    a00 = rows.index((0, 0, 0, 0))
    assert_allclose(np.asarray(coefficients[:, a00]), np.asarray(K_Q[:, 0]), rtol=1e-9)


def test_to_joint_faraday_inputs_and_refusals(galactic_basis):
    kernel, channels, reference, support, basis = galactic_basis
    samples = galactic_samples()
    moments = JointMoments.from_samples(samples, basis.index, reference)
    _, _, none = to_joint_faraday(basis, moments)
    assert none is None
    given = np.ones(len(joint_faraday_rows(basis.index)))
    _, _, back = to_joint_faraday(basis, moments, absolute_next=given)
    assert_allclose(np.asarray(back), given)
    with pytest.raises(ValueError):
        to_joint_faraday(basis, moments, absolute_next=given[:-1])
    main_text = build_stub_basis(
        kernel, channels, Truncation(0, 2, 1), reference, support
    )
    with pytest.raises(ValueError, match="depth_degree"):
        to_joint_faraday(
            main_text, JointMoments.from_samples(samples, main_text.index, reference)
        )
    with pytest.raises(ValueError, match="index"):
        to_joint_faraday(
            basis, JointMoments.from_samples(samples, main_text.index, reference)
        )


# -- burn_screen_check ------------------------------------------------------------------


def test_burn_screen_check_exact_for_fixed_line_kernel():
    rng = np.random.default_rng(4)
    c = rng.standard_normal((3, 2, 2, 2, 2, 2))
    line_nu = np.array([2.0e8, 5.0e8])
    kernel = PolynomialTestKernel(c, line_nu, gamma0=5.0, B0=1.0)
    channels = Channels.bump(line_nu, 0.05 * line_nu)
    check = burn_screen_check(
        kernel, channels, 5.3, 0.9, 0.2, -0.4, 0.7, 0.3, amplitude=2.0
    )
    assert isinstance(check, BurnCheck)
    scale = np.max(np.abs(np.asarray(check.nominal)))
    assert_allclose(
        np.asarray(check.screened), np.asarray(check.nominal), atol=1e-14 * scale
    )
    assert check.difference.kind == "measured" and check.difference.value.shape == (
        2,
        4,
    )
    assert np.all(np.asarray(check.difference.value)[:, [0, 3]] == 0)
    # Sign pin: exp(i tau mean - tau^2 sigma^2 / 2) with tau = 2 lam^2 (Burn: exp(2 i mean lam^2 - 2 sigma^2 lam^4)).
    lam = np.asarray(check.wavelengths_m)
    assert_allclose(lam, C_SI_M / line_nu)
    P0 = 2.0 * np.asarray(kernel.channel_modes(channels, 5.3, 0.9, 0.2, -0.4).P[0])
    burn = P0 * np.exp(2j * 0.7 * lam**2) * np.exp(-2 * 0.09 * lam**4)
    assert_allclose(np.asarray(check.nominal), burn, rtol=1e-13)


def test_burn_screen_check_channel_width_residual():
    """Line-by-line screen vs rotation at the centre: exact only for a delta-like channel."""
    kernel = HarmonicKernel(12)
    gamma, B, mu, eta = GAMMA0, B0, 0.3, 0.5
    tau_star = 2 * (C_SI_M / NU_STAR) ** 2
    mean, sigma = 0.8 / tau_star, 0.3 / tau_star
    residuals = {}
    centre = float(kernel.line_frequencies(gamma, B, mu, eta)[3])  # the m = 4 line
    for width in (1e-4, 0.3):
        channels = Channels.bump([centre], [width * centre])
        check = burn_screen_check(kernel, channels, gamma, B, mu, eta, mean, sigma)
        residuals[width] = float(
            np.asarray(check.difference.value)[0, 1]
            / np.abs(np.asarray(check.nominal)[0])
        )
    assert residuals[1e-4] < 1e-6
    assert residuals[0.3] > 1e-4
    assert residuals[0.3] > 100 * residuals[1e-4]


# -- fixed_harmonic_comparison ------------------------------------------------------------


def narrow_population(width, seed=2):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, 11)
    alpha0, theta0 = 0.9, 1.1
    return (
        PopulationSamples(
            GAMMA0 * (1 + width * 0.1 * t),
            B0 * (1 + width * 0.1 * t**2),
            np.cos(alpha0 + width * 0.1 * t),
            np.cos(theta0 + width * 0.05 * (t - 0.3)),
            0.2 + 0.5 * t,
            S_DEPTH * (3.0 + 0.4 * t),
            weights=rng.uniform(0.5, 1.5, 11),
        ),
        alpha0,
        theta0,
    )


def fixed_harmonic_oracle(samples, harmonics):
    """Direct fixed-harmonic average with SciPy Bessel functions (``I, Q, U, V``)."""
    w = np.asarray(samples.normalised_weights())
    out = []
    for m in harmonics:
        I, Q, V, _ = oracle_lines(
            m,
            np.asarray(samples.gamma),
            np.asarray(samples.B),
            np.asarray(samples.mu),
            np.asarray(samples.eta),
        )
        P = np.exp(2j * np.asarray(samples.phi)) * Q
        out.append([w @ I, w @ P.real, w @ P.imag, w @ V])
    return np.array(out)


def test_fixed_harmonic_comparison_returns_both_remainders():
    kernel = HarmonicKernel(20, n_outer=16, n_inner=16)
    channels = Channels.bump([3.0 * NU_STAR], [0.4 * 3.0 * NU_STAR])
    reference = Reference(GAMMA0, B0, 3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH))
    support = Support((16.0, 24.0), (0.8, 1.2), (0.0, 10 * S_DEPTH), truncated=False)
    basis = build_stub_basis(kernel, channels, Truncation(1, 1, 1), reference, support)
    harmonics = (2, 5)
    samples, alpha0, theta0 = narrow_population(1.0)
    expansion = build_expansion(harmonics, GAMMA0, alpha0, theta0, B=B0)
    result = fixed_harmonic_comparison(
        basis,
        expansion,
        samples,
        alpha0=alpha0,
        theta0=theta0,
        kernel=kernel,
        amplitude=1.5,
    )
    assert isinstance(result, HarmonicComparison)
    assert result.harmonics == harmonics
    assert result.channel_finite.shape == (1, 4) and result.channel_direct.shape == (
        1,
        4,
    )
    assert result.fixed_finite.shape == (2, 4) and result.fixed_direct.shape == (2, 4)
    assert (
        result.channel_remainder.kind == "measured"
        and result.fixed_remainder.kind == "measured"
    )
    assert_allclose(
        np.asarray(result.channel_remainder.value),
        np.abs(np.asarray(result.channel_finite) - np.asarray(result.channel_direct)),
    )
    assert_allclose(
        np.asarray(result.fixed_direct),
        1.5 * fixed_harmonic_oracle(samples, harmonics),
        rtol=1e-9,
    )
    assert np.all(np.isfinite(np.asarray(result.fixed_remainder.value)))
    # The quadratic fixed-harmonic average converges as the population narrows.
    narrow, _, _ = narrow_population(0.5)
    narrower = fixed_harmonic_comparison(
        basis, expansion, narrow, alpha0=alpha0, theta0=theta0, amplitude=1.5
    )
    assert narrower.channel_direct is None
    assert narrower.channel_remainder.kind == "unbounded"
    wide_rem = np.max(
        np.asarray(result.fixed_remainder.value)
        / np.abs(np.asarray(result.fixed_direct)[:, :1])
    )
    narrow_rem = np.max(
        np.asarray(narrower.fixed_remainder.value)
        / np.abs(np.asarray(narrower.fixed_direct)[:, :1])
    )
    assert narrow_rem < wide_rem / 4
    assert "not comparable" in result.note
    # The fixed-harmonic quadratic prediction is the manuscript's mixed average.
    fixed_I = np.asarray(result.fixed_finite)[:, 0]
    assert_allclose(fixed_I, np.asarray(result.fixed_direct)[:, 0], rtol=5e-2)
    assert jnp.all(jnp.isfinite(result.fixed_finite))
