"""Tests of ``syncmoments.model.kernels``: shared helpers, the ``KernelModel`` protocol,
``required_m_max`` and ``PolynomialTestKernel`` exactness (continuum: ``test_continuum``).
"""

import math
import sys
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

import syncmoments  # noqa: F401
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import (
    ContinuumKernel,
    KernelModel,
    Modes,
    PolynomialTestKernel,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    legendre_norm,
    legendre_table,
    phase_weights,
    required_m_max,
)
from syncmoments.model.moments import Support
from syncmoments.model.phase import TaylorPhase

from _harmonic_oracles import REFERENCE


def nu_star(gamma0, B0):
    return E_ESU * B0 / (2 * np.pi * gamma0 * M_E * C_CGS)


def benchmark_channels(B0=1.0):
    y = np.array([2.0, 4.0, 8.0])
    return Channels.bump(y * nu_star(20.0, B0), 0.65 * y * nu_star(20.0, B0))


# -- shared helpers -------------------------------------------------------------


def test_legendre_table_matches_scipy():
    x = np.linspace(-1, 1, 23)
    for degree in (0, 1, 2, 5, 8):
        table = np.asarray(legendre_table(jnp.asarray(x), degree))
        assert table.shape == (23, degree + 1)
        for n in range(degree + 1):
            assert_allclose(table[:, n], eval_legendre(n, x), rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        legendre_table(x, -1)


def test_gather_pairs_and_norm():
    index = MomentIndex.build(Truncation(2, 1, 0))
    grid = jnp.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    got = np.asarray(gather_pairs(grid, index.pairs))
    for i, (l, k) in enumerate(index.pairs):
        assert got[0, i] == l * 2 + k and got[1, i] == 6 + l * 2 + k
    assert_allclose(
        np.asarray(legendre_norm(index.pairs)),
        [(2 * l + 1) * (2 * k + 1) / 4 for l, k in index.pairs],
    )
    with pytest.raises(ValueError):
        gather_pairs(grid, ((5, 0),))


def test_phase_weights_none_and_shape_contract():
    tau = jnp.array([1.0, 2.0])
    w = phase_weights(None, tau, 0.0, 1.0)
    assert w.shape == (1, 2) and np.all(np.asarray(w) == 1) and jnp.iscomplexobj(w)
    w = np.asarray(phase_weights(TaylorPhase(2), tau, 0.5, 3.0))
    t = np.asarray(tau)
    expected = np.exp(1j * t * 0.5)[None] * np.stack(
        [np.ones(2), 1j * t * 3.0, (1j * t * 3.0) ** 2 / 2]
    )
    assert_allclose(w, expected, rtol=1e-14)

    class Bad:
        n_weights = 3

        def __call__(self, tau, *, depth_ref, s_depth):
            return jnp.ones((2,) + tau.shape)

    with pytest.raises(ValueError, match="n_weights"):
        phase_weights(Bad(), tau, 0.0, 1.0)


def test_check_scalar_point_shape_and_value_errors():
    with pytest.raises(ValueError, match="scalar"):
        check_scalar_point(jnp.ones(2), 1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="real"):
        check_scalar_point(1.0 + 0j, 1.0, 0.0, 0.0)
    invalid = [
        (0.5, 1.0, 0.0, 0.0),
        (2.0, -1.0, 0.0, 0.0),
        (2.0, 1.0, 1.5, 0.0),
        (2.0, 1.0, 0.0, np.nan),
    ]
    for point in invalid:
        with pytest.raises(Exception):
            eqx.filter_jit(check_scalar_point)(*point)
    out = check_scalar_point(2.0, 1.0, 1.0, -1.0)
    assert [float(v) for v in out] == [2.0, 1.0, 1.0, -1.0]


def test_protocol_membership_and_namedtuples():
    kernels = (
        HarmonicKernel(3),
        ContinuumKernel(),
        PolynomialTestKernel(np.zeros((3, 1, 1, 1, 1, 1)), [1.0]),
    )
    for kernel in kernels:
        assert isinstance(kernel, KernelModel)
        assert isinstance(kernel.LABEL, tuple) and isinstance(kernel.components, tuple)
        hash(kernel.describe())
    modes = Modes(I=jnp.zeros(2), V=jnp.zeros(2), P=jnp.zeros((1, 2)))
    assert modes.I.shape == (2,) and modes.P.shape == (1, 2)
    projected = ProjectedModes(
        I=jnp.zeros((2, 3)), V=jnp.zeros((2, 3)), P=jnp.zeros((1, 2, 3))
    )
    assert projected.P.shape == (1, 2, 3)


# -- required_m_max ----------------------------------------------------------


def test_required_m_max_matches_manuscript_harmonic_cutoff():
    support = Support(gamma=(16.0, 24.0), B=(0.8, 1.2), depth=(0.0, 1.0))
    assert required_m_max(support, benchmark_channels()) == 40
    sys.path.insert(0, REFERENCE)
    try:
        from validation.full_response import harmonic_cutoff
    finally:
        sys.path.pop(0)
    assert harmonic_cutoff(24.0, 0.8) == 40
    # Closed form: ceil((1 + beta_max) nu_hi,max / nu_B,min).
    beta_max = math.sqrt(1 - 1 / 24.0**2)
    nu_B_min = E_ESU * 0.8 / (2 * np.pi * 24.0 * M_E * C_CGS)
    assert required_m_max(support, benchmark_channels()) == math.ceil(
        (1 + beta_max) * 13.2 * nu_star(20.0, 1.0) / nu_B_min
    )
    # Mildly relativistic support: (1 + beta) < 2 tightens the manuscript's D < 2.
    small = Support(gamma=(1.2, 2.0), B=(0.8, 1.2), depth=(0.0, 1.0))
    channels = Channels.bump([3.7 * nu_star(2.0, 0.8)], [0.5 * nu_star(2.0, 0.8)])
    assert (
        required_m_max(small, channels) == math.ceil((1 + math.sqrt(0.75)) * 4.2) == 8
    )
    assert required_m_max(small, channels) < math.ceil(2 * 4.2) == 9
    with pytest.raises(ValueError, match="gamma_max"):
        required_m_max(SimpleNamespace(gamma=(1.0, 1.0), B=(1.0, 2.0)), channels)
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda g: required_m_max(
                SimpleNamespace(gamma=(1.0, g), B=(1.0, 2.0)), channels
            )
        )(24.0)


# -- polynomial test kernel ------------------------------------------------------------


def polynomial_setup(seed=3, n_ch=2, L=2, N=2):
    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((3, n_ch, L + 1, L + 1, N + 1, N + 1))
    line_nu = np.array([1.5e8, 4.0e8])[:n_ch]
    kernel = PolynomialTestKernel(
        coefficients, line_nu, gamma0=20.0, B0=2.0, s_gamma=20.0, s_B=2.0
    )
    return kernel, coefficients, line_nu


def polynomial_oracle(coefficients, gamma, B, mu, eta):
    z_g, z_B = (gamma - 20.0) / 20.0, (B - 2.0) / 2.0
    L, N = coefficients.shape[2] - 1, coefficients.shape[4] - 1
    p_l = eval_legendre(np.arange(L + 1), mu)
    p_k = eval_legendre(np.arange(L + 1), eta)
    pg, pb = z_g ** np.arange(N + 1), z_B ** np.arange(N + 1)
    return np.einsum("sjlkrt,l,k,r,t->sj", coefficients, p_l, p_k, pg, pb)


def test_polynomial_kernel_channel_modes_are_exact():
    kernel, coefficients, line_nu = polynomial_setup()
    gamma, B, mu, eta = 23.0, 1.7, 0.3, -0.8
    phase = TaylorPhase(2)
    modes = jax.jit(
        lambda g, b: kernel.channel_modes(
            None, g, b, mu, eta, phase=phase, depth_ref=1.5, s_depth=0.5
        )
    )(gamma, B)
    expected = polynomial_oracle(coefficients, gamma, B, mu, eta)
    assert_allclose(np.asarray(modes.I), expected[0], rtol=1e-13)
    assert_allclose(np.asarray(modes.V), expected[2], rtol=1e-13)
    tau = 2 * (C_SI_M / line_nu) ** 2
    weights = np.exp(1j * tau * 1.5)[None] * np.stack(
        [np.ones(2), 1j * tau * 0.5, (1j * tau * 0.5) ** 2 / 2]
    )
    assert_allclose(np.asarray(modes.P), expected[1][None] * weights, rtol=1e-13)
    with pytest.raises(ValueError, match="n_ch"):
        kernel.channel_modes(benchmark_channels(), gamma, B, mu, eta)
    assert kernel.physical_error(None).kind == "bound"
    assert kernel.truncation_error(None, None).kind == "not_applicable"


@pytest.mark.parametrize(
    "truncation", [Truncation(2, 2, 2), Truncation(1, 2, 1), Truncation(3, 4, 0)]
)
def test_polynomial_kernel_projection_is_exact_legendre_projection(truncation):
    kernel, coefficients, line_nu = polynomial_setup()
    gamma, B = 18.0, 2.6
    index = MomentIndex.build(truncation)
    projected = kernel.angular_projection(None, gamma, B, truncation=truncation)
    assert projected.I.shape == (2, index.n_lk)
    nodes, weights = leggauss(12)  # exact for polynomial degree <= 23
    values = np.array(
        [
            [polynomial_oracle(coefficients, gamma, B, m, e) for e in nodes]
            for m in nodes
        ]
    )
    for i, (l, k) in enumerate(index.pairs):
        pl, pk = eval_legendre(l, nodes), eval_legendre(k, nodes)
        oracle = (
            (2 * l + 1)
            * (2 * k + 1)
            / 4
            * np.einsum("a,b,absj->sj", weights * pl, weights * pk, values)
        )
        assert_allclose(
            np.asarray(projected.I[:, i]), oracle[0], rtol=1e-12, atol=1e-13
        )
        assert_allclose(
            np.asarray(projected.V[:, i]), oracle[2], rtol=1e-12, atol=1e-13
        )
        assert_allclose(
            np.asarray(projected.P[0, :, i]), oracle[1], rtol=1e-12, atol=1e-13
        )

    # Nested derivatives in z reproduce the polynomial coefficients exactly (at z = 0 too).
    def in_z(z):
        return kernel.angular_projection(
            None, 20.0 + 20.0 * z[0], 2.0 + 2.0 * z[1], truncation=truncation
        ).I

    hessian = np.asarray(jax.jacfwd(jax.jacfwd(in_z))(jnp.zeros(2)))
    for i, (l, k) in enumerate(index.pairs):
        if l <= 2 and k <= 2:
            assert_allclose(
                hessian[:, i, 0, 1], coefficients[0, :, l, k, 1, 1], rtol=1e-12
            )
            assert_allclose(
                hessian[:, i, 0, 0], 2 * coefficients[0, :, l, k, 2, 0], rtol=1e-12
            )
        else:
            assert np.all(hessian[:, i] == 0)


def test_polynomial_kernel_validation():
    with pytest.raises(ValueError):
        PolynomialTestKernel(np.zeros((2, 1, 1, 1, 1, 1)), [1.0])
    with pytest.raises(ValueError):
        PolynomialTestKernel(np.zeros((3, 2, 1, 1, 1, 1)), [1.0])
