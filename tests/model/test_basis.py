"""Tests of ``syncmoments.model.basis``: exactness on the polynomial oracle, nested
derivatives against finite differences and the manuscript Section 5.3.1 benchmark
coefficients (refusals, JIT/AD, provenance and convergence: ``test_basis_build``
and ``test_basis_convergence``)."""

import itertools
import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

import syncmoments  # noqa: F401
from syncmoments.constants import C_SI_M
from syncmoments.model.basis import (
    KernelTerms,
    SpectralBasis,
    build_basis,
)
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference
from syncmoments.model.phase import TaylorPhase

from _basis_fixtures import (
    SMALL_SUPPORT,
    five_point,
    polynomial_setup,
    small_channels,
    small_harmonic,
    taylor_weights,
)
from _harmonic_oracles import (
    B0,
    BENCH_SUPPORT,
    GAMMA0,
    REFERENCE,
    S_DEPTH,
    UNITS,
    benchmark_channels,
)

# -- polynomial oracle: exact coefficients, factorials, b index, Q/U split ----------------


def test_polynomial_basis_reproduces_coefficients_exactly():
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    truncation = Truncation(2, 2, 2)
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, convergence=False
    )
    assert isinstance(basis, SpectralBasis)
    index = basis.index
    assert (basis.I_basis.shape, basis.P_basis.shape) == ((2, index.n0), (2, index.n2))
    assert jnp.iscomplexobj(basis.P_basis) and not jnp.iscomplexobj(basis.I_basis)
    I_b, V_b, P_b = (
        np.asarray(a) for a in (basis.I_basis, basis.V_basis, basis.P_basis)
    )
    for pos, (l, k, r, s, b) in enumerate(index.h0):
        even = (l + k) % 2 == 0
        assert_allclose(
            I_b[:, pos], coeff[0, :, l, k, r, s] * even, rtol=1e-12, atol=1e-13
        )
        assert_allclose(
            V_b[:, pos], coeff[2, :, l, k, r, s] * (not even), rtol=1e-12, atol=1e-13
        )
    tau = 2 * (C_SI_M / line_nu) ** 2
    w = taylor_weights(tau, 1.5, 0.5, 2)  # (3, n_ch)
    for pos, (l, k, r, s, b) in enumerate(index.h2):
        assert_allclose(
            P_b[:, pos], coeff[1, :, l, k, r, s] * w[b], rtol=1e-12, atol=1e-13
        )
    # Response matrix layout: channel-major, Stokes I, Q, U, V; Q/U real-imag split.
    C = np.asarray(basis.response_matrix())
    n0, n2 = index.n0, index.n2
    assert C.shape == (8, index.n_real)
    for j in range(2):
        assert_allclose(C[4 * j, :n0], I_b[j]) and np.all(C[4 * j, n0:] == 0)
        assert_allclose(C[4 * j + 3, :n0], V_b[j]) and np.all(C[4 * j + 3, n0:] == 0)
        assert np.all(C[4 * j + 1, :n0] == 0) and np.all(C[4 * j + 2, :n0] == 0)
        assert_allclose(C[4 * j + 1, n0 : n0 + n2], P_b[j].real)
        assert_allclose(C[4 * j + 1, n0 + n2 :], -P_b[j].imag)
        assert_allclose(C[4 * j + 2, n0 : n0 + n2], P_b[j].imag)
        assert_allclose(C[4 * j + 2, n0 + n2 :], P_b[j].real)
    assert basis.n_ch == 2 and basis.certified_orders == (0, 1, 2, 3)
    assert isinstance(basis.kernel_terms, KernelTerms)
    assert basis.kernel_terms.physical_kernel.kind == "bound"
    assert basis.kernel_terms.harmonic_truncation.kind == "not_applicable"
    assert basis.kernel_terms.numerical.kind == "unbounded"
    assert basis.kernel_terms.screen_exponent.kind == "not_applicable"


def test_polynomial_contraction_reproduces_direct_population_average():
    """``C m`` equals the direct weighted average of the polynomial kernel (depth at the
    reference so that the depth Taylor factor is exact)."""
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    truncation = Truncation(2, 2, 2)
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, convergence=False
    )
    rng = np.random.default_rng(11)
    S = 40
    gamma = rng.uniform(15.0, 25.0, S)
    B = rng.uniform(1.5, 2.5, S)
    mu, eta = rng.uniform(-1, 1, (2, S))
    phi = rng.uniform(0, 2 * np.pi, S)
    weights = rng.uniform(0.5, 2.0, S)
    samples = PopulationSamples(gamma, B, mu, eta, phi, np.full(S, 1.5), weights)
    m = np.asarray(
        JointMoments.from_samples(samples, basis.index, reference).to_vector()
    )
    pred = (np.asarray(basis.response_matrix()) @ m).reshape(2, 4)
    w = weights / weights.sum()
    z_g, z_B = (gamma - 20.0) / 2.0, (B - 2.0) / 0.2
    tau = 2 * (C_SI_M / line_nu) ** 2
    direct = np.zeros((2, 4))
    for n in range(S):
        p_l = eval_legendre(np.arange(3), mu[n])
        p_k = eval_legendre(np.arange(3), eta[n])
        pg, pb = z_g[n] ** np.arange(3), z_B[n] ** np.arange(3)
        val = np.einsum("sjlkrt,l,k,r,t->sj", coeff, p_l, p_k, pg, pb)
        P = val[1] * np.exp(2j * phi[n]) * np.exp(1j * tau * 1.5)
        direct += w[n] * np.stack([val[0], P.real, P.imag, val[2]], axis=1)
    assert_allclose(pred, direct, rtol=1e-11, atol=1e-13 * np.max(np.abs(direct)))


# -- harmonic derivatives against finite differences -----------------------------------


def test_harmonic_basis_derivatives_match_five_point_differences():
    kernel, channels, reference, support = small_harmonic()
    truncation = Truncation(2, 2, 2)
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, convergence=False
    )
    index = basis.index
    phase = TaylorPhase(2)
    project = jax.jit(
        lambda z: kernel.angular_projection(
            channels,
            GAMMA0 * (1 + z[0]),
            B0 * (1 + z[1]),
            phase=phase,
            depth_ref=3.0 * S_DEPTH,
            s_depth=S_DEPTH,
            truncation=index,
        )
    )
    step = 1e-3
    values = {}
    for i, j in itertools.product(range(-2, 3), repeat=2):
        p = project(jnp.array([i * step, j * step]))
        values[i, j] = (np.asarray(p.I), np.asarray(p.V), np.asarray(p.P))
    I_b, V_b, P_b = (
        np.asarray(a) for a in (basis.I_basis, basis.V_basis, basis.P_basis)
    )
    scale_I = np.max(np.abs(I_b))
    scale_P = np.max(np.abs(P_b))
    for pos, (l, k, r, s, b) in enumerate(index.h0):
        i = index.pairs.index((l, k))
        fd_I = five_point({key: v[0][:, i] for key, v in values.items()}, step, r, s)
        fd_V = five_point({key: v[1][:, i] for key, v in values.items()}, step, r, s)
        norm = math.factorial(r) * math.factorial(s)
        even = (l + k) % 2 == 0
        assert_allclose(I_b[:, pos], fd_I / norm * even, atol=1e-6 * scale_I)
        assert_allclose(V_b[:, pos], fd_V / norm * (not even), atol=1e-6 * scale_I)
    for pos, (l, k, r, s, b) in enumerate(index.h2):
        i = index.pairs.index((l, k))
        fd_P = five_point({key: v[2][b, :, i] for key, v in values.items()}, step, r, s)
        norm = math.factorial(r) * math.factorial(s)
        assert_allclose(P_b[:, pos], fd_P / norm, atol=1e-6 * scale_P)
    assert scale_I > 0 and scale_P > 0 and np.max(np.abs(V_b)) > 0


def test_third_order_columns_match_richardson_of_second_order():
    """``N = 3`` columns (certified order 3) against Richardson differences of the
    order-2 nested derivative."""
    channels = small_channels()
    kernel = HarmonicKernel(8, n_outer=12, n_inner=12)
    reference = Reference(GAMMA0, B0, depth_ref=0.0, scales=(GAMMA0, B0, S_DEPTH))
    truncation = Truncation(1, 1, 3, depth_degree=0)
    basis = build_basis(
        kernel,
        channels,
        truncation,
        reference,
        support=SMALL_SUPPORT,
        convergence=False,
    )
    index = basis.index

    def f(z):
        return kernel.angular_projection(
            channels,
            GAMMA0 * (1 + z[0]),
            B0 * (1 + z[1]),
            phase=TaylorPhase(0),
            truncation=index,
        ).I

    second = jax.jit(jax.jacfwd(jax.jacfwd(f)))
    I_b = np.asarray(basis.I_basis)
    scale = np.max(np.abs(I_b))
    h = 2e-3
    for pos, (l, k, r, s, b) in enumerate(index.h0):
        if r + s != 3:
            continue
        i = index.pairs.index((l, k))
        axis = 0 if r > 0 else 1
        lower = (r - 1, s) if r > 0 else (r, s - 1)
        idx = (0,) * lower[0] + (1,) * lower[1]
        e = np.zeros(2)
        e[axis] = h

        def g(x):
            return np.asarray(second(jnp.asarray(x)))[:, i][(slice(None),) + idx]

        d1 = (g(e) - g(-e)) / (2 * h)
        d2 = (g(e / 2) - g(-e / 2)) / h
        fd = (4 * d2 - d1) / 3 / (math.factorial(r) * math.factorial(s))
        even = (l + k) % 2 == 0
        assert_allclose(I_b[:, pos], fd * even, atol=1e-6 * scale)


# -- manuscript benchmark (Section 5.3.1 coefficients) ---------------------------------


@pytest.fixture(scope="module")
def manuscript_coefficients():
    """The manuscript's five-point coefficients at the two steps it reports
    (1e-3 and 5e-4) plus their Richardson combination (removes the h^4 term)."""
    sys.path.insert(0, REFERENCE)
    try:
        from validation import full_response, full_response_product
    finally:
        sys.path.pop(0)
    coarse = full_response_product.coefficients(64, 1e-3, 40)
    fine = full_response_product.coefficients(64, 5e-4, 40)
    return full_response.POWERS, coarse, fine, (16 * fine - coarse) / 15


def test_benchmark_basis_matches_manuscript_coefficients(
    manuscript_coefficients, record_property
):
    powers, coarse, fine, extrapolated = manuscript_coefficients
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    reference = Reference(
        GAMMA0, B0, depth_ref=4.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    basis = build_basis(
        kernel,
        channels,
        Truncation(8, 8, 2),
        reference,
        support=BENCH_SUPPORT,
        convergence=False,
    )
    index = basis.index
    assert (index.n0, index.n2) == (81 * 6, 41 * 10)
    got = {
        "I": np.asarray(basis.I_basis),
        "V": np.asarray(basis.V_basis),
        "P": np.asarray(basis.P_basis),
    }
    lookup = {p: t for t, p in enumerate(powers)}

    def expected(coeff):
        out = {name: np.zeros_like(got[name]) for name in got}
        for pos, (l, k, r, s, b) in enumerate(index.h0):
            even = (l + k) % 2 == 0
            row = coeff[lookup[(r, s, 0)], :, :, l, k]
            out["I"][:, pos] = row[0].real * UNITS * even
            out["V"][:, pos] = row[1].real * UNITS * (not even)
        for pos, (l, k, r, s, b) in enumerate(index.h2):
            out["P"][:, pos] = coeff[lookup[(r, s, b)], 2, :, l, k] * UNITS
        return out

    reference_coeff = expected(extrapolated)
    single = {step: expected(c) for step, c in (("1e-3", coarse), ("5e-4", fine))}
    for name in ("I", "V", "P"):
        scale = np.max(np.abs(reference_coeff[name]))
        assert scale > 0
        error = np.max(np.abs(got[name] - reference_coeff[name])) / scale
        assert error < 1e-6, (name, error)
        # The single-step oracles differ from the AD columns by their own h^4
        # truncation error (second-order columns): recorded, and the 16x ratio checked.
        raw = {
            step: np.max(np.abs(got[name] - val[name])) / scale
            for step, val in single.items()
        }
        record_property(f"benchmark_{name}_vs_richardson", error)
        record_property(f"benchmark_{name}_vs_step_1e-3", raw["1e-3"])
        record_property(f"benchmark_{name}_vs_step_5e-4", raw["5e-4"])
        assert raw["5e-4"] < 2e-6
        assert 8 < raw["1e-3"] / raw["5e-4"] < 32
    assert basis.kernel_terms.harmonic_truncation.kind == "bound"
    numerics = dict(basis.provenance.numerics)
    # At the reference, y_m = m / D with D in (1 - beta, 1 + beta): the highest line
    # meeting the top support edge y = 13.2 is floor(13.2 (1 + beta)).
    beta = math.sqrt(1 - 1 / GAMMA0**2)
    assert numerics["m_range"] == (1, math.floor(13.2 * (1 + beta))) == (1, 26)
    assert numerics["n_nodes"] == 256 and numerics["cells"] > 0
    assert_allclose(numerics["width_ratio_min"], 1.3)
