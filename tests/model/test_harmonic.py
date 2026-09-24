"""Tests of ``synchro.model.harmonic.HarmonicKernel``: channel modes, derivatives,
Bessel resolution classes, truncation dispatch (projections: ``test_harmonic_projection``).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.constants import C_CGS, E_ESU, M_E
from synchro.model.channels import Channels
from synchro.model.errors import ErrorTerm
from synchro.model.harmonic import HarmonicKernel, auto_nodes, harmonic_lines
from synchro.model.index import Truncation
from synchro.model.kernels import required_m_max
from synchro.model.moments import PopulationSamples, Reference, Support
from synchro.model.phase import TaylorPhase

from _harmonic_oracles import (
    BENCH_SUPPORT,
    NU_STAR,
    S_DEPTH,
    benchmark_channels,
    bump,
    oracle_lines,
    oracle_modes,
    rel_err,
    richardson,
)


@pytest.mark.parametrize(
    "gamma,B,mu,eta",
    [
        (21.0, 1.1, 0.3, -0.6),
        (20.0, 1.0, 0.0, 0.5),
        (24.0, 0.8, 0.9, 1 - 1e-6),
        (16.0, 1.2, -0.7, -(1 - 1e-6)),
        (18.5, 1.05, 1 - 1e-6, 0.2),
        (1.5, 20.0, 0.2, 0.3),
    ],
)
def test_channel_modes_match_numpy_line_sum(gamma, B, mu, eta):
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    phase = TaylorPhase(2)
    depth_ref = 4.3 * S_DEPTH
    modes = kernel.channel_modes(
        channels, gamma, B, mu, eta, phase=phase, depth_ref=depth_ref, s_depth=S_DEPTH
    )
    I, V, P = oracle_modes(channels, 40, gamma, B, mu, eta, 2, depth_ref, S_DEPTH)
    scale = np.max(np.abs(I))
    assert modes.I.shape == (3,) and modes.V.shape == (3,) and modes.P.shape == (3, 3)
    # rtol 1e-12 on every channel that carries lines inside the bump; the atol covers
    # channels reached only on the exponential tail of R (values ~1e-11 of the largest),
    # where 1/(1 - t^2) amplifies the 1e-16 roundoff of nu_m to ~1e-11 of that value.
    assert_allclose(np.asarray(modes.I), I, rtol=1e-12, atol=1e-11 * scale)
    assert_allclose(np.asarray(modes.V), V, rtol=1e-12, atol=1e-11 * scale)
    assert_allclose(np.asarray(modes.P), P, rtol=1e-12, atol=1e-11 * scale)
    assert_allclose(
        np.asarray(kernel.line_frequencies(gamma, B, mu, eta)),
        oracle_lines(np.arange(1, 41), gamma, B, mu, eta)[3],
        rtol=1e-14,
    )


def test_channel_modes_jit_vmap_and_chunking_agree():
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    compiled = jax.jit(lambda g, b: kernel.channel_modes(channels, g, b, 0.3, -0.6))
    single = compiled(21.0, 1.1)
    batched = jax.vmap(lambda g: compiled(g, 1.1))(jnp.array([19.0, 21.0, 23.0]))
    assert_allclose(np.asarray(batched.I[1]), np.asarray(single.I), rtol=1e-14)
    for chunk in (1, 7, 40, 200):
        other = HarmonicKernel(40, m_chunk=chunk).channel_modes(
            channels, 21.0, 1.1, 0.3, -0.6
        )
        assert_allclose(np.asarray(other.I), np.asarray(single.I), rtol=1e-14)
        assert_allclose(np.asarray(other.P), np.asarray(single.P), rtol=1e-14)


def test_zero_field_far_channels_and_axial_points_are_finite():
    channels = benchmark_channels()
    kernel = HarmonicKernel(12)
    zero = kernel.channel_modes(channels, 20.0, 0.0, 0.3, 0.2)
    assert np.all(np.asarray(zero.I) == 0) and np.all(np.asarray(zero.P) == 0)
    far = Channels.bump([1e4 * NU_STAR], [10.0 * NU_STAR])
    assert np.all(np.asarray(kernel.channel_modes(far, 20.0, 1.0, 0.3, 0.2).I) == 0)
    truncation = Truncation(2, 2, 0)
    projected = kernel.angular_projection(far, 20.0, 1.0, truncation=truncation)
    assert np.all(np.asarray(projected.I) == 0) and np.all(np.asarray(projected.V) == 0)
    jac = jax.jacfwd(
        lambda g: kernel.angular_projection(far, g, 1.0, truncation=truncation).I
    )(20.0)
    assert np.all(np.asarray(jac) == 0)
    # Along the field (|mu| = 1) nothing is radiated; on the axis (|eta| = 1) m = 1 survives.
    along = kernel.channel_modes(channels, 20.0, 1.0, 1.0, 0.3)
    assert np.all(np.asarray(along.I) == 0)
    doppler = 1 - 0.3 * np.sqrt(1 - 1 / 400)  # D at mu = 0.3, eta = 1
    line = Channels.bump([NU_STAR / doppler], [0.2 * NU_STAR])
    axis = HarmonicKernel(3).channel_modes(line, 20.0, 1.0, 0.3, 1.0)
    assert np.isfinite(float(axis.I[0])) and float(axis.I[0]) > 0
    assert_allclose(
        float(axis.I[0]), oracle_lines(1, 20.0, 1.0, 0.3, 1.0)[0], rtol=1e-12
    )


def test_v_flips_sign_under_simultaneous_reversal():
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    kwargs = dict(phase=TaylorPhase(1), depth_ref=2.0 * S_DEPTH, s_depth=S_DEPTH)
    a = kernel.channel_modes(channels, 21.0, 1.1, 0.3, -0.6, **kwargs)
    b = kernel.channel_modes(channels, 21.0, 1.1, -0.3, 0.6, **kwargs)
    assert_allclose(np.asarray(a.I), np.asarray(b.I), rtol=1e-14)
    assert_allclose(np.asarray(a.P), np.asarray(b.P), rtol=1e-14)
    assert_allclose(np.asarray(a.V), -np.asarray(b.V), rtol=1e-14)
    assert np.max(np.abs(np.asarray(a.V))) > 1e-3 * np.max(np.abs(np.asarray(a.I)))


# -- derivatives ------------------------------------------------------------------


def test_channel_modes_derivatives_match_richardson_to_third_order():
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    phase = TaylorPhase(1)

    def f(gB):
        modes = kernel.channel_modes(
            channels,
            gB[0],
            gB[1],
            0.3,
            -0.6,
            phase=phase,
            depth_ref=3.0 * S_DEPTH,
            s_depth=S_DEPTH,
        )
        return jnp.concatenate(
            [modes.I, modes.V, modes.P.real.ravel(), modes.P.imag.ravel()]
        )

    x = np.array([21.0, 1.1])
    raw = [f, jax.jacfwd(f)]
    raw.append(jax.jacfwd(raw[-1]))
    raw.append(jax.jacfwd(raw[-1]))
    orders = [jax.jit(fn) for fn in raw]
    values = [np.asarray(fn(jnp.asarray(x))) for fn in orders]
    for order in (1, 2, 3):
        assert np.all(np.isfinite(values[order]))
        lower = lambda y: np.asarray(orders[order - 1](jnp.asarray(y)))  # noqa: E731
        for i in range(2):
            fd = richardson(lower, x, i, 1e-3 * x[i])
            assert rel_err(values[order][..., i], fd) < 1e-6, (order, i)


# -- Bessel resolution classes -----------------------------------------------------


@pytest.mark.parametrize("m", [16, 48, 112, 240, 496, 1008])
def test_bessel_resolution_classes(m):
    points = [
        (5.0, 2.0, 0.3, 0.6),
        (50.0, 0.5, -0.2, 0.8),
        (1.5, 30.0, 0.7, -0.4),
        (5.0, 2.0, 1e-3, 1e-3),
    ]
    count = auto_nodes(m)
    assert count == 2 ** math.ceil(math.log2(max(128, 4 * m + 64)))
    compared = 0
    for gamma, B, mu, eta in points:
        base = np.stack(
            [
                np.asarray(a)
                for a in harmonic_lines(float(m), gamma, B, mu, eta, n_nodes=count)[:3]
            ]
        )
        doubled = np.stack(
            [
                np.asarray(a)
                for a in harmonic_lines(float(m), gamma, B, mu, eta, n_nodes=2 * count)[
                    :3
                ]
            ]
        )
        oracle = np.stack(oracle_lines(m, gamma, B, mu, eta)[:3])
        scale = np.max(np.abs(oracle))
        assert np.all(np.isfinite(base))
        if (
            scale < 1e-200
        ):  # x << m: below double-precision relevance, only "negligible"
            assert np.max(np.abs(base)) < 1e-200 and np.max(np.abs(doubled)) < 1e-200
            continue
        compared += 1
        assert_allclose(base, doubled, rtol=1e-9, atol=1e-9 * scale)
        assert_allclose(base, oracle, rtol=1e-10, atol=1e-10 * scale)
    assert compared >= 2
    # One class below the automatic count fails the resolution guard (NaN, not aliasing).
    below = harmonic_lines(float(m), 5.0, 2.0, 0.3, 0.6, n_nodes=count // 2)[0]
    assert np.isnan(float(below)) or count // 2 >= 2 * m + 2


def test_high_harmonic_channel_sum_matches_numpy():
    gamma, B = 5.0, 10.0
    nu_B = E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)
    channels = Channels.bump([200.0 * nu_B], [40.0 * nu_B])
    kernel = HarmonicKernel(300)
    assert kernel.resolution() == 2048
    modes = kernel.channel_modes(channels, gamma, B, 0.4, 0.5)
    I, V, P = oracle_modes(channels, 300, gamma, B, 0.4, 0.5, 0, 0.0, 1.0)
    assert_allclose(np.asarray(modes.I), I, rtol=1e-11)
    assert_allclose(np.asarray(modes.V), V, rtol=1e-11)
    assert_allclose(np.asarray(modes.P[0]), P[0], rtol=1e-11)


# -- truncation, physical error, provenance ----------------------------------------


def test_required_m_max_dispatch_on_both_sides():
    channels = benchmark_channels()
    assert required_m_max(BENCH_SUPPORT, channels) == 40
    kinds = {
        m: HarmonicKernel(m).truncation_error(BENCH_SUPPORT, channels).kind
        for m in (39, 40, 41)
    }
    assert kinds == {39: "unbounded", 40: "bound", 41: "bound"}
    bound = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, channels)
    assert bound.value.shape == (3, 4) and np.all(np.asarray(bound.value) == 0)
    assert bound.manuscript_term == "E_num"
    # With a probe but no population the term stays unbounded; with a reference it is an estimate.
    probed = HarmonicKernel(30, tail_probe=5)
    assert probed.truncation_error(BENCH_SUPPORT, channels).kind == "unbounded"
    reference = Reference(gamma0=24.0, B0=0.8)
    estimate = probed.truncation_error(BENCH_SUPPORT, channels, reference=reference)
    assert estimate.kind == "estimate" and estimate.value.shape == (3, 4)
    assert "reference" in estimate.note
    # Lines 31..35 at gamma=24, B=0.8 reach channel 3 (y <= 13.2) only near D -> 2; never channel 1.
    assert float(estimate.value[2, 0]) > 0 and float(estimate.value[0, 0]) == 0
    # ceil(39.6) = 40 is conservative: line 40 itself never meets a channel on this support.
    probe_40 = HarmonicKernel(39, tail_probe=1).truncation_error(
        BENCH_SUPPORT, channels, reference=reference
    )
    assert np.all(np.asarray(probe_40.value) == 0)


def test_truncation_estimate_matches_numpy_tail_on_samples():
    channels = benchmark_channels()
    kernel = HarmonicKernel(10, tail_probe=8, m_chunk=3)
    rng = np.random.default_rng(7)
    S = 5
    gamma, B = 20 + 2 * rng.standard_normal(S), 1 + 0.1 * rng.standard_normal(S)
    mu, eta = rng.uniform(-0.9, 0.9, (2, S))
    weights = rng.uniform(0.5, 2.0, S)
    samples = PopulationSamples(gamma, B, mu, eta, np.zeros(S), np.zeros(S), weights)
    support = Support(gamma=(16.0, 24.0), B=(0.8, 1.2), depth=(0.0, 1.0))
    term = kernel.truncation_error(support, channels, samples=samples)
    assert term.kind == "estimate" and term.value.shape == (3, 4)
    assert "samples" in term.note
    expected = np.zeros((3, 4))
    w = weights / weights.sum()
    for n in range(S):
        ms = np.arange(11, 19, dtype=float)
        I, Q, V, nu = oracle_lines(ms, gamma[n], B[n], mu[n], eta[n])
        R = bump(nu, np.asarray(channels.centres_hz), np.asarray(channels.widths_hz))
        columns = [R @ np.abs(I), R @ np.abs(Q), R @ np.abs(Q), R @ np.abs(V)]
        expected += w[n] * np.stack(columns, axis=-1)
    assert np.max(expected) > 0
    assert_allclose(
        np.asarray(term.value), expected, rtol=1e-11, atol=1e-13 * np.max(expected)
    )


def test_physical_error_describe_and_validation():
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)
    term = kernel.physical_error(channels)
    assert term.kind == "unbounded" and term.manuscript_term == "E_phys"
    assert "helical" in term.note and "declared" in term.note
    declared = ErrorTerm(jnp.zeros((3, 4)), "bound", "declared zero for the test")
    assert HarmonicKernel(40, E_phys=declared).physical_error(channels) is declared
    described = dict(kernel.describe())
    assert hash(kernel.describe())
    assert described["n_nodes"] == 256 and described["quadrature"] == "product"
    assert dict(HarmonicKernel(40, n_nodes=512).describe())["n_nodes"] == 512
    assert kernel.components == ("I", "Q", "V") and kernel.required_closures == ()
    assert "eq: smooth channel kernel" in kernel.LABEL
    bad = [
        dict(m_max=0),
        dict(m_max=40, n_nodes=128),
        dict(m_max=40, quadrature="grid"),
        dict(m_max=40, E_phys=1.0),
        dict(m_max=2.5),
        dict(m_max=40, n_outer=1),
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            HarmonicKernel(**kwargs)
    with pytest.raises(ValueError, match="quadrature"):
        kernel.angular_projection(
            channels, 20.0, 1.0, truncation=Truncation(1, 1, 0), quadrature="grid"
        )
    with pytest.raises(ValueError, match="scalar"):
        kernel.channel_modes(channels, jnp.ones(2), 1.0, 0.0, 0.0)
