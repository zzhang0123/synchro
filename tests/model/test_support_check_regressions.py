"""Regressions R10 and R16: samples outside the declared ``Support``.

The zero ``harmonic_truncation`` bound rests on ``m_max >= required_m_max(support)``,
which holds only if every emitting electron has ``gamma <= support.gamma[1]``
and ``B >= support.B[0]``. When concrete samples contradict the support, the
kernel and the direct average must not report that zero bound, and the direct
average must not report the ``Support.truncated=False`` declaration as a zero
excluded tail. Boundary cells: samples exactly on the support edges (inside)
and ``1e-9`` relative outside them; zero-weight and ``B = 0`` samples do not
emit and are ignored.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.kernels import required_m_max
from syncmoments.model.moments import PopulationSamples
from syncmoments.model.predict import direct_channel_average

from _harmonic_oracles import BENCH_SUPPORT, benchmark_channels
from _predict_helpers import harmonic_setup, samples_of


def grid_samples(gamma, B, n=8, weights=None):
    """``n x n`` Gauss-Legendre angular grid at one ``(gamma, B)`` point."""
    x, w = np.polynomial.legendre.leggauss(n)
    MU, ETA = np.meshgrid(x, x, indexing="ij")
    size = MU.size
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (size,))
    B = np.broadcast_to(np.asarray(B, dtype=float), (size,))
    weights = np.outer(w, w).ravel() if weights is None else weights
    zeros = np.zeros(size)
    return PopulationSamples(
        gamma, B, MU.ravel(), ETA.ravel(), zeros, zeros, weights=weights
    )


def test_r10_reviewer_case_is_not_a_zero_bound():
    """gamma = 24, B = 0.15 G on BENCH_SUPPORT (B >= 0.8): lines above 40 meet the channels."""
    channels = benchmark_channels()
    assert required_m_max(BENCH_SUPPORT, channels) == 40
    samples = grid_samples(24.0, 0.15, n=24)
    term = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, channels, samples=samples)
    assert term.kind == "unbounded", (term.kind, term.note)
    assert "outside the declared Support" in term.note
    assert "B" in term.note and "0.15" in term.note and "0.8" in term.note


def test_r10_outside_support_with_tail_probe_gives_a_positive_estimate():
    channels = benchmark_channels()
    samples = grid_samples(24.0, 0.15, n=8)
    term = HarmonicKernel(40, tail_probe=40).truncation_error(
        BENCH_SUPPORT, channels, samples=samples
    )
    assert term.kind == "estimate"
    assert "outside the declared Support" in term.note
    assert float(np.max(np.asarray(term.value)[:, 0])) > 0


@pytest.mark.parametrize(
    "gamma, B, variable",
    [
        (24.0 * (1 + 1e-9), 1.0, "gamma"),  # just above gamma_hi
        (30.0, 1.0, "gamma"),
        (20.0, 0.8 * (1 - 1e-9), "B"),  # just below B_lo
        (20.0, 1e-6, "B"),  # extreme: far below B_lo
    ],
)
def test_r10_violation_on_either_variable_is_named(gamma, B, variable):
    channels = benchmark_channels()
    samples = grid_samples(gamma, B, n=4)
    term = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, channels, samples=samples)
    assert term.kind == "unbounded"
    assert f"{variable} in [" in term.note


@pytest.mark.parametrize(
    "gamma, B",
    [(24.0, 0.8), (16.0, 1.2), (20.0, 1.0), (1.0 + 1e-6, 1.2)],
)
def test_r10_samples_on_the_harmonic_safe_side_keep_the_zero_bound(gamma, B):
    """Inside, on the edges, or below gamma_lo (which lowers every line): still bound."""
    channels = benchmark_channels()
    samples = grid_samples(gamma, B, n=4)
    term = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, channels, samples=samples)
    assert term.kind == "bound" and float(np.max(np.asarray(term.value))) == 0.0
    assert "checked" in term.note


def test_r10_zero_weight_and_zero_field_samples_do_not_count():
    channels = benchmark_channels()
    gamma = np.array([20.0, 50.0, 60.0])
    B = np.array([1.0, 1.0, 0.0])  # 50: zero weight; 60: B = 0 (no emission)
    zeros = np.zeros(3)
    samples = PopulationSamples(
        gamma, B, zeros, zeros, zeros, zeros, weights=np.array([1.0, 0.0, 1.0])
    )
    term = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, channels, samples=samples)
    assert term.kind == "bound"


def test_r10_traced_samples_keep_the_bound_with_the_hypothesis_stated():
    channels = benchmark_channels()
    kernel = HarmonicKernel(40)

    @eqx.filter_jit
    def traced(samples):
        return kernel.truncation_error(BENCH_SUPPORT, channels, samples=samples)

    term = traced(grid_samples(24.0, 0.15, n=2))
    assert term.kind == "bound"
    assert "not checked" in term.note and "Support" in term.note


def test_r10_no_samples_note_states_the_support_hypothesis():
    term = HarmonicKernel(40).truncation_error(BENCH_SUPPORT, benchmark_channels())
    assert term.kind == "bound"
    assert "inside the declared Support" in term.note and "not checked" in term.note


def test_r16_direct_average_outside_support_is_not_certified():
    kernel, channels, supp, ref, pop = harmonic_setup()
    outside = dict(pop)
    outside["gamma"] = pop["gamma"] * 3.0  # every sample above supp.gamma[1] = 6
    pred = direct_channel_average(
        samples_of(outside),
        kernel,
        channels,
        amplitude=1.0,
        reference=ref,
        support=supp,
    )
    trunc = pred.budget.harmonic_truncation
    assert trunc.kind == "unbounded", (trunc.kind, trunc.note)
    assert "gamma in [" in trunc.note
    tail = pred.budget.excluded_tail
    assert tail.kind == "unbounded", (tail.kind, tail.note)
    assert "Support.truncated=False" in tail.note and "gamma in [" in tail.note


def test_r16_direct_average_inside_support_keeps_its_certificates():
    kernel, channels, supp, ref, pop = harmonic_setup()
    pred = direct_channel_average(
        samples_of(pop), kernel, channels, amplitude=1.0, reference=ref, support=supp
    )
    assert pred.budget.harmonic_truncation.kind == "bound"
    assert pred.budget.excluded_tail.kind == "bound"
    assert "declared" in pred.budget.excluded_tail.note


def test_r16_depth_outside_support_voids_the_declared_complete_tail_only():
    kernel, channels, supp, ref, pop = harmonic_setup()
    deep = dict(pop)
    deep["depth"] = pop["depth"] + 10.0 * float(jnp.asarray(supp.depth[1]))
    pred = direct_channel_average(
        samples_of(deep), kernel, channels, amplitude=1.0, reference=ref, support=supp
    )
    # Depth does not enter required_m_max: the harmonic bound stands.
    assert pred.budget.harmonic_truncation.kind == "bound"
    tail = pred.budget.excluded_tail
    assert tail.kind == "unbounded" and "depth in [" in tail.note
