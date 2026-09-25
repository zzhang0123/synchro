"""Static chunk budgets of ``ContinuumKernel`` and of the direct average.

``ContinuumKernel.chunk_budget`` bounds the ``F``/``G`` integrand values of
one ``lax.map`` step of ``angular_projection`` (``eta`` nodes in blocks);
``direct_channel_average`` caps its samples per step by the kernel's
``samples_per_step`` (the per-sample ``channel_modes`` work of both kernels).
Blocking changes the summation order only, so every result must equal the
unchunked one to roundoff: ``1e-13`` relative to the largest entry of its
Stokes block (sums of <= 48 terms of one sign pattern; the observed
differences are a few ulp). Memory is checked through XLA's compiled
``memory_analysis`` (temporary bytes) as a ratio between a small budget and
an unbounded one on the same machine and jax version.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401
from syncmoments.model import _direct
from syncmoments.model.basis import build_basis
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import PopulationSamples, Reference, Support
from syncmoments.model.phase import GaussianScreen, TaylorPhase
from syncmoments.model.predict import direct_channel_average

RTOL = 1e-13
HUGE = 1 << 40  # a budget no example reaches: one block, no chunking
GAMMA, FIELD = 3000.0, 5e-6
TRUNCATION = Truncation(0, 2, 2, depth_degree=0)


def galactic_channels(n_ch=3, n_nu=32):
    centres = np.geomspace(0.1e9, 3.0e9, n_ch)
    return Channels.bump(centres_hz=centres, widths_hz=0.3 * centres, n_nu=n_nu)


def per_eta(kernel, channels):
    """``F`` and ``G`` integrand values of one ``eta`` node."""
    return 2 * channels.n_ch * channels.n_nu * kernel.n_nodes_F


def assert_block_close(actual, expected, rtol=RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.max(np.abs(expected))
    assert scale > 0.0
    assert np.max(np.abs(actual - expected)) <= rtol * scale


def projection(kernel, channels, phase=None):
    def f(gamma, B):
        modes = kernel.angular_projection(
            channels,
            gamma,
            B,
            phase=phase,
            depth_ref=30.0,
            s_depth=5.0,
            truncation=TRUNCATION if phase is not None else Truncation(0, 2, 2),
        )
        return modes.I, modes.P

    return f


@pytest.mark.parametrize("blocks", [1, 5, 48])
def test_continuum_projection_and_derivatives_independent_of_budget(blocks):
    channels = galactic_channels()
    screen = GaussianScreen(mean=30.0, sigma=5.0)
    base = ContinuumKernel(chunk_budget=HUGE)
    chunked = ContinuumKernel(chunk_budget=blocks * per_eta(base, channels))
    assert chunked.eta_block(channels) == blocks
    for phase in (None, screen):
        f0, f1 = projection(base, channels, phase), projection(chunked, channels, phase)
        second = lambda f: jax.jacfwd(  # noqa: E731
            jax.jacfwd(f, argnums=(0, 1)), argnums=(0, 1)
        )
        for want, got in zip(f0(GAMMA, FIELD), f1(GAMMA, FIELD)):
            assert_block_close(got, want)
        for want, got in zip(
            jax.tree.leaves(second(f0)(GAMMA, FIELD)),
            jax.tree.leaves(second(f1)(GAMMA, FIELD)),
        ):
            assert_block_close(got, want)


def test_continuum_basis_columns_independent_of_budget():
    channels = galactic_channels()
    reference = Reference(GAMMA, FIELD, 30.0, scales=(300.0, 5e-7, 5.0))
    support = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(0.0, 60.0))
    screen = GaussianScreen(mean=30.0, sigma=5.0)

    def columns(budget):
        basis = build_basis(
            ContinuumKernel(chunk_budget=budget),
            channels,
            TRUNCATION,
            reference,
            support=support,
            phase=screen,
            convergence=False,
        )
        return basis.I_basis, basis.P_basis

    for want, got in zip(columns(HUGE), columns(1)):
        assert_block_close(got, want)


def _temp_bytes(fn, *args):
    return jax.jit(fn).lower(*args).compile().memory_analysis().temp_size_in_bytes


def test_continuum_budget_bounds_compiled_temporaries():
    channels = galactic_channels(n_ch=8, n_nu=64)
    kernels = [ContinuumKernel(chunk_budget=b) for b in (HUGE, 1)]
    temps = []
    for kernel in kernels:
        f = projection(kernel, channels)
        second = jax.jacfwd(jax.jacfwd(f, argnums=(0, 1)), argnums=(0, 1))
        temps.append(_temp_bytes(second, GAMMA, FIELD))
    # 48 eta nodes in one block against one node per step: the working set of
    # the nested forward pass must shrink by far more than 4x.
    assert temps[1] < temps[0] / 4, temps


def test_continuum_budget_validation_and_provenance():
    for bad in (0, -1, 1.5, True, None):
        with pytest.raises(ValueError, match="chunk_budget"):
            ContinuumKernel(chunk_budget=bad)
    kernel = ContinuumKernel(chunk_budget=4096)
    assert dict(kernel.describe())["chunk_budget"] == 4096
    assert kernel.for_tangents(9).chunk_budget == 4096 // 9
    assert kernel.for_tangents(10**9).chunk_budget == 1
    assert ContinuumKernel().chunk_budget == 1 << 20
    channels = galactic_channels()
    assert ContinuumKernel(chunk_budget=1).eta_block(channels) == 1
    assert ContinuumKernel(chunk_budget=HUGE).eta_block(channels) == 48


def toy_samples(n, gamma=(20.0, 22.0), B=(0.9, 1.1)):
    rng = np.random.default_rng(3)
    return PopulationSamples(
        gamma=rng.uniform(*gamma, n),
        B=rng.uniform(*B, n),
        mu=rng.uniform(-0.9, 0.9, n),
        eta=rng.uniform(-0.9, 0.9, n),
        phi=rng.uniform(0.0, np.pi, n),
        depth=rng.uniform(0.0, 5.0, n),
        weights=rng.uniform(0.5, 1.5, n),
    )


def harmonic_channels():
    nu_star = 2.8e6 / 20.0
    return Channels.bump(centres_hz=[3 * nu_star, 5 * nu_star], widths_hz=[nu_star] * 2)


def test_sample_batch_respects_the_kernel_budget():
    channels = harmonic_channels()
    kernel = HarmonicKernel(40, chunk_budget=1 << 20)
    per_sample = 40 * kernel.resolution()  # channel_modes: all 40 harmonics per step
    batch = _direct.sample_batch(kernel, channels, 1024, 4096)
    assert batch == (1 << 20) // per_sample
    assert batch * per_sample <= kernel.chunk_budget
    assert _direct.sample_batch(kernel, channels, 16, 4096) == 16
    assert _direct.sample_batch(kernel, channels, 1024, 7) == 7
    tight = HarmonicKernel(40, chunk_budget=1)
    assert _direct.sample_batch(tight, channels, 1024, 4096) == 1
    continuum = ContinuumKernel(chunk_budget=1 << 20)
    gal = galactic_channels(n_ch=8, n_nu=64)
    assert _direct.sample_batch(continuum, gal, 256, 4096) == (1 << 20) // (
        2 * 8 * 64 * 128
    )


def test_direct_average_independent_of_budget_and_bounded():
    channels = harmonic_channels()
    samples = toy_samples(96)
    stokes = []
    for budget in (HUGE, 1 << 12):
        kernel = HarmonicKernel(8, chunk_budget=budget)
        pred = direct_channel_average(
            samples, kernel, channels, amplitude=1.0, phase=TaylorPhase(0),
            batch_size=512,
        )  # fmt: skip
        stokes.append(np.asarray(pred.stokes))
    for q in range(4):
        assert_block_close(stokes[1][:, q], stokes[0][:, q])

    def per_sample(kernel):
        def fn(gamma, B, mu, eta, phi, depth):
            s = PopulationSamples(gamma, B, mu, eta, phi, depth, weights=jnp.ones(96))
            return _direct._per_sample(
                s, kernel, channels, TaylorPhase(0), True, 0.0, 512
            )

        return fn

    leaves = [np.asarray(getattr(samples, n)) for n in _direct._LEAVES]
    big = _temp_bytes(per_sample(HarmonicKernel(8, chunk_budget=HUGE)), *leaves)
    small = _temp_bytes(per_sample(HarmonicKernel(8, chunk_budget=1 << 12)), *leaves)
    # 96 samples in one step against 4096 // (8 * 128) = 4 per step.
    assert small < big / 4, (small, big)


def test_continuum_direct_average_independent_of_budget():
    channels = galactic_channels()
    samples = toy_samples(24, gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6))
    stokes = [
        np.asarray(
            direct_channel_average(
                samples,
                ContinuumKernel(chunk_budget=budget),
                channels,
                amplitude=1.0,
                batch_size=256,
            ).stokes
        )
        for budget in (HUGE, 1)
    ]
    for q in range(3):
        assert_block_close(stokes[1][:, q], stokes[0][:, q])
