"""Bound helpers of ``synchro.model.bounds`` against explicit NumPy sums.

``basis_remainder`` and ``depth_error_bound`` are tested with minimal stand-in
objects exposing only the interface fields they consume (``basis.index``,
``basis.channels.n_ch``, ``channels.support``, ``kernel.channel_modes``).
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from synchro.model.bounds import (
    RemainderInputs,
    azimuth_factorisation_bound,
    depth_error_bound,
    screen_factorisation_bound,
)
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import PopulationSamples

C_SI_M = 2.99792458e8


class FakeChannels(NamedTuple):
    support: jnp.ndarray
    centres_hz: jnp.ndarray
    n_ch: int


class FakeBasis(NamedTuple):
    index: MomentIndex
    channels: FakeChannels


class Modes(NamedTuple):
    I: jnp.ndarray
    V: jnp.ndarray
    P: jnp.ndarray


def fake_channels(n_ch=3):
    lo = 1e8 * (1 + np.arange(n_ch))
    hi = 1.5 * lo
    return FakeChannels(
        support=jnp.asarray(np.stack([lo, hi], axis=1)),
        centres_hz=jnp.asarray((lo + hi) / 2),
        n_ch=n_ch,
    )


def fake_P(gamma, B, mu, eta, depth, channels):
    tau = 2 * (C_SI_M / channels.centres_hz) ** 2
    amp = gamma * B * (1 + 0.5 * mu) * (1 - 0.2 * eta)
    return amp * jnp.exp(1j * tau * depth) * (1 + 0.1j * jnp.arange(channels.n_ch))


class FakeKernel(eqx.Module):
    """One phase weight; ``P[0]`` is a smooth complex line-like kernel."""

    seen_phase: tuple = eqx.field(static=True, default=())

    def channel_modes(self, channels, gamma, B, mu, eta, *, phase, depth_ref, s_depth):
        P = fake_P(gamma, B, mu, eta, depth_ref, channels)
        return Modes(I=jnp.abs(P), V=jnp.zeros(channels.n_ch), P=P[None, :])


def seven_samples(seed=4):
    rng = np.random.default_rng(seed)
    return PopulationSamples(
        5 + rng.uniform(-1, 1, 7),
        2 + rng.uniform(-0.5, 0.5, 7),
        rng.uniform(-0.9, 0.9, 7),
        rng.uniform(-0.9, 0.9, 7),
        rng.uniform(0, 2 * np.pi, 7),
        1e-7 * rng.uniform(-1, 1, 7),
        weights=rng.uniform(0.5, 2, 7),
    )


# -- factorisation bounds ------------------------------------------------------


def test_screen_factorisation_bound_shape_and_columns():
    sigma = np.array([0.3, 1.2, 0.0, 2.0])
    phi = np.array([0.0, 0.6, 0.5, 1.0])
    term = screen_factorisation_bound(sigma, phi)
    assert term.kind == "bound"
    assert term.value.shape == (4, 4)
    expected = sigma * np.sqrt(1 - phi**2)
    assert_allclose(term.value[:, 1], expected, atol=1e-15)
    assert_allclose(term.value[:, 2], expected, atol=1e-15)
    assert np.all(term.value[:, 0] == 0) and np.all(term.value[:, 3] == 0)
    assert_allclose(term.value[3], 0.0)  # |Phi| = 1: factorisation exact
    assert_allclose(term.value[0, 1], 0.3)  # |Phi| = 0: full variance
    scaled = screen_factorisation_bound(sigma, phi, amplitude=2.5)
    assert_allclose(scaled.value, 2.5 * term.value, atol=1e-15)
    assert "screen factorisation error" in term.note
    compiled = jax.jit(screen_factorisation_bound)(jnp.asarray(sigma), jnp.asarray(phi))
    assert_allclose(compiled.value, term.value, atol=1e-15)
    grad = jax.grad(lambda p: screen_factorisation_bound(sigma, p).value.sum())(
        jnp.full(4, 0.5)
    )
    assert np.all(np.isfinite(grad))


@pytest.mark.parametrize(
    "sigma,phi",
    [
        ([-0.1, 0.2], [0.5, 0.5]),
        ([0.1, 0.2], [1.5, 0.5]),
        ([0.1, 0.2], [-0.1, 0.5]),
        ([np.nan, 0.2], [0.5, 0.5]),
    ],
)
def test_screen_factorisation_bound_rejects_values(sigma, phi):
    with pytest.raises(Exception):
        screen_factorisation_bound(np.array(sigma), np.array(phi))


def test_screen_factorisation_bound_rejects_shapes():
    with pytest.raises(ValueError):
        screen_factorisation_bound(np.array([0.1, 0.2]), np.array([0.5, 0.5, 0.5]))
    # (n_ch, n_line) is the per-line form (R06); 3-D and mismatched shapes are refused.
    with pytest.raises(ValueError):
        screen_factorisation_bound(np.zeros((1, 2, 2)), np.full((1, 2, 2), 0.5))
    with pytest.raises(ValueError):
        screen_factorisation_bound(np.array([[0.1, 0.2]]), np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        screen_factorisation_bound(np.array([0.1 + 0j]), np.array([0.5]))
    with pytest.raises(ValueError):
        azimuth_factorisation_bound(np.array([[0.1, 0.2]]), np.array([[0.5, 0.5]]))


def test_azimuth_factorisation_bound_scalar_and_vector_circular_moment():
    sigma = np.array([0.5, 1.5, 2.5])
    term = azimuth_factorisation_bound(sigma, 0.6)
    assert term.value.shape == (3, 4)
    assert_allclose(term.value[:, 1], sigma * 0.8, atol=1e-15)
    assert_allclose(term.value[:, 2], sigma * 0.8, atol=1e-15)
    assert np.all(term.value[:, [0, 3]] == 0)
    vector = azimuth_factorisation_bound(
        sigma, np.array([0.0, 0.6, 1.0]), amplitude=3.0
    )
    assert_allclose(
        vector.value[:, 1], 3.0 * sigma * np.array([1.0, 0.8, 0.0]), atol=1e-15
    )
    assert "angular factorisation error" in vector.note
    with pytest.raises(ValueError):
        azimuth_factorisation_bound(sigma, np.array([0.5, 0.5]))
    with pytest.raises(Exception):
        azimuth_factorisation_bound(sigma, 1.2)
    with pytest.raises(Exception):
        azimuth_factorisation_bound(-sigma, 0.5)


# -- depth error bound ---------------------------------------------------------


def depth_oracle(samples, channels, delta, amplitude=1.0):
    w = np.asarray(samples.normalised_weights())
    tau = 2 * (C_SI_M / np.asarray(channels.support)[:, 0]) ** 2
    absP = np.stack(
        [
            np.abs(
                np.asarray(
                    fake_P(
                        float(samples.gamma[n]),
                        float(samples.B[n]),
                        float(samples.mu[n]),
                        float(samples.eta[n]),
                        float(samples.depth[n]),
                        channels,
                    )
                )
            )
            for n in range(samples.size)
        ]
    )
    delta = np.broadcast_to(np.abs(delta), (samples.size,))
    return amplitude * tau * np.einsum("n,nj,n->j", w, absP, delta)


def test_depth_error_bound_matches_fake_kernel_average():
    samples = seven_samples()
    channels = fake_channels()
    delta = np.array([1e-9, 2e-9, -3e-9, 7e-10, 5e-10, 4e-9, 1e-8])
    zero = depth_error_bound(samples, FakeKernel(), channels, 0.0)
    assert np.all(np.asarray(zero.value) == 0)
    term = depth_error_bound(samples, FakeKernel(), channels, delta, amplitude=1.7)
    expected = depth_oracle(samples, channels, delta, 1.7)
    assert term.value.shape == (3, 4)
    assert_allclose(term.value[:, 1], expected, rtol=1e-13)
    assert_allclose(term.value[:, 2], expected, rtol=1e-13)
    assert np.all(term.value[:, [0, 3]] == 0)
    assert term.kind == "estimate" and "|K_P|" in term.note
    scalar = depth_error_bound(samples, FakeKernel(), channels, 2e-9)
    assert_allclose(
        scalar.value[:, 1], depth_oracle(samples, channels, 2e-9), rtol=1e-13
    )
    compiled = eqx.filter_jit(depth_error_bound)(
        samples, FakeKernel(), channels, jnp.asarray(delta), amplitude=1.7
    )
    assert_allclose(compiled.value, term.value, rtol=1e-13)
    batched = depth_error_bound(
        samples, FakeKernel(), channels, delta, amplitude=1.7, batch_size=2
    )
    assert_allclose(batched.value, term.value, rtol=1e-13)
    grad = jax.grad(
        lambda d: depth_error_bound(
            samples, FakeKernel(), channels, d, amplitude=1.7
        ).value.sum()
    )(jnp.asarray(delta))
    fd = np.array(
        [
            (
                depth_oracle(
                    samples, channels, delta + 1e-12 * (np.arange(7) == n), 1.7
                ).sum()
                * 2
                - depth_oracle(
                    samples, channels, delta - 1e-12 * (np.arange(7) == n), 1.7
                ).sum()
                * 2
            )
            / 2e-12
            for n in range(7)
        ]
    )
    assert_allclose(grad, fd, rtol=1e-4)


def test_depth_error_bound_with_absolute_channel_sum_is_a_bound():
    samples = seven_samples(8)
    channels = fake_channels(2)
    absolute = np.abs(np.random.default_rng(0).standard_normal((7, 2))) + 1
    term = depth_error_bound(
        samples, FakeKernel(), channels, 1e-9, absolute_channel_sum=absolute
    )
    w = np.asarray(samples.normalised_weights())
    tau = 2 * (C_SI_M / np.asarray(channels.support)[:, 0]) ** 2
    assert term.kind == "bound"
    assert_allclose(term.value[:, 1], tau * 1e-9 * (w @ absolute), rtol=1e-13)
    with pytest.raises(ValueError):
        depth_error_bound(
            samples, FakeKernel(), channels, 1e-9, absolute_channel_sum=absolute[:3]
        )
    with pytest.raises(ValueError):
        depth_error_bound(samples, FakeKernel(), channels, np.ones(3))
    with pytest.raises(Exception):
        depth_error_bound(samples, FakeKernel(), channels, np.nan)


# -- RemainderInputs -----------------------------------------------------------


def remainder_oracle(rho, H, absolute, N):
    fact = math.factorial(N + 1)
    I = rho[:, 0] + H[:, 0] @ absolute[0] / fact
    P = rho[:, 1] + H[:, 1] @ absolute[1] / fact
    V = rho[:, 2] + H[:, 2] @ absolute[0] / fact
    return np.stack([I, P, P, V], axis=1)


@pytest.mark.parametrize(
    "truncation", [Truncation(2, 2, 2), Truncation(1, 1, 0), Truncation(0, 0, 3)]
)
def test_basis_remainder_matches_explicit_sum(truncation):
    index = MomentIndex.build(truncation)
    basis = FakeBasis(index=index, channels=fake_channels(3))
    rng = np.random.default_rng(int(truncation.N))
    rho = rng.uniform(0, 1, (3, 3))
    H = rng.uniform(0, 2, (3, 3, index.n_lk))
    absolute = rng.uniform(0, 1, (2, index.n_lk))
    inputs = RemainderInputs(rho_ang=rho, H=H, absolute_moments=absolute)
    term = inputs.basis_remainder(basis, 2.0)
    assert term.kind == "bound" and term.value.shape == (3, 4)
    assert_allclose(
        term.value, 2.0 * remainder_oracle(rho, H, absolute, truncation.N), rtol=1e-14
    )
    assert "local response remainder" in term.note
    estimate = RemainderInputs(
        rho_ang=rho, H=H, absolute_moments=absolute, kind="estimate"
    )
    assert estimate.basis_remainder(basis, 1.0).kind == "estimate"
    compiled = eqx.filter_jit(lambda amp: inputs.basis_remainder(basis, amp))(2.0)
    assert_allclose(compiled.value, term.value, rtol=1e-14)


def test_basis_remainder_missing_inputs_are_unbounded():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis = FakeBasis(index=index, channels=fake_channels(2))
    term = RemainderInputs().basis_remainder(basis, 1.0)
    assert term.kind == "unbounded" and term.value is None
    for name in ("rho_ang", "H", "absolute_moments"):
        assert name in term.note
    partial = RemainderInputs(rho_ang=np.zeros((2, 3))).basis_remainder(basis, 1.0)
    assert (
        partial.kind == "unbounded"
        and "rho_ang" not in partial.note
        and "H" in partial.note
    )


def test_basis_remainder_rejects_shapes_values_and_kind():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis = FakeBasis(index=index, channels=fake_channels(2))
    ok = dict(
        rho_ang=np.zeros((2, 3)),
        H=np.zeros((2, 3, index.n_lk)),
        absolute_moments=np.zeros((2, index.n_lk)),
    )
    RemainderInputs(**ok).basis_remainder(basis, 1.0)
    with pytest.raises(ValueError):
        RemainderInputs(**{**ok, "rho_ang": np.zeros((3, 3))}).basis_remainder(
            basis, 1.0
        )
    with pytest.raises(ValueError):
        RemainderInputs(
            **{**ok, "H": np.zeros((2, 3, index.n_lk + 1))}
        ).basis_remainder(basis, 1.0)
    with pytest.raises(ValueError):
        RemainderInputs(
            **{**ok, "absolute_moments": np.zeros((3, index.n_lk))}
        ).basis_remainder(basis, 1.0)
    with pytest.raises(ValueError):
        RemainderInputs(**ok, kind="unbounded")
    with pytest.raises(Exception):
        RemainderInputs(**{**ok, "rho_ang": -np.ones((2, 3))})
    with pytest.raises(Exception):
        RemainderInputs(**{**ok, "H": np.full((2, 3, index.n_lk), np.nan)})
    with pytest.raises(Exception):
        RemainderInputs(**ok).basis_remainder(basis, -1.0)


def test_basis_remainder_appendix_c_layout():
    index = MomentIndex.build(Truncation(1, 1, 1, depth_degree=2))
    basis = FakeBasis(index=index, channels=fake_channels(2))
    rng = np.random.default_rng(3)
    rho = rng.uniform(0, 1, (2, 3))
    H = rng.uniform(0, 1, (2, 3, index.n_lk))
    absolute = rng.uniform(0, 1, (2, index.n_lk))
    n_a = index.n2 // 3
    tail = rng.uniform(0, 1, n_a)
    residual = rng.uniform(0, 1, 2)
    coefficients = rng.standard_normal((2, n_a)) + 1j * rng.standard_normal((2, n_a))
    inputs = RemainderInputs(
        rho_ang=rho,
        H=H,
        absolute_moments=absolute,
        depth_tail=tail,
        intrinsic_residual=residual,
    )
    without = inputs.basis_remainder(basis, 1.0)
    assert without.kind == "unbounded" and "depth_coefficients" in without.note
    term = inputs.basis_remainder(basis, 1.5, depth_coefficients=coefficients)
    tau = 2 * (C_SI_M / np.asarray(basis.channels.support)[:, 0]) ** 2
    P = residual + tau**3 / math.factorial(3) * (np.abs(coefficients) @ tail)
    I = rho[:, 0] + H[:, 0] @ absolute[0] / 2
    V = rho[:, 2] + H[:, 2] @ absolute[0] / 2
    assert_allclose(term.value, 1.5 * np.stack([I, P, P, V], axis=1), rtol=1e-13)
    assert "joint screen remainder" in term.note
    missing = RemainderInputs(
        rho_ang=rho, H=H, absolute_moments=absolute
    ).basis_remainder(basis, 1.0, depth_coefficients=coefficients)
    assert missing.kind == "unbounded" and "depth_tail" in missing.note
    with pytest.raises(ValueError):
        inputs.basis_remainder(basis, 1.0, depth_coefficients=coefficients[:, :-1])
