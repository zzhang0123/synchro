"""Tests of ``ContinuumKernel`` against SciPy quadrature of ``F``/``G``, the
``eta -> +/-1`` limits, the ``x_min`` dispatch and the ``eta``-only projection."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre, kv

import synchro  # noqa: F401
from synchro.constants import C_CGS, C_SI_M, E_ESU, M_E
from synchro.model.channels import Channels
from synchro.model.errors import ErrorTerm
from synchro.model.index import MomentIndex, Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.phase import TaylorPhase


def continuum_oracle(nu, gamma, B, eta):
    """NumPy/SciPy ``(K_I, K_Q)`` of ``eq: directional continuum``."""
    b_perp = B * np.sqrt(1 - eta**2)
    amplitude = np.sqrt(3) * E_ESU**3 * b_perp / (4 * np.pi * M_E * C_CGS**2)
    a_B = 3 * E_ESU * b_perp / (4 * np.pi * M_E * C_CGS)
    x = np.asarray(nu) / (a_B * gamma**2)
    F = np.array(
        [xi * quad(lambda u: kv(5 / 3, u), xi, np.inf)[0] for xi in np.atleast_1d(x)]
    )
    return amplitude * F.reshape(np.shape(x)), -amplitude * x * kv(2 / 3, x)


def galactic_channel(gamma=3000.0, B=5e-6, width=0.3, centre=3e8, n_nu=64):
    return Channels.bump([centre], [width * centre], n_nu=n_nu), gamma, B


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_channel_modes_match_scipy_quadrature():
    channels, gamma, B = galactic_channel(n_nu=128)
    kernel = ContinuumKernel()
    eta = 0.4
    phase = TaylorPhase(1)
    depth_ref, s_depth = 30.0, 100.0
    modes = kernel.channel_modes(
        channels, gamma, B, 0.0, eta, phase=phase, depth_ref=depth_ref, s_depth=s_depth
    )
    lo, hi = np.asarray(channels.support)[0]
    centre, width = float(channels.centres_hz[0]), float(channels.widths_hz[0])

    def response(nu):
        t = (nu - centre) / width
        return np.exp(1 - 1 / (1 - t * t)) if abs(t) < 1 else 0.0

    def integrand(nu, which):
        k_I, k_Q = continuum_oracle(nu, gamma, B, eta)
        tau = 2 * (C_SI_M / nu) ** 2
        if which == "I":
            return response(nu) * k_I
        w = np.exp(1j * tau * depth_ref) * (1j * tau * s_depth) ** which[1]
        value = response(nu) * k_Q * w
        return value.real if which[0] == "re" else value.imag

    options = dict(epsabs=0, epsrel=1e-11, limit=200)
    I_ref = quad(integrand, lo, hi, args=("I",), **options)[0]
    assert_allclose(float(modes.I[0]), I_ref, rtol=1e-8)
    assert modes.V.shape == (1,) and float(modes.V[0]) == 0.0
    assert modes.P.shape == (2, 1)
    for b in range(2):
        ref = complex(
            quad(integrand, lo, hi, args=(("re", b),), **options)[0],
            quad(integrand, lo, hi, args=(("im", b),), **options)[0],
        )
        assert_allclose(complex(modes.P[b, 0]), ref, rtol=1e-8)
    # Q is negative for the continuum convention K_Q = -A G (checked without the phase).
    assert float(kernel.channel_modes(channels, gamma, B, 0.0, eta).P[0, 0].real) < 0
    # mu is ignored.
    other = kernel.channel_modes(
        channels, gamma, B, 0.7, eta, phase=phase, depth_ref=depth_ref, s_depth=s_depth
    )
    assert np.array_equal(np.asarray(other.I), np.asarray(modes.I))


@pytest.mark.parametrize("eta", [1.0, -1.0, 1 - 1e-6, -(1 - 1e-6), 0.0])
def test_eta_limits_are_finite_symmetric_and_differentiable(eta):
    channels, gamma, B = galactic_channel()
    kernel = ContinuumKernel()
    f = jax.jit(lambda e: kernel.channel_modes(channels, gamma, B, 0.0, e).I[0])
    value = float(f(eta))
    assert np.isfinite(value) and value >= 0
    assert_allclose(value, float(f(-eta)), rtol=1e-13)
    if abs(eta) == 1.0:
        assert value == 0.0
    if abs(eta) == 1 - 1e-6:
        assert value < 1e-6 * float(f(0.0))
    derivative = float(jax.jacfwd(f)(eta))
    assert np.isfinite(derivative)
    if eta == 0.0:
        assert abs(derivative) < 1e-6 * float(f(0.0))  # even in eta


def test_x_min_dispatch_both_sides():
    channels, gamma, B = galactic_channel()
    eta = 0.3
    b_perp = B * np.sqrt(1 - eta**2)
    scale = 3 * E_ESU * b_perp / (4 * np.pi * M_E * C_CGS) * gamma**2
    x_min = 1e-3
    kernel = ContinuumKernel(x_min=x_min)
    for factor in (1 + 1e-9, 1.0):
        k_I, k_Q = kernel.kernels(jnp.asarray(x_min * factor * scale), gamma, B, eta)
        assert np.isfinite(float(k_I)) and float(k_I) > 0 and float(k_Q) < 0
    for factor in (1 - 1e-9, 0.5):
        with pytest.raises(Exception, match="x_min"):
            eqx.filter_jit(kernel.kernels)(
                jnp.asarray(x_min * factor * scale), gamma, B, eta
            )
    # Below x_min at a node with B_perp = 0 is not an error: the kernel is zero there.
    k_I, _ = kernel.kernels(jnp.asarray(0.5 * x_min * scale), gamma, B, 1.0)
    assert float(k_I) == 0.0
    # A channel whose lowest node falls below x_min raises; a slightly higher one passes.
    low = Channels.bump([1.2 * x_min * scale], [0.5 * x_min * scale])
    with pytest.raises(Exception, match="x_min"):
        kernel.channel_modes(low, gamma, B, 0.0, eta)
    high = Channels.bump([2.0 * x_min * scale], [0.5 * x_min * scale])
    assert np.isfinite(float(kernel.channel_modes(high, gamma, B, 0.0, eta).I[0]))
    with pytest.raises(ValueError):
        ContinuumKernel(x_min=0.0)
    with pytest.raises(ValueError):
        ContinuumKernel(n_nodes_F=8)


def test_projection_is_mu_free_and_matches_high_order_eta_rule():
    channels, gamma, B = galactic_channel()
    kernel = ContinuumKernel(n_eta=64)
    truncation = Truncation(1, 3, 1)
    index = MomentIndex.build(truncation, components=kernel.components)
    phase = TaylorPhase(1)
    kwargs = dict(phase=phase, depth_ref=5.0, s_depth=2.0)
    projected = jax.jit(
        lambda g, b: kernel.angular_projection(
            channels, g, b, truncation=truncation, **kwargs
        )
    )(gamma, B)
    assert projected.I.shape == (1, index.n_lk) and projected.P.shape == (
        2,
        1,
        index.n_lk,
    )
    assert np.all(np.asarray(projected.V) == 0)
    eta_nodes, eta_weights = leggauss(200)
    I_eta = np.array(
        [
            float(kernel.channel_modes(channels, gamma, B, 0.0, e).I[0])
            for e in eta_nodes
        ]
    )
    P_eta = np.array(
        [
            np.asarray(
                kernel.channel_modes(channels, gamma, B, 0.0, e, **kwargs).P[:, 0]
            )
            for e in eta_nodes
        ]
    )
    for i, (l, k) in enumerate(index.pairs):
        if l > 0:
            assert float(projected.I[0, i]) == 0.0
            assert np.all(np.asarray(projected.P[:, 0, i]) == 0)
            continue
        weights = (2 * k + 1) / 2 * eta_weights * eval_legendre(k, eta_nodes)
        assert_allclose(float(projected.I[0, i]), np.sum(weights * I_eta), rtol=1e-9)
        for b in range(2):
            assert_allclose(
                complex(projected.P[b, 0, i]), np.sum(weights * P_eta[:, b]), rtol=1e-9
            )
    jac = jax.jacfwd(
        lambda g: kernel.angular_projection(channels, g, B, truncation=truncation).I
    )(gamma)
    assert np.all(np.isfinite(np.asarray(jac)))
    with pytest.raises(ValueError, match="quadrature"):
        kernel.angular_projection(
            channels, gamma, B, truncation=truncation, quadrature="other"
        )


def test_error_terms_and_tail_bound():
    channels, gamma, B = galactic_channel()
    kernel = ContinuumKernel()
    phys = kernel.physical_error(channels)
    assert phys.kind == "unbounded" and "continuum" in phys.note
    assert phys.manuscript_term == "E_phys"
    declared = ErrorTerm(jnp.zeros((1, 4)), "bound", "declared")
    assert ContinuumKernel(E_phys=declared).physical_error(channels) is declared
    assert kernel.truncation_error(None, channels).kind == "not_applicable"
    tail = kernel.tail_bound(channels, gamma, B, 0.4)
    assert tail.kind == "bound" and tail.value.shape == (1, 4)
    value = np.asarray(tail.value)
    assert np.all(value >= 0) and value[0, 3] == 0 and value[0, 1] == value[0, 2]
    modes = kernel.channel_modes(channels, gamma, B, 0.0, 0.4)
    assert value[0, 0] < 1e-12 * float(modes.I[0])
    assert np.all(np.asarray(kernel.tail_bound(channels, gamma, B, 1.0).value) == 0)
    assert hash(kernel.describe()) and dict(kernel.describe())["n_eta"] == 48
