"""``RemainderInputs.from_samples`` (``syncmoments.model._remainder_probe``) against
NumPy oracles on the ``PolynomialTestKernel``: Frobenius norms of the order-``N+1``
derivative tensors at the probe points, absolute moments, the measured angular
residual, the ``app: depth moments`` inputs, and the covering inequality against the
direct average.
"""

from itertools import product as _product

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

import syncmoments  # noqa: F401
from syncmoments.constants import C_SI_M as _C
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.channels import Channels
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import PolynomialTestKernel
from syncmoments.model.moments import (
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)
from syncmoments.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis

P_LINE_NU = np.array([2.0e8, 6.0e8])
P_TAU = 2 * (_C / P_LINE_NU) ** 2
P_S_DEPTH = 0.4 / P_TAU[0]
P_DEPTH_REF = 1.0 * P_S_DEPTH
P_GAMMA0, P_B0 = 10.0, 3.0


def probe_kernel(seed=21, L=2, degree=2):
    """Parity-respecting polynomial kernel of total degree ``degree`` in ``z``."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, 2, L + 1, L + 1, degree + 1, degree + 1))
    for r, s_ in _product(range(degree + 1), repeat=2):
        if r + s_ > degree:
            c[..., r, s_] = 0.0
    for l, k in _product(range(L + 1), repeat=2):
        c[[0, 1] if (l + k) % 2 else [2], :, l, k] = 0.0
    kernel = PolynomialTestKernel(
        c, P_LINE_NU, gamma0=P_GAMMA0, B0=P_B0, s_gamma=P_GAMMA0, s_B=P_B0
    )
    return kernel, c


def probe_samples(seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, 6)
    return PopulationSamples(
        P_GAMMA0 * (1 + 0.3 * t),
        P_B0 * (1 + 0.2 * t**2 - 0.1 * t),
        np.clip(0.5 * t + 0.2 * rng.standard_normal(6), -0.9, 0.9),
        np.clip(-0.6 * t, -0.9, 0.9),
        0.5 + 1.2 * t,
        P_DEPTH_REF + P_S_DEPTH * (0.8 * t + 0.3),
        weights=rng.uniform(0.5, 2.0, 6),
    )


def probe_reference():
    return Reference(P_GAMMA0, P_B0, P_DEPTH_REF, scales=(P_GAMMA0, P_B0, P_S_DEPTH))


def probe_support():
    return Support((5.0, 15.0), (1.0, 5.0), (0.0, 10 * P_S_DEPTH), truncated=False)


def hessian_oracle(c, z, index):
    """Frobenius norms of the order-2 tensors of ``K_{X;lk}`` at ``z`` (``I, P, V``) and
    the ``(z_gamma, z_B)`` block for ``P``; ``c`` has total degree 2."""
    zg, zB, zd = z
    n_lk = index.n_lk
    out = np.zeros((2, 3, n_lk))
    out_2d = np.zeros((2, n_lk))
    for i, (l, k) in enumerate(index.pairs):
        for x, s_ in ((0, 0), (1, 1), (2, 2)):
            cc = c[s_, :, l, k]  # (n_ch, 3, 3) in (r, s)
            hess = np.zeros((2, 3, 3), dtype=complex)
            hess[:, 0, 0] = 2 * cc[:, 2, 0]
            hess[:, 1, 1] = 2 * cc[:, 0, 2]
            hess[:, 0, 1] = hess[:, 1, 0] = cc[:, 1, 1]
            if x == 1:
                K = cc[:, 0, 0] + cc[:, 1, 0] * zg + cc[:, 0, 1] * zB
                K += cc[:, 2, 0] * zg**2 + cc[:, 1, 1] * zg * zB + cc[:, 0, 2] * zB**2
                dg = cc[:, 1, 0] + 2 * cc[:, 2, 0] * zg + cc[:, 1, 1] * zB
                dB = cc[:, 0, 1] + 2 * cc[:, 0, 2] * zB + cc[:, 1, 1] * zg
                factor = 1j * P_TAU * P_S_DEPTH
                hess[:, 0, 2] = hess[:, 2, 0] = factor * dg
                hess[:, 1, 2] = hess[:, 2, 1] = factor * dB
                hess[:, 2, 2] = factor**2 * K
                out_2d[:, i] = np.sqrt(
                    np.sum(np.abs(hess[:, :2, :2]) ** 2, axis=(1, 2))
                )
            out[:, x, i] = np.sqrt(np.sum(np.abs(hess) ** 2, axis=(1, 2)))
    return out, out_2d


def angular_residual_oracle(c, samples, index):
    """``<|K_X(p) - sum_{lk retained} K_{X;lk}(z) P_l P_k|>`` for the polynomial kernel."""
    w = np.asarray(samples.normalised_weights())
    zg = (np.asarray(samples.gamma) - P_GAMMA0) / P_GAMMA0
    zB = (np.asarray(samples.B) - P_B0) / P_B0
    L = c.shape[2] - 1
    residual = np.zeros((2, 3))
    for n in range(samples.size):
        pg, pb = zg[n] ** np.arange(3), zB[n] ** np.arange(3)
        coeff = np.einsum("sjlkrt,r,t->sjlk", c, pg, pb)  # exact K_{X;lk}(z_n)
        pl = eval_legendre(np.arange(L + 1), float(samples.mu[n]))
        pk = eval_legendre(np.arange(L + 1), float(samples.eta[n]))
        full = np.einsum("sjlk,l,k->sj", coeff, pl, pk)
        retained = np.zeros_like(full)
        for l, k in index.pairs:
            retained += coeff[:, :, l, k] * pl[l] * pk[k]
        residual += w[n] * np.abs(full - retained)[[0, 1, 2]].T
    return residual


@pytest.fixture(scope="module")
def probe_case():
    kernel, c = probe_kernel()
    channels = Channels.bump(P_LINE_NU, 0.1 * P_LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(1, 1, 1), probe_reference(), probe_support()
    )
    samples = probe_samples()
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    return kernel, c, channels, basis, samples, inputs


def test_from_samples_envelopes_match_oracles(probe_case):
    kernel, c, channels, basis, samples, inputs = probe_case
    index = basis.index
    assert inputs.kind == "estimate"
    assert inputs.H.shape == (2, 3, index.n_lk)
    assert inputs.absolute_moments.shape == (2, index.n_lk)
    assert inputs.rho_ang.shape == (2, 3)
    assert inputs.depth_tail is None and inputs.intrinsic_residual is None
    z = np.asarray(probe_reference().z(samples.gamma, samples.B, samples.depth))
    points = [np.zeros(3)] + [t * zn for t in (0.5, 1.0) for zn in z]
    H = np.max([hessian_oracle(c, pt, index)[0] for pt in points], axis=0)
    assert_allclose(np.asarray(inputs.H), H, rtol=1e-11)
    w = np.asarray(samples.normalised_weights())
    legendre = np.array(
        [
            np.abs(
                eval_legendre(l, np.asarray(samples.mu))
                * eval_legendre(k, np.asarray(samples.eta))
            )
            for l, k in index.pairs
        ]
    ).T
    n2 = np.linalg.norm(z[:, :2], axis=1) ** 2
    n3 = np.linalg.norm(z, axis=1) ** 2
    assert_allclose(
        np.asarray(inputs.absolute_moments),
        [(w * n2) @ legendre, (w * n3) @ legendre],
        rtol=1e-12,
    )
    assert_allclose(
        np.asarray(inputs.rho_ang),
        angular_residual_oracle(c, samples, index),
        rtol=1e-11,
    )
    assert np.all(
        np.asarray(inputs.rho_ang) > 0
    )  # Legendre degree 2 truncated at L = 1


def test_from_samples_remainder_covers_the_finite_error(probe_case):
    kernel, c, channels, basis, samples, inputs = probe_case
    moments = JointMoments.from_samples(samples, basis.index, probe_reference())
    pred = predict(basis, moments, amplitude=1.3, errors=inputs)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.3, reference=probe_reference()
    )
    term = pred.budget.basis_remainder
    assert term.kind == "estimate"
    diff = np.abs(np.asarray(pred.stokes) - np.asarray(direct.stokes))
    scale = np.max(np.abs(np.asarray(direct.stokes)))
    assert np.max(diff) > 1e-3 * scale
    assert np.all(diff <= np.asarray(term.value) * (1 + 1e-10) + 1e-14 * scale)


def test_from_samples_options_and_refusals(probe_case):
    kernel, c, channels, basis, samples, inputs = probe_case
    index = basis.index
    without = RemainderInputs.from_samples(samples, basis, kernel=kernel)
    assert without.rho_ang is None and without.kind == "estimate"
    assert without.basis_remainder(basis, 1.0).kind == "unbounded"
    H = np.ones((2, 3, index.n_lk))
    rho = np.ones((2, 3))
    supplied = RemainderInputs.from_samples(
        samples, basis, derivative_envelope=H, angular_residual=rho
    )
    assert supplied.kind == "bound"
    assert_allclose(np.asarray(supplied.H), H)
    assert_allclose(
        np.asarray(supplied.absolute_moments), np.asarray(inputs.absolute_moments)
    )
    mixed = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, derivative_envelope=H, angular_residual="probe"
    )
    assert mixed.kind == "estimate"
    with pytest.raises(ValueError, match="kernel"):
        RemainderInputs.from_samples(samples, basis)
    with pytest.raises(ValueError):
        RemainderInputs.from_samples(samples, basis, derivative_envelope=H[:, :2])
    with pytest.raises(ValueError):
        RemainderInputs.from_samples(
            samples, basis, derivative_envelope=H, angular_residual=rho[:1]
        )
    with pytest.raises(ValueError):
        RemainderInputs.from_samples(
            samples, basis, kernel=kernel, derivative_envelope="bogus"
        )
    stub = dict(channels=channels, truncation=Truncation(1, 1, 1))
    uncertified = build_stub_basis(
        kernel,
        channels,
        Truncation(1, 1, 1),
        probe_reference(),
        probe_support(),
        certified_orders=(0,),
    )
    with pytest.raises(ValueError, match="certified"):
        RemainderInputs.from_samples(samples, uncertified, kernel=kernel)
    allowed = build_stub_basis(
        kernel,
        stub["channels"],
        stub["truncation"],
        probe_reference(),
        probe_support(),
        certified_orders=(0, 1, 2),
    )
    RemainderInputs.from_samples(samples, allowed, kernel=kernel, segment_points=(1.0,))
    beyond = build_stub_basis(
        kernel,
        stub["channels"],
        stub["truncation"],
        probe_reference(),
        probe_support(),
        certified_orders=(0, 1),
    )
    with pytest.raises(ValueError, match="certified"):  # order 2 not validated (R30)
        RemainderInputs.from_samples(samples, beyond, kernel=kernel)


def test_from_samples_appendix_c_layout():
    kernel, c = probe_kernel()
    channels = Channels.bump(P_LINE_NU, 0.1 * P_LINE_NU)
    basis = build_stub_basis(
        kernel,
        channels,
        Truncation(1, 1, 1, depth_degree=2),
        probe_reference(),
        probe_support(),
    )
    samples = probe_samples()
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    rows = [row for row in basis.index.h2 if row[4] == 0]
    assert inputs.depth_tail.shape == (len(rows),)
    assert inputs.intrinsic_residual.shape == (2,)
    w = np.asarray(samples.normalised_weights())
    zg = (np.asarray(samples.gamma) - P_GAMMA0) / P_GAMMA0
    zB = (np.asarray(samples.B) - P_B0) / P_B0
    dd = np.abs(np.asarray(samples.depth) - P_DEPTH_REF) ** 3
    for a, (l, k, r, s_, _) in enumerate(rows):
        chi = np.abs(
            eval_legendre(l, np.asarray(samples.mu))
            * eval_legendre(k, np.asarray(samples.eta))
            * zg**r
            * zB**s_
        )
        assert_allclose(float(inputs.depth_tail[a]), np.sum(w * chi * dd), rtol=1e-12)
    z = np.asarray(probe_reference().z(samples.gamma, samples.B, samples.depth))
    points = [np.zeros(3)] + [t * zn for t in (0.5, 1.0) for zn in z]
    H2 = np.max([hessian_oracle(c, pt, basis.index)[1] for pt in points], axis=0)
    expected = (
        np.asarray(inputs.rho_ang)[:, 1]
        + H2 @ np.asarray(inputs.absolute_moments)[0] / 2
    )
    assert_allclose(np.asarray(inputs.intrinsic_residual), expected, rtol=1e-11)
    moments = JointMoments.from_samples(samples, basis.index, probe_reference())
    term = predict(basis, moments, amplitude=1.0, errors=inputs).budget.basis_remainder
    assert term.kind == "estimate" and term.value.shape == (2, 4)
