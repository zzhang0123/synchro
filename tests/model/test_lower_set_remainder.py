"""Margin remainder of capped truncations in ``RemainderInputs`` (``bounds.py``).

Oracles: exact derivatives of the ``PolynomialTestKernel`` coefficient
polynomial along the telescoping paths ``p_k(t)``, exact weighted absolute
margin moments ``<|P_l P_k| |z^beta|>`` (NumPy, ``eval_legendre``), and the
covering inequality ``|predict - direct average| <= basis_remainder`` on a
discrete population. A capped basis is also checked against the uncapped
harmonic basis: the retained columns are the same numbers.
"""

import math

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

from syncmoments.model.basis import build_basis
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.channels import Channels
from syncmoments.model.index import Truncation
from syncmoments.model.moments import JointMoments
from syncmoments.model.predict import direct_channel_average, predict

from _basis_fixtures import small_harmonic
from _basis_stub import build_stub_basis
from test_remainder_probe import (
    P_LINE_NU,
    probe_kernel,
    probe_reference,
    probe_samples,
    probe_support,
)

CAPPED = Truncation(1, 1, 2, max_orders=(1, 1, None))
B_CAPPED = Truncation(1, 1, 2, max_orders=(None, 0, None))
DEPTH_CAPPED = Truncation(1, 1, 2, depth_degree=1, max_orders=(1, 1, None))
ORDERS = (0, 1, 2, 3)


def stub(truncation, certified_orders=ORDERS):
    kernel, c = probe_kernel(degree=2)
    channels = Channels.bump(P_LINE_NU, 0.1 * P_LINE_NU)
    basis = build_stub_basis(
        kernel,
        channels,
        truncation,
        probe_reference(),
        probe_support(),
        certified_orders=certified_orders,
    )
    return kernel, c, channels, basis


def displacements(samples):
    return np.asarray(probe_reference().z(samples.gamma, samples.B, samples.depth))


def polynomial_derivative(c, beta, point):
    """``d^beta`` of ``sum_rs c[..., r, s] z_g^r z_B^s`` at ``point`` (``b`` must be 0)."""
    if beta[2]:
        return np.zeros(c.shape[:-2])
    out = np.zeros(c.shape[:-2])
    for r in range(c.shape[-2]):
        for s in range(c.shape[-1]):
            if r >= beta[0] and s >= beta[1]:
                fall = math.perm(r, beta[0]) * math.perm(s, beta[1])
                out = out + c[..., r, s] * fall * point[0] ** (r - beta[0]) * point[
                    1
                ] ** (s - beta[1])
    return out


def path_points(zn, k, ts):
    out = []
    for t in ts:
        p = np.zeros(3)
        p[:k] = zn[:k]
        p[k] = t * zn[k]
        out.append(p)
    return out


def margin_oracle(c, basis, samples, component):
    """``max_{n, t} |d^beta K_{X;lk}(p_k(t))|`` over ``t in {0, 0.5, 1}``, ``(n_ch, n_lk, n_m)``."""
    rows = basis.truncation.margin_rows()
    z = displacements(samples)
    out = np.zeros((c.shape[1], basis.index.n_lk, len(rows)))
    for m, beta in enumerate(rows):
        k = next(i for i, b in enumerate(beta) if b)
        for zn in z:
            for point in path_points(zn, k, (0.0, 0.5, 1.0)):
                for i, (l, kk) in enumerate(basis.index.pairs):
                    value = polynomial_derivative(c[component, :, l, kk], beta, point)
                    out[:, i, m] = np.maximum(out[:, i, m], np.abs(value))
    return out


def moments_oracle(basis, samples):
    rows = basis.truncation.margin_rows()
    w = np.asarray(samples.normalised_weights())
    z = displacements(samples)
    mu, eta = np.asarray(samples.mu), np.asarray(samples.eta)
    out = np.zeros((basis.index.n_lk, len(rows)))
    for i, (l, k) in enumerate(basis.index.pairs):
        leg = np.abs(eval_legendre(l, mu) * eval_legendre(k, eta))
        for m, beta in enumerate(rows):
            out[i, m] = np.sum(w * leg * np.prod(np.abs(z) ** np.asarray(beta), axis=1))
    return out


@pytest.mark.parametrize("truncation", [CAPPED, B_CAPPED], ids=["gB", "B0"])
def test_capped_probe_fills_margin_envelopes_that_match_the_oracles(truncation):
    kernel, c, channels, basis = stub(truncation)
    samples = probe_samples()
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    n_m = len(truncation.margin_rows())
    assert inputs.H is None and inputs.absolute_moments is None
    assert inputs.margin_H.shape == (2, 3, basis.index.n_lk, n_m)
    assert inputs.kind == "estimate"
    for x, component in ((0, 0), (2, 2)):  # I and V (index 1 of c is Q)
        assert_allclose(
            np.asarray(inputs.margin_H[:, x]),
            margin_oracle(c, basis, samples, component),
            rtol=1e-11,
            atol=1e-13,
        )
    assert_allclose(
        np.asarray(inputs.margin_moments), moments_oracle(basis, samples), rtol=1e-12
    )


@pytest.mark.parametrize("truncation", [CAPPED, B_CAPPED], ids=["gB", "B0"])
def test_capped_remainder_covers_the_finite_error(truncation):
    kernel, c, channels, basis = stub(truncation)
    samples = probe_samples()
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    moments = JointMoments.from_samples(samples, basis.index, probe_reference())
    pred = predict(basis, moments, amplitude=1.3, errors=inputs)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.3, reference=probe_reference()
    )
    term = pred.budget.basis_remainder
    assert term.kind == "estimate" and "margin" in term.note
    diff = np.abs(np.asarray(pred.stokes) - np.asarray(direct.stokes))
    scale = np.max(np.abs(np.asarray(direct.stokes)))
    assert np.max(diff[:, [0, 3]]) > 1e-3 * scale  # the caps remove z^2 terms
    assert np.all(diff <= np.asarray(term.value) * (1 + 1e-10) + 1e-14 * scale)


def test_capped_depth_layout_uses_the_two_dimensional_margin():
    kernel, c, channels, basis = stub(DEPTH_CAPPED)
    samples = probe_samples()
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    rows = DEPTH_CAPPED.margin_rows()
    assert all(beta[2] == 0 for beta in rows)
    weights = np.array([1.0 / math.prod(map(math.factorial, b)) for b in rows])
    expected = np.asarray(inputs.rho_ang)[:, 1] + np.einsum(
        "clm,lm,m->c",
        np.asarray(inputs.margin_H[:, 1]),
        np.asarray(inputs.margin_moments),
        weights,
    )
    assert_allclose(np.asarray(inputs.intrinsic_residual), expected, rtol=1e-12)
    moments = JointMoments.from_samples(samples, basis.index, probe_reference())
    pred = predict(basis, moments, amplitude=1.0, errors=inputs)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=probe_reference()
    )
    term = pred.budget.basis_remainder
    assert term.kind == "estimate" and term.value.shape == (2, 4)
    diff = np.abs(np.asarray(pred.stokes) - np.asarray(direct.stokes))
    scale = np.max(np.abs(np.asarray(direct.stokes)))
    assert np.all(diff <= np.asarray(term.value) * (1 + 1e-10) + 1e-14 * scale)


def test_capped_truncation_refuses_the_operator_norm_inputs():
    kernel, c, channels, basis = stub(CAPPED)
    n_lk = basis.index.n_lk
    inputs = RemainderInputs(np.ones((2, 3)), np.ones((2, 3, n_lk)), np.ones((2, n_lk)))
    term = inputs.basis_remainder(basis, 1.0)
    assert term.kind == "unbounded"
    assert "margin_H" in term.note and "margin_moments" in term.note


def test_supplied_margin_envelopes_give_a_bound_and_are_shape_checked():
    kernel, c, channels, basis = stub(CAPPED)
    samples = probe_samples()
    n_m = len(CAPPED.margin_rows())
    H = np.full((2, 3, basis.index.n_lk, n_m), 2.0)
    rho = np.full((2, 3), 0.1)
    inputs = RemainderInputs.from_samples(
        samples, basis, derivative_envelope=H, angular_residual=rho
    )
    assert inputs.kind == "bound"
    assert_allclose(np.asarray(inputs.margin_H), H)
    rows = CAPPED.margin_rows()
    factor = np.array([1.0 / math.prod(map(math.factorial, b)) for b in rows])
    I = 0.1 + np.einsum("lm,m->", np.asarray(inputs.margin_moments), 2.0 * factor)
    term = inputs.basis_remainder(basis, 2.0)
    assert term.kind == "bound"
    assert_allclose(np.asarray(term.value)[:, 0], 2.0 * I, rtol=1e-13)
    with pytest.raises(ValueError, match="derivative_envelope"):
        RemainderInputs.from_samples(
            samples, basis, derivative_envelope=H[..., :1], angular_residual=rho
        )
    with pytest.raises(ValueError, match="margin_H"):
        RemainderInputs(rho, margin_H=np.ones((2, 3, 4)))


def test_capped_probe_needs_every_margin_order_validated():
    kernel, c, channels, basis = stub(CAPPED, certified_orders=(0, 1, 2))
    assert max(sum(b) for b in CAPPED.margin_rows()) == 3
    with pytest.raises(ValueError, match="certified"):
        RemainderInputs.from_samples(probe_samples(), basis, kernel=kernel)
    low = Truncation(1, 1, 2, depth_degree=0, max_orders=(1, 1, None))
    assert max(sum(b) for b in low.margin_rows()) == 3  # (2, 1, 0) is in S
    kernel, c, channels, basis = stub(low, certified_orders=(0, 1, 2))
    with pytest.raises(ValueError, match="certified"):
        RemainderInputs.from_samples(probe_samples(), basis, kernel=kernel)


def test_uncapped_inputs_keep_the_operator_norm_form():
    kernel, c, channels, basis = stub(Truncation(1, 1, 1))
    inputs = RemainderInputs.from_samples(
        probe_samples(), basis, kernel=kernel, angular_residual="probe"
    )
    assert inputs.margin_H is None and inputs.margin_moments is None
    term = inputs.basis_remainder(basis, 1.0)
    assert "margin" not in term.note and "(N+1)!" in term.note


def test_capped_harmonic_basis_keeps_the_uncapped_columns():
    kernel, channels, reference, support = small_harmonic()
    common = dict(support=support, convergence=False)
    full = build_basis(kernel, channels, Truncation(2, 2, 2), reference, **common)
    capped_t = Truncation(2, 2, 2, max_orders=(None, 1, 1))
    capped = build_basis(kernel, channels, capped_t, reference, **common)
    index, full_index = capped.index, full.index
    assert index.n0 < full_index.n0 and index.n2 < full_index.n2
    cols0 = [full_index.position(0, *row) for row in index.h0]
    cols2 = [full_index.position(2, *row) - full_index.n0 for row in index.h2]
    for name, cols in (("I_basis", cols0), ("V_basis", cols0), ("P_basis", cols2)):
        a = np.asarray(getattr(capped, name))
        b = np.asarray(getattr(full, name))[:, cols]
        assert_allclose(a, b, rtol=0, atol=1e-13 * np.max(np.abs(b)))
    assert dict(capped.provenance.truncation)["n0"] == index.n0
