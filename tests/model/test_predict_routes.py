"""``direct_channel_average`` phase routes and the real ``build_basis`` through
``predict`` on the polynomial oracle kernel (split from ``test_predict.py``)."""

import json

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

import synchro  # noqa: F401
from synchro.model.basis import build_basis
from synchro.model.channels import Channels
from synchro.model.index import Truncation
from synchro.model.moments import JointMoments
from synchro.model.phase import GaussianScreen, TaylorPhase
from synchro.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis
from _predict_helpers import (
    B0,
    DEPTH_REF,
    GAMMA0,
    LINE_NU,
    S_DEPTH,
    TAU,
    full_inputs,
    nine_atoms,
    polynomial_direct,
    polynomial_kernel,
    reference,
    samples_of,
    support,
)


@pytest.fixture(scope="module")
def poly_basis():
    kernel, c = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(2, 2, 2), reference(), support()
    )
    return kernel, c, channels, basis


def test_direct_average_screen_route_and_phase_options(poly_basis):
    kernel, c, channels, _ = poly_basis
    pop = nine_atoms(constant_depth=False)
    samples = samples_of(pop)
    screen = GaussianScreen(jnp.asarray(DEPTH_REF), jnp.asarray(0.3 * S_DEPTH))
    pred = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, phase=screen
    )
    w = pop["w"] / pop["w"].sum()
    expected = np.zeros(3, dtype=complex)
    for n in range(9):
        zg, zB = (pop["gamma"][n] - GAMMA0) / GAMMA0, (pop["B"][n] - B0) / B0
        pl, pk = eval_legendre(np.arange(3), pop["mu"][n]), eval_legendre(
            np.arange(3), pop["eta"][n]
        )
        KQ = np.einsum(
            "jlkrt,l,k,r,t->j", c[1], pl, pk, zg ** np.arange(3), zB ** np.arange(3)
        )
        expected += w[n] * np.exp(2j * pop["phi"][n]) * KQ
    expected *= np.exp(1j * TAU * DEPTH_REF - 0.5 * (TAU * 0.3 * S_DEPTH) ** 2)
    assert_allclose(
        np.asarray(pred.stokes[:, 1]) + 1j * np.asarray(pred.stokes[:, 2]),
        expected,
        rtol=1e-12,
    )
    names = [name for name, _ in pred.budget.assumption]
    assert names == ["independent_screen", "gaussian_screen"]
    assert all(term.kind == "unbounded" for _, term in pred.budget.assumption)
    assert "gaussian_screen" in {r.name for r in pred.provenance.assumptions}
    # A Taylor route of any degree means the exact per-emitter phase.
    taylor = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, phase=TaylorPhase(2)
    )
    assert_allclose(
        np.asarray(taylor.stokes), polynomial_direct(pop, c, 1.0), atol=1e-13
    )
    with pytest.raises(ValueError):
        direct_channel_average(samples, kernel, channels, amplitude=1.0, batch_size=0)


def test_real_build_basis_agrees_with_stub_and_direct(poly_basis):
    kernel, c, channels, stub = poly_basis
    basis = build_basis(
        kernel,
        channels,
        Truncation(2, 2, 2),
        reference(),
        support=support(),
        convergence=False,
    )
    pop = nine_atoms(constant_depth=False)
    samples = samples_of(pop)
    moments = JointMoments.from_samples(samples, basis.index, reference())
    real = predict(basis, moments, amplitude=2.0, **full_inputs(basis))
    stubbed = predict(stub, moments, amplitude=2.0, **full_inputs(stub))
    scale = np.max(np.abs(np.asarray(real.stokes)))
    assert_allclose(
        np.asarray(real.stokes), np.asarray(stubbed.stokes), atol=1e-13 * scale
    )
    assert_allclose(
        np.asarray(real.stokes),
        polynomial_direct(pop, c, 2.0, total_degree=2),
        atol=1e-12 * scale,
    )
    assert real.budget.numerical.kind == "unbounded"
    assert real.provenance.certified_orders == basis.certified_orders
    json.dumps(real.to_dict())


def test_scalar_statistical_input_broadcasts_over_moments(poly_basis):
    """``statistical_input=0`` (the value the unbounded note recommends) and any
    nonnegative scalar ``Delta`` are broadcast to ``(n_real,)``; other shapes raise."""
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    missing = predict(basis, moments, amplitude=2.0).budget.statistical_input
    assert missing.kind == "unbounded" and "statistical_input=" in missing.note
    zero = predict(basis, moments, amplitude=2.0, statistical_input=0)
    term = zero.budget.statistical_input
    assert term.kind == "bound" and term.value.shape == (basis.n_ch, 4)
    assert float(jnp.max(term.value)) == 0.0
    absC = np.abs(np.asarray(basis.response_matrix()))
    scalar = predict(basis, moments, amplitude=2.0, statistical_input=1e-3)
    vector = predict(
        basis,
        moments,
        amplitude=2.0,
        statistical_input=np.full(basis.index.n_real, 1e-3),
    )
    expected = 2.0 * (absC @ np.full(absC.shape[1], 1e-3)).reshape(-1, 4)
    for pred in (scalar, vector):
        assert_allclose(
            np.asarray(pred.budget.statistical_input.value), expected, rtol=1e-14
        )
    with pytest.raises(ValueError, match="shape"):
        predict(basis, moments, amplitude=2.0, statistical_input=np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        predict(basis, moments, amplitude=2.0, statistical_input=np.zeros((1, 1)))
