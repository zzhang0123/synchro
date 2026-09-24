"""Tests of ``synchro.model.predict``: exactness on the polynomial oracle kernel,
budget slots, JSON round trips, propagation, JIT/autodiff and the real
``build_basis``. The harmonic-kernel assumption inequalities are in
``test_predict_assumptions.py``.

The spectral basis is the test stand-in of ``_basis_stub`` (nested ``jacfwd``
of the kernel projections); oracles are NumPy sums over discrete populations.
"""

import json

import equinox as eqx
import jax
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model.channels import Channels
from synchro.model.errors import ErrorBudget, ErrorTerm
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, Reference
from synchro.model.predict import Prediction, direct_channel_average, predict

from _basis_stub import build_stub_basis, response_oracle
from _predict_helpers import (
    B0,
    DEPTH_REF,
    GAMMA0,
    LINE_NU,
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


# -- exactness on the polynomial kernel ---------------------------------------------


def test_predict_equals_direct_average_for_polynomial_kernel(poly_basis):
    kernel, c, channels, basis = poly_basis
    pop = nine_atoms()
    samples = samples_of(pop)
    moments = JointMoments.from_samples(samples, basis.index, reference())
    amplitude = 3.5
    pred = predict(basis, moments, amplitude=amplitude)
    direct = direct_channel_average(
        samples,
        kernel,
        channels,
        amplitude=amplitude,
        reference=reference(),
        support=support(),
    )
    oracle = polynomial_direct(pop, c, amplitude)
    scale = np.max(np.abs(oracle))
    assert isinstance(pred, Prediction) and isinstance(direct, Prediction)
    assert pred.stokes.shape == (3, 4)
    assert_allclose(np.asarray(pred.stokes), oracle, rtol=0, atol=1e-12 * scale)
    assert_allclose(np.asarray(direct.stokes), oracle, rtol=0, atol=1e-12 * scale)
    assert_allclose(
        np.asarray(pred.stokes), np.asarray(direct.stokes), atol=1e-12 * scale
    )
    # C layout: the NumPy contraction of the response matrix agrees.
    assert_allclose(
        np.asarray(pred.stokes),
        amplitude * response_oracle(basis, moments.to_vector()),
        rtol=1e-13,
    )
    assert direct.budget.basis_remainder.kind == "not_applicable"
    assert direct.budget.harmonic_truncation.kind == "not_applicable"
    assert direct.budget.statistical_input.kind == "bound"
    assert direct.moments.index.truncation == Truncation(0, 0, 0)
    assert_allclose(direct.moments.m0, [1.0])
    assert "direct" in direct.summary().lower()


def test_depth_taylor_truncation_is_the_only_difference(poly_basis):
    """With varying depth the finite response is the total-degree-N Taylor sum."""
    kernel, c, channels, basis = poly_basis
    pop = nine_atoms(constant_depth=False)
    samples = samples_of(pop)
    moments = JointMoments.from_samples(samples, basis.index, reference())
    pred = predict(basis, moments, amplitude=1.0)
    truncated = polynomial_direct(pop, c, 1.0, total_degree=2)
    exact = polynomial_direct(pop, c, 1.0)
    scale = np.max(np.abs(exact))
    assert_allclose(np.asarray(pred.stokes), truncated, atol=1e-12 * scale)
    assert np.max(np.abs(truncated - exact)[:, 1:3]) > 1e-4 * scale
    assert_allclose(truncated[:, [0, 3]], exact[:, [0, 3]], atol=1e-14 * scale)
    direct = direct_channel_average(samples, kernel, channels, amplitude=1.0)
    assert_allclose(np.asarray(direct.stokes), exact, atol=1e-12 * scale)


def test_predict_rejects_mismatched_index_and_reference(poly_basis):
    _, _, _, basis = poly_basis
    other = MomentIndex.build(Truncation(1, 1, 1))
    moments = JointMoments.from_samples(samples_of(nine_atoms()), other, reference())
    with pytest.raises(ValueError, match="index"):
        predict(basis, moments, amplitude=1.0)
    shifted = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, Reference(GAMMA0 * 1.01, B0, DEPTH_REF)
    )
    with pytest.raises(Exception):
        predict(basis, shifted, amplitude=1.0)
    with pytest.raises(Exception):
        predict(
            basis,
            JointMoments.from_samples(
                samples_of(nine_atoms()), basis.index, reference()
            ),
            amplitude=-1.0,
        )


# -- budget slots ------------------------------------------------------------------


def test_budget_slots_kinds_and_total(poly_basis):
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    pred = predict(basis, moments, amplitude=2.0, **full_inputs(basis))
    budget = pred.budget
    assert isinstance(budget, ErrorBudget)
    kinds = {name: term.kind for name, term in budget.terms()}
    assert kinds["basis_remainder"] == "bound"
    assert kinds["statistical_input"] == "bound"
    assert kinds["physical_kernel"] == "bound"  # polynomial kernel declares zero
    assert kinds["harmonic_truncation"] == "not_applicable"
    assert kinds["excluded_tail"] == "unbounded"  # support.truncated is None
    assert kinds["numerical"] == "unbounded"  # stub basis has no estimate
    assert kinds["screen_exponent"] == "not_applicable"
    assert kinds["depth_model"] == "bound"
    assert kinds["amplitude"] == "bound"
    assert budget.assumption == ()
    total = budget.total()
    assert total.kind == "unbounded" and set(budget.unbounded()) == {
        "excluded_tail",
        "numerical",
    }
    assert pred.propagate(np.eye(12)).kind == "unbounded"
    # Declared-complete support and a numerical estimate close the budget.
    closed = eqx.tree_at(
        lambda b: (b.support, b.kernel_terms.numerical),
        basis,
        (
            support(False),
            ErrorTerm(np.full((3, 4), 1e-3), "estimate", "two-quadrature"),
        ),
    )
    pred = predict(closed, moments, amplitude=2.0, **full_inputs(basis))
    total = pred.budget.total()
    assert total.kind == "estimate" and total.value.shape == (3, 4)
    assert_allclose(np.asarray(total.value), 2.0 * 1e-3)
    assert (
        pred.budget.excluded_tail.kind == "bound"
        and "declared" in pred.budget.excluded_tail.note
    )


def test_budget_values_follow_their_definitions(poly_basis):
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    index = basis.index
    rng = np.random.default_rng(2)
    delta = rng.uniform(0, 1e-3, index.n_real)
    amplitude, d_amp = 2.5, 0.2
    pred = predict(
        basis,
        moments,
        amplitude=amplitude,
        statistical_input=delta,
        amplitude_uncertainty=d_amp,
        excluded_tail=np.full((3, 4), 0.01),
    )
    C = np.asarray(basis.response_matrix())
    assert_allclose(
        np.asarray(pred.budget.statistical_input.value),
        amplitude * (np.abs(C) @ delta).reshape(3, 4),
        rtol=1e-13,
    )
    # delta A (|C m| + valued per-electron slots): the statistical term enters
    # through the cross term; excluded_tail is outside N_src (R05).
    assert_allclose(
        np.asarray(pred.budget.amplitude.value),
        d_amp
        * (
            np.abs(np.asarray(pred.stokes)) / amplitude
            + (np.abs(C) @ delta).reshape(3, 4)
        ),
        rtol=1e-13,
    )
    assert pred.budget.excluded_tail.kind == "bound"
    assert_allclose(np.asarray(pred.budget.excluded_tail.value), 0.01)
    # Truncated support without a tail term: required input missing.
    truncated = eqx.tree_at(lambda b: b.support, basis, support(True))
    term = predict(truncated, moments, amplitude=1.0).budget.excluded_tail
    assert term.kind == "unbounded" and "required" in term.note
    given = predict(
        truncated,
        moments,
        amplitude=1.0,
        excluded_tail=ErrorTerm(np.zeros((3, 4)), "estimate", "model"),
    )
    assert given.budget.excluded_tail.kind == "estimate"
    # Per-electron kernel terms scale with the amplitude.
    scaled = eqx.tree_at(
        lambda b: b.kernel_terms.harmonic_truncation,
        basis,
        ErrorTerm(np.ones((3, 4)), "estimate", "probe"),
    )
    assert_allclose(
        np.asarray(
            predict(scaled, moments, amplitude=4.0).budget.harmonic_truncation.value
        ),
        4.0,
    )


def test_to_dict_json_round_trip_and_summary(poly_basis):
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    pred = predict(basis, moments, amplitude=1.0, **full_inputs(basis))
    d = pred.to_dict()
    text = json.dumps(d)
    back = json.loads(text)
    assert set(back) >= {
        "stokes",
        "units",
        "amplitude",
        "moments",
        "budget",
        "provenance",
    }
    assert np.asarray(back["stokes"]).shape == (3, 4)
    assert set(back["moments"]) >= {"index", "vector", "labels"}
    assert back["budget"]["total"]["value"] is None
    assert set(back["budget"]["total"]["unbounded"]) == {"excluded_tail", "numerical"}
    assert back["budget"]["basis_remainder"]["kind"] == "bound"
    assert back["budget"]["harmonic_truncation"]["manuscript_term"] == "E_num"
    assert back["provenance"]["units"] == basis.provenance.units
    text = pred.summary()
    assert "unbounded" in text and "excluded_tail" in text
    closed = eqx.tree_at(
        lambda b: (b.support, b.kernel_terms.numerical),
        basis,
        (support(False), ErrorTerm(np.zeros((3, 4)), "bound", "exact")),
    )
    pred = predict(closed, moments, amplitude=1.0, **full_inputs(basis))
    back = json.loads(json.dumps(pred.to_dict()))
    assert (
        back["budget"]["total"]["value"] is not None
        and back["budget"]["total"]["kind"] == "bound"
    )
    assert "unbounded" not in pred.summary()


# -- propagate ----------------------------------------------------------------------


def test_propagate_with_and_without_response_uncertainty(poly_basis):
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    closed = eqx.tree_at(
        lambda b: (b.support, b.kernel_terms.numerical),
        basis,
        (
            support(False),
            ErrorTerm(np.full((3, 4), 2e-3), "estimate", "two-quadrature"),
        ),
    )
    pred = predict(closed, moments, amplitude=1.5, **full_inputs(basis))
    rng = np.random.default_rng(9)
    R = rng.standard_normal((5, 12))
    dR = np.abs(rng.standard_normal((5, 12))) * 1e-3
    envelope = np.asarray(pred.budget.total().value).ravel()
    stokes = np.asarray(pred.stokes).ravel()
    without = pred.propagate(R)
    assert without.kind == "unbounded" and "response_uncertainty" in without.note
    exact = pred.propagate(R, response_uncertainty=0.0)
    assert exact.kind == "estimate" and exact.value.shape == (5,)
    assert_allclose(np.asarray(exact.value), np.abs(R) @ envelope, rtol=1e-13)
    with_dr = pred.propagate(R, response_uncertainty=dR)
    assert_allclose(
        np.asarray(with_dr.value),
        np.abs(R) @ envelope + dR @ np.abs(stokes),
        rtol=1e-13,
    )
    with pytest.raises(ValueError):
        pred.propagate(R[:, :11])
    with pytest.raises(ValueError):
        pred.propagate(R, response_uncertainty=dR[:, :11])
    open_budget = predict(basis, moments, amplitude=1.5)
    assert open_budget.propagate(R, response_uncertainty=0.0).kind == "unbounded"


# -- JIT and autodiff ----------------------------------------------------------------


def test_predict_is_jit_and_grad_safe(poly_basis):
    _, _, _, basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    eager = predict(basis, moments, amplitude=2.0, **full_inputs(basis))
    compiled = eqx.filter_jit(
        lambda m, a: predict(basis, m, amplitude=a, **full_inputs(basis))
    )(moments, 2.0)
    assert_allclose(np.asarray(compiled.stokes), np.asarray(eager.stokes), rtol=1e-14)
    assert compiled.budget.total().kind == "unbounded"

    def total_I(m0, amplitude):
        m = eqx.tree_at(lambda x: x.m0, moments, m0)
        return predict(basis, m, amplitude=amplitude).stokes[:, 0].sum()

    g_m0, g_a = jax.grad(total_I, argnums=(0, 1))(moments.m0, 2.0)
    C = np.asarray(basis.response_matrix())
    assert_allclose(
        np.asarray(g_m0),
        2.0 * C.reshape(3, 4, -1)[:, 0, : basis.index.n0].sum(0),
        rtol=1e-12,
    )
    assert_allclose(float(g_a), float(eager.stokes[:, 0].sum()) / 2.0, rtol=1e-12)
