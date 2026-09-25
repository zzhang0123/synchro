"""Assumption triple, part (2) (FINAL_DESIGN Section 11) on the harmonic kernel:
``|predict(m_fac) - predict(m_joint)| <= budget.assumption`` and
``|predict(m_fac) - direct| <= assumption + basis_remainder`` with the probe ``H``.

The basis is the ``_basis_stub`` stand-in at ``Truncation(1, 1, 1)`` with
``m_max = required_m_max <= 40``; the direct reference is the discrete
population sum of ``kernel.channel_modes``.
"""

import equinox as eqx
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.model.assumptions import field_independent, independent_screen
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.index import Truncation
from syncmoments.model.moments import JointMoments
from syncmoments.model.phase import TaylorPhase
from syncmoments.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis
from _predict_helpers import harmonic_setup, samples_of


@pytest.fixture(scope="module")
def harmonic_case():
    kernel, channels, supp, ref, pop = harmonic_setup()
    basis = build_stub_basis(kernel, channels, Truncation(1, 1, 1), ref, supp)
    samples = samples_of(pop)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=ref, support=supp
    )
    remainder = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    return kernel, channels, supp, ref, samples, basis, direct, remainder


@pytest.mark.parametrize("make_map", [independent_screen, field_independent])
def test_assumption_terms_bound_the_reduced_prediction(harmonic_case, make_map):
    kernel, channels, supp, ref, samples, basis, direct, remainder = harmonic_case
    index = basis.index
    pm = make_map(index)
    joint = JointMoments.from_samples(samples, index, ref)
    fac = JointMoments.from_samples(
        samples, index, ref, parameter_map=pm, discrepancy="measured"
    )
    pred_joint = predict(basis, joint, amplitude=1.0, errors=remainder)
    pred_fac = predict(basis, fac, amplitude=1.0, errors=remainder)
    scale = np.max(np.abs(np.asarray(direct.stokes)[:, 0]))
    # The correlated population makes the reduced prediction differ by > 1e-3 relative.
    diff = np.abs(np.asarray(pred_fac.stokes) - np.asarray(pred_joint.stokes))
    assert np.max(diff) > 1e-3 * scale
    names = dict(pred_fac.budget.assumption)
    assert pm.name in names and names[pm.name].kind == "measured"
    assert pred_joint.budget.assumption == ()
    C = np.asarray(basis.response_matrix())
    allowance = (np.abs(C) @ np.asarray(fac.discrepancy.value)).reshape(-1, 4)
    assert_allclose(np.asarray(names[pm.name].value), allowance, rtol=1e-13)
    assert np.all(diff <= allowance * (1 + 1e-10) + 1e-13 * scale)
    # Finite-vs-direct: assumption + basis remainder with the probe H.
    basis_term = pred_fac.budget.basis_remainder
    assert basis_term.kind == "estimate"
    bound = allowance + np.asarray(basis_term.value)
    diff_direct = np.abs(np.asarray(pred_fac.stokes) - np.asarray(direct.stokes))
    assert np.all(diff_direct <= bound * (1 + 1e-6) + 1e-9 * scale)
    # The joint finite prediction alone is covered by the basis remainder.
    diff_joint = np.abs(np.asarray(pred_joint.stokes) - np.asarray(direct.stokes))
    assert np.all(
        diff_joint <= np.asarray(basis_term.value) * (1 + 1e-6) + 1e-9 * scale
    )
    assert np.max(diff_joint) > 0
    # Name in to_dict; without discrepancy="measured" the term and total are unbounded.
    d = pred_fac.to_dict()
    assert pm.name in d["budget"]["assumption"]
    assert d["budget"]["assumption"][pm.name]["kind"] == "measured"
    declared = JointMoments.from_samples(samples, index, ref, parameter_map=pm)
    open_pred = predict(basis, declared, amplitude=1.0, errors=remainder)
    assert dict(open_pred.budget.assumption)[pm.name].kind == "unbounded"
    assert f"assumption:{pm.name}" in open_pred.budget.unbounded()
    assert open_pred.budget.total().kind == "unbounded"
    assert pm.name in {r.name for r in open_pred.provenance.assumptions}


def test_harmonic_direct_average_matches_channel_modes_sum(harmonic_case):
    kernel, channels, supp, ref, samples, basis, direct, _ = harmonic_case
    w = np.asarray(samples.normalised_weights())
    expected = np.zeros((2, 4))
    for n in range(samples.size):
        modes = kernel.channel_modes(
            channels,
            samples.gamma[n],
            samples.B[n],
            samples.mu[n],
            samples.eta[n],
            phase=TaylorPhase(0),
            depth_ref=samples.depth[n],
            s_depth=1.0,
        )
        P = np.exp(2j * float(samples.phi[n])) * np.asarray(modes.P[0])
        expected += w[n] * np.stack(
            [np.asarray(modes.I), P.real, P.imag, np.asarray(modes.V)], axis=1
        )
    assert_allclose(np.asarray(direct.stokes), expected, rtol=1e-12)
    assert direct.budget.harmonic_truncation.kind == "bound"
    assert direct.budget.physical_kernel.kind == "unbounded"
    assert direct.budget.excluded_tail.kind == "bound"
    compiled = eqx.filter_jit(
        lambda s: direct_channel_average(
            s, kernel, channels, amplitude=1.0, reference=ref, support=supp
        ).stokes
    )(samples)
    assert_allclose(np.asarray(compiled), expected, rtol=1e-12)
