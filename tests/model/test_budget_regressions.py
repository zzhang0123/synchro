"""Regressions of the ``predict`` budget slot fillers (``_budget_terms``, ``predict``).

Each test reproduces a confirmed review finding: a basis-forced assumption
replaced by a structurally zero moment-level discrepancy (R00, R11), a second
assumption record handed a zero allowance (R01), a valued basis remainder on
a nonsmooth line-kernel basis (R04), the missing ``delta A`` times
per-electron-error cross term (R05), an unattributed discrepancy dropped next
to an explicit ``statistical_input`` (R13), and the per-column spectral-basis
envelope ``(4 n_ch, n_real)`` of the ``numerical`` slot (R20).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model import assumptions as A
from synchro.model.basis import build_basis
from synchro.model.bounds import RemainderInputs
from synchro.model.channels import Channels
from synchro.model.errors import ErrorTerm
from synchro.model.index import Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support
from synchro.model.phase import EmpiricalScreen
from synchro.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis
from _boundary_oracles import constant_kernel, tau_of
from _predict_helpers import (
    H_NU_STAR,
    LINE_NU,
    full_inputs,
    harmonic_setup,
    nine_atoms,
    polynomial_kernel,
    reference,
    samples_of,
    support,
)


@pytest.fixture(scope="module")
def poly_basis():
    kernel, _ = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(2, 2, 2), reference(), support(False)
    )
    exact = ErrorTerm.declared_zero("polynomial kernel is exact", shape=(3, 4))
    return eqx.tree_at(lambda b: b.kernel_terms.numerical, basis, exact)


def _exact_inputs(basis):
    inputs = full_inputs(basis)
    inputs.pop("amplitude_uncertainty")
    return inputs


# -- R00: screen route, independent_screen measured on b = 0 rows only -------------


def test_screen_route_forced_assumption_stays_unbounded_against_measured_zero():
    kernel, line = constant_kernel(1)
    channels = Channels.bump(line, 0.1 * line)
    tau = float(tau_of(line[0]))
    depth, phi = np.array([0.0, np.pi / tau]), np.array([0.0, np.pi / 2])
    ones = np.ones(2)
    samples = PopulationSamples(5 * ones, 2 * ones, 0.3 * ones, -0.2 * ones, phi, depth)
    ref = Reference(5.0, 2.0, 0.0, scales=(1.0, 1.0, 1.0))
    supp = Support(gamma=(2, 9), B=(0.5, 4), depth=(-1, 1), truncated=False)
    screen = EmpiricalScreen(jnp.asarray(depth), jnp.asarray([0.5, 0.5]))
    basis = build_basis(
        kernel, channels, Truncation(0, 0, 0, depth_degree=0), ref, support=supp,
        phase=screen,
    )  # fmt: skip
    pm = A.independent_screen(basis.index)
    moments = JointMoments.from_samples(
        samples, basis.index, ref, parameter_map=pm, discrepancy="measured"
    )
    assert np.all(np.asarray(moments.discrepancy.value) == 0.0)  # structural
    n_lk = basis.index.n_lk
    n_a = len([r for r in basis.index.h2 if r[4] == 0])
    errors = RemainderInputs(
        rho_ang=np.zeros((1, 3)), H=np.zeros((1, 3, n_lk)),
        absolute_moments=np.zeros((2, n_lk)), depth_tail=np.zeros(n_a),
        intrinsic_residual=np.zeros(1),
    )  # fmt: skip
    pred = predict(
        basis, moments, amplitude=1.0, errors=errors, amplitude_uncertainty=0.0,
        statistical_input=jnp.zeros(basis.index.n_real), depth_model=np.zeros((1, 4)),
    )  # fmt: skip
    truth = direct_channel_average(samples, kernel, channels, amplitude=1.0)
    s, t = np.asarray(pred.stokes)[0], np.asarray(truth.stokes)[0]
    assert abs(complex(s[1], s[2]) - complex(t[1], t[2])) > 0.5  # a real error
    term = dict(pred.budget.assumption)["independent_screen"]
    assert term.kind == "unbounded" and "forced" in term.note
    assert "assumption:independent_screen" in pred.budget.unbounded()
    assert pred.budget.total().kind == "unbounded"
    assert any("not propagated" in n for n in pred.provenance.notes)


# -- R11: ContinuumKernel forces isotropic pitch; L_mu = 0 moments cannot test it ---


def test_continuum_isotropic_pitch_stays_unbounded_against_measured_zero():
    channels = Channels.bump(np.array([1e9, 2e9]), np.array([3e8, 6e8]))
    ref = Reference(3000.0, 5e-6, depth_ref=0.0, scales=(300.0, 5e-7, 1.0))
    supp = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(-1.0, 1.0))
    basis = build_basis(
        ContinuumKernel(), channels, Truncation(0, 2, 2), ref, support=supp
    )
    rng = np.random.default_rng(0)
    size = 400
    mu = np.sign(rng.uniform(-1, 1, size)) * rng.uniform(0.99, 1.0, size)
    samples = PopulationSamples(
        rng.uniform(2700, 3300, size), rng.uniform(4.5e-6, 5.5e-6, size), mu,
        np.full(size, 0.5), np.zeros(size), np.zeros(size),
    )  # fmt: skip
    pm = A.isotropic_pitch(basis.index)
    moments = JointMoments.from_samples(
        samples, basis.index, ref, parameter_map=pm, discrepancy="measured"
    )
    pred = predict(
        basis, moments, amplitude=1.0, statistical_input=np.zeros(basis.index.n_real)
    )
    term = dict(pred.budget.assumption)["isotropic_pitch"]
    assert term.kind == "unbounded" and term.value is None
    assert "assumption:isotropic_pitch" in pred.budget.unbounded()
    assert [name for name, _ in pred.budget.assumption] == ["isotropic_pitch"]


# -- R01: every record gets its own term; no zero copies ---------------------------


def test_second_assumption_record_is_unbounded_not_a_zero_copy(poly_basis):
    basis = poly_basis
    index = basis.index
    base = JointMoments.from_samples(samples_of(nine_atoms()), index, reference())
    records = (A.field_independent(index).record(), A.azimuth_separable(index).record())
    delta = np.full(index.n_real, 1e-3)
    moments = JointMoments(
        index=index, m0=base.m0, m2=base.m2, reference=base.reference,
        assumptions=records, discrepancy=ErrorTerm(jnp.asarray(delta), "bound", "declared"),
    )  # fmt: skip
    pred = predict(basis, moments, amplitude=2.0, **_exact_inputs(basis))
    terms = dict(pred.budget.assumption)
    first, second = terms[records[0].name], terms[records[1].name]
    C = np.abs(np.asarray(basis.response_matrix()))
    assert first.kind == "bound"
    assert_allclose(
        np.asarray(first.value), 2.0 * (C @ delta).reshape(3, 4), rtol=1e-13
    )
    assert second.kind == "unbounded" and second.value is None
    assert f"assumption:{records[1].name}" in pred.budget.unbounded()


# -- R04: nonsmooth line-kernel basis ----------------------------------------------


def test_nonsmooth_line_kernel_basis_remainder_is_unbounded():
    kernel, _, supp, ref, pop = harmonic_setup()
    centres = np.array([2.0, 4.0]) * H_NU_STAR
    channels = Channels.tophat(centres, 0.5 * centres)
    basis = build_basis(
        kernel, channels, Truncation(1, 1, 1), ref, support=supp,
        allow_nonsmooth=True, convergence=False,
    )  # fmt: skip
    moments = JointMoments.from_samples(samples_of(pop), basis.index, ref)
    n_lk = basis.index.n_lk
    supplied = RemainderInputs(
        rho_ang=np.zeros((2, 3)), H=np.zeros((2, 3, n_lk)),
        absolute_moments=np.zeros((2, n_lk)), kind="bound",
    )  # fmt: skip
    term = predict(
        basis, moments, amplitude=1.0, errors=supplied
    ).budget.basis_remainder
    assert term.kind == "unbounded" and term.value is None
    assert "smoothness" in term.note


# -- R05: amplitude cross term ------------------------------------------------------


def test_amplitude_term_covers_delta_a_times_per_electron_error(poly_basis):
    basis = poly_basis
    index = basis.index
    m_true = np.asarray(
        JointMoments.from_samples(
            samples_of(nine_atoms()), index, reference()
        ).to_vector()
    )
    C = np.asarray(basis.response_matrix())
    delta = np.full(index.n_real, 0.01)
    delta[0] = 0.0
    sign = np.sign(C[0]) * np.sign(C[0] @ m_true)  # adversarial for channel 0, I
    m_hat = JointMoments.from_vector(index, m_true - sign * delta, reference())
    a_hat, d_a = 1.0, 0.5
    inputs = _exact_inputs(basis)
    inputs["statistical_input"] = delta
    pred = predict(basis, m_hat, amplitude=a_hat, amplitude_uncertainty=d_a, **inputs)
    total = pred.budget.total()
    assert total.kind == "bound"
    true = (a_hat + d_a) * (C @ m_true).reshape(3, 4)
    err = np.abs(true - np.asarray(pred.stokes))
    assert np.all(err <= np.asarray(total.value) * (1 + 1e-12))
    expected = d_a * (np.abs(C @ np.asarray(m_hat.to_vector())) + np.abs(C) @ delta)
    assert_allclose(
        np.asarray(pred.budget.amplitude.value), expected.reshape(3, 4), rtol=1e-12
    )


def test_amplitude_term_rescales_supplied_depth_model(poly_basis):
    basis = poly_basis
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    inputs = _exact_inputs(basis)
    inputs["depth_model"] = np.full((3, 4), 0.2)
    pred = predict(basis, moments, amplitude=2.0, amplitude_uncertainty=0.5, **inputs)
    per_electron = np.abs(np.asarray(pred.stokes)) / 2.0
    assert_allclose(
        np.asarray(pred.budget.amplitude.value),
        0.5 * (per_electron + 0.2 / 2.0),
        rtol=1e-12,
    )
    with pytest.raises(Exception):
        predict(basis, moments, amplitude=0.0, amplitude_uncertainty=0.5, **inputs)


# -- R13: unattributed discrepancy with explicit statistical input ------------------


def _with_discrepancy(basis, term):
    base = JointMoments.from_samples(samples_of(nine_atoms()), basis.index, reference())
    return JointMoments(
        index=base.index, m0=base.m0, m2=base.m2, reference=base.reference,
        m0_ext=base.m0_ext, discrepancy=term,
    )  # fmt: skip


def test_unattributed_discrepancy_adds_to_explicit_statistical_input(poly_basis):
    basis = poly_basis
    n_real = basis.index.n_real
    C = np.abs(np.asarray(basis.response_matrix()))
    disc = ErrorTerm(jnp.full(n_real, 0.05), "bound", "estimator error")
    moments = _with_discrepancy(basis, disc)
    stat = predict(
        basis, moments, amplitude=1.0, statistical_input=np.zeros(n_real)
    ).budget.statistical_input
    assert stat.kind == "bound"
    assert_allclose(np.asarray(stat.value), (C @ np.full(n_real, 0.05)).reshape(3, 4))
    explicit = ErrorTerm(jnp.full(n_real, 0.01), "estimate", "bootstrap")
    stat = predict(
        basis, moments, amplitude=1.0, statistical_input=explicit
    ).budget.statistical_input
    assert stat.kind == "estimate"
    assert_allclose(
        np.asarray(stat.value), (C @ np.full(n_real, 0.06)).reshape(3, 4), rtol=1e-13
    )
    open_disc = _with_discrepancy(basis, ErrorTerm.unbounded("unknown estimator"))
    stat = predict(
        basis, open_disc, amplitude=1.0, statistical_input=np.zeros(n_real)
    ).budget.statistical_input
    assert stat.kind == "unbounded"


# -- R20: per-column spectral-basis envelope ----------------------------------------


def test_per_column_numerical_envelope_is_contracted_with_abs_m(poly_basis):
    basis = poly_basis
    index = basis.index
    moments = JointMoments.from_samples(samples_of(nine_atoms()), index, reference())
    m = np.asarray(moments.to_vector())
    assert np.any(m < 0)
    rng = np.random.default_rng(4)
    envelope = rng.uniform(0, 1e-3, (12, index.n_real))
    closed = eqx.tree_at(
        lambda b: b.kernel_terms.numerical,
        basis,
        ErrorTerm(jnp.asarray(envelope), "estimate", "per-column dC"),
    )
    pred = predict(
        closed,
        moments,
        amplitude=3.0,
        amplitude_uncertainty=0.5,
        **_exact_inputs(basis),
    )
    term = pred.budget.numerical
    expected = 3.0 * (envelope @ np.abs(m)).reshape(3, 4)
    assert term.kind == "estimate" and term.value.shape == (3, 4)
    assert_allclose(np.asarray(term.value), expected, rtol=1e-13)
    # every perturbation |dC| <= envelope is covered by the contracted term
    dC = envelope * rng.choice([-1.0, 1.0], envelope.shape)
    assert np.all(np.abs(3.0 * (dC @ m)).reshape(3, 4) <= expected * (1 + 1e-12))
    # the amplitude cross term sees the per-electron (contracted) envelope
    per_electron = np.abs(np.asarray(pred.stokes)) / 3.0 + expected / 3.0
    assert_allclose(
        np.asarray(pred.budget.amplitude.value), 0.5 * per_electron, rtol=1e-12
    )
