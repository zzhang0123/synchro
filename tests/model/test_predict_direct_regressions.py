"""Round-2 regressions of ``predict`` and ``direct_channel_average``.

NEW-2: ``predict(samples=...)`` must run the Support check of
``kernel.truncation_error`` and of the declared-complete excluded tail, as the
direct average does. NEW-3: an unvalidated derivative-probe order leaves
``basis_remainder`` unbounded instead of aborting ``predict``. NEW-5: a
caller-supplied allowance (``assumption_allowances``) replaces the unbounded
term of a named assumption. NEW-6: the direct average's amplitude slot carries
the ``delta A`` times per-electron-error cross term. NEW-8: an exact zero
``amplitude_uncertainty`` is a zero ``bound``; a negative one raises. NEW-11:
the sample probe of ``predict`` uses the basis phase route.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model import assumptions as A
from synchro.model.basis import build_basis
from synchro.model.bounds import RemainderInputs, screen_factorisation_bound
from synchro.model.channels import Channels
from synchro.model.errors import ErrorTerm
from synchro.model.index import Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support
from synchro.model.phase import CumulantScreen, EmpiricalScreen, GaussianScreen
from synchro.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis
from _boundary_oracles import constant_kernel, tau_of
from _predict_helpers import (
    DEPTH_REF,
    LINE_NU,
    S_DEPTH,
    full_inputs,
    harmonic_setup,
    nine_atoms,
    polynomial_kernel,
    reference,
    samples_of,
    support,
)

# -- NEW-2: Support check on the predict(samples) path ------------------------------


@pytest.fixture(scope="module")
def harmonic_stub():
    kernel, channels, supp, ref, pop = harmonic_setup()
    basis = build_stub_basis(kernel, channels, Truncation(1, 1, 1), ref, supp)
    return kernel, channels, supp, ref, pop, basis


def test_new2_predict_matches_direct_slot_kinds_outside_support(harmonic_stub):
    kernel, channels, supp, ref, pop, basis = harmonic_stub
    out = dict(pop)
    out["gamma"] = pop["gamma"] * 3.0  # every sample above supp.gamma[1] = 6
    samples = samples_of(out)
    moments = JointMoments.from_samples(samples, basis.index, ref)
    pred = predict(basis, moments, amplitude=1.0, samples=samples, kernel=kernel)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=ref, support=supp
    )
    for name in ("harmonic_truncation", "excluded_tail"):
        got, want = getattr(pred.budget, name), getattr(direct.budget, name)
        assert got.kind == want.kind == "unbounded", (name, got.kind, got.note)
        assert "gamma in [" in got.note
    assert pred.budget.total().kind == "unbounded"


def test_new2_predict_inside_support_keeps_both_zero_bounds(harmonic_stub):
    kernel, channels, supp, ref, pop, basis = harmonic_stub
    samples = samples_of(pop)
    moments = JointMoments.from_samples(samples, basis.index, ref)
    pred = predict(basis, moments, amplitude=1.0, samples=samples, kernel=kernel)
    trunc, tail = pred.budget.harmonic_truncation, pred.budget.excluded_tail
    assert trunc.kind == "bound" and "checked inside" in trunc.note
    assert tail.kind == "bound" and float(np.max(np.asarray(tail.value))) == 0.0


# -- NEW-3: unvalidated probe order at N = 3 -----------------------------------------


def test_new3_predict_samples_at_n3_leaves_basis_remainder_unbounded():
    kernel, _ = polynomial_kernel(N=3)
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    basis = build_basis(
        kernel, channels, Truncation(2, 2, 3), reference(), support=support(),
        convergence=False,
    )  # fmt: skip
    assert 4 not in basis.certified_orders
    samples = samples_of(nine_atoms())
    moments = JointMoments.from_samples(samples, basis.index, reference())
    pred = predict(basis, moments, amplitude=1.0, samples=samples, kernel=kernel)
    term = pred.budget.basis_remainder
    assert term.kind == "unbounded" and term.value is None
    assert "N+1=4" in term.note and "errors=RemainderInputs" in term.note
    plain = predict(basis, moments, amplitude=1.0)
    assert_allclose(np.asarray(pred.stokes), np.asarray(plain.stokes), rtol=0)
    assert pred.budget.harmonic_truncation.kind == "not_applicable"


# -- NEW-5: caller-supplied assumption allowances ------------------------------------


def _two_ray_screen_case():
    """R00 population: two rays whose Faraday phases cancel the screen average."""
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
    return kernel, channels, tau, samples, basis, ref, screen


def _screen_bound(kernel, channels, tau, samples):
    """``screen_factorisation_bound`` from the per-ray incident polarisation."""
    incident = []
    for n in range(samples.size):
        one = PopulationSamples(
            *(np.asarray(getattr(samples, v))[n : n + 1] for v in "gamma B mu eta phi".split()),
            np.zeros(1),
        )  # fmt: skip
        s = np.asarray(
            direct_channel_average(one, kernel, channels, amplitude=1.0).stokes
        )
        incident.append(complex(s[0, 1], s[0, 2]))
    incident = np.asarray(incident)
    w = np.asarray(samples.normalised_weights())
    sigma = np.sqrt(w @ np.abs(incident - w @ incident) ** 2)
    phi_abs = np.abs(w @ np.exp(1j * tau * np.asarray(samples.depth)))
    return screen_factorisation_bound(jnp.asarray([sigma]), jnp.asarray([phi_abs]))


def test_new5_screen_allowance_closes_the_budget_and_bounds_the_error():
    kernel, channels, tau, samples, basis, ref, _ = _two_ray_screen_case()
    pm = A.independent_screen(basis.index)
    moments = JointMoments.from_samples(
        samples, basis.index, ref, parameter_map=pm, discrepancy="measured"
    )
    inputs = full_inputs(basis, n_ch=1)
    n_lk = basis.index.n_lk
    n_a = len([r for r in basis.index.h2 if r[4] == 0])
    inputs["errors"] = RemainderInputs(
        rho_ang=np.zeros((1, 3)), H=np.zeros((1, 3, n_lk)),
        absolute_moments=np.zeros((2, n_lk)), depth_tail=np.zeros(n_a),
        intrinsic_residual=np.zeros(1),
    )  # fmt: skip
    open_pred = predict(basis, moments, amplitude=1.0, **inputs)
    assert open_pred.budget.unbounded() == ("assumption:independent_screen",)
    allowance = _screen_bound(kernel, channels, tau, samples)
    pred = predict(
        basis, moments, amplitude=1.0,
        assumption_allowances={"independent_screen": allowance}, **inputs,
    )  # fmt: skip
    term = dict(pred.budget.assumption)["independent_screen"]
    assert term.kind == "bound" and "allowance supplied by caller" in term.note
    assert_allclose(np.asarray(term.value), np.asarray(allowance.value), rtol=0)
    total = pred.budget.total()
    assert pred.budget.unbounded() == () and total.kind == "bound"
    truth = np.asarray(
        direct_channel_average(samples, kernel, channels, amplitude=1.0).stokes
    )
    err = np.abs(truth - np.asarray(pred.stokes))
    assert np.max(err[:, 1:3]) > 0.5  # a real screen error
    # Two rays with |Phi_R| = 0 attain the Cauchy-Schwarz bound: equality up to roundoff.
    assert np.all(err <= np.asarray(total.value) * (1 + 1e-12) + 1e-15)
    assert any("allowance supplied by caller" in n for n in pred.provenance.notes)


def test_new5_direct_average_accepts_the_same_allowance():
    kernel, channels, tau, samples, _, _, screen = _two_ray_screen_case()
    allowance = _screen_bound(kernel, channels, tau, samples)
    pred = direct_channel_average(
        samples, kernel, channels, amplitude=2.0, phase=screen,
        assumption_allowances={"independent_screen": allowance.scaled(2.0)},
    )  # fmt: skip
    term = dict(pred.budget.assumption)["independent_screen"]
    assert term.kind == "bound" and "allowance supplied by caller" in term.note
    assert "assumption:independent_screen" not in pred.budget.unbounded()
    assert pred.budget.total().kind == "unbounded"  # numerical stays unbounded
    with pytest.raises(ValueError, match="unknown assumption"):
        direct_channel_average(
            samples, kernel, channels, amplitude=1.0, phase=screen,
            assumption_allowances={"isotropic_pitch": allowance},
        )  # fmt: skip


@pytest.fixture(scope="module")
def continuum_case():
    channels = Channels.bump(np.array([1e9, 2e9]), np.array([3e8, 6e8]))
    ref = Reference(3000.0, 5e-6, depth_ref=0.0, scales=(300.0, 5e-7, 1.0))
    supp = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(-1.0, 1.0))
    basis = build_basis(
        ContinuumKernel(), channels, Truncation(0, 2, 2), ref, support=supp
    )
    rng = np.random.default_rng(0)
    size = 50
    samples = PopulationSamples(
        rng.uniform(2700, 3300, size), rng.uniform(4.5e-6, 5.5e-6, size),
        rng.uniform(-1, 1, size), np.full(size, 0.5), np.zeros(size), np.zeros(size),
    )  # fmt: skip
    pm = A.isotropic_pitch(basis.index)
    moments = JointMoments.from_samples(
        samples, basis.index, ref, parameter_map=pm, discrepancy="measured"
    )
    return basis, moments


def test_new5_isotropic_pitch_allowance_replaces_the_forced_term(continuum_case):
    basis, moments = continuum_case
    declared = ErrorTerm(jnp.full((2, 4), 1e-3), "estimate", "declared pitch allowance")
    pred = predict(
        basis, moments, amplitude=2.0, amplitude_uncertainty=0.1,
        statistical_input=np.zeros(basis.index.n_real),
        assumption_allowances={"isotropic_pitch": declared},
    )  # fmt: skip
    term = dict(pred.budget.assumption)["isotropic_pitch"]
    assert term.kind == "estimate" and "allowance supplied by caller" in term.note
    assert "declared pitch allowance" in term.note
    assert_allclose(np.asarray(term.value), 1e-3, rtol=0)
    assert "assumption:isotropic_pitch" not in pred.budget.unbounded()
    assert "assumption:isotropic_pitch" not in pred.budget.amplitude.note
    assert pred.budget.total().kind == "unbounded"  # basis_remainder still open


@pytest.mark.parametrize(
    "allowances, match",
    [
        ({"independent_screen": ErrorTerm(jnp.ones((2, 4)), "bound")}, "unknown"),
        ({"isotropic_pitch": np.ones((2, 4))}, "ErrorTerm"),
        ({"isotropic_pitch": ErrorTerm(jnp.ones((3, 4)), "bound")}, "broadcast"),
    ],
)
def test_new5_invalid_allowances_raise(continuum_case, allowances, match):
    basis, moments = continuum_case
    with pytest.raises(ValueError, match=match):
        predict(basis, moments, amplitude=1.0, assumption_allowances=allowances)


# -- NEW-6: direct amplitude cross term -----------------------------------------------


def test_new6_direct_amplitude_slot_has_the_cross_term():
    kernel, channels, supp, ref, pop = harmonic_setup()
    screen = CumulantScreen(
        jnp.asarray([ref.depth_ref, 0.0, 0.0, 0.0]), g5_bound=jnp.asarray(1e-33)
    )
    kernel = eqx.tree_at(
        lambda k: k.E_phys, kernel, ErrorTerm(jnp.full((2, 4), 1e-3), "bound"),
        is_leaf=lambda x: x is None,
    )  # fmt: skip
    a, d_a = 2.0, 0.5
    pred = direct_channel_average(
        samples_of(pop), kernel, channels, amplitude=a, reference=ref,
        support=supp, phase=screen, amplitude_uncertainty=d_a,
        depth_model=np.full((2, 4), 0.2),
    )  # fmt: skip
    b = pred.budget
    assert b.screen_exponent.kind == "estimate"
    per_electron = np.abs(np.asarray(pred.stokes)) / a
    valued = [b.physical_kernel, b.harmonic_truncation, b.statistical_input]
    valued += [b.screen_exponent, b.depth_model]
    e = sum(np.asarray(t.value) for t in valued) / a
    assert float(np.max(np.asarray(b.physical_kernel.value))) > 0
    assert_allclose(np.asarray(b.amplitude.value), d_a * (per_electron + e), rtol=1e-12)
    assert b.amplitude.kind == "estimate"  # weakest of bound and the screen estimate
    assert "numerical" in b.amplitude.note


# -- NEW-8: exact zero and negative amplitude uncertainty ---------------------------


@pytest.fixture(scope="module")
def poly_case():
    kernel, _ = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(2, 2, 2), reference(), support(False)
    )
    moments = JointMoments.from_samples(
        samples_of(nine_atoms()), basis.index, reference()
    )
    return basis, moments


def test_new8_exact_zero_amplitude_uncertainty_is_a_zero_bound(poly_case):
    basis, moments = poly_case
    inputs = full_inputs(basis)
    inputs["statistical_input"] = ErrorTerm(
        jnp.full(basis.index.n_real, 1e-3), "estimate", "bootstrap"
    )
    for zero in (0.0, 0, np.float64(0.0), jnp.asarray(0.0)):
        inputs["amplitude_uncertainty"] = zero
        term = predict(basis, moments, amplitude=2.0, **inputs).budget.amplitude
        assert term.kind == "bound" and "declared exact" in term.note
        assert float(np.max(np.asarray(term.value))) == 0.0
    inputs["amplitude_uncertainty"] = 0.1
    assert predict(basis, moments, amplitude=2.0, **inputs).budget.amplitude.kind == (
        "estimate"
    )


@pytest.mark.parametrize("negative", [-0.1, -1e-300, jnp.asarray(-2.0)])
def test_new8_negative_amplitude_uncertainty_raises(poly_case, negative):
    basis, moments = poly_case
    with pytest.raises(Exception, match="amplitude_uncertainty"):
        predict(basis, moments, amplitude=1.0, amplitude_uncertainty=negative)


def test_new8_traced_amplitude_uncertainty_keeps_the_weakest_kind(poly_case):
    basis, moments = poly_case
    stat = ErrorTerm(jnp.full(basis.index.n_real, 1e-3), "estimate", "bootstrap")

    @eqx.filter_jit
    def run(d_a):
        return predict(
            basis, moments, amplitude=1.0, amplitude_uncertainty=d_a,
            statistical_input=stat,
        ).budget.amplitude  # fmt: skip

    term = run(jnp.asarray(0.0))
    assert term.kind == "estimate" and float(np.max(np.asarray(term.value))) == 0.0
    with pytest.raises(Exception, match="amplitude_uncertainty"):
        run(jnp.asarray(-1.0)).value.block_until_ready()


# -- NEW-11: the probe uses the basis phase route -------------------------------------


def test_new11_predict_probe_uses_the_basis_phase(monkeypatch):
    kernel, _ = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    screen = GaussianScreen(mean=DEPTH_REF, sigma=0.3 * S_DEPTH)
    basis = build_basis(
        kernel, channels, Truncation(2, 2, 2, depth_degree=0), reference(),
        support=support(False), phase=screen, convergence=False,
    )  # fmt: skip
    pop = nine_atoms(constant_depth=False)
    pop["depth"] = pop["depth"] + 2.0 * S_DEPTH
    samples = samples_of(pop)
    moments = JointMoments.from_samples(samples, basis.index, reference())
    seen = {}
    original = RemainderInputs.from_samples

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(RemainderInputs, "from_samples", spy)
    via_predict = predict(basis, moments, amplitude=1.0, samples=samples, kernel=kernel)
    assert seen.get("phase") is basis.phase
    monkeypatch.undo()
    explicit = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, phase=screen, angular_residual="probe"
    )
    want = predict(basis, moments, amplitude=1.0, errors=explicit).budget
    got = via_predict.budget.basis_remainder
    assert got.kind == want.basis_remainder.kind
    assert_allclose(
        np.asarray(got.value), np.asarray(want.basis_remainder.value), rtol=1e-12
    )
