"""Round-3 regressions of the direct average and the assumption allowances.

NEW2-2: ``direct_channel_average`` names the ``isotropic_pitch`` assumption a
``uniform_mu`` kernel (``ContinuumKernel``) forces, as ``build_basis`` does for
``predict``: an ``unbounded`` forced term that accepts an allowance and enters
the amplitude cross term. NEW2-3: an allowance of kind ``not_applicable`` is
refused (a forced assumption does arise on its route). NEW2-4: with traced
samples the declared-zero excluded tail states that the Support check did not
run, on both routes.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401
from syncmoments.model._allowances import check_allowances
from syncmoments.model.basis import build_basis
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.channels import Channels
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import (
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)
from syncmoments.model.phase import EmpiricalScreen
from syncmoments.model.predict import direct_channel_average, predict

from _boundary_oracles import constant_kernel

N_CH = 2


def _continuum_population(size=50):
    """Strongly anisotropic pitch (every ``mu = 0.95``) inside the Support."""
    rng = np.random.default_rng(0)
    return PopulationSamples(
        rng.uniform(2700, 3300, size), rng.uniform(4.5e-6, 5.5e-6, size),
        np.full(size, 0.95), np.full(size, 0.5), np.zeros(size), np.zeros(size),
    )  # fmt: skip


@pytest.fixture(scope="module")
def continuum():
    channels = Channels.bump(np.array([1e9, 2e9]), np.array([3e8, 6e8]))
    ref = Reference(3000.0, 5e-6, depth_ref=0.0, scales=(300.0, 5e-7, 1.0))
    supp = Support(
        gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(-1.0, 1.0), truncated=False
    )
    kernel = ContinuumKernel(
        E_phys=ErrorTerm(jnp.zeros((N_CH, 4)), "bound", "declared E_phys")
    )
    return kernel, channels, ref, supp, _continuum_population()


def _kinds(prediction):
    return [(name, term.kind) for name, term in prediction.budget.assumption]


# -- NEW2-2: the direct route names isotropic_pitch ----------------------------------


@pytest.mark.parametrize("screen", [False, True])
def test_new2_2_direct_forced_assumptions_match_predict(continuum, screen):
    kernel, channels, ref, supp, samples = continuum
    phase = EmpiricalScreen(jnp.asarray([-0.5, 0.5]), jnp.asarray([0.5, 0.5]))
    phase = phase if screen else None
    depth_degree = 0 if screen else None
    basis = build_basis(
        kernel, channels, Truncation(0, 2, 2, depth_degree=depth_degree), ref,
        support=supp, phase=phase,
    )  # fmt: skip
    moments = JointMoments.from_samples(samples, basis.index, ref)
    pred = predict(basis, moments, amplitude=1.0)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=ref, support=supp,
        phase=phase,
    )  # fmt: skip
    assert ("isotropic_pitch", "unbounded") in _kinds(pred)
    assert _kinds(direct) == _kinds(pred)
    names = [r.name for r in direct.provenance.assumptions]
    assert names == [r.name for r in basis.provenance.assumptions]
    assert "assumption:isotropic_pitch" in direct.budget.unbounded()
    term = dict(direct.budget.assumption)["isotropic_pitch"]
    assert "forced" in term.note and "ContinuumKernel" in term.note


def test_new2_2_direct_accepts_an_isotropic_pitch_allowance(continuum):
    kernel, channels, ref, supp, samples = continuum
    common = dict(amplitude=2.0, reference=ref, support=supp, amplitude_uncertainty=0.1)
    open_pred = direct_channel_average(samples, kernel, channels, **common)
    assert "assumption:isotropic_pitch" in open_pred.budget.amplitude.note
    declared = ErrorTerm(jnp.full((N_CH, 4), 1e-3), "estimate", "declared pitch")
    pred = direct_channel_average(
        samples, kernel, channels, **common,
        assumption_allowances={"isotropic_pitch": declared},
    )  # fmt: skip
    term = dict(pred.budget.assumption)["isotropic_pitch"]
    assert term.kind == "estimate" and "allowance supplied by caller" in term.note
    assert "assumption:isotropic_pitch" not in pred.budget.unbounded()
    assert "assumption:isotropic_pitch" not in pred.budget.amplitude.note
    # The allowance (Stokes units) enters the cross term divided by the amplitude.
    diff = np.asarray(pred.budget.amplitude.value) - np.asarray(
        open_pred.budget.amplitude.value
    )
    np.testing.assert_allclose(diff, 0.1 * 1e-3 / 2.0, rtol=1e-9)
    assert any("isotropic_pitch" in n for n in pred.provenance.notes)


# -- NEW2-3: not_applicable allowances are refused ------------------------------------


def test_new2_3_check_allowances_refuses_not_applicable():
    with pytest.raises(ValueError, match="not_applicable"):
        check_allowances(
            {"isotropic_pitch": ErrorTerm.not_applicable("caller says n/a")},
            ["isotropic_pitch"],
            N_CH,
        )


def test_new2_3_unbounded_and_valued_allowances_stay_accepted():
    for term in (
        ErrorTerm.unbounded("caller: no bound", "E_phys"),
        ErrorTerm(jnp.ones((N_CH, 4)), "measured", "m"),
    ):
        out = check_allowances({"a": term}, ["a"], N_CH)["a"]
        assert out.kind == term.kind and "allowance supplied by caller" in out.note


def test_new2_3_routes_refuse_not_applicable(continuum):
    kernel, channels, ref, supp, samples = continuum
    na = {"isotropic_pitch": ErrorTerm.not_applicable("caller says n/a")}
    basis = build_basis(kernel, channels, Truncation(0, 2, 2), ref, support=supp)
    moments = JointMoments.from_samples(samples, basis.index, ref)
    with pytest.raises(ValueError, match="not_applicable"):
        predict(basis, moments, amplitude=1.0, assumption_allowances=na)
    with pytest.raises(ValueError, match="not_applicable"):
        direct_channel_average(
            samples, kernel, channels, amplitude=1.0, reference=ref, support=supp,
            assumption_allowances=na,
        )  # fmt: skip


# -- NEW2-4: traced samples leave the declared-zero tail unchecked --------------------


def _two_ray():
    kernel, line = constant_kernel(1)
    channels = Channels.bump(line, 0.1 * line)
    ones = np.ones(2)
    samples = PopulationSamples(
        5 * ones, 2 * ones, 0.3 * ones, -0.2 * ones, np.zeros(2), np.zeros(2)
    )
    ref = Reference(5.0, 2.0, 0.0, scales=(1.0, 1.0, 1.0))
    supp = Support(gamma=(2, 9), B=(0.5, 4), depth=(-1, 1), truncated=False)
    return kernel, channels, samples, ref, supp


def _errors(basis):
    n_lk = basis.index.n_lk
    return RemainderInputs(
        rho_ang=np.zeros((1, 3)), H=np.zeros((1, 3, n_lk)),
        absolute_moments=np.zeros((2, n_lk)),
    )  # fmt: skip


def _assert_unchecked(term):
    assert term.kind == "bound", (term.kind, term.note)
    assert float(np.max(np.asarray(term.value))) == 0.0
    assert "declared zero" in term.note
    assert "not checked" in term.note and "traced" in term.note


def test_new2_4_direct_traced_samples_state_the_unchecked_tail():
    kernel, channels, samples, ref, supp = _two_ray()

    @eqx.filter_jit
    def run(s):
        return direct_channel_average(
            s, kernel, channels, amplitude=1.0, reference=ref, support=supp
        ).budget.excluded_tail

    _assert_unchecked(run(samples))
    eager = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=ref, support=supp
    ).budget.excluded_tail
    assert eager.kind == "bound" and "not checked" not in eager.note


def test_new2_4_predict_traced_samples_state_the_unchecked_tail():
    kernel, channels, samples, ref, supp = _two_ray()
    basis = build_basis(kernel, channels, Truncation(0, 0, 0), ref, support=supp)
    moments = JointMoments.from_samples(samples, basis.index, ref)
    errors = _errors(basis)

    @eqx.filter_jit
    def run(s):
        return predict(
            basis, moments, amplitude=1.0, errors=errors, samples=s, kernel=kernel
        ).budget.excluded_tail

    _assert_unchecked(run(samples))
    eager = predict(
        basis, moments, amplitude=1.0, errors=errors, samples=samples, kernel=kernel
    ).budget.excluded_tail
    assert eager.kind == "bound" and "not checked" not in eager.note
    no_samples = predict(basis, moments, amplitude=1.0).budget.excluded_tail
    assert no_samples.kind == "bound" and "traced" not in no_samples.note
