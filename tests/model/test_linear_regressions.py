"""Regression tests for review findings on ``fit_linear`` and its result builders.

R02/R03: the ``delta R S_hat`` term of ``extra eq: data error propagation``
enters the bias bound (unbounded without ``response_uncertainty``).
R08: a rank-deficient design leaves the bias of ``u`` unbounded.
R09: the rank decision is invariant under a rescaling of the design columns.
R21: ``tol`` reaches the identifiability report and the provenance notes;
``inflate`` diagnostics use the inflated weights. R23: the discrepancy bias
of the moments enters ``statistical_input``. R28: the ``inflate`` note names
``chi2``. Oracles: NumPy pseudo-inverses of the whitened design.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _linear_fixtures import setup, stokes_of, truth_of
from _nonlinear_oracles import StubBasis
from synchro.model.assumptions import no_assumption
from synchro.model.errors import ErrorTerm
from synchro.model.fit.diagnostics import identifiability
from synchro.model.fit.linear import fit_linear
from synchro.model.fit.observation import StokesData


def _u_of(result, pm):
    A = float(result.amplitude)
    return np.concatenate([[A], A * np.asarray(pm.flatten(result.theta))])


# -- R02 / R03: delta R S_hat ---------------------------------------------------------


def _response_problem(seed=7, n_rows=50):
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 1)
    S, _ = stokes_of(C, pm, theta, A, basis.reference)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((n_rows, C.shape[0]))
    dR = 0.01 * np.abs(rng.standard_normal(R.shape))
    u_true = np.concatenate([[A], A * flat])
    return basis, pm, S, R, dR, u_true


def _response_fit(basis, pm, S, R, dR_true, dR_declared):
    E = ErrorTerm(jnp.full((16, 4), 1e-9), "bound", "declared Stokes discrepancy")
    data = StokesData(
        (R + dR_true) @ S,
        np.full(R.shape[0], 1e-4),
        response=R,
        discrepancy=E,
        response_uncertainty=dR_declared,
    )
    return fit_linear(basis, data, pm, diagnostics=False)


def test_missing_response_uncertainty_leaves_bias_unbounded():
    basis, pm, S, R, dR, _ = _response_problem()
    result = _response_fit(basis, pm, S, R, dR, None)
    assert result.bias_bound.kind == "unbounded"
    assert "response_uncertainty" in result.bias_bound.note


def test_declared_response_uncertainty_enters_the_bias_bound():
    basis, pm, S, R, dR, u_true = _response_problem()
    result = _response_fit(basis, pm, S, R, dR, dR)
    bound = np.asarray(result.bias_bound.value)
    bias = np.abs(_u_of(result, pm) - u_true)
    # plug-in S_hat: an estimate, not a strict bound
    assert result.bias_bound.kind == "estimate"
    assert np.all(bias <= bound)
    larger = _response_fit(basis, pm, S, R, dR, 100.0 * dR)
    assert np.all(np.asarray(larger.bias_bound.value) > 10.0 * bound)


def test_zero_response_uncertainty_keeps_the_discrepancy_kind():
    basis, pm, S, R, _, u_true = _response_problem()
    zero = np.zeros_like(R)
    result = _response_fit(basis, pm, S, R, zero, zero)
    assert result.bias_bound.kind == "bound"
    bias = np.abs(_u_of(result, pm) - u_true)
    assert np.all(bias <= np.asarray(result.bias_bound.value) * (1 + 1e-9) + 1e-12)


# -- R08: rank deficiency ---------------------------------------------------------------


def test_rank_deficient_bias_bound_is_unbounded():
    index, basis, C = setup((1, 1, 1), n_ch=3, seed=5)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 2)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    zero = ErrorTerm(jnp.zeros((3, 4)), "bound", "declared exact")
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-6), discrepancy=zero)
    result = fit_linear(basis, data, pm, diagnostics=False)
    assert result.rank < pm.n_free() + 1
    assert result.bias_bound.kind == "unbounded"
    assert "rank" in result.bias_bound.note


# -- R09: column equilibration ------------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-100, 1e-14, 1e-11, 1e-6, 1e6, 1e100])
def test_rank_and_solution_invariant_under_column_rescaling(factor):
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 1)
    d, m = stokes_of(C, pm, theta, A, basis.reference)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    plain = fit_linear(basis, data, pm)
    scale = np.ones(index.n_real)
    scale[1::3] = factor  # moment rows 1, 4, 7, ... in rescaled coordinate units
    scaled_basis = StubBasis(index, basis.reference, C * scale[None, :])
    scaled = fit_linear(scaled_basis, data, pm)
    assert plain.rank == scaled.rank == pm.n_free() + 1
    assert scaled.identifiability.rank == scaled.rank
    m_scaled = np.asarray(scaled.moments.to_vector()) * scale
    assert_allclose(m_scaled, m, rtol=1e-6, atol=1e-8)
    assert_allclose(
        np.asarray(scaled.identifiability.singular_values),
        np.asarray(plain.identifiability.singular_values),
        rtol=1e-8,
    )


# -- R21: tol and the inflate weights -------------------------------------------------------


def test_tol_reaches_identifiability_and_notes():
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 1)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    s = np.asarray(fit_linear(basis, data, pm).identifiability.singular_values)
    tol = float(np.sqrt(s[-1] * s[-2]) / s[0])
    result = fit_linear(basis, data, pm, tol=tol)
    assert result.rank == pm.n_free()
    assert result.identifiability.tol == tol
    assert result.identifiability.rank == result.rank
    notes = [n for n in result.provenance.notes if "covariance None" in n]
    assert notes and all(f"{tol:g}" in n for n in notes)


def test_inflate_diagnostics_use_inflated_weights_and_note_chi2():
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 1)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    rng = np.random.default_rng(1)
    delta = 0.05 * np.abs(d).max() * rng.uniform(0.2, 1.0, d.size)
    noisy = d + delta * np.sign(rng.standard_normal(d.size))
    term = ErrorTerm(jnp.asarray(delta.reshape(-1, 4)), "bound", "declared")
    data = StokesData(noisy.reshape(-1, 4), np.full(d.size, 1e-4), discrepancy=term)
    result = fit_linear(basis, data, pm, discrepancy_policy="inflate")
    inflated = StokesData(noisy.reshape(-1, 4), 1e-4 + delta**2)
    oracle = identifiability(basis, inflated, pm)
    assert_allclose(
        np.asarray(result.identifiability.singular_values),
        np.asarray(oracle.singular_values),
        rtol=1e-10,
    )
    policy = [n for n in result.provenance.notes if "discrepancy_policy" in n]
    assert policy and "chi2" in policy[0]
    assert any("declared noise" in n for n in result.provenance.notes)


# -- R23: discrepancy bias in statistical_input ----------------------------------------------


def test_statistical_input_contains_the_discrepancy_bias_fixed_amplitude():
    """Fixed amplitude: ``m`` is affine in ``theta``, so the propagated bias is
    exact and ``|m_fit - m_true| <= Delta`` holds for noiseless data."""
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    flat, theta, A = truth_of(pm, 1)
    d, m = stokes_of(C, pm, theta, A, basis.reference)
    rng = np.random.default_rng(4)
    delta = 0.02 * np.abs(d).max() * rng.uniform(0.2, 1.0, d.size)
    shifted = d + delta * np.sign(rng.standard_normal(d.size))
    term = ErrorTerm(jnp.asarray(delta.reshape(-1, 4)), "bound", "declared")
    data = StokesData(shifted.reshape(-1, 4), np.full(d.size, 1e-4), discrepancy=term)
    result = fit_linear(basis, data, pm, amplitude=A)
    plain = fit_linear(
        basis,
        StokesData(shifted.reshape(-1, 4), np.full(d.size, 1e-4)),
        pm,
        amplitude=A,
    )
    stat = result.prediction.budget.statistical_input
    assert stat.kind == "estimate" and "bias" in stat.note
    error = np.abs(np.asarray(result.moments.to_vector()) - m)
    contracted = A * (np.abs(C) @ error).reshape(-1, 4)
    assert np.all(contracted <= np.asarray(stat.value) * (1 + 1e-9) + 1e-12)
    # without a declared discrepancy the moment bias is not bounded: explicit
    assert plain.prediction.budget.statistical_input.kind == "unbounded"
    assert "bias" in plain.prediction.budget.statistical_input.note
