"""End-to-end nonlinear fits on the harmonic kernel (``synchro.model.fit``).

Same pipeline as ``test_fit_integration.py`` (harmonic kernel, 16 bump
channels, ``Truncation(1, 1, 1)``, correlated atoms). Routes: ``fit_bfgs``
on the multilinear ``field_independent`` map started from ``project()`` of
the ``no_assumption`` least-squares solution, ``fit_bfgs`` on
``independent_screen`` (refused by ``fit_linear``), ``fit_nodal`` on the
atoms themselves, and the ``FitResult`` wrappers ``from_bfgs``/``from_nodal``
when ``synchro.model.fit.result`` imports. Oracles: ``jax.jacfwd`` Fisher
information at the truth, explicit atom weights, NumPy chi-square.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _integration_fixtures import (
    AMPLITUDE,
    N_CH,
    cached_basis,
    correlated_population,
    fit_linear,
    make_data,
    samples_of,
)
from synchro.model.assumptions import (
    field_independent,
    independent_screen,
    no_assumption,
    nodal,
)
from synchro.model.errors import ErrorTerm
from synchro.model.fit.diagnostics import feasibility_checks, identifiability
from synchro.model.fit.nonlinear import LogDensity, Transform, fit_bfgs, fit_nodal
from synchro.model.fit.result import FitResult
from synchro.model.moments import JointMoments


@pytest.fixture(scope="module")
def pipeline():
    kernel, basis = cached_basis()
    pop = correlated_population()
    joint = JointMoments.from_samples(samples_of(pop), basis.index, basis.reference)
    return kernel, basis, pop, joint


def fisher_at(density, z):
    J = jax.jacfwd(density.whitened_residual)(z)
    return np.asarray(J.T @ J)


def multilinear_truth(make_map, basis, joint):
    """``(pm, theta*, m*, z*)`` with the truth projected from the correlated population."""
    pm = make_map(basis.index)
    assert not pm.is_affine()
    theta = pm.project(joint, amplitude=AMPLITUDE)
    m = pm(theta, basis.reference).to_vector()
    return pm, theta, m, Transform(pm).forward(theta)


# -- BFGS ------------------------------------------------------------------------


def test_independent_screen_needs_the_nonlinear_route(pipeline):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true, z_true = multilinear_truth(independent_screen, basis, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=None)
    with pytest.raises(ValueError):
        fit_linear(basis, data, pm)
    report = identifiability(basis, data, pm, theta=theta_true)
    assert report.null_dim == 0 and "linearised" in report.note
    density = LogDensity(basis, data, pm)
    z0 = z_true + 0.3 * jnp.asarray(
        np.random.default_rng(0).standard_normal(z_true.shape)
    )
    out = fit_bfgs(density, z0, maxiter=3000)
    assert out.converged
    assert float(density.chi2(out.z)) < 1e-12 * float(density.chi2(z0))
    assert_allclose(np.asarray(out.z), np.asarray(z_true), atol=1e-6)
    fitted = pm(out.theta, basis.reference)
    assert [r.name for r in fitted.assumptions] == ["independent_screen"]
    assert_allclose(np.asarray(fitted.to_vector()), np.asarray(m_true), atol=1e-6)


def test_bfgs_field_independent_from_the_projected_wls_solution(pipeline):
    """Low noise (1e-4): WLS under ``no_assumption`` -> ``project`` -> BFGS polish."""
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true, z_true = multilinear_truth(field_independent, basis, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=2, noise=1e-4)
    wls = fit_linear(basis, data, no_assumption(basis.index))
    assert wls.rank == 28
    # the WLS moments violate the product structure; the projection restores it
    assert float(jnp.max(jnp.abs(pm.constraint_residual(wls.moments)))) > 1e-6
    theta0 = pm.project(wls.moments, amplitude=wls.amplitude)
    z0 = Transform(pm).forward(theta0)
    density = LogDensity(basis, data, pm)
    out = fit_bfgs(density, z0, maxiter=3000)
    assert out.converged and out.n_iter > 0
    chi2_fit, chi2_true = float(density.chi2(out.z)), float(density.chi2(z_true))
    assert chi2_fit <= chi2_true * (1 + 1e-9)
    dof = 64 - z_true.shape[0]
    assert abs(chi2_fit - dof) < 5 * np.sqrt(2 * dof)
    F = fisher_at(density, z_true)
    pulls = np.abs(np.asarray(out.z - z_true)) / np.sqrt(np.diag(np.linalg.pinv(F)))
    assert np.max(pulls) < 5.0
    fitted = pm(out.theta, basis.reference)
    assert [r.name for r in fitted.assumptions] == ["field_independent"]
    assert float(jnp.max(jnp.abs(pm.constraint_residual(fitted)))) < 1e-10


def test_bfgs_from_exact_wls_projection_stays_at_the_truth(pipeline):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true, z_true = multilinear_truth(field_independent, basis, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=None)
    wls = fit_linear(basis, data, no_assumption(basis.index))
    assert float(wls.chi2) < 1e-12
    z0 = Transform(pm).forward(pm.project(wls.moments, amplitude=wls.amplitude))
    assert_allclose(np.asarray(z0), np.asarray(z_true), atol=1e-8)
    density = LogDensity(basis, data, pm)
    out = fit_bfgs(density, z0, maxiter=100)
    assert out.converged and out.n_iter <= 2
    assert_allclose(np.asarray(out.z), np.asarray(z_true), atol=1e-8)


def test_bfgs_result_wrapper_reports_fisher_in_z(pipeline):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true, z_true = multilinear_truth(independent_screen, basis, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=4)
    density = LogDensity(basis, data, pm)
    result = out = fit_bfgs(density, z_true, maxiter=3000)
    assert isinstance(result, FitResult)
    assert result.method == "bfgs" and result.labels == density.transform.labels
    assert result.rank == z_true.shape[0] and result.covariance is not None
    assert result.dof == 64 - z_true.shape[0]
    assert_allclose(np.asarray(result.fisher), fisher_at(density, out.z), rtol=1e-10)
    assert_allclose(float(result.chi2), float(density.chi2(out.z)))
    assert result.converged == out.converged
    assert dict(result.prediction.budget.assumption)[pm.name].kind == "unbounded"
    # NEW-4: without a declared data.discrepancy the moment bias is not bounded,
    # so bias_bound and statistical_input are unbounded (as for fit_linear)
    assert result.bias_bound.kind == "unbounded"
    assert result.prediction.budget.statistical_input.kind == "unbounded"
    assert {r.name for r in result.provenance.assumptions} == {pm.name}
    assert_allclose(float(result.amplitude), float(jnp.exp(out.theta.log_amplitude)))
    full = json.loads(json.dumps(result.to_dict()))
    assert full["method"] == "bfgs" and set(full["parameters"]) == set(pm.labels())


# -- nodal -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def nodal_case(pipeline):
    kernel, basis, pop, joint = pipeline
    atoms = correlated_population(n=6)
    nodes = samples_of(atoms, weights=False)
    w_true = atoms["w"] / atoms["w"].sum()
    m_true = JointMoments.from_samples(
        samples_of(atoms), basis.index, basis.reference
    ).to_vector()
    data, sigma = make_data(basis, m_true, AMPLITUDE, seed=None)
    term = ErrorTerm.declared_zero("nodes are the atoms", shape=(N_CH, 4))
    out = fit_nodal(basis, data, nodes, iters=5000, step=0.1, discretisation=term)
    return nodes, w_true, m_true, data, sigma, term, out


def test_nodal_fit_recovers_atom_weights(pipeline, nodal_case):
    kernel, basis, pop, joint = pipeline
    nodes, w_true, m_true, data, sigma, term, out = nodal_case
    w = np.asarray(out.weights)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    assert np.max(np.abs(w - w_true)) < 1e-2
    assert abs(float(jnp.exp(out.theta.log_amplitude)) / AMPLITUDE - 1) < 1e-2
    assert 2 * out.fun < 0.1  # chi-square of the exact data
    assert out.discretisation is term and out.n_iter <= 5000
    pm = nodal(basis.index, nodes)
    assert pm.n_free() == 6 and pm.record().closure_kind == "nodal"
    fitted = pm(out.theta, basis.reference)
    assert_allclose(np.asarray(fitted.to_vector()), np.asarray(m_true), atol=2e-2)
    assert feasibility_checks(fitted, basis.index, basis.support).failed() == ()
    C = np.asarray(basis.response_matrix())
    model = float(jnp.exp(out.theta.log_amplitude)) * C @ np.asarray(fitted.to_vector())
    assert np.max(np.abs(model - np.asarray(data.data_vector()))) < 0.5 * sigma


def test_nodal_result_wrapper_carries_the_discretisation_term(pipeline, nodal_case):
    kernel, basis, pop, joint = pipeline
    nodes, w_true, m_true, data, sigma, term, result = nodal_case
    out = result
    assert isinstance(result, FitResult)
    assert result.method == "nodal" and result.labels[0] == "log_amplitude"
    assert len(result.labels) == 7 and result.weights is out.weights
    assert result.identifiability is None and result.discretisation is term
    # the softmax gauge direction is a null vector of the Fisher matrix
    assert result.rank == 6 and result.covariance is None
    gauge = np.concatenate([[0.0], np.ones(6)])
    F = np.asarray(result.fisher)
    assert np.max(np.abs(F @ gauge)) < 1e-8 * np.max(np.abs(F))
    assert_allclose(float(result.chi2), 2 * out.fun)
    assumption = dict(result.prediction.budget.assumption)
    assert assumption["nodal"].kind == "bound"
    assert np.max(np.abs(np.asarray(assumption["nodal"].value))) == 0.0
    assert "assumption:nodal" not in result.prediction.budget.unbounded()
    assert result.feasibility.failed() == ()
    full = json.loads(json.dumps(result.to_dict()))
    assert full["method"] == "nodal" and len(full["weights"]) == 6
