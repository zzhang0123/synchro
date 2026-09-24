"""End-to-end synthetic linear fits on the harmonic kernel (``synchro.model.fit``).

Pipeline: ``HarmonicKernel(m_max = required_m_max <= 40)`` on 16 bump
channels, ``Truncation(1, 1, 1)``, a correlated 12-atom population,
``d = A C m* + eta`` with uniform Gaussian noise. Oracles: NumPy ``lstsq``
and ``G^T Sigma^-1 G`` on the whitened design ``diag(1/sigma) C [c | P]``,
the estimator matrix ``|G^+ L^-1| |delta|`` for the bias bound, 200 noise
draws for the coverage of the reported covariance. The
nonlinear routes are in ``test_fit_integration_nonlinear.py``.
"""

import dataclasses
import json
import re

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _integration_fixtures import (
    AMPLITUDE,
    N_CH,
    S_DEPTH,
    cached_basis,
    correlated_population,
    fisher,
    fit_linear,
    isotropic_extension,
    make_data,
    model_of,
    samples_of,
    unknowns,
    whitened_design,
    with_amplitude,
)
from synchro.model.assumptions import gaussian_screen, isotropic_pitch, no_assumption
from synchro.model.errors import ErrorTerm
from synchro.model.fit.observation import StokesData
from synchro.model.moments import JointMoments


@pytest.fixture(scope="module")
def pipeline():
    kernel, basis = cached_basis()
    pop = correlated_population()
    joint = JointMoments.from_samples(samples_of(pop), basis.index, basis.reference)
    return kernel, basis, pop, joint


def truth_of(name, basis, pop, joint):
    """``(pm, theta*, m*)``: the model truth is the projection of a matching population."""
    if name == "gaussian_screen":
        pm, source = gaussian_screen(basis.index, 0.5 * S_DEPTH, 0.2 * S_DEPTH), joint
    else:
        pm = isotropic_pitch(basis.index)
        iso = samples_of(isotropic_extension(pop))
        source = JointMoments.from_samples(iso, basis.index, basis.reference)
        assert float(jnp.max(jnp.abs(pm.constraint_residual(source)))) < 1e-12
    theta = pm.project(source)
    return pm, theta, pm(theta, basis.reference).to_vector()


def odd_parity(label):
    """True for a real free-table label ``<... P_l(mu) P_k(eta) ...>`` with ``l + k`` odd."""
    if label.startswith(("Re ", "Im ", "amplitude")):
        return False
    degrees = [int(d) for d in re.findall(r"P_(\d+)\((?:mu|eta)\)", label)]
    return sum(degrees) % 2 == 1


# -- setup ---------------------------------------------------------------------


def test_pipeline_is_inside_the_harmonic_regime(pipeline):
    kernel, basis, pop, joint = pipeline
    assert kernel.m_max <= 40
    assert basis.channels.family == "bump" and basis.n_ch == N_CH
    C = np.asarray(basis.response_matrix())
    assert C.shape == (4 * N_CH, basis.index.n_real) == (64, 28)
    assert np.linalg.matrix_rank(C) == 28
    assert basis.kernel_terms.harmonic_truncation.kind == "bound"
    assert dict(basis.provenance.kernel)["name"] == "harmonic"


# -- rank-deficient and full-rank designs ----------------------------------------


def test_no_assumption_without_v_data_is_rank_deficient(pipeline):
    """Masking the V rows leaves the six odd-parity ``h0`` moments unconstrained."""
    kernel, basis, pop, joint = pipeline
    pm = no_assumption(basis.index)
    data, sigma = make_data(basis, joint.to_vector(), AMPLITUDE, seed=1)
    masked = StokesData(data.stokes, data.noise, mask=np.array([1, 1, 1, 0]))
    result = fit_linear(basis, masked, pm)
    report = result.identifiability
    assert report.labels == ("amplitude",) + pm.labels()
    odd = [i for i, label in enumerate(report.labels) if odd_parity(label)]
    n_u = pm.n_free() + 1
    assert len(odd) == sum((l + k) % 2 for l, k, *_ in basis.index.h0) == 6
    assert result.rank == n_u - len(odd) == 22 and result.covariance is None
    assert result.dof == masked.n_kept() - result.rank == 26
    assert report.null_dim == len(odd) == 6
    null = np.asarray(report.null_basis)
    keep = np.ones(n_u, bool)
    keep[odd] = False  # only the V-only (odd l + k) real moments are unconstrained
    assert np.max(np.abs(null[keep])) < 1e-10
    assert np.max(np.abs(null[odd])) > 0.5
    # the minimum-norm solution and its chi-square against NumPy lstsq
    Gw, dw = whitened_design(basis, masked, pm)
    rows = np.asarray(masked.kept_rows())
    u_np, *_ = np.linalg.lstsq(Gw[rows], dw[rows], rcond=None)
    u = unknowns(pm, result.theta, float(result.amplitude))
    assert_allclose(u, u_np, rtol=1e-8, atol=1e-10 * np.max(np.abs(u_np)))
    assert_allclose(float(result.chi2), np.sum((dw[rows] - Gw[rows] @ u_np) ** 2))
    # predictions reproduced: I, Q, U within noise, V (unconstrained) at zero
    stokes = np.asarray(result.prediction.stokes)
    d = np.asarray(masked.stokes)
    assert np.max(np.abs(stokes[:, :3] - d[:, :3])) < 5 * sigma
    assert np.max(np.abs(stokes[:, 3])) < 1e-9 * np.max(np.abs(d))
    assert_allclose(stokes.reshape(-1), model_of(basis, pm, u), rtol=1e-10, atol=1e-12)
    # a null vector leaves the I, Q, U prediction unchanged and moves V only
    shifted = model_of(basis, pm, u + 0.3 * null[:, 0] * np.max(np.abs(u)))
    assert_allclose(shifted.reshape(-1, 4)[:, :3], stokes[:, :3], rtol=1e-10)
    assert np.max(np.abs(shifted.reshape(-1, 4)[:, 3])) > 1e-6 * np.max(np.abs(d))
    assert result.bias_bound.kind == "unbounded"
    json.dumps(result.to_dict())


def test_no_assumption_with_all_stokes_is_full_rank(pipeline):
    """Other side of the rank dispatch: all four Stokes make ``no_assumption`` identifiable."""
    kernel, basis, pop, joint = pipeline
    pm = no_assumption(basis.index)
    data, _ = make_data(basis, joint.to_vector(), AMPLITUDE, seed=1)
    result = fit_linear(basis, data, pm)
    assert result.rank == pm.n_free() + 1 == 28
    assert result.covariance is not None and result.identifiability.null_dim == 0
    assert result.dof == 64 - 28


@pytest.mark.parametrize("name", ["isotropic_pitch", "gaussian_screen"])
def test_affine_fit_recovers_the_truth_within_fisher_errors(pipeline, name):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true = truth_of(name, basis, pop, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=11)
    result = fit_linear(basis, data, pm)
    n_u = pm.n_free() + 1
    assert result.rank == n_u and result.identifiability.null_dim == 0
    assert result.dof == 64 - n_u and result.converged
    Gw, _ = whitened_design(basis, data, pm)
    F = Gw.T @ Gw
    assert_allclose(np.asarray(result.fisher), F, rtol=1e-10, atol=1e-10 * F.max())
    theta_A = with_amplitude(theta_true, AMPLITUDE)
    linear = fisher(basis, data, pm, theta_A, coordinates="linear")
    assert_allclose(np.asarray(linear), F, rtol=1e-10, atol=1e-10 * F.max())
    # natural coordinates x = (A, theta): F_x = T^T F_u T with T = du/dx
    flat = np.asarray(pm.flatten(theta_true))
    T = np.block(
        [[1.0, np.zeros(flat.size)], [flat[:, None], AMPLITUDE * np.eye(flat.size)]]
    )
    natural, expected = np.asarray(fisher(basis, data, pm, theta_A)), T.T @ F @ T
    assert_allclose(natural, expected, rtol=1e-9, atol=1e-12 * np.max(np.abs(expected)))
    cov = np.asarray(result.covariance)
    # entries below 1e-9 of the largest are roundoff of the ill-conditioned inverse
    assert_allclose(cov, np.linalg.pinv(F), rtol=1e-6, atol=1e-9 * np.max(np.abs(cov)))
    u_true = unknowns(pm, theta_true, AMPLITUDE)
    u_fit = unknowns(pm, result.theta, float(result.amplitude))
    pulls = np.abs(u_fit - u_true) / np.sqrt(np.diag(cov))
    assert np.max(pulls) < 4.5
    dof = result.dof
    assert abs(float(result.chi2) - dof) < 5 * np.sqrt(2 * dof)
    expected = model_of(basis, pm, u_fit)  # A C m_fit in the u coordinates
    assert_allclose(
        np.asarray(result.prediction.stokes).reshape(-1), expected, rtol=1e-12
    )
    assert [r.name for r in result.moments.assumptions] == [pm.name]
    # the report lists necessary conditions; the noiseless fit satisfies all of them
    assert (
        result.feasibility.checks
        and "necessary" in result.feasibility.statement.lower()
    )
    exact, _ = make_data(basis, m_true, AMPLITUDE, seed=None)
    clean = fit_linear(basis, exact, pm)
    assert clean.feasibility.failed() == () and float(clean.chi2) < 1e-12
    assert_allclose(
        unknowns(pm, clean.theta, float(clean.amplitude)), u_true, rtol=1e-8
    )


def test_reported_covariance_has_gaussian_coverage(pipeline):
    """200 noise draws of the least-squares estimator on the fitted design."""
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true = truth_of("isotropic_pitch", basis, pop, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=12)
    result = fit_linear(basis, data, pm)
    Gw, _ = whitened_design(basis, data, pm)
    u_true = unknowns(pm, theta_true, AMPLITUDE)
    scale = np.sqrt(np.diag(np.asarray(result.covariance)))
    rng = np.random.default_rng(2024)
    pulls = []
    for _ in range(200):
        u_hat, *_ = np.linalg.lstsq(
            Gw, Gw @ u_true + rng.standard_normal(64), rcond=None
        )
        pulls.append((u_hat - u_true) / scale)
    pulls = np.abs(np.array(pulls))
    assert np.mean(pulls < 3.0) > 0.99
    assert 0.62 < np.mean(pulls < 1.0) < 0.74


# -- reports, discrepancy policies, assumption records -----------------------------


def test_identifiability_and_feasibility_reports_serialise(pipeline):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true = truth_of("isotropic_pitch", basis, pop, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=3)
    result = fit_linear(basis, data, pm)
    ident = json.loads(json.dumps(result.identifiability.to_dict()))
    assert ident["null_dim"] == 0 and ident["labels"][0] == "amplitude"
    assert set(ident["resolution"]) == set(result.identifiability.labels)
    assert len(ident["singular_values"]) == pm.n_free() + 1
    feas = json.loads(json.dumps(result.feasibility.to_dict()))
    names = {name for name, _, _ in feas["checks"]}
    assert "normalisation" in names and feas["not_computable"]
    assert [name for name, passed, _ in feas["checks"] if not passed] == list(
        result.feasibility.failed()
    )
    full = json.loads(json.dumps(result.to_dict()))
    for key in ("amplitude", "chi2", "rank", "dof", "prediction", "identifiability"):
        assert key in full
    assert full["rank"] == result.rank and full["dof"] == result.dof
    assert full["prediction"]["budget"]["assumption"][pm.name]["kind"] == "unbounded"
    assert full["bias_bound"]["kind"] == "unbounded"


def test_discrepancy_policies_with_a_violating_population(pipeline):
    """Data from the correlated (pitch-dependent) population fitted under
    ``isotropic_pitch``: the declared ``|delta| = A |C (m_joint - m*)|`` gives a bias
    bound that contains the actual bias; ``inflate`` widens the covariance."""
    kernel, basis, pop, joint = pipeline
    pm = isotropic_pitch(basis.index)
    theta_star = pm.project(joint)
    m_star = pm(theta_star, basis.reference).to_vector()
    C = np.asarray(basis.response_matrix())
    delta = AMPLITUDE * C @ (np.asarray(joint.to_vector()) - np.asarray(m_star))
    assert np.max(np.abs(delta)) > 0.1
    term = ErrorTerm(jnp.abs(jnp.asarray(delta)).reshape(-1, 4), "bound", "model error")
    data, _ = make_data(
        basis, joint.to_vector(), AMPLITUDE, seed=None, discrepancy=term
    )
    bias = fit_linear(basis, data, pm, discrepancy_policy="bias_bound")
    u_star = unknowns(pm, theta_star, AMPLITUDE)
    u_fit = unknowns(pm, bias.theta, float(bias.amplitude))
    assert bias.bias_bound.kind == "bound"
    bound = np.asarray(bias.bias_bound.value)
    assert bound.shape == u_star.shape and np.all(bound >= 0)
    actual = np.abs(u_fit - u_star)
    assert np.all(actual <= bound * (1 + 1e-8) + 1e-12 * np.max(bound))
    assert np.max(actual / bound) > 0.1  # the bound is not vacuous
    Gw, _ = whitened_design(basis, data, pm)
    expected = np.abs(np.linalg.pinv(Gw) / np.sqrt(np.asarray(data.noise))) @ np.abs(
        delta
    )
    assert_allclose(bound, expected, rtol=1e-8)
    inflate = fit_linear(basis, data, pm, discrepancy_policy="inflate")
    assert np.all(
        np.diag(np.asarray(inflate.covariance)) >= np.diag(np.asarray(bias.covariance))
    )
    assert float(inflate.chi2) < float(bias.chi2)
    assert "heuristic" in json.dumps(inflate.to_dict()).lower()
    # the inflated fit is a different estimator; its bias bound still contains its bias
    u_inflate = unknowns(pm, inflate.theta, float(inflate.amplitude))
    inflate_bound = np.asarray(inflate.bias_bound.value)
    assert np.all(np.abs(u_inflate - u_star) <= inflate_bound * (1 + 1e-8))
    with pytest.raises(ValueError):
        fit_linear(basis, data, pm, discrepancy_policy="bogus")
    plain, _ = make_data(basis, joint.to_vector(), AMPLITUDE, seed=None)
    assert fit_linear(basis, plain, pm).bias_bound.kind == "unbounded"
    with pytest.raises(ValueError):
        fit_linear(basis, plain, pm, discrepancy_policy="inflate")


def test_fit_prediction_budget_carries_the_fitted_assumption_records(pipeline):
    kernel, basis, pop, joint = pipeline
    pm, theta_true, m_true = truth_of("isotropic_pitch", basis, pop, joint)
    data, _ = make_data(basis, m_true, AMPLITUDE, seed=5)
    result = fit_linear(basis, data, pm)
    prediction = result.prediction
    names = dict(prediction.budget.assumption)
    assert set(names) == {pm.name} and names[pm.name].kind == "unbounded"
    assert f"assumption:{pm.name}" in prediction.budget.unbounded()
    records = {r.name: r for r in prediction.provenance.assumptions}
    assert records[pm.name].n_free == pm.n_free() == 13
    assert records[pm.name].closure_kind == "uniform_mu"
    assert records[pm.name] == result.moments.assumptions[0]
    assert result.provenance.assumptions == prediction.provenance.assumptions
    assert prediction.to_dict()["provenance"]["assumptions"][0]["name"] == pm.name
    # a map that declares its discrepancy turns the term into a bound A |C| Delta
    delta = np.full(basis.index.n_real, 1e-3)
    declared = dataclasses.replace(
        pm, discrepancy=ErrorTerm(jnp.asarray(delta), "bound")
    )
    bounded = fit_linear(basis, data, declared)
    term = dict(bounded.prediction.budget.assumption)[pm.name]
    assert term.kind == "bound"
    C = np.abs(np.asarray(basis.response_matrix()))
    expected = float(bounded.amplitude) * (C @ delta).reshape(-1, 4)
    assert_allclose(np.asarray(term.value), expected, rtol=1e-12)
    assert bounded.moments.assumptions[0].discrepancy_kind == "bound"
    assert f"assumption:{pm.name}" not in bounded.prediction.budget.unbounded()
