"""Noise, discrepancy and unresolved-direction terms of ``fit_combinations`` (T-006).

Oracles: Monte Carlo draws through the estimator (statistical tolerances
``6/sqrt(N)``), the exact identity ``x_hat - Pi x = K (delta + R dC a) + K n``,
attainment of ``|K| E`` by ``delta = E sign(K_i)``, the ``fit_linear`` bias
bound (drift guard) and rigorous coefficient bounds on populations inside the
declared support. Unconstrained terms are never zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from _reduction_fixtures import (
    SUPPORT,
    continuum_problem,
    matrix_basis,
    oracle_fit,
    random_columns,
    reference,
    stokes_data,
)
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.fit import (
    LinearRelation,
    StokesData,
    coefficient_bounds,
    fit_combinations,
    fit_linear,
    reduce_response,
)
from syncmoments.model.moments import JointMoments, PopulationSamples, Support
from test_combinations import _polynomial_problem


def _with(data, **kw):
    fields = dict(mask=data.mask, response=data.response, discrepancy=data.discrepancy)
    fields.update(kw)
    return StokesData(data.stokes, data.noise, **fields)


@pytest.mark.parametrize("dense", [False, True])
def test_noise_covariance_monte_carlo(dense):
    basis, _, _, data = continuum_problem(dense=dense)
    fit = fit_combinations(basis, data, max_sigma=0.1)
    N = 4000
    kept = list(data.kept_rows())
    noise = np.asarray(data.noise)
    cov = (np.diag(noise) if noise.ndim == 1 else noise)[np.ix_(kept, kept)]
    draws = np.linalg.cholesky(cov) @ np.random.default_rng(7).standard_normal(
        (len(kept), N)
    )
    beta = np.asarray(fit.beta_estimator) @ draws / np.asarray(fit.beta_sigma)[:, None]
    sample = beta @ beta.T / N
    assert np.max(np.abs(sample - np.eye(fit.n_retained))) < 6 / np.sqrt(N)
    x = np.asarray(fit.estimator) @ draws
    diag = np.diag(np.asarray(fit.covariance))
    live = diag > 1e-12 * diag.max()
    ratio = np.mean(x[live] ** 2, axis=1) / diag[live]
    assert np.max(np.abs(ratio - 1)) < 6 * np.sqrt(2 / N)


def test_K_delta_identity():
    basis, a_true, clean, data = continuum_problem()
    rng = np.random.default_rng(4)
    delta = 1e-2 * np.abs(clean).max() * rng.standard_normal(clean.shape)
    noise = np.sqrt(np.asarray(data.noise)[0]) * rng.standard_normal(clean.shape)
    stokes = clean + delta + noise
    obs = StokesData(stokes, data.noise, mask=data.mask)
    fit = fit_combinations(basis, obs, max_sigma=0.1)
    cmp = fit.against_truth(a_true, discrepancy=delta, noise=noise.reshape(-1))
    scale = np.max(np.abs(np.asarray(fit.x_hat)))
    assert np.max(np.abs(np.asarray(cmp.decomposition_residual))) <= 1e-10 * scale
    kept = list(obs.kept_rows())
    K, Pi = np.asarray(fit.estimator), np.asarray(fit.projector)
    direct = (
        np.asarray(fit.x_hat)
        - Pi @ a_true
        - K @ (delta.reshape(-1)[kept] + noise.reshape(-1)[kept])
    )
    assert np.max(np.abs(direct)) <= 1e-10 * scale
    np.testing.assert_allclose(
        np.asarray(cmp.beta_true), np.asarray(fit.combination_rows) @ a_true
    )


def test_envelope_propagation_is_attained():
    basis, _, clean, data = continuum_problem()
    E = (
        1e-3
        * np.abs(clean).max()
        * (1 + np.random.default_rng(1).uniform(size=clean.shape))
    )
    E[:, 3] = 0.0
    obs = _with(data, discrepancy=ErrorTerm(E, "bound", "declared envelope"))
    fit = fit_combinations(basis, obs, max_sigma=0.1)
    K = np.asarray(fit.estimator)
    kept = list(obs.kept_rows())
    E_k = E.reshape(-1)[kept]
    bound = np.asarray(fit.bias.value)
    for i in range(0, K.shape[0], 7):
        attained = K[i] @ (E_k * np.sign(K[i]))
        np.testing.assert_allclose(attained, bound[i], rtol=1e-12, atol=1e-300)
    rng = np.random.default_rng(2)
    for _ in range(20):
        delta = E_k * rng.uniform(-1, 1, E_k.size)
        assert np.all(np.abs(K @ delta) <= bound * (1 + 1e-12))
    assert fit.bias.kind == "bound"


def test_bias_kinds():
    basis, _, clean, data = continuum_problem()
    E = np.full((8, 4), 1e-3 * np.abs(clean).max())
    for kind in ("bound", "estimate", "measured"):
        fit = fit_combinations(
            basis, _with(data, discrepancy=ErrorTerm(E, kind, "x")), max_sigma=0.1
        )
        assert fit.bias.kind == kind == fit.beta_bias.kind
    none = fit_combinations(basis, data, max_sigma=0.1)
    assert none.bias.kind == "unbounded" and none.bias.value is None
    unb = fit_combinations(
        basis, _with(data, discrepancy=ErrorTerm.unbounded("x")), max_sigma=0.1
    )
    assert unb.bias.kind == "unbounded"
    R = np.eye(32)
    for unc, kind in (
        (None, "unbounded"),
        (np.zeros((32, 32)), "bound"),
        (1e-3 * np.ones((32, 32)), "estimate"),
    ):
        obs = StokesData(
            data.stokes,
            data.noise,
            mask=np.tile([1, 1, 1, 0], 8).astype(bool),
            response=R,
            response_uncertainty=unc,
            discrepancy=ErrorTerm(E.reshape(-1), "bound", "x"),
        )
        assert fit_combinations(basis, obs, max_sigma=0.1).bias.kind == kind


@pytest.mark.parametrize("uncertainty", [None, 1e-4])
def test_data_discrepancy_matches_fit_linear(uncertainty):
    basis, pm, data = _polynomial_problem()
    if uncertainty is not None:
        R = np.eye(64)
        data = StokesData(
            np.asarray(data.stokes).reshape(-1),
            data.noise,
            response=R,
            response_uncertainty=np.full((64, 64), uncertainty),
            discrepancy=ErrorTerm(
                np.asarray(data.discrepancy.value).reshape(-1), "bound", "d"
            ),
        )
    lin = fit_linear(basis, data, pm)
    fit = fit_combinations(basis, data, max_sigma=np.inf, parameter_map=pm)
    assert lin.rank == fit.n_retained == 28
    assert fit.bias.kind == lin.bias_bound.kind
    np.testing.assert_allclose(
        np.asarray(fit.bias.value), np.asarray(lin.bias_bound.value), rtol=1e-8
    )


def test_beta_bias_matches_rows():
    basis, _, clean, data = continuum_problem()
    E = np.full((8, 4), 1e-3 * np.abs(clean).max())
    fit = fit_combinations(
        basis, _with(data, discrepancy=ErrorTerm(E, "bound", "x")), max_sigma=0.1
    )
    kept = list(data.kept_rows())
    Kb = np.asarray(fit.beta_estimator)
    np.testing.assert_allclose(
        np.asarray(fit.beta_bias.value), np.abs(Kb) @ E.reshape(-1)[kept]
    )
    B, K = np.asarray(fit.combination_rows), np.asarray(fit.estimator)
    np.testing.assert_allclose(B @ K, Kb, atol=1e-12 * np.abs(Kb).max())


def _population(seed, n=64, corners=False):
    rng = np.random.default_rng(seed)
    lo, hi = np.array([8000.0, 4e-6, 1.7]), np.array([12000.0, 6e-6, 2.3])
    if corners:
        pick = rng.integers(0, 2, (n, 3))
        g, B, d = (lo + pick * (hi - lo)).T
        mu, eta = rng.choice([-1.0, 1.0], n), rng.choice([-1.0, 1.0], n)
    else:
        g, B, d = (lo + rng.uniform(size=(n, 3)) * (hi - lo)).T
        mu, eta = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    return PopulationSamples(
        gamma=g,
        B=B,
        mu=mu,
        eta=eta,
        phi=rng.uniform(0, 2 * np.pi, n),
        depth=d,
        weights=rng.uniform(0.1, 1.0, n),
    )


def _truth(layout, seed, corners=False, amplitude=1.7):
    moments = JointMoments.from_samples(
        _population(seed, corners=corners), layout.index, reference()
    )
    return layout.full_vector(moments, amplitude=amplitude)


def test_unresolved_unbounded_without_bound():
    basis, _, _, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=0.1)
    assert fit.unresolved.kind == "unbounded" and fit.unresolved.value is None
    assert "unbounded, not zero" in fit.unresolved.note
    slots = fit.to_dict()["slots"]
    assert any(s.get("unresolved") == "unbounded" for s in slots)


def test_unresolved_with_bound():
    basis, _, _, data = continuum_problem()
    layout = reduce_response(basis).layout
    a_true = _truth(layout, 3)
    clean = np.asarray(basis.response_matrix()) @ a_true[:52]
    obs = stokes_data(
        clean.reshape(-1, 4), sigma=1e-3 * np.abs(clean).max(), mask=data.mask
    )
    bound = coefficient_bounds(layout, SUPPORT, amplitude_bound=2.0)
    fit = fit_combinations(basis, obs, max_sigma=0.1, coefficient_bound=bound)
    assert fit.unresolved.kind == "bound"
    cmp = fit.against_truth(a_true)
    assert np.all(
        np.asarray(fit.unresolved.value)
        >= np.abs(np.asarray(cmp.unresolved_truth)) - 1e-14
    )


@pytest.mark.parametrize("corners", [False, True])
def test_coefficient_bounds_rigorous(corners):
    layout = reduce_response(continuum_problem()[0]).layout
    bound = np.asarray(coefficient_bounds(layout, SUPPORT, amplitude_bound=2.0).value)
    for seed in range(6):
        a = _truth(layout, seed, corners=corners, amplitude=2.0)
        assert np.all(np.abs(a) <= bound * (1 + 1e-12) + 1e-300)
    assert (
        coefficient_bounds(layout, SUPPORT, amplitude_bound=np.inf).kind == "unbounded"
    )
    assert coefficient_bounds(layout, None, amplitude_bound=1.0).kind == "unbounded"
    assert coefficient_bounds(layout, SUPPORT, amplitude_bound=-1.0).kind == "unbounded"
    wide = Support(
        gamma=(8000.0, 12000.0), B=(4e-6, 6e-6), depth=(1.7, 2.3), truncated=True
    )
    assert coefficient_bounds(layout, wide, amplitude_bound=1.0).kind == "bound"


def _approximate_problem(scale=1e-4):
    index, C = random_columns()
    C[:, 7] = -2.5 * C[:, 6] + scale * np.random.default_rng(9).standard_normal(
        C.shape[0]
    )
    a_true = np.random.default_rng(1).standard_normal(60)
    clean = (C @ a_true[:52]).reshape(-1, 4)
    data = stokes_data(clean, sigma=1e-3, mask=np.array([1, 1, 1, 0]))
    return matrix_basis(C, index=index), a_true, data


def test_approximate_reduction_bias():
    basis, a_true, data = _approximate_problem()
    rel = LinearRelation.proportional(7, 6, -2.5, scope="near pair", approximate=True)
    red = reduce_response(basis, declared=(rel,))
    bound = ErrorTerm(np.abs(a_true) + 0.5, "bound", "declared |a|")
    fit = fit_combinations(
        basis, data, reduction=red, max_sigma=1e3, coefficient_bound=bound
    )
    assert fit.reduction_bias.kind == "bound"
    kept = list(data.kept_rows())
    actual = np.abs(
        np.asarray(fit.estimator) @ (np.asarray(red.delta_C) @ a_true)[kept]
    )
    assert np.all(np.asarray(fit.reduction_bias.value) >= actual - 1e-15)
    assert np.max(actual) > 0
    free = fit_combinations(basis, data, reduction=red, max_sigma=1e3)
    assert free.reduction_bias.kind == "unbounded"
    exact = fit_combinations(basis, data, max_sigma=1e3)
    assert exact.reduction_bias.kind == "not_applicable"


def test_observable_summary():
    basis, _, clean, data = continuum_problem()
    E = np.full((8, 4), 1e-3 * np.abs(clean).max())
    fit = fit_combinations(
        basis, _with(data, discrepancy=ErrorTerm(E, "bound", "x")), max_sigma=0.1
    )
    B = np.asarray(fit.combination_rows)
    in_span = fit.observable(B)
    assert in_span.unresolved.kind == "not_applicable"
    np.testing.assert_allclose(
        np.asarray(in_span.value), np.asarray(fit.beta_hat), atol=1e-12
    )
    assert in_span.total().kind == "bound"
    J = np.random.default_rng(0).standard_normal((3, 60))
    generic = fit.observable(J, on="full")
    assert (
        generic.unresolved.kind == "unbounded" and generic.total().kind == "unbounded"
    )
    Cov, K = np.asarray(fit.covariance), np.asarray(fit.estimator)
    np.testing.assert_allclose(
        np.asarray(generic.noise_covariance), J @ Cov @ J.T, atol=1e-14
    )
    np.testing.assert_allclose(np.asarray(generic.gain), J @ K)
    kept = list(data.kept_rows())
    np.testing.assert_allclose(
        np.asarray(generic.bias.value), np.abs(J @ K) @ E.reshape(-1)[kept]
    )
    assert set(generic.null_sensitivity) == {"weak", "numerical_null", "analytic_null"}
    assert generic.reduction_bias.kind == "not_applicable"
    with pytest.raises(ValueError):
        fit.observable(np.ones((2, 7)))


def test_inflate_policy_semantics():
    basis, _, clean, data = continuum_problem()
    E = np.full((8, 4), 2e-3 * np.abs(clean).max())
    obs = _with(data, discrepancy=ErrorTerm(E, "bound", "x"))
    fit = fit_combinations(basis, obs, max_sigma=0.1, discrepancy_policy="inflate")
    kept = list(obs.kept_rows())
    inflated = StokesData(
        obs.stokes,
        np.asarray(obs.noise)
        + np.where(np.tile([1, 1, 1, 0], 8), E.reshape(-1) ** 2, 0.0),
        mask=obs.mask,
    )
    oracle = oracle_fit(fit.reduction.C, inflated, max_sigma=0.1)
    s = np.asarray(fit.singular_values)
    np.testing.assert_allclose(s[s > 1e-8 * s[0]], oracle["s"], rtol=1e-10)
    notes = " ".join(fit.provenance.notes)
    assert "heuristic" in notes and "chi2 under the declared noise" in notes
    assert fit.bias.kind == "bound" and len(kept) == 24
