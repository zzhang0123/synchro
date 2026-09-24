"""Tests of ``synchro.model.fit.linear``: coverage, discrepancy policies, the
bias bound, fixed amplitude with mask and response, and agreement with ``fit_bfgs``.

Oracles: exact Gaussian sampling for the coverage statistic (the estimator
is linear in the data), ``numpy.linalg.pinv`` for the estimator matrix
``K = G^+ L^-1`` of the bias bound, NumPy least squares with mask and
response, and the BFGS optimum of ``fit_bfgs`` on the same data.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _diagnostics_fixtures import random_population
from _linear_fixtures import oracle_design, setup, stokes_of, truth_of
from synchro.model.assumptions import isotropic_pitch
from synchro.model.errors import ErrorTerm
from synchro.model.fit.linear import POLICIES, fit_linear
from synchro.model.fit.nonlinear import LogDensity, fit_bfgs
from synchro.model.fit.observation import StokesData
from synchro.model.fit.result import FitResult
from synchro.model.moments import JointMoments

# -- coverage ------------------------------------------------------------------------


def test_three_sigma_coverage_over_200_seeds():
    """Coverage statistic (documented): isotropic_pitch at (1,1,1), 16 channels,
    u = (A, A theta) with 14 unknowns, Gaussian noise of variance 0.05^2 per row,
    200 fixed seeds. For every seed the estimator is linear in the data, so
    ``u_hat - u_true`` is exactly Gaussian with covariance ``(G^T Sigma^-1 G)^-1``.
    Asserted: (a) pooled over the 200 x 14 standardised errors, the fraction
    inside 3 sigma is at least 0.995 (expected 0.9973, 2800 draws); (b) for the
    amplitude alone, at least 0.99 of the seeds lie inside 3 sigma (expected
    0.9973, 200 draws); (c) the mean Mahalanobis distance ``(u_hat-u)^T F
    (u_hat-u)`` lies within 4 standard errors of its chi-square mean 14."""
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 11)
    d0, _ = stokes_of(C, pm, theta, A, basis.reference)
    sigma = 0.05
    u_true = np.concatenate([[A], A * flat])
    inside, inside_amp, maha = [], [], []
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        d = d0 + sigma * rng.standard_normal(d0.size)
        data = StokesData(d.reshape(-1, 4), np.full(d.size, sigma**2))
        result = fit_linear(basis, data, pm, diagnostics=False)
        assert result.covariance is not None and result.identifiability is None
        u_hat = np.concatenate(
            [
                [float(result.amplitude)],
                float(result.amplitude) * np.asarray(pm.flatten(result.theta)),
            ]
        )
        err = u_hat - u_true
        std = np.sqrt(np.diag(np.asarray(result.covariance)))
        inside.append(np.abs(err) <= 3 * std)
        inside_amp.append(abs(err[0]) <= 3 * std[0])
        maha.append(err @ np.asarray(result.fisher) @ err)
    n_u = u_true.size
    assert n_u == 14
    assert np.mean(inside) >= 0.995
    assert np.mean(inside_amp) >= 0.99
    assert abs(np.mean(maha) - n_u) <= 4 * np.sqrt(2 * n_u / 200)


# -- bias bound --------------------------------------------------------------------


@pytest.mark.parametrize("policy", POLICIES)
def test_bias_bound_contains_bias_from_violating_population(policy):
    """Data from a correlated population (isotropic pitch violated) fitted with
    ``isotropic_pitch``; ``delta = A C (m_joint - m_iso)`` is declared as the
    Stokes-space discrepancy. Without noise the fitted ``u`` differs from the
    projected truth by exactly ``K delta``, which the bound ``|K| |delta|`` contains."""
    index, basis, C = setup((1, 1, 1), n_ch=6, seed=21)
    pm = isotropic_pitch(index)
    rng = np.random.default_rng(31)
    pop = random_population(rng, 40)
    joint = JointMoments.from_samples(pop, index, basis.reference)
    theta_proj = pm.project(joint)
    m_iso = np.asarray(pm(theta_proj, basis.reference).to_vector())
    m_joint = np.asarray(joint.to_vector())
    A = 1.7
    assert np.abs(m_joint - m_iso).max() > 1e-3
    d = A * C @ m_joint
    delta_true = A * C @ (m_joint - m_iso)
    term = ErrorTerm(np.abs(delta_true).reshape(-1, 4), "bound", "declared |delta|")
    var = rng.uniform(0.5, 2.0, d.size)
    data = StokesData(d.reshape(-1, 4), var, discrepancy=term)
    result = fit_linear(basis, data, pm, discrepancy_policy=policy)
    flat_proj = np.asarray(pm.flatten(theta_proj))
    u_true = np.concatenate([[A], A * flat_proj])
    u_hat = np.concatenate(
        [
            [float(result.amplitude)],
            float(result.amplitude) * np.asarray(pm.flatten(result.theta)),
        ]
    )
    bias = np.abs(u_hat - u_true)
    bound = np.asarray(result.bias_bound.value)
    assert result.bias_bound.kind == "bound"
    assert bound.shape == u_true.shape
    assert np.all(bias <= bound * (1 + 1e-9) + 1e-12)
    assert np.any(bias > 0.1 * bound)  # the bound is not vacuous
    # oracle: K = G^+ L^-1 with the policy's whitening; the bound is |K| delta
    if policy == "inflate":
        var_fit = var + delta_true**2
        assert "heuristic" in " ".join(result.provenance.notes)
    else:
        var_fit = var
    P, c = (np.asarray(x) for x in pm.affine_pieces())
    G = np.diag(1 / np.sqrt(var_fit)) @ C @ np.column_stack([c, P])
    K = np.linalg.pinv(G) @ np.diag(1 / np.sqrt(var_fit))
    assert_allclose(bound, np.abs(K) @ np.abs(delta_true), rtol=1e-8)
    assert_allclose(np.asarray(result.fisher), G.T @ G, rtol=1e-9, atol=1e-9)
    assert_allclose(u_hat - u_true, K @ delta_true, atol=1e-9)


def test_inflate_requires_discrepancy_and_measured_kind_propagates():
    index, basis, C = setup((1, 1, 1), n_ch=5, seed=2)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 5)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-2))
    with pytest.raises(ValueError, match="discrepancy"):
        fit_linear(basis, data, pm, discrepancy_policy="inflate")
    measured = ErrorTerm(jnp.full((5, 4), 1e-3), "measured", "measured")
    data_m = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-2), discrepancy=measured)
    result = fit_linear(basis, data_m, pm, diagnostics=False)
    assert result.bias_bound.kind == "measured"
    unbounded = ErrorTerm.unbounded("no information")
    data_u = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-2), discrepancy=unbounded)
    assert (
        fit_linear(basis, data_u, pm, diagnostics=False).bias_bound.kind == "unbounded"
    )
    with pytest.raises(ValueError, match="discrepancy"):
        fit_linear(basis, data_u, pm, discrepancy_policy="inflate")


# -- fixed amplitude, mask and response ----------------------------------------------


def test_fixed_amplitude_and_masked_response_match_numpy_lstsq():
    index, basis, C = setup((1, 1, 1), n_ch=6, seed=17)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 6)
    rng = np.random.default_rng(8)
    R = rng.standard_normal((20, 24))
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    d = R @ d + 0.01 * rng.standard_normal(20)
    mask = np.ones(20, dtype=bool)
    mask[[3, 11]] = False
    data = StokesData(d, rng.uniform(0.5, 1.5, 20), mask=mask, response=R)
    fixed = fit_linear(basis, data, pm, amplitude=A)
    assert fixed.labels == pm.labels() and fixed.rank == pm.n_free()
    assert float(fixed.amplitude) == A and float(fixed.theta.log_amplitude) == np.log(A)
    G, dw, _, cols = oracle_design(C, pm, basis.reference, data, amplitude=False)
    c_w = G @ np.zeros(G.shape[1])  # placeholder to keep shapes explicit
    P, c = (np.asarray(x) for x in pm.affine_pieces())
    kept = list(data.kept_rows())
    L_inv = np.diag(1 / np.sqrt(np.asarray(data.noise)[kept]))
    offset = L_inv @ (R @ C @ c)[kept] * A
    theta_ref, *_ = np.linalg.lstsq(A * G, dw - offset - c_w, rcond=None)
    assert_allclose(np.asarray(pm.flatten(fixed.theta)), theta_ref, atol=1e-9)
    assert_allclose(np.asarray(fixed.fisher), A**2 * G.T @ G, rtol=1e-9, atol=1e-9)
    assert fixed.dof == 18 - pm.n_free()
    fitted = fit_linear(basis, data, pm)
    G2, dw2, _, _ = oracle_design(C, pm, basis.reference, data)
    u_ref, *_ = np.linalg.lstsq(G2, dw2, rcond=None)
    assert_allclose(float(fitted.amplitude), u_ref[0], rtol=1e-9)
    assert_allclose(
        np.asarray(pm.flatten(fitted.theta)), u_ref[1:] / u_ref[0], atol=1e-9
    )
    r = dw2 - G2 @ u_ref
    assert_allclose(float(fitted.chi2), r @ r, rtol=1e-8)


# -- agreement with fit_bfgs ------------------------------------------------------------


def test_fit_linear_matches_fit_bfgs_on_affine_map():
    """Zero noise: both reach the truth (agreement to 1e-8). With noise the BFGS
    optimum agrees with the closed-form solution to 1e-6 (the optimiser's
    gradient tolerance, not the linear algebra, sets this level)."""
    index, basis, C = setup((1, 1, 1), n_ch=10, seed=21)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 1, amplitude=1.4)
    d0, _ = stokes_of(C, pm, theta, A, basis.reference)
    rng = np.random.default_rng(2)
    for noise, atol in ((0.0, 1e-8), (0.05, 1e-6)):
        d = d0 + noise * rng.standard_normal(d0.size)
        data = StokesData(d.reshape(-1, 4), np.full(d.size, 0.05**2))
        linear = fit_linear(basis, data, pm)
        density = LogDensity(basis, data, pm)
        z0 = jnp.concatenate([jnp.log(jnp.asarray([A])), jnp.asarray(flat)])
        z0 = z0 + 0.1 * jnp.asarray(rng.standard_normal(z0.size))
        bfgs = fit_bfgs(density, z0, maxiter=3000, tol=1e-10)
        assert isinstance(bfgs, FitResult) and bfgs.converged
        assert_allclose(float(bfgs.amplitude), float(linear.amplitude), rtol=atol)
        assert_allclose(
            np.asarray(pm.flatten(bfgs.theta)),
            np.asarray(pm.flatten(linear.theta)),
            atol=atol,
        )
        assert_allclose(float(bfgs.chi2), float(linear.chi2), atol=1e-6, rtol=1e-6)
        assert_allclose(
            np.asarray(bfgs.moments.to_vector()),
            np.asarray(linear.moments.to_vector()),
            atol=atol,
        )
