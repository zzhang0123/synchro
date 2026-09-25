"""Tests of ``syncmoments.model.fit.linear`` (``fit_linear``, ``fisher``): recovery,
Fisher matrices, rank deficiency, the singular-value cutoff and validation.

Oracles: NumPy SVD least squares on the independently assembled whitened
design ``L^-1 R C [c | P]`` (``P, c`` from ``ParameterMap.affine_pieces``,
``C`` from the hand-written polynomial response of ``_nonlinear_oracles``),
``numpy.linalg.inv`` for the covariance and central differences for the
Fisher Jacobian of a nonlinear map. The policy, coverage and bias-bound
tests live in ``test_linear_policies.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _diagnostics_fixtures import random_population
from _linear_fixtures import oracle_design, setup, stokes_of, truth_of, with_amplitude
from syncmoments.model.assumptions import (
    ParameterMap,
    gaussian_screen,
    isotropic_pitch,
    no_assumption,
)
from syncmoments.model.fit.linear import fisher, fit_linear
from syncmoments.model.fit.nonlinear import Transform
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.fit.result import FitResult
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments

# -- zero-noise recovery on a full-rank configuration ------------------------------


def test_zero_noise_recovery_full_rank_isotropic_pitch_222():
    """isotropic_pitch at (2,2,2), 64 channels, all four Stokes: rank == n_free + 1.

    The truth is the projection of a correlated 60-atom population onto the
    map (a feasible tensor: the product of the rest-marginal with a uniform
    pitch distribution), so the feasibility report must pass as well."""
    index, basis, C = setup((2, 2, 2), n_ch=64, seed=5)
    pm = isotropic_pitch(index)
    joint = JointMoments.from_samples(
        random_population(np.random.default_rng(3), 60), index, basis.reference
    )
    theta, A = pm.project(joint), 2.3
    flat = np.asarray(pm.flatten(theta))
    d, m = stokes_of(C, pm, theta, A, basis.reference)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    result = fit_linear(basis, data, pm)
    assert isinstance(result, FitResult)
    assert result.rank == pm.n_free() + 1 == 58
    assert result.covariance is not None and result.fisher.shape == (58, 58)
    assert_allclose(float(result.amplitude), A, rtol=1e-10)
    assert_allclose(np.asarray(pm.flatten(result.theta)), flat, atol=1e-10)
    assert_allclose(float(result.theta.log_amplitude), np.log(A), atol=1e-10)
    assert_allclose(np.asarray(result.moments.to_vector()), m, atol=1e-10)
    assert float(result.chi2) < 1e-14
    assert result.converged and result.n_iter == 0
    assert result.dof == 256 - 58
    assert result.labels == ("amplitude",) + pm.labels()
    assert result.method == "linear"
    assert result.identifiability.null_dim == 0
    assert result.identifiability.rank == result.rank
    assert result.feasibility is not None and result.feasibility.failed() == ()
    assert_allclose(
        np.asarray(result.prediction.stokes).ravel(),
        d,
        rtol=0,
        atol=1e-10 * abs(d).max(),
    )
    assert result.bias_bound.kind == "unbounded"
    assert pm.record() in result.provenance.assumptions
    # immutability: the inputs are untouched and the result is an eqx.Module
    assert_allclose(np.asarray(data.stokes).ravel(), d)


# -- Fisher ------------------------------------------------------------------------


@pytest.mark.parametrize("noise", ["variances", "covariance"])
def test_fisher_equals_whitened_normal_matrix(noise):
    index, basis, C = setup((1, 1, 1), n_ch=8, seed=7)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 2)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    rng = np.random.default_rng(9)
    if noise == "variances":
        cov = rng.uniform(0.5, 2.0, d.size)
        Sigma = np.diag(cov)
    else:
        M = rng.standard_normal((d.size, d.size))
        cov = M @ M.T / d.size + np.eye(d.size)
        Sigma = cov
    data = StokesData(d.reshape(-1, 4), cov)
    result = fit_linear(basis, data, pm)
    G, _, _, _ = oracle_design(C, pm, basis.reference, data)
    P, c = (np.asarray(x) for x in pm.affine_pieces())
    Gu = C @ np.column_stack([c, P])
    F_u = Gu.T @ np.linalg.solve(Sigma, Gu)
    assert_allclose(np.asarray(result.fisher), F_u, rtol=1e-10, atol=1e-10)
    assert_allclose(np.asarray(result.fisher), G.T @ G, rtol=1e-10, atol=1e-10)
    cov = np.linalg.inv(F_u)
    assert_allclose(
        np.asarray(result.covariance), cov, rtol=1e-8, atol=1e-12 * np.abs(cov).max()
    )
    # fisher(): natural coordinates x = (A, theta) are related by u = (A, A theta)
    theta_A = with_amplitude(theta, A)
    F_x = np.asarray(fisher(basis, data, pm, theta_A))
    T = np.zeros((flat.size + 1, flat.size + 1))
    T[0, 0] = 1.0
    T[1:, 0] = flat
    T[1:, 1:] = A * np.eye(flat.size)
    assert_allclose(F_x, T.T @ F_u @ T, rtol=1e-9, atol=1e-9)
    F_lin = np.asarray(fisher(basis, data, pm, theta_A, coordinates="linear"))
    assert_allclose(F_lin, F_u, rtol=1e-9, atol=1e-9)
    # fixed amplitude: coordinates theta only
    F_fix = np.asarray(fisher(basis, data, pm, theta, amplitude=A))
    assert_allclose(
        F_fix, A**2 * (P.T @ C.T @ np.linalg.solve(Sigma, C @ P)), rtol=1e-9
    )
    # jit through the Parameters pytree
    jitted = jax.jit(lambda th: fisher(basis, data, pm, th))(theta_A)
    assert_allclose(np.asarray(jitted), F_x, rtol=1e-12)


def test_fisher_nonaffine_matches_finite_differences():
    index, basis, C = setup((1, 1, 2), n_ch=10, seed=8)
    pm = gaussian_screen(index, 0.5, 0.8, fit_hyper=True)
    assert not pm.is_affine()
    rng = np.random.default_rng(4)
    x = 0.3 * rng.standard_normal(pm.n_free() + 1)
    x[0] = 1.7  # amplitude
    x[-1] = 0.9  # sigma > 0
    theta = with_amplitude(pm.unflatten(jnp.asarray(x[1:])), x[0])
    d = rng.standard_normal(4 * 10)
    var = rng.uniform(0.5, 2.0, d.size)
    data = StokesData(d.reshape(-1, 4), var)

    def model(v):
        m = np.asarray(
            pm(pm.unflatten(jnp.asarray(v[1:])), basis.reference).to_vector()
        )
        return v[0] * (C @ m) / np.sqrt(var)

    h = 1e-5
    J = np.zeros((d.size, x.size))
    for a in range(x.size):
        e = np.zeros_like(x)
        e[a] = h
        J[:, a] = (model(x + e) - model(x - e)) / (2 * h)
    F = np.asarray(fisher(basis, data, pm, theta))
    assert_allclose(F, J.T @ J, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):
        fit_linear(basis, data, pm)


# -- rank deficiency and the null space --------------------------------------------


def test_rank_deficient_reports_none_covariance_and_null_vectors():
    index, basis, C = setup((1, 1, 1), n_ch=3, seed=5)
    pm = no_assumption(index)
    rng = np.random.default_rng(2)
    d = rng.standard_normal(12)
    data = StokesData(d.reshape(-1, 4), np.full(12, 0.1))
    result = fit_linear(basis, data, pm)
    assert result.rank == 12 < pm.n_free() + 1 == 28
    assert result.covariance is None
    assert result.fisher.shape == (28, 28)
    assert result.identifiability.null_dim == 16
    assert result.identifiability.rank == result.rank
    assert result.dof == 0
    # minimum equilibrated-norm solution reproduces the data exactly (12 rows,
    # rank 12); the columns are scaled to unit norm first (review finding R09)
    assert float(result.chi2) < 1e-18
    P, c = (np.asarray(x) for x in pm.affine_pieces())
    A = float(result.amplitude)
    u = np.concatenate([[A], A * np.asarray(pm.flatten(result.theta))])
    G, dw, _, cols = oracle_design(C, pm, basis.reference, data)
    scales = 1.0 / np.linalg.norm(G, axis=0)
    v_ref, *_ = np.linalg.lstsq(G * scales, dw, rcond=1e-8)
    assert_allclose(u, scales * v_ref, atol=1e-9)
    D = np.asarray(result.identifiability.column_scales)
    assert_allclose(D, scales, rtol=1e-12)
    for j in range(3):
        v = D * np.asarray(result.identifiability.null_basis)[:, j]
        u2 = u + 0.5 * v / np.abs(v).max()
        pred = u[0] * C @ (P @ (u[1:] / u[0]) + c)
        pred2 = u2[0] * C @ (P @ (u2[1:] / u2[0]) + c)
        assert_allclose(pred2, pred, atol=1e-9 * np.abs(pred).max())
        names = result.identifiability.null_components(j)
        assert names and all(n in result.identifiability.labels for n, _ in names)
    assert result.identifiability.labels == result.labels
    # the bias bound and the statistical input are still reported (min-norm estimator)
    assert result.bias_bound.kind == "unbounded"
    notes = " ".join(result.provenance.notes)
    assert "statistical_input of the prediction unbounded" in notes
    # the random data fix no sign of u[0]; the equilibrated minimum-norm
    # solution has u[0] < 0 here, so no prediction is formed (noted)
    if float(result.amplitude) > 0.0:
        assert result.prediction.budget.statistical_input.kind == "unbounded"
    else:
        assert result.prediction is None and "not positive" in notes


def test_tol_dispatch_both_sides():
    index, basis, C = setup((1, 1, 1), n_ch=8, seed=13)
    pm = isotropic_pitch(index)
    flat, theta, A = truth_of(pm, 3)
    d, _ = stokes_of(C, pm, theta, A, basis.reference)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-2))
    G, _, _, _ = oracle_design(C, pm, basis.reference, data)
    G = G / np.linalg.norm(G, axis=0)  # the cutoff acts on the equilibrated design
    s = np.linalg.svd(G, compute_uv=False)
    ratio = s[-1] / s[0]
    # both sides of the cutoff sigma <= tol sigma_max (1e-6 relative margins
    # keep the NumPy and jax singular values on the same side)
    # (the smallest singular value of this design is doubly degenerate, so the
    # rank drops by the multiplicity, which the oracle count reproduces)
    tols = (ratio * (1 - 1e-6), ratio * (1 + 1e-6), np.sqrt(s[-1] * s[-3]) / s[0])
    full, at, above = (
        fit_linear(basis, data, pm, tol=t, diagnostics=False) for t in tols
    )
    expected = [int(np.sum(s > t * s[0])) for t in tols]
    assert expected[0] == s.size and expected[1] < s.size and expected[2] < s.size
    assert full.rank == expected[0] and full.covariance is not None
    assert at.rank == expected[1] and at.covariance is None
    assert above.rank == expected[2] and above.covariance is None
    assert_allclose(float(full.amplitude), A, rtol=1e-9)
    assert float(above.chi2) > float(full.chi2)


# -- validation --------------------------------------------------------------------------


def test_fit_linear_rejects_bad_inputs():
    index, basis, C = setup((1, 1, 1), n_ch=4, seed=1)
    pm = isotropic_pitch(index)
    d = np.zeros(16)
    data = StokesData(d.reshape(-1, 4), np.ones(16))
    with pytest.raises(ValueError, match="fit_bfgs"):
        fit_linear(basis, data, gaussian_screen(index, 0.1, 0.5, fit_hyper=True))
    with pytest.raises(ValueError):
        fit_linear(basis, data, object())
    other = isotropic_pitch(MomentIndex.build(Truncation(1, 1, 2)))
    with pytest.raises(ValueError, match="index"):
        fit_linear(basis, data, other)
    with pytest.raises(ValueError, match="discrepancy_policy"):
        fit_linear(basis, data, pm, discrepancy_policy="bogus")
    for bad in ("bogus", 0.0, -1.0, np.array([1.0, 2.0])):
        with pytest.raises(ValueError, match="amplitude"):
            fit_linear(basis, data, pm, amplitude=bad)
    with pytest.raises(ValueError, match="tol"):
        fit_linear(basis, data, pm, tol=-1.0)
    with pytest.raises(ValueError, match="channels"):
        fit_linear(basis, StokesData(np.zeros((3, 4)), np.ones(12)), pm)
    assert isinstance(no_assumption(index), ParameterMap)
    Transform(pm)  # the affine map also has a transform; nothing else to check


def test_fisher_rejects_bad_inputs():
    index, basis, C = setup((1, 1, 1), n_ch=4, seed=1)
    pm = isotropic_pitch(index)
    data = StokesData(np.zeros((4, 4)), np.ones(16))
    theta = pm.unflatten(jnp.zeros(pm.n_free()))
    with pytest.raises(ValueError, match="log_amplitude"):
        fisher(basis, data, pm, theta)
    with pytest.raises(ValueError, match="coordinates"):
        fisher(basis, data, pm, with_amplitude(theta, 1.0), coordinates="polar")
    with pytest.raises(ValueError):
        fisher(basis, data, pm, with_amplitude(theta, 1.0), amplitude=2.0)
    with pytest.raises(ValueError):
        fisher(
            basis, data, isotropic_pitch(MomentIndex.build(Truncation(1, 1, 2))), theta
        )
