"""``fit_combinations`` against NumPy oracles under one declared metric (T-006).

The oracle (``_reduction_fixtures.oracle_fit``) whitens by an explicit
Cholesky factor, applies the response and the metric pull-back and takes an
SVD of the ungrouped design on the full ``C``. Grouped and ungrouped fits
must agree; an SVD of ``H`` alone must not.
"""

from __future__ import annotations

import inspect
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from _linear_fixtures import setup as linear_setup
from _reduction_fixtures import (
    MANUSCRIPT_REPRESENTATIVES,
    continuum_problem,
    oracle_fit,
    stokes_data,
)
from syncmoments.model.assumptions import ParameterMap, isotropic_pitch, no_assumption
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.fit import (
    CombinationFit,
    StokesData,
    fit_combinations,
    fit_linear,
    identifiability,
    reduce_response,
)
from syncmoments.model.fit import linear as linear_module
from syncmoments.model.fit._result_core import TOL

MAX_SIGMA = 0.1


def _close(a, b, rtol):
    a, b = np.asarray(a), np.asarray(b)
    scale = max(float(np.max(np.abs(b), initial=0.0)), 1e-300)
    assert a.shape == b.shape
    assert float(np.max(np.abs(a - b), initial=0.0)) <= rtol * scale


def _nonzero(fit):
    s = np.asarray(fit.singular_values)
    return s[s > fit.rank_tol * s[0]]


def test_grouped_equals_ungrouped_same_metric():
    basis, _, _, data = continuum_problem()
    grouped = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    flat = fit_combinations(
        basis,
        data,
        max_sigma=MAX_SIGMA,
        reduction=reduce_response(basis, relations="none"),
    )
    assert isinstance(grouped, CombinationFit)
    _close(_nonzero(flat), _nonzero(grouped), 1e-10)
    for name in ("estimator", "projector", "covariance", "beta_hat", "representative"):
        _close(getattr(flat, name), getattr(grouped, name), 1e-10)
    assert grouped.directions.analytic_null.shape[1] == 60 - 30
    assert flat.directions.analytic_null.shape[1] == 8
    assert (
        flat.directions.numerical_null.shape[1]
        == grouped.directions.numerical_null.shape[1] + 22
    )


def test_representative_choice_invariance():
    basis, _, _, data = continuum_problem()
    a = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    b = fit_combinations(
        basis,
        data,
        max_sigma=MAX_SIGMA,
        reduction=reduce_response(basis, representatives=MANUSCRIPT_REPRESENTATIVES),
    )
    for name in ("estimator", "projector", "covariance", "singular_values"):
        _close(getattr(b, name), getattr(a, name), 1e-12)
    Wa, Wb = np.asarray(a.directions.retained), np.asarray(b.directions.retained)
    _close(Wb @ Wb.T, Wa @ Wa.T, 1e-12)


def test_wrong_metric_detected():
    basis, _, _, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    red = fit.reduction
    oracle = oracle_fit(red.C, data, max_sigma=MAX_SIGMA)
    _close(_nonzero(fit), oracle["s"], 1e-10)
    for name, key in (("estimator", "K"), ("projector", "Pi"), ("covariance", "Cov")):
        _close(getattr(fit, name), oracle[key], 1e-9)
    kept = list(data.kept_rows())
    Hw = np.asarray(red.H)[kept] / np.sqrt(np.asarray(data.noise)[kept])[:, None]
    s_H = np.linalg.svd(Hw, compute_uv=False)
    top = min(len(s_H), len(oracle["s"]))
    assert np.max(np.abs(s_H[:top] / oracle["s"][:top] - 1.0)) > 0.01


def test_diagonal_metric_equals_rescaled_coordinates():
    basis, _, _, data = continuum_problem()
    weights = np.random.default_rng(3).uniform(0.2, 5.0, 60)
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA, metric=weights)
    oracle = oracle_fit(fit.reduction.C, data, M=np.diag(weights), max_sigma=MAX_SIGMA)
    _close(_nonzero(fit), oracle["s"], 1e-10)
    for name, key in (("projector", "Pi"), ("covariance", "Cov"), ("x_hat", "x_hat")):
        _close(getattr(fit, name), oracle[key], 1e-9)
    # rescaled coordinates a' = sqrt(w) a with the Euclidean metric
    C = np.asarray(fit.reduction.C) / np.sqrt(weights)[None, :]
    scaled = oracle_fit(C, data, max_sigma=MAX_SIGMA)
    _close(np.sqrt(weights) * np.asarray(fit.x_hat), scaled["x_hat"], 1e-9)
    euclid = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    assert not np.allclose(_nonzero(euclid), _nonzero(fit), rtol=1e-3)


def test_dense_metric_oracle():
    basis, _, _, data = continuum_problem()
    A = np.random.default_rng(5).standard_normal((60, 60))
    M = A @ A.T / 60 + np.eye(60)
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA, metric=M)
    oracle = oracle_fit(fit.reduction.C, data, M=M, max_sigma=MAX_SIGMA)
    _close(_nonzero(fit), oracle["s"], 1e-9)
    for name, key in (("projector", "Pi"), ("covariance", "Cov"), ("estimator", "K")):
        _close(getattr(fit, name), oracle[key], 1e-8)
    W = np.asarray(fit.directions.retained)
    np.testing.assert_allclose(
        W.T @ np.asarray(fit.metric) @ W, np.eye(W.shape[1]), atol=1e-10
    )


def test_nondiagonal_covariance():
    basis, _, _, data = continuum_problem(dense=True)
    assert np.asarray(data.noise).ndim == 2
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    oracle = oracle_fit(fit.reduction.C, data, max_sigma=MAX_SIGMA)
    _close(_nonzero(fit), oracle["s"], 1e-9)
    for name, key in (("estimator", "K"), ("projector", "Pi"), ("x_hat", "x_hat")):
        _close(getattr(fit, name), oracle[key], 1e-8)
    assert fit.noise_model == "covariance"


def _masked(values, mask):
    return stokes_data(values, sigma=1e-3 * np.max(np.abs(values)), mask=mask)


def test_mask_and_withheld_channels():
    basis, _, clean, data = continuum_problem()
    mask = np.ones((8, 4), dtype=bool)
    mask[:, 3] = False
    mask[[1, 4, 6]] = False
    d1 = _masked(clean, mask)
    fit = fit_combinations(basis, d1, max_sigma=MAX_SIGMA)
    oracle = oracle_fit(fit.reduction.C, d1, max_sigma=MAX_SIGMA)
    _close(fit.projector, oracle["Pi"], 1e-9)
    assert len(fit.kept_rows) == 15 and fit.stokes_hat.shape == (8, 4)
    assert fit.dof == 15 - fit.n_retained


def test_masked_rows_not_read():
    basis, _, clean, _ = continuum_problem()
    mask = np.ones((8, 4), dtype=bool)
    mask[:, 3] = False
    mask[[1, 4, 6]] = False
    changed = clean.copy()
    changed[~mask] = 1e3 * np.random.default_rng(0).standard_normal((~mask).sum())
    a = fit_combinations(basis, _masked(clean, mask), max_sigma=MAX_SIGMA)
    b = fit_combinations(
        basis,
        StokesData(changed, np.asarray(_masked(clean, mask).noise), mask=mask),
        max_sigma=MAX_SIGMA,
    )
    for name in ("x_hat", "beta_hat", "estimator", "singular_values", "stokes_hat"):
        np.testing.assert_array_equal(
            np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        )


@pytest.mark.parametrize("uncertainty", [0.0, 1e-3])
def test_observing_response(uncertainty):
    basis, _, clean, _ = continuum_problem()
    rng = np.random.default_rng(8)
    Rresp = rng.standard_normal((20, 32)) * np.tile([1, 1, 1, 0], 8)[None, :]
    d = Rresp @ clean.reshape(-1)
    data = StokesData(
        d,
        np.full(20, (1e-2 * np.abs(d).max()) ** 2),
        response=Rresp,
        response_uncertainty=np.full_like(Rresp, uncertainty),
        discrepancy=ErrorTerm(np.full(20, 1e-3 * np.abs(d).max()), "bound", "declared"),
    )
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    oracle = oracle_fit(fit.reduction.C, data, max_sigma=MAX_SIGMA)
    _close(_nonzero(fit), oracle["s"], 1e-9)
    _close(fit.estimator, oracle["K"], 1e-8)
    assert fit.bias.kind == ("bound" if uncertainty == 0.0 else "estimate")
    assert fit.response.shape == (20, 32)


def test_fixed_amplitude_offset():
    basis, a_true, clean, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA, amplitude=2.0)
    assert fit.labels == fit.reduction.layout.labels()[1:]
    offset = np.zeros(60)
    offset[0] = 2.0
    np.testing.assert_array_equal(np.asarray(fit.offset), offset)
    oracle = oracle_fit(
        fit.reduction.C, data, J=np.eye(60)[:, 1:], a_off=offset, max_sigma=MAX_SIGMA
    )
    _close(fit.x_hat, oracle["x_hat"], 1e-9)
    np.testing.assert_allclose(np.asarray(fit.representative)[0], 2.0)


@pytest.mark.parametrize("make", [isotropic_pitch, no_assumption])
@pytest.mark.parametrize("amplitude", ["fit", 1.5])
def test_affine_map_fitted_and_fixed_amplitude(make, amplitude):
    basis, _, _, data = continuum_problem()
    pm = make(basis.index)
    fit = fit_combinations(
        basis, data, max_sigma=MAX_SIGMA, parameter_map=pm, amplitude=amplitude
    )
    P, c = (np.asarray(v) for v in pm.affine_pieces(reference=basis.reference))
    if amplitude == "fit":
        J = np.vstack([np.column_stack([c, P]), np.zeros((8, P.shape[1] + 1))])
        assert fit.labels == ("amplitude",) + tuple(pm.labels())
        a_off = np.zeros(60)
    else:
        J = np.vstack([P, np.zeros((8, P.shape[1]))])
        assert fit.labels == tuple(pm.labels())
        a_off = 1.5 * np.concatenate([c, np.zeros(8)])
    np.testing.assert_array_equal(np.asarray(fit.jacobian), J)
    np.testing.assert_allclose(np.asarray(fit.offset), a_off)
    T = np.asarray(fit.reduction.T)
    assert fit.factor_Q.shape[0] == np.linalg.matrix_rank(T @ J)
    oracle = oracle_fit(fit.reduction.C, data, J=J, a_off=a_off, max_sigma=MAX_SIGMA)
    _close(fit.projector, oracle["Pi"], 1e-9)


def test_redundant_map_under_metric_raises():
    basis, _, _, data = continuum_problem()
    pm = no_assumption(basis.index)
    P, c = pm.affine_pieces(reference=basis.reference)
    P_dup = P.at[:, -1].set(P[:, 0])
    with mock.patch.object(
        ParameterMap, "affine_pieces", lambda self, reference=None: (P_dup, c)
    ):
        with pytest.raises(ValueError, match="redundant parameters"):
            fit_combinations(basis, data, max_sigma=MAX_SIGMA, parameter_map=pm)


def _polynomial_problem(seed=4):
    index, basis, C = linear_setup((1, 1, 1), n_ch=16, seed=3)
    pm = no_assumption(index)
    rng = np.random.default_rng(seed)
    m = np.asarray(
        pm(
            pm.unflatten(jnp.asarray(0.3 * rng.standard_normal(pm.n_free()))),
            basis.reference,
        ).to_vector()
    )
    clean = (2.0 * C @ m).reshape(-1, 4)
    sigma = 1e-3 * np.abs(clean).max()
    noisy = clean + sigma * rng.standard_normal(clean.shape)
    data = StokesData(
        noisy,
        np.full(clean.size, sigma**2),
        discrepancy=ErrorTerm(np.full((16, 4), 0.5 * sigma), "bound", "declared"),
    )
    return basis, pm, data


def test_full_rank_agrees_with_fit_linear():
    basis, pm, data = _polynomial_problem()
    lin = fit_linear(basis, data, pm)
    assert lin.rank == 28
    A = float(lin.amplitude)
    u = np.concatenate([[A], A * np.asarray(pm.flatten(lin.theta))])
    fit = fit_combinations(basis, data, max_sigma=np.inf, parameter_map=pm)
    assert fit.n_retained == 28 and fit.directions.analytic_null.shape[1] == 0
    _close(fit.x_hat, u, 1e-9)
    _close(fit.covariance, lin.covariance, 1e-8)


def test_rank_deficient_predictions_agree_with_fit_linear():
    """Equal rank decisions give equal predictions; the representatives differ.

    ``fit_linear`` ranks the column-equilibrated design and this path the
    metric design, so the two ranks can differ near ``rank_tol`` (16 channels
    here give 28 and 26); the comparison is made where they agree.
    """
    basis, _, _, data = continuum_problem(n_ch=8)
    pm = no_assumption(basis.index)
    lin = fit_linear(basis, data, pm)
    fit = fit_combinations(basis, data, max_sigma=np.inf, parameter_map=pm)
    assert lin.rank < 52 and lin.covariance is None
    assert lin.rank == fit.numerical_rank == fit.n_retained
    kept = list(data.kept_rows())
    sigma = np.sqrt(np.asarray(data.noise)[kept])
    lin_w = np.asarray(lin.prediction.stokes).reshape(-1)[kept] / sigma
    fit_w = np.asarray(fit.stokes_hat).reshape(-1)[kept] / sigma
    assert np.max(np.abs(fit_w - lin_w)) <= 1e-8 * np.max(np.abs(lin_w))
    assert abs(float(fit.chi2) - float(lin.chi2)) <= 1e-8 * float(lin.chi2)
    A = float(lin.amplitude)
    u = np.concatenate([[A], A * np.asarray(pm.flatten(lin.theta))])
    assert np.max(np.abs(np.asarray(fit.x_hat) - u)) > 1e-6 * np.max(np.abs(u))


def test_fit_linear_defaults_unchanged():
    def defaults(fn):
        return {
            k: p.default
            for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty
        }

    assert defaults(fit_linear) == {
        "amplitude": "fit",
        "discrepancy_policy": "bias_bound",
        "tol": 1e-8,
        "diagnostics": True,
    }
    assert defaults(linear_module.fisher) == {
        "amplitude": None,
        "coordinates": "natural",
    }
    assert defaults(identifiability)["tol"] == 1e-8
    assert TOL == 1e-8 and linear_module.POLICIES == ("bias_bound", "inflate")
    basis, _, _, data = continuum_problem()
    data = StokesData(
        data.stokes,
        data.noise,
        mask=data.mask,
        discrepancy=ErrorTerm(np.full((8, 4), 1e-30), "bound", "declared"),
    )
    lin = fit_linear(basis, data, no_assumption(basis.index))
    assert lin.covariance is None and lin.bias_bound.kind == "unbounded"


def test_invalid_fit_inputs():
    basis, _, _, data = continuum_problem()
    with pytest.raises(TypeError):
        fit_combinations(basis, data)
    for bad in (0.0, -1.0, float("nan"), "x"):
        with pytest.raises(ValueError, match="max_sigma"):
            fit_combinations(basis, data, max_sigma=bad)
    other, _, _, _ = continuum_problem(scales=(0.1, 0.3))
    with pytest.raises(ValueError, match="foreign"):
        fit_combinations(basis, data, max_sigma=1.0, reduction=reduce_response(other))
    from syncmoments.model.assumptions import independent_screen

    with pytest.raises(ValueError, match="fit_bfgs"):
        fit_combinations(
            basis, data, max_sigma=1.0, parameter_map=independent_screen(basis.index)
        )
    for metric in (
        np.ones(5),
        -np.ones(60),
        np.full(60, np.nan),
        np.triu(np.ones((60, 60))),
    ):
        with pytest.raises(ValueError, match="metric"):
            fit_combinations(basis, data, max_sigma=1.0, metric=metric)
    with pytest.raises(ValueError, match="inflate"):
        fit_combinations(basis, data, max_sigma=1.0, discrepancy_policy="inflate")
    with pytest.raises(ValueError, match="discrepancy_policy"):
        fit_combinations(basis, data, max_sigma=1.0, discrepancy_policy="x")
    small = stokes_data(np.ones((3, 4)))
    with pytest.raises(ValueError, match="channels"):
        fit_combinations(basis, small, max_sigma=1.0)
    for kw in ({"rank_tol": -1.0}, {"cluster_rtol": -1.0}, {"amplitude": -2.0}):
        with pytest.raises(ValueError):
            fit_combinations(basis, data, max_sigma=1.0, **kw)
    for bound in (ErrorTerm.unbounded("x"), ErrorTerm(np.ones(5), "bound", "x"), "x"):
        with pytest.raises(ValueError, match="coefficient_bound"):
            fit_combinations(basis, data, max_sigma=1.0, coefficient_bound=bound)
