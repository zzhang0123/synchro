"""Recovery, null perturbations and mode stability of ``fit_combinations`` (T-006).

A noise-free fit returns ``beta = B a`` and ``x_hat = Pi a``; perturbations
along analytic-null directions leave the SED (to rounding) and the fit
unchanged, and a weak perturbation changes the whitened SED by exactly
``s_weak |c|`` without moving ``beta_hat``. The retained modes are stable
under a quadrature refinement of the basis; the relation check and the
grouped/ungrouped agreement hold at the extreme tested scales.
"""

from __future__ import annotations

import numpy as np
import pytest

from _reduction_fixtures import continuum_problem, small_continuum_basis, stokes_data
from syncmoments.model.assumptions import isotropic_pitch
from syncmoments.model.fit import fit_combinations, reduce_response

MAX_SIGMA = 0.1


def _noise_free(basis, a_true, sigma_rel=1e-2):
    C = np.asarray(basis.response_matrix())
    clean = (C @ a_true[: C.shape[1]]).reshape(-1, 4)
    sigma = sigma_rel * np.max(np.abs(clean[:, 0]))
    return clean, stokes_data(clean, sigma=sigma, mask=np.array([1, 1, 1, 0]))


def test_noise_free_identifiable_recovery():
    basis, a_true, _, _ = continuum_problem()
    _, data = _noise_free(basis, a_true)
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    beta_true = np.asarray(fit.combination_rows) @ a_true
    scale = np.max(np.abs(beta_true))
    assert np.max(np.abs(np.asarray(fit.beta_hat) - beta_true)) <= 1e-10 * scale
    projected = np.asarray(fit.projector) @ a_true
    assert np.max(np.abs(np.asarray(fit.x_hat) - projected)) <= 1e-10 * np.max(
        np.abs(projected)
    )
    cmp = fit.against_truth(a_true)
    np.testing.assert_allclose(np.asarray(cmp.projected_truth), projected)
    np.testing.assert_allclose(
        np.asarray(cmp.unresolved_truth) + projected, a_true, atol=1e-12
    )


def test_null_perturbation_leaves_sed_unchanged():
    basis, a_true, _, _ = continuum_problem()
    clean, data = _noise_free(basis, a_true)
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    C = np.asarray(fit.reduction.C)
    base = C @ a_true
    for w in np.asarray(fit.directions.analytic_null).T[::5]:
        moved = C @ (a_true + 3.0 * w)
        assert np.max(np.abs(moved - base)) <= 1e-12 * np.max(np.abs(base))
        obs = stokes_data(
            moved.reshape(-1, 4), sigma=float(np.sqrt(data.noise[0])), mask=data.mask
        )
        alt = fit_combinations(basis, obs, max_sigma=MAX_SIGMA)
        np.testing.assert_allclose(
            np.asarray(alt.beta_hat), np.asarray(fit.beta_hat), rtol=1e-9, atol=1e-12
        )
    weak = np.asarray(fit.directions.weak)
    s = np.asarray(fit.singular_values)
    s_weak = s[[i for i, c in enumerate(fit.classes) if c == "weak"]]
    kept = list(data.kept_rows())
    sigma = np.sqrt(np.asarray(data.noise)[kept])
    HT = np.asarray(fit.reduction.H) @ np.asarray(fit.reduction.T)
    for i in (0, weak.shape[1] - 1):
        c = 0.7
        change = (HT @ (c * weak[:, i]))[kept] / sigma
        np.testing.assert_allclose(np.linalg.norm(change), s_weak[i] * c, rtol=1e-8)
        obs = stokes_data(
            (HT @ (a_true + c * weak[:, i])).reshape(-1, 4),
            sigma=float(np.sqrt(data.noise[0])),
            mask=data.mask,
        )
        alt = fit_combinations(basis, obs, max_sigma=MAX_SIGMA)
        np.testing.assert_allclose(
            np.asarray(alt.beta_hat),
            np.asarray(fit.beta_hat),
            atol=1e-9 * np.max(np.abs(np.asarray(fit.beta_hat))),
        )


def test_truth_outside_map_range_raises():
    basis, a_true, _, data = continuum_problem()
    fit = fit_combinations(
        basis, data, max_sigma=MAX_SIGMA, parameter_map=isotropic_pitch(basis.index)
    )
    with pytest.raises(ValueError, match="outside the range"):
        fit.against_truth(a_true)  # nonzero h0_ext slots: J has zero rows there
    with pytest.raises(ValueError):
        fit.against_truth(a_true[:10])


def _principal_angle(A, B):
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    cos = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.arccos(np.clip(cos.min(), -1.0, 1.0)))


def _stability(quad_a, quad_b):
    basis_a = small_continuum_basis(
        (0.2, 0.2), (0, 2, 2, None, None), "taylor", 8, quad_a
    )
    basis_b = small_continuum_basis(
        (0.2, 0.2), (0, 2, 2, None, None), "taylor", 8, quad_b
    )
    _, a_true, _, _ = continuum_problem()
    _, data = _noise_free(basis_a, a_true)
    fa = fit_combinations(basis_a, data, max_sigma=MAX_SIGMA)
    fb = fit_combinations(basis_b, data, max_sigma=MAX_SIGMA)
    np.testing.assert_array_equal(
        np.asarray(fa.reduction.T), np.asarray(fb.reduction.T)
    )
    assert fa.n_retained == fb.n_retained
    angle = _principal_angle(
        np.asarray(fa.directions.retained), np.asarray(fb.directions.retained)
    )
    change = np.abs(np.asarray(fa.beta_hat) - np.asarray(fb.beta_hat)) / np.asarray(
        fa.beta_sigma
    )
    return angle, float(change.max())


# The example quadrature (n_eta 64, n_nodes_F 128, n_nu 24) against its 1.5x and
# 2x refinements. The coarse test quadrature (16, 64, 8) is not converged enough
# for these tolerances (max|dC|/max|C| = 5e-4 against (24, 96, 12) gives a
# principal angle of 7.6e-3 and a beta_hat change of 0.37 sigma, measured
# 2026-09-26): mode stability is a property of the quadrature, not of the method.
def test_mode_stability_under_basis_refinement():
    angle, change = _stability((64, 128, 24), (96, 192, 32))
    assert angle < 1e-3 and change < 0.1


@pytest.mark.slow
def test_mode_stability_under_convergence_full():
    angle, change = _stability((64, 128, 24), (128, 256, 48))
    assert angle < 1e-3 and change < 0.1


@pytest.mark.parametrize("scales", [(0.02, 0.4), (0.4, 0.02), (0.02, 0.02), (0.4, 0.4)])
def test_scale_extremes_within_scope(scales):
    basis = small_continuum_basis(scales)
    grouped = reduce_response(basis)
    assert grouped.n_q == 30 and grouped.exact
    assert (
        max(
            r
            for r, rel in zip(grouped.relation_residuals, grouped.relations)
            if rel.source == "analytic"
        )
        <= 1e-10
    )
    C = np.asarray(basis.response_matrix())
    a = np.random.default_rng(0).standard_normal(60)
    clean = (C @ a[:52]).reshape(-1, 4)
    data = stokes_data(
        clean, sigma=1e-2 * np.abs(clean[:, 0]).max(), mask=np.array([1, 1, 1, 0])
    )
    g = fit_combinations(basis, data, max_sigma=MAX_SIGMA, reduction=grouped)
    u = fit_combinations(
        basis,
        data,
        max_sigma=MAX_SIGMA,
        reduction=reduce_response(basis, relations="none"),
    )
    sg, su = np.asarray(g.singular_values), np.asarray(u.singular_values)
    sg, su = sg[sg > 1e-8 * sg[0]], su[su > 1e-8 * su[0]]
    assert sg.shape == su.shape
    np.testing.assert_allclose(sg, su, rtol=1e-8)


def test_result_json_is_strict():
    import json

    basis, a_true, clean, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=MAX_SIGMA)
    out = json.loads(json.dumps(fit.to_dict(), allow_nan=False))
    claims = out["claims"]
    assert set(claims) == {"analytic", "numerical", "practical"}
    assert claims["analytic"]["null_dim"] == 30
    assert claims["numerical"]["rank"] == fit.numerical_rank
    assert claims["practical"]["n_retained"] == fit.n_retained
    assert out["limits"] and len(out["slot_status"]) == 60
    assert json.loads(json.dumps(fit.to_dict()))["max_sigma"] == MAX_SIGMA
    wide = fit_combinations(basis, data, max_sigma=np.inf)
    assert json.loads(json.dumps(wide.to_dict(), allow_nan=False))["max_sigma"] == "inf"
    obs = fit.observable(np.eye(60)[:2], on="full")
    json.dumps(obs.to_dict(), allow_nan=False)
    cmp = fit.against_truth(a_true, discrepancy=np.zeros((8, 4)), noise=np.zeros(32))
    json.dumps(cmp.to_dict(), allow_nan=False)
    assert fit.combination_labels()[0].startswith("beta_1 = ")
    np.testing.assert_allclose(
        fit.q_hat(), np.asarray(fit.reduction.T) @ fit.representative
    )
