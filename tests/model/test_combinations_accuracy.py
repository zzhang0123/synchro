"""Rounding accuracy, cluster covariance and determinism of ``fit_combinations`` (T-006).

``K C = Pi`` on the kept rows is an identity of the estimator; with the SVD's
own left singular vectors its rounding error is of order ``u cond`` (``u =
2^-53``, ``cond = s_max / s_min`` over the retained modes), which the NumPy
oracle (``_reduction_fixtures.oracle_fit``) attains. Rebuilding the left
vectors as ``G_w V_R / D`` squares ``cond``. The tests keep every numerically
ranked mode (``max_sigma = inf``), where ``cond`` is largest, and a retained
cluster whose members are split by less than ``cluster_rtol s_max``.
Pivot ties of the canonical cluster basis must not depend on rounding at the
level that separates two LAPACK builds (norms tied to about 1e-11).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from _reduction_fixtures import continuum_problem, matrix_basis, oracle_fit
from _reduction_fixtures import stokes_data
from syncmoments.model.assumptions import no_assumption
from syncmoments.model.fit import fit_combinations, fit_linear, reduce_response
from syncmoments.model.fit._combination_core import pivoted_basis
from syncmoments.model.fit._combination_errors import PI_RTOL
from syncmoments.model.index import MomentIndex, Truncation

U = 2.0**-53
C_ROUND = 16.0  # the oracle attains 0.3 to 0.7 u cond on these problems
INDEX = MomentIndex.build(Truncation(0, 0, 1), components=("I", "Q"))  # n_real 11


def _cond(fit):
    s = np.asarray(fit.singular_values)
    return float(s[0] / s[fit.n_retained - 1])


def _identity_error(fit, C_x, kept):
    K, Pi = np.asarray(fit.estimator), np.asarray(fit.projector)
    return float(np.max(np.abs(K @ C_x[kept] - Pi)))


@pytest.mark.parametrize("n_ch, dense", [(8, False), (12, True), (16, False)])
def test_estimator_identity_with_every_numerical_mode(n_ch, dense):
    basis, _, _, data = continuum_problem(n_ch=n_ch, dense=dense)
    fit = fit_combinations(basis, data, max_sigma=np.inf)
    assert fit.n_retained == fit.numerical_rank
    C = np.asarray(fit.reduction.C)
    kept = list(data.kept_rows())
    cond = _cond(fit)
    assert cond > 1e5, "the test needs an ill-conditioned retained set"
    assert _identity_error(fit, C, kept) <= C_ROUND * U * cond
    oracle = oracle_fit(C, data, max_sigma=np.inf, rank_tol=fit.rank_tol)
    assert oracle["n_R"] == fit.n_retained
    x_o = oracle["x_hat"]
    x_err = np.max(np.abs(np.asarray(fit.x_hat) - x_o)) / np.max(np.abs(x_o))
    assert x_err <= C_ROUND * U * cond
    np.testing.assert_allclose(np.asarray(fit.projector), oracle["Pi"], atol=1e-8)


def test_estimator_identity_agrees_with_fit_linear_at_equal_rank():
    basis, _, _, data = continuum_problem(n_ch=8)
    pm = no_assumption(basis.index)
    lin = fit_linear(basis, data, pm)
    fit = fit_combinations(basis, data, max_sigma=np.inf, parameter_map=pm)
    assert lin.rank == fit.n_retained == fit.numerical_rank
    C_x = np.asarray(fit.reduction.C) @ np.asarray(fit.jacobian)
    kept = list(data.kept_rows())
    cond = _cond(fit)
    assert _identity_error(fit, C_x, kept) <= C_ROUND * U * cond
    sigma = np.sqrt(np.asarray(data.noise)[kept])
    lin_w = np.asarray(lin.prediction.stokes).reshape(-1)[kept] / sigma
    fit_w = np.asarray(fit.stokes_hat).reshape(-1)[kept] / sigma
    assert np.max(np.abs(fit_w - lin_w)) <= 1e-10 * np.max(np.abs(lin_w))


def _split_pair(split, seed=4, n_ch=6):
    """Design with ``s_max = 1e6`` and a retained pair at ``10, 10 + split``."""
    rng = np.random.default_rng(seed)
    n = INDEX.n_real
    s = np.array([1e6, 1e4, 1e3, 10.0 + split, 10.0, 1.0, 0.1, 0, 0, 0, 0])
    Uo, _ = np.linalg.qr(rng.standard_normal((4 * n_ch, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    C = Uo @ np.diag(s) @ V.T
    data = stokes_data((C @ rng.standard_normal(n)).reshape(-1, 4), sigma=1.0)
    basis = matrix_basis(C, index=INDEX)
    return basis, data, reduce_response(basis, include_ext=False)


@pytest.mark.parametrize("split", [0.0, 5e-5, 9e-5])
def test_cluster_split_below_cluster_rtol_keeps_the_exact_estimator(split):
    basis, data, red = _split_pair(split)
    fit = fit_combinations(basis, data, max_sigma=0.2, reduction=red)
    assert (3, 4) in fit.clusters and fit.n_retained == 5
    C = np.asarray(red.C)
    kept = list(data.kept_rows())
    cond = _cond(fit)
    assert _identity_error(fit, C, kept) <= C_ROUND * U * cond
    oracle = oracle_fit(C, data, max_sigma=0.2, rank_tol=fit.rank_tol)
    K_o = oracle["K"]
    K_err = np.max(np.abs(np.asarray(fit.estimator) - K_o)) / np.max(np.abs(K_o))
    assert K_err <= C_ROUND * U * cond
    np.testing.assert_allclose(
        np.asarray(fit.covariance), oracle["Cov"], atol=1e-12, rtol=1e-9
    )


@pytest.mark.parametrize("split", [5e-5, 9e-5])
def test_cluster_beta_covariance_is_reported_and_bounded(split):
    basis, data, red = _split_pair(split)
    fit = fit_combinations(basis, data, max_sigma=0.2, reduction=red)
    K_beta = np.asarray(fit.beta_estimator)
    cov = K_beta @ K_beta.T  # unit noise on every kept row
    sd = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(np.asarray(fit.beta_sigma), sd, rtol=1e-12)
    corr = cov / np.outer(sd, sd)
    rho = (10.0 + split) / 10.0
    measured = abs(corr[3, 4])
    assert measured <= 0.5 * (rho**2 - 1.0) * (1 + 1e-6)
    assert measured > 0.1 * 0.5 * (rho**2 - 1.0)  # the split does correlate them
    notes = [n for n in fit.provenance.notes if "beta correlation" in n]
    assert notes and "[3, 4]" in notes[0]
    assert f"{measured:.2g}" in notes[0]
    limits = fit.to_dict()["limits"]
    assert not any("diagonal to cluster_rtol" in text for text in limits)
    assert any("(rho^2 - 1)/2" in text for text in limits)


# -- slot status and the unresolved entry (one threshold) ----------------------------


def _rank_deficient_fit():
    basis, _, _, data = continuum_problem()
    return fit_combinations(basis, data, max_sigma=0.1)


def test_slot_status_uses_the_unresolved_threshold():
    fit = _rank_deficient_fit()
    Pi = np.array(fit.projector)
    Pi[0] = 0.0
    Pi[0, 0] = 1.0
    Pi[1] = 0.0
    Pi[1, 1] = 1.0 + 0.5 * PI_RTOL
    Pi[2] = 0.0
    Pi[2, 2] = 1.0 + 50 * PI_RTOL  # between PI_RTOL and the former 1e-10
    edited = eqx.tree_at(lambda f: f.projector, fit, jnp.asarray(Pi))
    assert edited.slot_status()[:3] == ("resolved", "resolved", "mixed")


def test_resolved_slot_keeps_unbounded_unresolved_entry():
    fit = _rank_deficient_fit()
    assert fit.unresolved.kind == "unbounded"
    Pi = np.array(fit.projector)
    Pi[0] = 0.0
    Pi[0, 0] = 1.0
    edited = eqx.tree_at(lambda f: f.projector, fit, jnp.asarray(Pi))
    assert edited.slot_status()[0] == "resolved"
    slot = edited.to_dict()["slots"][0]
    assert slot["status"] == "resolved" and slot["unresolved"] == "unbounded"


def test_not_applicable_unresolved_has_no_slot_entry():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((4 * 5, INDEX.n_real))
    basis = matrix_basis(C, index=INDEX)
    data = stokes_data((C @ rng.standard_normal(INDEX.n_real)).reshape(-1, 4))
    fit = fit_combinations(
        basis, data, max_sigma=1e6, reduction=reduce_response(basis, include_ext=False)
    )
    assert fit.unresolved.kind == "not_applicable"
    assert all("unresolved" not in slot for slot in fit.to_dict()["slots"])


# -- canonical basis of a (near-)degenerate set ---------------------------------------


def _symmetric_projector(seed=1):
    """Projector onto ``span{f, P f}`` with ``P`` swapping slots 1<->2 and 3<->4."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal(6)
    f[1] = 3.0  # the largest columns are the tied pair 1, 2
    perm = [0, 2, 1, 4, 3, 5]
    Qb, _ = np.linalg.qr(np.column_stack([f, f[perm]]))
    P = Qb @ Qb.T
    norms = np.linalg.norm(P, axis=0)
    assert np.isclose(norms[1], norms[2], rtol=1e-14, atol=0)
    assert norms[1] == pytest.approx(norms.max(), rel=1e-14)
    return P


@pytest.mark.parametrize("size", [1e-11, 4e-11])
def test_pivot_near_tie_is_environment_independent(size):
    """Two builds that tie the pivot norms to ~1e-11 give the same basis."""
    P = _symmetric_projector()
    E = np.zeros_like(P)
    E[1, 1] = size  # raises column 1 above column 2 ...
    one = pivoted_basis(P + E, 2)
    two = pivoted_basis(P - E, 2)  # ... or column 2 above column 1
    np.testing.assert_allclose(one, two, atol=1e-9)
    np.testing.assert_allclose(one @ one.T, P, atol=1e-9)
