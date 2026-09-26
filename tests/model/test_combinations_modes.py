"""Mode thresholds, clusters and signs of ``fit_combinations`` (boundary rule, T-006).

Every threshold is tested on both sides, by the dispatcher function itself
(``_combination_core.classify`` / ``find_clusters`` on exact numbers) and
through the full fit on a design whose singular values are constructed
(``designed``), including extreme ``max_sigma``. Adjacent selections just
above and below a cutoff are compared directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from _reduction_fixtures import continuum_problem, matrix_basis, stokes_data
from syncmoments.model.fit import fit_combinations, reduce_response
from syncmoments.model.fit._combination_core import classify, find_clusters
from syncmoments.model.index import MomentIndex, Truncation

INDEX = MomentIndex.build(Truncation(0, 0, 1), components=("I", "Q"))  # n_real 11


def designed(s, seed=0, n_ch=5):
    """``(basis, data)`` whose whitened design (unit noise, all rows kept) has singular values ``s``."""
    rng = np.random.default_rng(seed)
    n = INDEX.n_real
    s = np.concatenate([np.asarray(s, dtype=float), np.zeros(n - len(s))])
    U, _ = np.linalg.qr(rng.standard_normal((4 * n_ch, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    C = U @ np.diag(s) @ V.T
    data = stokes_data((C @ rng.standard_normal(n)).reshape(-1, 4), sigma=1.0)
    return matrix_basis(C, index=INDEX), data


def _fit(s, **kw):
    basis, data = designed(s)
    red = reduce_response(basis, include_ext=kw.pop("include_ext", True))
    return fit_combinations(basis, data, reduction=red, **kw)


@pytest.mark.parametrize("max_sigma", [1e-6, 0.1, 1e6])
def test_threshold_boundary_inclusive(max_sigma):
    t = float(1.0 / max_sigma)
    s = np.array([100 * t, t * (1 + 1e-9), t, t * (1 - 1e-9), 1e-3 * t])
    got = classify(s, (), retain_at=t, null_below=1e-8 * s[0], notes=[])
    assert got == ("retained", "retained", "retained", "weak", "weak")
    fit = _fit(
        [100 * t, 10 * t, t * (1 + 1e-9), t * (1 - 1e-9), 0.1 * t],
        max_sigma=max_sigma,
        cluster_rtol=0.0,
    )
    assert fit.classes[:5] == ("retained",) * 3 + ("weak",) * 2
    assert fit.n_retained == 3 and fit.numerical_rank == 5


def test_adjacent_selections_agree_on_common_modes():
    s = [50.0, 20.0, 8.0, 3.0, 1.0, 0.3]
    k = 3
    above = _fit(s, max_sigma=(1 / s[k]) * (1 + 1e-6))
    below = _fit(s, max_sigma=(1 / s[k]) * (1 - 1e-6))
    assert above.n_retained == k + 1 and below.n_retained == k
    for name in ("beta_hat", "beta_sigma", "combination_rows"):
        np.testing.assert_allclose(
            np.asarray(getattr(above, name))[:k],
            np.asarray(getattr(below, name)),
            atol=1e-12,
        )
    w = np.asarray(above.directions.retained)[:, k]
    b = np.asarray(above.combination_rows)[k]
    np.testing.assert_allclose(
        np.asarray(above.projector) - np.asarray(below.projector),
        np.outer(w, b),
        atol=1e-12,
    )


def test_rank_tol_boundary():
    s = np.array([1.0, 0.1, 1e-5, 1e-9])
    at = classify(s, (), retain_at=np.inf, null_below=1e-5 * s[0], notes=[])
    assert at[2] == "numerical_null"  # strict: s = rank_tol s_max is not counted
    for rank_tol, cls in (
        (1e-5 * (1 - 1e-6), "weak"),
        (1e-5 * (1 + 1e-6), "numerical_null"),
    ):
        assert (
            classify(s, (), retain_at=np.inf, null_below=rank_tol, notes=[])[2] == cls
        )
        fit = _fit([1.0, 0.1, 1e-5, 1e-9], max_sigma=1e3, rank_tol=rank_tol)
        assert fit.classes[2] == cls
        assert fit.numerical_rank == (3 if cls == "weak" else 2)


def test_cluster_rtol_boundary():
    s = np.array([1.0, 0.5, 0.5 - 1e-6, 0.1])
    gap = s[1] - s[2]
    assert find_clusters(s, gap / s[0]) == ((1, 2),)
    assert find_clusters(s, gap / s[0] * (1 - 1e-9)) == ()
    assert find_clusters(np.array([1.0, 0.0, 0.0]), 1.0) == ()  # zeros never cluster
    assert find_clusters(np.array([1.0, 1.0, 0.0]), 0.0) == ((0, 1),)
    # through the fit; rounding-level singular values of the padded design may
    # form their own (null) cluster, so only clusters touching modes 1, 2 are checked
    fit = _fit(
        [1.0, 0.5, 0.5 - 1e-6, 0.1], max_sigma=1e3, cluster_rtol=gap * (1 + 1e-6)
    )
    assert (1, 2) in fit.clusters
    fit = _fit(
        [1.0, 0.5, 0.5 - 1e-6, 0.1], max_sigma=1e3, cluster_rtol=gap * (1 - 1e-3)
    )
    assert not any({1, 2} & set(c) for c in fit.clusters)


def test_extreme_thresholds():
    basis, _, _, data = continuum_problem()
    wide = fit_combinations(basis, data, max_sigma=np.inf)
    assert wide.n_retained == wide.numerical_rank
    none = fit_combinations(basis, data, max_sigma=1e-300)
    assert none.n_retained == 0 and none.beta_hat.shape == (0,)
    assert np.all(np.asarray(none.estimator) == 0.0)
    assert np.all(np.asarray(none.projector) == 0.0)
    assert none.unresolved.kind == "unbounded"
    assert any("no mode meets max_sigma" in n for n in none.provenance.notes)


def test_zero_design_raises():
    basis = matrix_basis(np.zeros((8, 52)))
    with pytest.raises(ValueError, match="nonzero singular value"):
        fit_combinations(basis, stokes_data(np.zeros((2, 4))), max_sigma=1.0)


def test_full_rank_all_retained():
    fit = _fit(np.linspace(20.0, 2.0, 11), max_sigma=1.0, include_ext=False)
    assert fit.n_retained == 11 and fit.directions.analytic_null.shape[1] == 0
    np.testing.assert_allclose(np.asarray(fit.projector), np.eye(11), atol=1e-12)
    assert fit.unresolved.kind == "not_applicable"
    assert set(fit.slot_status()) == {"resolved"}


def test_classification_partition():
    basis, _, _, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=0.1)
    d = fit.directions
    blocks = [
        np.asarray(x) for x in (d.retained, d.weak, d.numerical_null, d.analytic_null)
    ]
    assert sum(b.shape[1] for b in blocks) == 60
    assert blocks[0].shape[1] == fit.n_retained
    assert blocks[0].shape[1] + blocks[1].shape[1] == fit.numerical_rank
    allw = np.hstack(blocks)
    Mx = np.asarray(fit.metric)
    np.testing.assert_allclose(allw.T @ Mx @ allw, np.eye(60), atol=1e-10)
    C = np.asarray(fit.reduction.C)
    J = np.asarray(fit.jacobian)
    residual = C @ J @ blocks[3]
    assert np.max(np.abs(residual)) <= 1e-12 * np.max(np.abs(C))
    assert set(fit.classes) <= {"retained", "weak", "numerical_null"}


def _rotated(data, chi, perm=None):
    n_ch = data.n_ch
    R = np.zeros((4 * n_ch, 4 * n_ch))
    order = np.arange(n_ch) if perm is None else perm
    for j, src in enumerate(order):
        c, s = np.cos(chi[src]), np.sin(chi[src])
        R[4 * j, 4 * src] = 1.0
        R[4 * j + 1 : 4 * j + 3, 4 * src + 1 : 4 * src + 3] = [[c, -s], [s, c]]
        R[4 * j + 3, 4 * src + 3] = 1.0
    stokes = R @ np.asarray(data.stokes).reshape(-1)
    mask = np.asarray(data.mask).reshape(-1) if data.mask is not None else None
    noise = np.asarray(data.noise)
    return stokes_data(stokes, sigma=float(np.sqrt(noise[0])), mask=mask, response=R)


def test_repeated_singular_values_subspace():
    basis, _, _, data = continuum_problem()
    fit = fit_combinations(basis, data, max_sigma=0.1)
    assert fit.clusters, "the Q/U symmetry gives repeated singular values"
    chi = np.random.default_rng(2).uniform(0, np.pi, data.n_ch)
    perm = np.random.default_rng(3).permutation(data.n_ch)
    for other in (_rotated(data, chi), _rotated(data, np.zeros(data.n_ch), perm)):
        alt = fit_combinations(basis, other, max_sigma=0.1)
        assert alt.n_retained == fit.n_retained
        for cluster in fit.clusters:
            idx = [i for i in cluster if fit.classes[i] == "retained"]
            if not idx:
                continue
            W = np.asarray(fit.directions.retained)[:, idx]
            W2 = np.asarray(alt.directions.retained)[:, idx]
            np.testing.assert_allclose(W2 @ W2.T, W @ W.T, atol=1e-9)
        np.testing.assert_allclose(
            np.asarray(alt.projector), np.asarray(fit.projector), atol=1e-9
        )


def test_cluster_not_split_by_threshold():
    t = 10.0
    fit = _fit([100.0, t * (1 + 1e-12), t * (1 - 1e-12), 1.0], max_sigma=1 / t)
    assert (1, 2) in fit.clusters
    assert fit.classes[1] == fit.classes[2] == "weak"
    assert any("straddles 1/max_sigma" in n for n in fit.provenance.notes)


def test_sign_convention_data_independent():
    basis, _, clean, data = continuum_problem()
    other = stokes_data(
        clean * 0.5 + 1e-3,
        sigma=float(np.sqrt(np.asarray(data.noise)[0])),
        mask=data.mask,
    )
    a = fit_combinations(basis, data, max_sigma=0.1)
    b = fit_combinations(basis, other, max_sigma=0.1)
    for name in ("combination_rows", "singular_values", "estimator"):
        np.testing.assert_array_equal(
            np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        )
    W = np.asarray(a.directions.retained)
    top = np.argmax(np.abs(W), axis=0)
    assert np.all(W[top, np.arange(W.shape[1])] > 0)
    assert not np.allclose(np.asarray(a.beta_hat), np.asarray(b.beta_hat))
