"""Tests for ``syncmoments.model.fit.diagnostics.identifiability``.

Oracle: ``numpy.linalg.svd`` of an independently assembled whitened design
``L^-1 R C [c | P]`` (``P, c`` from ``ParameterMap.affine_pieces``; a
central-difference Jacobian for a nonlinear map), with its columns scaled to
unit norm (the column equilibration shared with ``fit_linear``; review
finding R09). Bases and data come from
``_diagnostics_fixtures`` (the real ``SpectralBasis``/``StokesData`` when
their modules import, private stubs otherwise). Feasibility tests live in
``test_feasibility.py``.
"""

import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from _diagnostics_fixtures import (
    make_basis,
    make_data,
    reference_222,
)
from syncmoments.model.assumptions import (
    gaussian_screen,
    isotropic_pitch,
    no_assumption,
)
from syncmoments.model.fit.diagnostics import (
    IdentifiabilityReport,
    identifiability,
)

# -- identifiability oracle -------------------------------------------------------


def oracle_design(basis, data, pm, *, reference=None, amplitude=True):
    """NumPy ``L^-1 R C [c | P]`` with masked rows removed."""
    C = np.asarray(basis.response_matrix())
    P, c = pm.affine_pieces(reference=reference)
    cols = np.asarray(P)
    if amplitude:
        cols = np.concatenate([np.asarray(c)[:, None], cols], axis=1)
    R = np.asarray(data.response) if data.response is not None else np.eye(C.shape[0])
    if data.mask is not None:
        keep = np.flatnonzero(np.broadcast_to(np.asarray(data.mask), (basis.n_ch, 4)))
        R = R[keep]
    noise = np.asarray(data.noise)
    if noise.ndim == 1:
        if data.mask is not None:
            noise = noise[keep]
        L_inv = np.diag(1 / np.sqrt(noise))
    else:
        L_inv = np.linalg.inv(np.linalg.cholesky(noise))
    return L_inv @ R @ C @ cols


def equilibrated(G):
    """``(G D, d)`` with ``d_i = 1/||G_i||`` (1 for a zero column)."""
    norms = np.linalg.norm(G, axis=0)
    d = np.where(norms > 0, 1.0 / np.where(norms > 0, norms, 1.0), 1.0)
    return G * d[None, :], d


def check_report(report, G, tol):
    G, d = equilibrated(G)
    assert_allclose(report.column_scales, d, rtol=1e-12)
    s = np.linalg.svd(G, compute_uv=False)
    assert_allclose(report.singular_values, s, rtol=1e-10, atol=1e-12)
    rank = int(np.sum(s > tol * s[0]))
    n_u = G.shape[1]
    assert report.null_dim == n_u - rank
    assert report.null_basis.shape == (n_u, report.null_dim)
    assert report.modes.shape == (n_u, n_u)
    assert report.resolution.shape == (n_u,)
    res = np.asarray(report.resolution)
    assert np.all(res >= -1e-12) and np.all(res <= 1 + 1e-12)
    assert_allclose(res.sum(), rank, atol=1e-9)
    if report.null_dim:
        scale = np.linalg.norm(G)
        assert np.linalg.norm(G @ np.asarray(report.null_basis)) <= 1e-8 * scale
        NB = np.asarray(report.null_basis)
        assert_allclose(NB.T @ NB, np.eye(report.null_dim), atol=1e-10)
    assert len(report.labels) == n_u


@pytest.mark.parametrize("n_ch", [3, 8])
def test_identifiability_matches_numpy_svd(index_111, n_ch):
    rng = np.random.default_rng(n_ch)
    basis, data = make_basis(index_111, rng, n_ch), make_data(n_ch, rng)
    pm = no_assumption(index_111)
    report = identifiability(basis, data, pm)
    assert isinstance(report, IdentifiabilityReport)
    check_report(report, oracle_design(basis, data, pm), 1e-8)
    assert report.labels == ("amplitude",) + pm.labels()
    if n_ch == 3:  # 12 data rows against 28 unknowns: rank deficient
        assert report.null_dim >= 16 and report.rank == 12
        assert len(report.weak(0.5)) > 0
    else:  # 32 rows, 28 unknowns, generic random basis: full rank
        assert report.null_dim == 0 and report.rank == 28
        assert report.weak(0.999) == ()
        assert_allclose(report.resolution, 1.0, atol=1e-10)


def test_null_vector_leaves_prediction_unchanged(index_111):
    rng = np.random.default_rng(3)
    basis, data = make_basis(index_111, rng, 3), make_data(3, rng)
    pm = no_assumption(index_111)
    report = identifiability(basis, data, pm)
    C = np.asarray(basis.response_matrix())
    P, c = (np.asarray(x) for x in pm.affine_pieces())
    theta = rng.standard_normal(pm.n_free())
    A = 2.5
    u = np.concatenate([[A], A * theta])
    v = np.asarray(report.column_scales) * np.asarray(report.null_basis)[:, 0]
    u2 = u + 0.7 * v / np.abs(v).max()
    A2, theta2 = u2[0], u2[1:] / u2[0]
    pred = A * C @ (P @ theta + c)
    pred2 = A2 * C @ (P @ theta2 + c)
    assert_allclose(pred2, pred, atol=1e-9 * np.abs(pred).max())
    named = report.null_components(0, cutoff=1e-3)
    assert named and all(name in report.labels for name, _ in named)


def test_identifiability_weak_and_resolution_labels(index_111):
    rng = np.random.default_rng(5)
    basis, data = make_basis(index_111, rng, 3), make_data(3, rng)
    pm = isotropic_pitch(index_111)
    report = identifiability(basis, data, pm)
    res = dict(zip(report.labels, np.asarray(report.resolution)))
    for threshold in (0.0, 0.3, 0.9, 1.0 + 1e-9):
        expect = tuple(name for name in report.labels if res[name] < threshold)
        assert report.weak(threshold) == expect
    d = report.to_dict()
    json.dumps(d)
    assert d["null_dim"] == report.null_dim
    assert set(d["resolution"]) == set(report.labels)


@pytest.mark.parametrize("amplitude", [True, False])
def test_identifiability_amplitude_column_switch(index_111, amplitude):
    rng = np.random.default_rng(11)
    basis, data = make_basis(index_111, rng, 4), make_data(4, rng)
    pm = isotropic_pitch(index_111)
    report = identifiability(basis, data, pm, amplitude=amplitude)
    G = oracle_design(basis, data, pm, amplitude=amplitude)
    check_report(report, G, 1e-8)
    assert len(report.labels) == pm.n_free() + int(amplitude)


def test_identifiability_tol_both_sides(index_111):
    rng = np.random.default_rng(2)
    basis, data = make_basis(index_111, rng, 8), make_data(8, rng)
    pm = no_assumption(index_111)
    full = identifiability(basis, data, pm)
    assert full.null_dim == 0
    s = np.asarray(full.singular_values)
    ratio = float(s[-1] / s[0])
    below = identifiability(basis, data, pm, tol=ratio * 0.5)
    above = identifiability(basis, data, pm, tol=float(np.sqrt(s[-1] * s[-2]) / s[0]))
    at = identifiability(basis, data, pm, tol=ratio)  # sigma <= tol sigma_max is null
    assert below.null_dim == 0
    assert above.null_dim == 1 and at.null_dim == 1
    assert_allclose(above.resolution.sum(), full.rank - 1, atol=1e-9)


@pytest.mark.parametrize("mask", [[True, True, True, False], "rows"])
def test_identifiability_with_mask(index_111, mask):
    rng = np.random.default_rng(21)
    if mask == "rows":
        mask = rng.uniform(size=(5, 4)) > 0.3
    basis, data = make_basis(index_111, rng, 5), make_data(5, rng, mask=mask)
    pm = no_assumption(index_111)
    report = identifiability(basis, data, pm)
    G = oracle_design(basis, data, pm)
    assert G.shape[0] == int(np.broadcast_to(np.asarray(mask), (5, 4)).sum())
    check_report(report, G, 1e-8)


def test_identifiability_with_response_and_covariance(index_111):
    rng = np.random.default_rng(8)
    R = rng.standard_normal((10, 12))
    basis = make_basis(index_111, rng, 3)
    data = make_data(3, rng, response=R, covariance=True)
    pm = no_assumption(index_111)
    report = identifiability(basis, data, pm)
    check_report(report, oracle_design(basis, data, pm), 1e-8)
    assert report.rank == 10


def test_identifiability_nonaffine_uses_jacobian(index_111):
    rng = np.random.default_rng(13)
    basis, data = make_basis(index_111, rng, 8), make_data(8, rng)
    ref = reference_222()
    pm = gaussian_screen(index_111, 0.5, 0.8, fit_hyper=True)
    assert not pm.is_affine()
    with pytest.raises(ValueError):
        identifiability(basis, data, pm)
    v0 = rng.standard_normal(pm.n_free())
    v0[-1] = abs(v0[-1]) + 0.2  # sigma (last hyper entry) must be positive
    theta = pm.unflatten(v0)
    report = identifiability(basis, data, pm, theta=theta, reference=ref)

    def m_of(v):
        return np.asarray(pm(pm.unflatten(v), ref).to_vector())

    J = np.empty((index_111.n_real, v0.size))
    h = 1e-5
    for a in range(v0.size):
        e = np.zeros_like(v0)
        e[a] = h
        J[:, a] = (m_of(v0 + e) - m_of(v0 - e)) / (2 * h)
    c = m_of(v0) - J @ v0
    C = np.asarray(basis.response_matrix())
    G = np.diag(1 / np.sqrt(np.asarray(data.noise))) @ C @ np.column_stack([c, J])
    G, _ = equilibrated(G)
    assert_allclose(
        report.singular_values,
        np.linalg.svd(G, compute_uv=False),
        rtol=1e-6,
        atol=1e-10,
    )
    assert report.labels == ("amplitude",) + pm.labels()
    # sigma enters only through <z_depth^2> (b = 2), absent at N = 1: a null direction
    assert report.null_dim == 1
    assert dict(report.null_components(0)) == pytest.approx(
        {"hyper:gaussian_depth:sigma": 1.0}, abs=1e-8
    ) or dict(report.null_components(0)) == pytest.approx(
        {"hyper:gaussian_depth:sigma": -1.0}, abs=1e-8
    )
    assert report.weak(0.5) == ("hyper:gaussian_depth:sigma",)


def test_identifiability_rejects_mismatched_index(index_111, index_222):
    rng = np.random.default_rng(0)
    basis, data = make_basis(index_111, rng, 3), make_data(3, rng)
    with pytest.raises(ValueError):
        identifiability(basis, data, no_assumption(index_222))
    with pytest.raises(ValueError):
        identifiability(basis, make_data(4, rng), no_assumption(index_111))
