"""``reduce_response``: exact grouping ``C = H T = H L Q`` before any data (T-006).

Oracles: the NumPy product ``H T`` against the padded response, the spec
rows of ``T`` at 20 %/20 % scales, the closed-form relation (R) at other
scales, and a jet-rank oracle built from binomial Taylor coefficients of
``B Phi(B gamma^2)`` (``_reduction_fixtures.continuum_jets``) for completeness.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from _reduction_fixtures import (
    MANUSCRIPT_REPRESENTATIVES,
    continuum_jets,
    manuscript_rows,
    matrix_basis,
    random_columns,
    reference,
    small_continuum_basis,
)
from syncmoments.model.fit import (
    CoefficientLayout,
    LinearRelation,
    ResponseReduction,
    reduce_response,
)
from syncmoments.model.fit._relations import lq_qr, lq_svd
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples

CASES = {
    "20/20": ((0.2, 0.2), (0, 2, 2, None, None)),
    "10/30": ((0.1, 0.3), (0, 2, 2, None, None)),
    "30/5": ((0.3, 0.05), (0, 2, 2, None, None)),
    "2/40": ((0.02, 0.4), (0, 2, 2, None, None)),
    "N0": ((0.2, 0.2), (0, 2, 0, None, None)),
    "N1": ((0.2, 0.2), (0, 2, 1, None, None)),
    "N3": ((0.2, 0.2), (0, 2, 3, None, None)),
    "depth_degree1": ((0.2, 0.2), (0, 2, 2, 1, None)),
    "cap_B1": ((0.2, 0.2), (0, 2, 2, None, (None, 1, None))),
    "cap_g1": ((0.2, 0.2), (0, 2, 2, None, (1, None, None))),
    "cap_g0": ((0.2, 0.2), (0, 2, 2, None, (0, None, None))),
}


def _reduction(case, **kw):
    scales, trunc = CASES[case]
    return reduce_response(small_continuum_basis(scales, trunc), **kw)


def _relative(x, scale):
    return float(np.max(np.abs(x))) / float(np.max(np.abs(scale)))


@pytest.mark.parametrize("case", sorted(CASES))
def test_C_equals_H_T_and_H_L_Q(case):
    red = _reduction(case)
    C, T, H = (np.asarray(x) for x in (red.C, red.T, red.H))
    L, Q = np.asarray(red.L), np.asarray(red.Q)
    assert _relative(C - H @ T, C) <= 1e-12
    assert _relative(C - H @ L @ Q, C) <= 1e-12
    np.testing.assert_allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-13)
    assert np.allclose(L, np.tril(L)) and np.all(np.diag(L) > 0)
    np.testing.assert_allclose(L @ L.T, T @ T.T, atol=1e-12 * np.abs(T).max() ** 2)
    assert red.exact and red.lq_method == "qr"
    assert red.max_delta_C <= 1e-12


def _groups(N, depth_degree=None):
    b_max = N if depth_degree is None else depth_degree
    m0 = 2 * (N + 1)
    m2 = 4 * sum(
        (N - b + 1) if depth_degree is None else (N + 1) for b in range(b_max + 1)
    )
    return m0 + m2


def test_group_counts():
    red = _reduction("20/20")
    assert (red.n_full, red.layout.n_active, red.n_q) == (60, 52, 30)
    assert len(red.layout.structural_zero) == 8
    assert red.layout.structural_zero == tuple(range(52, 60))
    for case, N in (("N0", 0), ("N1", 1), ("N3", 3)):
        assert _reduction(case).n_q == _groups(N)
    assert _reduction("depth_degree1").n_q == _groups(2, depth_degree=1)
    # caps drop relations whose (r+1, s) or (r, s+1) slot is missing
    capped = _reduction("cap_g1")
    analytic = [r for r in capped.relations if r.source == "analytic"]
    assert len(analytic) < len(
        [r for r in _reduction("20/20").relations if r.source == "analytic"]
    )
    assert capped.n_q == capped.layout.n_active - len(analytic)


def test_gamma_cap_zero_gives_no_relations():
    red = _reduction("cap_g0")
    assert all(r.source == "structural" for r in red.relations)
    assert red.n_q == red.layout.n_active


@pytest.mark.parametrize(
    "trunc",
    [(0, 2, N, None, None) for N in range(4)]
    + [(0, 2, 2, None, (None, 1, None)), (0, 2, 2, None, (1, None, None))]
    + [(0, 2, 3, 1, None)],
)
@pytest.mark.parametrize("eps", [(0.2, 0.2), (0.1, 0.3)])
def test_relation_set_complete_uncapped(trunc, eps):
    """Jets of random ``B Phi(B gamma^2)`` satisfy every relation and span ``n_q``."""
    L_mu, L_eta, N, depth_degree, caps = trunc
    index = MomentIndex.build(
        Truncation(L_mu, L_eta, N, depth_degree, max_orders=caps),
        components=("I", "Q"),
    )
    n_rows = 4 * 40
    C = continuum_jets(index, *eps, n_rows)
    red = reduce_response(
        matrix_basis(C, index=index, kernel="continuum", ref=reference(*eps)),
        relations="continuum",
    )
    Cp = np.asarray(red.C)
    assert np.linalg.matrix_rank(Cp, tol=1e-9 * np.abs(Cp).max()) == red.n_q
    assert _relative(Cp - np.asarray(red.H @ red.T), Cp) <= 1e-12


def test_manuscript_T_reproduced_with_representatives():
    red = _reduction("20/20", representatives=MANUSCRIPT_REPRESENTATIVES)
    rows = manuscript_rows(red.layout.index)
    assert sorted(rows) == sorted(red.representatives)
    T = np.asarray(red.T)
    for i, slot in enumerate(red.representatives):
        np.testing.assert_allclose(T[i], rows[slot], rtol=0, atol=1e-14)
    assert "q[M0[l=0,k=0;r=0,s=0]]" in red.group_labels
    assert any("0.2 a[M0[l=0,k=0;r=0,s=1]]" in f for f in red.group_formulas)


def test_manuscript_T_rejected_at_other_scales():
    basis = small_continuum_basis((0.1, 0.3))
    index = basis.index
    declared = []
    for k in (0, 2):
        pos = {rs: index.position(0, 0, k, *rs, 0) for rs in ((0, 0), (1, 0), (0, 1))}
        declared.append(
            LinearRelation(
                ((pos[(0, 1)], 1.0), (pos[(0, 0)], -0.2), (pos[(1, 0)], -0.5)),
                "declared",
                "the 20 %/20 % identity C_B = 0.2 C_0 + 0.5 C_g",
            )
        )
    with pytest.raises(ValueError, match="declare it approximate or correct it"):
        reduce_response(basis, relations="none", declared=tuple(declared))


def test_relation_coefficients_general_scales():
    eps_g, eps_B = 0.1, 0.3
    red = _reduction("10/30")
    index = red.layout.index
    analytic = {r.weights: r for r in red.relations if r.source == "analytic"}

    def expected(k, r, s, b=0, h=0, offset=0):
        def pos(rr, ss):
            return index.position(h, 0, k, rr, ss, b) + offset

        weights = {
            pos(r, s + 1): (s + 1) / eps_B,
            pos(r + 1, s): -(r + 1) / (2 * eps_g),
            pos(r, s): -(1 + r / 2 - s),
        }
        return {j: w for j, w in weights.items() if w != 0.0}

    for want in (
        expected(0, 0, 0),
        expected(2, 1, 0),
        expected(0, 0, 0, b=1, h=2, offset=index.n2),
    ):
        match = [
            dict(w)
            for w in analytic
            if set(dict(w)) == set(want)
            and all(np.isclose(dict(w)[j], want[j], rtol=1e-14) for j in want)
        ]
        assert match, want


def test_structural_zero_slots():
    red = _reduction("20/20")
    C = np.asarray(red.C)
    for slot, reason in zip(red.layout.structural_zero, red.layout.structural_reasons):
        assert np.all(C[:, slot] == 0.0) and reason
    # parity=False keeps odd-parity M2 rows whose P basis vanishes: structural
    index = MomentIndex.build(Truncation(0, 1, 1), parity=False, components=("I", "Q"))
    layout = CoefficientLayout.build(index, reference())
    odd = [e.slot for e in index.entries() if e.h == 2 and (e.l + e.k) % 2]
    assert set(odd) <= set(layout.structural_zero)
    C = np.zeros((8, index.n_real))
    C[:, 0] = 1.0
    ok = reduce_response(matrix_basis(C, index=index))
    assert set(odd) <= set(ok.layout.structural_zero)
    C[1, odd[0]] = 1e-30
    with pytest.raises(ValueError, match="structural"):
        reduce_response(matrix_basis(C, index=index))


def test_reduction_is_data_independent():
    params = inspect.signature(reduce_response).parameters
    assert not {"data", "moments", "noise", "truth", "stokes"} & set(params)
    trunc = (0, 2, 2, 0, None)
    a = reduce_response(small_continuum_basis((0.2, 0.2), trunc, "taylor", 8))
    b = reduce_response(small_continuum_basis((0.2, 0.2), trunc, "gaussian", 5))
    assert np.array_equal(np.asarray(a.T), np.asarray(b.T))
    assert a.representatives == b.representatives
    again = _reduction("20/20")
    first = _reduction("20/20")
    assert np.array_equal(np.asarray(again.T), np.asarray(first.T))
    assert again.group_labels == first.group_labels


def test_raw_scales_and_full_vector():
    red = _reduction("10/30")
    layout = red.layout
    rng = np.random.default_rng(4)
    n = 50
    samples = PopulationSamples(
        gamma=1e4 * (1 + 0.1 * rng.uniform(-1, 1, n)),
        B=5e-6 * (1 + 0.2 * rng.uniform(-1, 1, n)),
        mu=rng.uniform(-1, 1, n),
        eta=rng.uniform(-1, 1, n),
        phi=rng.uniform(0, 2 * np.pi, n),
        depth=2.0 + 0.3 * rng.uniform(-1, 1, n),
        weights=rng.uniform(0.5, 1.5, n),
    )
    moments = JointMoments.from_samples(samples, layout.index, reference(0.1, 0.3))
    a = layout.full_vector(moments, amplitude=2.0)
    raw = moments.to_raw_displacements()
    want = 2.0 * np.concatenate([raw.to_vector(), raw.m0_ext])
    np.testing.assert_allclose(layout.to_raw(a), want, rtol=1e-12)
    assert layout.n_full == a.shape[0] == 60
    no_ext = JointMoments(layout.index, moments.m0, moments.m2, moments.reference)
    with pytest.raises(ValueError, match="from_samples"):
        layout.full_vector(no_ext)
    other = JointMoments.from_samples(samples, layout.index, reference(0.2, 0.2))
    with pytest.raises(ValueError, match="reference"):
        layout.full_vector(other)


def test_invalid_inputs():
    index, C = random_columns()
    basis = matrix_basis(C, index=index)
    bad = C.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        reduce_response(matrix_basis(bad, index=index))
    for kw in (
        {"relations": "all"},
        {"representatives": "beta"},
        {"representatives": ((0, 0, 0), (0, 0, 0))},
        {"representatives": ((0, 0),)},
        {"check_rtol": -1.0},
        {"check_rtol": float("nan")},
        {"include_ext": "yes"},
    ):
        with pytest.raises(ValueError):
            reduce_response(basis, **kw)
    for rel in (
        LinearRelation(((99, 1.0), (0, -1.0)), "declared", "x"),
        LinearRelation(((0, 1.0), (0, -1.0)), "declared", "x"),
        LinearRelation(((0, np.inf), (1, -1.0)), "declared", "x"),
        LinearRelation(((0, 0.0), (1, 0.0)), "declared", "x"),
        LinearRelation(((0, 1.0), (1, -1.0)), "analytic", "x"),
        LinearRelation(((0, 1.0), (1, -1.0)), "declared", ""),
    ):
        with pytest.raises(ValueError):
            reduce_response(basis, declared=(rel,))
    with pytest.raises(ValueError):
        reduce_response(basis, declared=("not a relation",))


@pytest.mark.parametrize(
    "T",
    [
        np.random.default_rng(1).standard_normal((5, 9)),
        np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-9, 0.0]]),
        np.array([[2.0, 0.0, 0.0, 0.5]]),
    ],
)
def test_lq_qr_and_svd_agree(T):
    """Boundary rule: both factorisations evaluated directly on one full-row-rank ``T``."""
    for L, Q in (lq_qr(T), lq_svd(T)):
        np.testing.assert_allclose(L @ Q, T, atol=1e-13 * np.abs(T).max())
        np.testing.assert_allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-13)
    (L1, Q1), (L2, Q2) = lq_qr(T), lq_svd(T)
    np.testing.assert_allclose(Q1.T @ Q1, Q2.T @ Q2, atol=1e-13)
    np.testing.assert_allclose(L1 @ L1.T, L2 @ L2.T, atol=1e-12 * np.abs(T).max() ** 2)


def test_reduction_result_type_and_json():
    import json

    red = _reduction("20/20")
    assert isinstance(red, ResponseReduction)
    a = np.random.default_rng(0).standard_normal(red.n_full)
    np.testing.assert_allclose(red.grouped(a), np.asarray(red.T) @ a)
    assert red.predict_full(a).shape == (8, 4)
    summary = json.loads(json.dumps(red.to_dict(), allow_nan=False))
    assert summary["n_q"] == 30 and summary["exact"] is True
    assert "T" not in summary
    full = red.to_dict(arrays=True)
    assert np.asarray(full["T"]).shape == (30, 60)
    assert red.null_basis.shape == (60, 30)
    np.testing.assert_allclose(
        np.asarray(red.T) @ np.asarray(red.null_basis), 0.0, atol=1e-12
    )
