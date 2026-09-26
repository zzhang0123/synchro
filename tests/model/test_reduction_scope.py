"""Scope of ``reduce_response``: kernel fallback, declared and approximate relations (T-006).

Out-of-scope kernels get structural zeros only; ``relations="continuum"``
refuses them; the continuum identity fails numerically on the harmonic
kernel (the scope boundary). Declared relations are checked against
``check_rtol`` (tested on both sides of the threshold); approximate ones keep
``Delta C = C - H T``; a sampled near-collinearity is never promoted.
"""

from __future__ import annotations

import numpy as np
import pytest

from _reduction_fixtures import (
    harmonic_basis,
    matrix_basis,
    polynomial_basis,
    random_columns,
    small_continuum_basis,
    stokes_data,
)
from syncmoments.model.fit import (
    LinearRelation,
    find_column_relations,
    fit_combinations,
    reduce_response,
)


def _rho(C, weights):
    """Relative residual of ``sum_j w_j C_j = 0`` (independent NumPy oracle)."""
    num = np.max(np.abs(sum(w * C[:, j] for j, w in weights)))
    den = sum(abs(w) * np.max(np.abs(C[:, j])) for j, w in weights)
    return num / den


@pytest.mark.parametrize("make", [harmonic_basis, polynomial_basis])
def test_auto_on_out_of_scope_kernel_is_identity_with_note(make):
    basis = make()
    red = reduce_response(basis)
    assert all(r.source == "structural" for r in red.relations)
    assert red.n_q == red.layout.n_full - len(red.layout.structural_zero)
    assert any("structural zeros only" in n for n in red.notes)
    assert red.kernel == dict(basis.provenance.kernel)["name"]
    none = reduce_response(basis, relations="none")
    assert np.array_equal(np.asarray(none.T), np.asarray(red.T))


def test_continuum_relations_refused_for_harmonic():
    with pytest.raises(ValueError, match="harmonic"):
        reduce_response(harmonic_basis(), relations="continuum")
    C = np.random.default_rng(0).standard_normal((8, 52))
    with pytest.raises(ValueError, match="kernel"):
        reduce_response(matrix_basis(C), relations="continuum")


def test_continuum_identity_fails_numerically_on_harmonic():
    basis = harmonic_basis()
    index = basis.index
    s_g, s_B, _ = (float(s) for s in basis.reference.scales)
    eps_g, eps_B = s_g / float(basis.reference.gamma0), s_B / float(basis.reference.B0)
    C = np.asarray(basis.response_matrix())
    for k in (0, 2):
        weights = (
            (index.position(0, 0, k, 0, 1, 0), 1.0 / eps_B),
            (index.position(0, 0, k, 1, 0, 0), -1.0 / (2 * eps_g)),
            (index.position(0, 0, k, 0, 0, 0), -1.0),
        )
        assert _rho(C, weights) >= 1e-3
        with pytest.raises(ValueError, match="declare it approximate"):
            reduce_response(
                basis, declared=(LinearRelation(weights, "declared", "(R) at (0,0)"),)
            )


def _perturbed_pair(scale):
    index, C = random_columns()
    C[:, 12] = C[:, 11] + scale * np.random.default_rng(3).standard_normal(C.shape[0])
    return index, C


def test_check_rtol_boundary():
    index, C = _perturbed_pair(1e-9)
    rel = LinearRelation.identical(12, 11, scope="caller: equal columns")
    rho = _rho(C, rel.weights)
    red = reduce_response(matrix_basis(C, index=index), declared=(rel,), check_rtol=rho)
    assert red.exact and np.isclose(red.relation_residuals[-1], rho, rtol=1e-12)
    with pytest.raises(ValueError, match="declare it approximate"):
        reduce_response(
            matrix_basis(C, index=index), declared=(rel,), check_rtol=rho / (1 + 1e-6)
        )
    for extreme in (0.0, 1e-300):
        with pytest.raises(ValueError):
            reduce_response(
                matrix_basis(C, index=index), declared=(rel,), check_rtol=extreme
            )
    ok = reduce_response(matrix_basis(C, index=index), declared=(rel,), check_rtol=1.0)
    assert ok.exact


def test_declared_exact_relation_accepted_and_labelled():
    index, C = random_columns()
    basis = matrix_basis(C, index=index)
    plain = reduce_response(basis)
    rel = LinearRelation.identical(5, 4, scope="caller: channel 5 duplicates 4")
    red = reduce_response(basis, declared=(rel,))
    assert red.n_q == plain.n_q - 1 and red.exact
    record = [r for r in red.relations if r.source == "declared"]
    assert record and record[0].scope == "caller: channel 5 duplicates 4"
    text = str(red.to_dict())
    assert "declared by caller" in text


def test_declared_wrong_relation_rejected():
    index, C = random_columns()
    with pytest.raises(ValueError, match="declare it approximate or correct it"):
        reduce_response(
            matrix_basis(C, index=index),
            declared=(LinearRelation.identical(6, 4, scope="wrong"),),
        )


def test_declared_redundant_relation_dropped_with_note():
    index, C = random_columns()
    rels = (
        LinearRelation.identical(5, 4, scope="a"),
        LinearRelation.identical(4, 5, scope="b"),
    )
    red = reduce_response(matrix_basis(C, index=index), declared=rels)
    assert sum(r.source == "declared" for r in red.relations) == 1
    assert any("dropped" in n for n in red.notes)


def test_find_column_relations_are_approximate():
    index, C = random_columns()
    basis = matrix_basis(C, index=index)
    found = find_column_relations(basis)
    assert found and all(r.source == "approximate" for r in found)
    assert all("numerical candidate on 6 channels" in r.scope for r in found)
    pairs = {tuple(sorted(j for j, _ in r.weights)) for r in found}
    assert {(4, 5), (6, 7), (3,)} <= pairs
    ratio = [dict(r.weights) for r in found if {j for j, _ in r.weights} == {6, 7}]
    w = ratio[0]
    assert np.isclose(-w[6] / w[7], -2.5) or np.isclose(-w[7] / w[6], -2.5)
    assert not any({j for j, _ in r.weights} >= {8, 9, 10} for r in found)
    assert find_column_relations(reduce_response(basis)) == found
    with pytest.raises(ValueError):
        find_column_relations(basis, rtol=-1.0)


def test_approximate_relation_keeps_delta_C():
    index, C = random_columns()
    C[:, 7] = C[:, 7] + 1e-4 * np.random.default_rng(9).standard_normal(C.shape[0])
    rel = LinearRelation.proportional(
        7, 6, -2.5, scope="near-proportional pair", approximate=True
    )
    red = reduce_response(matrix_basis(C, index=index), declared=(rel,))
    assert red.exact is False and red.max_delta_C > 0.0
    Cp, H, T = (np.asarray(x) for x in (red.C, red.H, red.T))
    np.testing.assert_allclose(np.asarray(red.delta_C), Cp - H @ T, atol=1e-15)
    assert red.relation_residuals[-1] > 0.0


def test_numerical_zero_column_not_promoted():
    index, C = random_columns()
    red = reduce_response(matrix_basis(C, index=index))
    assert 3 in red.representatives
    assert not any(3 in dict(r.weights) for r in red.relations)
    stokes = (C @ np.random.default_rng(1).standard_normal(C.shape[1])).reshape(-1, 4)
    fit = fit_combinations(
        matrix_basis(C, index=index), stokes_data(stokes), reduction=red, max_sigma=10
    )
    e3 = np.zeros(fit.jacobian.shape[1])
    e3[3] = 1.0
    null = np.asarray(fit.directions.numerical_null)
    analytic = np.asarray(fit.directions.analytic_null)
    assert np.linalg.norm(null.T @ e3) > 0.99
    assert analytic.shape[1] == 0 or np.linalg.norm(analytic.T @ e3) < 1e-12


def test_near_collinear_columns_not_promoted():
    index, C = random_columns()
    C[:, 7] = -2.5 * C[:, 6] * (1 + 1e-9 * np.random.default_rng(2).uniform(size=24))
    red = reduce_response(matrix_basis(C, index=index))
    assert {6, 7} <= set(red.representatives)
    assert all(r.source == "structural" for r in red.relations)


def test_perturbed_T_detected():
    basis = small_continuum_basis((0.2, 0.2))
    index = basis.index
    pos = {rs: index.position(0, 0, 0, *rs, 0) for rs in ((0, 0), (1, 0), (0, 1))}
    wrong = LinearRelation(
        ((pos[(0, 1)], 1.0), (pos[(0, 0)], -0.21), (pos[(1, 0)], -0.5)),
        "declared",
        "perturbed 0.2 -> 0.21",
    )
    with pytest.raises(ValueError, match="declare it approximate"):
        reduce_response(basis, relations="none", declared=(wrong,))
    red = reduce_response(basis)
    T = np.array(red.T)
    row = [i for i, s in enumerate(red.representatives) if s == pos[(0, 0)]][0]
    T[row, pos[(0, 1)]] = 0.21
    C, H = np.asarray(red.C), np.asarray(red.H)
    assert np.max(np.abs(C - H @ T)) > 1e-3 * np.max(np.abs(C))
