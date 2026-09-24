"""Tests for ``synchro.model.fit.diagnostics.feasibility_checks``.

Oracles: explicit weighted sums over a discrete population for every moment
inequality (Cauchy-Schwarz, ``|<f^2 e^{2i phi}>| <= <f^2>``, moment
matrices) and ``scipy.special.eval_legendre`` on a fine grid for the
Legendre ranges. Crafted vectors violate one condition at a time.
"""

import json

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

from _diagnostics_fixtures import (
    population_support,
    random_population,
    reference_222,
)
from synchro.model.fit.diagnostics import FeasibilityReport, feasibility_checks
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, Support

# -- feasibility oracles ----------------------------------------------------------


def sample_arrays(pop, reference):
    w = np.asarray(pop.normalised_weights())
    z = np.asarray(reference.z(pop.gamma, pop.B, pop.depth))
    return w, z, np.asarray(pop.mu), np.asarray(pop.eta), np.asarray(pop.phi)


def names(report):
    return tuple(name for name, _, _ in report.checks)


def margin(report, name):
    return dict((n, m) for n, _, m in report.checks)[name]


def test_feasibility_from_samples_passes_everything(index_222):
    rng = np.random.default_rng(1)
    pop, ref = random_population(rng, 60), reference_222()
    moments = JointMoments.from_samples(pop, index_222, ref)
    report = feasibility_checks(moments, index_222, population_support())
    assert isinstance(report, FeasibilityReport)
    assert report.failed() == ()
    assert all(passed for _, passed, _ in report.checks)
    assert "necessary" in report.statement.lower() and "not certif" in report.statement
    assert "support_mean[z_depth]" in names(report)  # m0_ext present
    assert not any("not in the fitted vector" in item for item in report.not_computable)
    json.dumps(report.to_dict())


def test_feasibility_margins_match_explicit_sums(index_222):
    rng = np.random.default_rng(4)
    pop, ref = random_population(rng, 30), reference_222()
    moments = JointMoments.from_samples(pop, index_222, ref)
    report = feasibility_checks(moments, index_222, population_support())
    w, z, mu, eta, phi = sample_arrays(pop, ref)
    e2 = np.exp(2j * phi)
    zg, zB = z[:, 0], z[:, 1]
    assert_allclose(margin(report, "abs_bound[f=1]"), 1 - abs(np.sum(w * e2)))
    assert_allclose(
        margin(report, "cauchy_schwarz[f=z_gamma]"),
        np.sum(w * zg**2) - abs(np.sum(w * zg * e2)) ** 2,
    )
    assert_allclose(
        margin(report, "cauchy_schwarz[f=z_B]"),
        np.sum(w * zB**2) - abs(np.sum(w * zB * e2)) ** 2,
    )
    assert_allclose(
        margin(report, "abs_bound[f=z_gamma]"),
        np.sum(w * zg**2) - abs(np.sum(w * zg**2 * e2)),
    )
    assert_allclose(
        margin(report, "abs_bound[f=P_1(mu)]"),
        np.sum(w * mu**2) - abs(np.sum(w * mu**2 * e2)),
        atol=1e-14,
    )
    x = np.stack([np.ones_like(zg), zg, zB])
    M = (w * x) @ x.T
    assert_allclose(margin(report, "psd[1,z_gamma,z_B]"), np.linalg.eigvalsh(M)[0])
    assert_allclose(
        margin(report, "legendre_range[P_2(mu)]"),
        np.sum(w * eval_legendre(2, mu)) + 0.5,
    )
    assert_allclose(margin(report, "normalisation"), 0.0, atol=1e-15)


def test_feasibility_hermitian_matrix_matches_samples(index_222):
    rng = np.random.default_rng(6)
    pop, ref = random_population(rng, 50), reference_222()
    moments = JointMoments.from_samples(pop, index_222, ref)
    report = feasibility_checks(moments, index_222, population_support())
    name = [n for n in names(report) if n.startswith("hermitian_psd[")][0]
    w, z, mu, eta, phi = sample_arrays(pop, ref)
    zg, zB, zd = z.T
    e2 = np.exp(2j * phi)
    columns = {
        "1": np.ones_like(zg),
        "z_gamma": zg,
        "z_B": zB,
        "z_depth": zd,
        "P_1(mu) P_1(eta)": mu * eta,
        "e^{2i phi}": e2,
        "z_gamma e^{2i phi}": zg * e2,
        "z_B e^{2i phi}": zB * e2,
        "P_1(mu) P_1(eta) e^{2i phi}": mu * eta * e2,
        "z_depth e^{2i phi}": zd * e2,
    }
    listed = name[len("hermitian_psd[") : -1].split(",")
    assert set(listed) == set(columns)
    F = np.stack([columns[key] for key in listed])
    H = (w * np.conj(F)) @ F.T
    assert_allclose(margin(report, name), np.linalg.eigvalsh(H)[0], atol=1e-12)


def feasible_vector(index, rng, size=40):
    pop, ref = random_population(rng, size), reference_222()
    return np.asarray(JointMoments.from_samples(pop, index, ref).to_vector()), ref


def crafted(index, vector, ref, rows=()):
    """Copy of ``vector`` with ``rows = {(h, l, k, r, s, b, part): value}`` overwritten."""
    m = np.array(vector)
    for key, value in dict(rows).items():
        h, l, k, r, s, b, part = key
        slot = index.position(h, l, k, r, s, b) + (index.n2 if part == "im" else 0)
        m[slot] = value
    return JointMoments.from_vector(index, jnp.asarray(m), ref, tol=0.5)


def test_feasibility_crafted_violations(index_222):
    rng = np.random.default_rng(9)
    m, ref = feasible_vector(index_222, rng)
    support = population_support()
    base = feasibility_checks(crafted(index_222, m, ref), index_222, support)
    assert base.failed() == ()
    bad = crafted(
        index_222,
        m,
        ref,
        {(2, 0, 0, 0, 0, 0, "re"): 1.2, (2, 0, 0, 0, 0, 0, "im"): 0.0},
    )
    failed = feasibility_checks(bad, index_222, support).failed()
    assert "abs_bound[f=1]" in failed
    assert any(n.startswith("hermitian_psd") for n in failed)
    bad = crafted(
        index_222,
        m,
        ref,
        {(0, 0, 0, 1, 0, 0, "real"): 0.5, (0, 0, 0, 2, 0, 0, "real"): 0.1},
    )
    failed = feasibility_checks(bad, index_222, support).failed()
    assert "psd[1,z_gamma,z_B]" in failed
    bad = crafted(
        index_222,
        m,
        ref,
        {
            (2, 0, 0, 1, 0, 0, "re"): 0.9,
            (2, 0, 0, 1, 0, 0, "im"): 0.0,
            (0, 0, 0, 2, 0, 0, "real"): 0.5,
        },
    )
    failed = feasibility_checks(bad, index_222, support).failed()
    assert "cauchy_schwarz[f=z_gamma]" in failed
    bad = crafted(index_222, m, ref, {(0, 0, 0, 0, 0, 0, "real"): 1.01})
    assert "normalisation" in feasibility_checks(bad, index_222, support).failed()
    bad = crafted(index_222, m, ref, {(0, 2, 0, 0, 0, 0, "real"): -0.7})
    failed = feasibility_checks(bad, index_222, support).failed()
    assert "legendre_range[P_2(mu)]" in failed
    bad = crafted(index_222, m, ref, {(0, 0, 0, 1, 0, 0, "real"): 5.0})
    failed = feasibility_checks(bad, index_222, support).failed()
    assert "support_mean[z_gamma]" in failed and "support_bound[M0]" in failed


def test_feasibility_support_and_not_computable(index_222, index_111):
    rng = np.random.default_rng(12)
    m, ref = feasible_vector(index_222, rng)
    moments = crafted(index_222, m, ref)  # from_vector: no m0_ext
    report = feasibility_checks(moments, index_222, population_support())
    assert report.failed() == ()
    assert "support_mean[z_depth]" not in names(report)
    assert any("support_mean[z_depth]" in item for item in report.not_computable)
    assert any("cauchy_schwarz[f=z_depth]" in item for item in report.not_computable)
    # a support that excludes the population fails the mean checks
    shifted = Support((7.9, 12.0), (0.5, 2.0), (-2.0, 2.0))
    assert (
        "support_mean[z_gamma]"
        in feasibility_checks(moments, index_222, shifted).failed()
    )
    # no support: support checks are listed as not computable
    none = feasibility_checks(moments, index_222, None)
    assert not any(n.startswith("support") for n in names(none))
    assert any("support" in item for item in none.not_computable)
    # (1,1,1): the psd block and the P_1^2 bounds need N >= 2 and L >= 2
    small = JointMoments.from_samples(random_population(rng, 20), index_111, ref)
    rep = feasibility_checks(small, index_111, population_support())
    assert rep.failed() == ()
    assert "psd[1,z_gamma,z_B]" not in names(rep)
    assert any("psd[1,z_gamma,z_B]" in item for item in rep.not_computable)
    assert any("abs_bound[f=P_1(mu)]" in item for item in rep.not_computable)


def test_feasibility_boundary_single_atom_and_extreme_weights(index_222):
    rng = np.random.default_rng(15)
    ref = reference_222()
    pop = random_population(rng, 1)
    moments = JointMoments.from_samples(pop, index_222, ref)
    report = feasibility_checks(moments, index_222, population_support())
    assert report.failed() == ()  # |M2_00| = 1 exactly: margin 0 passes
    assert abs(margin(report, "abs_bound[f=1]")) < 1e-12
    big = random_population(rng, 12)
    big = type(big)(
        big.gamma,
        big.B,
        big.mu,
        big.eta,
        big.phi,
        big.depth,
        weights=1e300 * big.weights,
    )
    rep = feasibility_checks(
        JointMoments.from_samples(big, index_222, ref), index_222, population_support()
    )
    assert rep.failed() == ()


def test_feasibility_rejects_index_mismatch(index_222, index_111):
    rng = np.random.default_rng(0)
    pop, ref = random_population(rng, 5), reference_222()
    moments = JointMoments.from_samples(pop, index_222, ref)
    with pytest.raises(ValueError):
        feasibility_checks(moments, index_111, population_support())


def test_legendre_minimum_matches_dense_grid():
    from synchro.model.fit.diagnostics import legendre_range

    x = np.linspace(-1, 1, 200001)
    for l in range(0, 9):
        lo, hi = legendre_range(l)
        assert hi == 1.0
        assert_allclose(lo, eval_legendre(l, x).min(), atol=1e-8)


def test_feasibility_depth_layout(index_222):
    index = MomentIndex.build(Truncation(1, 1, 1, depth_degree=2))
    rng = np.random.default_rng(17)
    pop, ref = random_population(rng, 25), reference_222()
    moments = JointMoments.from_samples(pop, index, ref)
    report = feasibility_checks(moments, index, population_support())
    assert report.failed() == ()
    assert "cauchy_schwarz[f=z_depth]" in names(report)
