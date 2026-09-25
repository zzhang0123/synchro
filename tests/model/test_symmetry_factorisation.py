"""Symmetry closures combined with factorisations; ``fully_independent`` counts.

Derivation (checked below). A symmetry ``g`` acts variable by variable:
``pitch_symmetric`` flips ``mu``; ``field_reversal_symmetric`` flips ``mu``
and ``eta`` and shifts ``phi`` by ``pi`` (``e^{ih pi} = 1`` for ``h in {0,
2}``). Under a factorisation ``p = prod_G p_G``, ``g`` maps each group to
itself, so the marginal of ``p o g`` on ``G`` is ``p_G o g_G``. Joint
invariance ``p o g = p`` therefore holds iff every marginal is invariant,
``p_G o g_G = p_G``. A group entry ``<prod_{v in G} f_v^{e_v}>_G`` is odd
under ``g_G`` iff the flipped exponents on ``G`` have odd sum
(``P_l(-x) = (-1)^l P_l(x)``), and then it is zero. A row vanishes iff one
of its group factors vanishes. With ``mu`` and ``eta`` in different groups,
field reversal thus forces ``<P_odd(mu) ...>`` and ``<P_odd(eta) ...>`` to
zero, not only the odd-``l + k`` joint rows; with one group it reduces to the
joint rule.

Counts. ``fully_independent`` has one table per variable: ``N_gamma``,
``N_B`` and ``max_b`` powers of ``z``, ``L_mu`` and ``L_eta`` Legendre
degrees and the complex ``<e^{2i phi}>`` (2 real), so ``n_free = N_gamma +
N_B + max_b + L_mu + L_eta + 2`` with ``(N_gamma, N_B, max_b) =
truncation.caps()`` (``2N + max_b + ...`` only when uncapped). Field reversal
keeps the even degrees: ``L_mu`` -> ``floor(L_mu / 2)``, same for ``eta``
(pitch symmetry only for ``mu``). At ``(2, 2, 2)``: ``12`` -> ``10`` (field),
``11`` (pitch); at ``(8, 8, 2)``: ``24`` -> ``16`` (field), ``20`` (pitch).
``{eta} | rest`` with field reversal at ``(2, 2, 2)``: ``eta`` keeps
``P_2`` (1); ``rest`` keeps the ``M0`` projections ``l in {0, 2}`` times six
``(r, s)`` minus the unit (11) and the ``M2`` projections ``l in {0, 2}``
times ten ``(r, s, b)``, complex (40): 52.
"""

import itertools

import jax
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model.assumptions import (
    Factorisation,
    ParameterMap,
    field_reversal_symmetric,
    fully_independent,
    pitch_symmetric,
)
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference

REF = Reference(20.0, 2.0, depth_ref=1.5, scales=(2.0, 0.2, 0.5))
ETA_APART = (("gamma", "B", "mu", "phi", "depth"), ("eta",))

PINNED_SYMMETRIC = {
    (1, 1, 0): {"field": 2, "pitch": 3, "both": 2},
    (3, 1, 1): {"field": 6, "pitch": 7, "both": 6},
    (2, 2, 2): {"field": 10, "pitch": 11, "both": 10, "eta_apart": 52},
    (8, 8, 2): {"field": 16, "pitch": 20, "both": 16},
}


def product_population(symmetric):
    """Exact product grid; ``symmetric`` mirrors the ``mu``, ``eta`` marginals."""
    if symmetric:
        mu = ([-0.7, -0.2, 0.2, 0.7], [0.3, 0.2, 0.2, 0.3])
        eta = ([-0.9, -0.4, 0.4, 0.9], [0.1, 0.4, 0.4, 0.1])
    else:
        mu = ([-0.3, 0.4, 0.8], [0.2, 0.5, 0.3])
        eta = ([-0.6, 0.1, 0.5], [0.3, 0.3, 0.4])
    marginals = [
        ([18.0, 21.0, 23.0], [0.3, 0.5, 0.2]),
        ([1.8, 2.3], [0.6, 0.4]),
        mu,
        eta,
        ([0.3, 0.3 + np.pi, 1.1, 1.1 + np.pi], [0.2, 0.2, 0.3, 0.3]),
        ([1.0, 1.9], [0.45, 0.55]),
    ]
    atoms = list(itertools.product(*(zip(*m) for m in marginals)))
    values = [np.array([a[i][0] for a in atoms]) for i in range(6)]
    weights = np.prod([[a[i][1] for a in atoms] for i in range(6)], axis=0)
    return PopulationSamples(*values, weights=weights)


def symmetric_maps(index):
    fi = fully_independent(index)
    return {
        "field": fi.assume(field_reversal_symmetric(index)),
        "pitch": fi.assume(pitch_symmetric(index)),
        "both": fi.assume(pitch_symmetric(index), field_reversal_symmetric(index)),
        "eta_apart": ParameterMap.build(
            index, Factorisation(ETA_APART), field_reversal_symmetric(index).closures
        ),
    }


def odd_rows(index, flipped):
    """Mask over ``m``: rows whose ``mu`` (``l``) or ``eta`` (``k``) degree is odd."""
    rows = list(index.h0) + list(index.h2) * 2
    return np.array([any(row[i] % 2 for i in flipped) for row in rows])


@pytest.mark.parametrize("orders", sorted(PINNED_SYMMETRIC))
def test_symmetric_factorised_counts_are_pinned(orders):
    index = MomentIndex.build(Truncation(*orders))
    L_mu, L_eta, N = orders
    base = fully_independent(index).n_free()
    assert base == 2 * N + N + L_mu + L_eta + 2
    derived = {
        "field": base - (L_mu + 1) // 2 - (L_eta + 1) // 2,
        "pitch": base - (L_mu + 1) // 2,
        "both": base - (L_mu + 1) // 2 - (L_eta + 1) // 2,
    }
    maps = symmetric_maps(index)
    for kind, pinned in PINNED_SYMMETRIC[orders].items():
        pm = maps[kind]
        assert pm.n_free() == pinned == derived.get(kind, pinned), kind
        assert len(pm.labels()) == pm.n_free()
    labels = maps["field"].labels()
    assert "<P_1(mu)>" not in labels and "<P_1(eta)>" not in labels
    assert ("<P_2(mu)>" in labels) == (L_mu >= 2)
    assert ("<P_2(eta)>" in labels) == (L_eta >= 2)


@pytest.mark.parametrize("kind", ["field", "eta_apart"])
def test_map_range_respects_the_declared_symmetry(kind):
    """Every output of the map has zero odd-``l`` and odd-``k`` rows."""
    index = MomentIndex.build(Truncation(2, 2, 2))
    pm = symmetric_maps(index)[kind]
    vector = np.random.default_rng(3).normal(size=pm.n_free())
    m = np.asarray(pm(pm.unflatten(vector), REF).to_vector())
    odd = odd_rows(index, (0, 1))
    assert np.all(m[odd] == 0.0)
    assert np.max(np.abs(m[~odd])) > 0.1


@pytest.mark.parametrize("orders", [(1, 1, 0), (3, 1, 1), (2, 2, 2), (4, 4, 3)])
@pytest.mark.parametrize("kind", ["field", "pitch", "both", "eta_apart"])
def test_symmetric_product_population_is_reproduced_and_identifiable(kind, orders):
    index = MomentIndex.build(Truncation(*orders))
    pm = symmetric_maps(index)[kind]
    joint = JointMoments.from_samples(product_population(True), index, REF)
    theta = pm.project(joint)
    m_joint = np.asarray(joint.to_vector())
    m_map = np.asarray(pm(theta, REF).to_vector())
    # float64 weighted sums of 768 atoms: the mirrored rows cancel to ~1e-16
    assert_allclose(m_map, m_joint, rtol=0, atol=1e-14 * np.max(np.abs(m_joint)))
    vector = pm.flatten(theta)

    def m_of(v):
        return pm(pm.unflatten(v), REF).to_vector()

    jac = np.asarray(jax.jacfwd(m_of)(vector))
    assert np.linalg.matrix_rank(jac, tol=1e-10 * np.max(np.abs(jac))) == pm.n_free()


def test_asymmetric_product_population_shows_the_joint_discrepancy():
    """``M_{1,1} = <P_1(mu)> <P_1(eta)>`` is even in ``l + k`` but not allowed."""
    index = MomentIndex.build(Truncation(2, 2, 2))
    pm = symmetric_maps(index)["field"]
    samples = product_population(False)
    joint = JointMoments.from_samples(samples, index, REF)
    measured = JointMoments.from_samples(
        samples, index, REF, parameter_map=pm, discrepancy="measured"
    )
    delta = np.asarray(measured.discrepancy.value)
    m_joint = np.asarray(joint.to_vector())
    odd = odd_rows(index, (0, 1))
    assert_allclose(delta[odd], np.abs(m_joint[odd]), rtol=1e-14, atol=1e-17)
    slot = index.h0.index((1, 1, 0, 0, 0))
    assert odd[slot] and delta[slot] > 1e-3
    # kept rows are products of the projected marginals: rounding only
    assert np.max(delta[~odd]) < 1e-14 * np.max(np.abs(m_joint))
    assert pm.record().discrepancy_kind == "unbounded"


FULLY_INDEPENDENT = [
    ((2, 2, 2), {}, 12),
    ((2, 2, 2), {"max_orders": (1, 1, None)}, 10),
    ((2, 2, 3), {"max_orders": (0, None, None)}, 12),
    ((2, 2, 3, 1), {"max_orders": (1, 2, None)}, 10),
    ((2, 2, 3), {"max_orders": (None, None, 1)}, 13),
    ((8, 8, 2), {}, 24),
    ((8, 8, 2), {"max_orders": (1, 0, 1)}, 20),
]


@pytest.mark.parametrize("orders, kwargs, pinned", FULLY_INDEPENDENT)
def test_fully_independent_counts_follow_the_caps(orders, kwargs, pinned):
    truncation = Truncation(*orders, **kwargs)
    n_gamma, n_b, max_b = truncation.caps()
    derived = n_gamma + n_b + max_b + truncation.L_mu + truncation.L_eta + 2
    index = MomentIndex.build(truncation)
    pm = fully_independent(index)
    assert pm.n_free() == pinned == derived
    # field reversal removes the odd Legendre degrees of both marginals
    odd = (truncation.L_mu + 1) // 2 + (truncation.L_eta + 1) // 2
    field = pm.assume(field_reversal_symmetric(index))
    assert field.n_free() == derived - odd
