"""Symmetry assumptions ``pitch_symmetric`` and ``field_reversal_symmetric``.

Oracles: mirrored discrete populations (exactly symmetric by construction),
exact weighted sums, the parity of ``P_l(-x) = (-1)^l P_l(x)`` and
``e^{2i(phi + pi)} = e^{2i phi}``, and an independent free-parameter count.

Free-parameter count (one group ``VARS``, no factorisation): every retained
row of ``m`` except the unit moment is free, a complex ``M2`` row counts twice.
With ``p0`` the kept ``M0`` pairs, ``p2`` the kept ``M2`` pairs (``l + k``
even by ``eq: angular parity``), ``n_rs = C(N+2, 2)`` and ``n_rsb =
C(N+3, 3)``: ``n_free = p0 n_rs - 1 + 2 p2 n_rsb``. Pitch symmetry keeps
``l`` even, field reversal keeps ``l + k`` even, both keep ``l, k`` even.
At ``(2, 2, 2)`` (``n_rs = 6``, ``n_rsb = 10``): no assumption ``9*6 - 1 +
2*5*10 = 153``; pitch ``6*6 - 1 + 2*4*10 = 115``; field reversal ``5*6 - 1 +
2*5*10 = 129``; both ``4*6 - 1 + 2*4*10 = 103``. At ``(8, 8, 2)``: ``81*6 - 1
+ 2*41*10 = 1305``; ``45*6 - 1 + 2*25*10 = 769``; ``41*6 - 1 + 2*41*10 =
1065``; ``25*6 - 1 + 2*25*10 = 649``.
"""

import itertools
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model.assumptions import (
    VARS,
    Closure,
    Factorisation,
    ParameterMap,
    field_reversal_symmetric,
    independent_screen,
    isotropic_pitch,
    no_assumption,
    pitch_symmetric,
)
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference
from syncmoments.model.predict import predict

from _basis_fixtures import polynomial_setup
from _basis_stub import build_stub_basis

PINNED = {
    (2, 2, 2): {"none": 153, "pitch": 115, "field": 129, "both": 103},
    (8, 8, 2): {"none": 1305, "pitch": 769, "field": 1065, "both": 649},
}


def keeps(kind, l, k):
    return {
        "none": True,
        "pitch": l % 2 == 0,
        "field": (l + k) % 2 == 0,
        "both": l % 2 == 0 and k % 2 == 0,
    }[kind]


def count_free(L_mu, L_eta, N, kind):
    pairs = list(itertools.product(range(L_mu + 1), range(L_eta + 1)))
    p0 = sum(keeps(kind, l, k) for l, k in pairs)
    p2 = sum(keeps(kind, l, k) for l, k in pairs if (l + k) % 2 == 0)
    return p0 * comb(N + 2, 2) - 1 + 2 * p2 * comb(N + 3, 3)


def maps(index):
    return {
        "none": no_assumption(index),
        "pitch": pitch_symmetric(index),
        "field": field_reversal_symmetric(index),
        "both": pitch_symmetric(index).assume(field_reversal_symmetric(index)),
    }


@pytest.mark.parametrize("orders", sorted(PINNED))
def test_free_parameter_counts_are_pinned_and_derived(orders):
    index = MomentIndex.build(Truncation(*orders))
    for kind, pm in maps(index).items():
        assert pm.n_free() == PINNED[orders][kind] == count_free(*orders, kind)
        assert len(pm.labels()) == pm.n_free()


def reference():
    return Reference(20.0, 2.0, depth_ref=1.5, scales=(2.0, 0.2, 0.5))


def atoms(seed=4, n=9):
    """Correlated, asymmetric atoms (odd ``l`` and odd ``l + k`` moments nonzero)."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, n)
    return dict(
        gamma=20.0 + 3.0 * t,
        B=2.0 + 0.3 * t**2,
        mu=np.clip(0.2 + 0.6 * t, -0.95, 0.95),
        eta=np.clip(0.3 - 0.5 * t**2, -0.95, 0.95),
        phi=0.4 + 1.3 * t,
        depth=1.5 + 0.8 * t,
        w=rng.uniform(0.3, 2.0, n),
    )


def mirrored(pop, kind):
    image = dict(pop)
    image["mu"] = -pop["mu"]
    if kind == "field":
        image["eta"] = -pop["eta"]
        image["phi"] = pop["phi"] + np.pi
    return {key: np.concatenate([pop[key], image[key]]) for key in pop}


def samples_of(pop):
    keys = ("gamma", "B", "mu", "eta", "phi", "depth")
    return PopulationSamples(*(pop[key] for key in keys), weights=pop["w"])


def dropped_mask(index, kind):
    rows = list(index.h0) + list(index.h2) * 2
    return np.array([not keeps(kind, row[0], row[1]) for row in rows])


@pytest.mark.parametrize("kind", ["pitch", "field"])
def test_symmetric_populations_give_exact_zeros_and_reproduce_the_prediction(kind):
    kernel, _, _, ref, channels, support = polynomial_setup(L=2, N=2)
    truncation = Truncation(2, 2, 2)
    basis = build_stub_basis(kernel, channels, truncation, ref, support)
    index = basis.index
    pm = maps(index)[kind]
    samples = samples_of(mirrored(atoms(), kind))
    joint = JointMoments.from_samples(samples, index, ref)
    mapped = JointMoments.from_samples(samples, index, ref, parameter_map=pm)
    drop = dropped_mask(index, kind)
    assert drop.any()
    m_joint, m_map = np.asarray(joint.to_vector()), np.asarray(mapped.to_vector())
    assert np.all(m_map[drop] == 0.0)
    assert np.max(np.abs(m_joint[drop])) < 1e-14 * np.max(np.abs(m_joint))
    # slot 0 is the constant 1 in the map; the weighted sum gives 1 within an ulp
    assert_allclose(m_map[~drop], m_joint[~drop], rtol=0, atol=2.3e-16)
    a = np.asarray(predict(basis, joint, amplitude=1.7).stokes)
    b = np.asarray(predict(basis, mapped, amplitude=1.7).stokes)
    assert_allclose(b, a, rtol=0, atol=1e-13 * np.max(np.abs(a)))
    if kind == "field":
        assert np.all(b[:, 3] == 0.0)  # every V row has l + k odd
    record = mapped.assumptions[0]
    assert record.discrepancy_kind == "unbounded" and record.groups == (VARS,)
    assert record.closure_kind == pm.closures[0].kind


@pytest.mark.parametrize("kind", ["pitch", "field", "both"])
def test_asymmetric_populations_show_the_measured_discrepancy(kind):
    ref, index = reference(), MomentIndex.build(Truncation(2, 2, 2))
    samples = samples_of(atoms())
    joint = JointMoments.from_samples(samples, index, ref)
    measured = JointMoments.from_samples(
        samples, index, ref, parameter_map=maps(index)[kind], discrepancy="measured"
    )
    delta = np.asarray(measured.discrepancy.value)
    drop = dropped_mask(index, kind)
    m_joint = np.asarray(joint.to_vector())
    assert_allclose(delta[drop], np.abs(m_joint[drop]), rtol=1e-14)
    assert np.all(delta[~drop] == 0.0)
    assert np.max(delta) > 1e-2
    assert measured.discrepancy.kind == "measured"
    assert maps(index)[kind].record().discrepancy_kind == "unbounded"


def test_field_reversal_zeroes_the_v_response_that_the_population_has():
    kernel, _, _, ref, channels, support = polynomial_setup(L=2, N=2)
    basis = build_stub_basis(kernel, channels, Truncation(2, 2, 2), ref, support)
    samples = samples_of(atoms())
    joint = JointMoments.from_samples(samples, basis.index, ref)
    pm = field_reversal_symmetric(basis.index)
    mapped = JointMoments.from_samples(samples, basis.index, ref, parameter_map=pm)
    assert (
        np.max(np.abs(np.asarray(predict(basis, joint, amplitude=1.0).stokes)[:, 3]))
        > 0
    )
    assert np.all(np.asarray(predict(basis, mapped, amplitude=1.0).stokes)[:, 3] == 0.0)


def test_symmetries_combine_with_factorisations_and_closures():
    ref, index = reference(), MomentIndex.build(Truncation(2, 2, 2))
    samples = samples_of(atoms())
    screen = independent_screen(index)
    combined = pitch_symmetric(index).assume(screen)
    assert combined.factorisation.groups == screen.factorisation.groups
    assert [c.kind for c in combined.closures] == ["pitch_symmetric"]
    expected = np.array(
        JointMoments.from_samples(samples, index, ref, parameter_map=screen).to_vector()
    )
    got = np.asarray(
        JointMoments.from_samples(
            samples, index, ref, parameter_map=combined
        ).to_vector()
    )
    drop = dropped_mask(index, "pitch")
    expected[drop] = 0.0
    assert_allclose(got, expected, rtol=1e-14, atol=1e-16)
    both = pitch_symmetric(index).assume(field_reversal_symmetric(index))
    assert both.record().closure_kind == "pitch_symmetric+field_reversal_symmetric"
    same = pitch_symmetric(index).assume(pitch_symmetric(index))
    assert len(same.closures) == 1 and same.n_free() == PINNED[(2, 2, 2)]["pitch"]
    iso = isotropic_pitch(index).assume(pitch_symmetric(index))
    assert iso.n_free() == isotropic_pitch(index).n_free()


def test_symmetry_closures_are_validated():
    index = MomentIndex.build(Truncation(1, 1, 1))
    one = Factorisation((VARS,), "custom")
    ParameterMap.build(index, one, (Closure((), "field_reversal_symmetric"),))
    with pytest.raises(ValueError):
        ParameterMap.build(index, one, (Closure((), "pitch_symmetric", (1.0,)),))
    with pytest.raises(ValueError):
        ParameterMap.build(index, one, (Closure(("mu",), "pitch_symmetric"),))
    with pytest.raises(ValueError):
        ParameterMap.build(
            index, one, (Closure((), "pitch_symmetric"), Closure((), "pitch_symmetric"))
        )


def test_symmetric_maps_are_affine_jittable_and_differentiable():
    ref, index = reference(), MomentIndex.build(Truncation(2, 2, 2))
    pm = pitch_symmetric(index).assume(field_reversal_symmetric(index))
    assert pm.is_affine()
    theta = pm.project(JointMoments.from_samples(samples_of(atoms()), index, ref))
    vector = pm.flatten(theta)
    P, c = pm.affine_pieces()
    eager = np.asarray(pm(theta, ref).to_vector())
    assert_allclose(np.asarray(P @ vector + c), eager, rtol=1e-14, atol=1e-16)

    def m_of(v):
        return pm(pm.unflatten(v), ref).to_vector()

    assert_allclose(np.asarray(jax.jit(m_of)(vector)), eager, rtol=1e-14, atol=1e-16)
    jac = jax.jacfwd(m_of)(vector)
    assert_allclose(np.asarray(jac), np.asarray(P), rtol=1e-14, atol=1e-16)
    assert jnp.all(jac[jnp.asarray(dropped_mask(index, "both"))] == 0.0)
