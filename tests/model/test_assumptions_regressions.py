"""Regression tests for review findings on ``synchro.model.assumptions``.

R01: ``assume()`` must not hand one input map's discrepancy allowance to a
combined assumption that the other maps constrain further; the combined
allowance is then unbounded. R15/NEW-1: a fitted depth hyper-parameter is
recorded as a NaN placeholder with the closure kind suffixed ``" (fitted)"``,
so the record (a static field of ``JointMoments``) does not depend on
``theta``: moments of different fitted values difference, stack and share
one ``filter_jit`` trace; the fitted values live in ``theta`` and in the
``FitResult`` provenance notes. R27: the free-parameter count of
``fully_independent`` in the docstring holds in the ``app: depth moments``
layout. Oracles: exact weighted sums of ``JointMoments.from_samples`` on a
discrete population, and an independent parameter count.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from synchro.model.assumptions import (
    Closure,
    ParameterMap,
    fully_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
)
from synchro.model.errors import ErrorTerm
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference


def reference():
    return Reference(5.2, 2.1, 0.3, scales=(1.3, 0.7, 0.9))


def anisotropic_population(seed=3, n=2000):
    """Screen independent of the rest; pitch anisotropic (``mu ~ N(0.5, 0.3)``)."""
    rng = np.random.default_rng(seed)
    return PopulationSamples(
        5.0 + 0.5 * rng.standard_normal(n),
        2.0 + 0.3 * rng.standard_normal(n),
        np.clip(0.5 + 0.3 * rng.standard_normal(n), -0.99, 0.99),
        np.clip(0.2 * rng.standard_normal(n), -0.99, 0.99),
        rng.uniform(0.0, 2.0 * np.pi, n),
        0.3 + 0.8 * rng.standard_normal(n),
        weights=np.ones(n),
    )


def with_term(pm, term):
    return ParameterMap.build(
        pm.index,
        pm.factorisation,
        pm.closures,
        discrepancy=term,
        name=pm.name,
        label=pm.LABEL,
        free_hyper=pm.free_hyper,
    )


# -- R01 ------------------------------------------------------------------------------


def test_assume_drops_allowance_when_other_map_adds_constraint():
    index = MomentIndex.build(Truncation(2, 2, 2))
    ref = reference()
    samples = anisotropic_population()
    screen = independent_screen(index)
    measured = JointMoments.from_samples(
        samples, index, ref, parameter_map=screen, discrepancy="measured"
    ).discrepancy
    screen_d = with_term(screen, measured)
    both = screen_d.assume(isotropic_pitch(index))
    # The measured screen allowance does not cover the isotropic-pitch error.
    joint = JointMoments.from_samples(samples, index, ref)
    fac = both(both.project(joint), ref)
    error = np.abs(np.asarray(joint.to_vector() - fac.to_vector()))
    assert np.max(error) > 10.0 * float(np.max(np.asarray(measured.value)))
    assert both.discrepancy is None
    assert both.record().discrepancy_kind == "unbounded"
    assert fac.discrepancy is None
    reverse = isotropic_pitch(index).assume(screen_d)
    assert reverse.discrepancy is None
    assert reverse.record().discrepancy_kind == "unbounded"


def test_assume_drops_allowance_when_other_map_adds_a_closure():
    index = MomentIndex.build(Truncation(2, 2, 2))
    term = ErrorTerm(jnp.full(index.n_real, 1e-3), "estimate", "screen")
    screen_d = with_term(independent_screen(index), term)
    closed = screen_d.assume(gaussian_screen(index, 0.8, 0.6))
    assert closed.discrepancy is None
    assert closed.record().discrepancy_kind == "unbounded"


@pytest.mark.parametrize("other", ["no_assumption", "same_partition", "self_plain"])
def test_assume_keeps_allowance_when_others_add_no_constraint(other):
    index = MomentIndex.build(Truncation(2, 2, 2))
    term = ErrorTerm(jnp.full(index.n_real, 1e-3), "estimate", "screen")
    gauss_d = with_term(gaussian_screen(index, 0.8, 0.6), term)
    others = {
        "no_assumption": no_assumption(index),
        "same_partition": independent_screen(index),
        "self_plain": gaussian_screen(index, 0.8, 0.6),
    }
    combined = gauss_d.assume(others[other])
    assert combined.discrepancy is term
    assert combined.record().discrepancy_kind == "estimate"
    combined = others[other].assume(gauss_d)
    assert combined.discrepancy is term


# -- R15 / NEW-1 ----------------------------------------------------------------------


def fitted_theta(pm, hyper):
    vector = np.zeros(pm.n_free())
    vector[-len(hyper) :] = hyper
    return pm.unflatten(jnp.asarray(vector))


def fitted_map():
    index = MomentIndex.build(Truncation(1, 1, 2))
    return gaussian_screen(index, 0.5, 0.2, fit_hyper=True)


def test_fitted_hyper_record_is_a_theta_independent_placeholder():
    pm = fitted_map()
    out = pm(fitted_theta(pm, (5.0, 1.0)), reference())
    fixed = gaussian_screen(pm.index, 5.0, 1.0)
    np.testing.assert_allclose(
        np.asarray(out.m0_ext),
        np.asarray(fixed(fixed.project(out), reference()).m0_ext),
        atol=1e-12,
    )
    record = out.assumptions[0]
    assert record.closure_kind == "gaussian_depth (fitted)"
    assert len(record.hyper) == 2 and all(np.isnan(h) for h in record.hyper)
    assert record == pm.record()
    other = pm(fitted_theta(pm, (4.0, 2.0)), reference())
    assert other.assumptions == out.assumptions


def test_moments_of_two_fitted_thetas_share_one_pytree_structure():
    """NEW-1: differencing and stacking moments of different fitted hyper."""
    pm, ref = fitted_map(), reference()
    thetas = [(5.0, 1.0), (4.0, 2.0), (3.0, 1.5), (2.0, 0.5)]
    ms = [pm(fitted_theta(pm, h), ref) for h in thetas]
    diff = jax.tree.map(lambda a, b: a - b, ms[0], ms[1])
    np.testing.assert_allclose(
        np.asarray(diff.m0_ext), np.asarray(ms[0].m0_ext - ms[1].m0_ext), atol=0
    )
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *ms)
    assert stacked.m2.shape == (len(thetas),) + ms[0].m2.shape
    traces = {"n": 0}

    @eqx.filter_jit
    def total(m):
        traces["n"] += 1
        return jnp.sum(m.to_vector())

    for m in ms:
        total(m)
    assert traces["n"] == 1


def test_record_of_traced_fitted_hyper_matches_the_eager_record():
    pm, ref = fitted_map(), reference()
    call = jax.jit(lambda t: pm(t, ref))
    first = call(fitted_theta(pm, (5.0, 1.0)))
    second = call(fitted_theta(pm, (4.0, 2.0)))
    assert first.assumptions == second.assumptions == (pm.record(),)
    closed = jax.jit(
        lambda t: jax.lax.cond(True, lambda: pm(t, ref), lambda: pm(t, ref))
    )(fitted_theta(pm, (5.0, 1.0)))
    assert closed.assumptions == first.assumptions


def test_record_of_non_fitted_closure_keeps_constructor_hyper():
    index = MomentIndex.build(Truncation(1, 1, 2))
    screen = Closure(("depth",), "gaussian_depth", (jnp.asarray(0.5), jnp.asarray(0.2)))
    pm = independent_screen(index, screen=screen)
    theta = pm.unflatten(jnp.zeros(pm.n_free()))
    record = pm(theta, reference()).assumptions[0]
    assert record.hyper == (0.5, 0.2) and record.closure_kind == "gaussian_depth"
    assert jax.jit(lambda t: pm(t, reference()))(theta).assumptions[0].hyper == (
        0.5,
        0.2,
    )


# -- R27 ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orders", [(2, 2, 2, None), (2, 2, 2, 4), (2, 1, 2, 3), (2, 2, 3, 0), (1, 1, 2, 5)]
)
def test_fully_independent_count_matches_docstring(orders):
    L_mu, L_eta, N, depth_degree = orders
    truncation = Truncation(L_mu, L_eta, N, depth_degree=depth_degree)
    pm = fully_independent(MomentIndex.build(truncation))
    assert pm.n_free() == 2 * N + truncation.max_b() + L_mu + L_eta + 2
    assert "2N + max_b + L_mu + L_eta + 2" in fully_independent.__doc__
