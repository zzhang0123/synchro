"""``fixed_table`` closures combined with a symmetry closure.

Semantics under test. A closure combined with a symmetry represents its
symmetrised marginal: a ``fixed_table`` always takes the full table, the
shape it has in the same partition without the symmetry closures, and the
entries that the symmetry forces to zero (odd flipped exponents on the
group, see ``_factor.drops_row``) are dropped. The reduced table (only the
kept entries) is refused on both routes, because the map without the
symmetry, which ``assume`` combines, cannot take it. The record keeps the
full table and names the dropped entries.

Oracle. The symmetrised table (dropped entries set to zero) in the same
partition without the symmetry gives the same moments when every free
entry the symmetry drops is also set to zero: a dropped row then has a zero
group factor in both maps, and every kept row is the same product of the
same numbers. So ``m(symmetric map) == m(plain map, symmetrised tables)``
on every row, independently of the row-dropping code.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model import _factor as _f
from syncmoments.model.assumptions import (
    VARS,
    Closure,
    Factorisation,
    ParameterMap,
    Parameters,
    field_reversal_symmetric,
    pitch_symmetric,
)
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import Reference

REF = Reference(20.0, 2.0, depth_ref=1.5, scales=(2.0, 0.2, 0.5))
SINGLETONS = tuple((v,) for v in VARS)
MU_ETA = (("gamma",), ("B",), ("mu", "eta"), ("phi",), ("depth",))
MU_APART = (("gamma", "B", "eta", "phi", "depth"), ("mu",))
SYMMETRY = {"field": field_reversal_symmetric, "pitch": pitch_symmetric}
# partitions in canonical order, so ``groups.index`` matches the built map
assert all(_f.canonical_partition(g) == g for g in (SINGLETONS, MU_ETA, MU_APART))

# (symmetry, partition, closed group): mu or eta singletons, a joint
# (mu, eta) group, and a one-free-group (affine) partition.
CASES = [
    ("field", SINGLETONS, ("mu",)),
    ("field", SINGLETONS, ("eta",)),
    ("pitch", SINGLETONS, ("mu",)),
    ("field", MU_ETA, ("mu", "eta")),
    ("pitch", MU_ETA, ("mu", "eta")),
    ("field", MU_APART, ("mu",)),
]
# extreme low (every mu entry dropped at L_mu = 1), moderate, benchmark size
ORDERS = [(1, 1, 0), (3, 1, 1), (2, 2, 2), (8, 8, 2)]


def full_spec(index, groups, group):
    """Entries of ``group`` in the partition without any symmetry closure."""
    spec = _f.build_tables(index, groups, ())[0]
    return spec[groups.index(group)]


def fixed_closure(index, groups, group):
    """Full-length ``fixed_table`` over the entries without the symmetry."""
    spec = full_spec(index, groups, group)
    values = 0.1 + 0.05 * np.arange(len(spec))
    return Closure(group, "fixed_table", (jnp.asarray(values),)), spec


def symmetrise(closure, spec, kept):
    table = np.asarray(closure.hyper[0]).copy()
    for j, p in enumerate(spec):
        if p not in kept:
            table[j] = 0.0
    return Closure(closure.group, "fixed_table", (jnp.asarray(table),))


def with_odd_entries(closure, spec, kept, value):
    table = np.asarray(closure.hyper[0]).copy()
    for j, p in enumerate(spec):
        if p not in kept:
            table[j] = value
    return Closure(closure.group, "fixed_table", (jnp.asarray(table),))


def both_routes(index, kind, groups, closure):
    sym = SYMMETRY[kind](index)
    factorisation = Factorisation(groups, "fact")
    built = ParameterMap.build(index, factorisation, (closure, *sym.closures))
    plain = ParameterMap.build(index, factorisation, (closure,))
    assumed = plain.assume(sym)
    return built, assumed


def random_theta(pm, seed=0):
    return pm.unflatten(np.random.default_rng(seed).normal(size=pm.n_free()))


def plain_theta(pm, plain, theta):
    """``theta`` of the symmetric map on the plain map; dropped entries 0."""
    own = dict(zip(pm.free_groups(), theta.tables))
    tables = []
    for g in plain.free_groups():
        spec, mine = plain.tables_spec[g], pm.tables_spec[g]
        complex_ = any(plain._complex_mask(g))
        values = np.zeros(len(spec), dtype=complex if complex_ else float)
        for j, p in enumerate(spec):
            if g in own and p in mine:
                values[j] = own[g][mine.index(p)]
        tables.append(jnp.asarray(values))
    return Parameters(tables=tuple(tables))


def record_fields(pm):
    r = pm.record()
    return (r.groups, r.closure_kind, r.hyper, r.n_free, r.discrepancy_kind)


@pytest.mark.parametrize("orders", ORDERS)
@pytest.mark.parametrize("kind, groups, group", CASES)
def test_both_routes_accept_the_full_table_and_agree(kind, groups, group, orders):
    index = MomentIndex.build(Truncation(*orders))
    closure, spec = fixed_closure(index, groups, group)
    built, assumed = both_routes(index, kind, groups, closure)
    g = groups.index(group)
    kept = built.tables_spec[g]
    dropped = [p for p in spec if p not in kept]
    for attr in ("tables_spec", "gathers", "keep", "ext_ok"):
        assert getattr(built, attr) == getattr(assumed, attr), attr
    assert built.n_free() == assumed.n_free()
    assert built.labels() == assumed.labels()
    assert record_fields(built) == record_fields(assumed)
    # the record keeps the full table and names every dropped entry
    record = built.record()
    assert len(record.hyper) == len(spec)
    if dropped:
        assert "symmetrised" in record.closure_kind
        for p in dropped:
            assert _f.entry_label(group, p) in record.closure_kind
    else:
        assert "symmetrised" not in record.closure_kind

    theta = random_theta(built)
    m_built = np.asarray(built(theta, REF).to_vector())
    m_assumed = np.asarray(assumed(theta, REF).to_vector())
    assert np.array_equal(m_built, m_assumed)

    factorisation = Factorisation(groups, "fact")
    plain = ParameterMap.build(index, factorisation, (symmetrise(closure, spec, kept),))
    m_plain = np.asarray(plain(plain_theta(built, plain, theta), REF).to_vector())
    scale = max(np.max(np.abs(m_plain)), 1.0)
    assert_allclose(m_built, m_plain, rtol=0, atol=1e-15 * scale)


@pytest.mark.parametrize("orders", [(1, 1, 0), (2, 2, 2), (8, 8, 2)])
@pytest.mark.parametrize("kind, groups, group", CASES)
def test_nonzero_dropped_entries_do_not_change_the_prediction(
    kind, groups, group, orders
):
    """The symmetrised marginal of a table ignores its dropped entries."""
    index = MomentIndex.build(Truncation(*orders))
    closure, spec = fixed_closure(index, groups, group)
    zero, _ = both_routes(index, kind, groups, closure)
    kept = zero.tables_spec[groups.index(group)]
    loud = with_odd_entries(closure, spec, kept, 7.5)
    built, assumed = both_routes(index, kind, groups, loud)
    theta = random_theta(zero, seed=4)
    m_zero = np.asarray(zero(theta, REF).to_vector())
    for pm in (built, assumed):
        assert pm.n_free() == zero.n_free()
        assert np.array_equal(np.asarray(pm(theta, REF).to_vector()), m_zero)
    assert record_fields(built) == record_fields(assumed)
    if len(kept) < len(spec):  # the recorded input differs, the moments do not
        assert built.record().hyper != zero.record().hyper
        assert 7.5 in built.record().hyper


@pytest.mark.parametrize("orders", [(1, 1, 0), (2, 2, 2), (8, 8, 2)])
@pytest.mark.parametrize("kind, groups, group", CASES)
def test_other_table_shapes_are_refused_on_both_routes(kind, groups, group, orders):
    index = MomentIndex.build(Truncation(*orders))
    closure, spec = fixed_closure(index, groups, group)
    built, _ = both_routes(index, kind, groups, closure)
    n_kept = len(built.tables_spec[groups.index(group)])
    n = len(spec)
    shapes = {(n + 1,), (n, 1), (1, n)}
    if n_kept != n:
        shapes.add((n_kept,))  # the reduced table: refused, see module docstring
    for shape in shapes:
        bad = Closure(group, "fixed_table", (jnp.zeros(shape),))
        with pytest.raises(ValueError, match=rf"shape \({n},\)"):
            both_routes(index, kind, groups, bad)
        with pytest.raises(ValueError, match=rf"shape \({n},\)"):
            ParameterMap.build(
                index,
                Factorisation(groups, "fact"),
                (bad, *SYMMETRY[kind](index).closures),
            )


def test_reduced_table_error_names_the_dropped_entries():
    index = MomentIndex.build(Truncation(2, 2, 2))
    bad = Closure(("mu",), "fixed_table", (jnp.ones(1),))
    closures = (bad, *field_reversal_symmetric(index).closures)
    with pytest.raises(ValueError, match=r"shape \(2,\).*<P_1\(mu\)>"):
        ParameterMap.build(index, Factorisation(SINGLETONS, "fi"), closures)


def test_symmetrised_fixed_table_is_traceable_and_affine():
    """``jit`` over the map (table as a leaf) and ``affine_pieces`` agree."""
    index = MomentIndex.build(Truncation(2, 2, 2))
    closure, spec = fixed_closure(index, MU_APART, ("mu",))
    closure = with_odd_entries(closure, spec, ((2,),), -3.0)
    built, assumed = both_routes(index, "field", MU_APART, closure)
    assert built.is_affine() and assumed.is_affine()
    theta = random_theta(built, seed=9)
    eager = np.asarray(built(theta, REF).to_vector())
    traced = eqx.filter_jit(lambda pm, th: pm(th, REF).to_vector())(built, theta)
    assert_allclose(np.asarray(traced), eager, rtol=1e-15, atol=1e-300)
    P, c = built.affine_pieces()
    affine = np.asarray(P @ built.flatten(theta) + c)
    assert_allclose(affine, eager, rtol=1e-14, atol=1e-15)


def test_groups_without_dropped_entries_keep_their_table_shape():
    """A table on ``gamma`` is unchanged by a symmetry (no flipped variable)."""
    index = MomentIndex.build(Truncation(2, 2, 2))
    closure, spec = fixed_closure(index, SINGLETONS, ("gamma",))
    built, assumed = both_routes(index, "field", SINGLETONS, closure)
    assert built.tables_spec[0] == spec
    assert "symmetrised" not in built.record().closure_kind
    assert record_fields(built) == record_fields(assumed)
