"""Error terms, budgets and provenance records against plain-Python oracles."""

import dataclasses
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model.errors import (
    KINDS,
    MANUSCRIPT_TERMS,
    WEAKEST_ORDER,
    AssumptionRecord,
    ErrorBudget,
    ErrorTerm,
    Provenance,
)

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def budget_with(**overrides):
    """Nine zero bounds, then replace the named slots."""
    fields = {
        name: ErrorTerm.declared_zero(f"{name} declared zero")
        for name in ErrorBudget.SLOTS
    }
    fields.update(overrides)
    return ErrorBudget(**fields)


def record(name="independent_screen", **kw):
    base = dict(
        name=name,
        label="eq: independent screen moments",
        groups=(("depth",), ("gamma", "B", "mu", "eta", "phi")),
        closure_kind="none",
        hyper=(),
        n_free=115,
        discrepancy_kind="unbounded",
    )
    base.update(kw)
    return AssumptionRecord(**base)


def provenance(**kw):
    base = dict(
        package_version="0.1.0",
        kernel=(("name", "harmonic"), ("m_max", 40)),
        channels=(("family", "bump"), ("n_ch", 3)),
        truncation=(("L_mu", 2), ("L_eta", 2), ("N", 2), ("depth_degree", None)),
        reference=(("gamma0", 20.0), ("B0", 1.0)),
        support=(("gamma", (10.0, 30.0)), ("truncated", None)),
        phase_route=(("name", "taylor"), ("degree", 2)),
        quadrature=(("name", "product"), ("n_outer", 64)),
        assumptions=(record(),),
        units="erg/s/sr per electron",
        numerics=(("n_nodes", 256), ("m_range", (1, 40))),
        certified_orders=(0, 1, 2),
        finite_checks=("benchmark",),
        notes=("incident polarisation not modelled",),
    )
    base.update(kw)
    return Provenance(**base)


# ----------------------------------------------------------------------
# ErrorTerm
# ----------------------------------------------------------------------


def test_kinds_and_order_constants():
    assert KINDS == ("bound", "estimate", "measured", "unbounded", "not_applicable")
    assert WEAKEST_ORDER == ("bound", "measured", "estimate")
    assert set(MANUSCRIPT_TERMS) == set(ErrorBudget.SLOTS) | {"assumption"}
    assert MANUSCRIPT_TERMS["harmonic_truncation"] == "E_num"


def test_unbounded_and_not_applicable_have_no_value():
    u = ErrorTerm.unbounded("no tail bound supplied", manuscript_term="E_tail")
    assert u.value is None and u.kind == "unbounded"
    assert u.note == "no tail bound supplied" and u.manuscript_term == "E_tail"
    n = ErrorTerm.not_applicable("direct average")
    assert n.value is None and n.kind == "not_applicable" and n.note == "direct average"
    assert jax.tree_util.tree_leaves(u) == []


def test_declared_zero_is_a_zero_bound():
    z = ErrorTerm.declared_zero("continuum has no harmonic sum")
    assert z.kind == "bound"
    assert z.value.shape == (1, 4)
    assert_allclose(np.asarray(z.value), 0.0)
    assert "continuum" in z.note
    z3 = ErrorTerm.declared_zero("x", shape=(3, 4))
    assert z3.value.shape == (3, 4)


@pytest.mark.parametrize("kind", ["bound", "estimate", "measured"])
def test_valued_kinds_store_float_arrays(kind):
    t = ErrorTerm([[1, 2, 3, 4]], kind, note="n", manuscript_term="E_num")
    assert t.kind == kind and t.value.shape == (1, 4)
    assert t.value.dtype == jnp.float64
    assert_allclose(np.asarray(t.value), [[1, 2, 3, 4]])


@pytest.mark.parametrize(
    "value,kind",
    [
        (None, "bound"),
        ([1.0], "unbounded"),
        ([1.0], "not_applicable"),
        ([1.0], "exact"),
        ([1.0 + 1j], "bound"),
        ([1.0], 3),
    ],
)
def test_error_term_rejects_inconsistent_construction(value, kind):
    with pytest.raises(ValueError):
        ErrorTerm(value, kind)


def test_error_term_rejects_bad_static_strings():
    with pytest.raises(ValueError):
        ErrorTerm([1.0], "bound", note=3)
    with pytest.raises(ValueError):
        ErrorTerm([1.0], "bound", manuscript_term=None)


def test_error_term_rejects_negative_and_nonfinite_values():
    with pytest.raises(Exception):
        ErrorTerm(jnp.array([-1.0]), "bound")
    with pytest.raises(Exception):
        ErrorTerm(jnp.array([jnp.nan]), "estimate")
    with pytest.raises(Exception):
        ErrorTerm(jnp.array([jnp.inf]), "measured")


def test_scaled_keeps_kind_and_note():
    t = ErrorTerm(jnp.ones((2, 4)), "estimate", note="probe", manuscript_term="E_num")
    s = t.scaled(3.0)
    assert s.kind == "estimate" and s.note == "probe" and s.manuscript_term == "E_num"
    assert_allclose(np.asarray(s.value), 3.0)
    # per-channel factor broadcasts
    s2 = t.scaled(jnp.array([[1.0], [2.0]]))
    assert_allclose(np.asarray(s2.value), [[1.0] * 4, [2.0] * 4])
    # kinds without a value scale to themselves
    u = ErrorTerm.unbounded("x").scaled(5.0)
    assert u.value is None and u.kind == "unbounded" and u.note == "x"
    assert ErrorTerm.not_applicable().scaled(2.0).kind == "not_applicable"
    with pytest.raises(Exception):
        t.scaled(-1.0)


def test_scaled_under_jit_and_grad():
    t = ErrorTerm(jnp.ones((2, 4)), "bound")

    @jax.jit
    def f(amplitude):
        return t.scaled(amplitude).value.sum()

    assert float(f(2.0)) == 16.0
    assert float(jax.grad(f)(2.0)) == 8.0


def test_error_term_to_dict_round_trips_through_json():
    t = ErrorTerm(
        jnp.array([[1.0, 0.5, 0.25, 0.0]]), "bound", note="n", manuscript_term="E"
    )
    d = t.to_dict()
    assert d == {
        "value": [[1.0, 0.5, 0.25, 0.0]],
        "kind": "bound",
        "note": "n",
        "manuscript_term": "E",
    }
    assert json.loads(json.dumps(d)) == d
    u = ErrorTerm.unbounded("why").to_dict()
    assert u == {
        "value": None,
        "kind": "unbounded",
        "note": "why",
        "manuscript_term": "",
    }
    assert json.loads(json.dumps(u)) == u


def test_error_term_is_immutable():
    t = ErrorTerm(jnp.ones(4), "bound")
    with pytest.raises((AttributeError, TypeError)):
        t.kind = "estimate"  # type: ignore[misc]


# ----------------------------------------------------------------------
# ErrorBudget
# ----------------------------------------------------------------------


def test_slots_order():
    assert ErrorBudget.SLOTS == (
        "basis_remainder",
        "statistical_input",
        "physical_kernel",
        "harmonic_truncation",
        "excluded_tail",
        "numerical",
        "screen_exponent",
        "depth_model",
        "amplitude",
    )


def test_all_unbounded():
    b = ErrorBudget.all_unbounded("nothing supplied")
    assert b.unbounded() == ErrorBudget.SLOTS
    assert b.assumption == ()
    for name, term in b.terms():
        assert term.kind == "unbounded" and term.note == "nothing supplied"
    assert len(b.terms()) == 9
    total = b.total()
    assert total.kind == "unbounded" and total.value is None
    assert b.envelope() is None
    assert b.total().manuscript_term == "eq: channel error budget"


def test_total_is_unbounded_if_any_term_is():
    b = budget_with(excluded_tail=ErrorTerm.unbounded("no tail model"))
    assert b.unbounded() == ("excluded_tail",)
    t = b.total()
    assert t.kind == "unbounded" and t.value is None
    assert "excluded_tail" in t.note
    assert b.envelope() is None
    # any single slot being unbounded suffices
    for slot in ErrorBudget.SLOTS:
        bb = budget_with(**{slot: ErrorTerm.unbounded()})
        assert bb.total().kind == "unbounded"
        assert bb.unbounded() == (slot,)


def test_total_unbounded_via_assumption_term():
    b = budget_with(
        assumption=(("independent_screen", ErrorTerm.unbounded("no measure")),)
    )
    assert b.unbounded() == ("assumption:independent_screen",)
    assert b.total().kind == "unbounded"
    names = [n for n, _ in b.terms()]
    assert names == list(ErrorBudget.SLOTS) + ["assumption:independent_screen"]


def test_total_sums_values_and_broadcasts():
    n_ch = 3
    a = ErrorTerm(jnp.full((n_ch, 4), 0.5), "bound")
    c = ErrorTerm(jnp.array([[1.0, 2.0, 3.0, 4.0]]), "bound")  # (1, 4) broadcast
    s = ErrorTerm(jnp.array(0.25), "bound")  # scalar broadcast
    b = budget_with(basis_remainder=a, numerical=c, amplitude=s)
    total = b.total()
    assert total.kind == "bound"
    expected = 0.5 + np.array([[1.0, 2.0, 3.0, 4.0]]) + 0.25
    assert total.value.shape == (n_ch, 4)
    assert_allclose(np.asarray(total.value), np.broadcast_to(expected, (n_ch, 4)))
    assert_allclose(np.asarray(b.envelope()), np.asarray(total.value))


def test_total_broadcast_mismatch_raises_value_error():
    b = budget_with(
        basis_remainder=ErrorTerm(jnp.ones((3, 4)), "bound"),
        numerical=ErrorTerm(jnp.ones((2, 4)), "bound"),
    )
    with pytest.raises(ValueError):
        b.total()


@pytest.mark.parametrize(
    "kinds,expected",
    [
        (("bound", "bound"), "bound"),
        (("bound", "measured"), "measured"),
        (("measured", "bound"), "measured"),
        (("bound", "estimate"), "estimate"),
        (("measured", "estimate"), "estimate"),
        (("estimate", "measured"), "estimate"),
        (("estimate", "bound"), "estimate"),
        (("measured", "measured"), "measured"),
    ],
)
def test_total_kind_is_the_weakest(kinds, expected):
    b = budget_with(
        basis_remainder=ErrorTerm(jnp.ones(4), kinds[0]),
        statistical_input=ErrorTerm(2 * jnp.ones(4), kinds[1]),
    )
    total = b.total()
    assert total.kind == expected
    assert_allclose(np.asarray(total.value), 3.0)
    assert expected in total.note


def test_not_applicable_terms_are_skipped():
    b = budget_with(
        basis_remainder=ErrorTerm.not_applicable("direct average"),
        numerical=ErrorTerm(jnp.ones((2, 4)), "estimate"),
    )
    t = b.total()
    assert t.kind == "estimate"
    assert_allclose(np.asarray(t.value), 1.0)
    assert b.unbounded() == ()
    all_na = ErrorBudget(**{s: ErrorTerm.not_applicable() for s in ErrorBudget.SLOTS})
    t0 = all_na.total()
    assert t0.kind == "bound" and t0.value.shape == (1, 4)
    assert_allclose(np.asarray(t0.value), 0.0)


def test_assumption_terms_enter_the_sum():
    b = budget_with(
        assumption=(
            ("independent_screen", ErrorTerm(jnp.ones((2, 4)), "measured", "m")),
            ("azimuth_separable", ErrorTerm(2 * jnp.ones((2, 4)), "bound")),
        )
    )
    t = b.total()
    assert t.kind == "measured"
    assert_allclose(np.asarray(t.value), 3.0)
    names = [n for n, _ in b.terms()]
    assert names[-2:] == [
        "assumption:independent_screen",
        "assumption:azimuth_separable",
    ]


def test_budget_validation():
    with pytest.raises(ValueError):
        budget_with(numerical=None)
    with pytest.raises(ValueError):
        budget_with(numerical=1.0)
    with pytest.raises(ValueError):
        budget_with(assumption=(("a", 1.0),))
    with pytest.raises(ValueError):
        budget_with(assumption=((1, ErrorTerm.unbounded()),))
    with pytest.raises(ValueError):
        budget_with(
            assumption=(("a", ErrorTerm.unbounded()), ("a", ErrorTerm.unbounded()))
        )
    with pytest.raises(ValueError):
        budget_with(assumption=[("a", ErrorTerm.unbounded())])


def test_budget_to_dict_round_trips_and_names_manuscript_terms():
    b = budget_with(
        harmonic_truncation=ErrorTerm(jnp.zeros((1, 4)), "bound", "m_max >= required"),
        excluded_tail=ErrorTerm.unbounded("no tail"),
        assumption=(("independent_screen", ErrorTerm(jnp.ones((1, 4)), "measured")),),
    )
    d = b.to_dict()
    assert set(d) == set(ErrorBudget.SLOTS) | {"assumption", "total"}
    assert d["harmonic_truncation"]["manuscript_term"] == "E_num"
    assert d["numerical"]["manuscript_term"] == "E_num"
    assert d["excluded_tail"]["manuscript_term"] == "E_tail"
    assert d["physical_kernel"]["manuscript_term"] == "E_phys"
    assert d["excluded_tail"]["value"] is None
    assert d["assumption"] == {
        "independent_screen": {
            "value": [[1.0, 1.0, 1.0, 1.0]],
            "kind": "measured",
            "note": "",
            "manuscript_term": MANUSCRIPT_TERMS["assumption"],
        }
    }
    assert d["total"]["value"] is None and d["total"]["kind"] == "unbounded"
    assert d["total"]["unbounded"] == ["excluded_tail"]
    text = json.dumps(d)
    assert json.loads(text) == d
    # explicit manuscript_term on the term wins over the slot default
    b2 = budget_with(
        numerical=ErrorTerm(jnp.zeros(4), "bound", manuscript_term="custom")
    )
    assert b2.to_dict()["numerical"]["manuscript_term"] == "custom"
    assert b2.to_dict()["total"]["unbounded"] == []
    assert b2.to_dict()["total"]["value"] == [[0.0] * 4]


def test_budget_total_under_jit_and_grad():
    def f(x):
        b = budget_with(
            basis_remainder=ErrorTerm(x * jnp.ones((2, 4)), "estimate"),
            statistical_input=ErrorTerm(jnp.ones((2, 4)), "bound"),
        )
        return b.total().value.sum()

    assert float(jax.jit(f)(2.0)) == 24.0
    assert float(jax.grad(f)(2.0)) == 8.0


def test_budget_crosses_filter_jit_boundary():
    b = budget_with(
        assumption=(("azimuth_separable", ErrorTerm(jnp.ones(4), "bound")),),
        excluded_tail=ErrorTerm.unbounded("tail"),
    )

    @eqx.filter_jit
    def f(budget):
        return budget.total()

    out = f(b)
    assert out.kind == "unbounded" and out.value is None

    @eqx.filter_jit
    def g(budget):
        return budget

    back = g(b)
    assert back.assumption[0][0] == "azimuth_separable"
    assert back.excluded_tail.kind == "unbounded"


# ----------------------------------------------------------------------
# AssumptionRecord
# ----------------------------------------------------------------------


def test_assumption_record_hashable_and_json():
    r = record()
    assert hash(r) == hash(record())
    assert r == record()
    d = r.to_dict()
    assert d["groups"] == [["depth"], ["gamma", "B", "mu", "eta", "phi"]]
    assert d["n_free"] == 115 and d["hyper"] == []
    assert json.loads(json.dumps(d)) == d
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.name = "x"  # type: ignore[misc]


def test_assumption_record_normalises_and_validates():
    r = record(groups=[["depth"], ["gamma"]], hyper=[1, 2.5])
    assert r.groups == (("depth",), ("gamma",)) and r.hyper == (1.0, 2.5)
    assert hash(r)
    with pytest.raises(ValueError):
        record(name=3)
    with pytest.raises(ValueError):
        record(n_free=-1)
    with pytest.raises(ValueError):
        record(groups=(("depth", 1),))
    with pytest.raises(ValueError):
        record(hyper=("a",))


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def test_provenance_hashable_and_immutable():
    p = provenance()
    assert hash(p) == hash(provenance())
    assert p == provenance()
    assert {p: 1}[provenance()] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.units = "x"  # type: ignore[misc]


def test_provenance_rejects_unhashable_fields():
    with pytest.raises(ValueError):
        provenance(kernel=(("name", ["harmonic"]),))
    with pytest.raises(ValueError):
        provenance(notes=["a"])
    with pytest.raises(ValueError):
        provenance(assumptions=("independent_screen",))
    with pytest.raises(ValueError):
        provenance(certified_orders=(0, 1.5))
    with pytest.raises(ValueError):
        provenance(kernel=(("name", jnp.ones(2)),))


def test_provenance_to_dict_converts_pairs_and_round_trips():
    d = provenance().to_dict()
    assert d["kernel"] == {"name": "harmonic", "m_max": 40}
    assert d["truncation"] == {"L_mu": 2, "L_eta": 2, "N": 2, "depth_degree": None}
    assert d["support"] == {"gamma": [10.0, 30.0], "truncated": None}
    assert d["numerics"] == {"n_nodes": 256, "m_range": [1, 40]}
    assert d["certified_orders"] == [0, 1, 2]
    assert d["finite_checks"] == ["benchmark"]
    assert d["notes"] == ["incident polarisation not modelled"]
    assert d["assumptions"] == [record().to_dict()]
    assert d["package_version"] == "0.1.0" and d["units"] == "erg/s/sr per electron"
    assert json.loads(json.dumps(d)) == d


def test_provenance_with_notes_and_assumptions_are_functional():
    p = provenance()
    q = p.with_notes("observing response assumed exact", "second")
    assert p.notes == ("incident polarisation not modelled",)
    assert q.notes == p.notes + ("observing response assumed exact", "second")
    r = q.with_assumptions(record("gaussian_screen", hyper=(0.0, 1.0)))
    assert len(p.assumptions) == 1 and len(r.assumptions) == 2
    assert r.assumptions[1].name == "gaussian_screen"
    assert hash(r) != hash(p)
    assert p.with_notes() == p
    with pytest.raises(ValueError):
        p.with_notes(3)
    with pytest.raises(ValueError):
        p.with_assumptions("not a record")


def test_provenance_usable_as_static_field_in_jit():
    class Holder(eqx.Module):
        provenance: Provenance = eqx.field(static=True)
        x: jax.Array

    calls = []

    @jax.jit
    def f(h):
        calls.append(h.provenance.package_version)
        return h.x * len(h.provenance.notes)

    h = Holder(provenance(), jnp.ones(2))
    assert_allclose(np.asarray(f(h)), 1.0)
    f(Holder(provenance(), 2 * jnp.ones(2)))
    assert calls == ["0.1.0"]
    f(Holder(provenance().with_notes("more"), jnp.ones(2)))
    assert calls == ["0.1.0", "0.1.0"]
