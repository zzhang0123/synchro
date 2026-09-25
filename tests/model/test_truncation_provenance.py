"""``Truncation.max_orders`` in every output, and the ``syncmoments.model`` exports.

A capped truncation (``max_orders=(N_gamma, N_B, N_depth)``) retains a lower
set of ``(r, s, b)`` rows, so the flattened moment vector and the response
columns have a different meaning than for the uncapped cutoff with the same
``(L_mu, L_eta, N, depth_degree)``. The caps must therefore be recoverable
from the basis provenance (``_basis_checks.truncation_record``), from
``JointMoments.to_dict`` and from the test stub. Each check writes the record
through strict JSON (``allow_nan=False``), rebuilds the ``Truncation`` from
the parsed dict alone and compares it with the original (equality of the
static fields, which fixes the retained rows). Non-binding caps normalise to
``None`` in ``Truncation`` itself, so they must serialise like the default.
"""

import importlib
import json

import numpy as np
import pytest

import syncmoments.model as model
from syncmoments.model import assumptions, index as index_module
from syncmoments.model._basis_checks import truncation_record
from syncmoments.model.basis import build_basis
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples
from syncmoments.model.predict import predict

from _basis_fixtures import polynomial_setup
from _basis_stub import build_stub_basis

UNCAPPED = {
    "default": Truncation(2, 2, 2),
    "non_binding": Truncation(2, 2, 2, max_orders=(2, 5, None)),
    "depth_layout": Truncation(2, 2, 2, depth_degree=3),
}
CAPPED = {
    "no_B": Truncation(2, 2, 2, max_orders=(None, 0, None)),
    "all_capped": Truncation(2, 2, 2, max_orders=(1, 1, 1)),
    "depth_layout_capped": Truncation(
        2, 2, 2, depth_degree=3, max_orders=(1, None, None)
    ),
}
ALL = {**UNCAPPED, **CAPPED}
KEY_ORDER = [
    "L_mu",
    "L_eta",
    "N",
    "depth_degree",
    "max_orders",
    "parity",
    "components",
    "n0",
    "n2",
    "n_real",
]


def strict_round_trip(value):
    return json.loads(json.dumps(value, allow_nan=False))


def truncation_from(record):
    """``Truncation`` rebuilt from a parsed JSON record (lists back to tuples)."""
    caps = record["max_orders"]
    return Truncation(
        record["L_mu"],
        record["L_eta"],
        record["N"],
        record["depth_degree"],
        max_orders=None if caps is None else tuple(caps),
    )


def expected_caps(truncation):
    caps = truncation.max_orders
    return None if caps is None else list(caps)


def assert_identifies(record, truncation):
    """The parsed record fixes the truncation and its row counts."""
    assert record["max_orders"] == expected_caps(truncation)
    rebuilt = truncation_from(record)
    assert rebuilt == truncation and hash(rebuilt) == hash(truncation)
    index = MomentIndex(rebuilt)
    assert (record["n0"], record["n2"], record["n_real"]) == (
        index.n0,
        index.n2,
        index.n_real,
    )


def population(seed=4, S=9):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.0, 1.0, S)
    return PopulationSamples(
        20.0 + 3.0 * t,
        2.0 + 0.3 * t**2,
        np.clip(0.6 * t + 0.1 * rng.standard_normal(S), -0.95, 0.95),
        np.clip(-0.5 * t, -0.95, 0.95),
        0.4 + 1.1 * t,
        1.5 + 0.4 * t,
        weights=rng.uniform(0.5, 2.0, S),
    )


# -- the truncation normalisation the records rely on --------------------------


def test_non_binding_caps_serialise_like_the_default():
    assert UNCAPPED["non_binding"] == UNCAPPED["default"]
    assert UNCAPPED["non_binding"].max_orders is None
    assert all(t.is_capped() for t in CAPPED.values())
    assert not any(t.is_capped() for t in UNCAPPED.values())


# -- truncation_record (basis provenance) ------------------------------------


@pytest.mark.parametrize("name", sorted(ALL))
def test_truncation_record_carries_max_orders(name):
    truncation = ALL[name]
    record = truncation_record(MomentIndex(truncation))
    assert [key for key, _ in record] == KEY_ORDER
    assert dict(record)["max_orders"] == truncation.max_orders
    hash(record)  # a static field of Provenance must stay hashable


@pytest.mark.parametrize("name", sorted(ALL))
def test_basis_and_prediction_json_identify_the_truncation(name):
    truncation = ALL[name]
    kernel, _, _, reference, channels, support = polynomial_setup()
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, convergence=False
    )
    back = strict_round_trip(basis.provenance.to_dict())
    assert list(back["truncation"]) == KEY_ORDER
    assert_identifies(back["truncation"], truncation)

    moments = JointMoments.from_samples(population(), basis.index, reference)
    prediction = predict(basis, moments, amplitude=1.0)
    out = strict_round_trip(prediction.to_dict())
    assert_identifies(out["provenance"]["truncation"], truncation)
    assert_identifies(out["moments"]["index"], truncation)


def test_capped_and_uncapped_records_differ():
    records = {
        name: strict_round_trip(dict(truncation_record(MomentIndex(t))))
        for name, t in ALL.items()
    }
    assert records["non_binding"] == records["default"]
    distinct = [records[name] for name in ("default", *sorted(CAPPED))]
    for i, a in enumerate(distinct):
        for b in distinct[i + 1 :]:
            assert a != b
    # same (L_mu, L_eta, N, depth_degree): only max_orders tells them apart
    plain, capped = records["depth_layout"], records["depth_layout_capped"]
    assert {k: plain[k] for k in KEY_ORDER[:4]} == {k: capped[k] for k in KEY_ORDER[:4]}
    assert plain["max_orders"] is None and capped["max_orders"] == [1, None, None]


# -- JointMoments.to_dict -------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL))
def test_joint_moments_to_dict_round_trip(name):
    truncation = ALL[name]
    _, _, _, reference, _, _ = polynomial_setup()
    index = MomentIndex(truncation)
    moments = JointMoments.from_samples(population(), index, reference)
    back = strict_round_trip(moments.to_dict())
    assert back == moments.to_dict()
    record = back["index"]
    assert list(record) == [k for k in KEY_ORDER if k != "components"]
    assert_identifies(record, truncation)
    assert len(back["vector"]) == len(back["labels"]) == record["n_real"]


# -- the test stub --------------------------------------------------------------


@pytest.mark.parametrize("name", ["default", "no_B", "depth_layout_capped"])
def test_stub_provenance_records_max_orders(name):
    truncation = ALL[name]
    kernel, _, _, reference, channels, support = polynomial_setup()
    basis = build_stub_basis(kernel, channels, truncation, reference, support)
    back = strict_round_trip(basis.provenance.to_dict())
    assert back["truncation"]["max_orders"] == expected_caps(truncation)
    assert truncation_from(back["truncation"]) == truncation


# -- exports -----------------------------------------------------------------


NEW_EXPORTS = {
    "pitch_symmetric": assumptions.pitch_symmetric,
    "field_reversal_symmetric": assumptions.field_reversal_symmetric,
    "lower_set_margin": index_module.lower_set_margin,
}


@pytest.mark.parametrize("package", ["syncmoments.model", "syncmoments.model.fit"])
def test_every_exported_name_imports(package):
    module = importlib.import_module(package)
    names = module.__all__
    assert len(names) == len(set(names))
    assert not [n for n in names if n.startswith("_")]
    namespace = {}
    exec(f"from {package} import *", namespace)
    for name in names:
        assert getattr(module, name) is namespace[name]


@pytest.mark.parametrize("name", sorted(NEW_EXPORTS))
def test_new_public_names_are_exported(name):
    assert name in model.__all__
    assert getattr(model, name) is NEW_EXPORTS[name]


def test_exports_keep_their_module_group():
    names = model.__all__
    index_group = ["Entry", "MomentIndex", "Truncation", "lower_set_margin"]
    start = names.index("Entry")
    assert names[start : start + len(index_group)] == index_group
    assumption_group = [n for n in assumptions.__all__ if n != "VARS"]
    start = names.index(sorted(assumption_group)[0])
    assert names[start : start + len(assumption_group)] == sorted(assumption_group)
