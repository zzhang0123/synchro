"""Strict-JSON ``to_dict`` output for fitted-hyper maps (review item NEW2-1).

A map with ``free_hyper`` records NaN placeholders in its
``AssumptionRecord.hyper``. Every public ``to_dict`` must still be strict
JSON (``json.dumps(..., allow_nan=False)``) and survive a JSON round trip,
and a fit's prediction must carry the note with the fitted hyper values.
"""

from __future__ import annotations

import json
import math

import jax.numpy as jnp
import numpy as np
import pytest

from _nonlinear_oracles import polynomial_basis
from syncmoments.model.assumptions import gaussian_screen
from syncmoments.model.errors import AssumptionRecord, Provenance
from syncmoments.model.fit.nonlinear import LogDensity, Transform, fit_bfgs
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.predict import predict


def strict_round_trip(d):
    text = json.dumps(d, allow_nan=False)
    assert json.loads(text) == d


@pytest.fixture(scope="module")
def fitted_hyper_case():
    index = MomentIndex.build(Truncation(1, 1, 2))
    basis, C = polynomial_basis(index, n_ch=16, seed=22)
    pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)
    transform = Transform(pm)
    rng = np.random.default_rng(1)
    z_true = jnp.asarray(0.5 * rng.standard_normal(transform.n_params))
    moments = pm(transform.inverse(z_true), basis.reference)
    return basis, C, pm, z_true, moments


def test_fitted_hyper_record_holds_nan_placeholders(fitted_hyper_case):
    moments = fitted_hyper_case[4]
    record = moments.assumptions[0]
    assert record.closure_kind.endswith("(fitted)")
    assert all(math.isnan(h) for h in record.hyper)


def test_assumption_record_to_dict_is_strict_json():
    record = AssumptionRecord(
        "r",
        "l",
        (("mu",),),
        "gaussian_depth (fitted)",
        (math.nan, 1.0, math.inf),
        3,
        "none",
    )
    d = record.to_dict()
    assert d["hyper"] == [None, 1.0, None]
    strict_round_trip(d)


def test_provenance_to_dict_is_strict_json():
    record = AssumptionRecord("r", "l", (), "k (fitted)", (math.nan,), 0, "none")
    provenance = Provenance(
        "v", (("x", math.nan),), (), (), (), (), (), (), (record,), "u", (), (), (), ()
    )
    d = provenance.to_dict()
    assert d["kernel"] == {"x": None}
    assert d["assumptions"][0]["hyper"] == [None]
    strict_round_trip(d)


def test_joint_moments_to_dict_is_strict_json(fitted_hyper_case):
    strict_round_trip(fitted_hyper_case[4].to_dict())


def test_predict_to_dict_is_strict_json(fitted_hyper_case):
    basis, _, _, _, moments = fitted_hyper_case
    strict_round_trip(predict(basis, moments, amplitude=1.0).to_dict())


def test_fit_result_prediction_is_strict_json_and_records_fitted_hyper(
    fitted_hyper_case,
):
    basis, C, pm, z_true, moments = fitted_hyper_case
    d = np.asarray(float(jnp.exp(z_true[0])) * C @ np.asarray(moments.to_vector()))
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-2))
    density = LogDensity(basis, data, pm, lambda th: -jnp.log(th.hyper[0][1]))
    result = fit_bfgs(density, z_true + 0.1, maxiter=2000)
    assert result.prediction is not None
    strict_round_trip(result.to_dict())
    strict_round_trip(result.prediction.to_dict())
    note = [n for n in result.provenance.notes if "fitted hyper" in n]
    assert note
    assert note[0] in result.prediction.provenance.notes


def test_prediction_to_dict_is_strict_json_for_non_finite_stokes(fitted_hyper_case):
    """NEW3-1: a non-finite Stokes value is written as null, not NaN."""
    import equinox as eqx

    basis, _, _, _, moments = fitted_hyper_case
    pred = predict(basis, moments, amplitude=1.0)
    broken = eqx.tree_at(lambda p: p.stokes, pred, pred.stokes.at[0, 0].set(jnp.nan))
    d = broken.to_dict()
    assert d["stokes"][0][0] is None
    assert d["stokes"][0][1] == float(pred.stokes[0, 1])
    strict_round_trip(d)
