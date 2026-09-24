"""Regression R20: the ``numerical`` term of ``build_basis``.

Before the fix the term was ``sum_a |C_tensor,2x - C_product|_{ja}`` (a unit-
moment contraction of the other route's disagreement): it changed by eleven
orders of magnitude with the arbitrary ``Reference.scales`` and measured the
tensor route's error, not the product route's. Now the term is the per-column
envelope ``|C_route,2x - C|`` of shape ``(4 n_ch, n_real)`` of the primary
route refined against itself, which ``predict`` contracts with ``|m|``
(invariant under the scales, and ``>= |delta C m|`` by the triangle
inequality); the product-vs-tensor comparison is a named finite check.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model.basis import basis_convergence, build_basis
from synchro.model.channels import Channels
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support
from synchro.model.predict import predict

from _basis_fixtures import small_harmonic
from _harmonic_oracles import B0, GAMMA0, S_DEPTH

TRUNCATION = Truncation(1, 1, 1)
SCALES = {
    "natural": (GAMMA0, B0, S_DEPTH),
    "unit": (1.0, 1.0, 1.0),
    "small": (0.1, 0.01, 0.01 * S_DEPTH),
}


def population(seed=3, n=40):
    """Correlated atoms inside the small support (gamma 19.5..20.5, B 0.98..1.02)."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(-1, 1, (2, n))
    return PopulationSamples(
        GAMMA0 * (1 + 0.02 * u),
        B0 * (1 + 0.015 * (0.6 * u + 0.4 * v)),
        np.clip(0.3 * u + 0.2 * v, -0.9, 0.9),
        np.clip(0.5 * v - 0.1 * u, -0.9, 0.9),
        0.2 + 0.4 * u + 0.2 * v,
        S_DEPTH * (3.0 + 1.5 * u + 0.5 * v),
        weights=1.0 + 0.5 * rng.uniform(0, 1, n),
    )


def _prediction(scales, kernel=None, convergence="angular"):
    base_kernel, channels, _, support = small_harmonic()
    kernel = base_kernel if kernel is None else kernel
    reference = Reference(GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=scales)
    basis = build_basis(
        kernel,
        channels,
        TRUNCATION,
        reference,
        support=support,
        convergence=convergence,
    )
    moments = JointMoments.from_samples(population(), basis.index, reference)
    return basis, moments, predict(basis, moments, amplitude=1.0)


@pytest.fixture(scope="module")
def predictions():
    return {name: _prediction(scales) for name, scales in SCALES.items()}


def test_r20_numerical_is_a_per_column_envelope(predictions):
    basis, moments, pred = predictions["natural"]
    term = basis.kernel_terms.numerical
    assert term.kind == "estimate"
    assert term.value.shape == (4 * basis.n_ch, basis.index.n_real)
    assert "product route at 2x angular nodes" in term.note
    assert pred.budget.numerical.value.shape == (basis.n_ch, 4)
    assert "contracted with |m|" in pred.budget.numerical.note


def test_r20_contracted_numerical_term_is_invariant_under_the_scales(predictions):
    values = {
        name: np.asarray(pred.budget.numerical.value)
        for name, (_, _, pred) in predictions.items()
    }
    stokes = np.asarray(predictions["natural"][2].stokes)
    assert_allclose(stokes, np.asarray(predictions["small"][2].stokes), rtol=1e-8)
    for name in ("unit", "small"):
        assert_allclose(
            values[name], values["natural"], rtol=1e-6, atol=1e-12 * stokes[0, 0]
        )


def test_r20_numerical_envelopes_the_primary_route_error(predictions):
    """``sum_a |dC_ja||m_a|`` bounds ``|dC m|`` and tracks the 4x-node error."""
    basis, moments, pred = predictions["natural"]
    kernel, channels, reference, support = small_harmonic()
    fine = build_basis(
        HarmonicKernel(10, n_outer=64, n_inner=64),
        channels,
        TRUNCATION,
        basis.reference,
        support=support,
        convergence=False,
    )
    m = np.asarray(moments.to_vector())
    true = np.abs(
        (np.asarray(fine.response_matrix()) - np.asarray(basis.response_matrix())) @ m
    ).reshape(basis.n_ch, 4)
    estimate = np.asarray(pred.budget.numerical.value)
    stokes_I = abs(float(pred.stokes[0, 0]))
    assert np.all(estimate >= 0.5 * true), (estimate, true)
    # Not the old 2.3 x I: a 16-node product route is far better than that.
    assert float(np.max(estimate)) < 1e-2 * stokes_I, (estimate, stokes_I)


def test_r20_route_cross_check_is_a_named_finite_check():
    # NEW-9: the route check is opt-in (cross_route=True or convergence="full").
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel, channels, TRUNCATION, reference, support=support, cross_route=True
    )
    quadrature = dict(basis.provenance.quadrature)
    assert quadrature["route"] == "product" and quadrature["check_route"] == "tensor"
    assert quadrature["numerical_route"] == "product route at 2x angular nodes"
    checks = [c for c in basis.provenance.finite_checks if c.startswith("route check")]
    assert len(checks) == 1 and "tensor route at 2x angular nodes" in checks[0]


def test_r20_cross_route_envelope_is_available_on_request(predictions):
    basis = predictions["natural"][0]
    kernel = small_harmonic()[0]
    cross = basis_convergence(basis, kernel, factor=2, cross_route=True)
    assert cross.kind == "estimate"
    assert cross.value.shape == (4 * basis.n_ch, basis.index.n_real)
    assert "tensor route at 2x angular nodes" in cross.note


def test_r20_continuum_tail_rides_on_the_mass_column():
    kernel = ContinuumKernel()
    channels = Channels.bump([1e9], [2e8], n_nu=24)
    reference = Reference(1e3, 1e-5, scales=(1e3, 1e-5, 1.0))
    support = Support(gamma=(5e2, 5e3), B=(5e-6, 2e-5), depth=(0.0, 1.0))
    basis = build_basis(
        kernel, channels, Truncation(0, 1, 1), reference, support=support
    )
    term = basis.kernel_terms.numerical
    assert term.value.shape == (4, basis.index.n_real)
    tail = np.asarray(kernel.tail_bound(channels, 1e3, 1e-5, 0.0).value)
    mass = basis.index.position(0, 0, 0, 0, 0, 0)
    value = np.asarray(term.value).reshape(1, 4, -1)
    assert np.all(value[:, :, mass] >= tail - 1e-300)
    assert "mass column" in term.note


# -- NEW-9: the product-vs-tensor route check is opt-in -------------------------------


def _counted_build(monkeypatch, **kwargs):
    """``build_basis`` on the small harmonic setup, counting ``_build_core`` calls."""
    import synchro.model.basis as basis_module

    calls = []
    original = basis_module._build_core

    def counting(*args, **kw):
        calls.append(kw.get("quadrature"))
        return original(*args, **kw)

    monkeypatch.setattr(basis_module, "_build_core", counting)
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel, channels, TRUNCATION, reference, support=support, **kwargs
    )
    return basis, calls


def _route_checks(basis):
    return [c for c in basis.provenance.finite_checks if c.startswith("route check")]


def test_new9_default_build_does_exactly_one_extra_build(monkeypatch):
    basis, calls = _counted_build(monkeypatch)
    assert calls == [None, "product"]  # the basis, then the same route at 2x nodes
    quadrature = dict(basis.provenance.quadrature)
    assert quadrature["check_route"] is None
    assert quadrature["numerical_route"] == "product route at 2x angular nodes"
    assert _route_checks(basis) == []
    assert basis.kernel_terms.numerical.kind == "estimate"


@pytest.mark.parametrize(
    "kwargs",
    [{"cross_route": True}, {"convergence": "full"}],
    ids=["cross_route", "full"],
)
def test_new9_route_check_runs_on_request(monkeypatch, kwargs):
    basis, calls = _counted_build(monkeypatch, **kwargs)
    assert calls == [None, "product", "tensor"]
    assert dict(basis.provenance.quadrature)["check_route"] == "tensor"
    checks = _route_checks(basis)
    assert len(checks) == 1 and "not part of the budget" in checks[0]


def test_new9_full_without_route_check_and_numerical_unchanged(monkeypatch):
    basis, calls = _counted_build(monkeypatch, convergence="full", cross_route=False)
    assert calls == [None, "product"]
    assert _route_checks(basis) == []
    default, _ = _counted_build(monkeypatch)
    with_check, _ = _counted_build(monkeypatch, cross_route=True)
    # The route check never changes the budget term.
    assert_allclose(
        np.asarray(with_check.kernel_terms.numerical.value),
        np.asarray(default.kernel_terms.numerical.value),
        rtol=0,
        atol=0,
    )


def test_new9_cross_route_validation():
    kernel, channels, reference, support = small_harmonic()
    args = (kernel, channels, TRUNCATION, reference)
    with pytest.raises(ValueError, match="cross_route"):
        build_basis(*args, support=support, convergence=False, cross_route=True)
    with pytest.raises(ValueError, match="cross_route"):
        build_basis(*args, support=support, cross_route="yes")
