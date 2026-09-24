"""Boundary sweeps on the statistics side: ``is_affine`` dispatch, ``noise -> 0``,
weights ``1e300``, a single sample and softmax logits ``+/-40`` (and ``+/-700``).

Both sides of every dispatch are evaluated directly: affine maps must obey
the superposition identity and reproduce ``affine_pieces``, non-affine maps
must violate it and refuse ``affine_pieces``; the whitened data, the
identifiability SVD and the log density are evaluated at variances down to
``1e-300``; extreme weights and logits are compared with their normalised
counterparts. (The BFGS parameter cap ``2000 +/- 1`` is covered by
``test_nonlinear_boundaries.py``.)
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model.assumptions import (
    Parameters,
    azimuth_separable,
    field_independent,
    fully_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
)
from synchro.model.bounds import RemainderInputs
from synchro.model.fit.diagnostics import identifiability
from synchro.model.fit.nonlinear import LogDensity
from synchro.model.fit.observation import StokesData
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference
from synchro.model.predict import direct_channel_average, predict

from _basis_fixtures import polynomial_setup
from _nonlinear_oracles import polynomial_basis, random_nodes, samples_of
from synchro.model.basis import build_basis

REF = Reference(5.2, 2.1, 0.1, scales=(1.3, 0.7, 0.9))

AFFINE = {
    "no_assumption": lambda i: no_assumption(i),
    "isotropic_pitch": lambda i: isotropic_pitch(i),
    "gaussian_screen_fixed": lambda i: gaussian_screen(i, 0.8, 0.6),
}
NON_AFFINE = {
    "independent_screen": lambda i: independent_screen(i),
    "field_independent": lambda i: field_independent(i),
    "azimuth_separable": lambda i: azimuth_separable(i),
    "fully_independent": lambda i: fully_independent(i),
    "gaussian_screen_fitted": lambda i: gaussian_screen(i, 0.8, 0.6, fit_hyper=True),
}


def _random_theta(pm, seed):
    """Random free parameters; fitted hyper-parameters are made positive (sigma >= 0)."""
    theta = pm.unflatten(
        jnp.asarray(np.random.default_rng(seed).standard_normal(pm.n_free()))
    )
    if theta.hyper:
        return Parameters(
            tables=theta.tables, hyper=tuple(jnp.abs(h) + 0.1 for h in theta.hyper)
        )
    return theta


def _vector(pm, theta):
    return np.asarray(pm(theta, REF).to_vector())


# -- is_affine dispatch --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(AFFINE))
def test_affine_maps_obey_superposition_and_affine_pieces(index_111, name):
    pm = AFFINE[name](index_111)
    assert pm.is_affine()
    a, b = _random_theta(pm, 1), _random_theta(pm, 2)
    both = pm.unflatten(pm.flatten(a) + pm.flatten(b))
    zero = pm.unflatten(jnp.zeros(pm.n_free()))
    m_a, m_b, m_ab, m_0 = (_vector(pm, t) for t in (a, b, both, zero))
    scale = np.max(np.abs(m_ab)) + 1.0
    assert_allclose(m_ab - m_0, (m_a - m_0) + (m_b - m_0), atol=1e-12 * scale, rtol=0)
    P, c = pm.affine_pieces(reference=REF)
    assert_allclose(
        np.asarray(P) @ np.asarray(pm.flatten(a)) + np.asarray(c),
        m_a,
        atol=1e-12 * scale,
        rtol=0,
    )
    assert np.all(np.isfinite(np.asarray(P))) and np.all(np.isfinite(np.asarray(c)))


@pytest.mark.parametrize("name", sorted(NON_AFFINE))
def test_non_affine_maps_violate_superposition_and_refuse_affine_pieces(
    index_111, name
):
    pm = NON_AFFINE[name](index_111)
    assert not pm.is_affine()
    a, b = _random_theta(pm, 1), _random_theta(pm, 2)
    both = pm.unflatten(pm.flatten(a) + pm.flatten(b))
    zero = pm.unflatten(jnp.zeros(pm.n_free()))
    m_a, m_b, m_ab, m_0 = (_vector(pm, t) for t in (a, b, both, zero))
    residual = np.max(np.abs((m_ab - m_0) - (m_a - m_0) - (m_b - m_0)))
    assert np.all(np.isfinite(m_ab)) and residual > 1e-3 * (np.max(np.abs(m_ab)) + 1.0)
    with pytest.raises(ValueError, match="affine"):
        pm.affine_pieces(reference=REF)


def test_nodal_map_is_not_affine_and_softmax_is_gauge_invariant(index_111):
    nodes = samples_of(random_nodes(3, 6))
    pm = nodal(index_111, nodes)
    assert not pm.is_affine() and pm.n_free() == 6
    logits = jnp.asarray([0.3, -1.0, 2.0, 0.0, 0.5, -0.2])
    m = _vector(pm, Parameters(logits=logits))
    shifted = _vector(pm, Parameters(logits=logits + 7.0))
    assert_allclose(shifted, m, rtol=1e-13, atol=1e-15)
    with pytest.raises(ValueError):
        pm.affine_pieces(reference=REF)


# -- softmax logits +/-40 and +/-700 ------------------------------------------------------------


@pytest.mark.parametrize("magnitude", [40.0, 700.0])
def test_extreme_logits_match_normalised_weights(index_111, magnitude):
    """Logits ``+/-magnitude`` give finite moments equal to ``from_samples``
    with the normalised ``softmax`` weights (``exp(+/-700)`` is representable
    only after the max shift, which both sides must apply)."""
    pop = random_nodes(5, 6)
    nodes = samples_of(pop)
    pm = nodal(index_111, nodes)
    logits = magnitude * jnp.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    m = _vector(pm, Parameters(logits=logits))
    assert np.all(np.isfinite(m))
    weights = np.exp(np.asarray(logits) - magnitude)
    weights /= weights.sum()
    reference = PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=jnp.asarray(weights),
    )
    expected = np.asarray(
        JointMoments.from_samples(reference, index_111, REF).to_vector()
    )
    assert_allclose(m, expected, rtol=1e-12, atol=1e-14)
    # the -magnitude nodes carry weight exp(-2 magnitude): below roundoff of the +nodes
    kept = PopulationSamples(
        *(pop[k][::2] for k in ("gamma", "B", "mu", "eta", "phi", "depth"))
    )
    assert_allclose(
        m,
        np.asarray(JointMoments.from_samples(kept, index_111, REF).to_vector()),
        rtol=1e-12,
        atol=1e-14,
    )


# -- weights 1e300 and a single sample ---------------------------------------------------------


def test_weights_1e300_equal_unit_weights_through_predict():
    """Relative weights of order ``1e300`` normalise to the same measure as
    unit-scale weights through ``from_samples``, ``predict`` and the direct
    average (no overflow in the normalisation or the weighted sums)."""
    kernel, _, _, reference, channels, support = polynomial_setup(
        seed=8, n_ch=3, L=1, N=1
    )
    basis = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1),
        reference,
        support=support,
        convergence=False,
    )
    pop = random_nodes(9, 7)
    pop["depth"] = np.full(
        7, float(reference.depth_ref)
    )  # constant depth: the response is exact
    w = np.random.default_rng(1).uniform(0.5, 2.0, 7)
    arrays = tuple(pop[k] for k in ("gamma", "B", "mu", "eta", "phi", "depth"))
    unit = PopulationSamples(*arrays, weights=w)
    huge = PopulationSamples(*arrays, weights=1e300 * w)
    assert_allclose(np.asarray(huge.normalised_weights()), w / w.sum(), rtol=1e-15)
    a = predict(
        basis, JointMoments.from_samples(unit, basis.index, reference), amplitude=1.0
    )
    b = predict(
        basis, JointMoments.from_samples(huge, basis.index, reference), amplitude=1.0
    )
    assert_allclose(np.asarray(b.stokes), np.asarray(a.stokes), rtol=1e-13)
    da = direct_channel_average(
        unit, kernel, channels, amplitude=1.0, reference=reference
    )
    db = direct_channel_average(
        huge, kernel, channels, amplitude=1.0, reference=reference
    )
    assert_allclose(np.asarray(db.stokes), np.asarray(da.stokes), rtol=1e-13)
    assert np.all(np.isfinite(np.asarray(db.stokes)))
    assert_allclose(
        np.asarray(db.stokes), np.asarray(b.stokes), rtol=1e-10
    )  # constant depth: exact


def test_single_sample_is_exact_and_probe_is_finite():
    """One sample: the polynomial finite response equals the direct value at
    that point (a degree-``N`` polynomial is its own Taylor expansion), the
    remainder probe over a one-point population is finite, and the moments
    are the monomials at that point."""
    kernel, _, _, ref, channels, support = polynomial_setup(seed=4, n_ch=2, L=1, N=1)
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis = build_basis(
        kernel, channels, Truncation(1, 1, 1), ref, support=support, convergence=False
    )
    one = PopulationSamples([6.0], [2.5], [0.4], [-0.3], [1.1], [float(ref.depth_ref)])
    moments = JointMoments.from_samples(one, index, ref)
    z = np.asarray(ref.z(6.0, 2.5, float(ref.depth_ref)))
    assert_allclose(float(moments.get(0, 0, 0, 1, 0, 0)), z[0], rtol=1e-14)
    assert_allclose(float(moments.get(0, 1, 0, 0, 0, 0)), 0.4, rtol=1e-14)
    assert_allclose(complex(moments.get(2, 0, 0, 0, 0, 0)), np.exp(2.2j), rtol=1e-14)
    inputs = RemainderInputs.from_samples(
        one, basis, kernel=kernel, angular_residual="probe"
    )
    for name in ("rho_ang", "H", "absolute_moments"):
        value = getattr(inputs, name)
        assert value is not None and np.all(np.isfinite(np.asarray(value)))
    pred = predict(basis, moments, amplitude=2.0, errors=inputs)
    assert np.all(np.isfinite(np.asarray(pred.stokes)))
    term = pred.budget.basis_remainder
    assert term.kind == "estimate" and np.all(np.isfinite(np.asarray(term.value)))


# -- noise -> 0 ------------------------------------------------------------------------------------

VARIANCES = (1.0, 1e-8, 1e-16, 1e-32, 1e-300)


@pytest.mark.parametrize("variance", VARIANCES)
def test_whitening_identifiability_and_density_as_noise_vanishes(
    index_111, variance, record_property
):
    """Whitened data ``d / sqrt(var)`` stays finite down to ``var = 1e-300``;
    the identifiability singular values (of the column-equilibrated design,
    review finding R09), ``null_dim`` and the resolution are invariant; the log density is finite
    while ``chi^2`` is representable and overflows to ``inf`` (never NaN) beyond."""
    basis, C = polynomial_basis(index_111, n_ch=6, seed=11)
    pm = isotropic_pitch(index_111)
    theta = _random_theta(pm, 5)
    m = np.asarray(pm(theta, basis.reference).to_vector())
    d = 1.5 * C @ m
    data = StokesData(d.reshape(-1, 4), np.full(d.size, variance))
    whitened, L_inv = data.whitened()
    assert np.all(np.isfinite(np.asarray(whitened))) and np.all(
        np.isfinite(np.asarray(L_inv))
    )
    assert_allclose(np.asarray(whitened), d / np.sqrt(variance), rtol=1e-13)
    report = identifiability(basis, data, pm, reference=basis.reference)
    unit = identifiability(
        basis,
        StokesData(d.reshape(-1, 4), np.ones(d.size)),
        pm,
        reference=basis.reference,
    )
    assert_allclose(
        np.asarray(report.singular_values),
        np.asarray(unit.singular_values),
        rtol=1e-10,
    )
    assert_allclose(
        np.asarray(report.column_scales) / np.sqrt(variance),
        np.asarray(unit.column_scales),
        rtol=1e-10,
    )
    assert report.null_dim == unit.null_dim
    assert_allclose(
        np.asarray(report.resolution), np.asarray(unit.resolution), atol=1e-10
    )
    density = LogDensity(basis, data, pm)
    z = density.transform.forward(
        Parameters(log_amplitude=jnp.log(1.5), tables=theta.tables, hyper=theta.hyper)
    )
    value = float(density(z))
    record_property("log_density_at_truth", value)
    assert np.isfinite(value)  # the truth has zero residual at every variance
    off = z + 1e-3
    chi2 = float(density.chi2(off))
    grad = np.asarray(jax.grad(density)(off))
    if variance >= 1e-32:
        assert np.isfinite(chi2) and np.all(np.isfinite(grad))
    else:
        assert not np.isnan(chi2) and not np.any(np.isnan(grad))


def test_zero_and_negative_variances_are_refused():
    """Zero, negative and subnormal variances are refused (``variances must be
    positive``); the smallest normal float whitens to ``1/sqrt(tiny) = 6.7e153``."""
    d = np.ones((3, 4))
    tiny = np.finfo(float).tiny  # 2.2e-308
    for bad in (0.0, -1.0, 0.5 * tiny, 1e-308):
        with pytest.raises(Exception, match="variances must be positive"):
            jax.block_until_ready(StokesData(d, np.full(12, bad)).whitened())
    for ok in (tiny, 1e-300):
        whitened, L_inv = StokesData(d, np.full(12, ok)).whitened()
        assert np.all(np.isfinite(np.asarray(whitened)))
        assert_allclose(np.asarray(L_inv)[0, 0], 1 / np.sqrt(ok), rtol=1e-12)
