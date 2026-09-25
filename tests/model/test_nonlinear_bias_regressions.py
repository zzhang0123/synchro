"""Regression tests: the moment bias of ``fit_bfgs``/``fit_nodal`` (review NEW-4).

``fit_linear`` makes ``statistical_input`` unbounded when no data
discrepancy is declared (the moment bias is not constrained). The nonlinear
routes must agree: without a declared discrepancy their ``statistical_input``
is unbounded too; with one, ``fit_bfgs`` adds the linearised bias
``|dm/dz| |K| |delta|`` at the optimum, ``K = J^+ L^-1`` (``J`` the whitened
model Jacobian in ``z``), as an ``estimate``. Oracles: central-difference
Jacobians and ``numpy.linalg.pinv``; for a fixed amplitude and a table-only
map ``m`` is affine in ``z``, so the propagated bias bounds the actual
moment error of noiseless data shifted by the declared discrepancy.
"""

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

from _nonlinear_oracles import samples_of
from syncmoments.model.assumptions import isotropic_pitch
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.fit.linear import fit_linear
from syncmoments.model.fit.nonlinear import LogDensity, Transform, fit_bfgs, fit_nodal
from syncmoments.model.fit.observation import StokesData
from test_result import data_of, fd_jacobian, nodal_setup, setup

NOISE = 1e-2


def noisy_case():
    index, basis, C = setup((1, 1, 1), n_ch=10, seed=21)
    pm = isotropic_pitch(index)
    tr = Transform(pm)
    rng = np.random.default_rng(1)
    z_true = jnp.asarray(0.5 * rng.standard_normal(tr.n_params))
    m_true = pm(tr.inverse(z_true), basis.reference).to_vector()
    data, d = data_of(C, m_true, float(jnp.exp(z_true[0])), noise=NOISE, seed=3)
    return basis, C, pm, z_true, data, d


def test_bfgs_and_linear_agree_that_an_undeclared_bias_is_unbounded():
    basis, C, pm, z_true, data, _ = noisy_case()
    linear = fit_linear(basis, data, pm)
    bfgs = fit_bfgs(LogDensity(basis, data, pm), z_true, maxiter=3000)
    assert_allclose(
        np.asarray(bfgs.moments.to_vector()),
        np.asarray(linear.moments.to_vector()),
        atol=1e-8,
    )
    for result in (linear, bfgs):
        stat = result.prediction.budget.statistical_input
        assert stat.kind == "unbounded" and "bias" in stat.note
        assert result.bias_bound.kind == "unbounded"
    assert "no data.discrepancy declared" in bfgs.bias_bound.note


def test_bfgs_declared_discrepancy_adds_the_linearised_bias():
    basis, C, pm, z_true, data, d = noisy_case()
    delta = 0.01 * np.abs(d).max() * np.linspace(0.2, 1.0, d.size)
    term = ErrorTerm(jnp.asarray(delta.reshape(-1, 4)), "bound", "declared")
    declared = StokesData(data.stokes, data.noise, discrepancy=term)
    density = LogDensity(basis, declared, pm)
    result = fit_bfgs(density, z_true, maxiter=3000)
    z = np.asarray(result.z)

    def model(v):
        return -np.asarray(density.whitened_residual(jnp.asarray(v)))

    J = fd_jacobian(model, z)
    K = np.linalg.pinv(J) / np.sqrt(NOISE)  # L^-1 = I / sqrt(noise)
    expected = np.abs(K) @ delta
    bias = result.bias_bound
    assert bias.kind == "estimate" and "linearised at the optimum" in bias.note
    assert_allclose(np.asarray(bias.value), expected, rtol=1e-5)
    stat = result.prediction.budget.statistical_input
    assert stat.kind == "estimate" and "linearised at the optimum" in stat.note
    assert stat.value.shape == (basis.n_ch, 4)


def test_bfgs_bias_bounds_the_moment_error_when_m_is_affine_in_z():
    """Fixed amplitude, table-only map: ``m`` is affine in ``z``, so the
    linearisation is exact and ``A |C| |m_fit - m_true| <= statistical_input``
    for noiseless data shifted by the declared discrepancy."""
    index, basis, C = setup((1, 1, 1), n_ch=16, seed=3)
    pm = isotropic_pitch(index)
    rng = np.random.default_rng(4)
    theta = pm.unflatten(jnp.asarray(0.3 * rng.standard_normal(pm.n_free())))
    A = 1.7
    m = np.asarray(pm(theta, basis.reference).to_vector())
    d = A * C @ m
    delta = 0.02 * np.abs(d).max() * rng.uniform(0.2, 1.0, d.size)
    shifted = d + delta * np.sign(rng.standard_normal(d.size))
    term = ErrorTerm(jnp.asarray(delta.reshape(-1, 4)), "bound", "declared")
    data = StokesData(shifted.reshape(-1, 4), np.full(d.size, 1e-4), discrepancy=term)
    tr = Transform(pm, amplitude=False)
    density = LogDensity(basis, data, pm, transform=tr, amplitude=A)
    result = fit_bfgs(density, tr.forward(theta), maxiter=3000, tol=1e-12)
    stat = result.prediction.budget.statistical_input
    assert stat.kind == "estimate"
    error = np.abs(np.asarray(result.moments.to_vector()) - m)
    contracted = A * (np.abs(C) @ error).reshape(-1, 4)
    assert np.all(contracted <= np.asarray(stat.value) * (1 + 1e-6) + 1e-12)


def test_nodal_statistical_input_is_unbounded_with_or_without_discrepancy():
    """The softmax gauge makes the nodal Fisher singular: no covariance, so
    neither the noise part nor a linearised bias is formed."""
    index, basis, C, pop, X, w_true, A_true, data, d = nodal_setup()
    nodes = samples_of(pop)
    delta = 0.01 * np.abs(d).max() * np.ones(d.size)
    for discrepancy in (None, ErrorTerm(jnp.asarray(delta), "bound", "declared")):
        declared = StokesData(data.stokes, data.noise, discrepancy=discrepancy)
        result = fit_nodal(basis, declared, nodes, iters=200)
        assert result.prediction.budget.statistical_input.kind == "unbounded"
        assert result.bias_bound is None
