"""Tests of ``fit_bfgs`` and ``fit_nodal`` (``synchro.model.fit.nonlinear``).

Oracles: NumPy least squares for the affine truth, ``scipy.optimize.nnls``
for the simplex core (``u = A w >= 0``), ``scipy.special.eval_legendre``
node features against the chunked operator and against
``JointMoments.from_samples``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import nnls

from _nonlinear_oracles import (
    node_features,
    polynomial_basis,
    random_nodes,
    reference_of,
    samples_of,
)
from synchro.model.assumptions import (
    Parameters,
    gaussian_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
)
from synchro.model.errors import ErrorTerm
from synchro.model.fit import _nodal
from synchro.model.fit.nonlinear import (
    MAX_BFGS_PARAMS,
    LogDensity,
    Transform,
    fit_bfgs,
    fit_nodal,
)
from synchro.model.fit.result import FitResult
from synchro.model.fit.observation import StokesData
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments


def stokes_data(basis, m, amplitude, *, noise=1e-2):
    C = np.asarray(basis.response_matrix())
    d = amplitude * C @ np.asarray(m)
    return StokesData(d.reshape(-1, 4), np.full(d.size, noise))


def random_z(transform, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(transform.n_params))


# -- fit_bfgs --------------------------------------------------------------------


@pytest.mark.parametrize("name", ["isotropic_pitch", "gaussian_fit"])
def test_fit_bfgs_recovers_truth_at_zero_noise(name):
    """Affine map: exact recovery. Gaussian screen with fitted ``(mean, sigma)``
    at ``N = 2`` (``sigma`` enters through ``b = 2``): the prior ``-log sigma``
    cancels the log-Jacobian of the log slot, so the mode in ``z`` is the
    maximum-likelihood point and the truth is recovered as well."""
    if name == "gaussian_fit":
        index = MomentIndex.build(Truncation(1, 1, 2))
        basis, _ = polynomial_basis(index, n_ch=16, seed=22)
        pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)

        def prior(theta):
            return -jnp.log(theta.hyper[0][1])

    else:
        index = MomentIndex.build(Truncation(1, 1, 1))
        basis, _ = polynomial_basis(index, n_ch=10, seed=21)
        pm = isotropic_pitch(index)
        prior = None
    tr = Transform(pm)
    rng = np.random.default_rng(1)
    z_true = jnp.asarray(0.5 * rng.standard_normal(tr.n_params))
    truth = tr.inverse(z_true)
    data = stokes_data(
        basis, pm(truth, basis.reference).to_vector(), float(jnp.exp(z_true[0]))
    )
    density = LogDensity(basis, data, pm, prior)
    z0 = z_true + 0.2 * jnp.asarray(rng.standard_normal(tr.n_params))
    out = fit_bfgs(density, z0, maxiter=3000, tol=1e-9)
    assert isinstance(out, FitResult) and out.method == "bfgs"
    assert out.converged and out.n_iter > 0 and out.weights is None
    assert out.labels == density.transform.labels
    assert_allclose(np.asarray(out.z), np.asarray(z_true), atol=1e-8)
    assert_allclose(float(density.chi2(out.z)), 0.0, atol=1e-12)
    assert_allclose(out.fun, -float(density(out.z)), rtol=1e-9, atol=1e-12)
    assert_allclose(pm.flatten(out.theta), pm.flatten(truth), atol=1e-8)
    assert_allclose(out.theta.log_amplitude, z_true[0], atol=1e-8)


def test_fit_bfgs_matches_linear_least_squares_with_noise():
    """With noise the BFGS optimum of an affine map equals ``lstsq`` on ``(A, A theta)``."""
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, C = polynomial_basis(index, n_ch=10, seed=5)
    pm = isotropic_pitch(index)
    tr = Transform(pm)
    rng = np.random.default_rng(2)
    z_true = jnp.asarray(0.3 * rng.standard_normal(tr.n_params))
    truth = tr.inverse(z_true)
    d = (
        float(jnp.exp(z_true[0]))
        * C
        @ np.asarray(pm(truth, basis.reference).to_vector())
    )
    sigma = 0.05
    d = d + sigma * rng.standard_normal(d.size)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, sigma**2))
    density = LogDensity(basis, data, pm)
    out = fit_bfgs(density, z_true, maxiter=2000, tol=1e-8)
    P, c = pm.affine_pieces()
    G = C @ np.column_stack([np.asarray(c), np.asarray(P)])
    u, *_ = np.linalg.lstsq(G, d, rcond=None)
    A, theta = u[0], u[1:] / u[0]
    assert out.converged
    assert_allclose(float(jnp.exp(out.z[0])), A, rtol=1e-7)
    assert_allclose(np.asarray(out.z[1:]), theta, atol=1e-7)


def test_fit_bfgs_reports_failure_to_converge():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, _ = polynomial_basis(index, n_ch=10, seed=21)
    pm = no_assumption(index)
    tr = Transform(pm)
    data = stokes_data(
        basis, pm(tr.inverse(random_z(tr, 0)), basis.reference).to_vector(), 1.0
    )
    density = LogDensity(basis, data, pm)
    out = fit_bfgs(density, random_z(tr, 3), maxiter=1)
    assert out.n_iter <= 1 and not out.converged
    assert np.all(np.isfinite(np.asarray(out.z)))


# -- node operator ----------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 5000])
def test_node_operator_matches_numpy_features(n):
    index = MomentIndex.build(Truncation(1, 1, 2))
    pop = random_nodes(3, n)
    nodes = samples_of(pop)
    op = _nodal.NodeOperator(nodes, index, reference_of())
    X = node_features(pop, index)
    rng = np.random.default_rng(4)
    w = rng.uniform(0.1, 2.0, n)
    y = rng.standard_normal(index.n_real)
    assert_allclose(np.asarray(op.moments(w)), X.T @ w, rtol=1e-12, atol=1e-12)
    assert_allclose(np.asarray(op.adjoint(y)), X @ y, rtol=1e-12, atol=1e-12)
    # consistency with the package's own sample moments (normalised weights)
    expected = JointMoments.from_samples(samples_of(pop, w), index, reference_of())
    assert_allclose(
        np.asarray(op.moments(w)) / w.sum(),
        np.asarray(expected.to_vector()),
        atol=1e-13,
    )
    grad = jax.jit(jax.grad(lambda v: op.moments(v) @ y))(jnp.asarray(w))
    assert_allclose(np.asarray(grad), X @ y, rtol=1e-12, atol=1e-12)


# -- simplex core vs NNLS ----------------------------------------------------------------


@pytest.mark.parametrize("interior", [True, False])
def test_simplex_fit_matches_nnls(interior):
    rng = np.random.default_rng(11 if interior else 12)
    n_data, n = 12, 4
    G = rng.standard_normal((n_data, n))
    if interior:
        w_true = rng.uniform(0.2, 1.0, n)
        w_true /= w_true.sum()
        target = 2.0 * G @ w_true
    else:
        target = rng.standard_normal(n_data)  # optimum generally on the boundary
    u_ref, _ = nnls(G, target)
    w, amp, fun, n_iter, converged, kkt = _nodal.simplex_fit(
        lambda w: G @ w,
        lambda r: G.T @ r,
        target,
        np.full(n, 1.0 / n),
        iters=20000,
        step=0.1,
        tol=1e-15,
    )
    w = np.asarray(w)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    residual_ref = 0.5 * np.sum((target - G @ u_ref) ** 2)
    assert float(fun) <= residual_ref + 1e-8
    if interior:
        assert bool(converged)
        assert_allclose(w, w_true, atol=1e-7)
        assert_allclose(float(amp), 2.0, rtol=1e-7)
    else:
        assert_allclose(float(amp) * w, u_ref, atol=2e-4)
    assert float(kkt) < 1e-4 and int(n_iter) > 0


def test_simplex_fit_is_monotone_and_handles_zero_target():
    rng = np.random.default_rng(3)
    G = rng.standard_normal((6, 3))
    w, amp, fun, n_iter, converged, _ = _nodal.simplex_fit(
        lambda w: G @ w,
        lambda r: G.T @ r,
        np.zeros(6),
        np.ones(3) / 3,
        iters=50,
        step=0.5,
        tol=1e-12,
    )
    assert float(amp) == 0.0 and float(fun) == 0.0
    assert_allclose(np.asarray(w), np.ones(3) / 3)


# -- fit_nodal ---------------------------------------------------------------------


def nodal_setup(n_nodes=5, n_ch=6, seed=7):
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, C = polynomial_basis(index, n_ch=n_ch, seed=seed)
    pop = random_nodes(seed, n_nodes)
    rng = np.random.default_rng(seed + 1)
    w_true = rng.uniform(0.2, 1.0, n_nodes)
    w_true /= w_true.sum()
    X = node_features(pop, index)
    A_true = 2.5
    d = A_true * C @ (X.T @ w_true)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    return index, basis, pop, w_true, A_true, data


def test_fit_nodal_recovers_weights_on_true_nodes():
    index, basis, pop, w_true, A_true, data = nodal_setup()
    nodes = samples_of(pop)
    out = fit_nodal(basis, data, nodes, iters=5000, step=0.1)
    assert isinstance(out, FitResult) and out.method == "nodal" and out.converged
    w = np.asarray(out.weights)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-12
    assert_allclose(w, w_true, atol=1e-7)
    assert_allclose(float(jnp.exp(out.theta.log_amplitude)), A_true, rtol=1e-7)
    assert out.fun < 1e-10 and out.discretisation is None
    # theta reproduces the weights through the nodal map and the transform
    pm = nodal(index, nodes)
    moments = pm(out.theta, basis.reference).to_vector()
    expected = JointMoments.from_samples(samples_of(pop, w), index, basis.reference)
    assert_allclose(np.asarray(moments), np.asarray(expected.to_vector()), atol=1e-12)
    assert_allclose(np.asarray(out.z[1:]), np.log(w), atol=1e-12)
    assert_allclose(jax.nn.softmax(out.theta.logits), w, atol=1e-15)
    # the log density of the nodal map at the fitted point has zero chi-square
    density = LogDensity(basis, data, pm)
    assert float(density.chi2(out.z)) < 1e-9


def test_fit_nodal_starts_from_node_weights_and_passes_discretisation():
    index, basis, pop, w_true, A_true, data = nodal_setup(seed=9)
    term = ErrorTerm(jnp.zeros((6, 4)), "bound", "extra eq: nodal bound test")
    nodes = samples_of(pop, w_true)  # start at the truth
    out = fit_nodal(basis, data, nodes, iters=50, discretisation=term)
    assert out.discretisation is term
    assert_allclose(np.asarray(out.weights), w_true, atol=1e-9)
    assert out.n_iter <= 50
    with pytest.raises(ValueError):
        fit_nodal(basis, data, object())
    with pytest.raises(ValueError):
        fit_nodal(basis, data, nodes, discretisation=object())
    with pytest.raises(ValueError):
        fit_nodal(basis, data, nodes, iters=0)
    with pytest.raises(ValueError):
        fit_nodal(basis, data, nodes, step=0.0)


def test_fit_nodal_many_nodes_is_finite_and_feasible():
    """More nodes than data: the fit is not unique, but stays on the simplex."""
    index, basis, pop, w_true, A_true, data = nodal_setup(n_nodes=4100, n_ch=3, seed=13)
    out = fit_nodal(basis, data, samples_of(pop), iters=200, step=0.1)
    w = np.asarray(out.weights)
    assert np.all(np.isfinite(w)) and np.all(w >= 0) and abs(w.sum() - 1) < 1e-10
    assert np.isfinite(out.fun) and out.n_iter <= 200
    assert out.theta.logits.shape == (4100,)
    assert Parameters(logits=out.theta.logits).logits is out.theta.logits


# -- fit_bfgs: cap boundary --------------------------------------------------------------


def test_fit_bfgs_cap_both_sides():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, _ = polynomial_basis(index, n_ch=8, seed=11)
    pm = isotropic_pitch(index)
    data = stokes_data(basis, jnp.zeros(index.n_real).at[0].set(1.0), 1.0)
    density = LogDensity(basis, data, pm)
    with pytest.raises(ValueError, match="refuses"):
        fit_bfgs(density, jnp.zeros(MAX_BFGS_PARAMS + 1))
    for n in (MAX_BFGS_PARAMS, MAX_BFGS_PARAMS - 1):
        with pytest.raises(ValueError, match="shape"):
            fit_bfgs(density, jnp.zeros(n))
    with pytest.raises(ValueError):
        fit_bfgs(density, jnp.zeros((2, 7)))
    with pytest.raises(ValueError):
        fit_bfgs(density, jnp.zeros(density.n_params), maxiter=0)
    with pytest.raises(ValueError):
        fit_bfgs(density, jnp.zeros(density.n_params), restarts=-1)
    with pytest.raises(ValueError):
        fit_bfgs(density, jnp.zeros(density.n_params), line_search_maxiter=0)
    with pytest.raises(ValueError):
        fit_bfgs(object(), jnp.zeros(3))
