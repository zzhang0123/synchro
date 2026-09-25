"""Boundary and extreme cases of ``syncmoments.model.fit.nonlinear``.

Both sides of every dispatch are evaluated directly: the preconditioner
floor (a parameter with zero curvature), the box saturation of the tanh
transform, the ``positive=`` override of the log slots, zero-weight nodes
under multiplicative updates, and logits at ``+/-40`` through the nodal
log density.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _nonlinear_oracles import polynomial_basis, random_nodes, samples_of
from syncmoments.model.assumptions import Parameters, gaussian_screen, nodal
from syncmoments.model.fit import nonlinear
from syncmoments.model.fit.nonlinear import LogDensity, Transform, fit_bfgs, fit_nodal
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.index import MomentIndex, Truncation


def zero_noise_data(basis, m, amplitude, noise=1e-2):
    d = amplitude * np.asarray(basis.response_matrix()) @ np.asarray(m)
    return StokesData(d.reshape(-1, 4), np.full(d.size, noise))


# -- preconditioner floor: an unidentified log slot -------------------------------


def test_unidentified_sigma_keeps_finite_scale_and_recovers_the_rest():
    """At ``N = 1`` no ``b = 2`` moment exists, so ``sigma`` has zero curvature;
    the Gauss-Newton diagonal is floored (finite scale), the ``-log sigma``
    prior cancels the Jacobian, and every identified parameter is recovered
    while ``sigma`` stays at its start."""
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, _ = polynomial_basis(index, n_ch=10, seed=31)
    pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)
    tr = Transform(pm)
    rng = np.random.default_rng(5)
    z_true = jnp.asarray(0.4 * rng.standard_normal(tr.n_params))
    truth = tr.inverse(z_true)
    data = zero_noise_data(
        basis, pm(truth, basis.reference).to_vector(), float(jnp.exp(z_true[0]))
    )
    density = LogDensity(basis, data, pm, lambda th: -jnp.log(th.hyper[0][1]))
    scale = nonlinear._gauss_newton_scale(density, z_true)
    assert np.all(np.isfinite(np.asarray(scale)))
    assert float(scale[-1]) == float(jnp.max(scale))  # the floored slot is the largest
    z0 = z_true + 0.1 * jnp.asarray(rng.standard_normal(tr.n_params))
    out = fit_bfgs(density, z0, maxiter=2000, tol=1e-9)
    assert out.converged
    assert_allclose(np.asarray(out.z[:-1]), np.asarray(z_true[:-1]), atol=1e-7)
    assert_allclose(float(out.z[-1]), float(z0[-1]), atol=1e-12)
    # without preconditioning the same problem is still solved (both sides)
    plain = fit_bfgs(density, z0, maxiter=2000, tol=1e-9, precondition=False)
    assert np.all(np.isfinite(np.asarray(plain.z)))


def test_flat_prior_with_unidentified_log_slot_does_not_converge():
    """The log-Jacobian alone pushes ``sigma`` upward without bound; the line
    search fails on the overflowing step and BFGS reports no convergence."""
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, _ = polynomial_basis(index, n_ch=10, seed=31)
    pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)
    tr = Transform(pm)
    z_true = jnp.asarray(0.4 * np.random.default_rng(5).standard_normal(tr.n_params))
    truth = tr.inverse(z_true)
    data = zero_noise_data(
        basis, pm(truth, basis.reference).to_vector(), float(jnp.exp(z_true[0]))
    )
    out = fit_bfgs(LogDensity(basis, data, pm), z_true, maxiter=200)
    assert not out.converged
    assert float(out.z[-1]) >= float(z_true[-1])
    assert np.all(np.isfinite(np.asarray(out.z)))
    # precision-floor rule, both sides: a large predicted decrease is not converged
    assert not nonlinear._at_precision_floor(
        _Result(status=3, jac=np.ones(2), hess_inv=np.eye(2), fun=0.0)
    )
    assert nonlinear._at_precision_floor(
        _Result(status=3, jac=1e-7 * np.ones(2), hess_inv=np.eye(2), fun=0.0)
    )
    assert not nonlinear._at_precision_floor(
        _Result(status=0, jac=np.zeros(2), hess_inv=np.eye(2), fun=0.0)
    )


class _Result:
    def __init__(self, **fields):
        self.__dict__.update(fields)


# -- transform extremes ---------------------------------------------------------------


def test_box_saturation_and_positive_override(index_111):
    pm = gaussian_screen(index_111, 0.8, 0.6, fit_hyper=True)
    label = "hyper:gaussian_depth:mean"
    tr = Transform(pm, bounds={label: (-2.0, 3.0)})
    slot = tr.labels.index(label)
    for value in (-50.0, 50.0, 700.0):
        z = jnp.zeros(tr.n_params).at[slot].set(value)
        theta = tr.inverse(z)
        assert -2.0 <= float(theta.hyper[0][0]) <= 3.0
        assert np.isfinite(float(tr.log_det(z)))
        assert np.all(np.isfinite(jax.grad(tr.log_det)(z)))
    # closed form far in the tail: log((hi-lo)/2) + 2 log 2 - 2 |z|
    z = jnp.zeros(tr.n_params).at[slot].set(30.0)
    assert_allclose(
        float(tr.log_det(z)), np.log(2.5) + 2 * np.log(2) - 60.0, rtol=1e-12
    )
    # positive= override: no log slots, or the mean on a log scale
    none = Transform(pm, positive=())
    assert (
        "log" not in none.kinds and float(none.log_det(jnp.zeros(none.n_params))) == 0
    )
    mean_log = Transform(pm, positive=(label,))
    assert mean_log.kinds[slot] == "log" and mean_log.kinds[-1] == "identity"
    theta = mean_log.inverse(jnp.zeros(mean_log.n_params))
    assert_allclose(theta.hyper[0], [1.0, 0.0])


def test_forward_rejects_boundary_values_and_accepts_interior(index_111):
    pm = gaussian_screen(index_111, 0.8, 0.6, fit_hyper=True)
    tr = Transform(pm, bounds={"hyper:gaussian_depth:mean": (-2.0, 3.0)})
    base = tr.inverse(jnp.zeros(tr.n_params))

    def with_hyper(mean, sigma):
        return Parameters(
            log_amplitude=base.log_amplitude,
            tables=base.tables,
            hyper=(jnp.array([mean, sigma]),),
        )

    for mean, sigma in ((-2.0, 1.0), (3.0, 1.0), (0.0, 0.0)):
        with pytest.raises(Exception):
            tr.forward(with_hyper(mean, sigma))
    for mean, sigma in ((-1.999, 1e-8), (2.999, 1e8)):
        z = tr.forward(with_hyper(mean, sigma))
        assert np.all(np.isfinite(np.asarray(z)))
        assert_allclose(tr.inverse(z).hyper[0], [mean, sigma], rtol=1e-9)


# -- nodal extremes ---------------------------------------------------------------------


def test_zero_weight_nodes_stay_zero_and_extreme_logits_are_finite():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, C = polynomial_basis(index, n_ch=6, seed=41)
    pop = random_nodes(41, 6)
    start = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    w_true = np.array([0.3, 0.0, 0.2, 0.4, 0.0, 0.1])
    from _nonlinear_oracles import node_features

    d = 2.0 * C @ (node_features(pop, index).T @ w_true)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    out = fit_nodal(basis, data, samples_of(pop, start), iters=3000)
    w = np.asarray(out.weights)
    assert w[1] == 0.0 and w[4] == 0.0 and np.all(w >= 0)
    assert_allclose(w, w_true, atol=1e-7)
    pm = nodal(index, samples_of(pop))
    density = LogDensity(basis, data, pm)
    z = jnp.asarray([0.0, 40.0, -40.0, 40.0, -40.0, 40.0, -40.0])
    assert np.isfinite(float(density(z)))
    assert np.all(np.isfinite(jax.jit(jax.grad(density))(z)))
    assert np.all(np.isfinite(np.asarray(out.z)))  # log of floored weights
