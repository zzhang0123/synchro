"""Regression tests for review finding R22 on ``syncmoments.model.fit.nonlinear``.

``fit_bfgs`` from ``z0 = 0`` on ``gaussian_screen(fit_hyper=True)`` used
to abort with the ``gaussian_depth`` ``error_if`` when a line-search trial
point overflowed ``exp(z_sigma)``. The objective must reject such trial
points (``+inf``) so the fit returns, converged or not. Oracle: the
returned objective equals ``-LogDensity`` at the returned point and is
finite; trial points outside the evaluable range give ``+inf``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from _nonlinear_oracles import polynomial_basis
from syncmoments.model.assumptions import gaussian_screen
from syncmoments.model.fit import nonlinear
from syncmoments.model.fit.nonlinear import LogDensity, Transform, fit_bfgs
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.index import MomentIndex, Truncation


def noisy_density(seed, prior):
    index = MomentIndex.build(Truncation(1, 1, 2))
    basis, _ = polynomial_basis(index, n_ch=16, seed=22)
    C = np.asarray(basis.response_matrix())
    pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)
    transform = Transform(pm)
    assert transform.kinds[-1] == "log"
    rng = np.random.default_rng(seed)
    z_true = jnp.asarray(0.3 * rng.standard_normal(transform.n_params))
    theta = transform.inverse(z_true)
    m = np.asarray(pm(theta, basis.reference).to_vector())
    d = float(jnp.exp(theta.log_amplitude)) * C @ m
    sig = 1e-2
    noisy = d + np.sqrt(sig) * rng.standard_normal(d.size)
    data = StokesData(noisy.reshape(-1, 4), np.full(d.size, sig))
    return LogDensity(basis, data, pm, prior=prior), z_true


PRIORS = {"flat": None, "log_sigma": lambda t: -jnp.log(t.hyper[0][1])}


@pytest.mark.parametrize("prior", sorted(PRIORS))
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("precondition", [True, False])
def test_fit_bfgs_from_zero_returns_on_fitted_gaussian_screen(
    seed, prior, precondition
):
    density, _ = noisy_density(seed, PRIORS[prior])
    result = fit_bfgs(density, jnp.zeros(density.n_params), precondition=precondition)
    assert np.all(np.isfinite(np.asarray(result.z)))
    assert np.isfinite(result.fun)
    assert result.fun == pytest.approx(float(-density(result.z)), rel=1e-12)


def test_objective_rejects_overflowing_log_slot():
    density, z_true = noisy_density(1, None)
    objective = nonlinear._total_objective(density, jnp.ones(density.n_params))
    z_bad = z_true.at[-1].set(1e12)
    value, _ = jax.value_and_grad(objective)(z_bad)  # must not raise
    assert value == jnp.inf
    for z_bad in (z_true.at[0].set(jnp.nan), z_true.at[-1].set(-1e12)):
        assert objective(z_bad) == jnp.inf
    good = objective(z_true)
    assert float(good) == pytest.approx(float(-density(z_true)), rel=1e-14)
    assert np.all(np.isfinite(np.asarray(jax.grad(objective)(z_true))))
