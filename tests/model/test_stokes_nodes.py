"""Regression tests for the additive ``n_nodes`` keyword of ``stokes_harmonic``."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import jv, jvp

import synchro  # noqa: F401  (enables x64)
from synchro.constants import C_CGS, E_ESU, M_E
from synchro.stokes import stokes_harmonic


def _automatic_count(n):
    """The count ``bessel._j_resolution`` picks for a concrete order ``n``."""
    return 2 ** math.ceil(math.log2(max(128, 4 * n + 64)))


def _oracle(n, gamma, alpha, theta, B):
    beta = np.sqrt(1 - 1 / gamma**2)
    b_par, b_perp = beta * np.cos(alpha), beta * np.sin(alpha)
    D = 1 - b_par * np.cos(theta)
    x = n * b_perp * np.sin(theta) / D
    a_par = (np.cos(theta) - b_par) * b_perp * (jv(n - 1, x) + jv(n + 1, x)) / (2 * D)
    a_perp = b_perp * jvp(n, x)
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2 * np.pi * C_CGS) * n**2 / D**3
    return np.array(
        [
            pref * (a_par**2 + a_perp**2),
            pref * (a_par**2 - a_perp**2),
            2 * pref * a_par * a_perp,
        ]
    )


@pytest.mark.parametrize("n", [1, 2, 7, 40, 300])
def test_default_and_automatic_count_are_identical(n):
    gamma, alpha, theta, B = 5.0, 0.7, 1.1, 3.0
    default = stokes_harmonic(n, gamma, alpha, theta, B=B)
    explicit = stokes_harmonic(n, gamma, alpha, theta, B=B, n_nodes=_automatic_count(n))
    for a, b in zip(default, explicit):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert_allclose(np.asarray(default), _oracle(n, gamma, alpha, theta, B), rtol=1e-11)
    dimensionless = stokes_harmonic(n, gamma, alpha, theta, n_nodes=_automatic_count(n))
    for a, b in zip(dimensionless, stokes_harmonic(n, gamma, alpha, theta)):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_traced_order_with_static_nodes_is_finite_and_matches_concrete():
    gamma, alpha, theta, B = 5.0, 0.7, 1.1, 3.0

    @jax.jit
    def traced(n):
        return jnp.stack(stokes_harmonic(n, gamma, alpha, theta, B=B, n_nodes=256))

    for n in (1.0, 3.0, 17.0, 40.0):
        got = np.asarray(traced(jnp.asarray(n)))
        assert np.all(np.isfinite(got))
        assert_allclose(got, _oracle(int(n), gamma, alpha, theta, B), rtol=1e-11)
    batched = np.asarray(jax.vmap(traced)(jnp.arange(1.0, 41.0)))
    assert batched.shape == (40, 3) and np.all(np.isfinite(batched))
    # An unresolved order under the static count returns NaN rather than aliasing.
    assert np.all(np.isnan(np.asarray(traced(jnp.asarray(200.0)))))
