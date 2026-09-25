"""Boundary validation of the propagators' exponential (``syncmoments._expm``).

``expm_pade13`` dispatches on the squaring count ``n``: Pade 13 of ``A 2^-n``,
squared ``n`` times, ``n = ceil(log2(|A|_1 / theta_13))`` (0 below ``theta_13``).
At each threshold ``|A|_1 = theta_13 2^m`` both neighbouring methods (``n`` and
``n + 1`` squarings) are valid and are evaluated directly, bypassing the
dispatcher, and compared with each other and with SciPy, for the matrices
``[[-K ds, I], [0, 0]]`` of the propagators with an absorbing (thick, weak) and
a rotation-dominated ``K``, from ``|K ds| = 1e-300`` to ``theta_13 2^31``.

Tolerance ``1e-14 2^m``: a squaring rounds at the unit roundoff ``u`` and the
exponential of a rotation has condition number about ``|A|_1 = theta_13 2^m``,
so ``u |A|_1 = 6e-16 2^m``, with a factor 16 of margin (measured at most
``2.8e-15 2^m``). ``jax.scipy.linalg.expm`` uses ``floor`` instead, i.e. Pade 13
up to ``2 theta_13``; just below ``2^(m+1) theta_13`` the rotation case then
reaches a relative error of 1e-10 (m = 0) to 0.13 (m = 31) against SciPy,
which the test records as a regression bound for the ``floor`` method.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import syncmoments  # noqa: F401  (enables x64)
from syncmoments._expm import _THETA13, expm_pade13, expm_squared
from syncmoments.transfer import mueller_matrix

KS = {
    "thick": mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0),
    "weak": mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1),
    "rotation": mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.3, -0.2, 1.0),
}
EXPONENTS = (0, 1, 3, 10, 20, 31)
POSITIONS = {"below": 1 - 1e-9, "at": 1.0, "above": 1 + 1e-9}
MAX_SQUARINGS = 40


def _propagator_matrix(K, norm):
    """``[[-K ds, I], [0, 0]]`` with ``ds`` chosen so that ``|K ds|_1 = norm``."""
    K = np.asarray(K)
    A = np.zeros((8, 8))
    A[:4, :4] = -K * norm / np.max(np.sum(np.abs(K), axis=0))
    A[:4, 4:] = np.eye(4)
    return A


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), np.max(np.abs(b))))


def _squared(A, n):
    return np.asarray(
        expm_squared(jnp.asarray(A), jnp.asarray(float(n)), MAX_SQUARINGS)
    )


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("m", EXPONENTS)
@pytest.mark.parametrize("name", KS)
def test_neighbouring_squaring_counts_agree_at_each_threshold(name, m, position):
    A = _propagator_matrix(KS[name], _THETA13 * 2.0**m * POSITIONS[position])
    n = max(0, int(np.ceil(np.log2(np.max(np.sum(np.abs(A), axis=0)) / _THETA13))))
    tol = 1e-14 * 2.0**m
    chosen, more = _squared(A, n), _squared(A, n + 1)
    reference = sla.expm(A)
    assert _rel(chosen, more) < tol
    assert _rel(chosen, reference) < tol and _rel(more, reference) < tol
    assert _rel(expm_pade13(jnp.asarray(A), MAX_SQUARINGS), chosen) == 0.0


@pytest.mark.parametrize("m", EXPONENTS)
@pytest.mark.parametrize("name", KS)
def test_frechet_derivative_matches_scipy_at_the_thresholds(name, m):
    """Forward mode of the dispatcher (the rule's ``jax.jvp``) against
    ``scipy.linalg.expm_frechet`` on both sides of ``theta_13 2^m``."""
    E = np.zeros((8, 8))
    E[:4, :4] = -np.asarray(KS[name])
    for position in POSITIONS.values():
        A = _propagator_matrix(KS[name], _THETA13 * 2.0**m * position)
        f = lambda X: expm_pade13(X, MAX_SQUARINGS)  # noqa: E731
        got = jax.jvp(f, (jnp.asarray(A),), (jnp.asarray(E),))[1]
        assert _rel(got, sla.expm_frechet(A, E, compute_expm=False)) < 1e-14 * 2.0**m


@pytest.mark.parametrize("norm", (1e-300, 1e-8, 1.0))
def test_small_propagator_norms(norm):
    """``|K ds|`` down to 1e-300: the identity block keeps ``|A|_1 = 1``, n = 0."""
    for K in KS.values():
        A = _propagator_matrix(K, norm)
        assert _rel(expm_pade13(jnp.asarray(A), MAX_SQUARINGS), sla.expm(A)) < 1e-14


@pytest.mark.parametrize("m", (0, 10, 20, 31))
def test_floor_squaring_count_is_inaccurate_just_below_the_next_threshold(m):
    """The ``floor`` rule of ``jax.scipy.linalg.expm`` (Pade 13 up to
    ``2 theta_13``) at ``|A|_1 = 2^(m+1) theta_13 (1 - 1e-9)``, rotation case:
    measured 1.0e-10, 1.2e-7, 1.2e-4 and 0.13 for m = 0, 10, 20, 31, which
    is why the dispatcher takes ``ceil``."""
    A = _propagator_matrix(KS["rotation"], _THETA13 * 2.0 ** (m + 1) * (1 - 1e-9))
    floor_err = _rel(_squared(A, m), sla.expm(A))
    ceil_err = _rel(_squared(A, m + 1), sla.expm(A))
    assert floor_err > 1e-11 and floor_err > 100 * ceil_err
    assert ceil_err < 1e-14 * 2.0**m


@pytest.mark.parametrize("budget", (0, 3, 5))
def test_max_squarings_limit_matches_jax_expm(budget):
    """NaN exactly where ``jax.scipy.linalg.expm`` returns NaN: ``floor`` above
    the budget. Between (``ceil`` above, ``floor`` at the budget) ``floor`` is
    used and the result stays finite and accurate to the ``floor`` method."""
    for scale in (1.0, 1 + 1e-9, 2 * (1 - 1e-9), 2.0, 4.0):
        A = _propagator_matrix(KS["thick"], _THETA13 * 2.0**budget * scale)
        ours = np.asarray(expm_pade13(jnp.asarray(A), budget))
        theirs = np.asarray(jax.scipy.linalg.expm(jnp.asarray(A), max_squarings=budget))
        assert np.isnan(ours).any() == np.isnan(theirs).any(), scale
        if not np.isnan(ours).any():
            assert _rel(ours, sla.expm(A)) < 1e-11


def test_batched_members_equal_unbatched_values_and_derivatives():
    """``vmap`` over members that need 0, 5, 20 and 31 squarings: the squaring
    steps run while any member needs them and are selected per member, so each
    member equals its unbatched value; ``grad`` of a sum over the batch (the
    round-4 failure mode) equals the per-member gradients."""
    norms = [0.5, _THETA13 * 2**5, _THETA13 * 2**20, _THETA13 * 2**31]
    A = jnp.asarray(np.stack([_propagator_matrix(KS["weak"], x) for x in norms]))
    W = jnp.asarray(np.cos(np.arange(64.0)).reshape(8, 8))
    f = lambda X: jnp.sum(W * expm_pade13(X, MAX_SQUARINGS))  # noqa: E731
    batched = jax.vmap(lambda X: expm_pade13(X, MAX_SQUARINGS))(A)
    grads = jax.grad(lambda As: jnp.sum(jax.vmap(f)(As)))(A)
    for i in range(len(norms)):
        assert _rel(batched[i], expm_pade13(A[i], MAX_SQUARINGS)) < 1e-15
        assert _rel(grads[i], jax.grad(f)(A[i])) < 1e-15
