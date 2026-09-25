"""Autodiff of ``bessel_jn_neighbours`` at ``|x| << n`` (docstring claim).

The contour of order ``n`` is shifted by ``a = arccosh(n/|x|)``; the order
``n -+ 1`` integrands carry ``exp(+-a)`` and forward-mode tangents through the
quadrature amplify roundoff by about ``(2n/|x|)^k`` at derivative order ``k``.
At ``n = 1``, ``x = 1e-6`` (default count, 128 nodes) the third derivatives
of ``J_0`` and ``J_1'`` are off by more than 1 against SciPy (true values of
order ``3e-7``), while the order recurrence
(``syncmoments.model._bessel_recurrence.neighbours_recurrence``, used by
``HarmonicKernel(derivatives="analytic")``) agrees to ``1e-15``. The
docstring of ``bessel_jn_neighbours`` states this; this test pins the quoted
numbers in both environments.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import jvp

import syncmoments  # noqa: F401  (float64)
from syncmoments.bessel import bessel_jn_neighbours
from syncmoments.model._bessel_recurrence import neighbours_recurrence

X = 1e-6
NODES = 128  # the automatic count for n = 1 (``2^ceil(log2(max(128, 4 (n+1) + 64)))``)


def _nested(f, k):
    for _ in range(k):
        f = jax.jacfwd(f)
    return f


def _scipy(k):
    """``d^k/dx^k (J_0, J_2, J_1')`` at ``X``."""
    return np.array([jvp(0, X, k), jvp(2, X, k), jvp(1, X, k + 1)])


def _autodiff(k):
    return np.asarray(_nested(lambda y: jnp.stack(bessel_jn_neighbours(1, y)), k)(X))


def _recurrence(k):
    f = _nested(lambda y: jnp.stack(neighbours_recurrence(1.0, y, NODES)), k)
    return np.asarray(f(X))


def test_default_count_matches_the_quoted_one():
    explicit = _nested(
        lambda y: jnp.stack(bessel_jn_neighbours(1, y, n_nodes=NODES)), 3
    )(X)
    np.testing.assert_array_equal(_autodiff(3), np.asarray(explicit))


def test_autodiff_third_derivative_is_off_by_more_than_one_at_small_x():
    exact = _scipy(3)
    assert np.max(np.abs(exact)) < 1e-6
    error = np.abs(_autodiff(3) - exact)
    assert error[0] > 1.0 and error[2] > 1.0  # J_0''' and J_1''''
    assert np.abs(_autodiff(2) - _scipy(2)).max() > 1e-4  # already at k = 2


def test_recurrence_derivatives_agree_with_scipy_at_small_x():
    for k in (1, 2, 3):
        assert np.abs(_recurrence(k) - _scipy(k)).max() <= 1e-15


def test_docstring_states_the_warning_and_the_remedy():
    doc = bessel_jn_neighbours.__doc__
    assert "|x| << n" in doc
    assert "x = 1e-6" in doc
    assert "neighbours_recurrence" in doc
    assert 'derivatives="analytic"' in doc
