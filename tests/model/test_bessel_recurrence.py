"""Bessel derivatives by the order recurrence on one contour (``_bessel_recurrence``).

``bessel_band`` evaluates ``J_{m+k}(x)``, ``|k| <= width``, from one set of
``cos/sin(m t - x cosh(a) sin t)`` on the contour shifted for order ``m``;
negative orders use ``J_{-p} = (-1)^p J_p``. Derivatives follow from
``J_n' = (J_{n-1} - J_{n+1})/2`` on the same contour, which is also the
derivative of the discrete trapezoid sum at a fixed contour (the identity
holds node by node), so the recurrence reproduces automatic differentiation
of the quadrature up to roundoff wherever that is accurate. Checked against
SciPy ``jv/jvp`` over the Bessel resolution classes; finite checks, not
certificates.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import jv, jvp

import syncmoments  # noqa: F401
from syncmoments.bessel import bessel_jn_neighbours
from syncmoments.model._bessel_recurrence import (
    bessel_band,
    derivative_coefficients,
    neighbours_recurrence,
    taylor_neighbours,
)
from syncmoments.model._harmonic_cells import harmonic_lines

CLASSES = (1, 2, 16, 48, 112, 240, 496, 1008)
TINY = 1e-280  # SciPy's relative accuracy is not used below this magnitude


def _count(m):
    return 2 ** math.ceil(math.log2(max(128, 4 * m + 64)))


def _arguments(m, width):
    """Axis limit, the transition around ``x = m``, the resolution edge, ``x < 0``."""
    edge = _count(m) / 2 - m - 1 - width
    x = np.array(
        [1e-8, 1e-4, 1e-3, 0.1 * m, 0.5 * m, 0.9 * m, m * (1 - 1e-6), float(m)]
        + [1.01 * m, 1.0625 * m, edge, -0.5 * m, -1e-4]
    )
    return x[np.abs(x) <= edge]


def _rel(a, b):
    """Max relative difference over entries with ``|b| > TINY`` (scale per row)."""
    a, b = np.asarray(a), np.asarray(b)
    mask = np.abs(b) > TINY
    return float(np.max(np.abs(a - b)[mask] / np.abs(b)[mask])) if mask.any() else 0.0


def _rec(m, n):
    """The triple with recurrence derivatives, stacked; ``y -> (3,)``."""

    def f(y):
        return jnp.stack(neighbours_recurrence(float(m), y, n))

    return f


def _ad(m, n):
    """The triple with autodiff of the shared quadrature, stacked."""

    def f(y):
        return jnp.stack(bessel_jn_neighbours(float(m), y, n_nodes=n))

    return f


def _nested(fn, order):
    for _ in range(order):
        fn = jax.jacfwd(fn)
    return fn


@pytest.mark.parametrize("m", CLASSES)
def test_band_matches_scipy_including_negative_orders(m):
    width = 4
    xs = _arguments(m, width)
    band = np.asarray(bessel_band(jnp.full(xs.shape, float(m)), xs, width, _count(m)))
    orders = m + np.arange(-width, width + 1)
    ref = jv(orders[None, :], xs[:, None])
    # Relative to the largest |J| of the band at each x (orders far below m at
    # x << m are ~1e-300 and carry no weight in any derivative).
    scale = np.max(np.abs(ref), axis=1, keepdims=True)
    keep = scale[:, 0] > TINY  # x = 1e-8 at m >= 48: the whole band underflows
    # Measured (DEV): <= 5.7e-13 (m = 1008, x = 0.9 m).
    assert np.max(np.abs(band - ref)[keep] / scale[keep]) < 2e-12
    zero = np.asarray(bessel_band(float(m), 0.0, width, _count(m)))
    assert_allclose(zero, (orders == 0).astype(float), atol=0.0)


@pytest.mark.parametrize("m", CLASSES)
def test_recurrence_derivatives_match_scipy_through_order_four(m):
    n = _count(m)
    worst = 0.0
    for x in _arguments(m, 5):
        for k in range(0, 5):
            fn = _rec(m, n)
            got = np.asarray(_nested(fn, k)(float(x)))
            ref = np.array([jvp(m - 1, x, k), jvp(m + 1, x, k), jvp(m, x, k + 1)])
            scale = max(np.max(np.abs(ref)), TINY)
            worst = max(worst, float(np.max(np.abs(got - ref)) / scale))
    # Measured (DEV): 2.1e-10 at m = 1008, x = m, order four; elsewhere <= 8e-11.
    # SciPy's jvp is the same order-difference formula, so both carry the
    # cancellation of the fourth difference (|J| / |J''''| ~ 4e3 there).
    assert worst < 5e-10, worst


@pytest.mark.parametrize("m", [16, 48, 112, 240])
def test_recurrence_equals_autodiff_of_the_shared_quadrature(m):
    """Where AD of the quadrature is accurate (``x >= 0.1 m``) both agree."""
    n = _count(m)
    for x in [x for x in _arguments(m, 5) if x >= 0.1 * m]:
        for k in range(1, 4):
            rec = _rec(m, n)
            ad = _ad(m, n)
            a = np.asarray(_nested(ad, k)(float(x)))
            b = np.asarray(_nested(rec, k)(float(x)))
            assert np.max(np.abs(a - b)) <= 2e-12 * np.max(np.abs(a)), (x, k)


def test_recurrence_is_accurate_where_autodiff_of_the_contour_is_not():
    """At ``x = 1e-4``, ``m = 1`` the contour shift ``cosh a = m/x`` amplifies the
    roundoff of the differentiated integrand by ``(2 m/x)^k``; the recurrence
    with reflected negative orders keeps SciPy's values."""
    m, x = 1.0, 1e-4
    rec = _rec(m, 128)
    ad = _ad(m, 128)
    ref = np.array([jvp(0, x, 3), jvp(2, x, 3), jvp(1, x, 4)])
    got = np.asarray(_nested(rec, 3)(x))
    assert_allclose(got, ref, rtol=1e-8)
    assert np.max(np.abs(np.asarray(_nested(ad, 3)(x)) - ref) / np.abs(ref)) > 1e-3


@pytest.mark.parametrize("m", [1, 2, 48, 496])
def test_taylor_coefficients_equal_nested_recurrence(m):
    n, order = _count(m), 4
    xs = _arguments(m, order + 1)
    coeffs = np.asarray(
        derivative_coefficients(jnp.full(xs.shape, float(m)), xs, order, n)
    )
    assert coeffs.shape == (order + 1, 3, xs.size)
    for i, x in enumerate(xs):
        fn = _rec(m, n)
        for k in range(order + 1):
            ref = np.asarray(_nested(fn, k)(float(x)))
            # One band of width 5 against nested widths 2..5: summation order
            # only (measured <= 1.2e-12 relative on an entry 1e-3 of its row).
            scale = np.max(np.abs(ref))
            assert_allclose(coeffs[k, :, i], ref, rtol=1e-11, atol=1e-15 * scale)
    # The Taylor polynomial in h = x - x0 reproduces every derivative up to order.
    x0 = jnp.asarray(0.5 * m)
    c0 = derivative_coefficients(float(m), x0, order, n)
    poly = lambda y: jnp.stack(taylor_neighbours(c0, y - x0))  # noqa
    for k in range(order + 1):
        assert_allclose(
            np.asarray(_nested(poly, k)(0.5 * m)), np.asarray(c0[k]), rtol=1e-14
        )


def test_resolution_guard_and_parity():
    lower, upper, prime = neighbours_recurrence(40.0, 100.0, 128)
    assert np.isnan(lower) and np.isnan(upper) and np.isnan(prime)
    assert np.all(np.isnan(np.asarray(derivative_coefficients(40.0, 100.0, 2, 128))))
    a = np.asarray(
        jax.jacfwd(lambda y: jnp.stack(neighbours_recurrence(5.0, y, 128)))(-2.0)
    )
    ref = np.array([jvp(4, -2.0, 1), jvp(6, -2.0, 1), jvp(5, -2.0, 2)])
    assert_allclose(a, ref, rtol=1e-12)


def _max_quadrature_array(jaxpr, n_nodes):
    """Largest array (elements) with an axis of length ``n_nodes`` in a jaxpr tree."""
    worst = 0
    for eqn in jaxpr.eqns:
        for var in eqn.outvars:
            shape = getattr(var.aval, "shape", ())
            if n_nodes in shape:
                worst = max(worst, int(np.prod(shape)))
        for param in eqn.params.values():
            for sub in param if isinstance(param, (tuple, list)) else (param,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    worst = max(worst, _max_quadrature_array(inner, n_nodes))
    return worst


def test_no_tangent_reaches_the_bessel_quadrature():
    """Third-order nested ``jacfwd`` in ``(gamma, B)`` of the line powers at 16
    points: with the recurrence no array carries both the quadrature axis and a
    tangent axis; the autodiff rule (the probe's positive control) does."""
    n_nodes, points = 256, 16
    mu = jnp.linspace(0.1, 0.9, points)
    eta = jnp.linspace(-0.8, 0.7, points)

    def lines(bessel):
        def f(z):
            out = harmonic_lines(
                7.0, 20.0 + z[0], 1.0 + z[1], mu, eta, n_nodes=n_nodes, bessel=bessel
            )
            return jnp.stack(out)

        return _nested(f, 3)

    z = jnp.zeros(2)
    rec = jax.make_jaxpr(lines(None))(z).jaxpr
    auto = jax.make_jaxpr(lines("autodiff"))(z).jaxpr
    assert _max_quadrature_array(rec, n_nodes) <= points * n_nodes
    assert _max_quadrature_array(auto, n_nodes) >= 2 * points * n_nodes
    assert_allclose(
        np.asarray(lines(None)(z)), np.asarray(lines("autodiff")(z)), rtol=1e-11
    )
