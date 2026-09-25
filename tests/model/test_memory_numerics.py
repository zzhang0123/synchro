"""Shared Bessel quadrature and bounded-memory chunking of ``HarmonicKernel``.

Shared quadrature: ``syncmoments.bessel.bessel_jn_neighbours`` evaluates
``J_{m-1}``, ``J_{m+1}`` and ``J_m'`` from one set of ``cos/sin(m t - x sin t)``
on the contour shifted for order ``m``. Checked against SciPy ``jv/jvp`` over
the Bessel resolution classes, at small ``x`` (the axis limit), at ``x`` close
to ``m`` and at the resolution edge, against the separate quadratures of
``bessel_jn_and_prime``, and its derivatives against SciPy and Richardson
differences. These are finite checks, not accuracy certificates.

Chunking: the projections are summed over harmonics and over blocks of
angular points so that one ``lax.map`` step holds at most ``chunk_budget``
Bessel integrand values; the blocking changes only the summation order.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import jv, jvp

import syncmoments  # noqa: F401
from syncmoments.bessel import bessel_jn_and_prime, bessel_jn_neighbours
from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model._harmonic_cells import chunk_plan, harmonic_lines
from syncmoments.model._kernel_helpers import sine_from_cosine
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.phase import TaylorPhase

from _harmonic_oracles import GAMMA0, S_DEPTH, benchmark_channels, richardson

CLASSES = (1, 2, 16, 48, 112, 240, 496, 1008)
# Measured (DEV, jax 0.10.0): <= 8.3e-13 relative to SciPy for every function and
# class below; the separate quadratures
# of bessel_jn_and_prime sit at the same level.
SCIPY_RTOL = 2e-12
# Floor below which SciPy's own relative accuracy is not used (values ~1e-280).
TINY = 1e-280


def _count(m):
    return 2 ** math.ceil(math.log2(max(128, 4 * m + 64)))


def _arguments(m):
    """Axis limit, the transition region around ``x = m`` and the resolution edge."""
    edge = _count(m) / 2 - m - 1
    x = np.array(
        [1e-8, 1e-3, 0.1 * m, 0.5 * m, 0.9 * m, 0.99 * m]
        + [m * (1 - 1e-6), float(m), m * (1 + 1e-6), 1.01 * m, 1.0625 * m, edge]
    )
    return x[x <= edge]


def _rel(a, b):
    mask = np.abs(b) > TINY
    return float(np.max(np.abs(a - b)[mask] / np.abs(b)[mask]))


@pytest.mark.parametrize("m", CLASSES)
def test_neighbours_match_scipy_over_resolution_classes(m):
    x = _arguments(m)
    lower, upper, prime = (
        np.asarray(a) for a in bessel_jn_neighbours(float(m), x, n_nodes=_count(m))
    )
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert _rel(lower, jv(m - 1, x)) < SCIPY_RTOL
    assert _rel(upper, jv(m + 1, x)) < SCIPY_RTOL
    assert _rel(prime, jvp(m, x)) < SCIPY_RTOL
    # Same nodes, separate contours (v0.2.0 path): equal to the same level.
    value, sep_prime = bessel_jn_and_prime(
        np.array([m - 1.0, m, m + 1.0])[:, None], x[None], n_nodes=_count(m)
    )
    assert _rel(lower, np.asarray(value[0])) < SCIPY_RTOL
    assert _rel(upper, np.asarray(value[2])) < SCIPY_RTOL
    assert _rel(prime, np.asarray(sep_prime[1])) < SCIPY_RTOL


def test_neighbours_axis_limit_parity_and_resolution_guard():
    m = np.array([1.0, 2.0, 7.0])
    lower, upper, prime = bessel_jn_neighbours(m, 0.0, n_nodes=128)
    assert_allclose(np.asarray(lower), [1.0, 0.0, 0.0], atol=0)
    assert_allclose(np.asarray(upper), [0.0, 0.0, 0.0], atol=0)
    assert_allclose(np.asarray(prime), [0.5, 0.0, 0.0], atol=0)
    # J_n(-x) = (-1)^n J_n(x) and J_n'(-x) = (-1)^(n+1) J_n'(x).
    for n in (1.0, 2.0, 7.0):
        pos = np.stack([np.asarray(a) for a in bessel_jn_neighbours(n, 3.3)])
        neg = np.stack([np.asarray(a) for a in bessel_jn_neighbours(n, -3.3)])
        assert_allclose(neg, -((-1.0) ** n) * pos, rtol=1e-15)
    # Small x: J_{m+1}/J_{m-1} -> x^2 / (4 m (m+1)).
    small = bessel_jn_neighbours(5.0, 1e-6)
    assert_allclose(float(small[1] / small[0]), 1e-12 / 120, rtol=1e-10)
    # Outside m + 1 + |x| <= n_nodes/2, for m = 0 and for a non-integer order: NaN.
    bad = bessel_jn_neighbours(np.array([63.0, 0.0, 2.5]), np.ones(3), n_nodes=128)
    assert np.all(np.isnan(np.stack([np.asarray(a) for a in bad])))
    edge = bessel_jn_neighbours(62.0, 1.0, n_nodes=128)
    assert np.all(np.isfinite(np.stack([np.asarray(a) for a in edge])))


@pytest.mark.parametrize("m", [1, 16, 240])
def test_neighbours_derivatives_match_scipy_and_richardson(m):
    count = _count(m)
    grads = [
        jax.jit(jax.grad(lambda x, i=i: bessel_jn_neighbours(m, x, n_nodes=count)[i]))
        for i in range(3)
    ]
    edge = count / 2 - m - 1.01  # finite differences stay resolved
    for x in (0.3 * m, max(m - 0.5, 0.2), m * (1 + 1e-6), min(1.2 * m, edge)):
        got = np.array([float(g(x)) for g in grads])
        second = -jvp(m, x) / x - (1 - m**2 / x**2) * jv(m, x)
        expected = np.array([jvp(m - 1, x), jvp(m + 1, x), second])
        scale = np.max(np.abs(expected))
        assert_allclose(got, expected, rtol=1e-10, atol=1e-11 * scale)
        fn = lambda y: np.array(  # noqa: E731
            [float(a) for a in bessel_jn_neighbours(m, y[0], n_nodes=count)]
        )
        fd = richardson(fn, np.array([x, 0.0]), 0, 1e-3 * min(x, 1.0))
        assert_allclose(got, fd, rtol=1e-6, atol=1e-8 * scale)


def _separate_lines(m, gamma, B, mu, eta, n_nodes):
    """The v0.2.0 ``harmonic_lines``: three separate contours per point."""
    m, gamma, B, mu, eta = jnp.broadcast_arrays(
        *(jnp.asarray(v, dtype=float) for v in (m, gamma, B, mu, eta))
    )
    beta = jnp.sqrt(1.0 - 1.0 / gamma**2)
    b_par, b_perp = beta * mu, beta * sine_from_cosine(mu)
    st, D = sine_from_cosine(eta), 1.0 - beta * mu * eta
    x = m * b_perp * st / D
    value, prime = bessel_jn_and_prime(
        jnp.stack([m - 1.0, m, m + 1.0]), x[None], n_nodes=n_nodes
    )
    a_par = (eta - b_par) * b_perp * (value[0] + value[2]) / (2.0 * D)
    a_perp = b_perp * prime[1]
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2.0 * jnp.pi * C_CGS) * m**2 / D**3
    return pref * (a_par**2 + a_perp**2), pref * (a_par**2 - a_perp**2)


@pytest.mark.parametrize("gamma", [1.01, 2.0, 20.0, 50.0])
def test_harmonic_lines_equal_the_separate_quadratures(gamma):
    rng = np.random.default_rng(3)
    mu = np.concatenate([rng.uniform(-1, 1, 40), [1.0, -1.0, 0.0, 0.3, 1 - 1e-12]])
    eta = np.concatenate([rng.uniform(-1, 1, 40), [0.2, 0.5, 1.0, -1.0, 0.7]])
    for m in (1.0, 2.0, 17.0, 40.0):
        n_nodes = _count(int(m))
        I, Q, V, _ = harmonic_lines(m, gamma, 1.3, mu, eta, n_nodes=n_nodes)
        sep_I, sep_Q = _separate_lines(m, gamma, 1.3, mu, eta, n_nodes)
        scale = float(np.max(np.asarray(sep_I)))
        assert np.all(np.isfinite(np.asarray(V)))
        assert_allclose(np.asarray(I), sep_I, rtol=1e-12, atol=1e-13 * scale)
        assert_allclose(np.asarray(Q), sep_Q, rtol=1e-12, atol=1e-13 * scale)


# -- bounded working set ---------------------------------------------------------------


@pytest.mark.parametrize(
    "points,n_nodes,m_chunk,budget",
    [
        (1, 256, 64, 1 << 20),
        (8192, 256, 64, 1 << 20),
        (8192, 2048, 64, 1 << 20),
        (2304, 256, 64, 1 << 20),
        (64, 128, 3, 1 << 20),
        (8192, 65536, 64, 1 << 12),
        (1, 65536, 1, 1),
    ],
)
def test_chunk_plan_bounds_the_working_set(points, n_nodes, m_chunk, budget):
    harmonics, block = chunk_plan(points, n_nodes, m_chunk, budget)
    assert 1 <= harmonics <= m_chunk and 1 <= block <= points
    # One point with one harmonic is the smallest unit; otherwise the budget holds.
    assert harmonics * block * n_nodes <= max(budget, n_nodes)
    if points * n_nodes > budget:
        assert harmonics == 1


@pytest.mark.parametrize("quadrature", ["product", "tensor"])
def test_projection_and_derivatives_do_not_depend_on_blocking(quadrature):
    channels = benchmark_channels()
    truncation = MomentIndex.build(Truncation(2, 2, 1))
    kwargs = dict(n_outer=12, n_inner=12, n_mu=10, n_eta=10, quadrature=quadrature)

    def taylor(kernel):
        def f(z):
            modes = kernel.angular_projection(
                channels,
                GAMMA0 + z[0],
                1.0 + 0.05 * z[1],
                phase=TaylorPhase(1),
                depth_ref=4 * S_DEPTH,
                s_depth=S_DEPTH,
                truncation=truncation,
            )
            return jnp.concatenate([modes.I.ravel(), modes.V.ravel(), modes.P.ravel()])

        z = jnp.zeros(2)
        return np.asarray(f(z)), np.asarray(jax.jit(jax.jacfwd(f))(z))

    one_block = taylor(HarmonicKernel(40, m_chunk=64, chunk_budget=1 << 30, **kwargs))
    # The chunk_plan dispatch threshold: all points of one harmonic in one step
    # (budget = points * n_nodes) versus blocks of points (one value less).
    points = 2 * 12 * 12 if quadrature == "product" else 10 * 10
    threshold = points * HarmonicKernel(40).resolution()
    cases = ((1 << 12, 1), (5000, 7), (threshold, 64), (threshold - 1, 64))
    for budget, chunk in cases:
        blocked = taylor(
            HarmonicKernel(40, m_chunk=chunk, chunk_budget=budget, **kwargs)
        )
        for a, b in zip(blocked, one_block):
            assert_allclose(a, b, rtol=1e-12, atol=1e-14 * np.max(np.abs(b)))


# Cap on XLA's compiled temporaries of the nested second-order jacfwd below.
# Measured (default derivatives="analytic", recurrence rule): 7.80e7 bytes (DEV,
# jax 0.10.0) and 8.61e7 (NEW, jax 0.10.2). The same compile with
# derivatives="autodiff" (tangents through the Bessel quadrature) needs 2.37e8
# (DEV) and 5.65e8 (NEW); v0.2.0 needed 9.7e9 and 4.40e10. The cap is 2.3x above
# the larger analytic figure and below both autodiff figures.
JACFWD2_TEMP_CAP_BYTES = 2.0e8
# The analytic rule must stay well below the autodiff rule on the same compile
# (measured ratios autodiff/analytic: 3.0 in DEV, 6.6 in NEW).
AUTODIFF_OVER_ANALYTIC_MIN = 2.0


def _jacfwd2_temp_bytes(kernel):
    """XLA temp bytes of ``jit(jacfwd(jacfwd(f)))`` at the benchmark size
    (m_max = 40, 64 x 64 product nodes, ``Truncation(2, 2, 2)``); compiled only,
    not run. ``None`` when the backend has no memory analysis."""
    channels = benchmark_channels()
    index = MomentIndex.build(Truncation(2, 2, 2))

    def f(z):
        modes = kernel.angular_projection(
            channels,
            GAMMA0 + z[0],
            1.0 + 0.05 * z[1],
            phase=TaylorPhase(2),
            depth_ref=4 * S_DEPTH,
            s_depth=S_DEPTH,
            truncation=index,
        )
        return jnp.sum(modes.I) + jnp.sum(jnp.abs(modes.P))

    compiled = jax.jit(jax.jacfwd(jax.jacfwd(f))).lower(jnp.zeros(2)).compile()
    analysis = compiled.memory_analysis()
    return None if analysis is None else analysis.temp_size_in_bytes


def test_projection_temporaries_stay_bounded_under_second_derivatives(record_property):
    """The default kernel's second-derivative temporaries stay under
    ``JACFWD2_TEMP_CAP_BYTES`` and at most half of the ``derivatives="autodiff"``
    figure, so a regression that routes ``angular_projection`` tangents back
    through the Bessel quadrature fails in both environments."""
    temp = _jacfwd2_temp_bytes(HarmonicKernel(40))
    if temp is None:
        pytest.skip("XLA memory analysis not available on this backend")
    autodiff = _jacfwd2_temp_bytes(HarmonicKernel(40, derivatives="autodiff"))
    record_property("jacfwd2_temp_bytes", temp)
    record_property("jacfwd2_temp_bytes_autodiff", autodiff)
    assert temp < JACFWD2_TEMP_CAP_BYTES, temp
    assert temp * AUTODIFF_OVER_ANALYTIC_MIN < autodiff, (temp, autodiff)
