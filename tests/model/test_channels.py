"""Channel responses: NumPy oracles, edge smoothness, quadrature and AD safety."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad
from scipy.special import erf, expit

from synchro.model.channels import Channels

CENTRES = np.array([2.0e8, 4.0e8, 8.0e8])
WIDTHS = 0.65 * CENTRES
FAMILIES = ("bump", "planck_taper", "raised_cosine", "tophat", "gaussian")


def make(family, **kwargs):
    if family == "gaussian":
        return Channels.gaussian(CENTRES, 0.1 * CENTRES, **kwargs)
    return getattr(Channels, family)(CENTRES, WIDTHS, **kwargs)


def oracle(family, nu, taper=0.5, support_sigma=6.0):
    """Direct NumPy formulas, written independently of the JAX code."""
    nu = np.asarray(nu, float)
    out = np.zeros((CENTRES.size, *nu.shape))
    for j, (c, w) in enumerate(zip(CENTRES, WIDTHS)):
        if family == "gaussian":
            t = (nu - c) / (0.1 * c)
            inside = np.abs(t) < support_sigma
            out[j][inside] = np.exp(-0.5 * t[inside] ** 2)
            continue
        t = (nu - c) / w
        inside = np.abs(t) < 1
        if family == "bump":
            out[j][inside] = np.exp(1 - 1 / (1 - t[inside] ** 2))
        elif family == "raised_cosine":
            out[j][inside] = 0.5 * (1 + np.cos(np.pi * t[inside]))
        elif family == "tophat":
            out[j][inside] = 1.0
        elif family == "planck_taper":
            v = np.minimum((t + 1) / 2, (1 - t) / 2)
            value = np.ones_like(t)
            ramp = inside & (v < taper)
            value[ramp] = expit(taper / (taper - v[ramp]) - taper / v[ramp])
            out[j][inside] = value[inside]
    return out


def exact_integral(family, j, taper=0.5, support_sigma=6.0):
    """Closed-form band integral of the unit-peak response (Hz)."""
    w = WIDTHS[j]
    if family == "bump":
        x, gw = np.polynomial.legendre.leggauss(600)
        return w * float(np.sum(gw * np.exp(1 - 1 / (1 - x**2))))
    if family == "planck_taper":
        return 2 * w * (1 - taper)
    if family == "raised_cosine":
        return w
    if family == "tophat":
        return 2 * w
    sigma = 0.1 * CENTRES[j]
    return sigma * math.sqrt(2 * math.pi) * erf(support_sigma / math.sqrt(2))


@pytest.mark.parametrize("family", FAMILIES)
def test_values_match_numpy_formulas(family):
    channels = make(family)
    nu = np.linspace(0.1e8, 20e8, 4001).reshape(4001)
    expected = oracle(family, nu)
    actual = np.asarray(channels(nu))
    assert actual.shape == (3, 4001)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    assert np.all(actual[expected == 0] == 0)
    # Arbitrary leading shapes are preserved.
    grid = nu[:12].reshape(3, 4)
    assert channels(grid).shape == (3, 3, 4)
    assert_allclose(np.asarray(channels(grid)), oracle(family, grid), rtol=1e-14)
    assert channels.n_ch == 3


@pytest.mark.parametrize("taper", [0.05, 0.25, 0.5])
def test_planck_taper_values_flat_region_and_taper_bounds(taper):
    channels = Channels.planck_taper(CENTRES, WIDTHS, taper=taper)
    nu = np.linspace(0.3e8, 15e8, 2001)
    assert_allclose(np.asarray(channels(nu)), oracle("planck_taper", nu, taper))
    if taper < 0.5:
        flat = CENTRES + (1 - 2 * taper) * 0.999 * WIDTHS
        assert_allclose(np.diag(np.asarray(channels(flat))), 1.0, atol=1e-15)
    for bad in [0.0, -0.1, 0.51, 1.0]:
        with pytest.raises(ValueError):
            Channels.planck_taper(CENTRES, WIDTHS, taper=bad)


@pytest.mark.parametrize("family", FAMILIES)
def test_unit_integral_normalisation_integrates_to_one(family):
    channels = make(family, normalisation="unit_integral")
    unit_peak = make(family)
    scale = np.asarray(channels.peak_scale())
    assert_allclose(np.asarray(unit_peak.peak_scale()), 1.0)
    for j in range(3):
        lo, hi = np.asarray(channels.support)[j]
        assert_allclose(scale[j], 1 / exact_integral(family, j), rtol=1e-13)
        pieces = np.linspace(lo, hi, 9)
        total = sum(
            quad(
                lambda nu: float(channels(jnp.asarray(nu))[j]),
                a,
                b,
                epsabs=1e-14,
                epsrel=1e-14,
                limit=200,
            )[0]
            for a, b in zip(pieces[:-1], pieces[1:])
        )
        assert abs(total - 1) < 1e-12
    # The declared normalisation is a static, hashable part of the provenance.
    assert channels.normalisation == "unit_integral"
    assert hash(channels.describe()) == hash(channels.describe())
    assert channels.describe() != unit_peak.describe()


@pytest.mark.parametrize("family", ["bump", "planck_taper"])
@pytest.mark.parametrize("side", [-1, 1])
def test_smooth_families_are_flat_through_third_order_at_edges(family, side):
    channels = make(family)
    j = 1
    edge = CENTRES[j] + side * WIDTHS[j]
    w = WIDTHS[j]

    def scalar(nu):
        return channels(nu)[j]

    derivatives = [scalar]
    for _ in range(3):
        derivatives.append(jax.jacfwd(derivatives[-1]))
    # Dimensionless derivatives w^k d^k R / dnu^k just inside and outside.
    for delta in [1e-2, 5e-3, 2e-3]:
        inside = jnp.asarray(edge - side * delta * w)
        for k, fn in enumerate(derivatives):
            assert abs(float(fn(inside))) * w**k < 1e-8
        outside = jnp.asarray(edge + side * delta * w)
        for fn in derivatives:
            assert float(fn(outside)) == 0.0
    # Finite differences of the response itself straddling the edge.
    h = 1e-3 * w
    stencil = np.asarray(channels(edge + h * np.arange(-3, 4)))[j]
    third = (stencil[6] - 3 * stencil[5] + 3 * stencil[4] - stencil[3]) / h**3
    first = (stencil[4] - stencil[2]) / (2 * h)
    assert abs(first) * w < 1e-6 and abs(third) * w**3 < 1e-6
    assert channels.smoothness == 99


def test_raised_cosine_is_c1_with_a_second_derivative_jump():
    channels = Channels.raised_cosine(CENTRES, WIDTHS)
    j, w = 0, WIDTHS[0]
    edge = CENTRES[j] + w
    d1 = jax.jacfwd(lambda nu: channels(nu)[j])
    d2 = jax.jacfwd(d1)
    inside, outside = jnp.asarray(edge - 1e-9 * w), jnp.asarray(edge + 1e-9 * w)
    assert abs(float(channels(inside)[j])) < 1e-15
    assert abs(float(d1(inside))) * w < 1e-8
    assert float(d1(outside)) == 0.0 and float(d2(outside)) == 0.0
    assert_allclose(float(d2(inside)) * w**2, 0.5 * np.pi**2, rtol=1e-6)
    assert channels.smoothness == 1


@pytest.mark.parametrize("support_sigma", [4.0, 6.0, 8.0])
def test_gaussian_edge_jump_is_recorded(support_sigma):
    channels = Channels.gaussian(CENTRES, 0.1 * CENTRES, support_sigma=support_sigma)
    lo, hi = np.asarray(channels.support)[0]
    sigma = 0.1 * CENTRES[0]
    assert_allclose([lo, hi], CENTRES[0] + sigma * np.array([-1, 1]) * support_sigma)
    jump = math.exp(-0.5 * support_sigma**2)
    just_inside = float(channels(jnp.asarray(hi * (1 - 1e-12)))[0])
    assert_allclose(just_inside, jump, rtol=1e-6)
    assert float(channels(jnp.asarray(hi))[0]) == 0.0
    described = dict(channels.describe())
    assert_allclose(described["edge_jump"], jump)
    assert channels.smoothness == 0
    assert_allclose(np.asarray(channels.widths_hz), 0.1 * CENTRES)
    with pytest.raises(ValueError):
        Channels.gaussian(CENTRES, 0.1 * CENTRES, support_sigma=0.0)


def test_tophat_is_one_inside_zero_outside_smoothness_zero():
    channels = Channels.tophat(CENTRES, WIDTHS)
    edges = CENTRES[:, None] + WIDTHS[:, None] * np.array([-1, 1])
    values = np.asarray(channels(edges.ravel()))
    assert np.all(values[np.arange(3).repeat(2), np.arange(6)] == 0)
    inside = np.asarray(channels(CENTRES))
    assert np.all(np.diag(inside) == 1)
    assert channels.smoothness == 0


@pytest.mark.parametrize("delta", [1e-12, 1e-9, 1e-6])
def test_both_sides_of_the_unit_t_boundary(delta):
    for family in FAMILIES:
        channels = make(family)
        lo, hi = np.asarray(channels.support)[2]
        half = (hi - lo) / 2
        probe = np.array(
            [lo - delta * half, lo + delta * half, hi - delta * half, hi + delta * half]
        )
        values = np.asarray(channels(probe))[2]
        assert values[0] == 0.0 and values[3] == 0.0
        assert np.all(values[[1, 2]] >= 0)
        # Raised cosine: 1 - cos(pi delta) underflows below delta ~ 1e-8 (float64).
        positive = family in ("tophat", "gaussian")
        if positive or (family == "raised_cosine" and delta >= 1e-6):
            assert np.all(values[[1, 2]] > 0)
        if family in ("bump", "planck_taper") and delta <= 1e-9:
            assert np.all(values[[1, 2]] == 0)
        assert np.all(np.isfinite(values))


@pytest.mark.parametrize("family", FAMILIES)
def test_jacfwd_is_finite_everywhere_and_jit_matches(family):
    channels = make(family)
    nu = np.concatenate(
        [
            np.linspace(1e7, 3e9, 301),
            np.asarray(channels.support).ravel(),
            np.asarray(channels.support).ravel() * (1 + 1e-15),
            np.asarray(channels.support).ravel() * (1 - 1e-15),
            CENTRES,
        ]
    )
    jac = jax.jacfwd(lambda x: channels(x).sum(axis=0))(jnp.asarray(nu))
    assert np.all(np.isfinite(np.asarray(jac)))
    hess = jax.jacfwd(jax.jacfwd(lambda x: channels(x)[1]))(jnp.asarray(nu[:40]))
    assert np.all(np.isfinite(np.asarray(hess)))
    third = jax.jacfwd(jax.jacfwd(jax.jacfwd(lambda x: channels(x)[1].sum())))(
        jnp.asarray(nu[300:312])
    )
    assert np.all(np.isfinite(np.asarray(third)))
    compiled = jax.jit(channels.__call__)(jnp.asarray(nu))
    # Values below 1e-30 differ at 1e-12 relative between XLA and eager sigmoids.
    assert_allclose(
        np.asarray(compiled), np.asarray(channels(nu)), rtol=1e-12, atol=1e-30
    )

    # Differentiation with respect to the channel parameters (moving support).
    def moved(centres, widths):
        if family == "gaussian":
            moved_channels = Channels.gaussian(centres, widths)
        else:
            moved_channels = getattr(Channels, family)(centres, widths)
        return moved_channels(jnp.asarray(nu)).sum()

    grads = jax.grad(moved, argnums=(0, 1))(
        jnp.asarray(CENTRES),
        jnp.asarray(WIDTHS if family != "gaussian" else 0.1 * CENTRES),
    )
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


def test_from_table_nu_and_omega_arguments_agree():
    grid = np.linspace(1.0e8, 3.0e8, 41)
    response = np.zeros_like(grid)
    body = slice(5, 36)
    response[body] = 0.8 * np.sin(np.pi * (grid[body] - grid[5]) / (grid[35] - grid[5]))
    by_nu = Channels.from_table(grid, response, smoothness=0)
    by_omega = Channels.from_table(
        2 * np.pi * grid, response, smoothness=0, argument="omega"
    )
    probe = np.linspace(0.9e8, 3.1e8, 777)
    assert_allclose(np.asarray(by_nu(probe)), np.asarray(by_omega(probe)), rtol=1e-13)
    # Linear interpolation of the unit-peak table; exact zero outside the support.
    expected = np.interp(probe, grid, response / response.max())
    expected[(probe < grid[4]) | (probe > grid[36])] = 0
    assert_allclose(np.asarray(by_nu(probe))[0], expected, atol=1e-15)
    lo, hi = np.asarray(by_nu.support)[0]
    # sin(0) = 0 at grid[5]; the interpolant vanishes beyond grid[5] and grid[36].
    assert_allclose([lo, hi], [grid[5], grid[36]])
    assert by_nu.family == "table" and by_nu.n_ch == 1 and by_nu.smoothness == 0
    assert hash(by_nu.describe()) == hash(by_nu.describe())
    unit = Channels.from_table(
        grid, response, smoothness=0, normalisation="unit_integral"
    )
    area = np.trapezoid(expected, probe)
    assert_allclose(float(unit.peak_scale()[0]) * area, 1.0, rtol=2e-4)
    assert_allclose(
        float(unit.peak_scale()[0]), 1 / np.trapezoid(response / response.max(), grid)
    )
    jac = jax.jacfwd(lambda x: unit(x)[0])(jnp.asarray(probe[::50]))
    assert np.all(np.isfinite(np.asarray(jac)))
    # Two channels sharing one grid.
    two = Channels.from_table(
        grid, np.stack([response, np.roll(response, 2)]), smoothness=1
    )
    assert two.n_ch == 2 and two.support.shape == (2, 2)
    with pytest.raises(ValueError):
        Channels.from_table(grid, response, smoothness=0, argument="hz")
    with pytest.raises(ValueError):
        Channels.from_table(grid, response[:-1], smoothness=0)
    with pytest.raises(ValueError):
        Channels.from_table(grid, np.zeros_like(response), smoothness=0)
    with pytest.raises(ValueError):
        Channels.from_table(grid[::-1], response[::-1], smoothness=0)


@pytest.mark.parametrize("family", FAMILIES)
def test_gauss_legendre_nodes_and_weights_on_each_support(family):
    channels = make(family, n_nu=48)
    nodes, weights = np.asarray(channels.nodes), np.asarray(channels.weights)
    support = np.asarray(channels.support)
    assert nodes.shape == weights.shape == (3, 48) and channels.n_nu == 48
    assert np.all(nodes > support[:, :1]) and np.all(nodes < support[:, 1:])
    assert_allclose(weights.sum(axis=1), support[:, 1] - support[:, 0], rtol=1e-15)
    # Exact for polynomials of degree < 2 n_nu; check degree 7 in the scaled variable.
    x = (nodes - support[:, :1]) / (support[:, 1:] - support[:, :1])
    assert_allclose(
        (weights * x**7).sum(axis=1) / (support[:, 1] - support[:, 0]),
        1 / 8,
        rtol=1e-13,
    )
    unit = make(family, normalisation="unit_integral", n_nu=256)
    response = np.asarray(unit(unit.nodes))[np.arange(3), np.arange(3)]
    tolerance = 1e-8 if family == "planck_taper" else 1e-12
    assert_allclose(
        (np.asarray(unit.weights) * response).sum(axis=1), 1.0, atol=tolerance
    )


def test_width_ratio_peak_scale_and_describe_are_concrete_and_hashable():
    channels = Channels.bump(CENTRES, WIDTHS)
    ratio = channels.width_ratio()
    assert isinstance(ratio, np.ndarray)
    assert_allclose(ratio, 2 * 0.65)
    described = channels.describe()
    assert isinstance(described, tuple) and hash(described)
    fields = dict(described)
    assert fields["family"] == "bump" and fields["n_ch"] == 3
    assert fields["smoothness"] == 99 and fields["normalisation"] == "unit_peak"
    assert Channels.LABEL == "extra eq: channel kernel"
    assert_allclose(np.asarray(channels.peak_scale()), 1.0)
    for j in range(3):
        assert_allclose(np.asarray(channels(CENTRES[j]))[j], 1.0)


def test_shape_and_value_validation():
    with pytest.raises(ValueError):
        Channels.bump(CENTRES, WIDTHS[:2])
    with pytest.raises(ValueError):
        Channels.bump(np.ones((3, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError):
        Channels.bump(CENTRES, WIDTHS, normalisation="unit_area")
    with pytest.raises(ValueError):
        Channels.bump(CENTRES, WIDTHS, n_nu=0)
    for bad in [-WIDTHS, 0 * WIDTHS, CENTRES, np.array([np.nan, 1, 1]) * WIDTHS]:
        with pytest.raises(Exception):
            Channels.bump(CENTRES, bad)
    with pytest.raises(Exception):
        Channels.bump(-CENTRES, WIDTHS)
    with pytest.raises(Exception):
        Channels.bump(CENTRES, 1.5 * CENTRES)
    with pytest.raises(Exception):
        Channels.gaussian(CENTRES, 0.2 * CENTRES, support_sigma=6.0)
    with pytest.raises(ValueError):
        Channels.bump(CENTRES, WIDTHS)(jnp.asarray([1.0 + 1j]))


@pytest.mark.parametrize(
    "centres,widths",
    [
        (np.array([1e6]), np.array([1e0])),
        (np.array([1e12, 3e12]), np.array([0.999e12, 2.9e12])),
        (np.array([50e6]), np.array([49.99e6])),
    ],
)
def test_extreme_widths_stay_finite_and_normalised(centres, widths):
    for family in ("bump", "planck_taper", "raised_cosine", "tophat"):
        channels = getattr(Channels, family)(
            centres, widths, normalisation="unit_integral"
        )
        lo = float(channels.support[0, 0])
        probe = np.concatenate(
            [centres, centres + 0.5 * widths, centres + 2 * widths, [0.5 * lo]]
        )
        values = np.asarray(channels(probe))
        assert np.all(np.isfinite(values)) and np.all(values >= 0)
        assert np.all(values[:, -1] == 0)
        jac = jax.jacfwd(lambda x: channels(x).sum(axis=0))(jnp.asarray(probe))
        assert np.all(np.isfinite(np.asarray(jac)))
        assert_allclose(
            (
                np.asarray(channels.weights)
                * np.asarray(channels(channels.nodes))[
                    np.arange(centres.size), np.arange(centres.size)
                ]
            ).sum(axis=1),
            1.0,
            atol=1e-8,
        )
