"""Per-variable caps save derivative work in the basis build.

``Truncation(..., max_orders=(N_gamma, N_B, N_depth))`` keeps only the rows
``r <= N_gamma``, ``s <= N_B``. The build differentiates only the ``(r, s)``
multi-indices those rows need: ``_harmonic_taylor.level_plan`` picks, for the
nested forward passes, a direction set per level (``z_gamma``, ``z_B`` or
both) whose reachable multi-indices cover the needed lower set at the least
number of forward tangents. The columns must equal the uncapped build
restricted to the retained rows (spec: ``<= 1e-13`` relative per Stokes
block). Also here: the tangent-aware chunk plan (``HarmonicKernel._plan``;
counted only on rules that differentiate the Bessel contour) and the exact
parity zeros of ``angular_taylor``.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model._harmonic_taylor import (
    lower_closure,
    level_cost,
    level_plan,
    multi_indices,
    point_taylor,
    reachable,
)
from syncmoments.model.basis import _build_core
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import PolynomialTestKernel
from syncmoments.model.moments import Reference
from syncmoments.model.phase import TaylorPhase

RTOL = 1e-13  # spec: capped columns vs uncapped restricted, per Stokes block
CAPS_N2 = [(2, 1, None), (2, 0, None), (0, 2, None), (1, 1, None), (None, 0, 1)]
CAPS_N3 = [(3, 1, None), (1, 2, None), (2, 2, None), (0, 3, 1), (3, 0, None)]


def _needed(truncation):
    index = MomentIndex.build(truncation)
    return {(r, s) for (_, _, r, s, _) in index.h0 + index.h2}


# -- the level plan ------------------------------------------------------------------


@pytest.mark.parametrize("N", [0, 1, 2, 3, 4])
def test_uncapped_plan_is_the_full_nested_jacfwd(N):
    """No caps: every level takes both directions (bit-identical to v0.2.0 nesting)."""
    alphas = multi_indices(N)
    assert level_plan(alphas) == ((0, 1),) * N
    assert level_cost(level_plan(alphas)) == 3**N
    assert set(reachable(level_plan(alphas))) == set(alphas)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_capped_plans_cover_the_needed_set_at_minimal_cost(N):
    """Brute force over every nesting of length ``max |alpha|``: the plan covers
    the needed lower set, and no covering nesting has fewer forward tangents."""
    for caps in itertools.product((None, 0, 1, 2, 3), repeat=2):
        needed = _needed(Truncation(1, 1, N, max_orders=caps + (None,)))
        plan = level_plan(needed)
        assert needed <= set(reachable(plan))
        depth = max(r + s for r, s in needed)
        best = min(
            level_cost(seq)
            for seq in itertools.product(((0, 1), (0,), (1,)), repeat=depth)
            if needed <= set(reachable(seq))
        )
        assert level_cost(plan) == best, (caps, plan)


def test_benchmark_caps_cost_fewer_tangents():
    """``(8, 8, 2)``: 9 tangents uncapped, 6 with ``N_B = 1``, 4 with ``N_B = 0``."""
    costs = [
        level_cost(level_plan(_needed(Truncation(8, 8, 2, max_orders=caps))))
        for caps in (None, (2, 1, None), (2, 0, None))
    ]
    assert costs == [9, 6, 4]


def test_lower_closure_and_validation():
    assert lower_closure({(2, 1)}) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (2, 1))
    with pytest.raises(ValueError, match="multi-ind"):
        lower_closure({(-1, 0)})
    with pytest.raises(ValueError, match="multi-ind"):
        lower_closure(set())


def test_point_taylor_on_a_lower_set_matches_the_analytic_partials():
    """``f = exp(a z0 + b z1) (1 + z0^2 z1)``: partials in closed form (sympy-free)."""
    a, b = 0.7, -1.3

    def f(z):
        return jnp.exp(a * z[0] + b * z[1]) * (1.0 + z[0] ** 2 * z[1])

    def exact(r, s):
        # d^r_0 d^s_1 at 0 of e^{a z0 + b z1} + z0^2 z1 e^{a z0 + b z1}
        out = a**r * b**s
        if r >= 2 and s >= 1:
            out += r * (r - 1) * a ** (r - 2) * s * b ** (s - 1)
        return out

    for alphas in ({(3, 0), (1, 1)}, {(2, 1)}, {(0, 3)}, set(multi_indices(3))):
        got = point_taylor(f, alphas)
        closure = set(lower_closure(alphas))
        assert set(got) == closure
        for (r, s), value in got.items():
            assert_allclose(float(value), exact(r, s), rtol=1e-14, atol=1e-15)


# -- capped columns equal the uncapped build restricted to the retained rows --------


def _nu_B(gamma, B):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


def _setup(gamma0, B0_, m_max):
    nu_c = 3.0 * _nu_B(gamma0, B0_)
    channels = Channels.bump([nu_c], [0.5 * nu_c])
    s_depth = 1.0 / (2 * (C_SI_M / nu_c) ** 2)
    reference = Reference(
        gamma0,
        B0_,
        depth_ref=3.0 * s_depth,
        scales=(0.2 * (gamma0 - 1.0), 0.05 * B0_, s_depth),
    )
    return channels, reference


def _core(kernel, channels, reference, truncation):
    index = MomentIndex.build(truncation)
    phase = TaylorPhase(truncation.max_b())
    out = _build_core(kernel, channels, reference, phase, index=index, quadrature=None)
    return index, [np.asarray(a) for a in out]


_FULL = {}  # uncapped builds shared by the cap cases of one configuration


def _assert_restriction(kernel, channels, reference, N, caps, key, L=2):
    capped_index, capped = _core(
        kernel, channels, reference, Truncation(L, L, N, max_orders=caps)
    )
    if (key, N, L) not in _FULL:
        _FULL[(key, N, L)] = _core(kernel, channels, reference, Truncation(L, L, N))
    full_index, full = _FULL[(key, N, L)]
    errors = {}
    for k, rows_c, rows_f in (
        (0, capped_index.h0, full_index.h0),
        (1, capped_index.h0, full_index.h0),
        (2, capped_index.h2, full_index.h2),
    ):
        where = {row: i for i, row in enumerate(rows_f)}
        cols = [where[row] for row in rows_c]
        for part in (np.real, np.imag) if k == 2 else (np.real,):
            ref = part(full[k][:, cols])
            scale = np.max(np.abs(ref))
            diff = np.max(np.abs(part(capped[k]) - ref))
            errors[(k, part.__name__)] = diff / scale if scale > 0 else diff
    assert max(errors.values()) <= RTOL, errors
    return errors


@pytest.mark.parametrize("caps", CAPS_N2)
@pytest.mark.parametrize("derivatives", ["analytic", "autodiff"])
def test_capped_harmonic_columns_equal_uncapped_at_N2(caps, derivatives):
    channels, reference = _setup(2.0, 10.0, 12)
    kernel = HarmonicKernel(12, n_outer=16, n_inner=16, derivatives=derivatives)
    _assert_restriction(kernel, channels, reference, 2, caps, derivatives)


@pytest.mark.parametrize("caps", CAPS_N3)
@pytest.mark.parametrize("gamma0, B0_, m_max", [(1.01, 1e-2, 8), (50.0, 100.0, 30)])
def test_capped_harmonic_columns_equal_uncapped_at_extreme_corners(
    caps, gamma0, B0_, m_max
):
    """``N = 3`` at the low- and high-gamma corners of package C."""
    channels, reference = _setup(gamma0, B0_, m_max)
    kernel = HarmonicKernel(m_max, n_outer=16, n_inner=16)
    _assert_restriction(kernel, channels, reference, 3, caps, (gamma0, B0_))


@pytest.mark.parametrize("caps", [(2, 1, None), (0, 2, None)])
def test_capped_tensor_route_columns_equal_uncapped(caps):
    channels, reference = _setup(2.0, 10.0, 12)
    kernel = HarmonicKernel(12, n_mu=16, n_eta=16, quadrature="tensor")
    _assert_restriction(kernel, channels, reference, 2, caps, "tensor")


@pytest.mark.parametrize("caps", [(2, 1, None), (1, 0, None), (0, 2, 1)])
def test_capped_generic_kernel_columns_equal_uncapped(caps):
    """A kernel without ``angular_taylor`` goes through ``point_taylor`` of the
    projection; exact polynomial coefficients make the check sharp."""
    rng = np.random.default_rng(5)
    c = rng.standard_normal((3, 2, 3, 3, 4, 4))
    kernel = PolynomialTestKernel(
        c, np.array([2e8, 6e8]), gamma0=10.0, B0=3.0, s_gamma=10.0, s_B=3.0
    )
    reference = Reference(10.0, 3.0, 0.2, scales=(10.0, 3.0, 0.5))
    channels = Channels.bump([2e8, 6e8], [1e8, 3e8])
    _assert_restriction(kernel, channels, reference, 3, caps, "polynomial")


def test_angular_taylor_returns_only_the_requested_lower_set():
    channels, reference = _setup(2.0, 10.0, 12)
    kernel = HarmonicKernel(12, n_outer=16, n_inner=16)
    index = MomentIndex.build(Truncation(1, 1, 2))
    common = dict(phase=TaylorPhase(0), truncation=index)
    full = kernel.angular_taylor(
        channels, 2.0, 10.0, scales=(0.2, 0.5), order=2, **common
    )
    part = kernel.angular_taylor(
        channels, 2.0, 10.0, scales=(0.2, 0.5), order=2, orders={(2, 0)}, **common
    )
    assert set(part) == {(0, 0), (1, 0), (2, 0)}
    for alpha, modes in part.items():
        for name in ("I", "V", "P"):
            ref = np.asarray(getattr(full[alpha], name))
            got = np.asarray(getattr(modes, name))
            assert_allclose(got, ref, rtol=1e-13, atol=1e-13 * np.max(np.abs(ref)))
    with pytest.raises(ValueError, match="order"):
        kernel.angular_taylor(
            channels, 2.0, 10.0, scales=(1.0, 1.0), order=1, orders={(2, 0)}, **common
        )


# -- tangent-aware chunk plan -------------------------------------------------------


def test_taylor_plan_divides_the_budget_by_the_forward_tangents():
    """``_plan(points, tangents)``: one step holds at most ``chunk_budget``
    values counting every forward tangent; ``tangents=1`` is the plain plan."""
    kernel = HarmonicKernel(40)
    n = kernel.resolution()
    for points in (1, 100, 8192):
        assert kernel._plan(points) == kernel._plan(points, tangents=1)
        for tangents in (3, 9, 27, 64):
            harmonics, block = kernel._plan(points, tangents=tangents)
            assert harmonics * block * n * tangents <= max(kernel.chunk_budget, n)
            assert block <= kernel._plan(points)[1]


@pytest.mark.parametrize("derivatives", ["analytic", "autodiff"])
def test_derivatives_do_not_depend_on_the_blocking(derivatives):
    """A budget of 12 x n_nodes values splits the points into blocks (and, on
    the autodiff rule, the 9 tangents shrink them further): values change by
    summation order only."""
    channels, reference = _setup(2.0, 10.0, 12)
    index = MomentIndex.build(Truncation(2, 2, 2))
    common = dict(phase=TaylorPhase(1), truncation=index, depth_ref=0.3, s_depth=0.2)
    n_nodes = HarmonicKernel(12).resolution()
    out = []
    for budget in (1 << 30, 12 * n_nodes):
        kernel = HarmonicKernel(
            12, n_outer=16, n_inner=16, chunk_budget=budget, derivatives=derivatives
        )
        out.append(
            kernel.angular_taylor(
                channels, 2.0, 10.0, scales=(0.2, 0.5), order=2, **common
            )
        )
    for alpha in out[0]:
        for name in ("I", "V", "P"):
            ref = np.asarray(getattr(out[0][alpha], name))
            got = np.asarray(getattr(out[1][alpha], name))
            assert_allclose(got, ref, rtol=1e-12, atol=1e-13 * np.max(np.abs(ref)))


def test_parity_forbidden_pairs_stay_exactly_zero():
    """The ``mu < 0`` parity factors zero I and P for odd l + k and V for even
    l + k exactly, at every derivative order of ``angular_taylor``."""
    channels, reference = _setup(2.0, 10.0, 12)
    kernel = HarmonicKernel(12, n_outer=16, n_inner=16)
    index = MomentIndex.build(Truncation(3, 3, 2))
    out = kernel.angular_taylor(
        channels,
        2.0,
        10.0,
        scales=(0.2, 0.5),
        order=2,
        phase=TaylorPhase(0),
        truncation=index,
    )
    odd = np.array([(l + k) % 2 == 1 for l, k in index.pairs])
    for modes in out.values():
        assert np.all(np.asarray(modes.I)[:, odd] == 0.0)
        assert np.all(np.asarray(modes.P)[..., odd] == 0.0)
        assert np.all(np.asarray(modes.V)[:, ~odd] == 0.0)
    jax.clear_caches()
