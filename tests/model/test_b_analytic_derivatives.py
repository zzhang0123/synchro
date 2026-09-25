"""Analytic assembly of the ``(r, s)`` derivative tensors of ``HarmonicKernel``.

``HarmonicKernel(derivatives="analytic")`` (default) builds the basis from
``angular_taylor``: per-point factors (line powers times response, phase
weights and measure; the Legendre tables of the moving nodes) are
differentiated by nested ``jacfwd`` with the Bessel triple replaced by its
Taylor polynomial in ``x`` (coefficients from one recurrence band per
point), and the Legendre projection is assembled by the Leibniz rule.
``derivatives="autodiff"`` is nested ``jacfwd`` of ``angular_projection``
through the Bessel quadrature (the v0.2.0 structure, on the shared-contour
rule ``bessel_jn_neighbours``; not bit-identical to v0.2.0). Both are the
same AD of the same program (node motion included), so they must agree to
roundoff; checked per Stokes block and derivative order at the benchmark
and at extreme ``(gamma0, B0)`` corners through ``N = 3``. Agreement is not
a check against the derivative of the computed values: in affine-follow
cells and at an exact channel-edge/line coincidence at ``t = 0`` (a
channel edge at an integer multiple of ``nu_B(gamma0, B0)``) the two differ
by the rule's quadrature error (``product_cells``). The corners keep their
edges off integer multiples; one named test keeps a coincidence.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model import _basis_core as core
from syncmoments.model.basis import _build_core
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import Reference
from syncmoments.model.phase import TaylorPhase

from _harmonic_oracles import B0, GAMMA0, S_DEPTH, benchmark_channels

RTOL = 1e-12  # spec: per Stokes block, relative to the block's largest column
GAMMAS = (1.01, 2.0, 20.0, 50.0)
FIELDS = (1e-2, 10.0, 100.0)
# Channel centre (in nu_B) and m_max per gamma: lines m nu_B / (1 - beta t)
# cross the channel for several harmonics and both cell signs. The bump
# support is [0.5, 1.5] x centre; no edge is an integer multiple of nu_B, so no
# line meets an edge at t = 0 (the rule is not differentiable there).
CORNER = {1.01: (3.0, 8), 2.0: (3.0, 12), 20.0: (8.3, 16), 50.0: (10.3, 30)}


def _nu_B(gamma, B):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


def _corner(gamma0, B0_, quadrature="product", centre=None):
    y, m_max = CORNER[gamma0]
    y = y if centre is None else centre
    nu_c = y * _nu_B(gamma0, B0_)
    channels = Channels.bump([nu_c], [0.5 * nu_c])
    s_depth = 1.0 / (2 * (C_SI_M / nu_c) ** 2)
    reference = Reference(
        gamma0,
        B0_,
        depth_ref=3.0 * s_depth,
        scales=(0.2 * (gamma0 - 1.0), 0.05 * B0_, s_depth),
    )
    kwargs = dict(n_outer=16, n_inner=16, n_mu=16, n_eta=16, quadrature=quadrature)
    return m_max, kwargs, channels, reference


def _bases(kernel, channels, reference, truncation):
    index = MomentIndex.build(truncation)
    phase = TaylorPhase(truncation.max_b())
    out = _build_core(kernel, channels, reference, phase, index=index, quadrature=None)
    return index, [np.asarray(a) for a in out]


def _block_errors(index, new, ref):
    """``max|new - ref| / max|ref|`` per Stokes block and derivative order ``r+s``.

    Re and Im of ``P`` are separate blocks. Grouping by order is stricter than
    the spec's per-block scale: low orders are not measured against the
    (larger or smaller) high-order columns.
    """
    errors = {}
    groups = (("I", index.h0, 0), ("V", index.h0, 1), ("P", index.h2, 2))
    for name, rows, k in groups:
        parts = (
            [(name, np.real)] if name != "P" else [("ReP", np.real), ("ImP", np.imag)]
        )
        for label, part in parts:
            a, b = part(new[k]), part(ref[k])
            for n in sorted({r + s for (_, _, r, s, _) in rows}):
                cols = [i for i, (_, _, r, s, _) in enumerate(rows) if r + s == n]
                scale = np.max(np.abs(b[:, cols]))
                if scale == 0.0:
                    assert np.max(np.abs(a[:, cols])) == 0.0
                    continue
                errors[(label, n)] = float(
                    np.max(np.abs(a[:, cols] - b[:, cols])) / scale
                )
    return errors


def _compare(m_max, kwargs, channels, reference, truncation):
    _, new = _bases(HarmonicKernel(m_max, **kwargs), channels, reference, truncation)
    index, ref = _bases(
        HarmonicKernel(m_max, derivatives="autodiff", **kwargs),
        channels,
        reference,
        truncation,
    )
    errors = _block_errors(index, new, ref)
    assert errors and max(errors.values()) < RTOL, errors
    return errors


# -- equality with the nested-jacfwd path ---------------------------------------------


# Fast: every gamma at B0 = 10 G; the other eight corners of the (gamma0, B0)
# grid run with -m slow (all 12 pass in DEV and NEW). The channel is placed at a
# fixed multiple of nu_B(gamma0, B0), so B0 mainly rescales the frequencies.
FAST_CORNERS = {(g, 10.0) for g in GAMMAS}
CORNERS = [
    (g, b) if (g, b) in FAST_CORNERS else pytest.param(g, b, marks=pytest.mark.slow)
    for g in GAMMAS
    for b in FIELDS
]


@pytest.mark.parametrize("gamma0, B0_", CORNERS)
def test_analytic_bases_equal_nested_autodiff_at_corners(gamma0, B0_):
    """Product route, ``N = 3`` (the orders ``r + s <= 3`` contain ``N = 0..2``)."""
    m_max, kwargs, channels, reference = _corner(gamma0, B0_)
    _compare(m_max, kwargs, channels, reference, Truncation(2, 2, 3))


@pytest.mark.parametrize("gamma0", [1.01, 50.0])
def test_analytic_bases_equal_nested_autodiff_on_the_tensor_route(gamma0):
    m_max, kwargs, channels, reference = _corner(gamma0, 10.0, quadrature="tensor")
    _compare(m_max, kwargs, channels, reference, Truncation(2, 2, 2))


def test_analytic_bases_equal_nested_autodiff_in_affine_cells():
    """Low-gamma coincidence (R18): a channel edge meets a line near ``t = 0``, so
    cells with ``0 < lo < RHO_SWITCH hi`` move their nodes affinely."""
    gamma0 = 3.0 * (1 + 1e-9)
    nu_b = _nu_B(3.0, 1.0)
    channels = Channels.bump([3.0 * nu_b], [nu_b])
    reference = Reference(gamma0, 1.0, 0.0, scales=(0.1, 0.05, 1.0))
    kwargs = dict(n_outer=16, n_inner=16)
    _compare(6, kwargs, channels, reference, Truncation(2, 2, 2))


def test_analytic_bases_equal_nested_autodiff_at_an_edge_line_coincidence():
    """Singular regression: support ``[5, 15] nu_B`` puts lines ``m = 5`` and
    ``m = 15`` on a channel edge at ``t = 0`` (``lo = 0`` exactly), where AD
    freezes the edge; both paths must still be the same AD."""
    m_max, kwargs, channels, reference = _corner(50.0, 10.0, centre=10.0)
    _compare(m_max, kwargs, channels, reference, Truncation(2, 2, 2))


@pytest.mark.slow
def test_analytic_bases_equal_nested_autodiff_at_the_benchmark():
    reference = Reference(
        GAMMA0, B0, depth_ref=4.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    _compare(40, {}, benchmark_channels(), reference, Truncation(8, 8, 2))


# -- the analytic path under outer differentiation -----------------------------------


def test_outer_derivatives_of_the_analytic_core_match_autodiff():
    """``jax.grad`` and ``jax.jacfwd`` in ``gamma0`` through ``_build_core`` (as
    ``test_jit_ad`` and the R19 memory test use it) agree across the two paths."""
    m_max, kwargs, channels, reference = _corner(2.0, 10.0)
    index = MomentIndex.build(Truncation(1, 1, 1))
    phase = TaylorPhase(1)

    def loss(derivatives):
        kernel = HarmonicKernel(m_max, derivatives=derivatives, **kwargs)

        def f(gamma0):
            ref = Reference(gamma0, reference.B0, reference.depth_ref, reference.scales)
            I, V, P = _build_core(
                kernel, channels, ref, phase, index=index, quadrature=None
            )
            return jnp.sum(I) + jnp.sum(V) + jnp.sum(jnp.real(P)) + jnp.sum(jnp.imag(P))

        return f

    g0 = jnp.asarray(2.0)
    ref_grad = float(jax.grad(loss("autodiff"))(g0))
    assert_allclose(float(jax.grad(loss("analytic"))(g0)), ref_grad, rtol=1e-11)
    assert_allclose(float(jax.jacfwd(loss("analytic"))(g0)), ref_grad, rtol=1e-11)


def test_angular_projection_is_unchanged_by_the_switch():
    m_max, kwargs, channels, reference = _corner(2.0, 10.0)
    index = MomentIndex.build(Truncation(2, 2, 1))
    out = []
    for derivatives in ("analytic", "autodiff"):
        kernel = HarmonicKernel(m_max, derivatives=derivatives, **kwargs)

        def f(z, kernel=kernel):
            modes = kernel.angular_projection(
                channels,
                2.0 + 0.1 * z[0],
                10.0 + 0.5 * z[1],
                phase=TaylorPhase(1),
                depth_ref=reference.depth_ref,
                s_depth=reference.scales[2],
                truncation=index,
            )
            return jnp.concatenate([modes.I.ravel(), modes.V.ravel(), modes.P.ravel()])

        z = jnp.zeros(2)
        out.append((np.asarray(f(z)), np.asarray(jax.jacfwd(jax.jacfwd(f))(z))))
    for a, b in zip(*out):
        assert_allclose(a, b, rtol=1e-11, atol=1e-13 * np.max(np.abs(b)))


# -- switch, provenance and convergence rebuilds --------------------------------------


def test_switch_validation_description_and_refined_rebuilds():
    with pytest.raises(ValueError, match="derivatives"):
        HarmonicKernel(10, derivatives="jet")
    kernel = HarmonicKernel(10, chunk_budget=1 << 18, derivatives="autodiff")
    assert ("derivatives", "autodiff") in kernel.describe()
    assert ("derivatives", "analytic") in HarmonicKernel(10).describe()
    for cross in (False, True):
        alt = core.refined(kernel, benchmark_channels(), None, 2, False, cross=cross)[0]
        assert alt.chunk_budget == 1 << 18 and alt.derivatives == "autodiff"


def test_angular_taylor_returns_plain_partial_derivatives():
    """``angular_taylor(...)[(r, s)]`` is ``d^r_{z_gamma} d^s_{z_B}`` of the
    projection at ``(gamma0 + s_gamma z_gamma, B0 + s_B z_B)``, without
    ``1/(r! s!)`` (``build_core`` divides once)."""
    m_max, kwargs, channels, reference = _corner(2.0, 10.0)
    index = MomentIndex.build(Truncation(1, 1, 2))
    kernel = HarmonicKernel(m_max, **kwargs)
    scales = (0.3, 0.7)
    common = dict(phase=TaylorPhase(1), depth_ref=0.1, s_depth=0.2, truncation=index)
    out = kernel.angular_taylor(channels, 2.0, 10.0, scales=scales, order=2, **common)

    def f(z):
        modes = kernel.angular_projection(
            channels, 2.0 + scales[0] * z[0], 10.0 + scales[1] * z[1], **common
        )
        return modes.I, modes.V, modes.P

    hess = jax.jacfwd(jax.jacfwd(f))(jnp.zeros(2))
    for k, name in enumerate(("I", "V", "P")):
        for (r, s), idx in (((2, 0), (0, 0)), ((1, 1), (0, 1)), ((0, 2), (1, 1))):
            got = np.asarray(getattr(out[(r, s)], name))
            ref = np.asarray(hess[k][..., idx[0], idx[1]])
            assert_allclose(got, ref, rtol=1e-11, atol=1e-13 * np.max(np.abs(ref)))
