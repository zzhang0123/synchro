"""Boundary sweeps of the angular quadratures and the channel edges.

Product vs tensor quadrature at channel widths ``{0.05, 0.2, 0.65}`` (the
product route must stay converged between 48 and 96 outer nodes; the tensor
degradation is recorded), channel support edges through derivative order
``N + 1 = 4`` for every family on both sides of each edge, the smoothness
dispatch of ``build_basis``, derivatives through a line sitting on a support
edge and through an empty/non-empty cell transition, ``gaussian`` channels at
and ``support_sigma in {4, 6, 8}``. The continuum kernel sweeps live in
``test_boundaries_continuum.py``, the product-cell corner grid in
``test_boundaries_corners.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

import synchro  # noqa: F401
from synchro.model.basis import build_basis
from synchro.model.channels import Channels
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import Truncation
from synchro.model.moments import Reference, Support

from _basis_fixtures import small_harmonic
from _boundary_oracles import (
    BLOWUP,
    OK,
    SUPPORT_SIGMAS,
    WIDTHS,
    gyro_hz,
    pair_errors,
    scipy_lines,
)
from _harmonic_oracles import B0, GAMMA0, NU_STAR

# -- product vs tensor at three channel widths ---------------------------------------------


@pytest.mark.parametrize("width", WIDTHS)
def test_product_converged_between_48_and_96_tensor_recorded(width, record_property):
    channels = Channels.bump([2.0 * NU_STAR], [width * 2.0 * NU_STAR])
    t = Truncation(2, 2, 0)
    out = {}
    for n in (48, 96):
        kernel = HarmonicKernel(40, n_outer=n, n_inner=n, n_mu=n, n_eta=n)
        for route in ("product", "tensor"):
            out[route, n] = kernel.angular_projection(
                channels, GAMMA0, B0, truncation=t, quadrature=route
            )
    for name in ("I", "V", "P"):
        values = {k: np.asarray(getattr(v, name)) for k, v in out.items()}
        for v in values.values():
            assert np.all(np.isfinite(v))
        scale = np.max(np.abs(values["product", 96]))
        for v in values.values():
            assert np.max(np.abs(v)) <= BLOWUP * scale
        product, ok = pair_errors(values["product", 48], values["product", 96])
        assert ok and product < OK, (name, width, product)
        tensor, _ = pair_errors(values["tensor", 48], values["tensor", 96])
        cross, _ = pair_errors(values["tensor", 96], values["product", 96])
        record_property(f"{name}_product_48_vs_96", product)
        record_property(f"{name}_tensor_48_vs_96", tensor)
        record_property(f"{name}_tensor96_vs_product96", cross)
        if width == 0.65:  # the manuscript's channels: both routes agree here
            assert cross < OK, (name, cross)


# -- channel support edges through order N + 1 -----------------------------------------------

FAMILIES = ("bump", "planck_taper", "raised_cosine", "tophat", "gaussian")


def _channel(family):
    c, w = 1.0e8, 2.0e7
    if family == "gaussian":
        return Channels.gaussian([c], [w / 6.0]), 1.0e8 - w, 1.0e8 + w, w
    return getattr(Channels, family)([c], [w]), c - w, c + w, w


def _derivatives(channels, nu, order):
    f = lambda x: channels(jnp.asarray(x))[0]  # noqa: E731
    out = []
    for _ in range(order + 1):
        out.append(float(f(nu)))
        f = jax.grad(f)
    return np.array(out)


@pytest.mark.parametrize("family", FAMILIES)
def test_support_edges_through_order_four_both_sides(family):
    """Orders ``0..4`` at ``lo -/+ delta`` and ``hi -/+ delta`` (``delta`` in
    units of the half-width): outside every order is exactly zero and finite;
    inside, every order below the jump order decays towards the edge
    (``|v(delta)| <= 0.6 |v(2 delta)|`` or already below ``1e-12``) and the
    jump order tends to a nonzero limit (``v(delta)`` within 10 % of
    ``v(2 delta)``). The jump order is ``smoothness + 1`` for continuous
    families and ``0`` for the edge-jump families (``tophat``, ``gaussian``)."""
    channels, lo, hi, w = _channel(family)
    jump = 0 if channels.edge_jump() > 0 else min(channels.smoothness + 1, 5)
    for edge, sign in ((lo, +1.0), (hi, -1.0)):
        for delta in (1e-2 * w, 1e-3 * w):
            outside = _derivatives(channels, edge - sign * delta, 4)
            assert np.all(outside == 0.0)
        near = _derivatives(channels, edge + sign * 1e-3 * w, 4) * w ** np.arange(5)
        far = _derivatives(channels, edge + sign * 2e-3 * w, 4) * w ** np.arange(5)
        assert np.all(np.isfinite(near)) and np.all(np.isfinite(far))
        for n in range(min(jump, 5)):
            assert abs(near[n]) < 1e-12 or abs(near[n]) <= 0.6 * abs(far[n]), (
                family,
                n,
                near,
                far,
            )
        if jump < 5:
            assert abs(near[jump]) > 0 and abs(near[jump] - far[jump]) <= 0.1 * abs(
                near[jump]
            ), (family, near, far)
    # at the edge itself the response is exactly zero and differentiable
    assert float(channels(jnp.asarray(lo))[0]) == 0.0
    assert np.all(np.isfinite(_derivatives(channels, lo, 4)))


def test_smoothness_dispatch_of_build_basis_both_sides():
    """``smoothness >= N + 1`` builds; one below refuses unless ``allow_nonsmooth``."""
    kernel, _, reference, support = small_harmonic()
    channels = Channels.raised_cosine([2.0 * NU_STAR], [0.65 * 2.0 * NU_STAR])
    assert channels.smoothness == 1
    kw = dict(support=support, convergence=False)
    ok = build_basis(kernel, channels, Truncation(0, 0, 0), reference, **kw)
    assert not any("unbounded" in n for n in ok.provenance.notes)
    with pytest.raises(ValueError, match="smoothness"):
        build_basis(kernel, channels, Truncation(0, 0, 1), reference, **kw)
    allowed = build_basis(
        kernel, channels, Truncation(0, 0, 1), reference, allow_nonsmooth=True, **kw
    )
    assert any(
        n.startswith("basis_remainder unbounded") for n in allowed.provenance.notes
    )
    assert np.all(np.isfinite(np.asarray(allowed.I_basis)))


# -- derivatives through cell boundaries -------------------------------------------------------


def _richardson(fn, x, h):
    d1 = (fn(x + h) - fn(x - h)) / (2 * h)
    d2 = (fn(x + h / 2) - fn(x - h / 2)) / h
    return (4 * d2 - d1) / 3


def test_derivative_through_a_line_on_the_support_edge():
    """Line ``m = 5`` sits exactly at ``nu_lo`` at the evaluation point; the
    ``jacfwd`` of the channel sum in ``gamma`` and ``B`` is finite and equals
    a Richardson central difference (the ``R_j' d nu_m`` term through the edge)."""
    gamma, mu, eta = 5.0, 0.3, 0.6
    nu_lo, nu_hi = 2.0e8, 5.0e8
    beta = np.sqrt(1 - 1 / gamma**2)
    D = 1 - beta * mu * eta
    B = nu_lo * D / (5.0 * gyro_hz(gamma, 1.0))
    channels = Channels.bump([(nu_lo + nu_hi) / 2], [(nu_hi - nu_lo) / 2])
    kernel = HarmonicKernel(10)
    nu = np.asarray(kernel.line_frequencies(gamma, B, mu, eta))
    assert_allclose(nu[4], nu_lo, rtol=1e-13)

    def I_of(g, b):
        return kernel.channel_modes(channels, g, b, mu, eta).I[0]

    for i, (x0, other) in enumerate(((gamma, B), (B, gamma))):
        fn = (
            (lambda g: float(I_of(g, other)))
            if i == 0
            else (lambda b: float(I_of(other, b)))
        )
        ad = float(jax.jacfwd(I_of, argnums=i)(gamma, B))
        fd = _richardson(fn, x0, 1e-3 * x0)
        assert np.isfinite(ad)
        assert abs(ad - fd) <= 1e-6 * max(abs(fd), abs(float(I_of(gamma, B))) / x0)


def test_derivative_through_empty_cell_transition(record_property):
    """At ``gamma0`` line ``m = 3`` sits exactly at ``nu_hi`` for ``mu eta = 0``:
    its ``t > 0`` product cell is empty and opens for ``gamma > gamma0`` while
    the ``t < 0`` cell's inner endpoint crosses the resonance ``t = 0`` (the
    response vanishes to all orders there). With 32 x 32 cells the ``jacfwd``
    of the product projection equals a Richardson central difference to
    ``2e-6`` and a 200 x 200 tensor reference to ``2e-4``; the 16 x 16 slope
    kink at the transition (measured 3e-3, WARN band) is recorded."""
    gamma0, B = 3.0, 1.0
    nu_hi = 3.0 * gyro_hz(gamma0, B)
    channels = Channels.bump([0.75 * nu_hi], [0.25 * nu_hi])
    t = Truncation(0, 0, 0)
    tensor = HarmonicKernel(3, n_mu=200, n_eta=200)
    reference = float(
        jax.jacfwd(
            lambda g: tensor.angular_projection(
                channels, g, B, truncation=t, quadrature="tensor"
            ).I[0, 0]
        )(gamma0)
    )
    assert_allclose(reference, -1.6898620778e-18, rtol=1e-8)  # pinned tensor reference
    for n, tol in ((16, 1e-2), (32, 1e-3)):
        kernel = HarmonicKernel(3, n_outer=n, n_inner=n)

        def I_of(g, kernel=kernel):
            return kernel.angular_projection(channels, g, B, truncation=t).I[0, 0]

        ad = float(jax.jacfwd(I_of)(gamma0))
        fd = _richardson(lambda g: float(I_of(g)), gamma0, 2e-3 * gamma0)
        sides = [float(jax.jacfwd(I_of)(gamma0 * (1 + s))) for s in (-1e-4, 1e-4)]
        assert np.isfinite([ad, fd] + sides).all()
        kink = abs(ad - 0.5 * sum(sides)) / abs(reference)
        record_property(f"transition_kink_{n}", kink)
        record_property(f"ad_vs_tensor_{n}", abs(ad - reference) / abs(reference))
        assert abs(ad - reference) <= tol * abs(reference), (n, ad, reference)
        if n == 32:
            assert abs(ad - fd) <= 1e-5 * abs(reference), (ad, fd)
            assert kink < OK
        below, above = float(I_of(gamma0 * (1 - 1e-3))), float(
            I_of(gamma0 * (1 + 1e-3))
        )
        assert below > 0 and above > 0 and abs(above - below) < 1e-2 * below


# -- gaussian support_sigma ---------------------------------------------------------------------


@pytest.mark.parametrize("support_sigma", SUPPORT_SIGMAS)
def test_gaussian_support_sigma_edges_area_and_dropped_tail(
    support_sigma, record_property
):
    gamma, B = 5.0, 2.0
    sigma = 4.0 * gyro_hz(gamma, B)
    centre = 60.0 * gyro_hz(gamma, B)
    channels = Channels.gaussian([centre], [sigma], support_sigma=support_sigma)
    jump = np.exp(-0.5 * support_sigma**2)
    assert_allclose(channels.edge_jump(), jump, rtol=1e-14)
    lo, hi = (float(v) for v in channels.support[0])
    assert_allclose(
        (lo, hi),
        (centre - support_sigma * sigma, centre + support_sigma * sigma),
        rtol=1e-14,
    )
    delta = 1e-9 * sigma
    for edge in (lo, hi):
        assert (
            float(channels(jnp.asarray(edge - (1 if edge == lo else -1) * delta))[0])
            == 0.0
        )
        inside = float(
            channels(jnp.asarray(edge + (1 if edge == lo else -1) * delta))[0]
        )
        assert_allclose(inside, jump, rtol=1e-6)
    unit = Channels.gaussian(
        [centre], [sigma], support_sigma=support_sigma, normalisation="unit_integral"
    )
    area, _ = quad(
        lambda x: np.exp(-0.5 * ((x - centre) / sigma) ** 2),
        lo,
        hi,
        epsabs=0,
        epsrel=1e-13,
    )
    assert_allclose(float(unit.peak_scale()[0]), 1.0 / area, rtol=1e-12)
    # Dropped tail of the harmonic sum against an untruncated Gaussian oracle.
    m_max = 200  # >= required_m_max (196 at support_sigma = 8) on the support below
    modes = HarmonicKernel(m_max).channel_modes(channels, gamma, B, 0.4, 0.5)
    ms = np.arange(1, m_max + 1, dtype=float)
    I, _, _, nu = scipy_lines(ms, gamma, B, 0.4, 0.5)
    full = float(np.exp(-0.5 * ((nu - centre) / sigma) ** 2) @ I)
    rel = abs(float(modes.I[0]) - full) / full
    record_property("dropped_tail_rel", rel)
    assert np.isfinite(float(modes.I[0])) and rel <= 20 * jump + 1e-13
    # smoothness 0 < N + 1 = 1: refused by build_basis for line kernels, allowed with the flag
    kernel = HarmonicKernel(m_max, n_outer=8, n_inner=8)
    support = Support(gamma=(4.9, 5.1), B=(1.9, 2.1), depth=(0.0, 1.0))
    ref = Reference(gamma, B)
    with pytest.raises(ValueError, match="smoothness"):
        build_basis(
            kernel,
            channels,
            Truncation(0, 0, 0),
            ref,
            support=support,
            convergence=False,
        )
    basis = build_basis(
        kernel,
        channels,
        Truncation(0, 0, 0),
        ref,
        support=support,
        convergence=False,
        allow_nonsmooth=True,
    )
    assert np.isfinite(float(basis.I_basis[0, 0]))
