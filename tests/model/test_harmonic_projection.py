"""Angular projections of ``HarmonicKernel``: the manuscript's product-coordinate
reference, tensor-vs-product convergence, parity and nested derivatives."""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel, product_cells
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.phase import TaylorPhase

from _harmonic_oracles import (
    B0,
    GAMMA0,
    REFERENCE,
    NU_STAR,
    S_DEPTH,
    UNITS,
    Y_J,
    benchmark_channels,
    oracle_lines,
    rel_err,
    richardson,
)


@pytest.fixture(scope="module")
def manuscript_product():
    sys.path.insert(0, REFERENCE)
    try:
        from validation import full_response_product
    finally:
        sys.path.pop(0)
    return full_response_product


@pytest.mark.parametrize("q", [(0.0, 0.0, 0.0), (0.05, -0.03, 0.3)])
def test_product_projection_matches_manuscript_reference(manuscript_product, q):
    nodes = 24
    reference = manuscript_product.projected(np.array(q), nodes, 40)  # (3, ch, b, l, k)
    channels = benchmark_channels()
    kernel = HarmonicKernel(40, n_outer=nodes, n_inner=nodes)
    truncation = Truncation(8, 8, 2)
    index = MomentIndex.build(truncation)
    gamma, B = GAMMA0 * (1 + q[0]), B0 * (1 + q[1])
    depth_ref = (4.0 + q[2]) * S_DEPTH
    projected = jax.jit(
        lambda g, b: kernel.angular_projection(
            channels,
            g,
            b,
            phase=TaylorPhase(2),
            depth_ref=depth_ref,
            s_depth=S_DEPTH,
            truncation=truncation,
        )
    )(gamma, B)
    assert projected.I.shape == (3, 81) and projected.P.shape == (3, 3, 81)
    scale = np.max(np.abs(reference[0])) * UNITS
    for l, k in [(0, 0), (2, 0), (1, 1), (8, 8), (3, 2), (5, 7)]:
        i = index.pairs.index((l, k))
        expected_I = reference[0, :, 0, l, k].real * UNITS
        expected_V = reference[1, :, 0, l, k].real * UNITS
        assert_allclose(
            np.asarray(projected.I[:, i]), expected_I, rtol=1e-8, atol=1e-13 * scale
        )
        assert_allclose(
            np.asarray(projected.V[:, i]), expected_V, rtol=1e-8, atol=1e-13 * scale
        )
        for b in range(3):
            expected_P = reference[2, :, b, l, k] * UNITS / math.factorial(b)
            assert_allclose(
                np.asarray(projected.P[b, :, i]),
                expected_P,
                rtol=1e-8,
                atol=1e-13 * scale,
            )
    assert np.max(np.abs(reference[0, :, 0, 8, 8])) > 0


def test_product_cells_match_reference_cell_geometry(manuscript_product):
    """The cell bounds, nodes and weights are those of the manuscript's ``cells``."""
    channels = benchmark_channels()
    support = np.asarray(channels.support)
    gamma, B = GAMMA0 * 1.05, B0 * 0.97
    q = np.array([0.05, -0.03, 0.0])
    reference = {}
    for ch, mu, eta, measure, *_ in manuscript_product.cells(q, 16, 40):
        reference.setdefault(ch, []).append((mu, eta, measure))
    cell_index = {ch: 0 for ch in range(3)}
    for m in range(1, 41):
        for j in range(3):
            mu, eta, measure = (
                np.asarray(a)
                for a in product_cells(
                    float(m), gamma, B, support[j, 0], support[j, 1], 16, 16
                )
            )
            nu = oracle_lines(m, gamma, B, mu, eta)[3]
            u = (nu - Y_J[j] * NU_STAR) / (0.65 * Y_J[j] * NU_STAR)
            R = np.where(
                np.abs(u) < 1, np.exp(1 - 1 / (1 - np.minimum(u * u, 1 - 1e-12))), 0.0
            )
            for sign in (1, 0):  # the reference yields the t < 0 cell first
                if np.all(measure[sign] == 0):
                    continue
                ref_mu, ref_eta, ref_measure = reference[j][cell_index[j]]
                cell_index[j] += 1
                assert_allclose(mu[sign], ref_mu, rtol=1e-13)
                assert_allclose(eta[sign], ref_eta, rtol=1e-13)
                assert_allclose(
                    measure[sign] * R[sign],
                    ref_measure,
                    rtol=1e-12,
                    atol=1e-13 * np.max(ref_measure),
                )
    assert all(cell_index[j] == len(reference[j]) for j in range(3))


def test_tensor_matches_product_at_wide_channel_and_degrades_at_narrow(record_property):
    gamma, B = 21.0, 1.1
    truncation = Truncation(2, 2, 0)
    wide = Channels.bump([4.0 * NU_STAR], [0.65 * 4.0 * NU_STAR])
    product = HarmonicKernel(40, n_outer=128, n_inner=128).angular_projection(
        wide, gamma, B, truncation=truncation
    )
    tensor = HarmonicKernel(
        40, n_mu=320, n_eta=320, quadrature="tensor"
    ).angular_projection(wide, gamma, B, truncation=truncation)
    for name in ("I", "V", "P"):
        a, b = np.asarray(getattr(tensor, name)), np.asarray(getattr(product, name))
        assert rel_err(a, b) < 1e-6, name
    record_property(
        "tensor320_vs_product128_width0.65",
        rel_err(np.asarray(tensor.I), np.asarray(product.I)),
    )
    # 5 % channel: the product route stays converged, the tensor route degrades (recorded).
    narrow = Channels.bump([4.0 * NU_STAR], [0.05 * 4.0 * NU_STAR])
    coarse = HarmonicKernel(40, n_outer=64, n_inner=64).angular_projection(
        narrow, gamma, B, truncation=truncation
    )
    fine = HarmonicKernel(40, n_outer=128, n_inner=128).angular_projection(
        narrow, gamma, B, truncation=truncation
    )
    assert rel_err(np.asarray(coarse.I), np.asarray(fine.I)) < 1e-6
    tensor_narrow = HarmonicKernel(
        40, n_mu=96, n_eta=96, quadrature="tensor"
    ).angular_projection(narrow, gamma, B, truncation=truncation)
    degradation = rel_err(np.asarray(tensor_narrow.I), np.asarray(fine.I))
    record_property("tensor96_vs_product128_width0.05", degradation)
    assert np.isfinite(degradation)


@pytest.mark.parametrize("quadrature", ["product", "tensor"])
def test_parity_forbidden_projections_vanish(quadrature):
    channels = benchmark_channels()
    kernel = HarmonicKernel(
        20, n_outer=24, n_inner=24, n_mu=32, n_eta=32, quadrature=quadrature
    )
    index = MomentIndex.build(Truncation(3, 3, 0), parity=False)
    projected = kernel.angular_projection(
        channels, 20.5, 1.0, phase=TaylorPhase(0), truncation=index
    )
    odd = np.array([(l + k) % 2 == 1 for l, k in index.pairs])
    I, V, P = (np.asarray(a) for a in (projected.I, projected.V, projected.P[0]))
    largest = max(np.max(np.abs(I)), np.max(np.abs(V)))
    assert np.max(np.abs(I[:, odd])) < 1e-13 * largest
    assert np.max(np.abs(P[:, odd])) < 1e-13 * largest
    assert np.max(np.abs(V[:, ~odd])) < 1e-13 * largest
    assert np.max(np.abs(V[:, odd])) > 1e-3 * largest and np.max(np.abs(I[:, ~odd])) > 0
    if quadrature == "product":
        assert np.all(I[:, odd] == 0) and np.all(V[:, ~odd] == 0)


def test_projection_derivatives_match_richardson_to_third_order():
    channels = Channels.bump([4.0 * NU_STAR], [0.65 * 4.0 * NU_STAR])
    kernel = HarmonicKernel(12, n_outer=16, n_inner=16)
    truncation = Truncation(2, 2, 1)
    phase = TaylorPhase(1)

    def f(gB):
        p = kernel.angular_projection(
            channels,
            gB[0],
            gB[1],
            phase=phase,
            depth_ref=3.0 * S_DEPTH,
            s_depth=S_DEPTH,
            truncation=truncation,
        )
        return jnp.concatenate(
            [p.I.ravel(), p.V.ravel(), p.P.real.ravel(), p.P.imag.ravel()]
        )

    x = np.array([21.0, 1.1])
    raw = [f, jax.jacfwd(f)]
    raw.append(jax.jacfwd(raw[-1]))
    raw.append(jax.jacfwd(raw[-1]))
    orders = [jax.jit(fn) for fn in raw]
    values = [np.asarray(fn(jnp.asarray(x))) for fn in orders]
    for order in (1, 2, 3):
        assert np.all(np.isfinite(values[order]))
        lower = lambda y: np.asarray(orders[order - 1](jnp.asarray(y)))  # noqa: E731
        for i in range(2):
            fd = richardson(lower, x, i, 1e-3 * x[i])
            assert rel_err(values[order][..., i], fd) < 1e-6, (order, i)
