"""Boundary sweeps of the harmonic kernel (FINAL_DESIGN Section 11).

Both sides of every dispatch are evaluated directly, never through the
dispatcher alone: Bessel resolution classes at the class edges and the
resolution guard, ``required_m_max +/- 1``, ``m_max`` extremes, parity on and
off, and the corner grid ``gamma0 x B0 x mu x eta`` of the line powers. Every
cell asserts finiteness, ``rel_err < 1e-3`` between the two methods where
both apply, no blow-up beyond ``1e6`` times the reference, and four corner
cells are pinned to mpmath values. Companion files:
``test_boundaries_quadrature.py``, ``test_boundaries_corners.py``,
``test_boundaries_continuum.py`` (including the recorded harmonic-vs-continuum
``E_phys`` check), ``test_boundaries_depth.py``, ``test_boundaries_fit.py``;
shared oracles in ``_boundary_oracles.py``.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad
from scipy.special import jv, jvp

import synchro  # noqa: F401  (enables x64)
from synchro.bessel import bessel_jn_and_prime
from synchro.model.basis import build_basis
from synchro.model.harmonic import HarmonicKernel, auto_nodes, harmonic_lines
from synchro.model.index import MomentIndex, Truncation
from synchro.model.kernels import required_m_max
from synchro.model.channels import Channels
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support

from _basis_fixtures import small_harmonic
from _boundary_oracles import (
    ANGLE_CORNERS,
    B_CORNERS,
    BLOWUP,
    GAMMA_CORNERS,
    J_PINS,
    M_MAX_VALUES,
    OK,
    gyro_hz,
    rel_err,
    relative_channel,
    scipy_channel_sum,
    scipy_lines,
)

# -- Bessel resolution classes: both sides of every class edge -------------------------

CLASS_EDGES = (16, 48, 112, 240, 496, 1008)  # 4 m + 64 = 2^k exactly


@pytest.mark.parametrize("m", [e + d for e in CLASS_EDGES for d in (0, 1)])
def test_resolution_class_edges_agree_across_classes(m):
    """At ``m = edge`` and ``edge + 1`` (the two sides of the class switch) the
    automatic count, half of it (when the guard still admits it) and twice it
    give the same ``J_m, J_m'`` as SciPy at ``x/m in {0.3, 0.9, 0.999}``."""
    count = auto_nodes(m)
    assert count == 2 ** int(np.ceil(np.log2(max(128, 4 * m + 64))))
    x = m * np.array([0.3, 0.9, 0.999])
    ref = np.stack([jv(m, x), jvp(m, x)])
    for nodes in (count // 2, count, 2 * count):
        value, prime = bessel_jn_and_prime(float(m), x, n_nodes=nodes)
        got = np.stack([np.asarray(value), np.asarray(prime)])
        guarded = m + x > nodes / 2  # the guard side: NaN, never an aliased value
        assert np.all(np.isnan(got[:, guarded]))
        assert np.all(np.isfinite(got[:, ~guarded]))
        scale = np.max(np.abs(ref), axis=0)
        err = np.abs(got - ref) / np.maximum(scale, 1e-250)
        if np.any(~guarded):
            assert np.max(err[:, ~guarded]) < 1e-9, (m, nodes, err)


def test_resolution_guard_both_sides():
    """``n + |x| <= n_nodes/2`` admits the value, one half-unit beyond gives NaN."""
    inside = bessel_jn_and_prime(60.0, 4.0, n_nodes=128)[0]
    outside = bessel_jn_and_prime(60.0, 4.5, n_nodes=128)[0]
    assert np.isfinite(float(inside)) and np.isnan(float(outside))
    assert_allclose(float(inside), jv(60, 4.0), rtol=1e-10, atol=1e-250)


@pytest.mark.parametrize("m", [4, 40, 300])
def test_harmonic_kernel_node_count_validation_both_sides(m):
    """``n_nodes = 4 m_max + 2`` is accepted, ``4 m_max + 1`` refused; below
    ``m_max = 4`` the floor of 16 nodes takes over (both sides at 16 / 15)."""
    HarmonicKernel(m, n_nodes=4 * m + 2)
    with pytest.raises(ValueError, match="n_nodes"):
        HarmonicKernel(m, n_nodes=4 * m + 1)
    assert HarmonicKernel(m).resolution() == auto_nodes(m)
    HarmonicKernel(1, n_nodes=16)
    with pytest.raises(ValueError, match="n_nodes"):
        HarmonicKernel(1, n_nodes=15)


@pytest.mark.parametrize("cell", sorted(J_PINS))
def test_pinned_bessel_corners_match_mpmath(cell):
    m, x = cell
    j_ref, jp_ref = J_PINS[cell]
    value, prime = bessel_jn_and_prime(float(m), x, n_nodes=auto_nodes(m))
    assert_allclose(float(value), j_ref, rtol=1e-11)
    assert_allclose(float(prime), jp_ref, rtol=1e-11)
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    assert_allclose(float(mp.besselj(m, x)), j_ref, rtol=1e-14)


# -- corner grid of the line powers -----------------------------------------------------

CORNERS = list(itertools.product(GAMMA_CORNERS, B_CORNERS))


@pytest.mark.parametrize("gamma,B", CORNERS)
def test_line_powers_at_corners_finite_and_match_scipy(gamma, B):
    """``m in {1, 10, 300}`` x ``mu, eta in {-1+1e-6, 0, 1-1e-6}`` at every
    ``(gamma0, B0)`` corner: finite, ``rel_err < 1e-3`` against SciPy where the
    value is representable, no blow-up beyond ``1e6`` x the reference."""
    ms = np.array(M_MAX_VALUES, dtype=float)[:, None, None]
    mu = np.array(ANGLE_CORNERS)[None, :, None]
    eta = np.array(ANGLE_CORNERS)[None, None, :]
    got = harmonic_lines(ms, gamma, B, mu, eta, n_nodes=auto_nodes(300))
    ref = scipy_lines(ms, gamma, B, mu, eta)
    for name, a, b in zip("IQVn", got, ref):
        a, b = np.asarray(a), np.asarray(b)
        assert np.all(np.isfinite(a)), (name, a)
        scale = np.max(np.abs(b))
        assert np.all(np.abs(a) <= BLOWUP * max(scale, 1e-300)), name
        # rel_err on cells whose reference is representable relative to the line scale.
        floor = 1e-200 * max(scale, 1e-300)
        cells = np.abs(b) > floor
        assert np.all(rel_err(a[cells], b[cells]) < OK), (name, rel_err(a, b))
        assert np.all(np.abs(a[~cells]) <= 1e3 * floor + 1e-300), name
    I = np.asarray(got[0])
    assert np.all(I >= 0)


@pytest.mark.parametrize("gamma,B", CORNERS)
def test_channel_sum_at_corners_matches_scipy(gamma, B):
    """``channel_modes`` (``m_max = 300``, jitted once) at the nine angle corners
    of every ``(gamma0, B0)`` corner against the SciPy line sum."""
    channels = relative_channel(gamma, B, 150.0, 0.6)
    kernel = HarmonicKernel(300)
    modes = jax.jit(lambda g, b, mu, eta: kernel.channel_modes(channels, g, b, mu, eta))
    for mu, eta in itertools.product(ANGLE_CORNERS, ANGLE_CORNERS):
        out = modes(gamma, B, mu, eta)
        I, Q, V = (
            float(np.real(np.asarray(v).ravel()[0])) for v in (out.I, out.P, out.V)
        )  # Q = Re P: the lines carry no U before the sky factor
        rI, rQ, rV = (
            float(v[0]) for v in scipy_channel_sum(channels, 300, gamma, B, mu, eta)
        )
        assert np.isfinite([I, Q, V]).all()
        scale = max(abs(rI), 1e-300)
        for a, b in ((I, rI), (Q, rQ), (V, rV)):
            assert abs(a - b) <= 1e-9 * scale, (mu, eta, a, b)
        assert abs(I) <= BLOWUP * scale


# -- m_max extremes ---------------------------------------------------------------------


@pytest.mark.parametrize("m_max", M_MAX_VALUES)
def test_m_max_extremes_match_scipy_sum(m_max):
    gamma, B = 5.0, 2.0
    channels = relative_channel(gamma, B, 0.5 * (m_max + 1), 0.9)
    kernel = HarmonicKernel(m_max)
    assert kernel.resolution() == auto_nodes(m_max)
    for mu, eta in ((0.4, 0.5), (-0.7, 0.9), (1e-3, -1e-3)):
        modes = kernel.channel_modes(channels, gamma, B, mu, eta)
        I, Q, V = scipy_channel_sum(channels, m_max, gamma, B, mu, eta)
        assert np.all(np.isfinite(np.asarray(modes.I)))
        assert_allclose(np.asarray(modes.I), I, rtol=1e-10, atol=1e-10 * abs(I[0]))
        assert_allclose(
            np.asarray(modes.P[0]).real, Q, rtol=1e-10, atol=1e-10 * abs(I[0])
        )
        assert_allclose(np.asarray(modes.V), V, rtol=1e-10, atol=1e-10 * abs(I[0]))


# -- required_m_max +/- 1 -----------------------------------------------------------------


def test_required_m_max_plus_minus_one_on_channel_modes():
    """``R = required_m_max = ceil((1 + beta) nu_hi / nu_B)`` is conservative by
    exactly one: line ``R`` never meets the channel, line ``R - 1`` does at the
    extremal angle. So ``m_max = R - 1``, ``R``, ``R + 1`` and ``R + 5`` give
    identical modes at every angle, and ``R - 2`` differs wherever line
    ``R - 1`` lies inside the channel (both sides of the sufficiency edge)."""
    support = Support(gamma=(1.2, 1.5), B=(1.0, 2.0), depth=(0.0, 1.0))
    gamma, B = float(support.gamma[1]), float(support.B[0])
    beta = np.sqrt(1 - 1 / gamma**2)
    nu_hi = 2.9 * gyro_hz(gamma, B) / (1 + beta)  # (1 + beta) nu_hi / nu_B = 2.9
    channels = Channels.bump([2 * nu_hi / 3], [nu_hi / 3])
    R = required_m_max(support, channels)
    assert R == 3
    kernels = {m: HarmonicKernel(m) for m in (R - 2, R - 1, R, R + 1, R + 5)}
    differs = 0
    for t in (-0.9, -0.64, -0.3, 0.0, 0.5):
        mu, eta = np.sqrt(abs(t)), np.sign(t) * np.sqrt(abs(t))
        at = {
            m: k.channel_modes(channels, gamma, B, mu, eta) for m, k in kernels.items()
        }
        for m in (R - 1, R + 1, R + 5):
            assert_allclose(np.asarray(at[m].I), np.asarray(at[R].I), rtol=0, atol=0)
            assert_allclose(np.asarray(at[m].P), np.asarray(at[R].P), rtol=0, atol=0)
            assert_allclose(np.asarray(at[m].V), np.asarray(at[R].V), rtol=0, atol=0)
        assert np.isfinite(float(at[R].I[0]))
        nu = np.asarray(kernels[R].line_frequencies(gamma, B, mu, eta))
        inside = float(channels.support[0, 0]) < nu[R - 2] < nu_hi
        gap = float(np.abs(at[R].I - at[R - 2].I)[0])
        if inside:
            assert gap > 1e-6 * float(at[R].I[0]), (t, gap)
            differs += 1
        else:
            assert gap == 0.0
    assert differs >= 2
    # At the extremal angle line R sits above nu_hi and line R - 1 inside.
    mu, eta = 1.0 - 1e-6, -(1.0 - 1e-6)
    nu = np.asarray(kernels[R].line_frequencies(gamma, B, mu, eta))
    assert nu[R - 1] > nu_hi > nu[R - 2] > float(channels.support[0, 0])


def test_required_m_max_plus_minus_one_through_build_basis():
    kernel, channels, reference, support = small_harmonic()
    R = required_m_max(support, channels)
    assert R == 7
    kw = dict(support=support, convergence=False)
    truncation = Truncation(0, 0, 0)
    with pytest.raises(ValueError, match="required_m_max"):
        build_basis(
            HarmonicKernel(R - 1, n_outer=16, n_inner=16),
            channels,
            truncation,
            reference,
            **kw,
        )
    below = build_basis(
        HarmonicKernel(R - 1, n_outer=16, n_inner=16),
        channels,
        truncation,
        reference,
        allow_truncated=True,
        **kw,
    )
    assert below.kernel_terms.harmonic_truncation.kind == "unbounded"
    assert any("truncated" in note for note in below.provenance.notes)
    exact = build_basis(
        HarmonicKernel(R, n_outer=16, n_inner=16), channels, truncation, reference, **kw
    )
    above = build_basis(
        HarmonicKernel(R + 1, n_outer=16, n_inner=16),
        channels,
        truncation,
        reference,
        **kw,
    )
    for basis in (exact, above):
        term = basis.kernel_terms.harmonic_truncation
        assert term.kind == "bound" and float(jnp.max(term.value)) == 0.0
    assert_allclose(np.asarray(above.I_basis), np.asarray(exact.I_basis), rtol=1e-12)
    assert np.all(np.isfinite(np.asarray(below.I_basis)))


def test_galactic_regime_refused_even_when_truncated():
    kernel, channels, reference, _ = small_harmonic()
    galactic = Support(gamma=(1e3, 1e4), B=(1e-6, 1e-5), depth=(0.0, 1.0))
    assert required_m_max(galactic, channels) > 16368
    with pytest.raises(ValueError, match="ContinuumKernel"):
        build_basis(
            kernel,
            channels,
            Truncation(0, 0, 0),
            reference,
            support=galactic,
            allow_truncated=True,
            convergence=False,
        )


# -- parity on / off ----------------------------------------------------------------------


@pytest.mark.parametrize("quadrature", ["product", "tensor"])
def test_parity_off_keeps_odd_rows_whose_bases_vanish(quadrature):
    """With ``parity=False`` the odd ``(l, k)`` ``P`` projections are retained
    and vanish (exactly on the product route, to roundoff on the tensor
    route); the even projections equal the ``parity=True`` ones."""
    kernel, channels, reference, _ = small_harmonic()
    on = MomentIndex.build(Truncation(2, 2, 0))
    off = MomentIndex.build(Truncation(2, 2, 0), parity=False)
    assert on.n2 == 5 and off.n2 == 9 and len(off.h0) == len(on.h0)
    p_on = kernel.angular_projection(
        channels, reference.gamma0, reference.B0, truncation=on, quadrature=quadrature
    )
    p_off = kernel.angular_projection(
        channels, reference.gamma0, reference.B0, truncation=off, quadrature=quadrature
    )
    scale = float(np.max(np.abs(np.asarray(p_off.P))))
    odd = [i for i, (l, k) in enumerate(off.pairs) if (l + k) % 2]
    even = [i for i, (l, k) in enumerate(off.pairs) if not (l + k) % 2]
    assert np.all(np.isfinite(np.asarray(p_off.P)))
    odd_max = float(np.max(np.abs(np.asarray(p_off.P)[..., odd])))
    assert odd_max == 0.0 if quadrature == "product" else odd_max < 1e-12 * scale
    # ``pairs`` lists all nine pairs on both indices (the V rows need the odd ones).
    assert on.pairs == off.pairs and len(even) == 5 and len(odd) == 4
    assert_allclose(np.asarray(p_off.P), np.asarray(p_on.P), rtol=1e-13)
    assert_allclose(np.asarray(p_off.I), np.asarray(p_on.I), rtol=1e-13)
    on_odd = float(np.max(np.abs(np.asarray(p_on.P)[..., odd])))
    assert on_odd == odd_max


def test_parity_off_moments_reduce_to_parity_on_vector(index_111):
    off = MomentIndex.build(Truncation(1, 1, 1), parity=False)
    rng = np.random.default_rng(3)
    n = 20
    samples = PopulationSamples(
        5 + rng.uniform(size=n),
        1 + rng.uniform(size=n),
        rng.uniform(-1, 1, n),
        rng.uniform(-1, 1, n),
        rng.uniform(0, 6.3, n),
        rng.uniform(size=n),
    )
    ref = Reference(5.5, 1.5)
    m_on = JointMoments.from_samples(samples, index_111, ref)
    m_off = JointMoments.from_samples(samples, off, ref)
    for row in index_111.h2:
        assert_allclose(
            complex(m_off.get(2, *row)), complex(m_on.get(2, *row)), rtol=1e-13
        )
    assert off.n2 == 2 * index_111.n2  # (1,1,1): 4 pairs -> 2 even + 2 odd


def test_bump_area_pinned_to_mpmath():
    """``unit_integral`` bump channels use the band area ``w * 1.2069003224378762``."""
    from _boundary_oracles import BUMP_AREA_PIN

    width = 3.0e7
    channels = Channels.bump([1.0e8], [width], normalisation="unit_integral")
    assert_allclose(
        float(channels.peak_scale()[0]), 1.0 / (width * BUMP_AREA_PIN), rtol=1e-14
    )
    # Independent references: SciPy adaptive quadrature and mpmath at 30 digits
    # (1.20690032243787617534). A 400-node Gauss-Legendre sum is not used: its
    # summation roundoff is 2.7e-14 relative (measured), above the pin tolerance.
    area, _ = quad(lambda t: np.exp(1 - 1 / (1 - t**2)), -1, 1, epsrel=1e-13)
    assert_allclose(area, BUMP_AREA_PIN, rtol=1e-12)
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    exact = mp.quad(lambda t: mp.exp(1 - 1 / (1 - t**2)), [-1, 0, 1])
    assert_allclose(float(exact), BUMP_AREA_PIN, rtol=1e-15)
