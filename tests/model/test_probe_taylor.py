"""Remainder probe through ``angular_taylor`` and the sample-chunked tail probe.

``_remainder_probe.derivative_envelope`` / ``margin_envelope`` take the
order-``N+1`` (margin: ``|beta|``) derivatives of the projected kernels in
``z_3 = (z_gamma, z_B, z_depth)``. With ``method="taylor"`` (the default for
kernels that provide ``angular_taylor``) the ``(z_gamma, z_B)`` partials come
from ``angular_taylor`` at each probe point, one pass per depth order ``b``
with the weight ``d^b_{z_depth} exp(i tau (depth + s z)) = exp(i tau depth)
(i tau s)^b``. The symmetric tensor's
Frobenius norm is ``sqrt(sum_beta |beta|!/beta! |d^beta f|^2)``. Spec: equal
to the nested-``jacfwd`` probe (``method="jacfwd"``) to ``1e-12`` relative.

``_harmonic_tail.probe_estimate`` evaluates the probe harmonics in sample
blocks within ``chunk_budget`` (summation order only).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model import _harmonic_tail as tail
from syncmoments.model import _remainder_probe as probe
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import PopulationSamples, Reference
from syncmoments.model.phase import GaussianScreen

RTOL = 1e-12  # spec: new probe equals the nested-jacfwd probe


def _nu_B(gamma, B):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


def _setup(truncation, gamma0=2.0, B0=10.0, m_max=8):
    nu_c = 3.0 * _nu_B(gamma0, B0)
    channels = Channels.bump([nu_c, 1.3 * nu_c], [0.5 * nu_c, 0.6 * nu_c])
    s_depth = 1.0 / (2 * (C_SI_M / nu_c) ** 2)
    reference = Reference(
        gamma0,
        B0,
        depth_ref=3.0 * s_depth,
        scales=(0.2 * (gamma0 - 1.0), 0.05 * B0, s_depth),
    )
    basis = SimpleNamespace(
        index=MomentIndex.build(truncation), reference=reference, channels=channels
    )
    kernel = HarmonicKernel(m_max, n_outer=8, n_inner=8, n_mu=12, n_eta=12)
    rng = np.random.default_rng(11)
    S = 2
    s_gamma, s_B, _ = reference.scales
    samples = PopulationSamples(
        gamma0 + s_gamma * rng.uniform(-0.8, 0.8, S),
        B0 + s_B * rng.uniform(-0.8, 0.8, S),
        rng.uniform(-0.9, 0.9, S),
        rng.uniform(-0.9, 0.9, S),
        np.zeros(S),
        reference.depth_ref + s_depth * rng.uniform(-1, 1, S),
        weights=rng.uniform(0.5, 2.0, S),
    )
    return kernel, basis, samples


def _block_error(new, old):
    """``max |new - old| / max |old|`` per (channel, Stokes) block (axes 0, 1)."""
    new, old = np.asarray(new), np.asarray(old)
    worst = 0.0
    for c in range(old.shape[0]):
        for x in range(old.shape[1]):
            scale = np.max(np.abs(old[c, x]))
            diff = np.max(np.abs(new[c, x] - old[c, x]))
            worst = max(worst, diff / scale if scale > 0 else diff)
    return worst


PHASES = {"taylor": lambda: None, "screen": lambda: GaussianScreen(30.0, 5.0)}


def _cases(truncations, fast):
    """``(truncation, phase)`` for every phase; cases not in ``fast`` run with
    -m slow (the nested-``jacfwd`` reference in ``z_3`` compiles slowly; all
    pass in DEV and NEW)."""
    out = []
    for name, truncation in truncations.items():
        for phase in sorted(PHASES):
            marks = () if (name, phase) in fast else (pytest.mark.slow,)
            out.append(
                pytest.param(truncation, phase, marks=marks, id=f"{name}-{phase}")
            )
    return out


ENVELOPE_CASES = _cases(
    {
        "N1": Truncation(2, 2, 1),
        "N2": Truncation(2, 2, 2),
        "depth": Truncation(1, 2, 2, depth_degree=1),
    },
    fast={("N1", "taylor"), ("N1", "screen"), ("N2", "taylor")},
)
MARGIN_CASES = _cases(
    {
        "g2B1": Truncation(2, 2, 2, max_orders=(2, 1, None)),
        "B0": Truncation(2, 2, 2, max_orders=(None, 0, None)),
        "g1B1d1": Truncation(2, 1, 2, max_orders=(1, 1, 1)),
        "depth": Truncation(1, 2, 2, depth_degree=1, max_orders=(1, 1, None)),
    },
    fast={("g2B1", "taylor"), ("depth", "screen")},
)


@pytest.mark.parametrize("truncation, phase", ENVELOPE_CASES)
def test_taylor_envelope_equals_the_nested_jacfwd_envelope(truncation, phase):
    kernel, basis, samples = _setup(truncation)
    kw = dict(phase=PHASES[phase](), segment_points=(0.5, 1.0))
    H_new, H2_new = probe.derivative_envelope(
        samples, basis, kernel, method="taylor", **kw
    )
    H_old, H2_old = probe.derivative_envelope(
        samples, basis, kernel, method="jacfwd", **kw
    )
    assert np.max(np.asarray(H_old)) > 0
    assert _block_error(H_new, H_old) <= RTOL
    assert _block_error(H2_new[:, None], H2_old[:, None]) <= RTOL


@pytest.mark.slow
@pytest.mark.parametrize("gamma0, B0, m_max", [(1.01, 1e-2, 8), (50.0, 100.0, 30)])
def test_taylor_envelope_equals_nested_jacfwd_at_extreme_corners(gamma0, B0, m_max):
    kernel, basis, samples = _setup(Truncation(2, 2, 2), gamma0, B0, m_max)
    kw = dict(phase=None, segment_points=(1.0,))
    new = probe.derivative_envelope(samples, basis, kernel, method="taylor", **kw)
    old = probe.derivative_envelope(samples, basis, kernel, method="jacfwd", **kw)
    assert _block_error(new[0], old[0]) <= RTOL


@pytest.mark.parametrize("truncation, phase", MARGIN_CASES)
def test_taylor_margin_envelope_equals_the_nested_jacfwd_one(truncation, phase):
    kernel, basis, samples = _setup(truncation)
    kw = dict(phase=PHASES[phase](), segment_points=(0.5, 1.0))
    new = probe.margin_envelope(samples, basis, kernel, method="taylor", **kw)
    old = probe.margin_envelope(samples, basis, kernel, method="jacfwd", **kw)
    assert (
        new.shape
        == old.shape
        == (2, 3, basis.index.n_lk, len(truncation.margin_rows()))
    )
    assert _block_error(new, old) <= RTOL


def test_method_resolution_and_validation():
    kernel, basis, samples = _setup(Truncation(1, 1, 1))
    assert probe.probe_method(kernel, "auto") == "taylor"
    assert probe.probe_method(HarmonicKernel(4, derivatives="autodiff"), "auto") == (
        "jacfwd"
    )
    assert probe.probe_method(object(), "auto") == "jacfwd"
    with pytest.raises(ValueError, match="method"):
        probe.probe_method(kernel, "jet")
    with pytest.raises(ValueError, match="angular_taylor"):
        probe.probe_method(object(), "taylor")


def test_symmetric_frobenius_norm_counts_every_index_tuple():
    """``sum_beta |beta|!/beta! |T_beta|^2`` equals the sum over all index tuples."""
    rng = np.random.default_rng(3)
    for dims, order in ((2, 3), (3, 3), (3, 2), (2, 1)):
        values = {}
        full = np.zeros((dims,) * order, dtype=complex)
        for idx in np.ndindex(*(dims,) * order):
            beta = tuple(idx.count(i) for i in range(dims))
            if beta not in values:
                values[beta] = rng.standard_normal() + 1j * rng.standard_normal()
            full[idx] = values[beta]
        expected = np.sqrt(np.sum(np.abs(full) ** 2))
        got = probe.symmetric_frobenius(
            {beta: jnp.asarray(v) for beta, v in values.items()}, order, dims
        )
        np.testing.assert_allclose(float(got), expected, rtol=1e-14)


# -- tail probe in sample blocks ------------------------------------------------------


def _tail(budget, samples, reference, channels):
    kernel = HarmonicKernel(
        6, tail_probe=10, n_mu=6, n_eta=5, chunk_budget=budget, m_chunk=4
    )
    return (
        np.asarray(tail.probe_estimate(kernel, channels, samples, reference).value),
        kernel,
    )


def test_tail_probe_blocks_over_samples_change_only_the_summation_order():
    _, basis, _ = _setup(Truncation(1, 1, 1))
    reference, channels = basis.reference, basis.channels
    rng = np.random.default_rng(2)
    S = 37
    samples = PopulationSamples(
        2.0 + 0.3 * rng.uniform(-1, 1, S),
        10.0 * (1 + 0.1 * rng.uniform(-1, 1, S)),
        rng.uniform(-1, 1, S),
        rng.uniform(-1, 1, S),
        np.zeros(S),
        np.zeros(S),
        weights=rng.uniform(0.1, 3.0, S),
    )
    for where in (samples, None):  # weighted average and grid maximum
        ref, kernel = _tail(1 << 30, where, reference, channels)
        n_nodes = tail.tail_nodes(kernel)
        size = S if where is not None else kernel.n_mu * kernel.n_eta
        for budget in (n_nodes, 3 * n_nodes + 1, 7 * n_nodes):
            got, kernel = _tail(budget, where, reference, channels)
            harmonics, block = tail.sample_plan(kernel, size)
            assert harmonics * block * n_nodes <= max(budget, n_nodes)
            assert block < size
            # Relative to the channel's I column (|Q_m| <= I_m): the Q entries
            # come from cancelling Bessel differences, so their own relative
            # roundoff under another vmap width can exceed 1e-13 (NEW: 1.2e-13).
            assert np.max(np.abs(got - ref) / ref[:, :1]) <= 1e-13
        assert np.max(ref) > 0


def test_tail_probe_stays_differentiable_in_traced_samples():
    _, basis, _ = _setup(Truncation(1, 1, 1))
    reference, channels = basis.reference, basis.channels
    kernel = HarmonicKernel(6, tail_probe=10, chunk_budget=3000)
    S = 5
    base = np.linspace(-1, 1, S)

    def f(gamma):
        samples = PopulationSamples(
            gamma, 10.0 + 0.2 * base, 0.5 * base, -0.4 * base, 0 * base, 0 * base
        )
        return jnp.sum(tail.probe_estimate(kernel, channels, samples, reference).value)

    gamma = jnp.asarray(2.0 + 0.2 * base)
    grad = np.asarray(jax.jit(jax.grad(f))(gamma))
    h = 1e-6
    for n in range(S):
        e = np.zeros(S)
        e[n] = h
        fd = (float(f(gamma + e)) - float(f(gamma - e))) / (2 * h)
        np.testing.assert_allclose(grad[n], fd, rtol=1e-5, atol=1e-8 * abs(fd) + 1e-300)
