"""Working-set budget of the remainder probe and of the convergence rebuild.

``_remainder_probe.derivative_envelope`` / ``margin_envelope`` with
``method="jacfwd"`` nest ``jax.jacfwd`` of ``angular_projection`` in the three
coordinates ``z_3``. Each level carries the primal and three tangents, so an
order-``q`` nesting holds ``4^q`` copies of the projection's working set
(``_harmonic_taylor.level_cost``: ``prod (1 + |D_i|)`` with ``|D_i| = 3``).
The probe evaluates the projection with
``kernel.for_tangents(tangent_divisor(kernel, q))``: ``1`` for the harmonic
kernel with ``derivatives="analytic"``, ``4^q`` for every other kernel,
including the ``"autodiff"`` harmonic kernel (measured trade-offs in the
``_remainder_probe`` module docstring). ``angular_residual`` vmaps ``channel_modes``
and ``angular_projection`` over a sample batch: the batch is capped by
``_direct.sample_batch`` (the kernel's ``samples_per_step``) and the
projection runs at ``chunk_budget // batch``. Blocking changes only the
summation order: results agree with the unblocked probe to ``1e-13``
relative per (channel, Stokes) block.

``_basis_core.refined`` rebuilds a ``ContinuumKernel`` for the convergence
estimate and keeps the caller's ``chunk_budget``.
"""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

import syncmoments  # noqa: F401  (float64)
from syncmoments.model import _basis_core as core
from syncmoments.model import _remainder_probe as probe
from syncmoments.model.basis import basis_convergence, build_basis
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import PopulationSamples, Reference, Support
from syncmoments.model.phase import GaussianScreen

RTOL = 1e-13  # blocking changes only the summation order
BUDGET = 1 << 20


def _continuum(truncation, *, chunk_budget=BUDGET, n_eta=4):
    channels = Channels.bump([1.0e9, 2.0e9], [0.3e9, 0.6e9], n_nu=8)
    reference = Reference(3000.0, 5e-6, depth_ref=30.0, scales=(300.0, 5e-7, 5.0))
    basis = SimpleNamespace(
        index=MomentIndex.build(truncation), reference=reference, channels=channels
    )
    kernel = ContinuumKernel(n_nodes_F=16, n_eta=n_eta, chunk_budget=chunk_budget)
    rng = np.random.default_rng(3)
    S = 3
    samples = PopulationSamples(
        3000.0 + 300.0 * rng.uniform(-0.8, 0.8, S),
        5e-6 * (1.0 + 0.1 * rng.uniform(-0.8, 0.8, S)),
        rng.uniform(-0.9, 0.9, S),
        rng.uniform(0.2, 0.9, S),
        np.zeros(S),
        np.full(S, 30.0),
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


@pytest.fixture
def projection_budgets(monkeypatch):
    """Record ``chunk_budget`` of every kernel whose ``angular_projection`` is traced."""
    seen = []
    for cls in (ContinuumKernel, HarmonicKernel):
        original = cls.angular_projection

        def spy(self, *args, _original=original, **kwargs):
            seen.append(self.chunk_budget)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "angular_projection", spy)
    return seen


def test_jacfwd_envelope_divides_the_budget_by_the_nested_tangents(
    projection_budgets,
):
    kernel, basis, samples = _continuum(Truncation(0, 1, 1))
    probe.derivative_envelope(
        samples, basis, kernel, phase=GaussianScreen(30.0, 5.0), segment_points=(1.0,)
    )
    order = 2  # N + 1
    assert projection_budgets
    assert set(projection_budgets) == {BUDGET // 4**order}


def test_jacfwd_margin_envelope_divides_the_budget_per_order(projection_budgets):
    truncation = Truncation(0, 1, 2, depth_degree=0, max_orders=(2, 1, None))
    kernel, basis, samples = _continuum(truncation)
    probe.margin_envelope(
        samples, basis, kernel, phase=GaussianScreen(30.0, 5.0), segment_points=(1.0,)
    )
    orders = {sum(beta) for beta in truncation.margin_rows()}
    assert set(projection_budgets) == {BUDGET // 4**q for q in orders}


def _harmonic_case(truncation, *, chunk_budget=None, derivatives="autodiff"):
    kw = {} if chunk_budget is None else {"chunk_budget": chunk_budget}
    kernel = HarmonicKernel(
        4, n_outer=6, n_inner=6, n_mu=8, n_eta=8, derivatives=derivatives, **kw
    )
    nu_B = 2.8e6 * 10.0 / 2.0  # Hz, e B / (2 pi gamma m_e c) at gamma = 2, B = 10 G
    channels = Channels.bump([3 * nu_B, 4 * nu_B], [nu_B, 1.2 * nu_B], n_nu=8)
    reference = Reference(2.0, 10.0, depth_ref=0.0, scales=(0.2, 0.5, 1.0))
    index = MomentIndex.build(truncation)
    basis = SimpleNamespace(index=index, reference=reference, channels=channels)
    samples = PopulationSamples(
        np.array([2.05, 1.9]),
        np.array([10.2, 9.7]),
        np.array([0.3, -0.5]),
        np.array([0.4, 0.8]),
        np.zeros(2),
        np.zeros(2),
    )
    return kernel, basis, samples


def test_tangent_divisor_per_kernel():
    continuum = ContinuumKernel(n_nodes_F=16, n_eta=4)
    for q in (1, 2, 3, 4):
        assert probe.tangent_divisor(continuum, q) == 4**q
        assert probe.tangent_divisor(SimpleNamespace(), q) == 4**q  # unknown kernel
        assert probe.tangent_divisor(HarmonicKernel(4), q) == 1
        autodiff = HarmonicKernel(4, derivatives="autodiff")
        assert probe.tangent_divisor(autodiff, q) == 4**q
    assert probe.tangent_divisor(continuum, 0) == 1


@pytest.mark.parametrize(
    "derivatives, method, N, divisor",
    [
        ("analytic", "jacfwd", 0, 1),
        ("analytic", "jacfwd", 1, 1),
        ("autodiff", "auto", 0, 4),
        ("autodiff", "jacfwd", 1, 16),
    ],
)
def test_harmonic_jacfwd_probe_budget_per_derivative_rule(
    projection_budgets, derivatives, method, N, divisor
):
    kernel, basis, samples = _harmonic_case(
        Truncation(1, 1, N), derivatives=derivatives
    )
    probe.derivative_envelope(
        samples, basis, kernel, phase=None, segment_points=(1.0,), method=method
    )
    assert set(projection_budgets) == {kernel.chunk_budget // divisor}


def test_harmonic_jacfwd_margin_envelope_keeps_the_full_budget(projection_budgets):
    truncation = Truncation(1, 1, 2, max_orders=(1, 2, None))
    kernel, basis, samples = _harmonic_case(truncation, derivatives="analytic")
    probe.margin_envelope(
        samples, basis, kernel, phase=None, segment_points=(1.0,), method="jacfwd"
    )
    assert set(projection_budgets) == {kernel.chunk_budget}


@pytest.mark.parametrize("derivatives", ["analytic", "autodiff"])
@pytest.mark.parametrize("budget", [1, 3000])
def test_blocked_harmonic_jacfwd_probe_equals_the_unblocked_probe(budget, derivatives):
    truncation = Truncation(1, 1, 1)
    kernel, basis, samples = _harmonic_case(
        truncation, chunk_budget=budget, derivatives=derivatives
    )
    wide, _, _ = _harmonic_case(
        truncation, chunk_budget=1 << 40, derivatives=derivatives
    )
    kw = dict(phase=None, segment_points=(0.5, 1.0), method="jacfwd")
    H, H2 = probe.derivative_envelope(samples, basis, kernel, **kw)
    H_ref, H2_ref = probe.derivative_envelope(samples, basis, wide, **kw)
    assert np.max(np.asarray(H_ref)) > 0
    assert _block_error(H, H_ref) <= RTOL
    assert _block_error(H2[:, None], H2_ref[:, None]) <= RTOL


def test_angular_residual_caps_the_sample_batch(monkeypatch, projection_budgets):
    kernel, basis, samples = _continuum(Truncation(0, 1, 1))
    per_sample = 2 * 2 * 8 * 16  # F and G, n_ch n_nu n_nodes_F
    kernel = ContinuumKernel(n_nodes_F=16, n_eta=4, chunk_budget=2 * per_sample)
    batches = []
    original = jax.lax.map

    def spy(f, xs, *args, **kwargs):
        if isinstance(xs, tuple) and len(xs) == 5:
            batches.append(kwargs.get("batch_size"))
        return original(f, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, "map", spy)
    probe.angular_residual(
        samples, basis, kernel, phase=GaussianScreen(30.0, 5.0), batch_size=64
    )
    assert batches == [2]  # chunk_budget // per_sample, below samples.size = 3
    assert set(projection_budgets) == {kernel.chunk_budget // 2}


@pytest.mark.parametrize("budget", [1, 5000])
def test_blocked_probe_equals_the_unblocked_probe(budget):
    screen = GaussianScreen(30.0, 5.0)
    kw = dict(phase=screen, segment_points=(0.5, 1.0))
    truncation = Truncation(0, 1, 1)
    kernel, basis, samples = _continuum(truncation, chunk_budget=budget)
    wide, _, _ = _continuum(truncation, chunk_budget=1 << 40)
    H, H2 = probe.derivative_envelope(samples, basis, kernel, **kw)
    H_ref, H2_ref = probe.derivative_envelope(samples, basis, wide, **kw)
    assert np.max(np.asarray(H_ref)) > 0
    assert _block_error(H, H_ref) <= RTOL
    assert _block_error(H2[:, None], H2_ref[:, None]) <= RTOL
    rho = probe.angular_residual(samples, basis, kernel, phase=screen, batch_size=64)
    rho_ref = probe.angular_residual(samples, basis, wide, phase=screen, batch_size=64)
    assert _block_error(rho[:, :, None], rho_ref[:, :, None]) <= RTOL


# -- O2: the refined convergence kernel keeps the caller's budget -----------------


@pytest.mark.parametrize("full", [False, True])
def test_refined_continuum_kernel_keeps_the_user_chunk_budget(full):
    kernel = ContinuumKernel(n_nodes_F=16, n_eta=4, chunk_budget=12345)
    channels = Channels.bump([1.0e9, 2.0e9], [0.3e9, 0.6e9], n_nu=8)
    alt, alt_channels, _, _ = core.refined(kernel, channels, None, 2, full)
    assert alt.chunk_budget == 12345
    assert ("chunk_budget", 12345) in alt.describe()
    assert alt.n_eta == 8 and alt_channels.n_nu == 16
    assert alt.n_nodes_F == (32 if full else 16)


def test_convergence_estimate_is_unchanged_by_a_user_budget():
    channels = Channels.bump([1.0e9, 2.0e9], [0.3e9, 0.6e9], n_nu=8)
    reference = Reference(3000.0, 5e-6, depth_ref=30.0, scales=(300.0, 5e-7, 5.0))
    screen = GaussianScreen(30.0, 5.0)
    truncation = Truncation(0, 1, 1, depth_degree=0)
    support = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(0.0, 60.0))
    values = []
    for budget in (BUDGET, 3000):
        kernel = ContinuumKernel(n_nodes_F=16, n_eta=4, chunk_budget=budget)
        basis = build_basis(
            kernel,
            channels,
            truncation,
            reference,
            support=support,
            phase=screen,
            convergence=False,
        )
        values.append(np.asarray(basis_convergence(basis, kernel).value))
    scale = np.max(np.abs(values[0]))
    assert scale > 0
    assert np.max(np.abs(values[1] - values[0])) <= RTOL * scale
