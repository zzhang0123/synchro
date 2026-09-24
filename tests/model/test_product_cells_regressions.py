"""Regressions R18 and R19 of the product-cell route (``_harmonic_cells``).

R18: when a channel edge meets a line ``m nu_B(gamma0, B0)`` at ``t = mu eta = 0``
(the cell's lower edge ``lo`` crosses zero), the node map ``v = t^(1/4)`` has
the derivative ``lo^(-3/4)``, and nested ``jacfwd`` through the moving nodes
amplified the Gauss rule's discretisation error by up to ``4e6`` (relative row
error of ``C``). Boundary method: compare the default ``64``-node basis with a
refined build on both sides of the coincidence (``-1e-9, +1e-9, 1e-6``) at a
low harmonic setting (``m_max = 6``), and nested ``jacfwd`` at and just below
the coincidence at a high one (``m_max = 40``).

R19: reverse-mode ``jax.grad`` through ``_build_core`` stored every ``lax.map``
chunk's residuals (357.6 GB of XLA temporaries at the benchmark size). Checked
from XLA's compiled memory analysis only (nothing is executed).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import synchro  # noqa: F401
from synchro.constants import C_CGS, E_ESU, M_E
from synchro.model.basis import _build_core, build_basis
from synchro.model.channels import Channels
from synchro.model._harmonic_cells import RHO_SWITCH, harmonic_lines, product_cells
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import Reference, Support
from synchro.model.phase import TaylorPhase

from _harmonic_oracles import B0, GAMMA0, S_DEPTH, benchmark_channels

OFFSETS = (-1e-9, 1e-9, 1e-6)
# Channel [2, 4] nu_B(3): lines m = 2 and 4 meet its edges at t = 0 when gamma0 = 3.
LOW_GAMMA_C = 3.0
# Measured after the fix: <= 1.6e-7 (64 vs 128 nodes); before: 4.2e6 at +1e-9.
ROW_TOL = 1e-5
# The reference sits 1e-7 away (d2 changes by 3.9e-9 there); the 64-node AD
# columns are within 6.1e-6 of 256-node ones at the coincidence. Before: 6.9e12.
HIGH_TOL = 1e-4


def _nu_B(gamma, B=1.0):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


def _low_matrix(gamma0, nodes):
    kernel = HarmonicKernel(6, n_outer=nodes, n_inner=nodes)
    basis = build_basis(
        kernel,
        Channels.bump([3.0 * _nu_B(3.0)], [1.0 * _nu_B(3.0)]),
        Truncation(2, 2, 2),
        Reference(gamma0, 1.0, 0.0, scales=(0.1, 0.05, 1.0)),
        support=Support(gamma=(2.8, 3.2), B=(0.9, 1.1), depth=(0.0, 1.0)),
        allow_truncated=True,
        convergence=False,
    )
    return np.asarray(basis.response_matrix())


def test_r18_low_harmonic_basis_is_stable_across_a_line_edge_coincidence(
    record_property,
):
    worst = 0.0
    for offset in OFFSETS:
        gamma0 = LOW_GAMMA_C * (1 + offset)
        a, b = _low_matrix(gamma0, 64), _low_matrix(gamma0, 128)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        rows = np.sum(np.abs(a - b), axis=1) / np.sum(np.abs(b), axis=1)
        worst = max(worst, float(np.max(rows)))
    record_property("r18_low_worst_row_rel", worst)
    assert worst < ROW_TOL, worst


def test_r18_high_harmonic_second_derivative_at_the_coincidence(record_property):
    """m_max = 40, channel [0.7, 3.3] nu_*(20): line 3 meets the upper edge at
    t = 0 when gamma = 20 * 3 / 3.3. Nested jacfwd of Re P_00 at and just below
    the coincidence against the value 1e-7 above it."""
    channels = Channels.bump([2.0 * _nu_B(20.0)], [1.3 * _nu_B(20.0)])
    index = MomentIndex.build(Truncation(0, 0, 0))
    kernel = HarmonicKernel(40)

    def f(gamma):
        modes = kernel.angular_projection(channels, gamma, 1.0, truncation=index)
        return jnp.real(modes.P[0, 0, 0])

    d1 = jax.jit(jax.jacfwd(f))
    d2 = jax.jit(jax.jacfwd(jax.jacfwd(f)))
    gamma_c = 20.0 * 3 / 3.3
    ref1, ref2 = float(d1(gamma_c * (1 + 1e-7))), float(d2(gamma_c * (1 + 1e-7)))
    worst = 0.0
    for offset in (0.0, -1e-11, -1e-9):
        g = jnp.asarray(gamma_c * (1 + offset))
        e1 = abs(float(d1(g)) - ref1) / abs(ref1)
        e2 = abs(float(d2(g)) - ref2) / abs(ref2)
        worst = max(worst, e1, e2)
    record_property("r18_high_worst_rel", worst)
    assert worst < HIGH_TOL, worst


def _cell_moment_derivatives(gamma, rho_switch, nodes):
    """Value, d/dgamma and d2/dgamma2 of one smooth cell moment of harmonic 2 in
    the channel [2, 4] nu_B(3); its t > 0 cell has lo/hi = delta/(0.5 + delta)
    at gamma = 3 (1 + delta)."""
    nu_3 = _nu_B(3.0)
    channels = Channels.bump([3.0 * nu_3], [nu_3])

    def f(g):
        mu, eta, w = product_cells(
            2.0, g, 1.0, 2.0 * nu_3, 4.0 * nu_3, nodes, nodes, rho_switch=rho_switch
        )
        I, _, _, nu = harmonic_lines(2.0, g, 1.0, mu, eta, n_nodes=128)
        R = channels(jnp.where(nu > 0, nu, 1.0))[0]
        return jnp.sum(w * R * I * (1 + mu * eta)) * 1e20

    d1 = jax.jacfwd(f)
    return np.array([float(f(gamma)), float(d1(gamma)), float(jax.jacfwd(d1)(gamma))])


# delta -> lo/hi of the cell: 2e-6, 2e-3, RHO_SWITCH (the dispatch threshold), 0.5.
SWITCH_DELTAS = (1e-6, 1e-3, 0.5 * RHO_SWITCH / (1 - RHO_SWITCH), 0.5)


@pytest.mark.parametrize("delta", SWITCH_DELTAS)
def test_r18_dispatch_boundary_both_node_motions_and_the_dispatcher(delta):
    """Boundary method at the RHO_SWITCH dispatch: v-map motion (rho_switch=0)
    and affine-in-t motion (rho_switch=inf), bypassing the dispatcher, against
    a 256-node reference. Measured at 64 nodes: affine <= 2.2e-7 everywhere;
    v-map 2.7e-6 at the threshold and 8.1e2 at lo/hi = 2e-6 (second derivative)."""
    gamma = 3.0 * (1 + delta)
    reference = _cell_moment_derivatives(gamma, np.inf, 256)
    affine = _cell_moment_derivatives(gamma, np.inf, 64)
    vmap = _cell_moment_derivatives(gamma, 0.0, 64)
    dispatched = _cell_moment_derivatives(gamma, RHO_SWITCH, 64)
    rel = lambda a: np.abs(a - reference) / np.abs(reference)  # noqa: E731
    assert np.all(np.isfinite(vmap)) and np.all(np.isfinite(affine))
    assert np.max(rel(affine)) < 1e-6
    assert np.max(rel(dispatched)) < 1e-5
    if delta >= SWITCH_DELTAS[2]:
        # At and above the threshold both motions are valid and agree.
        assert np.max(np.abs(vmap - affine) / np.abs(affine)) < 1e-5


# -- R19 -------------------------------------------------------------------------------


def _loss(kernel, index, phase):
    channels = benchmark_channels()

    def f(gamma0):
        ref = Reference(gamma0, B0, depth_ref=4 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH))
        I, V, P = _build_core(
            kernel, channels, ref, phase, index=index, quadrature=None
        )
        return (jnp.sum(I) + jnp.sum(V) + jnp.sum(jnp.real(P))) * 1e17

    return f


def test_r19_reverse_mode_memory_stays_within_the_forward_mode_budget(record_property):
    """Benchmark size (m_max = 40, three channels, 64 product nodes, N = 1)."""
    f = _loss(
        HarmonicKernel(40), MomentIndex.build(Truncation(2, 2, 1)), TaylorPhase(1)
    )
    x = jnp.asarray(GAMMA0)
    temp = {}
    for name, transform in (("jacfwd", jax.jacfwd), ("grad", jax.grad)):
        compiled = jax.jit(transform(f)).lower(x).compile()
        analysis = compiled.memory_analysis()
        if analysis is None:
            pytest.skip("XLA memory analysis not available on this backend")
        temp[name] = analysis.temp_size_in_bytes
    record_property("r19_temp_bytes", temp)
    # Measured with the checkpoint: grad 5.8 GB, jacfwd 6.7 GB; without it grad 357.6 GB.
    assert temp["grad"] <= 2 * temp["jacfwd"], temp
    assert temp["grad"] < 16e9, temp
