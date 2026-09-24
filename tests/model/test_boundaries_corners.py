"""Product-cell corner grid of the harmonic kernel (FINAL_DESIGN Section 11).

``gamma0 in {1.01, 1.1, 2, 5, 20, 50}`` x ``B0 in {1e-2, 1, 10, 100}`` G with
one 65 % bump channel placed relative to the gyrofrequency, so that
``required_m_max <= 300`` at every corner (harmonic cells restricted to
``m <= 300``). Per corner the angular projection ``(L_mu, L_eta) = (2, 2)`` is
evaluated directly by two methods: the product cells at two resolutions
(asserted: ``rel_err < 1e-3`` per live pair, finite, no blow-up beyond
``1e6``) and the tensor route at 96 x 96 (recorded; it degrades to
``2e-3 .. 3.5e-3`` at ``gamma >= 20``, measured). ``B0`` enters only through
``nu_B`` and the ``omega_B^2`` prefactor, so the projections at ``B0`` and
``B0'`` must satisfy ``P(B0') = (B0'/B0)^2 P(B0)`` to roundoff: the second
cross-method check along the ``B0`` axis.

Measured (this machine, n = 32 vs 64, channel centre 10 nu_B): product
agreement 1.3e-10 (gamma <= 1.1), 1e-5 .. 3.4e-5 (gamma >= 2). With the
channel at 90 nu_B (``required_m_max`` 170 .. 297, slow): n = 32 vs 64 gives
4.5e-10 (gamma 1.01), 2e-10 (gamma 2), 1.7e-4 (gamma 20, V) and 1.2e-3
(gamma 50, V pair (1, 0), a cancelling entry at 1e-3 of the V scale; it
converges to 3e-9 between 96 and 128 cells), so the slow cells use 48 vs 96.
"""

import numpy as np
import pytest

import synchro  # noqa: F401
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import Truncation
from synchro.model.kernels import required_m_max
from synchro.model.moments import Support

from _boundary_oracles import (
    B_CORNERS,
    BLOWUP,
    GAMMA_CORNERS,
    OK,
    pair_errors,
    relative_channel,
    slow,
)

TRUNCATION = Truncation(2, 2, 0)
WIDTH = 0.65
M_CAP = 300


def _setup(gamma, B, m_centre):
    channels = relative_channel(gamma, B, m_centre, WIDTH)
    support = Support(gamma=(gamma, gamma), B=(B, B), depth=(0.0, 1.0))
    m_max = required_m_max(support, channels)
    assert 1 <= m_max <= M_CAP, m_max
    return channels, m_max


def _product(gamma, B, m_centre, n):
    channels, m_max = _setup(gamma, B, m_centre)
    kernel = HarmonicKernel(m_max, n_outer=n, n_inner=n)
    return kernel.angular_projection(channels, gamma, B, truncation=TRUNCATION)


def _tensor(gamma, B, m_centre, n=96):
    channels, m_max = _setup(gamma, B, m_centre)
    kernel = HarmonicKernel(m_max, n_mu=n, n_eta=n)
    return kernel.angular_projection(
        channels, gamma, B, truncation=TRUNCATION, quadrature="tensor"
    )


def _assert_converged(coarse, fine, label):
    """Finite, no blow-up, ``rel_err < OK`` on every live pair, parity zeros agree."""
    for name in ("I", "V", "P"):
        a, b = np.asarray(getattr(coarse, name)), np.asarray(getattr(fine, name))
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (label, name)
        scale = np.max(np.abs(b))
        assert scale > 0 and np.max(np.abs(a)) <= BLOWUP * scale, (label, name)
        err, dead_ok = pair_errors(a, b)
        assert dead_ok and err < OK, (label, name, err)


def _assert_B_scaling(low, high, ratio, label):
    """``P(B') = (B'/B)^2 P(B)``: exact up to roundoff in the line positions."""
    for name in ("I", "V", "P"):
        a = np.asarray(getattr(low, name)) * ratio**2
        b = np.asarray(getattr(high, name))
        scale = np.max(np.abs(b))
        assert np.max(np.abs(a - b)) <= 1e-11 * scale, (label, name)


@pytest.mark.parametrize("gamma", GAMMA_CORNERS)
def test_product_cells_at_B_extremes(gamma, record_property):
    """``B0 in {1e-2, 100}`` at every ``gamma0`` corner: product 32 vs 64 at
    ``1e-2`` (asserted), the ``B0^2`` scaling between the two ``B0`` extremes
    (asserted) and the tensor route against the product route (recorded)."""
    low, high = min(B_CORNERS), max(B_CORNERS)
    coarse = _product(gamma, low, 10.0, 32)
    fine = _product(gamma, low, 10.0, 64)
    _assert_converged(coarse, fine, (gamma, low))
    _assert_B_scaling(coarse, _product(gamma, high, 10.0, 32), high / low, gamma)
    tensor = _tensor(gamma, low, 10.0)
    for name in ("I", "V", "P"):
        err, _ = pair_errors(getattr(tensor, name), getattr(fine, name))
        record_property(f"{name}_tensor96_vs_product64", err)
        assert np.all(np.isfinite(np.asarray(getattr(tensor, name))))


@slow
@pytest.mark.slow
@pytest.mark.parametrize("gamma", GAMMA_CORNERS)
@pytest.mark.parametrize("B", [b for b in B_CORNERS if b not in (1e-2, 100.0)])
def test_product_cells_at_interior_B(gamma, B):
    """The interior ``B0`` values: converged and on the ``B0^2`` law of ``B0 = 1e-2``."""
    coarse = _product(gamma, B, 10.0, 32)
    _assert_converged(coarse, _product(gamma, B, 10.0, 64), (gamma, B))
    _assert_B_scaling(_product(gamma, 1e-2, 10.0, 32), coarse, B / 1e-2, (gamma, B))


@slow
@pytest.mark.slow
@pytest.mark.parametrize("gamma", [1.01, 2.0, 20.0, 50.0])
def test_product_cells_up_to_m_300(gamma, record_property):
    """Channel at 90 ``nu_B``: ``required_m_max`` 170 (gamma 1.01) to 297
    (gamma 20, 50); product 48 vs 96 asserted, intensities down to 1e-69
    (gamma 1.01, where line 90 carries ``beta^180``) stay finite and positive."""
    coarse = _product(gamma, 1.0, 90.0, 48)
    fine = _product(gamma, 1.0, 90.0, 96)
    _assert_converged(coarse, fine, gamma)
    assert float(np.asarray(fine.I)[0, 0]) > 0
    _, m_max = _setup(gamma, 1.0, 90.0)
    record_property("required_m_max", m_max)
