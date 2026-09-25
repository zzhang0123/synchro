"""Boundary sweeps of the continuum kernel and the ``E_phys`` finite check.

The ``x_min`` guard on both sides with mpmath pins of ``F`` and ``G``,
``eta -> +/-1``, and the harmonic-vs-continuum comparisons at
``gamma in {10, 20, 50}``: angular averages of both kernels on a channel at
``x ~ 0.3`` and a pitch-averaged SciPy line sum at ``gamma = 20``. The
harmonic-vs-continuum differences are recorded (``record_property``) as the
``E_phys`` finite check (FINAL_DESIGN Section 11), not a bound. Every cell
asserts finiteness and no blow-up beyond ``1e6``, and pins the signed
relative differences measured on 2026-09-24 (``E_PHYS_PINS``) to
``E_PHYS_DRIFT = 1e-3`` absolute as a drift guard: the 24-vs-48-cell
quadrature change of these numbers is at most 9e-5, while a sign flip of
``P`` or a wrong continuum normalisation moves them by order one.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.ultrarel import F, G

from _boundary_oracles import (
    BLOWUP,
    F_PINS,
    F_REFERENCE,
    G_PINS,
    G_REFERENCE,
    OK,
    gyro_hz,
    pair_errors,
    relative_channel,
    scipy_channel_sum,
    slow,
)

# Signed (h - c)/c of the E_phys cells: (I, P) per gamma for the angular averages
# (24 cells; 48 cells differ by <= 9e-5), (I, Q) for the pitch-averaged line sum.
E_PHYS_PINS = {
    10.0: (-0.018266, -0.054838),
    20.0: (-0.023888, -0.037252),
    50.0: (-0.006417, -0.018949),
    "pitch_averaged": (-0.025453, -0.024446),
}
E_PHYS_DRIFT = 1e-3


def _pin_signed(h, c, pin):
    """Same sign, and ``(h - c)/c`` within ``E_PHYS_DRIFT`` of the pinned value."""
    h, c = complex(h), complex(c)
    assert abs(h.imag) <= 1e-12 * abs(h) and abs(c.imag) <= 1e-12 * abs(c)
    assert np.sign(h.real) == np.sign(c.real) != 0
    rel = (h.real - c.real) / c.real
    assert abs(rel - pin) < E_PHYS_DRIFT, (rel, pin)
    return rel


# -- continuum x_min both sides and F/G pins -------------------------------------------------------


@pytest.mark.parametrize("x", sorted(F_PINS))
def test_continuum_F_G_pinned_to_mpmath(x):
    assert_allclose(float(F(jnp.asarray(x))), F_PINS[x], rtol=1e-9)
    assert_allclose(float(G(jnp.asarray(x))), G_PINS[x], rtol=1e-9)
    # The float64 pins against the mpmath literals (32 digits).
    assert_allclose(float(G_REFERENCE[x]), G_PINS[x], rtol=1e-13)
    assert_allclose(float(F_REFERENCE[x]), F_PINS[x], rtol=1e-13)


@pytest.mark.parametrize("x_min", [1e-6, 1e-3, 1e-1])
def test_continuum_x_min_both_sides(x_min):
    gamma, B, eta = 20.0, 1.0, 0.3
    kernel = ContinuumKernel(x_min=x_min)
    positive, amplitude, scale = kernel._geometry(
        jnp.asarray(gamma), jnp.asarray(B), jnp.asarray(eta)
    )
    # x = nu/scale is formed inside the kernel, so x == x_min itself is rounding
    # sensitive; the admitted side starts one part in 1e9 above it.
    for factor in (1.0 + 1e-9, 1.0 + 1e-6, 10.0):
        k_I, k_Q = kernel.kernels(jnp.asarray(x_min * factor * scale), gamma, B, eta)
        assert np.isfinite(float(k_I)) and float(k_I) > 0 and float(k_Q) < 0
        assert_allclose(
            float(k_I) / float(amplitude),
            float(F(jnp.asarray(x_min * factor))),
            rtol=1e-12,
        )
    for factor in (1.0 - 1e-9, 0.5):
        with pytest.raises(Exception, match="x_min"):
            jax.block_until_ready(
                kernel.kernels(jnp.asarray(x_min * factor * scale), gamma, B, eta)
            )
    # B_perp = 0 (eta = 1): below x_min is not an error and the kernel is zero.
    k_I, k_Q = kernel.kernels(jnp.asarray(0.5 * x_min * scale), gamma, B, 1.0)
    assert float(k_I) == 0.0 and float(k_Q) == 0.0


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_continuum_eta_to_plus_minus_one(sign):
    """``K(eta)`` decreases to zero as ``|eta| -> 1`` (``B_perp -> 0``), is even in
    ``eta``, finite with finite ``eta`` derivatives at every point including
    ``|eta| = 1``, and the ``eta`` projection agrees between 48 and 96 nodes."""
    gamma, B = 20.0, 1.0
    kernel = ContinuumKernel()
    nu = jnp.asarray(0.5 * 3 * gyro_hz(gamma, B) * gamma**2)  # x = 0.5 at eta = 0
    etas = [0.0, 0.5, 1 - 1e-2, 1 - 1e-4, 1 - 1e-6, 1 - 1e-9, 1.0]
    values = []
    for e in etas:
        k_I, k_Q = kernel.kernels(nu, gamma, B, sign * e)
        d = jax.jacfwd(lambda t: kernel.kernels(nu, gamma, B, t)[0])(
            jnp.asarray(sign * e)
        )
        assert np.isfinite([float(k_I), float(k_Q), float(d)]).all()
        mirror = kernel.kernels(nu, gamma, B, -sign * e)[0]
        assert float(mirror) == float(k_I)
        values.append(float(k_I))
    assert values[-1] == 0.0
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[0] > 0
    channels = Channels.bump([float(nu)], [0.3 * float(nu)])
    coarse = kernel.angular_projection(
        channels, gamma, B, truncation=Truncation(0, 2, 0)
    )
    fine = ContinuumKernel(n_eta=96).angular_projection(
        channels, gamma, B, truncation=Truncation(0, 2, 0)
    )
    err, ok = pair_errors(coarse.I, fine.I)
    assert ok and err < OK, err  # measured 2.2e-5 on a 30 % channel at x = 0.5


def test_continuum_matches_pitch_averaged_scipy_lines(record_property):
    """``E_phys`` point check at ``gamma = 20``, ``eta = 0`` on a 20 % channel at
    ``m ~ 180``: the continuum kernel is pitch-averaged (``mu`` ignored), so its
    reference is ``(1/2) int dmu sum_m S_m R_j(nu_m)`` of the SciPy harmonic
    lines (Gauss-Legendre in ``mu`` on three segments, converged to 1e-13
    between 400 and 1600 nodes per segment). Finite and no blow-up are
    asserted; the relative difference is recorded (measured -2.5 % for I and
    -2.4 % for Q), consistent with the angular-average cells of
    ``test_boundaries.py``."""
    gamma, B, eta = 20.0, 1.0, 0.0
    channels = Channels.bump([180.0 * gyro_hz(gamma, B)], [36.0 * gyro_hz(gamma, B)])
    continuum = ContinuumKernel().channel_modes(channels, gamma, B, 0.0, eta)
    cI, cQ = float(continuum.I[0]), complex(continuum.P[0, 0])
    total = np.zeros(3)
    for a, b in ((-1.0, -0.3), (-0.3, 0.3), (0.3, 1.0)):
        x, w = np.polynomial.legendre.leggauss(400)
        mu = 0.5 * (b - a) * x + 0.5 * (a + b)
        for mu_i, w_i in zip(mu, 0.5 * (b - a) * w):
            sums = scipy_channel_sum(channels, 260, gamma, B, mu_i, eta)
            total += w_i * np.array([v[0] for v in sums])
    hI, hQ = 0.5 * total[0], 0.5 * total[1]
    assert np.isfinite([cI, cQ, hI, hQ]).all() and cI > 0 and hI > 0
    assert hI < BLOWUP * cI and cI < BLOWUP * hI
    assert abs(hQ) < BLOWUP * abs(cQ) and abs(cQ) < BLOWUP * abs(hQ)
    pin_I, pin_Q = E_PHYS_PINS["pitch_averaged"]
    record_property(
        "continuum_vs_pitch_averaged_lines_I_rel", _pin_signed(hI, cI, pin_I)
    )
    record_property(
        "continuum_vs_pitch_averaged_lines_Q_rel", _pin_signed(hQ, cQ, pin_Q)
    )


# -- harmonic vs continuum: the E_phys finite check (recorded) ---------------------------


def _e_phys_cell(gamma, m_centre, m_max, n_cells):
    B = 1.0
    channels = relative_channel(gamma, B, m_centre, 0.2)
    harmonic = HarmonicKernel(m_max, n_outer=n_cells, n_inner=n_cells)
    continuum = ContinuumKernel()
    t = Truncation(0, 0, 0)
    h = harmonic.angular_projection(channels, gamma, B, truncation=t)
    c = continuum.angular_projection(channels, gamma, B, truncation=t)
    return (
        float(h.I[0, 0]),
        float(c.I[0, 0]),
        complex(h.P[0, 0, 0]),
        complex(c.P[0, 0, 0]),
    )


@pytest.mark.parametrize("gamma,m_centre,m_max", [(10.0, 45.0, 60), (20.0, 180.0, 200)])
def test_harmonic_vs_continuum_recorded(gamma, m_centre, m_max, record_property):
    """Angular averages ``(1/4) int int K`` of both kernels on a channel at
    ``x = nu/(a_B gamma^2) ~ 0.3`` (``m <= 300``): finite, no blow-up; the
    relative difference is the ``E_phys`` finite check, recorded and pinned
    (signed) against drift."""
    hI, cI, hP, cP = _e_phys_cell(gamma, m_centre, m_max, 24)
    assert np.isfinite([hI, cI, hP, cP]).all() and hI > 0 and cI > 0
    assert hI < BLOWUP * cI and cI < BLOWUP * hI
    pin_I, pin_P = E_PHYS_PINS[gamma]
    record_property("e_phys_I_rel", _pin_signed(hI, cI, pin_I))
    record_property("e_phys_P_rel", _pin_signed(hP, cP, pin_P))
    record_property("e_phys_values", (gamma, hI, cI, hP.real, cP.real))


@slow
@pytest.mark.slow
def test_harmonic_vs_continuum_gamma_50_recorded(record_property):
    hI, cI, hP, cP = _e_phys_cell(50.0, 250.0, 300, 24)
    assert np.isfinite([hI, cI, hP, cP]).all() and hI > 0 and cI > 0
    assert hI < BLOWUP * cI and cI < BLOWUP * hI
    pin_I, pin_P = E_PHYS_PINS[50.0]
    record_property("e_phys_I_rel", _pin_signed(hI, cI, pin_I))
    record_property("e_phys_P_rel", _pin_signed(hP, cP, pin_P))
