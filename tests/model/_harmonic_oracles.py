"""Independent NumPy/SciPy oracles shared by the harmonic-kernel tests (not collected)."""

import math
from pathlib import Path

import numpy as np
from scipy.special import jv, jvp

from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.channels import Channels
from syncmoments.model.moments import Support

# Verbatim copy of the manuscript's independent NumPy/SciPy reference
# (validation/full_response*.py and its saved results); see reference/README.md.
REFERENCE = str(Path(__file__).resolve().parent / "reference")
GAMMA0, B0 = 20.0, 1.0
NU_STAR = E_ESU * B0 / (2 * np.pi * GAMMA0 * M_E * C_CGS)
S_DEPTH = 1.0 / (2 * (C_SI_M / NU_STAR) ** 2)  # zeta = (varphi - varphi_ref)/S_DEPTH
# e^2 Omega0^2 / (2 pi c): converts the manuscript's dimensionless powers.
UNITS = E_ESU**2 * (E_ESU * B0 / (M_E * C_CGS)) ** 2 / (2 * np.pi * C_CGS)
Y_J = np.array([2.0, 4.0, 8.0])
BENCH_SUPPORT = Support(gamma=(16.0, 24.0), B=(0.8, 1.2), depth=(0.0, 10.0))


def benchmark_channels(width=0.65):
    return Channels.bump(Y_J * NU_STAR, width * Y_J * NU_STAR)


def oracle_lines(m, gamma, B, mu, eta):
    """``(I_m, Q_m, V_m, nu_m)`` with SciPy Bessel functions (erg/s/sr, Hz)."""
    m = np.asarray(m, dtype=float)
    beta = np.sqrt(1 - 1 / gamma**2)
    b_par, b_perp = beta * mu, beta * np.sqrt((1 - mu) * (1 + mu))
    st, D = np.sqrt((1 - eta) * (1 + eta)), 1 - beta * mu * eta
    x = m * b_perp * st / D
    a_par = (eta - b_par) * b_perp * (jv(m - 1, x) + jv(m + 1, x)) / (2 * D)
    a_perp = b_perp * jvp(m, x)
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2 * np.pi * C_CGS) * m**2 / D**3
    return (
        pref * (a_par**2 + a_perp**2),
        pref * (a_par**2 - a_perp**2),
        2 * pref * a_par * a_perp,
        m * wB / (2 * np.pi * D),
    )


def bump(nu, centres, widths):
    t = (nu[None, :] - centres[:, None]) / widths[:, None]
    inside = np.abs(t) < 1
    out = np.zeros_like(t)
    out[inside] = np.exp(1 - 1 / (1 - t[inside] ** 2))
    return out


def oracle_modes(channels, m_max, gamma, B, mu, eta, degree, depth_ref, s_depth):
    """Line sum ``sum_m S_m R_j(nu_m) w_b(tau_m)`` with NumPy (``I, V, P[b]``)."""
    ms = np.arange(1, m_max + 1, dtype=float)
    I, Q, V, nu = oracle_lines(ms, gamma, B, mu, eta)
    R = bump(nu, np.asarray(channels.centres_hz), np.asarray(channels.widths_hz))
    tau = 2 * (C_SI_M / nu) ** 2
    terms = [(1j * tau * s_depth) ** b / math.factorial(b) for b in range(degree + 1)]
    w = np.exp(1j * tau * depth_ref)[None] * np.stack(terms)
    return R @ I, R @ V, np.einsum("cm,bm->bc", R * Q, w)


def richardson(fn, x, i, h):
    """Central difference with one Richardson step (error ``O(h^4)``)."""
    e = np.zeros(2)
    e[i] = 1.0
    d1 = (fn(x + h * e) - fn(x - h * e)) / (2 * h)
    d2 = (fn(x + h / 2 * e) - fn(x - h / 2 * e)) / h
    return (4 * d2 - d1) / 3


def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
