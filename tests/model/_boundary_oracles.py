"""Shared oracles, grids and pins of the boundary sweeps (not collected).

Every reference here is NumPy/SciPy/mpmath, independent of the JAX path.
The mpmath pins were computed at 30 digits with the script recorded in the
comments; they are literals so that a regression shows as numeric drift even
when mpmath is not importable. ``slow`` marks the heaviest sweeps; the shared
``tests/conftest.py`` skips them unless ``-m slow`` or ``SYNCHRO_RUN_SLOW=1``
selects them.
"""

import math

import numpy as np
import pytest
from scipy.special import jv, jvp

from synchro.constants import C_CGS, C_SI_M, E_ESU, M_E
from synchro.model.channels import Channels
from synchro.model.kernels import PolynomialTestKernel
from synchro.model.moments import PopulationSamples, Reference, Support

slow = pytest.mark.slow  # gated by tests/conftest.py

# Verdict thresholds of the boundary-validation rule.
OK, WARN, BLOWUP = 1e-3, 1e-1, 1e6

# Corner grids of FINAL_DESIGN Section 11.
GAMMA_CORNERS = (1.01, 1.1, 2.0, 5.0, 20.0, 50.0)
B_CORNERS = (1e-2, 1.0, 10.0, 100.0)
ANGLE_CORNERS = (-1.0 + 1e-6, 0.0, 1.0 - 1e-6)
DEPTH_REFS = (0.0, 1e2, 1e4)
TAU_DELTA = (1e-3, 1.0, 10.0, 100.0)
DEPTH_DEGREES = (0, 1, 2, 4)
N_VALUES = (0, 1, 2, 3)
L_VALUES = (0, 4, 8)
M_MAX_VALUES = (1, 10, 300)
WIDTHS = (0.05, 0.2, 0.65)
SUPPORT_SIGMAS = (4.0, 6.0, 8.0)

# mpmath (dps=30): F(x) = x int_x^inf K_{5/3}, G(x) = x K_{2/3}(x).
F_PINS = {
    1e-6: 0.021493468615984583,
    1e-3: 0.21313906509145029,
    1.0: 0.65142281535536397,
    10.0: 0.00019223826430086897,
}
G_PINS = {
    1e-6: 0.010747641081108539,
    1e-3: 0.10746383549069977,
    1.0: 0.49447506210420827,
    10.0: 0.00018161187569530204,
}
# mpmath besselj(m, x) and its derivative at four corner cells.
J_PINS = {
    (300, 42.0): (3.4844960201597459e-219, 2.4644961977074486e-218),
    (300, 299.0): (0.057749229700424629, 0.0089504332971478199),
    (1009, 1000.0): (0.01450034767707447, 0.0022436553524633411),
    (16, 15.5): (0.14614535990147157, 0.061952124897489603),
    (1, 0.14): (0.069828640001156863, 0.4963299992247096),
}
BUMP_AREA_PIN = 1.2069003224378762  # mpmath quad of exp(1 - 1/(1 - t^2)) on [-1, 1]


def rel_err(a, b, floor=1e-300):
    """``|a - b| / max(|a|, |b|, floor)`` elementwise (the rule's definition).

    Complex inputs keep their imaginary parts (``|.|`` is the modulus)."""
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)


def gyro_hz(gamma, B):
    return E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)


def scipy_lines(m, gamma, B, mu, eta):
    """``(I, Q, V, nu)`` of harmonic ``m`` with SciPy Bessel functions (broadcast)."""
    m, gamma, B, mu, eta = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (m, gamma, B, mu, eta))
    )
    beta = np.sqrt(1 - 1 / gamma**2)
    b_par, b_perp = beta * mu, beta * np.sqrt((1 - mu) * (1 + mu))
    st, D = np.sqrt((1 - eta) * (1 + eta)), 1 - beta * mu * eta
    x = m * b_perp * st / D
    a_par = (eta - b_par) * b_perp * (jv(m - 1, x) + jv(m + 1, x)) / (2 * D)
    a_perp = b_perp * jvp(m, x)
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2 * np.pi * C_CGS) * m**2 / D**3
    I = pref * (a_par**2 + a_perp**2)
    return (
        I,
        pref * (a_par**2 - a_perp**2),
        2 * pref * a_par * a_perp,
        m * wB / (2 * np.pi * D),
    )


def bump_response(nu, centres, widths):
    t = (np.asarray(nu)[None, :] - centres[:, None]) / widths[:, None]
    out = np.zeros_like(t)
    inside = np.abs(t) < 1
    out[inside] = np.exp(1 - 1 / (1 - t[inside] ** 2))
    return out


def scipy_channel_sum(channels, m_max, gamma, B, mu, eta):
    """``sum_m S_m R_j(nu_m)`` for ``m = 1..m_max`` (``I, Q, V`` per channel)."""
    ms = np.arange(1, m_max + 1, dtype=float)
    I, Q, V, nu = scipy_lines(ms, gamma, B, mu, eta)
    R = bump_response(
        nu, np.asarray(channels.centres_hz), np.asarray(channels.widths_hz)
    )
    return R @ I, R @ Q, R @ V


def relative_channel(gamma, B, m_centre, width):
    """One bump channel centred on the harmonic ``m_centre`` of ``(gamma, B)``."""
    nu = m_centre * gyro_hz(gamma, B)
    return Channels.bump([nu], [width * nu])


# -- fixed-line polynomial kernel for the depth sweeps -------------------------------------


def constant_kernel(n_ch, c_I=1.0, c_Q=0.7, c_V=0.0, line_nu_hz=1.0e8):
    """Kernel constant in ``(gamma, B, mu, eta)``: ``K_I = c_I``, ``K_Q = c_Q``, ``K_V = c_V``."""
    coeff = np.zeros((3, n_ch, 1, 1, 1, 1))
    coeff[0, :, 0, 0, 0, 0] = c_I
    coeff[1, :, 0, 0, 0, 0] = c_Q
    coeff[2, :, 0, 0, 0, 0] = c_V
    line = np.full(n_ch, line_nu_hz)
    return PolynomialTestKernel(coeff, line, gamma0=5.0, B0=2.0), line


def depth_population(depth_ref, delta, n=41):
    """Uniform depths ``depth_ref + delta u`` on Gauss-Legendre nodes ``u`` in ``[-1, 1]``."""
    u, w = np.polynomial.legendre.leggauss(n)
    ones = np.ones(n)
    samples = PopulationSamples(
        5.0 * ones,
        2.0 * ones,
        0.3 * ones,
        -0.2 * ones,
        0.4 * ones,
        depth_ref + delta * u,
        weights=w,
    )
    absolute = {p: float(np.sum(w * np.abs(u) ** p) / np.sum(w)) for p in range(1, 7)}
    return samples, absolute


def depth_support(depth_ref, delta):
    return Support(
        gamma=(2.0, 9.0),
        B=(0.5, 4.0),
        depth=(depth_ref - delta, depth_ref + delta),
        truncated=False,
    )


def depth_reference(depth_ref, delta):
    return Reference(5.0, 2.0, depth_ref, scales=(1.0, 1.0, delta))


def taylor_remainder(x, degree, absolute_moment):
    """``|x|^{D+1}/(D+1)! <|u|^{D+1}>``: the Taylor bound of ``eq: local response remainder``."""
    return abs(x) ** (degree + 1) / math.factorial(degree + 1) * absolute_moment


def gaussian_grid(mean, sigma, n=1601, span=8.0):
    """Discrete Gaussian screen: uniform grid over ``mean +/- span sigma`` (trapezoid)."""
    x = mean + sigma * np.linspace(-span, span, n)
    w = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return x, w / w.sum()


def tau_of(nu_hz):
    return 2.0 * (C_SI_M / np.asarray(nu_hz, dtype=float)) ** 2


def pair_errors(a, b):
    """Per-pair relative error against the larger of the two, skipping pairs
    that vanish in both (parity zeros); returns the maximum."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)))
    live = np.maximum(np.abs(a), np.abs(b)) > 1e-8 * scale
    dead_ok = np.all(np.abs(a[~live] - b[~live]) <= 1e-8 * scale)
    return float(np.max(rel_err(a[live], b[live]))), bool(dead_ok)
