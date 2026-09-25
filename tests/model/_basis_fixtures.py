"""Shared configurations of the ``syncmoments.model.basis`` tests (not collected).

The polynomial oracle obeys ``eq: angular parity`` and keeps only ``r + s <= N``
so that the finite contraction is exact; the small harmonic configuration keeps
``required_m_max`` (7 for a ``2 nu_*`` channel) below ``m_max = 10``.
"""

import itertools
import math

import numpy as np

from syncmoments.model.channels import Channels
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.kernels import PolynomialTestKernel
from syncmoments.model.moments import Reference, Support

from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH


def polynomial_setup(seed=5, n_ch=2, L=2, N=2, s_gamma=2.0, s_B=0.2):
    """Random polynomial kernel with ``r + s > N`` coefficients zeroed (exact contraction)."""
    rng = np.random.default_rng(seed)
    coeff = rng.standard_normal((3, n_ch, L + 1, L + 1, N + 1, N + 1))
    for r, s in itertools.product(range(N + 1), repeat=2):
        if r + s > N:
            coeff[..., r, s] = 0.0
    # The oracle obeys eq: angular parity (I, Q on l+k even, V on l+k odd), as the
    # physical kernels do; the basis masks the forbidden rows by construction.
    for l, k in itertools.product(range(L + 1), repeat=2):
        if (l + k) % 2:
            coeff[0, :, l, k] = 0.0
            coeff[1, :, l, k] = 0.0
        else:
            coeff[2, :, l, k] = 0.0
    line_nu = np.array([1.5e8, 4.0e8, 9.0e8])[:n_ch]
    kernel = PolynomialTestKernel(
        coeff, line_nu, gamma0=20.0, B0=2.0, s_gamma=s_gamma, s_B=s_B
    )
    reference = Reference(20.0, 2.0, depth_ref=1.5, scales=(s_gamma, s_B, 0.5))
    channels = Channels.bump(line_nu, 0.2 * line_nu)
    support = Support(gamma=(10.0, 30.0), B=(1.0, 3.0), depth=(0.0, 3.0))
    return kernel, coeff, line_nu, reference, channels, support


def taylor_weights(tau, depth_ref, s_depth, degree):
    return np.exp(1j * tau * depth_ref)[None] * np.stack(
        [(1j * tau * s_depth) ** b / math.factorial(b) for b in range(degree + 1)]
    )


# Narrow support so that required_m_max (7 for a 2 nu_* channel) stays below m_max = 10.
SMALL_SUPPORT = Support(gamma=(19.5, 20.5), B=(0.98, 1.02), depth=(0.0, 5.0 * S_DEPTH))


def small_channels(width=0.65):
    return Channels.bump([2.0 * NU_STAR], [width * 2.0 * NU_STAR])


def small_harmonic():
    kernel = HarmonicKernel(10, n_outer=16, n_inner=16, n_mu=24, n_eta=24)
    reference = Reference(
        GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    return kernel, small_channels(), reference, SMALL_SUPPORT


def five_point(values, step, r, s):
    """Manuscript-style five-point stencil derivative ``d^r_0 d^s_1`` from a 5x5 grid."""
    stencils = {
        0: np.array([0, 0, 1, 0, 0.0]),
        1: np.array([1, -8, 0, 8, -1.0]) / (12 * step),
        2: np.array([-1, 16, -30, 16, -1.0]) / (12 * step**2),
    }
    out = 0.0
    for i, j in itertools.product(range(-2, 3), repeat=2):
        out = out + stencils[r][i + 2] * stencils[s][j + 2] * values[i, j]
    return out
