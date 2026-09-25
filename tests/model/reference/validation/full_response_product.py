"""Product-coordinate angular quadrature for full_response.

Splits every channel-support boundary analytically in t=mu*eta. In each sign
half t=+/-v^4, and log(mu) is integrated from log(abs(t)) to zero. This resolves
the logarithmic product-coordinate Jacobian without sampling a Dirac spectrum.
The negative-mu half is explicitly included using harmonic parity.
"""

import math
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.special import jv, jvp

from . import full_response as f


def cells(q, nodes, nmax, oracle=False):
    gamma, field, depth = f.GAMMA0 * (1 + q[0]), 1 + q[1], f.ZETA0 + q[2]
    beta = np.sqrt(1 - gamma**-2)
    gx, gw = leggauss(nodes)
    for m in range(1, nmax + 1):
        scale = m * field * f.GAMMA0 / gamma
        for channel, (centre, half) in enumerate(zip(f.CENTRES, f.HALF_WIDTHS)):
            lower = max(-1.0, (1 - scale / (centre - half)) / beta)
            upper = min(1.0, (1 - scale / (centre + half)) / beta)
            if upper <= lower:
                continue
            for sign in (-1, 1):
                lo, hi = (
                    (max(0.0, lower), upper) if sign > 0 else (max(0.0, -upper), -lower)
                )
                if hi <= lo:
                    continue
                a, b = lo**0.25, hi**0.25
                v = (a + b) / 2 + (b - a) / 2 * gx
                t = sign * v**4
                outer_w = gw * (b - a) / 2 * 4 * v**3
                log_min = np.log(np.abs(t))
                log_mu = log_min[:, None] * (1 - gx[None, :]) / 2
                mu = np.exp(log_mu)
                eta = t[:, None] / mu
                inner_w = -log_min[:, None] / 2 * gw[None, :]
                d = 1 - beta * t[:, None]
                y = scale / d
                sin_a = np.sqrt(np.maximum(0.0, 1 - mu**2))
                sin_t = np.sqrt(np.maximum(0.0, 1 - eta**2))
                x = m * beta * sin_a * sin_t / d
                parallel = (eta - beta * mu) / sin_t * jv(m, x)
                jp = jvp(m, x) if oracle else (jv(m - 1, x) - jv(m + 1, x)) / 2
                perpendicular = beta * sin_a * jp
                pref = field**2 * m * m / (gamma * gamma * d**3)
                intensity = pref * (parallel**2 + perpendicular**2)
                circular = 2 * pref * parallel * perpendicular
                qnat = pref * (parallel**2 - perpendicular**2)
                u = (y - centre) / half
                response = np.exp(1 - 1 / (1 - u * u))
                measure = outer_w[:, None] * inner_w * response
                phase = np.exp(1j * depth / y**2)
                yield channel, mu, eta, measure, intensity, circular, qnat, phase, y


def projected(q, nodes, nmax):
    out = np.zeros((3, 3, 3, f.LMAX + 1, f.LMAX + 1), complex)
    ell = np.arange(f.LMAX + 1)
    parity = (-1.0) ** (ell[:, None] + ell[None, :])
    norm = (2 * ell[:, None] + 1) * (2 * ell[None, :] + 1) / 4
    for ch, mu, eta, w, i, v, qnat, phase, y in cells(q, nodes, nmax):
        pm, pe = legvander(mu, f.LMAX), legvander(eta, f.LMAX)

        def integrate(a):
            return np.einsum("ij,ijl,ijk->lk", w * a, pm, pe, optimize=True)

        out[0, ch, 0] += integrate(i) * (1 + parity) * norm
        out[1, ch, 0] += integrate(v) * (1 - parity) * norm
        for b in range(3):
            out[2, ch, b] += (
                integrate(qnat * phase * (1j / y**2) ** b) * (1 + parity) * norm
            )
    return out


def coefficients(nodes, step, nmax):
    values = {}
    for i in range(-2, 3):
        for j in range(-2, 3):
            values[i, j] = projected(np.array([i * step, j * step, 0.0]), nodes, nmax)
    stencils = {
        0: np.array([0, 0, 1, 0, 0.0]),
        1: np.array([1, -8, 0, 8, -1.0]) / (12 * step),
        2: np.array([-1, 16, -30, 16, -1.0]) / (12 * step**2),
    }
    out = np.zeros((len(f.POWERS), 3, 3, f.LMAX + 1, f.LMAX + 1), complex)
    for index, (r, s, b) in enumerate(f.POWERS):
        for i in range(-2, 3):
            for j in range(-2, 3):
                out[index] += (
                    stencils[r][i + 2] * stencils[s][j + 2] * values[i, j][:, :, b]
                )
        out[index] /= math.factorial(r) * math.factorial(s) * math.factorial(b)
    return out
