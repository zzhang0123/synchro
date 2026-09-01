"""Validation of the SED layer: spectral index, curvature, absolute emissivity.

1. power_law_emissivity_rel == numerical double integral (gamma + pitch angle).
2. spectral index alpha_s = (p-1)/2 exactly.
3. curvature Delta_alpha_s = Var(p)/4 from a Gaussian p-distribution.
4. absolute emissivity (CGS) == (sqrt3 e^3 C B/(4 pi m_e c^2)) * j_num(nu/nu_B).
"""

from __future__ import annotations

import numpy as np
from synchro.ultrarel import F
import jax.numpy as jnp
from synchro.sed import (
    spectral_index, spectral_curvature, power_law_emissivity_rel,
    power_law_emissivity_abs, E_ESU, M_E, C_CGS,
    los_intensity, spectral_index_cumulants,
)


def j_num_double(p, nu, ng=120, na=80):
    """int dgamma gamma^-p int dalpha sin^2(alpha) F(nu/(1.5 gamma^2 sin alpha))."""
    g = np.geomspace(0.03, 300.0, ng)
    a = np.linspace(0.04, np.pi - 0.04, na)
    dl = np.log(g[1] / g[0]); da = a[1] - a[0]
    tot = 0.0
    for gi in g:
        row = sum(np.sin(ai)**2 * float(F(jnp.asarray(nu / (1.5 * gi**2 * np.sin(ai))))) * da
                  for ai in a)
        tot += gi**-p * row * gi * dl
    return tot


def test_rel_emissivity():
    out = []
    for p in [2.0, 2.5, 3.0]:
        num = j_num_double(p, 1.0)
        ana = power_law_emissivity_rel(1.0, p)
        out.append((p, num, ana, num / ana))
    return out


def test_spectral_index():
    out = []
    for p in [2.2, 2.5, 3.0]:
        nus = np.logspace(0, 2, 8)
        j = np.array([power_law_emissivity_rel(nu, p) for nu in nus])
        alpha = -np.polyfit(np.log(nus), np.log(j), 1)[0]
        out.append((p, alpha, spectral_index(p)))
    return out


def test_curvature():
    pbar, sigp = 2.5, 0.2
    ps = np.linspace(pbar - 4 * sigp, pbar + 4 * sigp, 41)
    w = np.exp(-0.5 * (ps - pbar)**2 / sigp**2); w /= w.sum()
    nus = np.logspace(0, 2, 10)
    lnj = np.array([np.log(np.sum(w * np.array(
        [power_law_emissivity_rel(nu, p) for p in ps]))) for nu in nus])
    c2 = np.polyfit(np.log(nus), lnj, 2)[0]
    return 2 * c2, spectral_curvature(sigp**2)


def test_abs_emissivity(p=2.5, nu=1e9, C=1.0, B=1e-6):
    """Absolute formula == (prefactor) * relative formula(nu/nu_B), algebraically."""
    nu_B = E_ESU * B / (2 * np.pi * M_E * C_CGS)
    pref = np.sqrt(3) * E_ESU**3 * C * B / (4 * np.pi * M_E * C_CGS**2)
    rel = power_law_emissivity_rel(nu / nu_B, p)
    abs_ = power_law_emissivity_abs(nu, p, C, B)
    return pref * rel, abs_, abs_ / (pref * rel)


def test_los_cumulants():
    """LOS-varying p -> ln I(nu) Taylor coefficients = cumulants of alpha_s."""
    s = np.linspace(0, 1, 4000)
    j0 = np.ones_like(s)
    p = 2.5 + 0.3 * np.sin(2 * np.pi * s) + 0.1 * np.sin(4 * np.pi * s)
    mean, var, skew = spectral_index_cumulants(s, j0, p)
    nus = np.logspace(-1.5, 1.5, 40)
    lnI = np.array([np.log(los_intensity(nu, s, j0, p)) for nu in nus])
    c = np.polyfit(np.log(nus), lnI, 3)
    return (-mean, var / 2, -skew / 6), (c[2], c[1], c[0])


def test_running_index(p0=2.5, a=0.1):
    """Log-parabola p(gamma) -> d alpha_s / d ln nu = a/2."""
    from synchro.sed import running_spectral_index, log_parabola_running_index
    nus = np.logspace(-2, 2, 60)
    alpha = running_spectral_index(nus, lambda g: p0 + a * np.log(g))
    slope = np.polyfit(np.log(nus), alpha, 1)[0]
    pred = log_parabola_running_index(nus, p0, a)
    return slope, a / 2.0, alpha[30], pred[30]


if __name__ == "__main__":
    print("=== 1. Relative emissivity == numerical double integral ===")
    for p, num, ana, r in test_rel_emissivity():
        print(f"  p={p}: num={num:.6e} analytic={ana:.6e} ratio={r:.6f}")

    print("=== 2. Spectral index alpha_s = (p-1)/2 ===")
    for p, a, t in test_spectral_index():
        print(f"  p={p}: alpha_s={a:.4f} theory={t:.4f}")

    print("=== 3. Curvature Delta_alpha_s = Var(p)/4 ===")
    c, t = test_curvature()
    print(f"  Delta_alpha_s={c:.6f}  Var(p)/4={t:.6f}")

    print("=== 4. Absolute emissivity (CGS) == prefactor * rel(nu/nu_B) ===")
    num, abs_, r = test_abs_emissivity()
    print(f"  pref*rel={num:.6e} analytic={abs_:.6e} ratio={r:.6f}")

    print("=== 5. LOS-varying p: ln I coefficients == cumulants of alpha_s ===")
    theo, num = test_los_cumulants()
    print(f"  linear: -<a>={theo[0]:+.6f} vs fit {num[0]:+.6f}")
    print(f"  quadratic: Var/2={theo[1]:+.6f} vs fit {num[1]:+.6f}")
    print(f"  cubic: -Skew/6={theo[2]:+.6f} vs fit {num[2]:+.6f}")

    print("=== 6. Running spectral index (log-parabola): d a_s/d ln nu = a/2 ===")
    slope, a2, a_mid, p_mid = test_running_index()
    print(f"  slope={slope:.4f}  a/2={a2:.4f}   a_s(nu=1)={a_mid:.4f} vs pred {p_mid:.4f}")
