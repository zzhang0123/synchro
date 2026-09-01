"""Validation of the Kirchhoff layer (emissivity / self-absorption).

Checks:
1. Power-law N ~ gamma^-p: j ~ nu^-(p-1)/2, alpha ~ nu^-(p+4)/2, S ~ nu^(5/2).
2. Kirchhoff closure: S_nu is independent of the distribution amplitude.
3. Moment expansion (2nd order) of j and alpha matches direct quadrature for
   a narrow distribution.
"""

from __future__ import annotations

import numpy as np

from synchro.kirchhoff import (
    emissivity, absorption, source_function, derivative_weighted,
    P_derivatives, moment_expansion_emissivity, moment_expansion_absorption,
    absorption_moments, emissivity_from_moments, absorption_from_moments,
    absorbed_intensity_from_moments, inverse_gamma_moment,
    emissivity_Q, absorption_Q,
)


def _power_law_grid(p, gmin=1.0, gmax=2000.0, ng=600):
    g = np.geomspace(gmin, gmax, ng)
    return g, g**-p


def test_powerlaw_slopes(p=2.5, nu_c_ref=1.0):
    g, N = _power_law_grid(p)
    nus = np.logspace(1.0, 4.0, 10)
    j = np.array([emissivity(N, g, nu, nu_c_ref) for nu in nus])
    a = np.array([absorption(N, g, nu, nu_c_ref) for nu in nus])
    s = j / a
    sj = np.polyfit(np.log(nus), np.log(j), 1)[0]
    sa = np.polyfit(np.log(nus), np.log(a), 1)[0]
    ss = np.polyfit(np.log(nus), np.log(s), 1)[0]
    return sj, sa, ss, -(p - 1) / 2, -(p + 4) / 2, 2.5


def test_closure(p=2.5, nu_c_ref=1.0):
    g, N = _power_law_grid(p)
    nu = 100.0
    s1 = source_function(N, g, nu, nu_c_ref)
    s2 = source_function(10.0 * N, g, nu, nu_c_ref)
    return s1, s2, abs(s1 - s2) / abs(s1)


def test_moment_expansion(gamma0=30.0, sigma=3.0, nu_c_ref=1.0, nu=100.0):
    g = np.linspace(1.0, 200.0, 800)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    gw = derivative_weighted(N, g)

    # direct
    j_ex = emissivity(N, g, nu, nu_c_ref)
    a_ex = absorption(N, g, nu, nu_c_ref)

    # moment expansion (2nd order)
    P0, P1, P2 = P_derivatives(nu, gamma0, nu_c_ref)
    j_me = moment_expansion_emissivity(N, g, gamma0, P0, P1, P2)
    a_me = moment_expansion_absorption(gw, g, gamma0, P0, P1, P2, nu)

    return (j_ex, j_me, a_ex, a_me,
            abs(j_ex - j_me) / abs(j_ex), abs(a_ex - a_me) / abs(a_ex))


def test_closure_form(gamma0=30.0, sigma=3.0):
    """Kirchhoff closure in moment form: D_k from (M_k, <1/g>) vs direct."""
    g = np.linspace(1.0, 200.0, 4000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    gw = derivative_weighted(N, g)
    dg = g - gamma0

    M0 = np.trapezoid(N, g)
    M1 = np.trapezoid(N * dg, g)
    M2 = np.trapezoid(N * dg**2, g)
    inv_gamma = np.trapezoid(N / g, g)                    # exact inverse moment
    inv_gamma_ser = inverse_gamma_moment(M0, M1, M2, gamma0)  # geometric series

    D0, D1, D2 = absorption_moments(M0, M1, M2, gamma0, inv_gamma)
    D0_d = np.trapezoid(gw, g)
    D1_d = np.trapezoid(gw * dg, g)
    D2_d = np.trapezoid(gw * dg**2, g)
    exact_err = (abs(D0 - D0_d) / abs(D0_d), abs(D1 - D1_d) / abs(D1_d),
                 abs(D2 - D2_d) / abs(D2_d))
    approx_err = abs(inv_gamma_ser - inv_gamma) / abs(inv_gamma)
    return exact_err, approx_err


def test_end_to_end(gamma0=30.0, sigma=3.0, nu_c_ref=1.0, L=1e4):
    """Absorbed I_nu from moments only vs direct full-N quadrature."""
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    dg = g - gamma0
    M0 = np.trapezoid(N, g)
    M1 = np.trapezoid(N * dg, g)
    M2 = np.trapezoid(N * dg**2, g)
    inv_gamma = np.trapezoid(N / g, g)

    nus = np.logspace(1.0, 3.0, 12)
    I_direct = np.array([source_function(N, g, nu, nu_c_ref)
                         * (1 - np.exp(-absorption(N, g, nu, nu_c_ref) * L))
                         for nu in nus])
    I_mom = np.array([float(absorbed_intensity_from_moments(
        nu, gamma0, nu_c_ref, M0, M1, M2, L, inv_gamma)) for nu in nus])
    rel = np.abs(I_direct - I_mom) / np.abs(I_direct)
    return nus, I_direct, I_mom, rel


def test_polarisation_fraction():
    """Thin (p+1)/(p+7/3) and thick -3/(6p+13) polarisation fractions."""
    g = np.geomspace(0.01, 2000.0, 2000)
    out = []
    for p in [2.0, 2.5, 3.0, 4.0]:
        N = g**-p
        jF = emissivity(N, g, 1.0, 1.0)
        jG = emissivity_Q(N, g, 1.0, 1.0)
        aF = absorption(N, g, 1.0, 1.0)
        aG = absorption_Q(N, g, 1.0, 1.0)
        Pi_thin = jG / jF
        Sp = (jF + jG) / (aF + aG)
        Sm = (jF - jG) / (aF - aG)
        Pi_thick = (Sp - Sm) / (Sp + Sm)
        out.append((p, Pi_thin, (p + 1) / (p + 7 / 3),
                    Pi_thick, -3 / (6 * p + 13)))
    return out


if __name__ == "__main__":
    print("=== 1. Power-law slopes ===")
    sj, sa, ss, e_j, e_a, e_s = test_powerlaw_slopes()
    print(f"  j slope = {sj:.3f} (expect {e_j})")
    print(f"  alpha slope = {sa:.3f} (expect {e_a})")
    print(f"  S = j/alpha slope = {ss:.3f} (expect {e_s})")

    print("=== 2. Kirchhoff closure (S independent of amplitude) ===")
    s1, s2, rel = test_closure()
    print(f"  S = {s1:.6e}  S(10N) = {s2:.6e}  rel diff = {rel:.2e}")

    print("=== 3. Moment expansion vs direct quadrature ===")
    j_ex, j_me, a_ex, a_me, ej, ea = test_moment_expansion()
    print(f"  j: exact={j_ex:.6e} moment={j_me:.6e} rel err={ej:.2e}")
    print(f"  a: exact={a_ex:.6e} moment={a_me:.6e} rel err={ea:.2e}")

    print("=== 4. Kirchhoff closure in moment form (D_k from M_k + <1/g>) ===")
    exact_err, approx_err = test_closure_form()
    print(f"  exact closure rel err: D0={exact_err[0]:.2e}  D1={exact_err[1]:.2e}  D2={exact_err[2]:.2e}")
    print(f"  <1/g> geometric-series approx rel err = {approx_err:.2e}")

    print("=== 5. End-to-end absorbed I_nu (moments vs direct) ===")
    nus, Id, Im, rel = test_end_to_end()
    print(f"  max rel err over nu = {rel.max():.2e}")
    for nu, r in zip(nus[::3], rel[::3]):
        print(f"    nu={nu:8.1f}: rel err={r:.2e}")

    print("=== 6. Polarised absorption: thin & thick polarisation fractions ===")
    for p, pt, pt_th, pk, pk_th in test_polarisation_fraction():
        print(f"  p={p}: thin {pt:.4f} (theory {pt_th:.4f})  thick {pk:+.4f} (theory {pk_th:+.4f})")
