"""Figure for Sec. 6 (polarised radiative transfer).

Three panels, one per structural claim of the section:

(a) Kirchhoff closure -- the absorbed spectrum built from the emissivity
    moments (M0, M1, M2) plus <1/gamma> alone, against the direct quadrature
    over the full N(gamma).  No new statistics enter the absorption.
(b) Polarisation through the self-absorption turnover for power-law electron
    spectra: the thin fraction (p+1)/(p+7/3) flips sign to the self-absorbed
    -3/(6p+13), because the parallel source function dominates when thick.
(c) Magnus expansion of the path-ordered LOS transfer: Omega1 (LOS-averaged
    coefficients = first moments) versus Omega1+Omega2 (the rotation-conversion
    commutator = cross-cumulants), against the exact slab chain.

Generates figures/transfer.pdf.
"""

from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

from synchro.kirchhoff import (emissivity, absorption, emissivity_Q,
                               absorption_Q, source_function,
                               absorbed_intensity_from_moments)
from synchro.transfer import mueller_matrix, transfer_los
from synchro.magnus import magnus_S

rc("text", usetex=False)
rc("font", family="serif", size=11)
rc("mathtext", fontset="cm")


# --- (a) Kirchhoff closure: moments vs direct ------------------------------
def panel_closure(ax, gamma0=30.0, sigma=3.0, nu_c_ref=1.0, L=1e4):
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    dg = g - gamma0
    M0 = np.trapezoid(N, g)
    M1 = np.trapezoid(N * dg, g)
    M2 = np.trapezoid(N * dg**2, g)
    inv_gamma = np.trapezoid(N / g, g)

    nus = np.logspace(0.3, 3.2, 60)
    I_direct = np.array([source_function(N, g, nu, nu_c_ref)
                         * (-np.expm1(-absorption(N, g, nu, nu_c_ref) * L))
                         for nu in nus])
    I_mom = np.array([float(absorbed_intensity_from_moments(
        nu, gamma0, nu_c_ref, M0, M1, M2, L, inv_gamma)) for nu in nus])

    ax.loglog(nus, I_direct, "k-", lw=2.0, label=r"direct $\int N(\gamma)\,d\gamma$")
    ax.loglog(nus, I_mom, "C3--", lw=1.8,
              label=r"from $(M_0,M_1,M_2,\langle\gamma^{-1}\rangle)$")
    # nu^{5/2} source-function asymptote, anchored in the thick regime
    k = np.argmin(np.abs(nus - 3.0))
    ax.loglog(nus[:k + 12], I_direct[k] * (nus[:k + 12] / nus[k]) ** 2.5,
              ":", color="0.5", lw=1.4, label=r"$\propto\nu^{5/2}$ (thick)")
    ax.set_xlabel(r"$\nu\,/\,\nu_{c,\rm ref}$")
    ax.set_ylabel(r"$I_\nu$ (relative units)")
    ax.set_title("(a) Kirchhoff closure from moments")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    rel = np.max(np.abs(I_direct - I_mom) / np.abs(I_direct))
    ax.text(0.04, 0.93, rf"max rel. err $= {rel:.1e}$", transform=ax.transAxes,
            fontsize=8.5, va="top")


# --- (b) polarisation through the turnover --------------------------------
def panel_polarisation(ax, nu_c_ref=1.0):
    g = np.geomspace(0.01, 2000.0, 2000)
    taus = np.logspace(-2.5, 3.0, 70)
    for c, p in zip(["C0", "C1", "C2"], [2.0, 2.5, 3.0]):
        N = g**-p
        jF = emissivity(N, g, 1.0, nu_c_ref)
        jG = emissivity_Q(N, g, 1.0, nu_c_ref)
        aF = absorption(N, g, 1.0, nu_c_ref)
        aG = absorption_Q(N, g, 1.0, nu_c_ref)
        S_perp, S_par = (jF + jG) / (aF + aG), (jF - jG) / (aF - aG)
        # tau is the Stokes-I optical depth; the two modes scale with it
        t_perp = taus * (aF + aG) / aF
        t_par = taus * (aF - aG) / aF
        f_perp, f_par = -np.expm1(-t_perp), -np.expm1(-t_par)
        I = S_perp * f_perp + S_par * f_par
        Q = S_perp * f_perp - S_par * f_par
        ax.semilogx(taus, Q / I, "-", color=c, lw=1.8, label=rf"$p={p}$")
        ax.axhline((p + 1) / (p + 7 / 3), color=c, ls=":", lw=1.0)
        ax.axhline(-3 / (6 * p + 13), color=c, ls="--", lw=1.0)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel(r"$\tau_\nu = \alpha_I L$")
    ax.set_ylabel(r"$\Pi = Q/I$")
    ax.set_title("(b) Polarisation across the turnover")
    ax.text(0.04, 0.30, r"$\cdots\;\Pi_{\rm thin}=\frac{p+1}{p+7/3}$",
            transform=ax.transAxes, fontsize=8.5)
    ax.text(0.04, 0.19, r"$--\;\Pi_{\rm thick}=\frac{-3}{6p+13}$",
            transform=ax.transAxes, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")


# --- (c) Magnus: Omega1 vs Omega1 + Omega2 --------------------------------
def panel_magnus(ax, ds=1.0):
    """Two-region sightline: one rotation-dominated, one conversion-dominated.

    NOTE the generic input Stokes vector.  The leading commutator
    [K_conv, K_rot] generates a U-V mixing, which annihilates a pure-Q input,
    so S0 = (0,1,0,0) hides the Omega2 effect and both orders then appear to
    converge at the same rate.
    """
    S0 = jnp.array([1.0, 0.6, 0.3, 0.2])
    depths = np.logspace(-2.0, 0.0, 30)
    e1, e2 = [], []
    for a in depths:
        K_s = jnp.stack([
            mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a),   # rotation rho_V
            mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, a, 0.0),   # conversion rho_U
        ])
        S_ex = transfer_los(S0, jnp.zeros((2, 4)), K_s, ds)
        e1.append(float(jnp.max(jnp.abs(S_ex - magnus_S(S0, K_s, ds, order=1)))))
        e2.append(float(jnp.max(jnp.abs(S_ex - magnus_S(S0, K_s, ds, order=2)))))
    e1, e2 = np.array(e1), np.array(e2)
    d = depths * ds
    ax.loglog(d, e1, "C0-", lw=1.8, label=r"$\Omega_1$ (first moments)")
    ax.loglog(d, e2, "C3-", lw=1.8, label=r"$\Omega_1+\Omega_2$ (+ cross-cumulants)")
    k = 6
    ax.loglog(d, e1[k] * (d / d[k]) ** 2, ":", color="C0", lw=1.1)
    ax.loglog(d, e2[k] * (d / d[k]) ** 3, ":", color="C3", lw=1.1)
    ax.text(0.55, 0.30, r"$\propto\,\psi^{2}$", color="C0", transform=ax.transAxes,
            fontsize=9)
    ax.text(0.55, 0.10, r"$\propto\,\psi^{3}$", color="C3", transform=ax.transAxes,
            fontsize=9)
    ax.set_ylim(1e-11, 1e0)
    ax.set_xlabel(r"rotation--conversion depth $\psi$")
    ax.set_ylabel(r"$\max|S - S_{\rm Magnus}|$")
    ax.set_title("(c) Non-commutativity via Magnus")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panel_closure(axes[0])
    panel_polarisation(axes[1])
    panel_magnus(axes[2])
    plt.tight_layout()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "figures", "transfer.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
