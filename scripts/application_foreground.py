"""Illustrative foreground from an amplitude-weighted Gaussian index mixture.

Assume locally unbroken power laws, achromatic mixing weights, and a Gaussian
PDF of electron index p weighted by brightness at nu0. Then beta=(p+3)/2 and
T=A(nu/nu0)^(-mean(beta))*exp[Var(p)*ln(nu/nu0)^2/8]. This is a specified
Gaussian closure for an empirical-SED mixture; general electron-number
moments do not supply these weights. Non-Gaussian statistics or chromatic
mixing require their own discrepancy. The overlaid signal has arbitrary
relative amplitude; this plot is not a21-cm precision or inference validation.
"""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

from synchro.sed import spectral_curvature

rc("text", usetex=False)
rc("font", family="serif", size=11)
rc("mathtext", fontset="cm")


def brightness_temperature_moments(nu, nu0, p_mean, var_p, A=1.0):
    """T_b(nu) = A (nu/nu0)^-beta_mean * exp[Var(p) ln^2(nu/nu0)/8], beta=(p+3)/2."""
    beta_mean = (p_mean + 3.0) / 2.0
    return A * (nu / nu0) ** (-beta_mean) * np.exp(var_p * np.log(nu / nu0) ** 2 / 8.0)


def main():
    # --- physical inputs (Galactic synchrotron, 21-cm band) ---
    p_mean, var_p = 2.5, 0.04  # electron-spectrum index and its variance
    beta_mean = (p_mean + 3.0) / 2.0  # brightness-temperature index = 2.75
    delta_beta = spectral_curvature(var_p)  # Var(p)/4 = 0.01

    nu0 = 150.0  # reference frequency [MHz]
    nu = np.linspace(50.0, 200.0, 400)

    Tb = brightness_temperature_moments(nu, nu0, p_mean, var_p)
    Tb_pure = (nu / nu0) ** (-beta_mean)  # pure power law (Var(p)=0)

    # --- toy 21-cm global signal (a shallow Gaussian absorption at z ~ 17) ---
    nu21 = 1420.4 / (1.0 + 17.0)  # 21-cm line redshifted to z=17 -> ~79 MHz
    depth, width = 0.015, 3.0  # relative depth and width [MHz]
    signal = -depth * np.exp(-0.5 * ((nu - nu21) / width) ** 2)

    # --- figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.loglog(nu, Tb, "C0-", lw=1.6, label="Gaussian index mixture")
    ax.loglog(nu, Tb_pure, "k--", lw=1.0, label=r"power law $\nu^{-\beta}$")
    ax.set_xlabel(r"Frequency $\nu$ [MHz]")
    ax.set_ylabel(r"$T_b(\nu)$ [a.u.]")
    ax.set_title(r"(a) Foreground brightness temperature")
    ax.legend(fontsize=9)
    ax.set_xlim(nu.min(), nu.max())
    ax.set_ylim(0.9 * min(Tb.min(), Tb_pure.min()), 1.1 * max(Tb.max(), Tb_pure.max()))

    ax = axes[1]
    # fractional curvature: T_b / T_b_pure - 1  ~  Var(p)/8 ln^2(nu/nu0)
    curv = Tb / Tb_pure - 1.0
    ax.semilogx(nu, curv * 100, "C0-", lw=1.6)
    x = np.linspace(50, 200, 200)
    ax.semilogx(
        x,
        (var_p / 8.0) * np.log(x / nu0) ** 2 * 100,
        "k--",
        lw=1.0,
        label=r"$\mathrm{Var}(p)\,\ln^2(\nu/\nu_0)/8$",
    )
    # toy signal (not to scale; shown on same panel for context)
    ax.semilogx(nu, signal * 100, "C3-", lw=1.0, alpha=0.8, label="21-cm signal (toy)")
    ax.set_xlabel(r"Frequency $\nu$ [MHz]")
    ax.set_ylabel(r"Deviation [%]")
    ax.set_title(r"(b) Curvature = $\frac{1}{2}\Delta\beta\ln^2(\nu/\nu_0)$")
    ax.legend(fontsize=9)
    ax.set_ylim(-3, 3)

    fig.suptitle("Gaussian index-mixture illustration", fontsize=12, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    import os

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "figures",
        "foreground_demo.pdf",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # --- printed summary ---
    print(f"  <p> = {p_mean},  Var(p) = {var_p}")
    print(f"  brightness-temperature index  <beta> = (p+3)/2 = {beta_mean:.3f}")
    print(f"  spectral curvature  Delta_beta = Var(p)/4 = {delta_beta:.3f}")
    print(f"  ln T_b = -{beta_mean:.3f} ln(nu/nu0) + {var_p/8:.4f} ln^2(nu/nu0) + ...")
    edge = 50.0
    print(
        f"  curvature at {edge:.0f} MHz (band edge) = "
        f"{(var_p/8.0)*np.log(edge/nu0)**2*100:.2f}% of the power law"
    )


if __name__ == "__main__":
    main()
