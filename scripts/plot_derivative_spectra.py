"""
Compute and plot the Schott harmonic Stokes parameters and their
derivative spectra (dI/dgamma, dI/dalpha) for synchrotron radiation.

This script generates Figure 2 of the paper:
  - Panel (a): I_n, Q_n, V_n vs harmonic number n for several gamma values
  - Panel (b): derivative spectra dI_n/dgamma and dI_n/dalpha vs n

All quantities in CGS units; plots are normalised for clarity.
"""

import numpy as np
from scipy.special import jv, jvp
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)
rc("font", family="serif", size=11)


# --- Physical functions ---

def bessel_arg(n, beta, alpha, theta):
    """Bessel function argument x_n = n * beta_perp * sin(theta) / (1 - beta_par * cos(theta))."""
    beta_perp = beta * np.sin(alpha)
    beta_par = beta * np.cos(alpha)
    return n * beta_perp * np.sin(theta) / (1.0 - beta_par * np.cos(theta))


def stokes_harmonic(n, gamma, alpha, theta):
    """
    Compute normalised Stokes (I_n, Q_n, V_n) at harmonic n.

    Returns values normalised by the common prefactor e^2 omega_B^2 / (8 pi^2 c),
    so the returned quantities are dimensionless functions of (n, gamma, alpha, theta).
    """
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    beta_perp = beta * np.sin(alpha)
    beta_par = beta * np.cos(alpha)
    x = bessel_arg(n, beta, alpha, theta)

    Jn = jv(n, x)
    Jnp = jvp(n, x, 1)

    denom = (1.0 - beta_par * np.cos(theta))**2
    geom_par = (np.cos(theta) - beta_par)**2 / (np.sin(theta)**2)

    P_par = n**2 * geom_par * Jn**2 / denom
    P_perp = n**2 * beta_perp**2 * Jnp**2 / denom

    I_n = P_par + P_perp
    Q_n = P_par - P_perp
    V_n = 2.0 * n**2 * beta_perp * (np.cos(theta) - beta_par) / (
        np.sin(theta) * denom
    ) * Jn * Jnp

    return I_n, Q_n, V_n


def derivative_gamma(n, gamma, alpha, theta, dgamma=0.01):
    """Numerical derivative dS_n/dgamma via central differences."""
    I_p, Q_p, V_p = stokes_harmonic(n, gamma + dgamma, alpha, theta)
    I_m, Q_m, V_m = stokes_harmonic(n, gamma - dgamma, alpha, theta)
    return (
        (I_p - I_m) / (2 * dgamma),
        (Q_p - Q_m) / (2 * dgamma),
        (V_p - V_m) / (2 * dgamma),
    )


def derivative_alpha(n, gamma, alpha, theta, dalpha=0.01):
    """Numerical derivative dS_n/dalpha via central differences."""
    I_p, Q_p, V_p = stokes_harmonic(n, gamma, alpha + dalpha, theta)
    I_m, Q_m, V_m = stokes_harmonic(n, gamma, alpha - dalpha, theta)
    return (
        (I_p - I_m) / (2 * dalpha),
        (Q_p - Q_m) / (2 * dalpha),
        (V_p - V_m) / (2 * dalpha),
    )


# --- Plotting ---

def main():
    theta = np.pi / 3       # observation angle 60 deg from B
    alpha_ref = np.pi / 4   # pitch angle 45 deg
    n_max = 80
    harmonics = np.arange(1, n_max + 1)

    gammas = [2, 5, 10, 20]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- Panel (a): I_n for several gamma ---
    ax = axes[0, 0]
    for gamma, col in zip(gammas, colors):
        I_vals = np.array([stokes_harmonic(n, gamma, alpha_ref, theta)[0] for n in harmonics])
        I_vals /= I_vals.max() if I_vals.max() > 0 else 1.0
        ax.plot(harmonics, I_vals, color=col, label=rf"$\gamma = {gamma}$", lw=1.2)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"$I_n$ (normalised)")
    ax.set_title(r"(a) Intensity spectrum $I_n$")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)

    # --- Panel (b): Q_n / I_n (polarisation fraction) ---
    ax = axes[0, 1]
    for gamma, col in zip(gammas, colors):
        I_vals = np.array([stokes_harmonic(n, gamma, alpha_ref, theta)[0] for n in harmonics])
        Q_vals = np.array([stokes_harmonic(n, gamma, alpha_ref, theta)[1] for n in harmonics])
        mask = I_vals > 1e-30
        pol_frac = np.where(mask, Q_vals / I_vals, 0.0)
        ax.plot(harmonics[mask], pol_frac[mask], color=col, label=rf"$\gamma = {gamma}$", lw=1.2)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"$Q_n / I_n$")
    ax.set_title(r"(b) Linear polarisation fraction")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- Panel (c): dI_n/dgamma (energy derivative spectrum) ---
    ax = axes[1, 0]
    gamma_ref = 10
    dI_dgamma = np.array([derivative_gamma(n, gamma_ref, alpha_ref, theta)[0] for n in harmonics])
    dQ_dgamma = np.array([derivative_gamma(n, gamma_ref, alpha_ref, theta)[1] for n in harmonics])
    scale = np.abs(dI_dgamma).max()
    ax.plot(harmonics, dI_dgamma / scale, "b-", label=r"$\partial I_n / \partial\gamma$", lw=1.2)
    ax.plot(harmonics, dQ_dgamma / scale, "r--", label=r"$\partial Q_n / \partial\gamma$", lw=1.2)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"Derivative spectrum (normalised)")
    ax.set_title(rf"(c) Energy derivative ($\gamma_0 = {gamma_ref}$)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- Panel (d): dI_n/dalpha (pitch angle derivative spectrum) ---
    ax = axes[1, 1]
    dI_dalpha = np.array([derivative_alpha(n, gamma_ref, alpha_ref, theta)[0] for n in harmonics])
    dQ_dalpha = np.array([derivative_alpha(n, gamma_ref, alpha_ref, theta)[1] for n in harmonics])
    dV_dalpha = np.array([derivative_alpha(n, gamma_ref, alpha_ref, theta)[2] for n in harmonics])
    scale = np.abs(dI_dalpha).max()
    ax.plot(harmonics, dI_dalpha / scale, "b-", label=r"$\partial I_n / \partial\alpha$", lw=1.2)
    ax.plot(harmonics, dQ_dalpha / scale, "r--", label=r"$\partial Q_n / \partial\alpha$", lw=1.2)
    ax.plot(harmonics, dV_dalpha / scale, "g:", label=r"$\partial V_n / \partial\alpha$", lw=1.5)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"Derivative spectrum (normalised)")
    ax.set_title(rf"(d) Pitch angle derivative ($\alpha_0 = \pi/4$)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    fig.suptitle("Synchrotron harmonic spectra and derivative spectra", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = "../figures/derivative_spectra.pdf"
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
