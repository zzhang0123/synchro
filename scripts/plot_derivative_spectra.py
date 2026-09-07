"""
Compute and plot the Schott harmonic Stokes parameters and their derivative
spectra using the synchro package (corrected normalisation, autodiff).

Generates figures/derivative_spectra.pdf:
  (a) I_n vs harmonic number n for several gamma
  (b) Q_n / I_n (linear polarisation fraction)
  (c) energy derivative spectra dI/dgamma, dQ/dgamma at gamma_0
  (d) pitch-angle derivative spectra dI/dalpha, dQ/dalpha, dV/dalpha

Derivative spectra are computed at fixed physical B by JAX autodiff of the
resolved numerical Bessel integrals; quadrature error remains separate.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

from synchro.stokes import stokes_harmonic

rc("text", usetex=False)
rc("font", family="serif", size=11)
rc("mathtext", fontset="cm")


def stokes_vec(n, gamma, alpha, theta, *, B=None):
    """Vector harmonic response; B [G] selects physical fixed-field power."""
    return stokes_harmonic(n, gamma, alpha, theta, B=B)


def main():
    theta = np.pi / 3
    alpha_ref = np.pi / 4
    n_max = 80
    harmonics = jnp.arange(1, n_max + 1, dtype=jnp.float64)

    gammas = [2, 5, 10, 20]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- Panel (a): I_n for several gamma ---
    ax = axes[0, 0]
    for gamma, col in zip(gammas, colors):
        I, _, _ = stokes_vec(harmonics, gamma, alpha_ref, theta)
        I = np.asarray(I)
        I = I / I.max()
        ax.plot(
            np.arange(1, n_max + 1), I, color=col, label=rf"$\gamma = {gamma}$", lw=1.2
        )
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"$I_n$ (normalised)")
    ax.set_title(r"(a) Intensity spectrum $I_n$")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.5)

    # --- Panel (b): Q_n / I_n ---
    ax = axes[0, 1]
    for gamma, col in zip(gammas, colors):
        I, Q, _ = stokes_vec(harmonics, gamma, alpha_ref, theta)
        I = np.asarray(I)
        Q = np.asarray(Q)
        mask = I > 1e-30
        frac = np.where(mask, Q / I, 0.0)
        ax.plot(
            np.arange(1, n_max + 1)[mask],
            frac[mask],
            color=col,
            label=rf"$\gamma = {gamma}$",
            lw=1.2,
        )
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"$Q_n / I_n$")
    ax.set_title(r"(b) Linear polarisation fraction")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- Panels (c),(d): derivative spectra via autodiff ---
    gamma_ref = 10.0
    p_ref = jnp.array([gamma_ref, alpha_ref, theta])

    def f(p):
        I, Q, V = stokes_vec(harmonics, p[0], p[1], p[2], B=5e-6)
        return jnp.stack([I, Q, V])  # (3, N)

    J = jax.jacfwd(f)(p_ref)  # (3, N, 3): [Stokes, harmonic, param(gamma,alpha,theta)]
    J = np.asarray(J)
    dI_dg, dQ_dg = J[0, :, 0], J[1, :, 0]  # d/dgamma
    dI_da, dQ_da, dV_da = J[0, :, 1], J[1, :, 1], J[2, :, 1]  # d/dalpha
    ns = np.arange(1, n_max + 1)

    # --- Panel (c): energy derivative ---
    ax = axes[1, 0]
    scale = np.abs(dI_dg).max()
    ax.plot(ns, dI_dg / scale, "b-", label=r"$\partial I_n / \partial\gamma$", lw=1.2)
    ax.plot(ns, dQ_dg / scale, "r--", label=r"$\partial Q_n / \partial\gamma$", lw=1.2)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"Derivative spectrum (normalised)")
    ax.set_title(rf"(c) Energy derivative, fixed $B$ ($\gamma_0 = {gamma_ref}$)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # --- Panel (d): pitch-angle derivative ---
    ax = axes[1, 1]
    scale = np.abs(dI_da).max()
    ax.plot(ns, dI_da / scale, "b-", label=r"$\partial I_n / \partial\alpha$", lw=1.2)
    ax.plot(ns, dQ_da / scale, "r--", label=r"$\partial Q_n / \partial\alpha$", lw=1.2)
    ax.plot(ns, dV_da / scale, "g:", label=r"$\partial V_n / \partial\alpha$", lw=1.5)
    ax.set_xlabel(r"Harmonic number $n$")
    ax.set_ylabel(r"Derivative spectrum (normalised)")
    ax.set_title(r"(d) Pitch-angle derivative ($\alpha_0 = \pi/4$)")
    ax.legend(fontsize=9)
    ax.set_xlim(1, n_max)
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    fig.suptitle(
        "Synchrotron harmonic spectra and derivative spectra", fontsize=13, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outpath = os.path.join(repo_root, "figures", "derivative_spectra.pdf")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved figure to {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
