"""Validation of the remaining RT pieces: Magnus, conversion, cross-cumulants,
and the moment-driven LOS.

Checks:
1. Magnus expansion: exp(Omega1 [+Omega2]) vs exact LOS chain (homogeneous).
2. Faraday conversion: cold-plasma rho_Q ~ nu^-3, and ratio signed Stokes rQ/rV.
3. Cross-cumulant: correlated emissivity & rotation along the LOS != uncorrelated.
4. Moment-driven slab: thin limit, self-absorbed I, Faraday rotation.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from syncmoments.magnus import magnus_S
from syncmoments.transfer import mueller_matrix, transfer_los
from syncmoments.conversion import (
    conversion_rotation_ratio,
    mueller_rotation,
    mueller_conversion,
)
from syncmoments.kirchhoff import (
    emissivity_from_moments,
    absorbed_intensity_from_moments,
)
from syncmoments.los_moments import moment_driven_slab


# --- 1. Magnus -----------------------------------------------------------
def test_magnus(n_slabs=12, ds=0.05):
    rng = np.random.default_rng(0)
    K_s = []
    for _ in range(n_slabs):
        rV = 0.3 * rng.standard_normal()
        rQ = 0.3 * rng.standard_normal()
        K_s.append(mueller_matrix(0.0, 0.0, 0.0, 0.0, rQ, 0.0, rV))
    K_s = jnp.stack(K_s)
    S0 = jnp.array([0.0, 1.0, 0.0, 0.0])

    S_exact = transfer_los(S0, jnp.zeros((n_slabs, 4)), K_s, ds)
    S_o1 = magnus_S(S0, K_s, ds, order=1)
    S_o2 = magnus_S(S0, K_s, ds, order=2)
    e1 = float(jnp.max(jnp.abs(S_exact - S_o1)))
    e2 = float(jnp.max(jnp.abs(S_exact - S_o2)))
    return e1, e2


# --- 2. Faraday conversion ----------------------------------------------
def test_conversion():
    nu = 1e9  # 1 GHz
    n_e, B_par, B_perp = 0.01, 1e-6, 1e-6
    rV = mueller_rotation(nu, n_e, B_par)
    rQ = mueller_conversion(nu, n_e, B_perp)
    # scaling: rho_Q ~ nu^-3, rho_V ~ nu^-2
    rV2 = mueller_rotation(2 * nu, n_e, B_par)
    rQ2 = mueller_conversion(2 * nu, n_e, B_perp)
    ratio_scaling = (rQ2 / rQ) / (rV2 / rV)  # should be (2^-3)/(2^-2) = 1/2
    qv = conversion_rotation_ratio(nu, B_perp, B_par)
    return rV, rQ, qv, ratio_scaling


# --- 3. Cross-cumulant (emissivity ~ rotation along the LOS) ---------------
def test_cross_cumulant(n_slabs=200, L=1.0, delta=0.3):
    s = np.linspace(0, L, n_slabs + 1)[:-1] + 0.5 * L / n_slabs
    ds = L / n_slabs
    f = np.sin(2 * np.pi * s / L)  # common field fluctuation
    eps_Q = 1.0 * (1.0 + delta * f)  # emissivity ~ B(s)
    rV_s = 0.5 * (1.0 + delta * f)  # rotation ~ B(s)

    # exact: correlated emissivity & rotation
    eps_s = jnp.stack(
        [
            jnp.zeros(n_slabs),
            jnp.asarray(eps_Q),
            jnp.zeros(n_slabs),
            jnp.zeros(n_slabs),
        ],
        axis=1,
    )
    K_s = jnp.stack([mueller_matrix(0, 0, 0, 0, 0, 0, r) for r in rV_s])
    S_exact = transfer_los(jnp.zeros(4), eps_s, K_s, ds)
    P_exact = S_exact[1] + 1j * S_exact[2]

    # uncorrelated: uniform emissivity = <eps>, same rotation
    eps_mean = np.mean(eps_Q)
    eps_u = jnp.stack(
        [
            jnp.zeros(n_slabs),
            jnp.full(n_slabs, eps_mean),
            jnp.zeros(n_slabs),
            jnp.zeros(n_slabs),
        ],
        axis=1,
    )
    S_unc = transfer_los(jnp.zeros(4), eps_u, K_s, ds)
    P_unc = S_unc[1] + 1j * S_unc[2]

    return complex(P_exact), complex(P_unc), abs(P_exact - P_unc)


# --- 4. Moment-driven LOS -------------------------------------------------
def _gaussian_moments(gamma0=30.0, sigma=3.0):
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    dg = g - gamma0
    return (
        np.trapezoid(N, g),
        np.trapezoid(N * dg, g),
        np.trapezoid(N * dg**2, g),
        np.trapezoid(N / g, g),
    )


def test_moment_slab(nu=100.0, nu_c_ref=1.0, gamma0=30.0):
    M0, M1, M2, ig = _gaussian_moments(gamma0)
    L = 1e4

    # (a) thin limit: L small -> I ~ j L
    S_thin = moment_driven_slab(nu, gamma0, nu_c_ref, M0, M1, M2, ig, L=1e-2)
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    thin_err = abs(float(S_thin[0]) - float(j) * 1e-2) / abs(float(j) * 1e-2)

    # (b) self-absorbed I (no rotation): matches Kirchhoff closed form
    S_abs = moment_driven_slab(nu, gamma0, nu_c_ref, M0, M1, M2, ig, L=L)
    I_ref = absorbed_intensity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, L, ig)
    abs_err = abs(float(S_abs[0]) - float(I_ref)) / abs(float(I_ref))

    # (c) Physical slab at finite internal Faraday depth, with CGS throughout.
    from syncmoments.los_moments import moment_driven_slab_cgs
    from syncmoments.kirchhoff import cgs_coefficients_from_moments

    nu_hz, ne, bpar, bperp, g0 = 1e8, 0.03, 2e-6, 5e-6, 2500.0
    number = 1e-15  # suppress absorption, retaining finite internal rotation
    length = 1.8 / mueller_rotation(nu_hz, ne, bpar)
    source = cgs_coefficients_from_moments(
        nu_hz, g0, bperp, number, 0.0, 0.0, number / g0
    )
    S_rot = moment_driven_slab_cgs(
        nu_hz, g0, bperp, number, 0.0, 0.0, number / g0, length, n_e=ne, B_par=bpar
    )
    expected = float(source[1]) * length * np.exp(0.9j) * np.sinc(0.9 / np.pi)
    # Cold conversion is included in the numerical result; its tiny depth
    # and the finite absorption depth are separate terms in this comparison.
    rot_err = abs(complex(S_rot[1], S_rot[2]) - expected) / abs(expected)
    return thin_err, abs_err, rot_err


if __name__ == "__main__":
    print("=== 1. Magnus expansion vs exact chain ===")
    e1, e2 = test_magnus()
    print(f"  order-1 err = {e1:.2e}   order-2 err = {e2:.2e}")

    print("=== 2. Faraday conversion (cold plasma) ===")
    rV, rQ, qv, rs = test_conversion()
    print(f"  rV(1 GHz) = {rV:.3e} cm^-1, rQ(1 GHz) = {rQ:.3e} cm^-1")
    print(
        f"  signed Stokes rQ/rV = {qv:.3e},  nu-scaling ratio = {rs:.4f} (expect 0.5)"
    )

    print("=== 3. Cross-cumulant (emissivity ~ rotation) ===")
    Pex, Pun, dP = test_cross_cumulant()
    print(f"  correlated P = {Pex:.6f}")
    print(f"  uncorrelated P = {Pun:.6f}")
    print(f"  |difference| = {dP:.3e}  (cross-cumulant contribution)")

    print("=== 4. Moment-driven LOS slab ===")
    thin, ab, rot = test_moment_slab()
    print(f"  thin-limit rel err = {thin:.2e}")
    print(f"  self-absorbed rel err = {ab:.2e}")
    print(f"  Faraday rotation rel err = {rot:.2e}")
