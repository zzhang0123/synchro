"""Validation of the remaining RT pieces: Magnus, conversion, cross-cumulants,
and the moment-driven LOS.

Checks:
1. Magnus expansion: exp(Omega1 [+Omega2]) vs exact LOS chain (homogeneous).
2. Faraday conversion: cold-plasma rho_Q ~ nu^-3, and ratio rho_Q/rho_V.
3. Cross-cumulant: correlated emissivity & rotation along the LOS != uncorrelated.
4. Moment-driven slab: thin limit, self-absorbed I, Faraday rotation.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from synchro.magnus import omega1, omega2, magnus_S
from synchro.transfer import mueller_matrix, transfer_los, faraday_rotation_matrix
from synchro.conversion import (
    rotation_coefficient, conversion_coefficient, conversion_rotation_ratio,
    mueller_rotation, mueller_conversion,
)
from synchro.kirchhoff import (
    emissivity_from_moments, absorption_from_moments, absorbed_intensity_from_moments,
)
from synchro.los_moments import moment_driven_slab


# --- 1. Magnus -----------------------------------------------------------
def test_magnus(n_slabs=12, ds=0.05):
    rng = np.random.default_rng(0)
    K_s = []
    for _ in range(n_slabs):
        rV = 0.3 * rng.standard_normal()
        rU = 0.3 * rng.standard_normal()
        K_s.append(mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, rU, rV))
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
    rU = mueller_conversion(nu, n_e, B_perp)
    # scaling: rho_Q ~ nu^-3, rho_V ~ nu^-2
    rV2 = mueller_rotation(2 * nu, n_e, B_par)
    rU2 = mueller_conversion(2 * nu, n_e, B_perp)
    ratio_scaling = (rU2 / rU) / (rV2 / rV)  # should be (2^-3)/(2^-2) = 1/2
    qv = conversion_rotation_ratio(nu, B_perp, B_par)
    return rV, rU, qv, ratio_scaling


# --- 3. Cross-cumulant (emissivity ~ rotation along the LOS) ---------------
def test_cross_cumulant(n_slabs=200, L=1.0, delta=0.3):
    s = np.linspace(0, L, n_slabs + 1)[:-1] + 0.5 * L / n_slabs
    ds = L / n_slabs
    f = np.sin(2 * np.pi * s / L)  # common field fluctuation
    eps_Q = 1.0 * (1.0 + delta * f)          # emissivity ~ B(s)
    rV_s = 0.5 * (1.0 + delta * f)           # rotation ~ B(s)

    # exact: correlated emissivity & rotation
    eps_s = jnp.stack([jnp.zeros(n_slabs), jnp.asarray(eps_Q),
                       jnp.zeros(n_slabs), jnp.zeros(n_slabs)], axis=1)
    K_s = jnp.stack([mueller_matrix(0, 0, 0, 0, 0, 0, r) for r in rV_s])
    S_exact = transfer_los(jnp.zeros(4), eps_s, K_s, ds)
    P_exact = S_exact[1] + 1j * S_exact[2]

    # uncorrelated: uniform emissivity = <eps>, same rotation
    eps_mean = np.mean(eps_Q)
    eps_u = jnp.stack([jnp.zeros(n_slabs), jnp.full(n_slabs, eps_mean),
                       jnp.zeros(n_slabs), jnp.zeros(n_slabs)], axis=1)
    S_unc = transfer_los(jnp.zeros(4), eps_u, K_s, ds)
    P_unc = S_unc[1] + 1j * S_unc[2]

    return complex(P_exact), complex(P_unc), abs(P_exact - P_unc)


# --- 4. Moment-driven LOS -------------------------------------------------
def _gaussian_moments(gamma0=30.0, sigma=3.0):
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma**2)
    dg = g - gamma0
    return (np.trapezoid(N, g), np.trapezoid(N * dg, g),
            np.trapezoid(N * dg**2, g), np.trapezoid(N / g, g))


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

    # (c) Faraday rotation: thin + rotation rotates (Q,U) by rV L
    n_e, B_par = 0.01, 1e-6
    rV = mueller_rotation(1e9, n_e, B_par)  # use real frequency for rV
    # rotate the intrinsic (Q0, U0) by angle rV*L and compare to transfer
    Q0, U0 = 0.2, 0.1
    S_rot = moment_driven_slab(nu, gamma0, nu_c_ref, M0, M1, M2, ig,
                               L=1e-2, Q0=Q0, U0=U0, n_e=n_e, B_par=B_par)
    # thin => I ~ j L, Q,U ~ j L * R(rV L) @ [Q0, U0]
    R = faraday_rotation_matrix(rV * 1e-2)
    Qexp = j * 1e-2 * (R[1, 1] * Q0 + R[1, 2] * U0)
    Uexp = j * 1e-2 * (R[2, 1] * Q0 + R[2, 2] * U0)
    rot_err = (abs(float(S_rot[1]) - float(Qexp)) + abs(float(S_rot[2]) - float(Uexp))) \
        / (abs(float(Qexp)) + abs(float(Uexp)))
    return thin_err, abs_err, rot_err


if __name__ == "__main__":
    print("=== 1. Magnus expansion vs exact chain ===")
    e1, e2 = test_magnus()
    print(f"  order-1 err = {e1:.2e}   order-2 err = {e2:.2e}")

    print("=== 2. Faraday conversion (cold plasma) ===")
    rV, rU, qv, rs = test_conversion()
    print(f"  rV(1 GHz) = {rV:.3e} cm^-1, rU(1 GHz) = {rU:.3e} cm^-1")
    print(f"  rho_Q/rho_V = {qv:.3e},  nu-scaling ratio = {rs:.4f} (expect 0.5)")

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
