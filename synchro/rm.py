"""Rotation measure (RM) and conversion measure (CM): constants and moments.

Defines the Faraday rotation coefficient and its moment-expansion content.

Conventions (CGS, Gaussian units):
    chi  = RM * lambda^2            position-angle rotation (radians)
    RM   = C_RM * int n_e B_par ds   [rad cm^-2 in CGS; see rm_rad_per_m2]
    C_RM = e^3 / (2 pi m_e^2 c^4)

The practical constant is RM[rad/m^2] = 0.812 * int n_e[cm^-3] B_par[uG] d(s/pc).

Faraday depolarisation is the *second cumulant* statement: for a Gaussian RM
distribution the complex polarisation P = Q + i U is damped as
    <P> = P_0 exp(2 i <RM> lam^2) exp(-2 Var(RM) lam^4).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

E_ESU = 4.80320427e-10  # elementary charge [esu]
M_E = 9.1093837e-28  # electron mass [g]
C_CGS = 2.99792458e10  # speed of light [cm/s]

# e^3 / (2 pi m_e^2 c^4)   [rad cm^-2 per (G/cm)]
C_RM_CGS = E_ESU**3 / (2.0 * np.pi * M_E**2 * C_CGS**4)

# practical Faraday constant
RM_PER_UNIT = 0.812  # rad m^-2 per (pc cm^-3 uG)


def rotation_measure_rad_m2(path_cgs):
    """RM [rad/m^2] from the CGS path integral ``int n_e B_par ds`` [G/cm]."""
    return C_RM_CGS * path_cgs * 1e4  # cm^-2 -> m^-2


def rotation_measure_practical(n_e_cm3, B_par_uG, s_pc):
    """RM [rad/m^2] = 0.812 * int n_e[cm^-3] B_par[uG] d(s/pc)."""
    return RM_PER_UNIT * np.trapezoid(n_e_cm3 * B_par_uG, s_pc)


def rotation_angle(lam, path_cgs):
    """Position-angle Faraday rotation chi = RM * lam^2 [rad].

    ``lam`` in cm; ``path_cgs`` = int n_e B_par ds in [G/cm].
    """
    return C_RM_CGS * lam**2 * path_cgs


def burn_depolarisation(P0, mean_rm, var_rm, lam):
    """Complex polarisation P = Q + i U after Faraday rotation + dispersion.

    Gaussian RM (mean ``mean_rm`` [rad/m^2], variance ``var_rm`` [rad^2/m^4]):

        <P(lam)> = P0 * exp(2 i mean_rm lam^2) * exp(-2 var_rm lam^4)

    ``P0`` may be complex (Q0 + i U0). ``lam`` in metres.
    """
    return P0 * jnp.exp(2j * mean_rm * lam**2) * jnp.exp(-2.0 * var_rm * lam**4)


def gaussian_rm_cumulants(rms, sigmas, weights=None):
    """First two cumulants of a sampled RM distribution (rad/m^2).

    Returns (mean_rm, var_rm).  Weights default to uniform.
    """
    rms = jnp.asarray(rms)
    if weights is None:
        w = jnp.ones_like(rms) / rms.size
    else:
        w = jnp.asarray(weights) / jnp.sum(weights)
    mean = jnp.sum(w * rms)
    var = jnp.sum(w * (rms - mean) ** 2)
    return mean, var
