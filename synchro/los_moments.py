"""Moment-driven line-of-sight: build transfer coefficients from moments.

This is the end-to-end junction of the RT layer: given the electron-distribution
moments (M0, M1, M2, <1/gamma>), the thermal density n_e and field components
(B_par, B_perp), build the emissivity/absorption (Kirchhoff closure) and the
Faraday rotation/conversion (cold plasma) and propagate a uniform slab through
synchro.transfer.

The emissivity/absorption are in *relative* units (Kirchhoff prefactors c^2/8pi
omitted); the source function S = j/alpha and the transfer I = S(1 - e^-tau) are
self-consistent in these units.  Polarised absorption (alpha_Q, alpha_V) is not
yet moment-expanded and is set to zero (deferred; see RT_FRAMEWORK §2.2).
"""

from __future__ import annotations

import jax.numpy as jnp

from .kirchhoff import emissivity_from_moments, absorption_from_moments
from .transfer import mueller_matrix, transfer_los
from .conversion import mueller_rotation, mueller_conversion


def moment_driven_slab(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma,
                       L, n_slabs=64, Q0=0.0, U0=0.0,
                       n_e=None, B_par=None, B_perp=None):
    """Observed Stokes (I, Q, U, V) of a uniform slab from moments only.

    Parameters
    ----------
    nu : frequency (arbitrary unit consistent with nu_c_ref)
    gamma0, nu_c_ref : reference Lorentz factor and critical frequency
    M0, M1, M2 : emissivity moments int N (gamma-gamma0)^k dgamma
    inv_gamma : <1/gamma> (inverse moment, for the Kirchhoff closure)
    L : slab length; n_slabs : slab discretisation
    Q0, U0 : intrinsic linear polarisation fractions (eps_Q = Q0*j etc.)
    n_e, B_par, B_perp : thermal density [cm^-3] and field [G] for
        Faraday rotation/conversion (optional).
    """
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    a = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)

    eps = jnp.array([j, Q0 * j, U0 * j, 0.0])
    rV = mueller_rotation(nu, n_e, B_par) if (n_e is not None and B_par is not None) else 0.0
    rU = mueller_conversion(nu, n_e, B_perp) if (n_e is not None and B_perp is not None) else 0.0
    # aI = a, polarised absorption = 0, rotation rV, conversion rU
    K = mueller_matrix(a, 0.0, 0.0, 0.0, 0.0, rU, rV)

    S0 = jnp.zeros(4)
    ds = L / n_slabs
    eps_s = jnp.stack([eps] * n_slabs)
    K_s = jnp.stack([K] * n_slabs)
    return transfer_los(S0, eps_s, K_s, ds)
