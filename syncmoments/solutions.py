"""Limiting solutions of the polarised transfer equation (moment-expanded).

These are the analytic benchmarks the full transfer in :mod:`syncmoments.transfer`
must reproduce, expressed in the moment/cumulant language of the paper.
"""

from __future__ import annotations


from .rm import burn_depolarisation
from .transfer import optical_depth_factor


def thin_limit(eps, ds):
    """Optically thin, rotation-free limit: S = eps ds."""
    return eps * ds


def self_absorbed_source_function(nu, p, B):
    """Power-law synchrotron source function S_nu ~ nu^(5/2) B^(-1/2).

    Up to an overall constant; ``p`` only affects the (omitted) prefactor, not
    the scaling.
    """
    return nu**2.5 * B**-0.5


def power_law_emissivity(nu, p, B):
    """Optically-thin emissivity scaling ~ nu^(-(p-1)/2) B^((p+1)/2)."""
    return nu ** (-(p - 1.0) / 2.0) * B ** ((p + 1.0) / 2.0)


def power_law_absorption(nu, p, B):
    """Synchrotron self-absorption scaling ~ nu^(-(p+4)/2) B^((p+2)/2)."""
    return nu ** (-(p + 4.0) / 2.0) * B ** ((p + 2.0) / 2.0)


def uniform_source_intensity(nu, p, B, L, eps0=1.0, alpha0=1.0):
    """I_nu = S_nu (1 - e^-tau) for a uniform source of depth L.

    Returns I_nu up to overall scale; ``eps0``/``alpha0`` set the relative
    emissivity/absorption amplitude at nu=1.
    """
    eps = eps0 * power_law_emissivity(nu, p, B)
    alpha = alpha0 * power_law_absorption(nu, p, B)
    tau = alpha * L
    return eps * L * optical_depth_factor(tau)


def thin_rotation_depolarisation(Q0, U0, mean_rm, var_rm, lam):
    """Thin + Faraday rotation: complex polarisation after Gaussian RM.

    Wrapper around :func:`syncmoments.rm.burn_depolarisation`; returns (Q, U).
    """
    P = burn_depolarisation(Q0 + 1j * U0, mean_rm, var_rm, lam)
    return P.real, P.imag
