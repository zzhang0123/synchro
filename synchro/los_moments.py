"""Moment-driven line-of-sight: build transfer coefficients from moments.

This is the end-to-end junction of the RT layer: given the electron-distribution
moments (M0, M1, M2, <1/gamma>), the thermal density n_e and field components
(B_par, B_perp), build the emissivity/absorption (Kirchhoff closure) and the
Faraday rotation/conversion (cold plasma) and propagate a uniform slab through
synchro.transfer.

The emissivity/absorption are in *relative* units (Kirchhoff prefactors c^2/8pi
omitted); the source function S = j/alpha and the transfer I = S(1 - e^-tau) are
self-consistent in these units.

Boundary: this is a JAX/differentiable module (it takes moments as scalars and
returns a jnp array); the gamma-integrals that produce the moments live on the
NumPy precompute side in ``synchro.kirchhoff``.

Two polarisation modes, selected by ``polarised_absorption``:

* ``False`` (default) -- the intrinsic linear polarisation is imposed by hand
  through ``Q0``/``U0`` and only the Stokes-I absorption is propagated
  (alpha_Q = alpha_U = alpha_V = 0).  Use this to study Faraday rotation of a
  prescribed intrinsic polarisation.
* ``True`` -- the Kirchhoff-consistent mode.  Both eps_Q and alpha_Q come from
  the *same* moments through the G-kernel of ``synchro.kirchhoff``, so the slab
  reproduces the perpendicular/parallel source functions and hence the correct
  self-absorbed polarisation fraction.  ``Q0``/``U0`` must then be left at zero,
  because an imposed polarisation fraction is not Kirchhoff-consistent with an
  absorption computed from the same distribution.

Circular absorption alpha_V is not included in either mode: it is suppressed by
O(1/gamma) ~ 1e-4 for the diffuse foreground (see the main paper, Sec. 6).
"""

from __future__ import annotations

import jax.numpy as jnp

from .kirchhoff import (emissivity_from_moments, absorption_from_moments,
                        emissivity_Q_from_moments, absorption_Q_from_moments)
from .transfer import mueller_matrix, transfer_los
from .conversion import mueller_rotation, mueller_conversion


def moment_driven_slab(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma,
                       L, n_slabs=64, Q0=0.0, U0=0.0,
                       n_e=None, B_par=None, B_perp=None,
                       polarised_absorption=False):
    """Observed Stokes (I, Q, U, V) of a uniform slab from moments only.

    Parameters
    ----------
    nu : frequency (arbitrary unit consistent with nu_c_ref)
    gamma0, nu_c_ref : reference Lorentz factor and critical frequency
    M0, M1, M2 : emissivity moments int N (gamma-gamma0)^k dgamma
    inv_gamma : <1/gamma> (inverse moment, for the Kirchhoff closure)
    L : slab length; n_slabs : slab discretisation
    Q0, U0 : intrinsic linear polarisation fractions (eps_Q = Q0*j etc.).
        Only used when ``polarised_absorption`` is False.
    n_e, B_par, B_perp : thermal density [cm^-3] and field [G] for
        Faraday rotation/conversion (optional).
    polarised_absorption : if True, take eps_Q and alpha_Q from the same
        moments (Kirchhoff-consistent, natural basis with U = 0) instead of
        imposing Q0/U0 with alpha_Q = 0.  See the module docstring.

    Raises
    ------
    ValueError
        If ``polarised_absorption`` is True while ``Q0`` or ``U0`` is non-zero,
        which would silently mix an imposed polarisation with a Kirchhoff-closed
        absorption.
    """
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    a = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)

    if polarised_absorption:
        if Q0 != 0.0 or U0 != 0.0:
            raise ValueError(
                "polarised_absorption=True derives eps_Q from the moments; "
                "leave Q0=U0=0 (an imposed polarisation fraction is not "
                "Kirchhoff-consistent with the absorption from the same "
                "distribution)."
            )
        # Natural basis: U = 0 by symmetry, so alpha_U = 0 as well.
        eps_Q = emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
        aQ = absorption_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2,
                                       inv_gamma)
        eps_U = 0.0
    else:
        eps_Q, eps_U = Q0 * j, U0 * j
        aQ = 0.0

    eps = jnp.array([j, eps_Q, eps_U, 0.0])
    rV = mueller_rotation(nu, n_e, B_par) if (n_e is not None and B_par is not None) else 0.0
    rU = mueller_conversion(nu, n_e, B_perp) if (n_e is not None and B_perp is not None) else 0.0
    # alpha_V = 0: suppressed by O(1/gamma) for the diffuse foreground.
    K = mueller_matrix(a, aQ, 0.0, 0.0, 0.0, rU, rV)

    S0 = jnp.zeros(4)
    ds = L / n_slabs
    eps_s = jnp.stack([eps] * n_slabs)
    K_s = jnp.stack([K] * n_slabs)
    return transfer_los(S0, eps_s, K_s, ds)
