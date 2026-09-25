"""Local moment approximations connected to uniform-slab transfer.

The reduced demonstration and physical CGS interface are deliberately explicit.
Both use a local second-order energy-kernel Taylor approximation, whose error
must be assessed independently. They do not implement the manuscript's positive
finite-cumulant PDF reconstruction. The continuum wrappers assume isotropic UR
pitch angles; intrinsic V and alphaV are omitted here. Their omission requires
its own propagated error, while generic transfer.mueller_matrix retains alphaV.
"""

from __future__ import annotations

import numbers

import equinox as eqx
import jax.numpy as jnp

from .kirchhoff import (
    emissivity_from_moments,
    absorption_from_moments,
    emissivity_Q_from_moments,
    absorption_Q_from_moments,
    cgs_coefficients_from_moments,
)
from .transfer import mueller_matrix, transfer_slab
from .conversion import mueller_rotation, mueller_conversion


def moment_driven_slab(
    nu,
    gamma0,
    nu_c_ref,
    M0,
    M1,
    M2,
    inv_gamma,
    L,
    n_slabs=64,
    Q0=0.0,
    U0=0.0,
    n_e=None,
    B_par=None,
    B_perp=None,
    polarised_absorption=False,
):
    """Reduced-units demonstration; use moment_driven_slab_cgs for physical RT.

    nu, nu_c_ref and L have the reduced coefficient units of kirchhoff.py.
    Physical Faraday parameters are rejected because those require Hz/cm/CGS
    emission and absorption too. With polarised_absorption=True, signed jQ
    and alphaQ derive from the same moments; otherwise Q0,U0 are prescribed
    fractions. That switch and n_slabs are static configuration. n_slabs is
    retained for API compatibility; a uniform slab needs one exponential.
    Default absorption closure assumes vanishing energy endpoint terms.
    """
    if any(x is not None for x in (n_e, B_par, B_perp)):
        raise ValueError(
            "physical Faraday inputs require moment_driven_slab_cgs (Hz, Gauss, cm)"
        )
    if not isinstance(n_slabs, int) or isinstance(n_slabs, bool) or n_slabs < 1:
        raise ValueError("n_slabs must be a positive static integer")
    if polarised_absorption:
        if isinstance(Q0, numbers.Real) and isinstance(U0, numbers.Real):
            if Q0 != 0 or U0 != 0:
                raise ValueError("polarised_absorption=True requires Q0=U0=0")
        Q0 = eqx.error_if(
            jnp.asarray(Q0),
            (jnp.asarray(Q0) != 0) | (jnp.asarray(U0) != 0),
            "polarised_absorption=True requires Q0=U0=0",
        )
    j = emissivity_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2)
    a = absorption_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
    if polarised_absorption:
        q = emissivity_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2) + 0 * Q0
        aq = absorption_Q_from_moments(nu, gamma0, nu_c_ref, M0, M1, M2, inv_gamma)
        u = 0.0
    else:
        q, u, aq = Q0 * j, U0 * j, 0.0
    eps = jnp.array([j, q, u, 0.0])
    K = mueller_matrix(a, aq, 0.0, 0.0, 0.0, 0.0, 0.0)
    return transfer_slab(jnp.zeros(4), eps, K, L)


def moment_driven_slab_cgs(
    nu_hz,
    gamma0,
    B_perp_gauss,
    M0,
    M1,
    M2,
    inv_gamma,
    L_cm,
    *,
    n_e=0.0,
    B_par=0.0,
    phi=0.0,
    S0=None,
    boundary_terms=(0.0, 0.0, 0.0),
):
    """Physical directional continuum Stokes of one uniform slab in CGS.

    nu_hz: Hz; fields: Gauss; thermal n_e and electron moments: cm^-3;
    L_cm: cm; phi: projected-field azimuth in the fixed observer basis.
    S0 and output: erg/s/cm^2/Hz/sr. The signed natural rQ mixes U and V;
    rotating the projected field gives rQ*cos(2phi), rQ*sin(2phi).
    See kirchhoff.cgs_coefficients_from_moments for approximation/endpoints.
    Cold-plasma Faraday coefficients require high-frequency validity and are
    independent of the relativistic emitting-electron moments.
    """
    plasma = jnp.asarray([L_cm, n_e, B_par, phi])
    plasma = eqx.error_if(
        plasma,
        jnp.any(~jnp.isfinite(plasma)) | (plasma[0] < 0) | (plasma[1] < 0),
        "cgs slab needs finite L_cm>=0, n_e>=0, B_par, phi",
    )
    L_cm, n_e, B_par, phi = plasma
    jI, jQ, aI, aQ = cgs_coefficients_from_moments(
        nu_hz,
        gamma0,
        B_perp_gauss,
        M0,
        M1,
        M2,
        inv_gamma,
        boundary_terms=boundary_terms,
    )
    co, si = jnp.cos(2 * phi), jnp.sin(2 * phi)
    rV = mueller_rotation(nu_hz, n_e, B_par)
    rQ = mueller_conversion(nu_hz, n_e, B_perp_gauss)
    eps = jnp.array([jI, jQ * co, jQ * si, 0.0])
    K = mueller_matrix(aI, aQ * co, aQ * si, 0.0, rQ * co, rQ * si, rV)
    return transfer_slab(jnp.zeros(4) if S0 is None else S0, eps, K, L_cm)
