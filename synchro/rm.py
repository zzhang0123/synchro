"""Rotation measure (RM) and conversion measure (CM): constants and moments.

Defines the Faraday rotation coefficient and its moment-expansion content.

Conventions (CGS, Gaussian units):
    chi  = RM * lambda^2            position-angle rotation (radians)
    RM   = C_RM * int n_e B_par ds   [rad cm^-2 in CGS; see rm_rad_per_m2]
    C_RM = e^3 / (2 pi m_e^2 c^4)

The rounded practical constant is RM[rad/m^2] = 0.812 * int n_e[cm^-3]
B_par[uG] d(s/pc); use the CGS conversion for the unrounded coefficient.

For an external Gaussian RM screen with a common incident complex polarization
(or one independent of RM), P = Q + i U is averaged as
    <P> = P_0 exp(2 i <RM> lam^2) exp(-2 Var(RM) lam^4).
Weighted mean and variance alone do not establish that Gaussian closure.
Distributed emission inside a rotating slab has a different transfer law.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .constants import E_ESU, M_E, C_CGS

# e^3 / (2 pi m_e^2 c^4)   [rad cm^-2 per (G/cm)]
C_RM_CGS = E_ESU**3 / (2.0 * np.pi * M_E**2 * C_CGS**4)

# practical Faraday constant
RM_PER_UNIT = 0.812  # rad m^-2 per (pc cm^-3 uG)


def rotation_measure_rad_m2(path_cgs: ArrayLike) -> jax.Array:
    """RM [rad/m^2] from the CGS path integral ``int n_e B_par ds`` [G/cm]."""
    return C_RM_CGS * jnp.asarray(path_cgs) * 1e4  # cm^-2 -> m^-2


def rotation_measure_practical(
    n_e_cm3: ArrayLike, B_par_uG: ArrayLike, s_pc: ArrayLike
) -> float:
    """NumPy path quadrature with the rounded coefficient0.812 [rad/m^2].

    Supply matching finite one-dimensional samples on an increasing path, with
    nonnegative density. This preprocessing function is not JIT/autodiff code;
    path quadrature error and rounding of the practical coefficient remain.
    """
    density, field, path = (
        np.asarray(value, dtype=float) for value in (n_e_cm3, B_par_uG, s_pc)
    )
    if (
        path.ndim != 1
        or path.size < 2
        or density.shape != path.shape
        or field.shape != path.shape
        or not all(np.all(np.isfinite(value)) for value in (density, field, path))
        or np.any(density < 0)
        or np.any(np.diff(path) <= 0)
    ):
        raise ValueError(
            "require finite matching 1D path samples, nonnegative density and increasing positions"
        )
    return float(RM_PER_UNIT * np.trapezoid(density * field, path))


def rotation_angle(lam: ArrayLike, path_cgs: ArrayLike) -> jax.Array:
    """Position-angle Faraday rotation chi = RM * lam^2 [rad].

    ``lam`` in cm; ``path_cgs`` = int n_e B_par ds in [G/cm].
    """
    return C_RM_CGS * jnp.asarray(lam) ** 2 * jnp.asarray(path_cgs)


def burn_depolarisation(
    P0: ArrayLike, mean_rm: ArrayLike, var_rm: ArrayLike, lam: ArrayLike
) -> jax.Array:
    """Average P = Q + i U through a specified external Gaussian RM screen.

    Gaussian RM (mean ``mean_rm`` [rad/m^2], variance ``var_rm`` [rad^2/m^4]):

        <P(lam)> = P0 * exp(2 i mean_rm lam^2) * exp(-2 var_rm lam^4)

    ``P0`` may be complex (Q0 + i U0), common to the rays or independent of
    RM; ``lam`` is in metres. Inputs must be finite and broadcast-compatible,
    with var_rm>=0. A non-Gaussian screen instead requires its characteristic
    function and a discrepancy for the Gaussian closure. This expression does
    not include internally distributed emission, absorption or conversion.
    """
    P0, mean_rm, var_rm, lam = jnp.broadcast_arrays(P0, mean_rm, var_rm, lam)
    if any(jnp.iscomplexobj(value) for value in (mean_rm, var_rm, lam)):
        raise ValueError("RM mean, variance and wavelength must be real")
    invalid = (
        jnp.any(~jnp.isfinite(P0))
        | jnp.any(~jnp.isfinite(mean_rm))
        | jnp.any(~jnp.isfinite(var_rm))
        | jnp.any(~jnp.isfinite(lam))
        | jnp.any(var_rm < 0)
    )
    P0 = eqx.error_if(
        P0, invalid, "Gaussian screen requires finite inputs and nonnegative variance"
    )
    return P0 * jnp.exp(2j * mean_rm * lam**2) * jnp.exp(-2.0 * var_rm * lam**4)


def gaussian_rm_cumulants(
    rms: ArrayLike, weights: ArrayLike | None = None
) -> tuple[jax.Array, jax.Array]:
    """First two cumulants of a sampled RM distribution (rad/m^2).

    Returns population (mean_rm, var_rm), without a sample-unbiased correction.
    The legacy name does not assert Gaussianity. These two statistics alone
    do not determine the external-screen characteristic function. Samples and
    optional weights are finite matching 1D arrays; weights are nonnegative
    with positive total mass. This function supports JIT and differentiation.
    """
    rms = jnp.asarray(rms)
    if rms.ndim != 1 or rms.size == 0:
        raise ValueError("RM samples must be a nonempty 1D array")
    if jnp.iscomplexobj(rms):
        raise ValueError("RM samples must be real")
    rms = rms.astype(jnp.result_type(rms, 1.0))
    rms = eqx.error_if(rms, jnp.any(~jnp.isfinite(rms)), "RM samples must be finite")
    if weights is None:
        w = jnp.ones_like(rms) / rms.size
    else:
        weights = jnp.asarray(weights)
        if jnp.iscomplexobj(weights):
            raise ValueError("weights must be real")
        weights = weights.astype(jnp.result_type(rms, 1.0))
        if weights.shape != rms.shape:
            raise ValueError("weights must match the 1D RM samples")
        weights = eqx.error_if(
            weights,
            jnp.any(~jnp.isfinite(weights))
            | jnp.any(weights < 0)
            | (jnp.max(weights) <= 0),
            "weights must be finite, nonnegative and have positive total mass",
        )
        # Relative weights may have an overflowing unscaled sum. Rescaling
        # changes neither the normalized distribution nor its derivatives.
        # Preserve this intermediate under XLA: reassociating the two divides
        # can recreate the overflowing original weight sum inside JIT.
        scaled = jax.lax.optimization_barrier(
            weights / jax.lax.stop_gradient(jnp.max(weights))
        )
        w = scaled / jnp.sum(scaled)
    mean = jnp.sum(w * rms)
    var = jnp.sum(w * (rms - mean) ** 2)
    moments = jnp.stack([mean, var])
    moments = eqx.error_if(
        moments, jnp.any(~jnp.isfinite(moments)), "RM moment arithmetic overflowed"
    )
    return moments[0], moments[1]
