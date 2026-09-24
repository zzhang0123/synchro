"""Rotation measure (RM) and conversion measure (CM): constants and moments.

Defines the Faraday rotation coefficient and its moment-expansion content.

Conventions (CGS, Gaussian units)::

    chi  = RM * lambda^2            position-angle rotation (radians)
    RM   = C_RM * int n_e B_par ds   [rad cm^-2 in CGS; see rm_rad_per_m2]
    C_RM = e^3 / (2 pi m_e^2 c^4)

The rounded practical constant is RM[rad/m^2] = 0.812 * int n_e[cm^-3]
B_par[uG] d(s/pc); use the CGS conversion for the unrounded coefficient.

For an external Gaussian RM screen with a common incident complex polarization
(or one independent of RM), P = Q + i U is averaged as::

    <P> = P_0 exp(2 i <RM> lam^2) exp(-2 Var(RM) lam^4).

Weighted mean and variance alone do not establish that Gaussian closure.
Distributed emission is implemented by the joint emission-depth average in
``synchro.faraday``; its depths are measured from each emitter to the observer.
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
    return float(faraday_depth_practical(n_e_cm3, B_par_uG, s_pc)[0])


def faraday_depth_practical(
    n_e_cm3: ArrayLike, B_par_uG: ArrayLike, s_pc: ArrayLike
) -> np.ndarray:
    """Depth from each path node to the observer, in rad/m^2 (NumPy).

    Positions increase towards the observer at the last node. Integrate
    ``0.812 * n_e * B_parallel`` from each node to that endpoint by trapezoids;
    positive field points towards the observer. Field reversals and nonmonotone
    depth are allowed. Add a separately supplied foreground depth to every node
    if the path stops before the observer. No monotonic-depth inversion is used.

    Matching finite 1D arrays, nonnegative density and strictly increasing
    positions are required. This is preprocessing, not JIT/autodiff code.
    Grid quadrature and the rounded coefficient need separate error control.
    These depths must be paired with emission at the same nodes, not treated
    as a distribution of total sightline RMs.
    """
    arrays = tuple(np.asarray(value) for value in (n_e_cm3, B_par_uG, s_pc))
    if any(np.iscomplexobj(value) for value in arrays):
        raise ValueError("density, field and path must be real")
    density, field, path = (value.astype(float) for value in arrays)
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
    with np.errstate(over="ignore", invalid="ignore"):
        integrand = density * field
        cells = RM_PER_UNIT * (integrand[:-1] / 2 + integrand[1:] / 2) * np.diff(path)
        depths = np.r_[np.cumsum(cells[::-1])[::-1], 0.0]
    if not np.all(np.isfinite(depths)):
        raise ValueError("Faraday-depth quadrature overflowed")
    return depths


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
    rms, w = _screen_samples(rms, weights)
    mean = jnp.sum(w * rms)
    var = jnp.sum(w * (rms - mean) ** 2)
    moments = jnp.stack([mean, var])
    moments = eqx.error_if(
        moments, jnp.any(~jnp.isfinite(moments)), "RM moment arithmetic overflowed"
    )
    return moments[0], moments[1]


# Descriptive name; preserve the historical API and function identity.
rm_moments = gaussian_rm_cumulants


def _screen_samples(rms, weights, *, sample_name="RM"):
    """Shared real-sample validation and overflow-safe relative weights."""
    rms = jnp.asarray(rms)
    if rms.ndim != 1 or rms.size == 0:
        raise ValueError(f"{sample_name} samples must be a nonempty 1D array")
    if jnp.iscomplexobj(rms):
        raise ValueError(f"{sample_name} samples must be real")
    rms = rms.astype(jnp.result_type(rms, 1.0))
    rms = eqx.error_if(
        rms, jnp.any(~jnp.isfinite(rms)), f"{sample_name} samples must be finite"
    )
    if weights is None:
        w = jnp.ones_like(rms) / rms.size
    else:
        weights = jnp.asarray(weights)
        if jnp.iscomplexobj(weights):
            raise ValueError("weights must be real")
        weights = weights.astype(jnp.result_type(rms, 1.0))
        if weights.shape != rms.shape:
            raise ValueError(f"weights must match the 1D {sample_name} samples")
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
    return rms, w


def screen_polarisation(
    P0: ArrayLike,
    rms: ArrayLike,
    lam: ArrayLike,
    weights: ArrayLike | None = None,
) -> jax.Array:
    """Average ``P0 * exp(2j * RM * lam**2)`` over a discrete external screen.

    RM samples are a nonempty 1D array in rad/m^2, wavelengths in metres.
    ``P0`` is a complex scalar shared by rays or a matching 1D array allowing
    correlation with RM. Wavelengths can have any shape; the output has that
    shape, with no implicit ray/frequency broadcasting. Weights are finite,
    nonnegative relative masses, normalized internally. All inputs must be
    finite; RM and wavelengths must be real. JIT and differentiation are
    supported. A scan uses O(number of wavelengths) working storage in forward
    evaluation; reverse-mode AD may retain per-ray intermediates.

    This is the supplied discrete-screen average, not a Gaussian closure or
    a cumulant truncation. Sampling/quadrature error and physical screen-model
    error remain external inputs. It excludes internal emission, absorption
    and conversion. For fixed normalized weights w, perturbations obey::

        |delta P| <= sum(w*|delta P0|)
                     + 2*lam^2*sum(w*|P0|*|delta RM|).

    Changed normalized weights additionally contribute
    ``max(|P0|)*sum(|delta w|)``, evaluated consistently at the intermediate screen.
    These are input-error bounds, not floating-point or sampling certificates.
    """
    rms, w = _screen_samples(rms, weights)
    incident, lam = jnp.asarray(P0), jnp.asarray(lam)
    if incident.ndim != 0 and incident.shape != rms.shape:
        raise ValueError("P0 must be scalar or match the 1D RM samples")
    if jnp.iscomplexobj(lam):
        raise ValueError("wavelengths must be real")
    incident = eqx.error_if(
        incident, jnp.any(~jnp.isfinite(incident)), "P0 must be finite"
    )
    lam = lam.astype(jnp.result_type(lam, 1.0))
    lam = eqx.error_if(lam, jnp.any(~jnp.isfinite(lam)), "wavelengths must be finite")
    lam2 = lam**2
    lam2 = eqx.error_if(lam2, jnp.any(~jnp.isfinite(lam2)), "screen phase overflowed")
    incident = jnp.broadcast_to(incident, rms.shape)
    dtype = jnp.result_type(incident, rms, lam, 1j)

    def accumulate(total, ray):
        rm, amplitude, weight = ray
        phase = (2 * lam2) * rm
        phase = eqx.error_if(
            phase, jnp.any(~jnp.isfinite(phase)), "screen phase overflowed"
        )
        return total + weight * amplitude * jnp.exp(1j * phase), None

    result, _ = jax.lax.scan(
        accumulate, jnp.zeros(lam.shape, dtype=dtype), (rms, incident, w)
    )
    return eqx.error_if(
        result, jnp.any(~jnp.isfinite(result)), "screen average overflowed"
    )
