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
``syncmoments.faraday``; its depths are measured from each emitter to the observer.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from .constants import E_ESU, M_E, C_CGS

C_RM_CGS = E_ESU**3 / (2.0 * np.pi * M_E**2 * C_CGS**4)  # rad cm^-2 per (G/cm)
RM_PER_UNIT = 0.812  # practical Faraday constant, rad m^-2 per (pc cm^-3 uG)


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
    """Position-angle Faraday rotation chi = RM * lam^2 [rad]; ``lam`` in cm,
    ``path_cgs`` = int n_e B_par ds in [G/cm]."""
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


rm_moments = gaussian_rm_cumulants  # historical name, same function object


_WIDE, _WIDE_BIAS = jnp.float64, 1023  # weights are normalised in float64 (x64)
_HEADROOM = 64  # tangent headroom of _normalised_weights, see _headroom_shift
_INVALID_WEIGHTS = "weights must be finite, nonnegative and have positive total mass"
_SUBNORMAL_MAX = (
    "weights: the largest weight is below the smallest normal float64 (2^-1022, "
    "about 2.2e-308), which XLA CPU flushes to zero; rescale the weights"
)


def _float_layout(dtype):
    """``(integer dtype of the same width, mantissa bits, exponent bias)``."""
    info = jnp.finfo(dtype)
    itype = {16: jnp.int16, 32: jnp.int32, 64: jnp.int64}.get(info.bits)
    if itype is None:
        raise ValueError(f"unsupported float dtype {dtype} (use 16, 32 or 64 bits)")
    return itype, info.nmant, info.maxexp - 1


def _pow2(k, dtype):
    """``2.0**k`` for integer ``k`` in the normal range ``[1 - bias, bias]`` (from bits)."""
    itype, nmant, bias = _float_layout(dtype)
    exponent = (jnp.asarray(k) + bias).astype(itype)
    return jax.lax.bitcast_convert_type(exponent << nmant, dtype)


def _times_pow2(x, k):
    """``x * 2**k``, integer ``k`` in ``[2 (1 - bias), 2 bias]``, by two normal
    factors: no spurious over/underflow; exact unless the result is subnormal."""
    half = k // 2
    return x * _pow2(half, x.dtype) * _pow2(k - half, x.dtype)


def _weights_invalid(w):
    """True unless ``w`` is finite, sign-bit nonnegative, with a positive entry; reads
    the bits (on XLA CPU ``-5e-324 < 0`` is False, a mask accepted it under JIT)."""
    itype = _float_layout(w.dtype)[0]
    bits = jax.lax.bitcast_convert_type(w, itype)
    negative = (bits < 0) & (bits != jnp.iinfo(itype).min)  # -0.0: a zero weight
    return jnp.any(~jnp.isfinite(w)) | jnp.any(negative) | (jnp.max(bits) <= 0)


def _scaled_weights(w):
    """``(w / 2^top, top)``, ``top = floor(log2 max w)``, in float64 from the bits
    ``sig 2^e`` of 16 to 64-bit ``w`` (subnormals included); peak in ``[1, 2)``."""
    itype, nmant, bias = _float_layout(w.dtype)
    magnitude = jnp.maximum(jax.lax.bitcast_convert_type(w, itype), 0)  # -0.0 -> 0
    field = (magnitude >> nmant).astype(jnp.int32)
    sig = (magnitude & ((1 << nmant) - 1)).astype(jnp.int64)
    sig = jnp.where(field > 0, sig | (1 << nmant), sig).astype(_WIDE)
    e = jnp.maximum(field, 1) - (bias + nmant)  # w = sig * 2^e exactly
    log2 = jnp.where(sig > 0, jnp.frexp(sig)[1] - 1 + e, -(bias + nmant + 1))
    top = jnp.max(log2)
    k = e - top
    floor = 2 * (1 - _WIDE_BIAS)
    return jnp.where(k >= floor, _times_pow2(sig, jnp.maximum(k, floor)), 0.0), top


def _headroom_shift(top, headroom):
    """``s = clip(-top - (1022 - headroom), 0, headroom)``: the tangent factor
    ``2^-(top + s)`` stays below ``2^(1022 - headroom)``; 0 for larger ``max(w)``."""
    return jnp.clip(-top - (_WIDE_BIAS - 1 - headroom), 0, headroom)


@functools.partial(jax.custom_jvp, nondiff_argnums=(1,))
def _relative_weights_core(w, headroom):
    """``w / max(w) * 2^-s`` in float64, ``s = _headroom_shift(top, headroom)``."""
    scaled, top = _scaled_weights(w)
    return scaled / jnp.max(scaled) * _pow2(-_headroom_shift(top, headroom), _WIDE)


@_relative_weights_core.defjvp
def _relative_weights_jvp(headroom, primals, tangents):
    (w,), (dw,) = primals, tangents
    scaled, top = jax.lax.stop_gradient(_scaled_weights(w))
    out = _relative_weights_core(w, headroom)  # re-entered for higher orders
    shift = top + _headroom_shift(top, headroom)
    return out, _times_pow2(dw.astype(out.dtype), -shift) / jnp.max(scaled)


def _relative_weights_wide(w, headroom=0):
    """``w / max(w) 2^-s`` in float64 (``_headroom_shift``); refuses subnormal max."""
    w = jnp.asarray(w)
    return _relative_weights_core(
        eqx.error_if(w, _scaled_weights(w)[1] < 1 - _WIDE_BIAS, _SUBNORMAL_MAX),
        headroom,
    )


def _relative_weights(w):
    """``w / max(w)`` in the dtype of 1D ``w`` (1 to one ulp at max), scaled in float64
    by the exact ``2^-floor(log2 max w)``, so no subnormal reciprocal forms (jax
    0.10.2 CPU: ``5e307 / 1e308 = 0`` via ``1e-308``). Ratios below ``2^-1022``
    flush (at most ``n 2^-1022`` of the mass); a subnormal maximum is refused.
    Bit-identical to v0.2.0's ``w / max(w)`` where that stayed normal. The scale
    is a constant of differentiation: the tangent is ``dw / max(w)``."""
    w = jnp.asarray(w)
    return _relative_weights_wide(w).astype(w.dtype)


@jax.custom_jvp
def _sum_tangent(r, R, dr):
    """JVP of ``r / R``, ``R = sum(r)``, with the plain division JVP's operations
    (bit-identical); reverse mode transposes it, forward mode uses the rule."""
    dR = jnp.sum(dr)  # this order reproduces the division's JVP bits
    return dr / R + -dR * r * jax.lax.integer_pow(R, -2)


@_sum_tangent.defjvp
def _sum_tangent_jvp(primals, tangents):
    """``d2p[dr, y] = -((dr - p dR) Y + (y - p Y) dR) / R^2`` (``Y = sum y``), rank 2:
    with ``x = dr 2^-k`` (``|x| < 1``), ``Rb = R 2^s`` (in ``[1, n]``) it is
    ``-2^(k+2s) ((x - p X) Y + (y - p Y) X) / Rb^2``: summed, then scaled exactly."""
    (r, R, dr), (y, _, ddr) = primals, tangents
    s = 1 - jnp.frexp(jnp.max(r))[1]  # max(r) 2^s in [1, 2)
    k = jnp.frexp(jnp.max(jnp.abs(dr)))[1]  # |dr| < 2^k
    x, p, Rb = _times_pow2(dr, -k), r / R, _times_pow2(R, s)
    X, Y = jnp.sum(x), jnp.sum(y)
    S = ((x - p * X) * Y + (y - p * Y) * X) / Rb**2
    e = jnp.clip(k + 2 * s, 2 * (1 - _WIDE_BIAS), 2 * _WIDE_BIAS)
    return _sum_tangent(r, R, dr), _sum_tangent(r, R, ddr) - _times_pow2(S, e)


@jax.custom_jvp
def _divide_by_sum(r):
    """``r / sum(r)`` for float64 ``r`` with positive sum; JVP ``_sum_tangent``."""
    return r / jnp.sum(r)


@_divide_by_sum.defjvp
def _divide_by_sum_jvp(primals, tangents):
    (r,), (dr,), R = primals, tangents, jnp.sum(primals[0])  # as the division's JVP
    if r.shape[-1] == 1:  # p = 1: every derivative is zero (x - x = +0, as before)
        return r / R, dr - dr
    return r / R, _sum_tangent(r, R, dr)


@eqx.filter_jit
def _normalised_weights(weights):
    """Validated ``w / sum(w)`` in float64 (one compiled helper per shape/dtype).
    For ``max(w) < 2^-958`` relative weights and tangents ``dw / max(w)`` carry an
    exact ``2^-s``, ``s <= 64`` (ratios down to ``2^-1022`` kept), so the summed
    tangent overflows only where the derivative of ``w / sum(w)`` nearly does.
    Second derivatives: ``_sum_tangent``. Eager refusals: ``EquinoxRuntimeError``."""
    weights = eqx.error_if(weights, _weights_invalid(weights), _INVALID_WEIGHTS)
    # Barrier: XLA must not reassociate back to the overflowing raw weight sum.
    scaled = jax.lax.optimization_barrier(_relative_weights_wide(weights, _HEADROOM))
    return _divide_by_sum(scaled)


@eqx.filter_jit  # the string ``sample_name`` is static
def _validated_samples(rms, weights, sample_name):
    """Finite-sample check and normalised weights (``None``: uniform), one
    compiled call per shape, dtype and name, as ``_normalised_weights``."""
    message = f"{sample_name} samples must be finite"
    rms = eqx.error_if(rms, jnp.any(~jnp.isfinite(rms)), message)
    if weights is None:
        return rms, jnp.ones_like(rms) / rms.size
    return rms, _normalised_weights(weights).astype(rms.dtype)


def _screen_samples(rms, weights, *, sample_name="RM"):
    """Shared real-sample validation and overflow-safe normalised weights: float
    (16 to 64 bits) or integer weights are normalised in float64 and returned in
    the samples' dtype. Shape and dtype errors raise ``ValueError`` before tracing."""
    rms = jnp.asarray(rms)
    if rms.ndim != 1 or rms.size == 0:
        raise ValueError(f"{sample_name} samples must be a nonempty 1D array")
    if jnp.iscomplexobj(rms):
        raise ValueError(f"{sample_name} samples must be real")
    rms = rms.astype(jnp.result_type(rms, 1.0))
    if weights is not None:
        weights = jnp.asarray(weights)
        if jnp.iscomplexobj(weights):
            raise ValueError("weights must be real")
        if weights.shape != rms.shape:
            raise ValueError(f"weights must match the 1D {sample_name} samples")
        weights = weights.astype(jnp.result_type(weights, 1.0))
        _float_layout(weights.dtype)  # refuse unsupported widths
    # Exact power-of-two rescaling (_relative_weights): no overflowing weight sum.
    return _validated_samples(rms, weights, sample_name)


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
