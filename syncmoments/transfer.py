"""Polarised radiative transfer along a line of sight (JAX).

Solves  dS/ds = eps - K S,  S = (I, Q, U, V),  with the Mueller matrix

    K = [ aI  aQ  aU  aV ]
        [ aQ  aI  rV -rU ]
        [ aU -rV  aI  rQ ]
        [ aV  rU -rQ  aI ]

where a* are absorption coefficients and r* are Faraday rotation (rV) and
conversion (rQ, rU) coefficients.

Convention: with the rotation block [[0, rV], [-rV, 0]] the complex
polarisation P = Q + i U rotates as  P -> P exp(i rV ds)  over a length ds.
The physical Faraday position-angle rotation is  chi = RM lam^2, and the
Stokes rotation angle is 2 chi, so  rV ds = 2 RM lam^2  (i.e. rV = 2 dchi/ds).

The matrix exponential is the analytic constant-coefficient solution;
its implementation has finite-precision error. A LOS is a chain of slabs
in traversal order. Approximating a varying medium by slabs incurs a separate
discretisation error; refine the path and propagate that discrepancy.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
import numpy as np
from jax.scipy.linalg import expm

from ._expm import any_member as _any_member
from ._expm import expm_pade13
from .rm import _pow2


def mueller_matrix(aI, aQ, aU, aV, rQ, rU, rV):
    """Build the 4x4 Mueller matrix K from absorption and rotation coefficients."""
    rows = [[aI, aQ, aU, aV], [aQ, aI, rV, -rU], [aU, -rV, aI, rQ], [aV, rU, -rQ, aI]]
    return jnp.array(rows)


def transfer_slab(S, eps, K, ds, *, max_squarings=32):
    """Constant-coefficient propagation of shape-(4,) Stokes through ``ds``.

    Uses the augmented 5x5 exponential ``exp([[-K, eps], [0, 0]] ds) @ [S; 1]``
    (first four entries), well-defined for singular K (pure rotation). The
    source column is carried as ``eps ds 2^-k`` with ``[S; 2^k]``, exact since
    the map is linear in ``eps``; ``k = floor(log2 max|eps ds|) + 1`` clipped to
    ``[0, 1022]`` (float64), built from its bit pattern. Without the scale the
    scaling and squaring of ``expm`` overflows to NaN for an optically thick,
    bright slab; no reciprocal of a data-dependent scale is formed (jax 0.10.2
    CPU flushed the subnormal ``1/s`` to zero). Entries of ``eps ds`` below
    ``max|eps ds| 2^-1021`` become subnormal after scaling and flush to zero on
    XLA CPU, an absolute error below that bound.

    Derivatives use ``out = Phi S + G eps``, ``Phi = e^{-K ds}``,
    ``G = int_0^ds e^{-K t} dt``: the tangent ``Phi dS + G deps + dPhi S + dG eps``
    comes from one 8x8 exponential ``[[Phi, G / ds], [0, I]]`` and its forward
    derivative (``_tangent``), not through the scaled source column. The rule is
    linear in the tangents and re-enters itself for the primal, so ``jvp``,
    ``vjp``, ``jacfwd``, ``jacrev``, ``hessian``, ``vmap`` (reverse mode through
    a batched ``K`` or ``ds`` included), ``jit`` and ``lax.scan`` apply. Where
    JAX differentiates the value itself (the primal of a ``vjp`` under a further
    derivative; ``lax.scan`` under ``vjp``) it carries the same derivatives from
    exposure 896 on (``_slab_value``). Tested against SciPy and autodiff of
    ``Phi S + G eps`` to 1e-13 of the largest entry up to
    ``max|eps ds| = 1.7e308``, second derivatives in every nesting. A derivative
    beyond the float64 range overflows to inf without an error.

    ``max_squarings`` is static. The default permits strongly absorbing slabs
    (a regression checks tau=1e6); this does not certify arbitrary large or
    ill-conditioned K. The core is compiled once per shape (``filter_jit``).

    A non-finite result is refused (``equinox.error_if``; eagerly as
    ``EquinoxRuntimeError``, under ``jit``, ``vmap``, derivatives and inside
    ``transfer_los``) naming the cause: non-finite inputs, ``eps * ds`` beyond
    float64, more than ``max_squarings`` squarings (NaN), or a value beyond
    float64 (accumulated intensity or extreme gain). 0.2.0 returned NaN or inf.
    """
    S, eps, K, ds = map(jnp.asarray, (S, eps, K, ds))
    if S.shape != (4,) or eps.shape != (4,) or K.shape != (4, 4) or ds.ndim != 0:
        raise ValueError("shape contract: S and eps (4,), K (4,4), scalar ds")
    return _slab_compiled(S, eps, K, ds, _squaring_budget(max_squarings))


def _squaring_budget(max_squarings) -> int:
    """``max_squarings`` as a Python int: it is static, and numpy scalars would
    otherwise be traced. Integral floats are accepted as in 0.2.0."""
    if isinstance(max_squarings, (bool, np.bool_)):
        raise TypeError("max_squarings must be a non-negative integer, got a bool")
    if isinstance(max_squarings, (int, np.integer)):
        value = int(max_squarings)
    elif (
        isinstance(max_squarings, (float, np.floating))
        and float(max_squarings).is_integer()
    ):
        value = int(max_squarings)
    else:
        raise TypeError(
            f"max_squarings must be a non-negative integer, got {max_squarings!r}"
        )
    if value < 0:
        raise ValueError(f"max_squarings must be a non-negative integer, got {value}")
    return value


@eqx.filter_jit
def _slab_compiled(S, eps, K, ds, max_squarings):
    """One compilation per shape (an eager ``lax.cond`` recompiled every call)."""
    return _slab(S, eps, K, ds, max_squarings, None)


# Under vmap equinox reports the message of the largest index in the batch, so
# members that pass carry index 0 and the causes are ordered by priority.
_REFUSALS = (
    "transfer_slab: non-finite result",
    "transfer_slab: the result overflowed float64 (accumulated intensity "
    "S e^{-K ds} + G eps beyond the range, or extreme gain: negative "
    "absorption, e^{-K ds} beyond the floating-point range)",
    "transfer_slab: the matrix exponential returned NaN: |K ds| needs more than "
    "max_squarings squarings (raise max_squarings or split the slab), or an "
    "intermediate of an extreme gain overflowed",
    "transfer_slab: eps * ds overflows the floating-point range; rescale the "
    "source units or split the slab",
    "transfer_slab: non-finite input (S, eps, K or ds contains NaN or inf)",
)


def _diagnose(out, S, eps, K, ds):
    """``(bad, index)`` for ``equinox.branched_error_if`` over ``_REFUSALS``."""
    inputs_ok = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in (S, eps, K, ds)]))
    column_ok = jnp.all(jnp.isfinite(eps * ds))
    bad = ~(inputs_ok & jnp.all(jnp.isfinite(out)))
    cause = jnp.where(jnp.any(jnp.isnan(out)), 2, 1)
    cause = jnp.where(~inputs_ok, 4, jnp.where(~column_ok, 3, cause))
    return bad, jnp.where(bad, cause, 0)


def _refuse(values, out, S, eps, K, ds):
    return eqx.branched_error_if(values, *_diagnose(out, S, eps, K, ds), _REFUSALS)


@partial(jax.custom_jvp, nondiff_argnums=(4, 5))
def _slab(S, eps, K, ds, max_squarings, route):
    out = _slab_value(S, eps, K, ds, max_squarings, route)
    return _refuse(out, out, S, eps, K, ds)


def _slab_value(S, eps, K, ds, max_squarings, route=None):
    """The 5x5 value; where JAX differentiates this primal (the primal of a
    ``jvp``/``vjp`` under a further derivative; ``lax.scan`` under ``vjp``) its
    derivatives come from ``_companion`` once ``_uses_companion`` (plain autodiff
    loses the source tangent ``deps ds 2^-k`` to underflow). ``route`` (static)
    fixes the branch; ``transfer_los`` decides once per path (1.4x per slab)."""
    value = _plain_value(S, eps, K, ds, max_squarings)
    operands = (value, *(a.astype(value.dtype) for a in (S, eps, K, ds)))
    companion = partial(_companion, max_squarings=max_squarings)
    if route is None:
        use = _any_member(_uses_companion(eps, K, ds))
        return jax.lax.cond(use, companion, lambda value, *_: value, *operands)
    return companion(*operands) if route else value


def _plain_value(S, eps, K, ds, max_squarings):
    """Augmented 5x5 exponential with the source column scaled by ``2^-k``."""
    dtype = jnp.result_type(S, K, eps, ds, 1.0)
    # A coordinate rescaling only: freezing it preserves derivatives of the
    # analytic map, and avoids the undefined derivative of ||eps|| at zero.
    column = (eps * ds).astype(dtype)
    k = _source_exponent(jax.lax.stop_gradient(jnp.max(jnp.abs(column))))
    real = jnp.finfo(dtype).dtype
    M = jnp.zeros((5, 5), dtype=dtype)
    M = M.at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(column * _pow2(-k, real))
    y0 = jnp.concatenate([S.astype(dtype), _pow2(k, real).astype(dtype)[None]])
    return (expm(M, max_squarings=max_squarings) @ y0)[:4]


def _companion(value, S, eps, K, ds, max_squarings):
    """``value`` bit for bit (minus an exact ``+0``, so ``-0.0`` survives) with
    the derivatives of ``lin = Phi S + G eps``."""
    lin = _linear(S, eps, K, ds, max_squarings)
    return jax.lax.stop_gradient(value) - (jax.lax.stop_gradient(lin) - lin)


@partial(jax.custom_jvp, nondiff_argnums=(4,))
def _linear(S, eps, K, ds, max_squarings):
    """``Phi S + G eps`` (non-finite entries 0 in the value); tangent as the slab's."""
    Phi, G = _propagators(K, ds, max_squarings)
    return _finite_primal(Phi @ S + G @ eps)


@partial(_linear.defjvp, symbolic_zeros=True)
def _linear_jvp(max_squarings, primals, tangents):
    out = _linear(*primals, max_squarings)
    return out, _tangent(primals, tangents, out.dtype, max_squarings)


def _companion_exponent(dtype):
    """896 in float64 (``maxexp - 128``); 0, i.e. always, in float32."""
    return jnp.finfo(dtype).maxexp - 128


def _uses_companion(eps, K, ds):
    """``_exposure >= _companion_exponent``, from primal values only."""
    column = jax.lax.stop_gradient((eps * ds).astype(jnp.result_type(eps, K, ds, 1.0)))
    k = _source_exponent(jnp.max(jnp.abs(column)))
    return _exposure(k, K, ds) >= _companion_exponent(column.dtype)


def _exposure(k, K, ds):
    """``k + max(0, -e(ds)) + max(0, e(max|K ds|))``, ``e`` the binary exponent:
    the plain path's source tangent ``deps ds 2^-k``, reduced by the squarings."""
    e_ds = jnp.frexp(jnp.abs(ds))[1]
    e_tau = jnp.frexp(jnp.max(jnp.abs(K * ds)))[1]
    return k + jnp.maximum(0, -e_ds) + jnp.maximum(0, e_tau)


@jax.custom_jvp
def _finite_primal(x):
    """``x`` with non-finite entries set to 0 in the value; tangent unchanged."""
    return jnp.where(jnp.isfinite(x), x, 0)


@_finite_primal.defjvp
def _finite_primal_jvp(primals, tangents):
    return _finite_primal(primals[0]), tangents[0]


def _propagators(K, ds, max_squarings):
    """``(Phi, G) = (e^{-K ds}, int_0^ds e^{-K t} dt)`` from one 8x8 exponential.

    The identity block (not ``I ds``) keeps the norm at ``max(|K ds|, 1)``, at
    most that of the value's 5x5 exponential, so they exceed ``max_squarings``
    only where the value is refused. ``expm_pade13`` has no branch on a member's
    value, so the rule's ``jax.jvp`` of it transposes under a batched ``vmap``."""
    A = jnp.zeros((8, 8), dtype=K.dtype)
    A = A.at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4, dtype=K.dtype))
    E = expm_pade13(A, max_squarings)
    return E[:4, :4], E[:4, 4:] * ds


def _is_zero(t):
    return isinstance(t, SymbolicZero) or t.dtype == jax.dtypes.float0


def _split_exponent(S, column):
    """``max(0, e - maxexp // 2)``, ``e`` the exponent of ``max(|S|, |eps ds|)``."""
    top = jnp.maximum(jnp.max(jnp.abs(S)), jnp.max(jnp.abs(column)))
    top = jax.lax.stop_gradient(top)
    return jnp.maximum(0, jnp.frexp(top)[1] - jnp.finfo(top.dtype).maxexp // 2)


def _tangent(primals, tangents, dtype, max_squarings):
    """``Phi dS + G deps + dPhi S + dG eps``, linear in the tangents.

    ``dPhi S + dG eps`` is ``((dPhi 2^a) (S 2^-j) + (dG 2^a) (eps 2^-j)) 2^(j - a)``,
    ``a = j // 2`` (``_split_exponent``): neither the forward pass nor its
    transpose (``jacrev`` in ``K`` or ``ds``) forms ``dG eps`` or the cotangent
    ``ct eps`` at full scale, which overflowed near 1.8e308 before 0.3.0. The
    factors are exact powers of two (all 1 for ``j = 0``)."""
    S, eps, K, ds = (jnp.asarray(p, dtype) for p in primals)
    dS, deps, dK, dds = tangents
    if _is_zero(dK) and _is_zero(dds):
        Phi, G = _propagators(K, ds, max_squarings)
        tangent = jnp.zeros(4, dtype)
    else:
        j = _split_exponent(S, eps * ds)
        a = j // 2
        dK, dds = (
            jnp.zeros_like(p) if _is_zero(t) else t.astype(dtype) * _pow2(a, dtype)
            for p, t in ((K, dK), (ds, dds))
        )
        (Phi, G), (dPhi, dG) = jax.jvp(
            lambda K_, ds_: _propagators(K_, ds_, max_squarings), (K, ds), (dK, dds)
        )
        down = _pow2(-j, dtype)
        tangent = (dPhi @ (S * down) + dG @ (eps * down)) * _pow2(j - a, dtype)
    if not _is_zero(dS):
        tangent = tangent + Phi @ dS.astype(dtype)
    if not _is_zero(deps):
        tangent = tangent + G @ deps.astype(dtype)
    return tangent


@partial(_slab.defjvp, symbolic_zeros=True)
def _slab_jvp(max_squarings, route, primals, tangents):
    """The slab is linear in ``(S, eps)``: ``out = Phi S + G eps`` (``_tangent``)."""
    S, eps, K, ds = primals
    # Re-enter the rule for the primal so higher derivatives see it again.
    out = _slab(S, eps, K, ds, max_squarings, route)
    # The refusal is applied to every residual the tangent uses, so that it
    # survives dead-code elimination under jit(jacrev) (the value is unused).
    out, *residuals = _refuse((out, S, eps, K, ds), out, S, eps, K, ds)
    return out, _tangent(residuals, tangents, out.dtype, max_squarings)


def _source_exponent(magnitude):
    """``k = floor(log2 magnitude) + 1`` clipped to ``[0, 1022]`` (float64): ``2^k``,
    ``2^-k`` normal; the scaled column is below 1 (below 4 from ``2^1022`` on)."""
    bias = jnp.finfo(magnitude.dtype).maxexp - 1
    return jnp.clip(jnp.frexp(magnitude)[1], 0, bias - 1)


def transfer_los(S0, eps_s, K_s, ds, *, max_squarings=32):
    """Chain of constant-coefficient slabs along the LOS.

    ``eps_s``: (N, 4) emissivity per slab; ``K_s``: (N, 4, 4) Mueller matrices;
    ``ds``: scalar slab length (or per-slab (N,)).  Returns final Stokes (4,).
    Each slab is ``transfer_slab``: a non-finite slab is refused inside the
    scan with its cause, and derivatives use its linear tangent rule. The
    derivative route of the value (``_slab_value``) is chosen once per path.
    Compiled once per shape (``filter_jit``); eager refusals raise
    ``EquinoxRuntimeError``.
    """
    S0, eps_s, K_s, ds = map(jnp.asarray, (S0, eps_s, K_s, ds))
    n = eps_s.shape[0] if eps_s.ndim == 2 else -1
    shapes = (S0.shape, eps_s.shape, K_s.shape)
    if shapes != ((4,), (n, 4), (n, 4, 4)) or ds.shape not in ((), (n,)):
        raise ValueError(
            "shape contract: S0 (4,), eps_s (N,4), K_s (N,4,4), ds scalar or (N,)"
        )
    budget = _squaring_budget(max_squarings)
    if eps_s.shape[0] == 0:
        return S0
    return _los_compiled(S0, eps_s, K_s, ds, budget)


@eqx.filter_jit
def _los_compiled(S0, eps_s, K_s, ds, max_squarings):
    S0 = S0.astype(jnp.result_type(S0, eps_s, K_s, ds, 1.0))
    lengths = jnp.broadcast_to(ds, eps_s.shape[:1])
    use = _any_member(jnp.any(jax.vmap(_uses_companion)(eps_s, K_s, lengths)))

    def path(route):
        def body(S, x):
            eps, K, d = (*x, ds) if ds.ndim == 0 else x
            return _slab(S, eps, K, d, max_squarings, route), None

        xs = (eps_s, K_s) if ds.ndim == 0 else (eps_s, K_s, ds)
        return lambda: jax.lax.scan(body, S0, xs)[0]

    return jax.lax.cond(use, path(True), path(False))


def optical_depth_factor(tau):
    """(1-exp(-tau))/tau with value and derivatives continued through zero.

    A fourth-degree series is used for ``|tau| < 1e-4`` (next term ``<= 1.4e-23``).
    Negative optical depths describe gain; overflow for extreme gain remains
    a numerical-domain limit, not a physical saturation model.
    """
    tau = jnp.asarray(tau)
    small = jnp.abs(tau) < 1e-4
    regular = -jnp.expm1(-tau) / jnp.where(small, 1.0, tau)
    series = 1 - tau / 2 + tau**2 / 6 - tau**3 / 24 + tau**4 / 120
    return jnp.where(small, series, regular)


def faraday_rotation_matrix(theta):
    """Stokes-space rotation of (Q, U) by angle ``theta``: P -> P exp(i theta)."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    rows = [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0, 0, 1.0]]
    return jnp.array(rows)


def conversion_matrix(psi):
    """Faraday conversion: rotation of the (Q, V) plane by angle ``psi``.

    Q' = Q cos(psi) + V sin(psi),  V' = -Q sin(psi) + V cos(psi).
    Converts linear Q into circular V (and back).
    """
    c, s = jnp.cos(psi), jnp.sin(psi)
    rows = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, 0.0, s],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -s, 0, c],
    ]
    return jnp.array(rows)
