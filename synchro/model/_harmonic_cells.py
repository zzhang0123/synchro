"""Line powers and product-coordinate cells of the harmonic kernel (private).

``harmonic_lines`` is the recurrence of ``synchro.stokes`` at a traced float
harmonic index; ``product_cells`` is the JAX port of the cell geometry in the
manuscript's ``validation/full_response_product.py``. See
``synchro.model.harmonic`` for the contract and what is not certified.
"""

from __future__ import annotations

from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..bessel import bessel_jn_and_prime
from ..constants import C_CGS, E_ESU, M_E
from ._kernel_helpers import gyrofrequency_hz, sine_from_cosine

# |t| floor of the product cells: nodes of a non-empty cell satisfy |t| >= 1e-17
# for n_outer <= 128, so the floor never moves a node that carries weight.
_T_FLOOR = 1e-100
# Cells with 0 < lo < RHO_SWITCH * hi move their nodes affinely in t under AD
# (product_cells); the v-map would amplify the edge velocity by up to
# RHO_SWITCH^(-3/4) = 5.6 at the switch.
RHO_SWITCH = 0.1


def auto_nodes(m_max: int) -> int:
    """``2^ceil(log2(max(128, 4 m_max + 64)))``: resolves ``m + |x| <= 2 m_max + 1``.

    Dimensionless node count for ``synchro.bessel``; the guard is a
    resolution statement, not an accuracy certificate.
    """
    return 2 ** math.ceil(math.log2(max(128, 4 * int(m_max) + 64)))


@lru_cache(maxsize=16)
def _leggauss_np(n: int):
    return np.polynomial.legendre.leggauss(int(n))


def leggauss(n: int) -> tuple[jax.Array, jax.Array]:
    """Gauss-Legendre nodes and weights as fresh JAX constants (no cached tracers)."""
    x, w = _leggauss_np(n)
    return jnp.asarray(x), jnp.asarray(w)


def harmonic_lines(m, gamma, B, mu, eta, *, n_nodes):
    """Natural-basis line powers and frequency at harmonic ``m`` (all broadcast).

    Returns ``(I_m, Q_m, V_m, nu_m)``: erg/s/sr per electron and Hz. ``m`` may
    be a traced float (integral values); ``n_nodes`` is the static Bessel
    resolution (``m + 1 + |x| <= n_nodes/2`` or NaN). ``sqrt(1 - c^2)`` is
    evaluated as ``sqrt((1-c)(1+c))`` and the viewing axis ``|eta| = 1`` and
    the field direction ``|mu| = 1`` use the exact recurrence limits with a
    zero tangent of the vanishing sine. ``B`` in Gauss. Assumes the vacuum
    helical-orbit reference of ``synchro.stokes``; not certified beyond it,
    and the Bessel resolution guard is not an accuracy certificate.
    """
    m, gamma, B, mu, eta = jnp.broadcast_arrays(
        *(jnp.asarray(v, dtype=float) for v in (m, gamma, B, mu, eta))
    )
    beta = jnp.sqrt(1.0 - 1.0 / gamma**2)
    b_par = beta * mu
    b_perp = beta * sine_from_cosine(mu)
    st = sine_from_cosine(eta)
    D = 1.0 - b_par * eta
    x = m * b_perp * st / D
    orders = jnp.stack([m - 1.0, m, m + 1.0])
    value, prime = bessel_jn_and_prime(orders, x[None], n_nodes=n_nodes)
    a_par = (eta - b_par) * b_perp * (value[0] + value[2]) / (2.0 * D)
    a_perp = b_perp * prime[1]
    wB = E_ESU * B / (gamma * M_E * C_CGS)
    pref = E_ESU**2 * wB**2 / (2.0 * jnp.pi * C_CGS) * m**2 / D**3
    I = pref * (a_par**2 + a_perp**2)
    Q = pref * (a_par**2 - a_perp**2)
    V = 2.0 * pref * a_par * a_perp
    return I, Q, V, m * wB / (2.0 * jnp.pi * D)


def _quarter_root(y):
    safe = jnp.where(y > 0.0, y, 1.0)
    return jnp.where(y > 0.0, safe**0.25, 0.0)


def product_cells(
    m, gamma, B, nu_lo, nu_hi, n_outer, n_inner, *, rho_switch=RHO_SWITCH
):
    """Nodes and weights of the two sign cells of harmonic ``m`` in ``[nu_lo, nu_hi]``.

    Returns ``(mu, eta, measure)`` of shape ``(2, n_outer, n_inner)`` covering
    ``mu > 0`` only; ``measure`` is the ``dmu deta`` weight (zero for empty
    cells). Row 0 is ``t > 0``, row 1 is ``t < 0``. Cell bounds in
    ``t = mu eta`` follow the manuscript (``lower = max(-1, (1 - m nu_B/nu_lo)
    /beta)``, ``upper = min(1, (1 - m nu_B/nu_hi)/beta)``), ``t = sign v^4``
    with ``v`` on Gauss-Legendre nodes in ``[lo^(1/4), hi^(1/4)]`` and
    ``log mu`` on Gauss-Legendre nodes in ``[log|t|, 0]``. Empty cells
    (``hi <= lo``) get zero weight and fixed interior nodes so that every
    downstream expression and its derivatives stay finite.

    Derivatives in ``(gamma, B)`` (values are unaffected): a cell with
    ``lo = 0`` or ``lo >= rho_switch * hi`` differentiates this rule itself
    (AD equals the derivative of the computed values). For ``0 < lo <
    rho_switch * hi`` the ``v``-map moves interior nodes by up to
    ``(hi/lo)^(3/4)`` times the edge velocity (``d lo^(1/4)/d lo =
    lo^(-3/4)/4``, unbounded as a channel edge meets a line at ``t = 0``);
    there the nodes instead follow the bounds affinely in ``t``
    (``_affine_follow``), an exact change of variables, so AD gives the rule
    applied to the derivative of the pulled-back integrand. The two differ
    by the rule's quadrature error, not by a bound. Frequencies in Hz, ``B``
    in Gauss. Assumes ``nu_m = m nu_B / (1 - beta mu eta)`` (the vacuum line
    positions); the Gauss-Legendre counts are declared, and their
    convergence is estimated, not bounded.
    """
    gx_o, gw_o = leggauss(n_outer)
    gx_i, gw_i = leggauss(n_inner)
    beta = jnp.sqrt(1.0 - 1.0 / gamma**2)
    scale = m * gyrofrequency_hz(gamma, B)
    lower = jnp.maximum(-1.0, (1.0 - scale / nu_lo) / beta)
    upper = jnp.minimum(1.0, (1.0 - scale / nu_hi) / beta)
    lo = jnp.clip(jnp.stack([lower, -upper]), 0.0, 1.0)
    hi = jnp.clip(jnp.stack([upper, -lower]), lo, 1.0)
    lo_f, hi_f = jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)
    regular = (lo_f == 0.0) | (lo_f >= rho_switch * hi_f)
    filled = (hi_f > lo_f)[:, None]
    a = jnp.where(filled, _quarter_root(jnp.where(regular, lo, lo_f))[:, None], 0.25)
    b = jnp.where(filled, _quarter_root(jnp.where(regular, hi, hi_f))[:, None], 0.75)
    v = (a + b) / 2.0 + (b - a) / 2.0 * gx_o
    outer = jnp.where(filled, gw_o * (b - a) / 2.0 * 4.0 * v**3, 0.0)  # (2, n_outer)
    t_aff, outer_aff = _affine_follow(v**4, outer, lo, hi, filled)
    t = jnp.where(regular[:, None], v**4, t_aff)
    outer = jnp.where(regular[:, None], outer, outer_aff)
    log_t = jnp.log(jnp.maximum(t, _T_FLOOR))[..., None]  # (2, n_outer, 1)
    mu = jnp.exp(log_t * (1.0 - gx_i) / 2.0)
    sign = jnp.array([1.0, -1.0])[:, None, None]
    eta = sign * jnp.exp(log_t * (1.0 + gx_i) / 2.0)
    inner = -log_t / 2.0 * gw_i
    return mu, eta, outer[..., None] * inner


def _affine_follow(t_frozen, w_frozen, lo, hi, filled):
    """``(t, w)`` equal to the frozen values, moving as ``lo + (hi - lo) s``.

    ``s = (t - lo)/(hi - lo)`` and ``w/(hi - lo)`` are frozen; ``lo``, ``hi``
    carry the derivatives. Empty cells keep zero weight.
    """
    lo_f, hi_f = jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi)
    span = (hi - lo)[:, None]
    span_f = jnp.where(filled, jax.lax.stop_gradient(span), 1.0)
    s = (t_frozen - lo_f[:, None]) / span_f
    t = t_frozen + (lo - lo_f)[:, None] * (1.0 - s) + (hi - hi_f)[:, None] * s
    return t, w_frozen * jnp.where(filled, span / span_f, 0.0)


def static_int(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a static integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


def sum_harmonics(fn, ms, chunk):
    """``sum_m fn(m)`` with ``fn`` vmapped over chunks inside ``lax.map``.

    ``fn`` maps a scalar harmonic to a pytree of arrays; padded entries
    (harmonic 1 with zero weight) do not contribute. Each chunk is
    ``jax.checkpoint``-ed: reverse mode recomputes a chunk instead of storing
    its residuals (357.6 GB -> 5.8 GB of XLA temporaries for ``jax.grad`` at
    the benchmark size); values and forward mode are unchanged.
    """
    ms = jnp.asarray(ms, dtype=float)
    chunk = max(1, min(int(chunk), ms.size))
    pad = (-ms.size) % chunk
    padded = jnp.concatenate([ms, jnp.ones(pad)]).reshape(-1, chunk)
    mask = jnp.concatenate([jnp.ones(ms.size), jnp.zeros(pad)]).reshape(-1, chunk)

    @jax.checkpoint
    def batch(args):
        m_batch, mask_batch = args
        out = jax.vmap(fn)(m_batch)
        return jax.tree.map(
            lambda a: jnp.sum(
                a * mask_batch.reshape((-1,) + (1,) * (a.ndim - 1)), axis=0
            ),
            out,
        )

    stacked = jax.lax.map(batch, (padded, mask))
    return jax.tree.map(lambda a: jnp.sum(a, axis=0), stacked)


__all__ = [
    "auto_nodes",
    "leggauss",
    "harmonic_lines",
    "product_cells",
    "RHO_SWITCH",
    "static_int",
    "sum_harmonics",
]
