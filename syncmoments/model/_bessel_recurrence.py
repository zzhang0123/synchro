"""Bessel derivatives by the order recurrence on one contour (private).

The harmonic kernel needs ``(J_{m-1}, J_{m+1}, J_m')(x)`` and their
derivatives in ``x`` through the Taylor order of the basis. Automatic
differentiation of the contour quadrature (``syncmoments.bessel``) carries every
forward-mode tangent through all ``n_nodes`` integrand values. Here the
quadrature is evaluated at the primal ``x`` only:

* ``bessel_band(m, x, width, n_nodes)`` returns ``J_{m+k}(x)``, ``|k| <=
  width``, from one set of ``cos(phi)``, ``sin(phi)``, ``E`` on the contour
  ``t + i a`` of ``bessel_jn_neighbours`` (``phi = m t - |x| cosh(a) sin t``,
  ``E = exp(|x| sinh(a) cos t)``; order ``m + k`` uses ``exp(-(m + k) a) E
  e^{i (phi + k t)}``). Negative orders use ``J_{-p} = (-1)^p J_p`` (on the
  order-``m`` contour they lose ``(2m/|x|)^p`` in roundoff), ``x < 0`` the
  parity ``(-1)^n``, and ``x = 0`` the exact ``J_n(0) = delta_{n0}``.
  Its ``custom_jvp`` is ``J_n' = (J_{n-1} - J_{n+1})/2`` from the band of
  width ``width + 1``; nested derivatives recurse.
* ``neighbours_recurrence(m, x, n_nodes)`` has the values of
  ``bessel_jn_neighbours`` and the recurrence derivatives.
* ``derivative_coefficients(m, x0, order, n_nodes)`` gives ``d^k/dx^k`` of the
  triple at ``x0`` for ``k <= order`` from one band of width ``order + 1``
  (``J_n^{(k)} = 2^{-k} sum_j (-1)^j C(k, j) J_{n-k+2j}``);
  ``taylor_neighbours(coeffs, h)`` is the degree-``order`` polynomial in
  ``h = x - x0``: its derivatives through ``order`` at ``h = 0`` equal those
  of the triple, and outer derivatives in ``x0`` stay exact through the band's
  own rule.

Equality with autodiff: at a fixed contour ``d/dx`` of the order-``n``
integrand is ``-i sin(tau)`` times it, which is half the difference of the
order ``n - 1`` and ``n + 1`` integrands node by node, so the recurrence is
the derivative of the same trapezoid sums up to roundoff (and the reflection
of negative orders). Autodiff of the shifted contour amplifies roundoff by
about ``(2m/|x|)^k`` at order ``k`` for ``|x| << m``; the recurrence does not
for non-negative orders. Precision is checked against SciPy over the harmonic
resolution classes (``tests/model/test_bessel_recurrence.py``), not bounded.
Dimensionless; ``m`` is an integral order (traced float allowed, not
differentiated); ``n_nodes`` is static. Resolution guard as
``bessel_jn_neighbours``: ``m + 1 + |x| <= n_nodes/2`` or NaN (a resolution
statement, not an accuracy certificate).
"""

from __future__ import annotations

from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from ..bessel import bessel_jn_neighbours

RULES = ("recurrence", "autodiff")


@lru_cache(maxsize=32)
def _tables(n_nodes: int, width: int):
    """``sin t``, ``cos t``, ``t`` and ``cos(k t)/n``, ``sin(k t)/n`` for ``|k| <= width``."""
    t = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    k = np.arange(-width, width + 1)
    return (
        np.sin(t),
        np.cos(t),
        t,
        np.cos(np.outer(t, k)) / n_nodes,
        np.sin(np.outer(t, k)) / n_nodes,
    )


def _band_values(m, x, width, n_nodes):
    sin_t, cos_t, t, cos_kt, sin_kt = _tables(n_nodes, width)
    m, x = jnp.broadcast_arrays(
        jnp.asarray(m, dtype=float), jnp.asarray(x, dtype=float)
    )
    ax = jnp.abs(x)
    safe_x = jnp.where(ax > 0, ax, 1.0)
    shifted = (m > safe_x) & (m > 0)
    d = jnp.sqrt(jnp.maximum(m**2 - safe_x**2, 0.0))
    shift = jnp.where(shifted, jnp.log(jnp.maximum(m + d, 1.0)) - jnp.log(safe_x), 0.0)
    xsinh = jnp.where(shifted, d, 0.0)
    xcosh = jnp.where(shifted, m, safe_x)
    phase = m[..., None] * t - xcosh[..., None] * sin_t
    envelope = jnp.exp(xsinh[..., None] * (cos_t - 1.0))
    c, s = envelope * jnp.cos(phase), envelope * jnp.sin(phase)
    offsets = np.arange(-width, width + 1)
    orders = m[..., None] + offsets
    values = jnp.exp(xsinh[..., None] - orders * shift[..., None]) * (
        c @ cos_kt - s @ sin_kt
    )
    # J_{-p} = (-1)^p J_p: order -p sits at offset -2m - k, inside the band.
    source = jnp.where(orders < 0, -2.0 * m[..., None] - offsets, offsets) + width
    source = jnp.clip(source, 0, 2 * width).astype(jnp.int32)
    values = jnp.take_along_axis(values, source, axis=-1)
    odd = jnp.where(jnp.mod(orders, 2.0) == 0.0, 1.0, -1.0)
    values = jnp.where(orders < 0, odd * values, values)
    values = jnp.where((x < 0)[..., None], odd * values, values)
    return jnp.where((ax == 0)[..., None], jnp.where(orders == 0, 1.0, 0.0), values)


@partial(jax.custom_jvp, nondiff_argnums=(2, 3))
def bessel_band(m, x, width, n_nodes):
    """``J_{m+k}(x)`` for ``k = -width..width``, shape ``(*broadcast, 2 width + 1)``."""
    return _band_values(m, x, int(width), int(n_nodes))


@bessel_band.defjvp
def _bessel_band_jvp(width, n_nodes, primals, tangents):
    m, x = primals
    dx = jnp.asarray(tangents[1], dtype=float)
    full = bessel_band(m, x, width + 1, n_nodes)
    slope = (full[..., :-2] - full[..., 2:]) / 2.0
    return full[..., 1:-1], slope * dx[..., None]


@lru_cache(maxsize=16)
def _triple_matrix(order: int) -> np.ndarray:
    """``(order+1, 3, 2 order + 3)``: band weights of ``d^k (J_{m-1}, J_{m+1}, J_m')``."""
    width = order + 1
    out = np.zeros((order + 1, 3, 2 * width + 1))
    for k in range(order + 1):
        for f, (centre, power) in enumerate(((-1, k), (1, k), (0, k + 1))):
            for j in range(power + 1):
                offset = centre - power + 2 * j
                out[k, f, offset + width] += (
                    (-1) ** j * math.comb(power, j) / 2.0**power
                )
    return out


def _valid(m, x, n_nodes):
    m = jnp.asarray(m, dtype=float)
    return (m >= 1) & (m == jnp.floor(m)) & (m + 1 + jnp.abs(x) <= n_nodes / 2)


def derivative_coefficients(m, x0, order, n_nodes):
    """``d^k/dx^k (J_{m-1}, J_{m+1}, J_m')`` at ``x0``, shape ``(order+1, 3, *shape)``.

    One band of width ``order + 1``; NaN outside the resolution guard.
    """
    order = int(order)
    band = bessel_band(m, x0, order + 1, int(n_nodes))
    coeffs = jnp.moveaxis(
        band @ jnp.asarray(_triple_matrix(order).reshape(-1, band.shape[-1]).T), -1, 0
    )
    coeffs = coeffs.reshape((order + 1, 3) + band.shape[:-1])
    return jnp.where(_valid(m, x0, n_nodes), coeffs, jnp.nan)


def taylor_neighbours(coeffs, h):
    """The triple as the degree-``order`` Taylor polynomial in ``h = x - x0``.

    ``coeffs`` from :func:`derivative_coefficients` (linear in them). One
    Horner scheme per function on the unstacked coefficient rows (a stacked
    ``(3, ...)`` Horner under nested ``jacfwd`` was twice as slow).
    """
    h = jnp.asarray(h, dtype=float)
    out = []
    for f in range(3):
        value = coeffs[-1, f]
        for k in range(coeffs.shape[0] - 1, 0, -1):
            value = coeffs[k - 1, f] + value * (h / k)
        out.append(value)
    return tuple(out)


@partial(jax.custom_jvp, nondiff_argnums=(2,))
def neighbours_recurrence(m, x, n_nodes):
    """``(J_{m-1}, J_{m+1}, J_m')``: values of ``bessel_jn_neighbours``, recurrence derivatives."""
    return bessel_jn_neighbours(m, x, n_nodes=int(n_nodes))


@neighbours_recurrence.defjvp
def _neighbours_jvp(n_nodes, primals, tangents):
    m, x = primals
    dx = jnp.asarray(tangents[1], dtype=float)
    values = neighbours_recurrence(m, x, n_nodes)
    slope = derivative_coefficients(m, x, 1, n_nodes)[1]
    slope = jnp.where(jnp.isnan(slope), 0.0, slope)  # as autodiff of the NaN guard
    return values, tuple(slope[i] * dx for i in range(3))


def bessel_rule(rule, n_nodes):
    """``(m, x) -> (J_{m-1}, J_{m+1}, J_m')`` for ``rule`` in :data:`RULES` or a callable."""
    if callable(rule):
        return rule
    if rule is None or rule == "recurrence":
        return lambda m, x: neighbours_recurrence(m, x, int(n_nodes))
    if rule == "autodiff":
        return lambda m, x: bessel_jn_neighbours(m, x, n_nodes=int(n_nodes))
    raise ValueError(f"bessel rule must be one of {RULES} or a callable, got {rule!r}")


__all__ = [
    "RULES",
    "bessel_band",
    "derivative_coefficients",
    "taylor_neighbours",
    "neighbours_recurrence",
    "bessel_rule",
]
