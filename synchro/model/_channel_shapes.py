"""Private helpers for ``synchro.model.channels``: unit-peak shapes and areas.

Every shape function takes the frequency array already broadcast to
``(n_ch, *shape)`` together with the boolean mask ``inside`` of the open
support ``(nu_lo, nu_hi)`` and returns the unit-peak response evaluated with
safe arguments (finite value and finite nested derivatives everywhere); the
caller masks the result to exactly zero outside the support.
"""

from __future__ import annotations

from functools import lru_cache
import math

import jax
import jax.numpy as jnp
import numpy as np

FAMILIES = ("bump", "planck_taper", "raised_cosine", "tophat", "gaussian", "table")
NORMALISATIONS = ("unit_peak", "unit_integral")
SMOOTH = 99
# Below this value of 1 - t^2 (bump) or v/taper (Planck) the response is below
# exp(-999), which is exactly zero in float64; flooring there removes the
# 0 * inf products in the derivatives without changing any returned value.
FLOOR = 1e-3


@lru_cache(maxsize=1)
def bump_area() -> float:
    """``int_{-1}^{1} exp(1 - 1/(1 - t^2)) dt`` (512-node rule, error < 1e-15)."""
    x, w = np.polynomial.legendre.leggauss(512)
    return float(np.sum(w * np.exp(1 - 1 / (1 - x**2))))


@lru_cache(maxsize=8)
def legendre_rule(n_nu: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n_nu)


def unit_peak_area(family: str, taper: float, support_sigma: float) -> float:
    """Band integral of the unit-peak response divided by ``widths_hz``."""
    if family == "bump":
        return bump_area()
    if family == "planck_taper":  # each half taper integrates to taper/2
        return 2.0 * (1.0 - taper)
    if family == "raised_cosine":
        return 1.0
    if family == "tophat":
        return 2.0
    return math.sqrt(2 * math.pi) * math.erf(support_sigma / math.sqrt(2))


def _expand(values, nu):
    return values.reshape((values.shape[0],) + (1,) * (nu.ndim - 1))


def unit_peak_shape(family, nu, inside, *, centres, widths, support, taper, table):
    """Unit-peak response with safe arguments (see module docstring)."""
    c, w = _expand(centres, nu), _expand(widths, nu)
    lo, hi = _expand(support[:, 0], nu), _expand(support[:, 1], nu)
    if family == "bump":
        t = jnp.where(inside, (nu - c) / w, 0.0)
        return jnp.exp(1.0 - 1.0 / jnp.maximum(1.0 - t * t, FLOOR))
    if family == "planck_taper":
        u = (nu - lo) / (hi - lo)
        v = jnp.where(inside, jnp.minimum(u, 1.0 - u), 0.5 * taper)
        v = jnp.clip(v, taper * FLOOR, taper * (1.0 - FLOOR))
        ramp = jax.nn.sigmoid(taper / (taper - v) - taper / v)
        return jnp.where(jnp.minimum(u, 1.0 - u) >= taper, 1.0, ramp)
    if family == "raised_cosine":
        return 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.where(inside, (nu - c) / w, 1.0)))
    if family == "tophat":
        return jnp.ones_like(nu * c)
    if family == "gaussian":
        return jnp.exp(-0.5 * jnp.where(inside, (nu - c) / w, 0.0) ** 2)
    grid, rows = (jnp.asarray(part) for part in table)
    flat = jnp.where(inside, nu, c).reshape(nu.shape[0], -1)
    values = jax.vmap(lambda x, r: jnp.interp(x, grid, r))(flat, rows)
    return values.reshape(nu.shape)


def table_support(grid: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Support ``(lo, hi)`` per row: nonzero samples extended to adjacent zeros."""
    lo, hi = [], []
    for row in table:
        nonzero = np.flatnonzero(row)
        lo.append(grid[max(nonzero[0] - 1, 0)])
        hi.append(grid[min(nonzero[-1] + 1, grid.size - 1)])
    return np.asarray(lo), np.asarray(hi)


def table_edge_jump(grid: np.ndarray, rows: np.ndarray, support: np.ndarray) -> float:
    edges = [np.interp(support[j], grid, rows[j]) for j in range(rows.shape[0])]
    return float(np.max(edges))
