"""Shared helpers of the kernel models (private; re-exported by ``kernels``).

Units and shapes follow ``syncmoments.model.kernels``: frequencies in Hz,
``tau = 2 (C_SI_M / nu)^2`` in m^2, Legendre projections over the pairs of
``MomentIndex.pairs`` with the ``(2l+1)(2k+1)/4`` coefficient factor. Nothing
here is a physical model; the helpers are exact only in their own algebra,
which the package tests validate (finite checks, not certificates).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C_CGS, C_SI_M, E_ESU, M_E
from .index import MomentIndex


class Modes(NamedTuple):
    """Channel-integrated natural-basis modes at one ``(gamma, B, mu, eta)``.

    ``I``, ``V``: ``(n_ch,)`` real; ``P``: ``(n_weights, n_ch)`` complex, the
    ``Q`` modes times the phase weights, without ``exp(2 i phi)``.
    Units: per-electron channel Stokes (erg/s/sr for ``unit_peak`` channels,
    per Hz for ``unit_integral``). Assumes the kernel's own physical model;
    not certified beyond it (``eq: smooth channel kernel``).
    """

    LABEL = "eq: smooth channel kernel"
    I: jax.Array
    V: jax.Array
    P: jax.Array


class ProjectedModes(NamedTuple):
    """Legendre-projected modes for the pairs of ``MomentIndex.pairs``.

    ``I``, ``V``: ``(n_ch, n_lk)`` real; ``P``: ``(n_weights, n_ch, n_lk)``
    complex. The ``(2l+1)(2k+1)/4`` Legendre coefficient factor is included.
    Same units as :class:`Modes`. The projection quadrature is the kernel's
    declared route; its error is estimated, not bounded (``numerical`` slot).
    """

    LABEL = "eq: channel derivative coefficients"
    I: jax.Array
    V: jax.Array
    P: jax.Array


def legendre_table(x, degree: int) -> jax.Array:
    """``P_0..P_degree`` at ``x`` by the three-term recurrence; ``(*x.shape, degree+1)``.

    Dimensionless; exact recurrence, no assumption, nothing certified beyond it.
    """
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or degree < 0
    ):
        raise ValueError("degree must be a nonnegative static integer")
    x = jnp.asarray(x)
    table = [jnp.ones_like(x)]
    if degree >= 1:
        table.append(x)
    for n in range(1, degree):
        table.append(((2 * n + 1) * x * table[n] - n * table[n - 1]) / (n + 1))
    return jnp.stack(table, axis=-1)


def phase_coordinate(nu_hz) -> jax.Array:
    """``tau = 2 (C_SI_M / nu)^2`` [m^2] for ``nu`` in Hz (``nu > 0``).

    Assumes the pure-rotation phase ``2 lambda^2 depth``; nothing certified.
    """
    return 2.0 * (C_SI_M / jnp.asarray(nu_hz)) ** 2


def phase_weights(phase, tau, depth_ref, s_depth) -> jax.Array:
    """``phase(tau, depth_ref=, s_depth=)`` or one unit weight when ``phase is None``.

    Returns ``(n_weights, *tau.shape)`` complex; ``tau`` in m^2, depths in
    rad/m^2. The route's own assumptions apply; nothing certified here.
    """
    tau = jnp.asarray(tau)
    if phase is None:
        return jnp.ones((1, *tau.shape), dtype=jnp.result_type(tau, 1j))
    weights = jnp.asarray(phase(tau, depth_ref=depth_ref, s_depth=s_depth))
    if weights.shape != (phase.n_weights, *tau.shape):
        raise ValueError(
            "phase weights must have shape (n_weights, *tau.shape); got "
            f"{weights.shape} for tau {tau.shape}"
        )
    return weights


def gyrofrequency_hz(gamma, B) -> jax.Array:
    """``nu_B = e B / (2 pi gamma m_e c)`` [Hz]; ``B`` in Gauss.

    The vacuum relativistic gyrofrequency (no plasma correction assumed or
    certified); broadcasts over its arguments.
    """
    return E_ESU * jnp.asarray(B) / (2.0 * jnp.pi * jnp.asarray(gamma) * M_E * C_CGS)


def legendre_norm(pairs) -> jax.Array:
    """``(2l+1)(2k+1)/4`` for each ``(l, k)`` pair; shape ``(n_lk,)``.

    Dimensionless Legendre normalisation; no assumption, nothing certified.
    """
    arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
    return jnp.asarray((2 * arr[:, 0] + 1) * (2 * arr[:, 1] + 1) / 4.0)


def gather_pairs(grid, pairs) -> jax.Array:
    """Select ``grid[..., l, k]`` per pair: ``(..., L_mu+1, L_eta+1) -> (..., n_lk)``.

    Pure indexing in the units of ``grid``; no assumption, nothing certified.
    """
    arr = np.asarray(pairs, dtype=int).reshape(-1, 2)
    if arr.size and (
        np.max(arr[:, 0]) >= grid.shape[-2] or np.max(arr[:, 1]) >= grid.shape[-1]
    ):
        raise ValueError("pairs exceed the projected Legendre grid")
    return grid[..., arr[:, 0], arr[:, 1]]


def resolve_index(truncation, components) -> MomentIndex:
    """Accept a ``Truncation`` or a ready ``MomentIndex``.

    Builds ``MomentIndex.build(truncation, components=components)`` when
    needed (dimensionless bookkeeping); no assumption, nothing certified.
    """
    if isinstance(truncation, MomentIndex):
        return truncation
    return MomentIndex.build(truncation, components=tuple(components))


def check_scalar_point(gamma, B, mu, eta):
    """Trace-time shape checks and ``error_if`` value checks of a kernel point.

    ``gamma`` dimensionless, ``B`` in Gauss, ``mu``, ``eta`` cosines; scalars
    only (``ValueError`` otherwise). Validation only: assumes and certifies
    nothing about the physics.

    Returns float arrays. ``gamma >= 1``, ``B >= 0``, ``|mu|, |eta| <= 1``,
    all finite.
    """
    arrays = [jnp.asarray(v) for v in (gamma, B, mu, eta)]
    for name, arr in zip(("gamma", "B", "mu", "eta"), arrays):
        if arr.ndim != 0:
            raise ValueError(f"{name} must be a scalar; got shape {arr.shape}")
        if jnp.iscomplexobj(arr):
            raise ValueError(f"{name} must be real")
    gamma, B, mu, eta = (arr.astype(jnp.result_type(arr, 1.0)) for arr in arrays)
    stacked = jnp.stack([gamma, B, mu, eta])
    invalid = (
        jnp.any(~jnp.isfinite(stacked))
        | (gamma < 1.0)
        | (B < 0.0)
        | (jnp.abs(mu) > 1.0)
        | (jnp.abs(eta) > 1.0)
    )
    stacked = eqx.error_if(
        stacked,
        invalid,
        "kernel point requires finite gamma >= 1, B >= 0, |mu|,|eta| <= 1",
    )
    return stacked[0], stacked[1], stacked[2], stacked[3]


def safe_sqrt(y):
    """``sqrt(max(y, 0))`` with a zero tangent at ``y <= 0`` (no ``0 * inf``)."""
    positive = y > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, y, 1.0)), 0.0)


def sine_from_cosine(c):
    """``sqrt(1 - c^2)`` as ``sqrt((1 - c)(1 + c))``: no cancellation at ``|c| -> 1``."""
    return safe_sqrt((1.0 - c) * (1.0 + c))


def monomials(z, count):
    """``[1, z, ..., z^(count-1)]`` by cumulative products (derivative-safe at ``z = 0``)."""
    powers = [jnp.ones_like(z)]
    for _ in range(1, count):
        powers.append(powers[-1] * z)
    return jnp.stack(powers)


__all__ = [
    "Modes",
    "ProjectedModes",
    "legendre_table",
    "phase_coordinate",
    "phase_weights",
    "gyrofrequency_hz",
    "legendre_norm",
    "gather_pairs",
    "resolve_index",
    "check_scalar_point",
    "safe_sqrt",
    "sine_from_cosine",
    "monomials",
]
