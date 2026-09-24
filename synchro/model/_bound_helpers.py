"""Shared validation helpers of ``synchro.model.bounds`` (private)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .. import constants

C_SI_M = getattr(constants, "C_SI_M", 2.99792458e8)  # m/s


def _as_float(value, name):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return value.astype(jnp.result_type(value, 1.0))


def _nonnegative(value, name):
    value = eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite")
    return eqx.error_if(value, jnp.any(value < 0), f"{name} must be nonnegative")


def _amplitude(amplitude):
    amplitude = _as_float(amplitude, "amplitude")
    if amplitude.ndim != 0:
        raise ValueError("amplitude must be a scalar")
    return _nonnegative(amplitude, "amplitude")


def _stokes(n_ch, *, I=None, P=None, V=None):
    """Assemble ``(n_ch, 4)`` from per-component columns (``P`` fills Q and U)."""
    zero = jnp.zeros(n_ch)
    P = zero if P is None else P
    return jnp.stack([zero if I is None else I, P, P, zero if V is None else V], axis=1)


def _tau_max(channels):
    """Largest phase coordinate ``2 (c/nu)^2`` on each channel support (m^2)."""
    support = jnp.asarray(channels.support)
    if support.ndim != 2 or support.shape[1] != 2:
        raise ValueError("channels.support must have shape (n_ch, 2)")
    return 2.0 * (C_SI_M / support[:, 0]) ** 2
