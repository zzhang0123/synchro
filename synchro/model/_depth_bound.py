"""Depth-model term of ``eq: channel error budget`` (private helper of ``bounds``).

``depth_error_bound`` implements ``N_src tau <|K_P| |delta varphi|>`` of the
manuscript's perturbation bounds (main.tex app: perturbation bounds); it is
re-exported by ``synchro.model.bounds``, whose module docstring states the
units, shapes and what is not certified.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ._bound_helpers import _amplitude, _as_float, _nonnegative, _stokes, _tau_max
from .errors import ErrorTerm


def depth_error_bound(
    samples,
    kernel,
    channels,
    delta_depth,
    *,
    amplitude=1.0,
    phase=None,
    absolute_channel_sum=None,
    batch_size=1024,
) -> ErrorTerm:
    """``amplitude tau_max <|K_P| |delta depth|>`` per channel (``eq: channel error budget``).

    ``delta_depth`` (rad/m^2) is a scalar or ``(S,)`` per-emitter depth
    error; ``tau_max = 2 (c/nu_lo)^2`` is the largest phase coordinate on
    each channel support. ``|K_P|`` is ``|Modes.P[0]|`` from
    ``kernel.channel_modes`` at each sample with ``depth_ref`` set to the
    sample's own depth and ``phase`` passed through (``None`` selects the
    kernel's default route). Because the modulus of the phased line sum is
    below the absolute line sum, the result is ``kind="estimate"`` unless
    ``absolute_channel_sum`` ``(S, n_ch)`` (``sum_m |Q_m R_j|`` or
    ``int |R_j K_P| dnu`` per sample) is supplied, which gives ``"bound"``.
    Zeros in ``I, V``. Samples are mapped in batches of ``batch_size``.
    Units: ``amplitude`` times the per-electron channel Stokes units;
    ``tau_max`` in m^2 times ``delta_depth`` in rad/m^2 is a phase. Assumes
    the per-emitter depth error is bounded by ``delta_depth``; not
    certified: that bound, and the sampling of the population.
    """
    w = samples.normalised_weights()
    size = samples.size
    delta = _as_float(delta_depth, "delta_depth")
    if delta.ndim == 0:
        delta = jnp.broadcast_to(delta, (size,))
    if delta.shape != (size,):
        raise ValueError(f"delta_depth must be a scalar or have shape ({size},)")
    delta = eqx.error_if(
        delta, jnp.any(~jnp.isfinite(delta)), "delta_depth must be finite"
    )
    tau = _tau_max(channels)
    if absolute_channel_sum is None:
        absolute = _kernel_modulus(samples, kernel, channels, phase, batch_size)
        kind = "estimate"
        note = (
            "eq: channel error budget depth term tau_max <|K_P| |delta depth|> with |K_P| the "
            "modulus of the phased channel line sum (below sum_m |Q_m R_j|): estimate"
        )
    else:
        absolute = _as_float(absolute_channel_sum, "absolute_channel_sum")
        if absolute.shape != (size, channels.n_ch):
            raise ValueError(
                f"absolute_channel_sum must have shape ({size}, {channels.n_ch})"
            )
        absolute = _nonnegative(absolute, "absolute_channel_sum")
        kind = "bound"
        note = (
            "eq: channel error budget depth term tau_max <sum_m |Q_m R_j| |delta depth|> "
            "with the supplied absolute channel sum"
        )
    average = (w * jnp.abs(delta)) @ absolute
    value = _amplitude(amplitude) * _stokes(channels.n_ch, P=tau * average)
    return ErrorTerm(value=value, kind=kind, note=note, manuscript_term="E_phys")


def _kernel_modulus(samples, kernel, channels, phase, batch_size):
    """``(S, n_ch)`` moduli of the reference polarised channel kernel per sample."""

    def one(leaf):
        gamma, B, mu, eta, depth = leaf
        modes = kernel.channel_modes(
            channels,
            gamma,
            B,
            mu,
            eta,
            phase=phase,
            depth_ref=depth,
            s_depth=jnp.ones(()),
        )
        return jnp.abs(modes.P[0])

    leaves = (samples.gamma, samples.B, samples.mu, samples.eta, samples.depth)
    return jax.lax.map(one, leaves, batch_size=min(int(batch_size), samples.size))


__all__ = ["depth_error_bound"]
