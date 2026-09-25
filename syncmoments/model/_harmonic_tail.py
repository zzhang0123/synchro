"""Omitted-harmonic probe estimate of ``HarmonicKernel.truncation_error`` (private).

Split out of ``harmonic.py`` to keep that file under 400 lines. The estimate
sums ``|S_m R_j(nu_m)|`` over the probe harmonics ``m_max+1 .. m_max+tail_probe``
(natural-basis ``|Q_m|`` in both the ``Q`` and ``U`` columns), either averaged
over concrete or traced samples ``(gamma, B, mu, eta, weights)`` or maximised
over the kernel's ``n_mu x n_eta`` Gauss-Legendre angular grid at the
reference ``(gamma0, B0)``. Harmonics beyond the probe are not included, so
the result is an ``estimate``, never a bound. Memory: the samples (or grid
points) run in blocks inside ``lax.map`` so that one step holds at most
``kernel.chunk_budget`` Bessel integrand values (``sample_plan``); blocking
changes the summation order only. Units: per-electron channel Stokes,
``(n_ch, 4)``; ``B`` in Gauss, Hz.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..rm import _screen_samples
from ._harmonic_cells import (
    auto_nodes,
    chunk_plan,
    harmonic_lines,
    leggauss,
    sum_harmonics,
)
from .errors import ErrorTerm


def _points(kernel, samples, reference):
    """``(gamma, B, mu, eta, weights, where)`` of the probe evaluation."""
    if samples is not None:
        gamma, w = _screen_samples(
            jnp.asarray(samples.gamma), getattr(samples, "weights", None)
        )
        B, mu, eta = (
            jnp.asarray(getattr(samples, k), dtype=float) for k in ("B", "mu", "eta")
        )
        return gamma, B, mu, eta, w, "averaged over the supplied samples"
    x_mu, _ = leggauss(kernel.n_mu)
    x_eta, _ = leggauss(kernel.n_eta)
    MU, ETA = jnp.meshgrid(x_mu, x_eta, indexing="ij")
    mu, eta = MU.ravel(), ETA.ravel()
    gamma = jnp.full(mu.shape, jnp.asarray(reference.gamma0, dtype=float))
    B = jnp.full(mu.shape, jnp.asarray(reference.B0, dtype=float))
    where = (
        "maximised over the angular grid at the reference (gamma0, B0), "
        "not a population average"
    )
    return gamma, B, mu, eta, None, where


def tail_nodes(kernel) -> int:
    """Bessel resolution of the probe harmonics ``m_max + 1 .. m_max + tail_probe``."""
    return auto_nodes(kernel.m_max + kernel.tail_probe)


def sample_plan(kernel, size) -> tuple[int, int]:
    """``(harmonics, block)`` per ``lax.map`` step for ``size`` probe points."""
    return chunk_plan(size, tail_nodes(kernel), kernel.m_chunk, kernel.chunk_budget)


def _per_point(kernel, channels, arrays, harmonics):
    """``(n_ch, S_pad, 4)`` probe sums per point, blocks of ``arrays[0].shape[1]``."""
    m_max, probe = kernel.m_max, kernel.tail_probe
    tail = jnp.arange(m_max + 1, m_max + probe + 1, dtype=float)
    n_nodes = tail_nodes(kernel)

    @jax.checkpoint
    def step(block):
        gamma, B, mu, eta = block

        def one(m):
            I, Q, V, nu = harmonic_lines(m, gamma, B, mu, eta, n_nodes=n_nodes)
            nu_safe = jnp.where(nu > 0.0, nu, 1.0)
            response = channels(nu_safe) * (B > 0.0)  # (n_ch, block)
            return jnp.stack([response * jnp.abs(s) for s in (I, Q, Q, V)], axis=-1)

        return sum_harmonics(one, tail, harmonics)

    sums = jax.lax.map(step, arrays)  # (n_blocks, n_ch, block, 4)
    return jnp.moveaxis(sums, 0, 1).reshape(sums.shape[1], -1, 4)


def probe_estimate(kernel, channels, samples, reference, prefix=""):
    """``estimate`` ErrorTerm ``(n_ch, 4)`` of the omitted harmonics (module docstring)."""
    gamma, B, mu, eta, w, where = _points(kernel, samples, reference)
    size = gamma.shape[0]
    harmonics, block = sample_plan(kernel, size)
    pad = (-size) % block
    arrays = tuple(
        jnp.pad(a, (0, pad), mode="edge").reshape(-1, block)
        for a in (gamma, B, mu, eta)
    )
    sums = _per_point(kernel, channels, arrays, harmonics)[:, :size]  # (n_ch, S, 4)
    if w is not None:
        value = jnp.einsum("csk,s->ck", sums, w)
    else:
        value = jnp.max(sums, axis=1)
    m_max, probe = kernel.m_max, kernel.tail_probe
    return ErrorTerm(
        value,
        "estimate",
        f"{prefix}sum over probe harmonics {m_max + 1}..{m_max + probe} of "
        f"|S_m R_j| {where}",
        "E_num",
    )


__all__ = ["probe_estimate", "sample_plan", "tail_nodes"]
