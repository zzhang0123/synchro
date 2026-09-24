"""Sample probes of the remainder inputs (private helper of ``bounds.py``).

Implements ``RemainderInputs.from_samples``: the order-``N+1`` derivative
envelopes ``H`` by nested ``jax.jacfwd`` of the kernel's Legendre projections
in ``z = (z_gamma, z_B, z_depth)``, the absolute moments
``<|P_l P_k| ||z||^{N+1}>``, the angular residual ``<|K - K_L|>`` measured on
the samples, and the ``app: depth moments`` inputs. Units follow ``synchro.model.bounds``.
Nothing here is a bound: the Frobenius norm of the derivative tensor at
finitely many probe points is an estimate of the operator-norm supremum over
the reference-to-support segments (``kind="estimate"``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ._kernel_helpers import legendre_table
from .phase import TaylorPhase


def _probe_route(phase):
    """``(route, sample_depth)``: Taylor routes use the exact per-emitter phase."""
    if phase is None or isinstance(phase, TaylorPhase):
        return TaylorPhase(0), True
    return phase, False


def _point_function(basis, kernel, route, sample_depth):
    """``z (3,) -> (I, V, P0)`` projected modes at ``q0 + diag(s) z``."""
    reference, channels, index = basis.reference, basis.channels, basis.index
    s_gamma, s_B, s_depth = reference.scales

    def fn(z):
        depth = (
            reference.depth_ref + s_depth * z[2]
            if sample_depth
            else reference.depth_ref
        )
        modes = kernel.angular_projection(
            channels,
            reference.gamma0 + s_gamma * z[0],
            reference.B0 + s_B * z[1],
            phase=route,
            depth_ref=depth,
            s_depth=jnp.ones(()),
            truncation=index,
        )
        return modes.I, modes.V, modes.P[0]

    return fn


def _frobenius(tensor, order, dims):
    """Frobenius norm over the trailing ``order`` axes restricted to ``dims`` each."""
    for i in range(order):
        tensor = jnp.take(tensor, jnp.arange(dims), axis=-1 - i)
    flat = tensor.reshape(tensor.shape[: tensor.ndim - order] + (-1,))
    return jnp.sqrt(jnp.sum(jnp.abs(flat) ** 2, axis=-1))


def derivative_envelope(samples, basis, kernel, *, phase, segment_points):
    """``(H (n_ch, 3, n_lk), H_2d (n_ch, n_lk))`` by nested ``jacfwd`` at the probe points.

    Probe points: ``z = 0`` and ``t z_n`` for every sample ``n`` and every
    ``t`` in ``segment_points``; ``H`` is the maximum Frobenius norm of the
    order-``N+1`` tensor in ``z_3`` (``I, V`` do not depend on ``z_depth``);
    ``H_2d`` restricts the ``P`` tensor to ``(z_gamma, z_B)`` (``app: depth moments``).
    """
    order = basis.index.truncation.N + 1
    route, sample_depth = _probe_route(phase)
    fn = _point_function(basis, kernel, route, sample_depth)
    for _ in range(order):
        fn = jax.jacfwd(fn)
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    points = [jnp.zeros((1, 3))] + [float(t) * z for t in segment_points]
    points = jnp.concatenate(points, axis=0)

    def norms(point):
        dI, dV, dP = fn(point)
        return (
            _frobenius(dI, order, 2),
            _frobenius(dV, order, 2),
            _frobenius(dP, order, 3),
            _frobenius(dP, order, 2),
        )

    nI, nV, nP, nP2 = jax.lax.map(jax.jit(norms), points)
    H = jnp.stack([nI.max(0), nP.max(0), nV.max(0)], axis=1)
    return H, nP2.max(0)


def absolute_moments(samples, basis):
    """``(2, n_lk)``: rows ``<|P_l P_k| ||z_2||^{N+1}>`` and the same with ``z_3``."""
    index = basis.index
    N = index.truncation.N
    w = samples.normalised_weights()
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    legendre = _pair_legendre(samples, index)
    norm2 = jnp.linalg.norm(z[:, :2], axis=1) ** (N + 1)
    norm3 = jnp.linalg.norm(z, axis=1) ** (N + 1)
    return jnp.stack([(w * norm2) @ legendre, (w * norm3) @ legendre])


def _pair_legendre(samples, index):
    """``|P_l(mu) P_k(eta)|`` for the retained pairs, ``(S, n_lk)``."""
    pairs = np.asarray(index.pairs, dtype=int).reshape(-1, 2)
    L_mu, L_eta = index.truncation.L_mu, index.truncation.L_eta
    p_l = legendre_table(samples.mu, L_mu)[:, pairs[:, 0]]
    p_k = legendre_table(samples.eta, L_eta)[:, pairs[:, 1]]
    return jnp.abs(p_l * p_k)


def angular_residual(samples, basis, kernel, *, phase, batch_size):
    """``<|K_X(p) - sum_lk K_{X;lk}(gamma, B) P_l(mu) P_k(eta)|>``, ``(n_ch, 3)``.

    Exact residual of the Legendre truncation for the discrete population
    (the natural ``Q`` residual for ``P``, phase of unit modulus); the
    ``(gamma, B)`` dependence is kept exact.
    """
    index, channels = basis.index, basis.channels
    route, sample_depth = _probe_route(phase)
    pairs = np.asarray(index.pairs, dtype=int).reshape(-1, 2)
    L_mu, L_eta = index.truncation.L_mu, index.truncation.L_eta
    depth_ref = basis.reference.depth_ref

    def one(leaf):
        gamma, B, mu, eta, depth = leaf
        depth = depth if sample_depth else depth_ref
        modes = kernel.channel_modes(
            channels,
            gamma,
            B,
            mu,
            eta,
            phase=route,
            depth_ref=depth,
            s_depth=jnp.ones(()),
        )
        projected = kernel.angular_projection(
            channels,
            gamma,
            B,
            phase=route,
            depth_ref=depth,
            s_depth=jnp.ones(()),
            truncation=index,
        )
        weights = (
            legendre_table(mu, L_mu)[pairs[:, 0]]
            * legendre_table(eta, L_eta)[pairs[:, 1]]
        )
        return jnp.stack(
            [
                jnp.abs(modes.I - projected.I @ weights),
                jnp.abs(modes.P[0] - projected.P[0] @ weights),
                jnp.abs(modes.V - projected.V @ weights),
            ],
            axis=1,
        )

    leaves = (samples.gamma, samples.B, samples.mu, samples.eta, samples.depth)
    residual = jax.lax.map(one, leaves, batch_size=min(int(batch_size), samples.size))
    return jnp.einsum("s,scx->cx", samples.normalised_weights(), residual)


def depth_rows(index):
    """``app: depth moments`` row list ``a = (l, k, r, s)``: the ``h2`` rows with ``b = 0``."""
    return tuple(row for row in index.h2 if row[4] == 0)


def depth_tail(samples, basis):
    """``<|chi_a| |depth - depth_ref|^{L+1}>`` over the ``app: depth moments`` rows, ``(n_a,)``."""
    index = basis.index
    L = index.truncation.depth_degree
    rows = np.asarray(depth_rows(index), dtype=int).reshape(-1, 5)
    w = samples.normalised_weights()
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    p_l = legendre_table(samples.mu, index.truncation.L_mu)[:, rows[:, 0]]
    p_k = legendre_table(samples.eta, index.truncation.L_eta)[:, rows[:, 1]]
    chi = jnp.abs(p_l * p_k * z[:, :1] ** rows[:, 2] * z[:, 1:2] ** rows[:, 3])
    tail = jnp.abs(samples.depth - basis.reference.depth_ref) ** (L + 1)
    return (w * tail) @ chi


def intrinsic_residual(rho_ang, H_2d, absolute, N):
    """``rho_P + sum_lk H_2d,lk <|P_l P_k| ||z_2||^{N+1}> / (N+1)!`` per channel."""
    return rho_ang[:, 1] + (H_2d @ absolute[0]) / math.factorial(N + 1)


__all__ = [
    "derivative_envelope",
    "absolute_moments",
    "angular_residual",
    "depth_rows",
    "depth_tail",
    "intrinsic_residual",
]
