"""Taylor blocks of the channel projection for ``build_core`` (private).

Split out of ``_basis_core.py`` to keep that file under 400 lines.
``taylor_blocks`` returns ``D^{r,s}`` (no ``1/(r! s!)``) of the projected
modes in ``(z_gamma, z_B)`` at ``z = 0`` for the ``(r, s)`` the retained rows
need (all ``r + s <= N`` without per-variable caps):
``HarmonicKernel.angular_taylor`` for ``derivatives="analytic"``, nested
forward passes of ``angular_projection`` (``point_taylor``) otherwise.
``columns`` scatters them into basis columns. Units and shapes follow
``syncmoments.model.basis``; nothing here certifies quadrature accuracy.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ._harmonic_taylor import level_cost, level_plan, lower_closure, point_taylor

# -- the differentiated projection -------------------------------------------------------


def projection_in_z(kernel, channels, reference, phase, index, quadrature):
    """``z -> (I, V, P)`` projected modes at ``gamma0 + s_gamma z0, B0 + s_B z1``."""
    s_gamma, s_B, s_depth = reference.scales

    def f(z):
        modes = kernel.angular_projection(
            channels,
            reference.gamma0 + s_gamma * z[0],
            reference.B0 + s_B * z[1],
            phase=phase,
            depth_ref=reference.depth_ref,
            s_depth=s_depth,
            truncation=index,
            quadrature=quadrature,
        )
        return modes.I, modes.V, modes.P

    return f


def taylor_tensors(f, order):
    """``(f, Df, ..., D^order f)`` at ``z = 0`` in one nested forward pass.

    Each derivative tensor has ``n`` trailing axes of length 2 (the order of
    the axes is immaterial by symmetry of mixed partials). Lower orders ride
    along as ``has_aux`` outputs, so the primal work is not repeated.
    """

    def level0(z):
        value = f(z)
        return value, (value,)

    fn = level0
    for _ in range(order):

        def level(z, prev=fn):
            out, lower = jax.jacfwd(prev, has_aux=True)(z)
            return out, lower + (out,)

        fn = level
    return fn(jnp.zeros(2))[1]


def columns(rows, index, block_of, n_ch, dtype, weight_axis):
    """Scatter ``D^{r,s} T[pair] / (r! s!)`` (weight ``b`` if ``weight_axis``) into columns.

    ``block_of(r, s)`` is ``D^{r,s}`` of the projected modes, ``(n_ch, n_lk)``
    (``(n_weights, n_ch, n_lk)`` with ``weight_axis``).
    """
    lookup = {pair: i for i, pair in enumerate(index.pairs)}
    groups = {}
    for pos, (l, k, r, s, b) in enumerate(rows):
        groups.setdefault((r, s, b), []).append((pos, lookup[(l, k)]))
    out = jnp.zeros((n_ch, len(rows)), dtype=dtype)
    for (r, s, b), items in groups.items():
        block = block_of(r, s)
        if weight_axis:
            block = block[b]
        positions = [pos for pos, _ in items]
        pairs = [pair for _, pair in items]
        norm = math.factorial(r) * math.factorial(s)
        out = out.at[:, positions].set(block[:, pairs] / norm)
    return out


def needed_orders(index) -> tuple[tuple[int, int], ...]:
    """Lower closure of the ``(r, s)`` of the retained rows (``h0`` and ``h2``).

    Equal to every ``r + s <= N`` without per-variable caps; with
    ``Truncation.max_orders`` it drops the multi-indices no row uses.
    """
    return lower_closure({(r, s) for (_, _, r, s, _) in index.h0 + index.h2})


def taylor_blocks(kernel, channels, reference, phase, index, quadrature):
    """``{(r, s): (D^{r,s} I, D^{r,s} V, D^{r,s} P)}`` of the projection at ``z = 0``.

    Only the ``(r, s)`` of :func:`needed_orders` are differentiated.
    ``HarmonicKernel(derivatives="analytic")`` uses its ``angular_taylor``
    (no tangent through the Bessel quadrature, Leibniz-assembled Legendre
    contraction); every other kernel, and ``derivatives="autodiff"``, uses
    nested forward passes of ``angular_projection`` (``point_taylor``;
    ``jax.jacfwd`` at every level without caps, as ``taylor_tensors``), with
    the chunk budget divided by the forward tangents (``for_tangents``).
    """
    alphas = needed_orders(index)
    if getattr(kernel, "derivatives", None) == "analytic":
        s_gamma, s_B, s_depth = reference.scales
        modes = kernel.angular_taylor(
            channels,
            reference.gamma0,
            reference.B0,
            scales=(s_gamma, s_B),
            phase=phase,
            depth_ref=reference.depth_ref,
            s_depth=s_depth,
            truncation=index,
            quadrature=quadrature,
            order=max(r + s for r, s in alphas),
            orders=alphas,
        )
        return {alpha: (m.I, m.V, m.P) for alpha, m in modes.items()}
    if callable(getattr(kernel, "for_tangents", None)):
        kernel = kernel.for_tangents(level_cost(level_plan(alphas)))
    f = projection_in_z(kernel, channels, reference, phase, index, quadrature)
    return point_taylor(f, alphas)


class DepthPartial:
    """One phase weight ``d^b/dz^b exp(i tau (depth_ref + s_depth z))`` at ``z = 0``.

    ``= exp(i tau depth_ref) (i tau s_depth)^b`` (``b!`` times the Taylor
    phase weight ``w_b``); ``tau`` in m^2, depths in rad/m^2.
    """

    n_weights = 1

    def __init__(self, b: int):
        self.b = int(b)

    def __call__(self, tau, *, depth_ref, s_depth):
        tau = jnp.asarray(tau)
        weight = jnp.exp(1j * tau * depth_ref) * (1j * tau * s_depth) ** self.b
        return weight[None]


def partials_in_z3(basis, kernel, route, sample_depth, betas):
    """``point (3,) -> {(r, s, b): (I, V, P0)}``: ``d^(r,s,b)`` in ``z_3`` at the point.

    Covers every ``beta = (r, s, b)`` of ``betas`` (and the lower ``(r, s)``
    at each ``b``): one ``angular_taylor`` pass per depth order ``b`` with the
    single weight :class:`DepthPartial` ``(b)`` on the per-emitter (Taylor)
    route. ``I, V`` do not depend on ``z_depth``, nor does ``P`` on a screen
    route; those partials are zero.
    """
    reference, channels, index = basis.reference, basis.channels, basis.index
    s_gamma, s_B, s_depth = reference.scales
    wanted = {tuple(int(v) for v in beta) for beta in betas}
    by_b = {}
    for r, s, b in wanted:
        keep = sample_depth or b == 0
        by_b.setdefault(b if keep else 0, set()).add((r, s) if keep else (0, 0))

    def fn(point):
        if sample_depth:
            depth, scale = reference.depth_ref + s_depth * point[2], s_depth
        else:
            depth, scale = reference.depth_ref, jnp.ones(())
        out = {}
        for b, alphas in sorted(by_b.items()):
            alphas = lower_closure(alphas)
            modes = kernel.angular_taylor(
                channels,
                reference.gamma0 + s_gamma * point[0],
                reference.B0 + s_B * point[1],
                scales=(s_gamma, s_B),
                phase=DepthPartial(b) if sample_depth else route,
                depth_ref=depth,
                s_depth=scale,
                truncation=index,
                order=max(r + s for r, s in alphas),
                orders=alphas,
            )
            for (r, s), m in modes.items():
                zero = jnp.zeros_like
                I, V = (m.I, m.V) if b == 0 else (zero(m.I), zero(m.V))
                out[(r, s, b)] = (I, V, m.P[0])
        zeros = tuple(jnp.zeros_like(a) for a in next(iter(out.values())))
        return {beta: out.get(beta, zeros) for beta in set(out) | wanted}

    return fn


def symmetric_frobenius(values, order, dims):
    """Frobenius norm of a symmetric order-``order`` tensor over ``dims`` axes.

    ``values[beta]`` is the entry ``d^beta`` (``beta`` of length ``dims``);
    every ``beta`` with ``|beta| = order`` counts ``order!/beta!`` times.
    """
    total = 0.0
    for beta, value in values.items():
        if len(beta) == dims and sum(beta) == order:
            count = math.factorial(order) / math.prod(map(math.factorial, beta))
            total = total + count * jnp.abs(value) ** 2
    return jnp.sqrt(total)


__all__ = [
    "projection_in_z",
    "taylor_tensors",
    "needed_orders",
    "taylor_blocks",
    "columns",
    "DepthPartial",
    "partials_in_z3",
    "symmetric_frobenius",
]
