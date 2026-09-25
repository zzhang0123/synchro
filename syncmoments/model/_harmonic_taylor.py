"""Angular routes of ``HarmonicKernel`` with Taylor-coefficient outputs (private).

``project(kernel, channels, lift, ..., alphas, rule)`` returns, for every
``(r, s)`` of the lower set ``alphas`` (``multi_indices(order)`` for the
full total degree), ``d^r_{z_gamma} d^s_{z_B}`` at ``z = 0`` of the grids
``(I (n_ch, L_mu+1, L_eta+1), V, P (n_w, n_ch, L_mu+1, L_eta+1))`` of the
projection at ``(gamma, B) = lift(z)`` (no ``1/(r! s!)``). ``{(0, 0)}`` is
the plain projection (``angular_projection``).

Per angular point the factors ``q X_m R_j(nu_m) w_b(tau_m) dmu deta`` (the
"streams", ``X = I, V, Q``; ``P`` complex) and, on the product route, the
Legendre tables ``P_l(mu)``, ``P_k(eta)`` of the moving nodes are
differentiated by nested forward passes in ``z`` (``point_taylor``); the
projection ``sum_p q_p P_l(mu_p) P_k(eta_p)`` is then assembled by the
Leibniz rule ``D^a (A B) = sum_{b <= a} C(a, b) D^b A D^{a-b} B`` instead of
differentiating the ``(L+1)^2``-wide contraction itself. With
``rule="taylor"`` the Bessel triple of ``harmonic_lines`` is its
degree-``max |alpha|`` Taylor polynomial in ``h = x - x0`` about the node
argument ``x0`` at ``z = 0`` (``_bessel_recurrence``):
one quadrature band per point and no tangent through the ``n_nodes``
integrand values; outer derivatives in ``lift``'s leaves stay exact. The
node motion of ``product_cells`` (both ``gamma`` and ``B`` move the cell
bounds through ``nu_B`` and ``beta``) is differentiated as in the
nested-``jacfwd`` path, so both are the same AD of the same program (node
motion included) and differ by summation order only (checked to ``1e-12``
per Stokes block in ``tests/model/test_b_analytic_derivatives.py``). That AD
is the derivative of the computed values only in cells that differentiate
the rule itself: in affine-follow cells (``0 < lo < RHO_SWITCH hi``) it is
the rule applied to the derivative of the pulled-back integrand, and at an
exact channel-edge/line coincidence at ``t = 0`` the rule is not
differentiable (``product_cells``). There the two differ by the rule's
quadrature error, which is estimated (refined-rule ``numerical``), not
bounded.

Nested forward passes (``level_plan``): level ``i`` pushes the tangents of a
direction set ``D_i`` (``(0, 1)`` = both, ``(0,)`` = ``z_gamma``, ``(1,)`` =
``z_B``); the order-``n`` output holds ``d_{i_1} .. d_{i_n}`` with ``i_j`` in
``D_j``, so ``(r, s)`` is reachable when the first ``n = r + s`` levels have
at most ``r`` ``z_gamma``-only and at most ``s`` ``z_B``-only sets. The plan
is the nesting of least forward-tangent count ``prod (1 + |D_i|)`` that
reaches every needed index; for a total-degree set it is ``jacfwd`` in both
directions at every level (``3^N`` tangents, as before), while per-variable
caps (``Truncation.max_orders``) drop directions (e.g. ``(8, 8, 2)`` with
``N_B = 1``: 6 instead of 9; ``N_B = 0``: 4). Units, shapes and what is not
certified: ``syncmoments.model.harmonic``.
"""

from __future__ import annotations

from functools import lru_cache
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from ._bessel_recurrence import derivative_coefficients, taylor_neighbours
from ._harmonic_cells import (
    harmonic_lines,
    leggauss,
    line_argument,
    product_cells,
    sum_harmonics,
    sum_point_blocks,
)
from ._kernel_helpers import legendre_table, phase_coordinate, phase_weights

_DIRECTIONS = ((0, 1), (0,), (1,))


def multi_indices(order: int) -> tuple[tuple[int, int], ...]:
    """``(r, s)`` with ``r + s <= order``, by total order."""
    return tuple((r, n - r) for n in range(order + 1) for r in range(n, -1, -1))


def lower_closure(alphas) -> tuple[tuple[int, int], ...]:
    """Downward closure of ``alphas`` (``(r, s)`` pairs), ordered as :func:`multi_indices`."""
    try:
        pairs = {(int(r), int(s)) for r, s in alphas}
    except (TypeError, ValueError) as exc:
        raise ValueError("multi-indices must be (r, s) pairs of integers") from exc
    if not pairs or min(min(p) for p in pairs) < 0:
        raise ValueError("multi-indices must be a nonempty set of (r, s) >= 0")
    closure = {(i, j) for r, s in pairs for i in range(r + 1) for j in range(s + 1)}
    return tuple(sorted(closure, key=lambda a: (a[0] + a[1], -a[0])))


def reachable(plan) -> tuple[tuple[int, int], ...]:
    """``(r, s)`` whose derivative the nesting ``plan`` computes (module docstring)."""
    out = [(0, 0)]
    for n in range(1, len(plan) + 1):
        only_g = sum(1 for d in plan[:n] if d == (0,))
        only_B = sum(1 for d in plan[:n] if d == (1,))
        out += [(r, n - r) for r in range(n, -1, -1) if r >= only_g and n - r >= only_B]
    return tuple(out)


def level_cost(plan) -> int:
    """Forward tangents per primal of the nesting: ``prod (1 + |D_i|)``."""
    return math.prod(1 + len(d) for d in plan)


@lru_cache(maxsize=64)
def _plan(closure):
    nestings = itertools.product(_DIRECTIONS, repeat=max(map(sum, closure)))
    covering = [seq for seq in nestings if set(closure) <= set(reachable(seq))]
    return min(covering, key=level_cost)  # first of least cost: all-both if tied


def level_plan(alphas) -> tuple[tuple[int, ...], ...]:
    """Least-tangent nesting (direction set per level) reaching every ``alphas``."""
    return _plan(lower_closure(alphas))


def _position(plan, alpha):
    """Index into the order-``|alpha|`` tensor of ``plan`` holding ``d^alpha``."""
    r, s = alpha
    n, index = r + s, []
    for i, dirs in enumerate(plan[:n]):
        if dirs == (0, 1):
            pick = 0 if r > sum(1 for d in plan[i + 1 : n] if d == (0,)) else 1
            index.append(pick)
        else:
            pick = dirs[0]
            index.append(0)
        r, s = (r - 1, s) if pick == 0 else (r, s - 1)
    return tuple(index)


def _below(alpha):
    r, s = alpha
    for i in range(r + 1):
        for j in range(s + 1):
            yield (i, j), (r - i, s - j), math.comb(r, i) * math.comb(s, j)


def _level(prev, dirs):
    """One nested forward pass of ``prev`` in the directions ``dirs``."""
    if dirs == (0, 1):

        def level(z):
            out, lower = jax.jacfwd(prev, has_aux=True)(z)
            return out, lower + (out,)

        return level
    direction = dirs[0]

    def level(z):  # jacfwd restricted to one basis vector
        basis = jnp.zeros((1,) + z.shape, z.dtype).at[0, direction].set(1.0)

        def push(e):
            return jax.jvp(prev, (z,), (e,), has_aux=True)

        _, out, lower = jax.vmap(push, out_axes=(None, -1, None))(basis)
        return out, lower + (out,)

    return level


def point_taylor(fn, alphas) -> dict:
    """``{(r, s): d^r_0 d^s_1 fn(0)}`` over the lower closure of ``alphas``.

    ``fn: (2,) -> pytree``; ``alphas`` is an int (total degree) or a set of
    ``(r, s)``. Nested forward passes of :func:`level_plan` (``jacfwd`` in
    both directions at every level for a total-degree set).
    """
    closure = (
        multi_indices(alphas) if isinstance(alphas, int) else lower_closure(alphas)
    )
    plan = level_plan(closure)

    def level0(z):
        value = fn(z)
        return value, (value,)

    g = level0
    for dirs in plan:
        g = _level(g, dirs)
    tensors = g(jnp.zeros(2))[1]
    return {
        alpha: jax.tree.map(
            lambda a, i=_position(plan, alpha): a[(Ellipsis,) + i],
            tensors[sum(alpha)],
        )
        for alpha in closure
    }


def _bessel(rule, m, x0_of, order, n_nodes):
    """``harmonic_lines`` Bessel rule; ``"taylor"`` expands about ``x0_of()``."""
    if rule != "taylor":
        return rule
    x0 = x0_of()
    coeffs = derivative_coefficients(m, x0, order, n_nodes)
    return lambda m_, x: taylor_neighbours(coeffs, x - x0)


def _leibniz_product(D, alphas):
    """``D^a sum_p q_p P_l(mu_p) P_k(eta_p)`` for ``a`` in ``alphas``, every stream ``q``.

    ``D[a] = (streams, P_l table (P, L_mu+1), P_k table (P, L_eta+1))`` with
    streams ``(..., P)`` (real or complex); returns ``{a: tuple of (...,
    L_mu+1, L_eta+1)}``. ``D^a (q P_l P_k) = sum_b C(a, b) D^b (q P_l)
    D^{a-b} P_k`` and likewise for ``q P_l``.
    """
    n = len(D[(0, 0)][0])
    A = {
        alpha: tuple(
            sum(c * D[b][0][i][..., None] * D[rest][1] for b, rest, c in _below(alpha))
            for i in range(n)
        )
        for alpha in alphas
    }
    return {
        alpha: tuple(
            sum(
                c * jnp.einsum("...pl,pk->...lk", A[b][i], D[rest][2])
                for b, rest, c in _below(alpha)
            )
            for i in range(n)
        )
        for alpha in alphas
    }


def _prepare(kernel, alphas, points, rule):
    """``(closure, order, n_nodes, (harmonics, block))``; only rules that push
    tangents through the Bessel contour (not ``"taylor"``) count them."""
    closure = lower_closure(alphas)
    tangents = 1 if rule == "taylor" else level_cost(level_plan(closure))
    plan = kernel._plan(points, tangents=tangents)
    return closure, max(sum(a) for a in closure), kernel.resolution(), plan


def project_product(
    kernel, channels, lift, phase, depth_ref, s_depth, degrees, alphas, rule
):
    """Product-cell route (see ``harmonic.py``); returns ``{(r, s): (I, V, P)}``."""
    L_mu, L_eta = degrees
    support = jnp.asarray(channels.support, dtype=float)
    size = 2 * kernel.n_outer * kernel.n_inner
    alphas, order, n_nodes, (chunk, block) = _prepare(kernel, alphas, size, rule)
    block = min(block, size)
    pad = (-size) % block  # sum_point_blocks pads the last block with the last point
    ell = np.arange(L_mu + 1)[:, None] + np.arange(L_eta + 1)[None, :]
    parity = jnp.asarray((-1.0) ** ell)

    def per_channel(carry, args):
        nu_lo, nu_hi, j = args

        def one(m):
            def nodes(z, start):
                """Block of the cell nodes at ``lift(z)``; a slice, not a gather."""
                gamma, B = lift(z)
                cells = product_cells(
                    m, gamma, B, nu_lo, nu_hi, kernel.n_outer, kernel.n_inner
                )
                return (gamma, B) + tuple(
                    jax.lax.dynamic_slice_in_dim(
                        jnp.pad(a.reshape(-1), (0, pad), mode="edge"), start, block
                    )
                    for a in cells
                )

            def points(arrays, mask):
                start = arrays[0][0]

                def x0_of():
                    gamma, _, mu, eta, _ = nodes(jnp.zeros(2), start)
                    return line_argument(m, gamma, mu, eta)

                bessel = _bessel(rule, m, x0_of, order, n_nodes)

                def factors(z):
                    gamma, B, mu, eta, measure = nodes(z, start)
                    I, Q, V, nu = harmonic_lines(
                        m, gamma, B, mu, eta, n_nodes=n_nodes, bessel=bessel
                    )
                    nu_safe = jnp.where(nu > 0.0, nu, 1.0)
                    response = jnp.take(channels(nu_safe), j, axis=0) * (B > 0.0)
                    w = phase_weights(
                        phase, phase_coordinate(nu_safe), depth_ref, s_depth
                    )
                    q = measure * response * mask
                    return (
                        (q * I, q * V, (q * Q)[None] * w),
                        legendre_table(mu, L_mu),
                        legendre_table(eta, L_eta),
                    )

                D = point_taylor(factors, alphas)
                return _leibniz_product(D, alphas)

            return sum_point_blocks(points, (jnp.arange(size),), block)

        return carry, sum_harmonics(one, kernel.harmonics(), chunk)

    _, grids = jax.lax.scan(
        per_channel,
        None,
        (support[:, 0], support[:, 1], jnp.arange(channels.n_ch)),
    )
    # grids[a] = (I, V (n_ch, L+1, L+1), P (n_ch, n_w, L+1, L+1)); the mu < 0 half:
    # I, Q even and V odd under (mu, eta) -> (-mu, -eta).
    return {
        alpha: (
            I * (1.0 + parity),
            V * (1.0 - parity),
            jnp.moveaxis(P, 0, 1) * (1.0 + parity),
        )
        for alpha, (I, V, P) in grids.items()
    }


def project_tensor(
    kernel, channels, lift, phase, depth_ref, s_depth, degrees, alphas, rule
):
    """Tensor Gauss-Legendre route; fixed nodes, so only the streams carry ``z``."""
    L_mu, L_eta = degrees
    x_mu, w_mu = leggauss(kernel.n_mu)
    x_eta, w_eta = leggauss(kernel.n_eta)
    MU, ETA = jnp.meshgrid(x_mu, x_eta, indexing="ij")
    p_l = legendre_table(x_mu, L_mu) * w_mu[:, None]
    p_k = legendre_table(x_eta, L_eta) * w_eta[:, None]
    # Flattened tensor points: (P,) coordinates and (P, L+1) weighted tables.
    flat = (
        MU.reshape(-1),
        ETA.reshape(-1),
        jnp.repeat(p_l, kernel.n_eta, axis=0),
        jnp.tile(p_k, (kernel.n_mu, 1)),
    )
    size = kernel.n_mu * kernel.n_eta
    alphas, order, n_nodes, (chunk, block) = _prepare(kernel, alphas, size, rule)

    def one(m):
        def points(arrays, mask):
            mu, eta, pl, pk = arrays

            def x0_of():
                return line_argument(m, lift(jnp.zeros(2))[0], mu, eta)

            bessel = _bessel(rule, m, x0_of, order, n_nodes)

            def factors(z):
                gamma, B = lift(z)
                I, Q, V, nu = harmonic_lines(
                    m, gamma, B, mu, eta, n_nodes=n_nodes, bessel=bessel
                )
                nu_safe = jnp.where(nu > 0.0, nu, 1.0)
                response = channels(nu_safe) * ((B > 0.0) * mask)  # (n_ch, P)
                w = phase_weights(phase, phase_coordinate(nu_safe), depth_ref, s_depth)
                return response * I, response * V, (response * Q)[None] * w[:, None]

            D = point_taylor(factors, alphas)
            return {
                a: tuple(jnp.einsum("...cp,pl,pk->...clk", q, pl, pk) for q in D[a])
                for a in D
            }

        return sum_point_blocks(points, flat, block)

    return sum_harmonics(one, kernel.harmonics(), chunk)


ROUTES = {"product": project_product, "tensor": project_tensor}

__all__ = [
    "multi_indices",
    "lower_closure",
    "level_plan",
    "level_cost",
    "reachable",
    "point_taylor",
    "project_product",
    "project_tensor",
    "ROUTES",
]
