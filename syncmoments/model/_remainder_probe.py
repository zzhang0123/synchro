"""Sample probes of the remainder inputs (private helper of ``bounds.py``).

Implements ``RemainderInputs.from_samples``: the order-``N+1`` derivative
envelopes ``H`` of the kernel's Legendre projections in ``z = (z_gamma, z_B,
z_depth)``, the absolute moments
``<|P_l P_k| ||z||^{N+1}>``, the angular residual ``<|K - K_L|>`` measured on
the samples, and the ``app: depth moments`` inputs. For a capped truncation
(``Truncation.is_capped()``) ``margin_envelope`` and ``margin_moments`` give
the per-multi-index inputs of the lower-set margin form
(``_remainder_route.lower_set_margin``). Derivatives (``method``):
``"taylor"`` (default for a kernel with ``angular_taylor`` and
``derivatives="analytic"``) takes the ``(z_gamma, z_B)`` partials from
``angular_taylor`` at each probe point, one pass per depth order ``b`` with
the single weight ``d^b_{z_depth} exp(i tau (depth + s_depth z_depth)) =
exp(i tau depth) (i tau s_depth)^b`` (``_basis_taylor.partials_in_z3``);
``"jacfwd"`` nests ``jax.jacfwd`` of ``angular_projection`` in ``z_3``. Both
differentiate the same computed projection (tests: ``1e-12`` relative).
Units follow ``syncmoments.model.bounds``.

Memory: an order-``q`` nested ``jacfwd`` in ``z_3`` carries the primal and
three tangents per level, ``4^q`` copies of the projection's working set;
the projection runs with ``kernel.for_tangents(tangent_divisor(kernel, q))``.
The divisor is ``4^q`` (one ``eta`` node or angular point at the floor)
except for the harmonic kernel with ``derivatives="analytic"`` (recurrence
rule for the Bessel triple), which keeps its budget. Measured peaks, JAX
0.10.0 / 0.10.2, benchmark ``L = 8``, ``N = 2``, ``q = 3``: ``4^q`` lowers
that harmonic probe only from 2.2 / 2.0 to 1.7 / 1.8 GB and makes it 1.8 to
2.5 times slower per point; it lowers the ``"autodiff"`` harmonic probe from
4.0 / 10.1 to 2.0 / 2.9 GB (1.7 times slower) and the continuum probe of
README example (b) from 1.85 / 5.0 to 1.16 / 1.85 GB.
``angular_residual`` vmaps ``channel_modes`` and ``angular_projection`` over
a sample batch capped by ``_direct.sample_batch`` (the kernel's
``samples_per_step``) and runs the projection at ``chunk_budget // batch``.
Blocking changes only the summation order.
Nothing here is a bound: the Frobenius norm of the derivative tensor at
finitely many probe points is an estimate of the operator-norm supremum over
the reference-to-support segments, and the margin envelope at finitely many
path points is an estimate of its path supremum (``kind="estimate"``).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ._basis_taylor import partials_in_z3, symmetric_frobenius
from ._direct import sample_batch
from ._harmonic_taylor import multi_indices
from ._kernel_helpers import legendre_table
from .phase import TaylorPhase

METHODS = ("auto", "taylor", "jacfwd")
_COPIES_PER_LEVEL = 4  # primal + one tangent per coordinate of z_3


def probe_method(kernel, method="auto") -> str:
    """``"taylor"`` or ``"jacfwd"`` for ``kernel`` (module docstring)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    has_taylor = callable(getattr(kernel, "angular_taylor", None))
    if method == "taylor" and not has_taylor:
        raise ValueError("method='taylor' needs a kernel with angular_taylor")
    if method == "auto":
        analytic = getattr(kernel, "derivatives", "analytic") == "analytic"
        return "taylor" if has_taylor and analytic else "jacfwd"
    return method


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


def _for_tangents(kernel, copies):
    """``kernel.for_tangents(copies)`` when the kernel has it, else ``kernel``."""
    divide = getattr(kernel, "for_tangents", None)
    return divide(int(copies)) if callable(divide) else kernel


def tangent_divisor(kernel, order) -> int:
    """``chunk_budget`` divisor of an order-``order`` nested ``jacfwd``: ``4^order``,
    ``1`` for the harmonic kernel with ``derivatives="analytic"`` (module docstring)."""
    analytic = getattr(kernel, "derivatives", None) == "analytic"
    if getattr(kernel, "name", None) == "harmonic" and analytic:
        return 1
    return _COPIES_PER_LEVEL ** int(order)


def _nested_jacfwd(basis, kernel, route, sample_depth, order):
    """Order-``order`` nested ``jacfwd`` of :func:`_point_function` in ``z_3``,
    at ``chunk_budget // tangent_divisor(kernel, order)``."""
    kernel = _for_tangents(kernel, tangent_divisor(kernel, order))
    fn = _point_function(basis, kernel, route, sample_depth)
    for _ in range(order):
        fn = jax.jacfwd(fn)
    return fn


def _taylor_norms(basis, kernel, route, sample_depth, order):
    """Point ``-> (|D^order I|, |D^order V|, |D^order P|_3, |D^order P|_2)``."""
    betas = [(r, s, order - r - s) for r, s in multi_indices(order)]
    fn = partials_in_z3(basis, kernel, route, sample_depth, betas)

    def norms(point):
        d = fn(point)
        plane = {beta[:2]: v for beta, v in d.items() if beta[2] == 0}
        return (
            symmetric_frobenius({b: v[0] for b, v in plane.items()}, order, 2),
            symmetric_frobenius({b: v[1] for b, v in plane.items()}, order, 2),
            symmetric_frobenius({b: v[2] for b, v in d.items()}, order, 3),
            symmetric_frobenius({b: v[2] for b, v in plane.items()}, order, 2),
        )

    return norms


def _frobenius(tensor, order, dims):
    """Frobenius norm over the trailing ``order`` axes restricted to ``dims`` each."""
    for i in range(order):
        tensor = jnp.take(tensor, jnp.arange(dims), axis=-1 - i)
    flat = tensor.reshape(tensor.shape[: tensor.ndim - order] + (-1,))
    return jnp.sqrt(jnp.sum(jnp.abs(flat) ** 2, axis=-1))


def derivative_envelope(
    samples, basis, kernel, *, phase, segment_points, method="auto"
):
    """``(H (n_ch, 3, n_lk), H_2d (n_ch, n_lk))`` of the order-``N+1`` derivatives.

    Probe points: ``z = 0`` and ``t z_n`` for every sample ``n`` and every
    ``t`` in ``segment_points``; ``H`` is the maximum Frobenius norm of the
    order-``N+1`` tensor in ``z_3`` (``I, V`` do not depend on ``z_depth``);
    ``H_2d`` restricts the ``P`` tensor to ``(z_gamma, z_B)`` (``app: depth
    moments``). ``method``: see :func:`probe_method`.
    """
    order = basis.index.truncation.N + 1
    route, sample_depth = _probe_route(phase)
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    points = [jnp.zeros((1, 3))] + [float(t) * z for t in segment_points]
    points = jnp.concatenate(points, axis=0)
    if probe_method(kernel, method) == "taylor":
        norms = _taylor_norms(basis, kernel, route, sample_depth, order)
    else:
        fn = _nested_jacfwd(basis, kernel, route, sample_depth, order)

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


def _path_points(z, k, ts, depth_layout):
    """``p_k(t) = (z_1..z_{k-1}, t z_k, 0..)`` per sample and ``t``, ``(len(ts) S, 3)``.

    In the ``app: depth moments`` layout the depth coordinate stays at the
    sample (the 2D margin acts at the emitter depth).
    """
    points = []
    for t in ts:
        point = jnp.concatenate(
            [z[:, :k], t * z[:, k : k + 1], jnp.zeros_like(z[:, k + 1 :])], axis=1
        )
        if depth_layout:
            point = point.at[:, 2].set(z[:, 2])
        points.append(point)
    return jnp.concatenate(points, axis=0)


def margin_envelope(samples, basis, kernel, *, phase, segment_points, method="auto"):
    """``(n_ch, 3, n_lk, n_m)``: ``max |d^beta K_{X;lk}|`` on the margin paths.

    For each ``beta`` of ``truncation.margin_rows()`` (order ``n_m``) the
    maximum modulus over the path points ``p_k(t)``, ``k`` the first nonzero
    coordinate of ``beta``, ``t`` in ``{0} union segment_points``, every
    sample; ``X`` in the order ``I, P, V``. ``method``: see
    :func:`probe_method` (``"jacfwd"``: nested ``jacfwd`` per order).
    """
    rows = basis.index.truncation.margin_rows()
    route, sample_depth = _probe_route(phase)
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    depth_layout = basis.index.truncation.depth_degree is not None
    ts = (0.0,) + tuple(float(t) for t in segment_points)
    if probe_method(kernel, method) == "taylor":
        return _taylor_margin(basis, kernel, route, sample_depth, rows, z, ts)
    derivatives = {
        q: _nested_jacfwd(basis, kernel, route, sample_depth, q)
        for q in sorted({sum(beta) for beta in rows})
    }
    columns = [None] * len(rows)
    for k in range(3):
        chosen = [m for m, beta in enumerate(rows) if _first(beta) == k]
        points = _path_points(z, k, ts, depth_layout) if chosen else None
        for q in sorted({sum(rows[m]) for m in chosen}):
            group = [m for m in chosen if sum(rows[m]) == q]
            axes = [_axes(rows[m]) for m in group]
            envelope = jax.lax.map(
                jax.jit(lambda pt, d=derivatives[q], a=axes: _picked(d(pt), a)), points
            )
            envelope = jnp.abs(envelope).max(0)
            for j, m in enumerate(group):
                columns[m] = envelope[..., j]
    return jnp.stack(columns, axis=-1)


def _taylor_margin(basis, kernel, route, sample_depth, rows, z, ts):
    """``margin_envelope`` through ``angular_taylor``, one pass per path group ``k``."""
    depth_layout = basis.index.truncation.depth_degree is not None
    columns = [None] * len(rows)
    for k in range(3):
        group = [m for m, beta in enumerate(rows) if _first(beta) == k]
        if not group:
            continue
        betas = [rows[m] for m in group]
        fn = partials_in_z3(basis, kernel, route, sample_depth, betas)

        def picked(point, fn=fn, betas=betas):
            d = fn(point)
            return jnp.stack(
                [jnp.stack([d[b][0], d[b][2], d[b][1]], axis=1) for b in betas],
                axis=-1,
            )

        points = _path_points(z, k, ts, depth_layout)
        envelope = jnp.abs(jax.lax.map(jax.jit(picked), points)).max(0)
        for j, m in enumerate(group):
            columns[m] = envelope[..., j]
    return jnp.stack(columns, axis=-1)


def _first(beta):
    return next(i for i, b in enumerate(beta) if b)


def _axes(beta):
    return tuple(i for i, b in enumerate(beta) for _ in range(b))


def _picked(tensors, axes):
    """``(n_ch, 3, n_lk, len(axes))`` entries ``d^beta`` of ``(I, V, P)`` in ``I, P, V`` order."""
    dI, dV, dP = tensors

    def pick(t):
        return jnp.stack([t[(Ellipsis,) + a] for a in axes], axis=-1)

    return jnp.stack([pick(dI), pick(dP), pick(dV)], axis=1)


def margin_moments(samples, basis):
    """``(n_lk, n_m)``: ``<|P_l P_k| |z^beta|>`` over the margin rows (exact sums)."""
    rows = np.asarray(basis.index.truncation.margin_rows(), dtype=int).reshape(-1, 3)
    w = samples.normalised_weights()
    z = basis.reference.z(samples.gamma, samples.B, samples.depth)
    mono = jnp.prod(jnp.abs(z)[:, None, :] ** rows[None], axis=-1)
    return jnp.einsum("s,sl,sm->lm", w, _pair_legendre(samples, basis.index), mono)


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
    ``(gamma, B)`` dependence is kept exact. Samples per ``lax.map`` step:
    ``batch_size`` capped by ``_direct.sample_batch``; the projection runs at
    ``chunk_budget // batch`` (module docstring).
    """
    index, channels = basis.index, basis.channels
    batch = sample_batch(kernel, channels, batch_size, samples.size)
    projector = _for_tangents(kernel, batch)
    route, sample_depth = _probe_route(phase)
    pairs = np.asarray(index.pairs, dtype=int).reshape(-1, 2)
    L_mu, L_eta = index.truncation.L_mu, index.truncation.L_eta
    depth_ref = basis.reference.depth_ref

    def one(leaf):
        gamma, B, mu, eta, depth = leaf
        depth = depth if sample_depth else depth_ref
        kw = dict(phase=route, depth_ref=depth, s_depth=jnp.ones(()))
        modes = kernel.channel_modes(channels, gamma, B, mu, eta, **kw)
        projected = projector.angular_projection(
            channels, gamma, B, truncation=index, **kw
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
    residual = jax.lax.map(one, leaves, batch_size=batch)
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
    "METHODS",
    "probe_method",
    "tangent_divisor",
    "symmetric_frobenius",
    "derivative_envelope",
    "absolute_moments",
    "angular_residual",
    "depth_rows",
    "depth_tail",
    "intrinsic_residual",
    "margin_envelope",
    "margin_moments",
]
