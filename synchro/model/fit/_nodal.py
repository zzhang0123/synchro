"""Nodal (fixed-node, nonnegative-weight) fit core for ``fit_nodal``.

Private helper of :mod:`synchro.model.fit.nonlinear`. Implements
``extra eq: nodal distribution`` as a linear model in the node weights:
``m(w) = X^T w`` with ``X`` the ``(S, n_real)`` matrix of row monomials
``P_l(mu_i) P_k(eta_i) z_gamma^r z_B^s z_depth^b`` (and the real/imaginary
parts of the ``e^{2i phi}`` rows) of every node, in the ``to_vector`` layout
of ``JointMoments``. ``X`` is never stored: both ``X^T w`` and ``X y`` are
chunked ``lax.scan`` sweeps over the nodes, so the memory is
``O(S) + O(chunk * n_real)``.

The optimiser is a monotone exponentiated-gradient (multiplicative) method
on the probability simplex with the amplitude profiled out at every step:
``w <- w exp(-eta g) / sum(.)`` with ``g`` the gradient of the half
chi-square at the profiled amplitude, the relative step ``eta`` grown on an
accepted step and halved on a rejected one. It keeps ``w >= 0`` and
``sum w = 1`` at every iterate (a zero weight stays zero). Convergence is
judged by the simplex stationarity residual; nothing here certifies that
the optimum is reached to a given accuracy beyond the reported flag.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

CHUNK = 4096
STEP_GROW = 1.2
STEP_MIN = 1e-14


def _legendre_table(x, degree):
    columns = [jnp.ones_like(x)]
    if degree >= 1:
        columns.append(x)
    for n in range(1, degree):
        columns.append(((2 * n + 1) * x * columns[n] - n * columns[n - 1]) / (n + 1))
    return jnp.stack(columns, axis=-1)


def _power_table(x, degree):
    columns = [jnp.ones_like(x)]
    for _ in range(degree):
        columns.append(columns[-1] * x)
    return jnp.stack(columns, axis=-1)


def _features(leaf, rows, maxima, n0, reference):
    """``(chunk, n_real)`` real features of one chunk of nodes."""
    mu, eta, gamma, B, phi, depth = leaf
    z = reference.z(gamma, B, depth)
    tables = (
        _legendre_table(mu, maxima[0]),
        _legendre_table(eta, maxima[1]),
        _power_table(z[..., 0], maxima[2]),
        _power_table(z[..., 1], maxima[3]),
        _power_table(z[..., 2], maxima[4]),
    )
    chi = jnp.ones(mu.shape + (rows.shape[0],), dtype=mu.dtype)
    for table, exponent in zip(tables, rows.T):
        chi = chi * table[..., exponent]
    real, cplx = chi[:, :n0], chi[:, n0:]
    c, s = jnp.cos(2.0 * phi)[:, None], jnp.sin(2.0 * phi)[:, None]
    return jnp.concatenate([real, cplx * c, cplx * s], axis=1)


class NodeOperator:
    """Chunked ``X^T w`` and ``X y`` for fixed nodes (see the module docstring)."""

    def __init__(self, nodes, index, reference):
        self.reference = reference
        rows = np.concatenate(
            [index.exponents(index.h0), index.exponents(index.h2)], axis=0
        )
        self.rows = jnp.asarray(rows)
        self.maxima = tuple(int(m) for m in rows.max(axis=0)) if rows.size else (0,) * 5
        self.n0 = int(index.n0)
        self.n_real = int(index.n_real)
        self.size = int(nodes.size)
        self.chunk = min(self.size, CHUNK)
        self.pad = (-self.size) % self.chunk
        self.leaves = tuple(
            self._padded(v).reshape(-1, self.chunk)
            for v in (nodes.mu, nodes.eta, nodes.gamma, nodes.B, nodes.phi, nodes.depth)
        )

    def _padded(self, value):
        return jnp.pad(jnp.asarray(value, dtype=float), (0, self.pad), mode="edge")

    def _chunk_features(self, leaf):
        return _features(leaf, self.rows, self.maxima, self.n0, self.reference)

    def moments(self, w):
        """``X^T w`` (``(n_real,)``, unnormalised) for weights ``w`` ``(S,)``."""
        w_chunks = jnp.pad(jnp.asarray(w, dtype=float), (0, self.pad)).reshape(
            -1, self.chunk
        )

        def body(acc, xs):
            w_c, leaf = xs
            return acc + w_c @ self._chunk_features(leaf), None

        init = jnp.zeros(self.n_real)
        out, _ = jax.lax.scan(body, init, (w_chunks, self.leaves))
        return out

    def adjoint(self, y):
        """``X y`` (``(S,)``) for a moment-space vector ``y`` ``(n_real,)``."""
        y = jnp.asarray(y, dtype=float)

        def body(carry, leaf):
            return carry, self._chunk_features(leaf) @ y

        _, out = jax.lax.scan(body, None, self.leaves)
        return out.reshape(-1)[: self.size]


def simplex_fit(forward, adjoint, target, w0, *, iters, step, tol):
    """Monotone exponentiated-gradient fit of ``0.5 ||target - A forward(w)||^2``.

    ``forward(w) -> (n,)`` is linear in ``w`` and ``adjoint`` is its
    transpose; ``A`` is profiled at every iterate (``<Gw, t>/<Gw, Gw>``) and
    kept at its previous positive value when the profile is not positive,
    so the joint objective in ``(w, A)`` never increases. The update is
    ``w <- softmax(log w - eta g)`` with ``g`` the gradient at the profiled
    amplitude; ``eta`` starts at ``step / (max g0 - min g0)`` (so ``step`` is
    a fraction of the initial gradient range), grows by ``STEP_GROW`` after
    an accepted step and halves after a rejected one (``f`` may never
    increase). Returns ``(w, A, fun, n_iter, converged, kkt)`` with ``fun``
    the final half chi-square and ``kkt`` the simplex stationarity residual
    ``max_i w_i |g_i - <g, w>| / (1 + max_i |g_i|)``; ``converged`` is
    ``kkt <= tol``. The loop also stops when ``eta`` underflows
    ``STEP_MIN`` (no accepted step possible in floating point).
    """
    target = jnp.asarray(target, dtype=float)
    w0 = jnp.asarray(w0, dtype=float)

    def profile(w, amp_prev):
        """Profiled amplitude, or the previous one when the profile is <= 0."""
        gw = forward(w)
        denom = gw @ gw
        safe = jnp.where(denom > 0.0, denom, 1.0)
        amp = jnp.where(denom > 0.0, (gw @ target) / safe, 0.0)
        amp = jnp.where(amp > 0.0, amp, amp_prev)
        r = target - amp * gw
        return amp, r, 0.5 * (r @ r)

    def gradient(amp, r):
        return -amp * adjoint(r)

    def stationarity(w, g):
        return jnp.max(w * jnp.abs(g - g @ w)) / (1.0 + jnp.max(jnp.abs(g)))

    gw0 = forward(w0)
    scale = jnp.sqrt(gw0 @ gw0)
    fallback = jnp.sqrt(target @ target) / jnp.where(scale > 0.0, scale, 1.0)
    amp0, r0, f0 = profile(w0, fallback)
    g0 = gradient(amp0, r0)
    spread0 = jnp.max(g0) - jnp.min(g0)
    eta0 = float(step) / jnp.where(spread0 > 0.0, spread0, 1.0)
    state = (w0, amp0, f0, g0, eta0, 0, stationarity(w0, g0) <= tol, False)

    def cond(state):
        _, _, _, _, _, k, converged, stuck = state
        return (k < iters) & ~converged & ~stuck

    def body(state):
        w, amp, f, g, eta, k, _, _ = state
        w_new = jax.nn.softmax(jnp.log(w) - eta * (g - jnp.min(g)))
        amp_new, r_new, f_new = profile(w_new, amp)
        accept = f_new <= f
        g_new = jnp.where(accept, gradient(amp_new, r_new), g)
        w = jnp.where(accept, w_new, w)
        amp = jnp.where(accept, amp_new, amp)
        f = jnp.where(accept, f_new, f)
        eta = jnp.where(accept, eta * STEP_GROW, 0.5 * eta)
        converged = stationarity(w, g_new) <= tol
        return (w, amp, f, g_new, eta, k + 1, converged, eta < STEP_MIN)

    w, amp, f, g, _, k, converged, _ = jax.lax.while_loop(cond, body, state)
    return w, amp, f, k, converged, stationarity(w, g)


__all__ = ["NodeOperator", "simplex_fit", "CHUNK"]
