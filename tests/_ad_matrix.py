"""Transform harness of the AD-transform matrix (``tests/test_ad_transform_matrix*``).

A ``Group`` is one entry point restricted to one differentiable argument
group: ``f(x)`` maps the group's pytree ``x`` to a real 1D array (complex
outputs as real and imaginary parts), ``ref`` is an independent plain-``jnp``
implementation of the same map. Each transform is applied to ``f`` and
compared with the same quantity assembled from plain forward-mode autodiff
of ``ref`` (``jacfwd`` and ``jacfwd(jacfwd)``), so a transform that the
package's custom rules break cannot also break the expected value.

Comparison: per leaf, ``max|got - want| <= RTOL * scale`` with ``scale`` the
leaf's largest ``|want|`` (the whole result's largest when the leaf is
identically zero; then an exact zero is required if that is zero too), and
every entry of ``got`` finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

RTOL = 1e-12
BATCH = 4  # vmap batch; nested vmap uses (2, 2)


@dataclass(frozen=True)
class Group:
    """Entry point ``name`` restricted to the argument ``keys``."""

    name: str
    keys: tuple
    f: Callable
    ref: Callable
    x: dict
    positive: frozenset = field(default_factory=frozenset)


def _unit(shape, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=shape)


def batch(group):
    """``BATCH`` perturbed copies of ``group.x`` (positive keys multiplicatively)."""
    out = {}
    for i, (key, value) in enumerate(sorted(group.x.items())):
        value = np.asarray(value)
        u = _unit((BATCH, *value.shape), 100 + i)
        if key in group.positive:
            out[key] = jnp.asarray(value * (1 + 0.05 * u))
        else:
            spread = np.abs(value) + 0.1 * np.max(np.abs(value)) + 1e-3
            out[key] = jnp.asarray(value + 0.05 * u * spread)
    return out


def tangent(group, seed=7):
    """A fixed tangent pytree like ``group.x``."""
    return {
        k: jnp.asarray(_unit(np.shape(v), seed + i))
        for i, (k, v) in enumerate(sorted(group.x.items()))
    }


def cotangent(group):
    n = np.asarray(group.ref(group.x)).shape[0]
    return jnp.asarray(_unit((n,), 3))


def _scalar(fun, c):
    return lambda x: jnp.vdot(c, fun(x))


def _member(xb, i):
    return jax.tree_util.tree_map(lambda a: a[i], xb)


def _pair_batch(xb):
    return jax.tree_util.tree_map(lambda a: a.reshape(2, BATCH // 2, *a.shape[1:]), xb)


def _unpair(tree):
    return jax.tree_util.tree_map(lambda a: a.reshape(BATCH, *a.shape[2:]), tree)


class Expected:
    """Quantities from plain forward-mode autodiff of ``group.ref`` (jitted)."""

    def __init__(self, group):
        c = cotangent(group)
        s = _scalar(group.ref, c)
        self.c, self.v, self.xb = c, tangent(group), batch(group)
        self.value = jax.jit(group.ref)
        self.jac = jax.jit(jax.jacfwd(group.ref))
        self.grad = jax.jit(jax.jacfwd(s))
        self.hess = jax.jit(jax.jacfwd(jax.jacfwd(s)))
        self.jvp = jax.jit(lambda x, v: jax.jvp(group.ref, (x,), (v,))[1])

    def members(self, fun):
        return jnp.stack([fun(_member(self.xb, i)) for i in range(BATCH)])

    def stacked(self, fun):
        per = [fun(_member(self.xb, i)) for i in range(BATCH)]
        return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)

    def block_hess(self):
        """``hessian`` of a sum over the batch: per-member Hessians on the diagonal."""
        per = [self.hess(_member(self.xb, i)) for i in range(BATCH)]
        out = {}
        for a, row in per[0].items():
            out[a] = {}
            for b in row:
                sa, sb = np.shape(self.xb[a])[1:], np.shape(self.xb[b])[1:]
                full = np.zeros((BATCH, *sa, BATCH, *sb))
                for i in range(BATCH):
                    full[(i, *[slice(None)] * len(sa), i)] = per[i][a][b]
                out[a][b] = full
        return out

    def hvp(self, x):
        h = self.hess(x)
        return {
            a: sum(
                jnp.tensordot(h[a][b], self.v[b], axes=np.ndim(self.v[b]))
                for b in self.v
            )
            for a in h
        }


def _scan_sum(s, xb):
    return jax.lax.scan(lambda acc, x: (acc + s(x), None), jnp.zeros(()), xb)[0]


def transforms(group, e):
    """``{name: (thunk_got, thunk_want)}`` for every transform of the matrix."""
    f, x, xb, c, v = group.f, group.x, e.xb, e.c, e.v
    s = _scalar(f, c)
    vs = lambda xb_: jnp.sum(jax.vmap(s)(xb_))  # noqa: E731
    scan_value_grad = jax.value_and_grad(lambda xb_: _scan_sum(s, xb_))
    scan = lambda xr: _scan_sum(s, xr)  # noqa: E731
    nested, pair = jax.vmap(jax.vmap(s)), _pair_batch(xb)
    want_scan = lambda: (  # noqa: E731
        jnp.sum(e.members(lambda y: jnp.vdot(c, e.value(y)))),
        e.stacked(e.grad),
    )
    return {
        "eager": (lambda: f(x), lambda: e.value(x)),
        "jit": (lambda: jax.jit(f)(x), lambda: e.value(x)),
        "vmap": (lambda: jax.vmap(f)(xb), lambda: e.members(e.value)),
        "nested_vmap": (
            lambda: jax.vmap(jax.vmap(f))(_pair_batch(xb)),
            lambda: e.members(e.value).reshape(2, BATCH // 2, -1),
        ),
        "grad_sum_vmap": (lambda: jax.grad(vs)(xb), lambda: e.stacked(e.grad)),
        "jit_grad_sum_vmap": (
            lambda: jax.jit(jax.grad(vs))(xb),
            lambda: e.stacked(e.grad),
        ),
        "vmap_grad": (lambda: jax.vmap(jax.grad(s))(xb), lambda: e.stacked(e.grad)),
        "jacfwd": (lambda: jax.jacfwd(f)(x), lambda: e.jac(x)),
        "jacrev": (lambda: jax.jacrev(f)(x), lambda: e.jac(x)),
        "hessian": (lambda: jax.hessian(s)(x), lambda: e.hess(x)),
        "jit_hessian": (lambda: jax.jit(jax.hessian(s))(x), lambda: e.hess(x)),
        "rev_rev": (lambda: jax.jacrev(jax.jacrev(s))(x), lambda: e.hess(x)),
        "fwd_fwd": (lambda: jax.jacfwd(jax.jacfwd(s))(x), lambda: e.hess(x)),
        "rev_fwd": (lambda: jax.jacrev(jax.jacfwd(s))(x), lambda: e.hess(x)),
        "linearize": (lambda: jax.linearize(f, x)[1](v), lambda: e.jvp(x, v)),
        "vjp": (lambda: jax.vjp(f, x)[1](c)[0], lambda: e.grad(x)),
        "jvp_of_linearize_primal": (
            lambda: jax.jvp(lambda y: jax.linearize(f, y)[0], (x,), (v,))[1],
            lambda: e.jvp(x, v),
        ),
        "jvp_of_vjp_primal": (
            lambda: jax.jvp(lambda y: jax.vjp(f, y)[0], (x,), (v,))[1],
            lambda: e.jvp(x, v),
        ),
        "jvp_of_vjp": (
            lambda: jax.jvp(lambda y: jax.vjp(f, y)[1](c)[0], (x,), (v,))[1],
            lambda: e.hvp(x),
        ),
        "vjp_of_jvp": (
            lambda: jax.vjp(lambda y: jax.jvp(s, (y,), (v,))[1], x)[1](jnp.ones(()))[0],
            lambda: e.hvp(x),
        ),
        "checkpoint_grad": (
            lambda: jax.grad(jax.checkpoint(s))(x),
            lambda: e.grad(x),
        ),
        "scan": (lambda: scan_value_grad(xb), want_scan),
        "vmap_scan": (
            lambda: _unpair(jax.grad(lambda p: jnp.sum(jax.vmap(scan)(p)))(pair)),
            lambda: e.stacked(e.grad),
        ),
        "grad_nested_vmap": (
            lambda: _unpair(jax.grad(lambda p: jnp.sum(nested(p)))(pair)),
            lambda: e.stacked(e.grad),
        ),
        "checkpoint_grad_sum_vmap": (
            lambda: jax.grad(jax.checkpoint(vs))(xb),
            lambda: e.stacked(e.grad),
        ),
        "vmap_hessian": (
            lambda: jax.vmap(jax.hessian(s))(xb),
            lambda: e.stacked(e.hess),
        ),
        "hessian_sum_vmap": (lambda: jax.hessian(vs)(xb), e.block_hess),
    }


TRANSFORMS = (
    "eager", "jit", "vmap", "nested_vmap", "grad_sum_vmap", "jit_grad_sum_vmap",
    "vmap_grad", "jacfwd", "jacrev", "hessian", "jit_hessian", "rev_rev",
    "fwd_fwd", "rev_fwd", "linearize", "vjp", "jvp_of_linearize_primal",
    "jvp_of_vjp_primal", "jvp_of_vjp", "vjp_of_jvp", "checkpoint_grad", "scan",
    "vmap_scan", "grad_nested_vmap", "checkpoint_grad_sum_vmap", "vmap_hessian",
    "hessian_sum_vmap",
)  # fmt: skip


def mismatch(got, want, rtol=RTOL):
    """``None`` when ``got`` matches ``want`` (module docstring), else a message."""
    got_leaves, got_tree = jax.tree_util.tree_flatten(got)
    want_leaves, want_tree = jax.tree_util.tree_flatten(want)
    if got_tree != want_tree:
        return f"structure {got_tree} != {want_tree}"
    top = max((float(np.max(np.abs(w), initial=0.0)) for w in want_leaves), default=0)
    worst = 0.0
    for g, w in zip(got_leaves, want_leaves):
        g, w = np.asarray(g, dtype=float), np.asarray(w, dtype=float)
        if g.shape != w.shape:
            return f"shape {g.shape} != {w.shape}"
        if not np.all(np.isfinite(g)):
            return "non-finite entries"
        scale = float(np.max(np.abs(w), initial=0.0)) or top
        err = float(np.max(np.abs(g - w), initial=0.0))
        if scale == 0.0:
            if err != 0.0:
                return f"expected exact zeros, error {err:.3g}"
            continue
        worst = max(worst, err / scale)
    return None if worst <= rtol else f"relative error {worst:.3g} > {rtol:g}"


def check(group, name, expected=None):
    """Run transform ``name`` on ``group``; ``None`` on success, else a message."""
    e = expected if expected is not None else Expected(group)
    got, want = transforms(group, e)[name]
    return mismatch(got(), want())
