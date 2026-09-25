"""Second derivatives that pass through the value of ``transfer_slab``.

An outer derivative of the primal output of a ``jvp``, ``vjp`` or
``linearize`` differentiates the value computation, not the tangent rule; so
does ``lax.scan`` under ``vjp``, whose partial evaluation inlines the primal.
Up to round 3 of 0.3.0 that was plain autodiff through the source column scaled
by ``2^-k``, which loses the source tangent to underflow at extreme scale:
relative error 0.115 at ``max|eps ds| = 1e306`` (thick K, ds 20), 1.7e-2 for two
chained slabs and 3.3e-2 for ``transfer_los`` (weak K, ds 1). The primal now
re-enters the rule, and above the exposure threshold its value carries the
derivatives of ``Phi S + G eps`` (``tests/test_transfer_companion_boundaries.py``).

Reference: plain autodiff of ``Phi S + G eps`` from the 8x8 exponential,
``exp([[-K ds, I], [0, 0]]) = [[Phi, G / ds], [0, I]]``, which does not depend
on the scale of ``eps``. Tolerance 1e-13 of the largest entry; measured at most
6e-16 after the fix.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.rm import _pow2
from syncmoments.transfer import (
    _source_exponent,
    mueller_matrix,
    transfer_los,
    transfer_slab,
)

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
WEAK = mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1)
S0 = jnp.array([1.0, 0.2, -0.1, 0.05])
D = jnp.array([1.0, 0.3, -0.2, 0.1])
SCALES = (1.0, 1e297, 1e303, 1e306)
WRAPS = {"eager": lambda f: f, "jit": jax.jit}
TOL = 1e-13


def ref_slab(S, eps, K, ds):
    A = jnp.zeros((8, 8)).at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4))
    E = jax.scipy.linalg.expm(A)
    return E[:4, :4] @ S + (E[:4, 4:] * ds) @ eps


def plain_5x5(S, eps, K, ds):
    """The value as the plain implementation computes it (no derivative rule)."""
    column = eps * ds
    k = _source_exponent(jax.lax.stop_gradient(jnp.max(jnp.abs(column))))
    M = jnp.zeros((5, 5), dtype=column.dtype).at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(column * _pow2(-k, column.dtype))
    y0 = jnp.concatenate([S, _pow2(k, column.dtype)[None]])
    return (jax.scipy.linalg.expm(M, max_squarings=32) @ y0)[:4]


def scan_of(slab):
    def los(S, eps_s, K_s, ds):
        body = lambda S, x: (slab(S, x[0], x[1], ds), None)  # noqa: E731
        return jax.lax.scan(body, S, (eps_s, K_s))[0]

    return los


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("K,ds", [(THICK, 20.0), (WEAK, 1.0)], ids=["thick", "weak"])
def test_forward_derivative_of_the_jvp_primal(K, ds, scale, wrap):
    """``jacfwd`` of the primal output of a ``jvp`` (critic p1: 0.115 at 1e306)."""
    eps = D * scale / ds

    def primal_of_jvp(slab):
        return lambda e: jax.jvp(lambda e_: slab(S0, e_, K, ds), (e,), (D,))[0]

    got = WRAPS[wrap](jax.jacfwd(primal_of_jvp(transfer_slab)))(eps)
    assert rel(got, jax.jacfwd(primal_of_jvp(ref_slab))(eps)) < TOL


@pytest.mark.parametrize("scale", (1.0, 1e306))
def test_two_chained_slabs_in_every_second_order_mode(scale):
    """Two slabs sharing ``K``: all four nestings of ``jacfwd``/``jacrev`` for
    ``d2 out / dK d eps`` (critic p2: forward outer modes off by 1.7e-2)."""

    def two(slab):
        return lambda e, K: slab(slab(S0, e, K, 1.0), 0.5 * e, 0.7 * K, 1.0)

    eps = D * scale
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            got = outer(inner(two(transfer_slab), 1), 0)(eps, WEAK)
            want = outer(inner(two(ref_slab), 1), 0)(eps, WEAK)
            assert rel(got, want) < TOL, (outer.__name__, inner.__name__)


@pytest.mark.parametrize("scale", (1.0, 1e306))
def test_two_chained_slabs_hessian_is_symmetric(scale):
    """The full Hessian of a scalar in ``(eps, K)``: every order agrees with
    the reference and the mixed blocks are transposes of each other."""
    x0 = jnp.concatenate([D * scale, WEAK.ravel()])
    w = jnp.array([1.0, -0.5, 0.25, 2.0])

    def loss(slab):
        def f(x):
            e, K = x[:4], x[4:].reshape(4, 4)
            return w @ slab(slab(S0, e, K, 1.0), 0.5 * e, 0.7 * K, 1.0)

        return f

    want = jax.hessian(loss(ref_slab))(x0)
    for order in (jax.hessian, lambda f: jax.jacfwd(jax.jacfwd(f))):
        H = order(loss(transfer_slab))(x0)
        assert rel(H, want) < TOL
        assert rel(H.T, H) < TOL
    H = jax.jacrev(jax.jacrev(loss(transfer_slab)))(x0)
    assert rel(H, want) < TOL


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("scales", [(1.0, 1.0), (1e306, 1e306), (1.0, 1e306)])
def test_line_of_sight_forward_over_reverse(scales, wrap):
    """``transfer_los`` (``lax.scan``): ``jacfwd`` in ``eps_s`` of the primal of
    ``vjp`` and ``linearize`` in ``K_s``, and of ``jacrev`` in ``K_s`` (critic
    p3: 2.5e-2 and 3.3e-2). The mixed path takes the companion route."""
    eps_s = jnp.stack([D * scales[0], 0.5 * D * scales[1]])
    K_s = jnp.stack([WEAK, 0.7 * WEAK])
    ours = lambda e, K: transfer_los(S0, e, K, 1.0)  # noqa: E731
    ref = lambda e, K: scan_of(ref_slab)(S0, e, K, 1.0)  # noqa: E731
    wraps = {
        "vjp primal": lambda f: lambda e: jax.vjp(lambda K: f(e, K), K_s)[0],
        "linearize primal": lambda f: lambda e: jax.linearize(lambda K: f(e, K), K_s)[
            0
        ],
        "jacrev K": lambda f: lambda e: jax.jacrev(f, 1)(e, K_s),
    }
    for name, outer in wraps.items():
        got = WRAPS[wrap](jax.jacfwd(outer(ours)))(eps_s)
        assert rel(got, jax.jacfwd(outer(ref))(eps_s)) < TOL, name


@pytest.mark.parametrize("scale", (1.0, 1e306))
def test_line_of_sight_hessian_orders_agree(scale):
    """Three slabs, a scalar of the final Stokes vector, all 60 inputs: every
    nesting agrees with the reference and is symmetric."""
    K_s = jnp.stack([THICK * 0.1, WEAK, mueller_matrix(0.2, 0, 0, 0, 0, 0, 1.3)])
    eps_s = jnp.asarray(np.outer([1.0, -0.5, 0.25], D) * scale / 2.0)
    x0 = jnp.concatenate([eps_s.ravel(), K_s.ravel()])
    w = jnp.array([1.0, -0.5, 0.25, 2.0])

    def loss(los):
        return lambda x: w @ los(S0, x[:12].reshape(3, 4), x[12:].reshape(3, 4, 4), 2.0)

    want = np.asarray(jax.hessian(loss(scan_of(ref_slab)))(x0))
    f = loss(transfer_los)
    orders = {
        "fwd-rev": jax.jit(jax.hessian(f)),
        "fwd-fwd": jax.jacfwd(jax.jacfwd(f)),
        "rev-rev": jax.jacrev(jax.jacrev(f)),
        "rev-fwd": jax.jacrev(jax.jacfwd(f)),
    }
    for name, order in orders.items():
        H = order(x0)
        assert rel(H, want) < TOL and rel(H.T, H) < TOL, name


def _random_slabs(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = 10.0 ** rng.uniform(-3, 2)
        K = mueller_matrix(
            a, *(a * rng.uniform(-0.9, 0.9, 3) / 2), *rng.uniform(-5, 5, 3)
        )
        ds = 10.0 ** rng.uniform(-2, 1.5)
        top = 10.0 ** rng.uniform(-300, 307.5)
        eps = rng.uniform(-1, 1, 4) * (top * min(ds, 1.0)) / ds
        S = rng.uniform(-1, 1, 4) * 10.0 ** rng.uniform(-3, 3)
        yield jnp.asarray(S), jnp.asarray(eps), K, jnp.asarray(ds)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
def test_primal_is_bit_identical_over_random_slabs(wrap):
    """Slabs with ``max|eps ds|`` from 1e-300 to 3e307 (both sides of the
    threshold; 300 under jit, 60 eagerly): the value, also as the primal of a
    ``jvp``, equals the plain 5x5 value."""
    ours, plain = WRAPS[wrap](transfer_slab), WRAPS[wrap](plain_5x5)
    primal_of_jvp = WRAPS[wrap](
        lambda S, e, K, d: jax.jvp(lambda e_: transfer_slab(S, e_, K, d), (e,), (e,))[0]
    )
    for S, eps, K, ds in _random_slabs(300 if wrap == "jit" else 60, seed=4):
        expected = plain(S, eps, K, ds)
        assert_array_equal(ours(S, eps, K, ds), expected)
        assert_array_equal(primal_of_jvp(S, eps, K, ds), expected)


def test_signed_zero_of_the_value_is_kept():
    """The companion subtracts an exact ``+0``, so a ``-0.0`` entry survives."""
    for scale in (1.0, 1e306):
        eps = jnp.array([1.0, 0.0, 0.0, 0.0]) * scale
        args = (jnp.array([0.0, -0.0, 0.0, -0.0]), eps, jnp.eye(4), 1.0)
        got = np.asarray(transfer_slab(*args))
        assert_array_equal(np.signbit(got), np.signbit(np.asarray(plain_5x5(*args))))


def test_vmap_batch_mixing_normal_and_extreme_sources():
    """Under ``vmap`` the branch is chosen once for the batch (any member above
    the threshold): values stay bit-identical and derivatives exact for both."""
    eps_s = jnp.stack([jnp.stack([D, 0.5 * D]), jnp.stack([D, 0.5 * D]) * 1e306])
    K_s = jnp.stack([WEAK, 0.7 * WEAK])
    ours = lambda e: transfer_los(S0, e, K_s, 1.0)  # noqa: E731
    ref = lambda e: scan_of(ref_slab)(S0, e, K_s, 1.0)  # noqa: E731
    values = jax.jit(jax.vmap(ours))(eps_s)
    for i in range(2):
        assert_array_equal(values[i], ours(eps_s[i]))

    def outer(f):
        return lambda e: jax.jacfwd(lambda e_: jax.vjp(f, e_)[0])(e)

    J = jax.jit(jax.vmap(outer(ours)))(eps_s)
    for i in range(2):
        assert rel(J[i], outer(ref)(eps_s[i])) < TOL
