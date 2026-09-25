"""Closed-form second tangent of ``w / sum(w)``: identity, boundaries, transforms.

``syncmoments.rm._divide_by_sum`` computes ``p = r / R`` (``R = sum r``) with
the operations of the plain division and its JVP, so primal and first
derivatives must be bit-identical to the round-4 formulation
``optimization_barrier(core(w)) / sum(...)``. Its tangent ``_sum_tangent`` has a
closed-form JVP (forward-mode second derivatives)::

    d2p[dr, y] = -[(dr - p dR) Y + (y - p Y) dR] / R^2,

evaluated as ``-2^(k + 2s) S / Rb^2`` with ``x = dr 2^-k`` (``max |x| < 1``),
``Rb = R 2^s`` in ``[1, n]`` and ``S = (x - p X) Y + (y - p Y) X``: all scaling
is by exact powers of two applied once, after the sum.

Boundary validation (``~/.claude/rules/common/boundary-validation.md``): the
headroom shift ``s`` switches on at ``max(w) = 2^-958``. Both second-order
routes, the closed form (forward over forward) and the plain transposed
division (forward over reverse), are evaluated directly on log-weight
Hessians (representable) at ``2^-958 * 2^{-2..2}`` and at extreme ``n`` and
weight ratios, and must agree with each other and with the oracle.
"""

from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_weight_normalisation_oracle import (
    FACTOR,
    MODES,
    OFFSETS,
    PATTERN,
    TARGETS,
    classify,
    failures,
    hessian_theta,
    hessian_w,
    linear,
    variance,
    X,
)
from syncmoments.rm import (
    _HEADROOM,
    _normalised_weights,
    _relative_weights_wide,
    gaussian_rm_cumulants,
)
from syncmoments.expansion import mixed_moments


@eqx.filter_jit  # compiled as one unit, like ``_normalised_weights``
def _round4(w):
    """The round-4 formulation: plain division after the (unchanged) core."""
    scaled = jax.lax.optimization_barrier(_relative_weights_wide(w, _HEADROOM))
    return scaled / jnp.sum(scaled)


CASES = {
    "unit": PATTERN,
    "tiny": PATTERN * 1e-300,
    "headroom-edge": PATTERN * 2.0**-958,
    "subnormal-edge": np.array([1.0, 2.0**-1030, 2.0**-1022, 3.0]) * 2.0**-970,
    "huge": PATTERN * 1.7e308,
    "zeros": np.array([0.0, 1e-300, 3e-301, 0.0]),
    "wide": np.array([1.0, 2.0**-60, 2.0**-900, 0.25]) * 1e-200,
}


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("case", CASES)
def test_primal_and_first_derivatives_bit_identical_to_round4(case, jit):
    w = jnp.asarray(CASES[case])
    rng = np.random.default_rng(7)
    wrap = jax.jit if jit else (lambda f: f)
    pairs = [(wrap(_normalised_weights)(w), wrap(_round4)(w))]
    for d in [w, rng.normal(size=4), rng.normal(size=4) * 1e-300, np.eye(4)[1]]:
        d = jnp.asarray(d)
        jvp = lambda f: wrap(lambda w, d: jax.jvp(f, (w,), (d,))[1])(w, d)  # noqa: E731
        vjp = lambda f: wrap(lambda w, c: jax.vjp(f, w)[1](c)[0])(w, d)  # noqa: E731
        pairs += [(jvp(_normalised_weights), jvp(_round4))]
        pairs += [(vjp(_normalised_weights), vjp(_round4))]
    for J in (jax.jacfwd, jax.jacrev):
        pairs.append((wrap(J(_normalised_weights))(w), wrap(J(_round4))(w)))
    for got, ref in pairs:
        assert np.asarray(got).tobytes() == np.asarray(ref).tobytes()


@pytest.mark.parametrize("scale", [1.0, 1e-160, 1e-300, 2.0**-1022])
def test_second_tangent_of_p_is_exact_or_signed_inf_in_forward_mode(scale):
    """``jacfwd(jacfwd(p))`` never returns NaN: the closed form scales once."""
    w = jnp.asarray(PATTERN * scale)
    for wrap in (lambda f: f, jax.jit):
        second = wrap(jax.jacfwd(jax.jacfwd(_normalised_weights)))(w)
        for i in range(4):
            oracle = hessian_w(np.asarray(w), linear(np.eye(4)[i]))
            assert failures(second[i], oracle) == {}


def _theta_hessians(w, target):
    f, derivatives = TARGETS[target]
    g = lambda t: f(jnp.exp(t))  # noqa: E731
    theta = jnp.log(jnp.asarray(w))
    closed = np.asarray(jax.jacfwd(jax.jacfwd(g))(theta))
    plain = np.asarray(jax.hessian(g)(theta))
    return closed, plain, hessian_theta(np.exp(np.asarray(theta)), derivatives)


@pytest.mark.parametrize("step", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("target", TARGETS)
def test_both_routes_agree_across_the_headroom_threshold(target, step):
    closed, plain, oracle = _theta_hessians(PATTERN * 2.0 ** (-958 + step), target)
    assert failures(closed, oracle) == {} and failures(plain, oracle) == {}
    rel = np.max(np.abs(closed - plain)) / np.max(np.abs(plain))
    assert rel < 1e-13


EXTREMES = {
    "n2": np.array([1.0, 0.5]),
    "n64": np.linspace(0.01, 1.0, 64),
    "ratios": np.array([1.0, 2.0**-40, 2.0**-400, 0.5]),
}


@pytest.mark.parametrize("log2_max", [0, -700, -960, -1015])
@pytest.mark.parametrize("pattern", EXTREMES)
def test_both_routes_agree_at_extreme_sizes_and_ratios(pattern, log2_max):
    x = np.linspace(-1.0, 2.0, EXTREMES[pattern].size)
    w = jnp.asarray(EXTREMES[pattern] * 2.0**log2_max)
    g = lambda t: gaussian_rm_cumulants(jnp.asarray(x), jnp.exp(t))[1]  # noqa: E731
    theta = jnp.log(w)
    closed = np.asarray(jax.jacfwd(jax.jacfwd(g))(theta))
    plain = np.asarray(jax.hessian(g)(theta))
    oracle = hessian_theta(np.exp(np.asarray(theta)), variance(x))
    assert failures(closed, oracle) == {} and failures(plain, oracle) == {}


@pytest.mark.parametrize("mode", MODES)
def test_single_weight_has_zero_second_derivatives(mode):
    """``n = 1``: ``p = 1``; round 4 refused this at ``1e-300``."""
    for scale in (1.0, 1e-300):
        got = MODES[mode](
            lambda w: mixed_moments(np.ones((1, 1)), np.ones(1), w)[1][0]
        )(jnp.asarray([scale]))
        assert np.asarray(got).tolist() == [[0.0]]


@pytest.mark.parametrize("scale", [1e-300, 2.0**-1022])
@pytest.mark.parametrize("target", ["mixed_moments", "rm_variance"])
def test_small_direction_hessian_vector_products_in_the_headroom(target, scale):
    """Direction ``max(w) e_1``: ``H v`` and ``v^T H v`` are representable and
    exact (round 4 refused them)."""
    f, derivatives = TARGETS[target]
    w = jnp.asarray(PATTERN * scale)
    v = jnp.asarray(np.eye(4)[0] * scale)
    H, S = hessian_w(np.asarray(w), derivatives)
    s = Fraction(float(v[0]))
    for got in (
        jax.jvp(jax.grad(f), (w,), (v,))[1],
        jax.grad(lambda u: jax.jvp(f, (u,), (v,))[1])(w),
    ):
        labels = [classify(got[a], H[a][0] * s, S[a][0] * s) for a in range(4)]
        assert labels == ["ok"] * 4
    vhv = jax.jvp(lambda u: jax.jvp(f, (u,), (v,))[1], (w,), (v,))[1]
    assert classify(vhv, H[0][0] * s * s, S[0][0] * s * s) == "ok"


SHIFTS = [0.0, -300.0, -690.0, -700.0, -708.0]
BATCH = jnp.stack([jnp.log(jnp.asarray(PATTERN * 4)) + s for s in SHIFTS])
BATCH_MODES = {
    "vmap-fwd-rev": lambda g: jax.vmap(jax.hessian(g)),
    "vmap-fwd-fwd": lambda g: jax.vmap(jax.jacfwd(jax.jacfwd(g))),
    "vmap-rev-rev": lambda g: jax.vmap(jax.jacrev(jax.jacrev(g))),
    "jit-vmap-rev-fwd": lambda g: jax.jit(jax.vmap(jax.jacrev(jax.jacfwd(g)))),
}
SUM_MODES = {
    "hessian-of-sum": jax.hessian,
    "jacfwd-grad-of-sum": lambda s: jax.jacfwd(jax.grad(s)),
    "jit-jacrev-grad-of-sum": lambda s: jax.jit(jax.jacrev(jax.grad(s))),
}


@pytest.mark.parametrize("mode", list(BATCH_MODES) + list(SUM_MODES))
@pytest.mark.parametrize("target", TARGETS)
def test_batched_log_weight_hessians(target, mode):
    """``vmap`` over five weight vectors, and reverse mode through ``vmap``
    (Hessian of a sum over the batch: block diagonal)."""
    f, derivatives = TARGETS[target]
    g = lambda t: f(jnp.exp(t))  # noqa: E731
    if mode in BATCH_MODES:
        blocks = np.asarray(BATCH_MODES[mode](g)(BATCH))
    else:
        full = np.asarray(SUM_MODES[mode](lambda B: jnp.sum(jax.vmap(g)(B)))(BATCH))
        blocks = np.stack([full[i, :, i, :] for i in range(len(SHIFTS))])
        off = [np.delete(full[i], i, axis=1) for i in range(len(SHIFTS))]
        assert all(np.all(o == 0) for o in off)
    for i, theta in enumerate(np.asarray(BATCH)):
        assert failures(blocks[i], hessian_theta(np.exp(theta), derivatives)) == {}


@pytest.mark.parametrize("target", ["mixed_moments", "rm_variance"])
def test_third_derivatives_match_the_plain_division(target):
    """The closed form is a function of ``(r, dr, y)`` whose own derivatives
    are exact (powers of two ``2^k``, ``2^s`` are piecewise constant)."""
    f = TARGETS[target][0]
    w = jnp.asarray(PATTERN)
    if target == "rm_variance":
        ref = lambda w: (lambda p: p @ (X - p @ X) ** 2)(w / jnp.sum(w))  # noqa: E731
    else:
        ref = lambda w: (w / jnp.sum(w)) @ (OFFSETS[:, 0] * FACTOR)  # noqa: E731
    reference = np.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(ref)))(w))
    for third in (
        jax.jacfwd(jax.jacfwd(jax.jacfwd(f))),
        jax.jacrev(jax.hessian(f)),
        jax.jacfwd(jax.hessian(f)),
        jax.jacrev(jax.jacrev(jax.jacrev(f))),
    ):
        got = np.asarray(third(w))
        assert np.max(np.abs(got - reference)) <= 1e-14 * np.max(np.abs(reference))


def test_float32_weights_second_derivatives():
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0], jnp.float32)
    w32 = jnp.asarray(PATTERN, jnp.float32) * jnp.float32(1e-10)
    h32 = np.asarray(jax.hessian(lambda w: gaussian_rm_cumulants(x, w)[1])(w32))
    w64 = jnp.asarray(np.asarray(w32, np.float64))
    h64 = np.asarray(
        jax.hessian(lambda w: gaussian_rm_cumulants(x.astype(float), w)[1])(w64)
    )
    assert h32.dtype == np.float32 and np.all(np.isfinite(h32))
    np.testing.assert_allclose(h32, h64, rtol=1e-5, atol=1e-5 * np.max(np.abs(h64)))
