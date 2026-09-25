"""Derivatives of ``mixed_moments`` at the smallest normal weights (T-004 round 3, H2).

``mixed_moments`` normalises its weights with ``syncmoments.rm._normalised_weights``,
the helper used by every other weighted average. Round 2 normalised with its
own ``scaled / sum(scaled)`` after ``_relative_weights`` (no tangent headroom):
at ``w = full(4, 2^-1022)`` a ones tangent made the intermediate
``sum(dw) / max(w) = 2^1024`` overflow, and ``jax.jvp`` returned
``(-inf, [-inf, nan], [[-inf, nan], [nan, -inf]])`` next to a finite primal.

Oracles (NumPy, exact algebra). With ``p = w / W``, ``W = sum(w)`` and any
per-sample quantity ``g`` (``f``, ``f q_j`` or ``f q_j q_k``),
``E[g] = sum_i p_i g_i`` has::

    d E[g] / d w_k        = (g_k - E[g]) / W
    d2 E[g] / d w_k d w_l = -(g_k + g_l - 2 E[g]) / W^2

A uniform tangent leaves ``p`` unchanged, so its jvp is exactly zero. At
``W = 2^-1020`` the first derivatives reach ``|g| 2^1020`` (finite, below
``1.8e308``); the Hessian there is of order ``2^2040`` and not representable,
so the Hessian is checked at ``max(w)`` in ``{1e-150, 1, 1e150}``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.expansion import mixed_moments

OFFSETS = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, -1.0], [0.4, 0.5]])
FACTORS = {"ones": np.ones(4), "varied": np.array([1.0, 1.5, 0.5, 1.25])}
MIN_NORMAL = 2.0**-1022
WRAPS = {"eager": lambda f: f, "jit": jax.jit}


def _g(factor):
    """Per-sample ``(g0, g1, g2)`` of the three raw mixed moments."""
    return (
        factor,
        factor[:, None] * OFFSETS,
        factor[:, None, None] * OFFSETS[:, :, None] * OFFSETS[:, None, :],
    )


def _oracle_grad(factor, w):
    """``d E[g] / d w_k`` as arrays ``(4,)``, ``(2, 4)``, ``(2, 2, 4)``, computed
    in units of ``max(w)`` and rescaled by an exact power of two."""
    top = np.frexp(w.max())[1] - 1
    u = np.ldexp(w, -top)
    p = u / u.sum()
    out = []
    for g in _g(factor):
        mean = np.tensordot(p, g, axes=(0, 0))
        d = (np.moveaxis(g, 0, -1) - mean[..., None]) / u.sum()
        out.append(np.ldexp(d, -top))
    return out


def _moments(factor):
    return lambda w: mixed_moments(OFFSETS, factor, w)


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("factor", FACTORS)
def test_uniform_tangent_at_min_normal_gives_exact_zero(factor, wrap):
    """The repro of the round-2 critic: every tangent is 0 and the primal is
    unchanged; E[W] is exactly zero."""
    w0 = jnp.full(4, MIN_NORMAL)
    f = FACTORS[factor]
    primal, tangent = WRAPS[wrap](lambda w: jax.jvp(_moments(f), (w,), (jnp.ones(4),)))(
        w0
    )
    assert float(tangent[0]) == 0.0
    for t in tangent:
        t = np.asarray(t)
        assert np.all(np.isfinite(t))
        assert_allclose(t, 0.0, atol=1e-300)
    for got, expected in zip(primal, mixed_moments(OFFSETS, f)):
        assert_allclose(np.asarray(got), np.asarray(expected), rtol=4e-16)


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize(
    "w",
    [np.full(4, MIN_NORMAL), np.array([4.0, 2.0, 1.0, 1.0]) * 2.0**-1024],
    ids=["equal", "pattern"],
)
def test_nonuniform_tangent_at_min_normal_matches_oracle(w, factor, wrap):
    f = FACTORS[factor]
    t = np.array([1.0, 2.0, 3.0, 4.0]) / 8
    _, tangent = WRAPS[wrap](lambda v: jax.jvp(_moments(f), (v,), (jnp.asarray(t),)))(
        jnp.asarray(w)
    )
    for got, d in zip(tangent, _oracle_grad(f, w)):
        expected = d @ t
        scale = np.abs(d).max() * np.abs(t).sum()
        assert np.all(np.isfinite(np.asarray(got)))
        assert_allclose(np.asarray(got), expected, rtol=0, atol=8e-16 * scale)


@pytest.mark.parametrize("transform", ["grad", "jacfwd", "jit-jacfwd", "jacrev"])
@pytest.mark.parametrize(
    "w",
    [np.full(4, MIN_NORMAL), np.array([4.0, 2.0, 1.0, 1.0]) * 2.0**-1024],
    ids=["equal", "pattern"],
)
def test_first_derivatives_at_min_normal_match_oracle(w, transform):
    f = FACTORS["varied"]
    d0, d1, d2 = _oracle_grad(f, w)
    wj = jnp.asarray(w)
    if transform == "grad":
        got = [jax.grad(lambda v: _moments(f)(v)[0])(wj)]
        want = [d0]
    else:
        make = {"jacfwd": jax.jacfwd, "jacrev": jax.jacrev}.get(transform, jax.jacfwd)
        jac = make(_moments(f))
        jac = jax.jit(jac) if transform.startswith("jit") else jac
        got, want = jac(wj), [d0, d1, d2]
    for g, e in zip(got, want):
        g = np.asarray(g)
        assert np.all(np.isfinite(g))
        assert_allclose(g, e, rtol=0, atol=8e-16 * np.abs(e).max())


@pytest.mark.parametrize("scale", [1e-150, 1.0, 1e150])
def test_hessian_matches_oracle(scale):
    f = FACTORS["varied"]
    w = np.array([4.0, 2.0, 1.0, 3.0]) / 4 * scale
    p = w / w.sum()
    g = f[:, None] * OFFSETS
    mean = p @ g
    expected = -(g[:, None, :] + g[None, :, :] - 2 * mean) / w.sum() ** 2
    expected = np.moveaxis(expected, -1, 0)  # (P, 4, 4), as jax.hessian
    for wrap in WRAPS.values():
        got = np.asarray(wrap(jax.hessian(lambda v: _moments(f)(v)[1]))(jnp.asarray(w)))
        assert np.all(np.isfinite(got))
        assert_allclose(got, expected, rtol=0, atol=1e-14 * np.abs(expected).max())
