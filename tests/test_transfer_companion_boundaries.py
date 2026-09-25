"""Boundary validation of the slab value's derivative dispatcher.

``_slab_value`` returns the plain 5x5 value. Its derivatives (taken where JAX
inlines the primal: the primal output of a ``jvp``/``vjp`` under a further
derivative, ``lax.scan`` under ``vjp``) come from one of two methods, chosen
from primal values only:

* plain: autodiff of the 5x5 exponential, source tangent scaled by ``ds 2^-k``;
* companion: the derivatives of ``Phi S + G eps`` (the slab's tangent rule),
  at the price of one 8x8 exponential in the primal (unconditionally: 1.9-2.1x
  for jit ``transfer_los``, 1.4-1.6x for jit(vmap); hence the dispatch).

The companion is used when ``_exposure = k + max(0, -e(ds)) + max(0, e(|K ds|))``
reaches ``T = maxexp - 128`` (896 in float64). ``transfer_slab`` decides per
slab with a ``lax.cond``, ``transfer_los`` once per path (any slab). Measured
(DEV, jax 0.10.0) on ``|K|`` in 1e-6..1e6, ``ds`` in 1e-9..1e6, ``|K ds| <= 2e4``:
the plain path's first and second derivatives first exceed 1e-13 of their
largest entry at exposure 950-1014 (``k`` alone: 920-1018, hence the ``ds`` and
``|K ds|`` terms), so ``T`` leaves a margin of at least 54 binary orders; the
error grows about twofold per unit of exposure. Below ``T`` both methods are
evaluated directly and must agree; above it only the companion is required to.

Reference: plain autodiff of ``Phi S + G eps`` from the 8x8 exponential
(forward mode, which stays finite at 1.7e308).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import (
    _companion,
    _companion_exponent,
    _exposure,
    _plain_value,
    _slab_value,
    _uses_companion,
    mueller_matrix,
    transfer_los,
)

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
WEAK = mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1)
KS = {"thick": THICK, "weak": WEAK}
S0 = jnp.array([1.0, 0.2, -0.1, 0.05])
D = jnp.array([1.0, 0.3, -0.2, 0.1])
T = 896
TOL = 1e-13


def plain(S, e, K, ds):
    return _plain_value(S, e, K, ds, 32)


def companion(S, e, K, ds):
    return _companion(plain(S, e, K, ds), S, e, K, ds, 32)


def dispatched(S, e, K, ds):
    return _slab_value(S, e, K, ds, 32)


def reference(S, e, K, ds):
    A = jnp.zeros((8, 8)).at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4))
    E = jax.scipy.linalg.expm(A)
    return E[:4, :4] @ S + (E[:4, 4:] * ds) @ e


def _derivatives(f, ref=False):
    """First derivatives in ``eps`` and ``K`` (both modes) and ``d2/dK deps``."""
    rev = jax.jacfwd if ref else jax.jacrev

    def all_of(S, e, K, ds):
        return {
            "eps fwd": jax.jacfwd(f, 1)(S, e, K, ds),
            "eps rev": rev(f, 1)(S, e, K, ds),
            "K fwd": jax.jacfwd(f, 2)(S, e, K, ds),
            "K rev": rev(f, 2)(S, e, K, ds),
            "S fwd": jax.jacfwd(f, 0)(S, e, K, ds),
            "K eps": jax.jacfwd(jax.jacfwd(f, 2), 1)(S, e, K, ds),
        }

    return jax.jit(all_of)


METHODS = {
    "plain": _derivatives(plain),
    "companion": _derivatives(companion),
    "dispatched": _derivatives(dispatched),
    "reference": _derivatives(reference, ref=True),
}


def _errors(got, want):
    out = {}
    for name in want:
        a, b = np.asarray(got[name]), np.asarray(want[name])
        with np.errstate(all="ignore"):
            out[name] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    return out


def _source_at_exposure(K, ds, exposure):
    """``eps = D 0.75 2^k / ds`` with ``k`` chosen so that ``_exposure == exposure``."""
    offset = int(_exposure(0, K, jnp.asarray(ds)))
    return D * (0.75 * 2.0 ** (exposure - offset)) / ds


def _exposure_of(e, K, ds):
    k = int(np.frexp(float(jnp.max(jnp.abs(e * ds))))[1])
    return int(_exposure(max(k, 0), K, jnp.asarray(ds)))


def test_threshold_is_pinned():
    assert int(_companion_exponent(jnp.float64)) == T
    assert int(_companion_exponent(jnp.float32)) == 0


NEAR = list(itertools.product(KS, (1.0, 20.0), (-8, -1, 0, 1, 8)))


@pytest.mark.parametrize("K,ds,delta", NEAR)
def test_methods_agree_at_the_threshold(K, ds, delta):
    """At ``T + delta`` both methods, evaluated directly, agree with each other
    and with the reference (measured at most 1e-15), and the value is the same."""
    K = KS[K]
    e = _source_at_exposure(K, ds, T + delta)
    assert _exposure_of(e, K, ds) == T + delta
    args = (S0, e, K, jnp.asarray(ds))
    want = METHODS["reference"](*args)
    p, c = METHODS["plain"](*args), METHODS["companion"](*args)
    assert max(_errors(p, want).values()) < TOL
    assert max(_errors(c, want).values()) < TOL
    assert max(_errors(p, c).values()) < TOL
    assert_array_equal(companion(*args), plain(*args))


EXTREMES = (1.0, 1e-300, 1e250, 1e297, 1e306, 1.7e308)
FAR = list(itertools.product(KS, (1.0, 20.0), EXTREMES))


@pytest.mark.parametrize("K,ds,top", FAR)
def test_extreme_sources_route_to_an_accurate_method(K, ds, top):
    """``max|eps ds|`` from 1e-300 to 1.7e308: the companion is exact everywhere
    (forward and reverse, also at 1.7e308), the dispatcher uses it exactly from
    ``T`` on, and below ``T`` the plain path is accurate too."""
    K = KS[K]
    e = D * top / ds
    args = (S0, e, K, jnp.asarray(ds))
    want = METHODS["reference"](*args)
    c, d = METHODS["companion"](*args), METHODS["dispatched"](*args)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in c.values())
    assert max(_errors(c, want).values()) < TOL
    chosen = c if _exposure_of(e, K, ds) >= T else METHODS["plain"](*args)
    for name in want:
        assert_array_equal(np.asarray(d[name]), np.asarray(chosen[name]))
    assert max(_errors(d, want).values()) < TOL
    assert_array_equal(dispatched(*args), plain(*args))
    assert bool(_uses_companion(e, K, jnp.asarray(ds))) == (_exposure_of(e, K, ds) >= T)


@pytest.mark.parametrize("delta", (-1, 0))
def test_line_of_sight_routes_once_for_the_whole_path(delta):
    """A path with one slab at exposure ``T + delta`` and one ordinary slab: the
    route is the companion for both slabs iff ``delta >= 0``, and ``jacfwd`` of
    the ``vjp`` primal (the inlined value) matches the reference either way."""
    e_hi = _source_at_exposure(WEAK, 1.0, T + delta)
    eps_s, K_s = jnp.stack([D, e_hi]), jnp.stack([WEAK, 0.7 * WEAK])

    def outer(slab_path):
        return jax.jacfwd(lambda e: jax.vjp(lambda K: slab_path(e, K), K_s)[0])

    def ref_path(e, K):
        body = lambda S, x: (reference(S, x[0], x[1], 1.0), None)  # noqa: E731
        return jax.lax.scan(body, S0, (e, K))[0]

    got = outer(lambda e, K: transfer_los(S0, e, K, 1.0))(eps_s)
    want = np.asarray(outer(ref_path)(eps_s))
    assert np.max(np.abs(np.asarray(got) - want)) < TOL * np.max(np.abs(want))
    uses = jax.vmap(_uses_companion)(eps_s, K_s, jnp.ones(2))
    assert [bool(u) for u in uses] == [False, delta >= 0]


@pytest.mark.parametrize("K,ds", [("thick", 20.0), ("weak", 1.0)])
def test_plain_method_fails_above_the_threshold(K, ds):
    """Why the dispatcher exists: at 1e306 the plain path's ``eps`` derivatives
    are off by 3e-2 (weak) to 0.1 (thick), the round-3 critic's finding."""
    K = KS[K]
    args = (S0, D * 1e306 / ds, K, jnp.asarray(ds))
    err = _errors(METHODS["plain"](*args), METHODS["reference"](*args))
    assert max(err.values()) > 1e-3
