"""Boundary validation of the value's exponential at its squaring thresholds.

The value of ``transfer_slab`` is ``exp(M) [S; 2^k]`` for the 5x5 matrix
``M = [[-K ds, eps ds 2^-k], [0, 0]]``, by ``syncmoments._expm.expm_pade13``:
Pade 13 of ``M 2^-n`` squared ``n`` times, ``n = ceil(log2(|M|_1 / theta_13))``.
At each threshold ``|M|_1 = theta_13 2^m`` the neighbouring methods (``n`` and
``n + 1`` squarings) are evaluated directly, bypassing the dispatcher, and
compared with each other and with SciPy, for an absorbing (thick, weak) and a
rotation-dominated ``K``, ``m`` from 0 to 31, and source columns
``max|eps ds|`` of 1e-300, 1 and 1e300 (``k`` = 0, 1, 997).

Tolerance ``1e-14 2^m`` of the largest entry, as for the propagators
(``tests/test_transfer_expm_boundaries.py``): the rotation's exponential has
condition number about ``|M|_1``, so roundoff is ``u |M|_1 = 6e-16 2^m``.

``max_squarings`` keeps the 0.3.0 refusal set: the value is NaN (and refused
naming ``max_squarings``) exactly where ``jax.scipy.linalg.expm`` returns NaN,
i.e. ``floor(log2(|M|_1 / theta_13)) > max_squarings``. Where ``ceil`` exceeds
the budget and ``floor`` does not, ``floor`` is used, with its accuracy.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import syncmoments  # noqa: F401  (enables x64)
from syncmoments._expm import _THETA13, expm_pade13, expm_squared
from syncmoments.rm import _pow2
from syncmoments.transfer import (
    _plain_value,
    _source_exponent,
    mueller_matrix,
    transfer_slab,
)

KS = {
    "thick": mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0),
    "weak": mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1),
    "rotation": mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.3, -0.2, 1.0),
}
EXPONENTS = (0, 1, 3, 10, 20, 31)
POSITIONS = {"below": 1 - 1e-9, "at": 1.0, "above": 1 + 1e-9}
SOURCES = (1e-300, 1.0, 1e300)
S = jnp.array([1.0, 0.6, -0.3, 0.1])
D = jnp.array([0.5, 0.2, -0.1, 0.05])
BUDGET = 40


def _slab_at(K, norm, top):
    """``(eps, ds)`` with ``|K ds|_1 = norm`` and ``max|eps ds| = top``."""
    ds = norm / float(np.max(np.sum(np.abs(np.asarray(K)), axis=0)))
    return D * (top / 0.5 / ds), ds


def _matrix(eps, K, ds):
    column = eps * ds
    k = _source_exponent(jnp.max(jnp.abs(column)))
    M = jnp.zeros((5, 5)).at[:4, :4].set(-K * ds)
    return M.at[:4, 4].set(column * _pow2(-k, column.dtype)), _pow2(k, column.dtype)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    size = max(np.max(np.abs(a)), np.max(np.abs(b)), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(a - b)) / size)  # 0 when both vanish (thick, m = 31)


def _squared(M, n):
    return np.asarray(expm_squared(M, jnp.asarray(float(n)), BUDGET))


@pytest.mark.parametrize("top", SOURCES)
@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("m", EXPONENTS)
@pytest.mark.parametrize("name", KS)
def test_neighbouring_squaring_counts_agree_for_the_value(name, m, position, top):
    K = KS[name]
    eps, ds = _slab_at(K, _THETA13 * 2.0**m * POSITIONS[position], top)
    M, scale = _matrix(eps, K, ds)
    norm = float(np.max(np.sum(np.abs(np.asarray(M)), axis=0)))
    n = max(0, int(np.ceil(np.log2(norm / _THETA13))))
    tol = 1e-14 * 2.0**m
    chosen, more, reference = _squared(M, n), _squared(M, n + 1), sla.expm(M)
    assert _rel(chosen, more) < tol
    assert _rel(chosen, reference) < tol and _rel(more, reference) < tol
    # The dispatcher takes n, and the value is that exponential applied to [S; 2^k].
    np.testing.assert_array_equal(expm_pade13(M, BUDGET), chosen)
    y0 = jnp.concatenate([S, scale[None]])
    expected = jnp.sum(jnp.asarray(chosen)[:4] * y0, axis=1)  # as _plain_value
    np.testing.assert_array_equal(_plain_value(S, eps, K, ds, BUDGET), expected)
    # Compiled, XLA rounds the Pade evaluation and squarings differently from the
    # eager ops (measured up to 5.7e-8 at m = 31), within the same tolerance.
    got = transfer_slab(S, eps, K, ds, max_squarings=BUDGET)
    assert _rel(got, expected) < tol


@pytest.mark.parametrize("m", (0, 3, 10, 20))
def test_floor_count_of_the_0_3_0_value_misses_just_below_the_next_threshold(m):
    """The 0.3.0 value (``floor``: ``n = m`` squarings at ``|M|_1`` just below
    ``2^(m+1) theta_13``) against SciPy for the rotation ``K``; ``ceil`` takes
    ``m + 1``. Measured floor errors are recorded as a lower bound."""
    eps, ds = _slab_at(KS["rotation"], _THETA13 * 2.0 ** (m + 1) * (1 - 1e-9), 1.0)
    M, _ = _matrix(eps, KS["rotation"], ds)
    floor_err, ceil_err = (_rel(_squared(M, n), sla.expm(M)) for n in (m, m + 1))
    assert floor_err > 1e-11 and floor_err > 100 * ceil_err
    assert ceil_err < 1e-14 * 2.0**m


def _refused(S, eps, K, ds, budget, dtype):
    args = (jnp.asarray(a, dtype) for a in (S, eps, K))
    try:
        out = transfer_slab(*args, dtype(ds), max_squarings=budget)
    except eqx.EquinoxRuntimeError as err:
        assert "max_squarings" in str(err)
        return True, None
    return False, np.asarray(out, np.float64)


@pytest.mark.parametrize("dtype", (np.float64, np.float32))
@pytest.mark.parametrize("budget", (0, 3, 5))
def test_budget_refusal_is_the_floor_rule(budget, dtype):
    """Refused iff ``floor(log2(|M|_1 / theta_13)) > max_squarings`` (the NaN of
    ``jax.scipy.linalg.expm`` in float64); otherwise finite and accurate to the
    method used (``floor`` where ``ceil`` exceeds the budget). In float32 0.3.0
    used Pade 7 with ``theta_7 = 3.93``, so its refusal set began at a norm
    ``theta_13 / theta_7 = 1.37`` times lower; it now matches float64."""
    K = KS["thick"]
    # Not at 2.0 exactly: there floor(log2) turns on the last bit of the norm.
    for scale in (1.0, 1 + 1e-6, 2 * (1 - 1e-6), 2 * (1 + 1e-6), 4.0):
        eps, ds = _slab_at(K, _THETA13 * 2.0**budget * scale, 1.0)
        M, _ = _matrix(eps, K, ds)
        norm = float(np.max(np.sum(np.abs(np.asarray(M)), axis=0)))
        floor = max(0, int(np.floor(np.log2(norm / _THETA13))))
        refused, out = _refused(S, eps, K, ds, budget, dtype)
        assert refused == (floor > budget), scale
        if dtype == np.float64:
            jax_nan = np.isnan(
                np.asarray(jax.scipy.linalg.expm(M, max_squarings=budget))
            )
            assert refused == jax_nan.any(), scale
        if not refused:
            y0 = np.concatenate([np.asarray(S), [1.0]])
            M1 = np.asarray(M).copy()
            M1[:4, 4] = np.asarray(eps) * ds
            want = (sla.expm(M1) @ y0)[:4]
            tol = 1e-11 if dtype == np.float64 else 1e-4
            assert _rel(out, want) < tol, scale
