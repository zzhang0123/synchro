"""Jacobians in ``K`` and ``ds`` near the float64 limit, forward and reverse.

The ``(K, ds)`` tangent of a slab is ``dPhi S + dG eps``. Its transpose forms
the cotangent ``ct eps`` (and ``ct S``) before the transposed Frechet derivative
of the 8x8 exponential; up to round 3 of 0.3.0 that overflowed near
``max|eps ds| = 1.7e308`` (thick K, ds 20: 16 non-finite entries of
``jacrev``, true entries below 3.5e305; ``jacfwd`` was right to 3e-15). The rule
now splits the power of two ``2^j`` of the largest of ``|S|`` and ``|eps ds|``
(``j > 0`` only above ``2^512``) across the forward pass and its transpose.

Oracle: SciPy ``expm_frechet`` of ``[[-K ds, I], [0, 0]]``, evaluated in NumPy.
Tolerance 1e-13 of the largest entry; measured at most 3e-15.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import mueller_matrix, transfer_slab

KS = {
    "weak": mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1),
    "thick": mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0),
}
S0 = np.array([1.0, 0.2, -0.1, 0.05])
D = np.array([1.0, 0.3, -0.2, 0.1])
MODES = {"jacfwd": jax.jacfwd, "jacrev": jax.jacrev}
WRAPS = {"eager": lambda f: f, "jit": jax.jit}


def _block(K, ds):
    A = np.zeros((8, 8))
    A[:4, :4] = -np.asarray(K) * ds
    A[:4, 4:] = np.eye(4)
    return A


def _d_out_dK(S, eps, K, ds):
    """``d out / dK_ab = dPhi S + dG eps`` from SciPy's Frechet derivative."""
    A, J = _block(K, ds), np.empty((4, 4, 4))
    for a, b in itertools.product(range(4), range(4)):
        E = np.zeros((8, 8))
        E[a, b] = -ds
        dE = expm_frechet(A, E, compute_expm=False)
        J[:, a, b] = dE[:4, :4] @ S + (dE[:4, 4:] * ds) @ eps
    return J


def _rel(got, want):
    got = np.asarray(got)
    assert np.all(np.isfinite(got)), f"{int(np.sum(~np.isfinite(got)))} non-finite"
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


CELLS = list(itertools.product(KS, (1.0, 20.0), (1e308, 1.7e308)))


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("K,ds,top", CELLS)
def test_mueller_jacobian_near_the_float64_limit(K, ds, top, mode, wrap):
    """``max|eps ds| = top``: finite and equal to SciPy in both modes."""
    K = KS[K]
    eps = D * top / ds
    jac = WRAPS[wrap](MODES[mode](transfer_slab, argnums=2))
    got = jac(jnp.asarray(S0), jnp.asarray(eps), K, ds)
    assert _rel(got, _d_out_dK(S0, eps, K, ds)) < 1e-13


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("K", KS)
def test_mueller_jacobian_with_the_state_near_the_limit(K, mode):
    """``|S| = 1.5e308`` through an absorbing slab: the ``dPhi S`` term."""
    K, ds = KS[K], 1.0
    S, eps = 1.5e308 * D, D
    got = MODES[mode](transfer_slab, argnums=2)(jnp.asarray(S), jnp.asarray(eps), K, ds)
    assert _rel(got, _d_out_dK(S, eps, K, ds)) < 1e-13


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("K,ds,top", CELLS)
def test_length_derivative_near_the_float64_limit(K, ds, top, mode):
    """``d out / d ds = Phi (eps - K S)``; measured against ``max(|eps|, |K| |out|)``
    because the two terms cancel near saturation (their sum would overflow)."""
    K = KS[K]
    eps = D * top / ds
    E, Kn = expm(_block(K, ds)), np.asarray(K)
    Phi, G = E[:4, :4], E[:4, 4:] * ds
    want = Phi @ (eps - Kn @ S0)
    jac = MODES[mode](transfer_slab, argnums=3)
    got = np.asarray(jac(jnp.asarray(S0), jnp.asarray(eps), K, ds))
    assert np.all(np.isfinite(got))
    out = Phi @ S0 + G @ eps
    size = max(np.max(np.abs(eps)), np.max(np.abs(Kn)) * np.max(np.abs(out)))
    assert np.isfinite(size)
    assert np.max(np.abs(got - want)) < 1e-13 * size


def test_normal_range_rule_is_unscaled():
    """Below ``2^512`` the split exponent is 0: all factors are exactly 1."""
    from syncmoments.transfer import _split_exponent

    for top in (1.0, 1e150, 1e154):
        assert int(_split_exponent(jnp.asarray(S0), jnp.asarray(D * top))) == 0
    assert int(_split_exponent(jnp.asarray(S0), jnp.asarray(D * 1.7e308))) == 512
    assert int(_split_exponent(jnp.asarray(S0 * 1e160), jnp.asarray(D))) == 20
