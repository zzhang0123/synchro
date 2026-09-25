"""Derivatives of ``transfer_slab`` from the normal range to the float64 limit.

The slab map is linear in the state and the source,

    out = Phi S + G eps,   Phi = e^{-K ds},   G = int_0^ds e^{-K t} dt,

so ``d out / d S = Phi`` and ``d out / d eps = G`` do not depend on the scale of
``eps``. Before 0.3.0 forward-mode tangents of ``eps`` were carried through the
power-of-two scaled source column (``deps ds 2^-k``, about 1e-297 at the largest
sources) and lost precision in the Frechet derivative of ``expm``: relative
error 1e-4 at ``max|eps ds| = 1e303`` and 0.1 at 1e306 for the thick ``K`` below.

Oracles are SciPy (an implementation independent of JAX's): ``Phi`` and ``G``
from ``expm([[-K ds, I], [0, 0]])`` (top-left, top-right times ``ds``) and
their derivatives in ``K`` from ``expm_frechet`` of the same block matrix.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import expm, expm_frechet

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
CASES = {
    "thick-ds20": (THICK, 20.0),
    "weak-ds20": (WEAK, 20.0),
    "weak-ds1": (WEAK, 1.0),
}
SCALES = (1e297, 1e299, 1e303, 1e305, 1e306, 1e307)
DIRECTION = np.array([1.0, 0.3, -0.2, 0.1])  # max|eps ds| = scale
S0 = jnp.array([1.0, 0.2, -0.1, 0.05])
WRAPS = {"eager": lambda f: f, "jit": jax.jit}


def _block(K, ds):
    A = np.zeros((8, 8))
    A[:4, :4] = -np.asarray(K) * ds
    A[:4, 4:] = np.eye(4)
    return A


def _phi_g(K, ds):
    E = expm(_block(K, ds))
    return E[:4, :4], E[:4, 4:] * ds


def _dG_dK(K, ds):
    """``dG_ij / dK_ab`` as an array ``[i, j, a, b]`` (SciPy Frechet derivative)."""
    A, out = _block(K, ds), np.empty((4, 4, 4, 4))
    for a in range(4):
        for b in range(4):
            E = np.zeros((8, 8))
            E[a, b] = -ds
            out[:, :, a, b] = expm_frechet(A, E, compute_expm=False)[:4, 4:] * ds
    return out


def _reference_primal(S, eps, K, ds, max_squarings=32):
    """The 0.3.0 primal without the custom derivative rule (plain autodiff)."""
    dtype = jnp.result_type(S, K, eps, ds, 1.0)
    column = (eps * ds).astype(dtype)
    k = _source_exponent(jax.lax.stop_gradient(jnp.max(jnp.abs(column))))
    real = jnp.finfo(dtype).dtype
    M = jnp.zeros((5, 5), dtype=dtype)
    M = M.at[:4, :4].set(-K * ds)
    M = M.at[:4, 4].set(column * _pow2(-k, real))
    y0 = jnp.concatenate([S.astype(dtype), _pow2(k, real).astype(dtype)[None]])
    return (jax.scipy.linalg.expm(M, max_squarings=max_squarings) @ y0)[:4]


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_source_and_state_jacobians_are_scale_free(case, scale, wrap):
    """``jacfwd`` and ``jacrev`` give ``G`` (source) and ``Phi`` (state) at every
    scale. Tolerance 1e-13 of the largest entry: measured 4e-16 to 6e-14
    (JAX Pade expm against SciPy), independent of the scale."""
    K, ds = CASES[case]
    Phi, G = _phi_g(K, ds)
    eps = jnp.asarray(DIRECTION * scale / ds)
    for jac in (jax.jacfwd, jax.jacrev):
        J_eps = WRAPS[wrap](jac(transfer_slab, argnums=1))(S0, eps, K, ds)
        J_S = WRAPS[wrap](jac(transfer_slab, argnums=0))(S0, eps, K, ds)
        assert _rel(J_eps, G) < 1e-13 and _rel(J_S, Phi) < 1e-13


@pytest.mark.parametrize("scale", (1.0,) + SCALES)
@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_length_derivative_is_the_right_hand_side(case, scale):
    """``d out / d ds = Phi (eps - K S) = eps - K out`` (the ODE at the far end).

    Near saturation the two terms of ``eps - K out`` cancel (thick case: 1e-34
    against 0.05), so the error is measured against ``|eps| + |K| |out|``."""
    K, ds = CASES[case]
    Phi, G = _phi_g(K, ds)
    eps = DIRECTION * scale / ds
    expected = Phi @ (eps - np.asarray(K) @ np.asarray(S0))
    out = Phi @ np.asarray(S0) + G @ eps
    size = np.max(np.abs(eps)) + np.max(np.abs(K)) * np.max(np.abs(out))
    for jac in (jax.jacfwd, jax.jacrev):
        got = jac(transfer_slab, argnums=3)(S0, jnp.asarray(eps), K, ds)
        assert np.max(np.abs(np.asarray(got) - expected)) < 1e-13 * size


def _flat(case, scale):
    K, ds = CASES[case]
    eps = DIRECTION * scale / ds
    x = jnp.concatenate([S0, jnp.asarray(eps), K.ravel(), jnp.array([ds])])

    def f(x):
        return transfer_slab(x[:4], x[4:8], x[8:24].reshape(4, 4), x[24])

    return f, x


@pytest.mark.parametrize("scale", (1.0, 1e297, 1e303, 1e306, 1e307))
@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_hessian_is_finite_symmetric_and_order_independent(case, scale):
    """Second derivatives in all 25 inputs: forward-over-reverse (``hessian``),
    reverse-over-forward and forward-over-forward agree, each is symmetric, and
    the source-Mueller block equals SciPy's ``dG / dK`` (it does not depend on
    ``eps``). A tangent rule that differentiates the scaled primal in ``eps``
    passes the one-sided block but fails the symmetry (measured 0.2-2 at 1e306)."""
    f, x = _flat(case, scale)
    H = np.asarray(jax.hessian(f)(x))
    assert np.all(np.isfinite(H))
    for other in (jax.jacrev(jax.jacfwd(f))(x), jax.jacfwd(jax.jacfwd(f))(x)):
        assert_allclose(np.asarray(other), H, rtol=0, atol=1e-13 * np.max(np.abs(H)))
    assert_allclose(np.swapaxes(H, 1, 2), H, rtol=0, atol=1e-13 * np.max(np.abs(H)))
    K, ds = CASES[case]
    dG = _dG_dK(K, ds)  # [i, j, a, b]
    mixed = H[:, 4:8, 8:24].reshape(4, 4, 4, 4)
    assert _rel(mixed, dG) < 1e-12


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_normal_range_matches_plain_autodiff_of_the_primal(case):
    """Below the extreme range the custom rule reproduces plain autodiff of the
    unchanged primal in every input, to 1e-13 of the largest entry (for ``ds``,
    of ``|eps| + |K| |out|``: its two terms cancel near saturation)."""
    K, ds = CASES[case]
    args = (S0, jnp.asarray(DIRECTION * 0.7), K, jnp.asarray(ds))
    out = np.asarray(_reference_primal(*args))
    ds_size = np.max(np.abs(args[1])) + np.max(np.abs(K)) * np.max(np.abs(out))
    for argnums in range(4):
        for jac in (jax.jacfwd, jax.jacrev):
            ref = np.asarray(jac(_reference_primal, argnums=argnums)(*args))
            got = jac(transfer_slab, argnums=argnums)(*args)
            size = ds_size if argnums == 3 else np.max(np.abs(ref))
            assert_allclose(got, ref, rtol=0, atol=1e-13 * size)


PRIMAL_INPUTS = (
    (S0, DIRECTION * 0.7, WEAK, 0.9),
    (S0, DIRECTION * 1e306, THICK, 20.0),
    (jnp.zeros(4), [4.6e307, -3e307, 0.5, 7.0], jnp.zeros((4, 4)), 1.0),
    (S0, DIRECTION, mueller_matrix(0, 0, 0, 0, 0.3, 0.2, 1.8), 1.0),
    (S0, DIRECTION, 1e6 * jnp.eye(4), 1.0),
)


# Inputs 1 and 2 hold sources beyond the float32 range.
PRIMAL_DTYPES = [(i, jnp.float64) for i in range(len(PRIMAL_INPUTS))] + [
    (i, jnp.float32) for i in (0, 3, 4)
]


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("inputs,dtype", PRIMAL_DTYPES)
def test_primal_is_bit_identical_to_the_plain_implementation(inputs, dtype, wrap):
    """The custom rule leaves the value unchanged, also as the primal of a jvp."""
    S, eps, K, ds = PRIMAL_INPUTS[inputs]
    S, eps, K = (jnp.asarray(a, dtype) for a in (S, eps, K))
    expected = WRAPS[wrap](_reference_primal)(S, eps, K, ds)
    got = WRAPS[wrap](transfer_slab)(S, eps, K, ds)
    assert got.dtype == expected.dtype
    assert_array_equal(got, expected)
    primal_out, _ = jax.jvp(lambda e: transfer_slab(S, e, K, ds), (eps,), (eps,))
    assert_array_equal(primal_out, WRAPS["eager"](_reference_primal)(S, eps, K, ds))


def _los_oracle(K_s, ds):
    """``d S_N / d eps_i = Phi_{N-1} ... Phi_{i+1} G_i``."""
    n = len(K_s)
    out = np.empty((4, n, 4))
    for i in range(n):
        J = _phi_g(K_s[i], ds)[1]
        for j in range(i + 1, n):
            J = _phi_g(K_s[j], ds)[0] @ J
        out[:, i, :] = J
    return out


@pytest.mark.parametrize("scale", (1.0, 1e306))
def test_line_of_sight_scan_jit_vmap_and_both_modes(scale):
    """``transfer_los`` (``lax.scan``) under ``jit``, ``jacfwd``, ``jacrev`` and
    ``vmap`` against the product of slab propagators."""
    K_s = jnp.stack([THICK * 0.1, WEAK, mueller_matrix(0.2, 0, 0, 0, 0, 0, 1.3)])
    ds = 2.0
    eps_s = jnp.asarray(np.outer([1.0, -0.5, 0.25], DIRECTION) * scale / ds)
    expected = _los_oracle(np.asarray(K_s), ds)

    def f(e):
        return transfer_los(jnp.zeros(4), e, K_s, ds)

    for jac in (jax.jacfwd, jax.jacrev):
        for wrap in WRAPS.values():
            assert _rel(wrap(jac(f))(eps_s), expected) < 1e-13
    batch = jnp.stack([eps_s, 2.0 * eps_s])
    J = jax.jit(jax.vmap(jax.jacfwd(f)))(batch)
    assert _rel(J[0], expected) < 1e-13 and _rel(J[1], expected) < 1e-13
    values = jax.vmap(f)(batch)
    assert_array_equal(values[0], f(eps_s))
