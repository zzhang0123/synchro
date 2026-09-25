"""``transfer_slab`` and ``transfer_los`` refuse a non-finite result.

Up to 0.2.0 a non-finite slab was returned as NaN (or inf) without an error:
an overflowing ``eps * ds``, an exhausted ``max_squarings`` budget, extreme gain
and non-finite inputs. Each is now refused with a message naming the cause,
eagerly, under ``jit``, under ``vmap`` and inside the ``lax.scan`` of
``transfer_los``; a derivative of a refused value is refused as well.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import mueller_matrix, transfer_los, transfer_slab

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
ONES = jnp.ones(4)
# An overflowing value is not blamed on gain alone: an absorbing slab overflows
# by accumulation (the round-3 message said "extreme gain" for both).
ACCUMULATED = r"overflowed float64 \(accumulated intensity"
GAIN = r"overflowed float64 .*or extreme gain: negative absorption"
# (S, eps, K, ds, max_squarings, message fragment)
REFUSED = {
    "eps-ds-overflow": (
        jnp.zeros(4),
        jnp.array([4e307, 1.2e307, -8e306, 4e306]),
        THICK,
        50.0,
        32,
        r"eps \* ds overflows",
    ),
    "squaring-budget": (
        jnp.zeros(4),
        ONES,
        100.0 * jnp.eye(4),
        1.0,
        1,
        "max_squarings",
    ),
    "large-tau-budget": (
        jnp.zeros(4),
        ONES,
        1e12 * jnp.eye(4),
        1.0,
        32,
        "max_squarings",
    ),
    "extreme-gain": (ONES, ONES, -800.0 * jnp.eye(4), 1.0, 32, GAIN),
    # Absorbing slab, S e^{-K ds} + G eps = 1.5e308 (1 - 1e-3) + 1e308 (critic p9).
    "accumulation": (
        jnp.array([1.5e308, 0.0, 0.0, 0.0]),
        jnp.array([1e308, 0.0, 0.0, 0.0]),
        1e-3 * jnp.eye(4),
        1.0,
        32,
        ACCUMULATED,
    ),
    "nan-K": (
        ONES,
        ONES,
        jnp.eye(4).at[1, 2].set(jnp.nan),
        1.0,
        32,
        "non-finite input",
    ),
    "inf-S": (ONES.at[0].set(jnp.inf), ONES, jnp.eye(4), 1.0, 32, "non-finite input"),
    "nan-ds": (ONES, ONES, jnp.eye(4), jnp.nan, 32, "non-finite input"),
}
WRAPS = {"eager": lambda f: f, "jit": jax.jit}
# Eager calls raise EquinoxRuntimeError (the cores are compiled with
# equinox.filter_jit); under an outer jax.jit the error is a JaxRuntimeError.
RAISES = {"eager": eqx.EquinoxRuntimeError, "jit": jax.errors.JaxRuntimeError}


def _slab(case):
    S, eps, K, ds, budget, _ = REFUSED[case]
    return (lambda S, e, K, d: transfer_slab(S, e, K, d, max_squarings=budget)), (
        S,
        eps,
        K,
        jnp.asarray(ds),
    )


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("case", REFUSED, ids=list(REFUSED))
def test_non_finite_slab_is_refused_with_its_cause(case, wrap):
    f, args = _slab(case)
    with pytest.raises(RAISES[wrap], match=REFUSED[case][-1]):
        jax.block_until_ready(WRAPS[wrap](f)(*args))


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("case", ("eps-ds-overflow", "extreme-gain", "squaring-budget"))
def test_derivatives_of_a_refused_slab_are_refused(case, wrap):
    """Also under ``jit(jacrev)``, where the value itself is dead code: the
    check sits on the propagators the tangent rule uses."""
    f, args = _slab(case)
    for jac in (jax.jacfwd, jax.jacrev):
        for argnums in (1, 2):
            with pytest.raises(RAISES[wrap], match=REFUSED[case][-1]):
                jax.block_until_ready(WRAPS[wrap](jac(f, argnums=argnums))(*args))


def test_vmap_refuses_when_one_member_is_non_finite():
    eps = jnp.stack([ONES, jnp.array([4e307, 1.2e307, -8e306, 4e306])])

    def f(e):
        return transfer_slab(jnp.zeros(4), e, THICK, 50.0)

    with pytest.raises(RAISES["eager"], match=r"eps \* ds overflows"):
        jax.block_until_ready(jax.vmap(f)(eps))
    with pytest.raises(RAISES["jit"], match=r"eps \* ds overflows"):
        jax.block_until_ready(jax.jit(jax.vmap(f))(eps))
    finite = jax.jit(jax.vmap(f))(jnp.stack([ONES, 2.0 * ONES]))
    np.testing.assert_allclose(finite[1], 2.0 * finite[0], rtol=1e-15)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
def test_line_of_sight_refuses_a_non_finite_slab(wrap):
    """The refusal fires inside the scan: the second of three slabs overflows."""
    eps_s = jnp.stack([ONES, jnp.array([4e307, 1.2e307, -8e306, 4e306]), ONES])
    K_s = jnp.stack([jnp.eye(4), THICK, jnp.eye(4)])

    def f(e):
        return transfer_los(jnp.zeros(4), e, K_s, 50.0)

    with pytest.raises(RAISES[wrap], match=r"eps \* ds overflows"):
        jax.block_until_ready(WRAPS[wrap](f)(eps_s))
    with pytest.raises(RAISES[wrap], match=GAIN):
        gain = K_s.at[2].set(-800.0 * jnp.eye(4))
        jax.block_until_ready(
            WRAPS[wrap](lambda K: transfer_los(ONES, jnp.ones((3, 4)), K, 1.0))(gain)
        )
    with pytest.raises(RAISES[wrap], match=ACCUMULATED):
        bright = jnp.tile(jnp.array([1e308, 0.0, 0.0, 0.0]), (3, 1))
        absorbing = jnp.tile(1e-3 * jnp.eye(4), (3, 1, 1))
        jax.block_until_ready(
            WRAPS[wrap](lambda e: transfer_los(jnp.zeros(4), e, absorbing, 1.0))(bright)
        )
    with pytest.raises(RAISES[wrap], match=r"eps \* ds overflows"):
        jax.block_until_ready(WRAPS[wrap](jax.jacrev(f))(eps_s))


def test_finite_results_near_the_limits_are_not_refused():
    """The checks are on the value only: the largest finite cases still pass."""
    big = transfer_slab(jnp.zeros(4), jnp.array([1.79e308, 0, 0, 0]), jnp.eye(4), 1.0)
    assert np.isfinite(np.asarray(big)).all()
    thick = transfer_slab(jnp.zeros(4), ONES, 1e6 * jnp.eye(4), 1.0)
    assert np.isfinite(np.asarray(thick)).all()
    gain = transfer_slab(ONES, ONES, -700.0 * jnp.eye(4), 1.0)
    assert np.isfinite(np.asarray(gain)).all()
