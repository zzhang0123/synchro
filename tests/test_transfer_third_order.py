"""Third derivatives of ``transfer_slab`` (through ``syncmoments._expm``).

Before 0.4.0 these were checked only by round-5 probes (1e-16 to 7e-14 against
autodiff of ``jax.scipy.linalg.expm``). Third derivatives differentiate the
tangent rule twice more: in ``ds`` through the length identity
``d out / d ds = Phi (eps - K S)`` and ``Phi``'s own derivatives, in ``K``
through the forward derivative of the 8x8 propagator exponential.

References: in ``ds``, the analytic ``d3 out / d ds3 = K^2 Phi (eps - K S)``
with SciPy's ``Phi``; in ``K``, forward-mode autodiff (three levels) of
``Phi S + G eps`` from ``jax.scipy.linalg.expm`` of ``[[-K ds, I], [0, 0]]``.
Tolerance: ``1e-12`` of the size of the terms (``ds``: ``|K|^2 |Phi|
(|eps| + |K| |S|)``, where ``eps - K S`` may cancel; ``K``: the largest entry of
the reference), measured at most 7e-14 in the round-5 probes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import mueller_matrix, transfer_slab

TOL = 1e-12
NESTINGS = {
    "fff": lambda f: jax.jacfwd(jax.jacfwd(jax.jacfwd(f))),
    "rrr": lambda f: jax.jacrev(jax.jacrev(jax.jacrev(f))),
    "frf": lambda f: jax.jacfwd(jax.jacrev(jax.jacfwd(f))),
    "jit rfr": lambda f: jax.jit(jax.jacrev(jax.jacfwd(jax.jacrev(f)))),
}


def _slabs(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        scale = 10.0 ** rng.uniform(-1, 0.7)
        K = np.asarray(mueller_matrix(*(rng.normal(size=7) * scale)))
        K = K + np.eye(4) * abs(rng.normal())
        ds = float(10 ** rng.uniform(-1, 0.5))
        yield rng.normal(size=4), rng.normal(size=4), K, ds


SLABS = list(_slabs(6, seed=9))


@pytest.mark.parametrize("nesting", NESTINGS)
@pytest.mark.parametrize("index", range(len(SLABS)))
def test_third_derivative_in_length(index, nesting):
    S, eps, K, ds = SLABS[index]
    Phi = sla.expm(-K * ds)
    truth = K @ K @ Phi @ (eps - K @ S)
    aK = np.abs(K)
    size = np.max(aK @ aK @ np.abs(Phi) @ (np.abs(eps) + aK @ np.abs(S)))

    def f(d):
        return transfer_slab(jnp.asarray(S), jnp.asarray(eps), jnp.asarray(K), d)

    got = np.asarray(NESTINGS[nesting](f)(jnp.asarray(ds)))
    assert np.max(np.abs(got - truth)) <= TOL * size


def _reference(S, eps, K, ds):
    A = jnp.zeros((8, 8)).at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4))
    E = jax.scipy.linalg.expm(A)
    return E[:4, :4] @ S + (E[:4, 4:] * ds) @ eps


W = jnp.array([1.0, -2.0, 0.5, 1.0])
S1 = jnp.array([1.0, 0.2, 0.1, 0.05])
E1 = jnp.array([2.0, 0.1, -0.3, 0.0])
K_CASES = {
    "mixed": mueller_matrix(1.2, 0.3, -0.2, 0.1, 0.4, -0.3, 2.0),
    "rotation": mueller_matrix(0.05, 0.01, 0.0, 0.0, 0.3, 0.1, 9.0),
    "absorbing": mueller_matrix(3.0, 1.0, 0.5, 0.2, 0.0, 0.0, 0.0),
}


def _k_third(slab, ds, nesting):
    return NESTINGS[nesting](lambda k: W @ slab(S1, E1, k, ds))


@pytest.mark.slow
@pytest.mark.parametrize("nesting", NESTINGS)
@pytest.mark.parametrize("ds", (0.7, 3.0))
@pytest.mark.parametrize("case", K_CASES)
def test_third_derivative_in_the_mueller_matrix(case, ds, nesting):
    K = K_CASES[case]
    want = np.asarray(_k_third(_reference, ds, "fff")(K))
    got = np.asarray(_k_third(transfer_slab, ds, nesting)(K))
    assert np.max(np.abs(got - want)) <= TOL * np.max(np.abs(want))


def test_third_order_through_a_batched_sum():
    """Reverse over two forward levels of a sum over ``vmap`` in ``K`` (batched
    reverse mode at third order) against the forward-mode reference."""
    Ks = jnp.stack(list(K_CASES.values()))
    v = jnp.full_like(Ks, 0.1)

    def total(slab):
        return lambda Ks: jnp.sum(jax.vmap(lambda k: W @ slab(S1, E1, k, 3.0))(Ks))

    def second(F):
        inner = lambda K3: jax.jvp(F, (K3,), (v,))[1]  # noqa: E731
        return lambda K2: jax.jvp(inner, (K2,), (v,))[1]

    got = np.asarray(jax.grad(second(total(transfer_slab)))(Ks))
    want = np.asarray(jax.jacfwd(second(total(_reference)))(Ks))
    assert np.max(np.abs(got - want)) <= TOL * np.max(np.abs(want))
