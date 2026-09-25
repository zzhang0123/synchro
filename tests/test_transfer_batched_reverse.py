"""Reverse mode through ``vmap`` over a batched ``K`` or ``ds``.

In round 4 of 0.3.0 the slab's tangent rule called ``jax.jvp`` of the 8x8
exponential inside the ``custom_jvp`` rule. Under ``vmap`` with a batched ``K``
the ``lax.switch`` in ``expm`` becomes a select that wraps the untaken operands,
here the linear tangent, in ``stop_gradient``, which has no transpose rule:
``grad``, ``vjp``, ``jacrev``, ``hessian`` and HVPs of a sum over ``vmap`` in
``K`` or ``ds`` raised ``NotImplementedError`` (v0.2.0 differentiated them).

Checks, for ``transfer_slab``, ``transfer_los`` (per-ray ``K``, per-slab ``ds``)
and ``moment_driven_slab`` over frequencies, eagerly and under ``jit``:

* every transform against forward-mode autodiff of the 8x8 reference
  ``exp([[-K ds, I], [0, 0]]) = [[Phi, G / ds], [0, I]]`` (independent of the
  package's rule), at ``max|eps ds|`` of order 1 and 1e306 (companion route of
  the value, split exponent ``a > 0`` of the tangent rule), to 1e-13 of the
  largest entry of each block;
* ``grad`` of the sum over ``vmap`` against ``vmap(grad)``;
* v0.2.0 values at order-1 scale, computed once with the v0.2.0 export (the
  package then named ``synchro``) in jax 0.10.0 and pinned below.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.los_moments import moment_driven_slab
from syncmoments.transfer import mueller_matrix, transfer_los, transfer_slab

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
WEAK = mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1)
S0 = jnp.array([1.0, 0.2, -0.1, 0.05])
D = jnp.array([1.0, 0.3, -0.2, 0.1])
W = jnp.array([1.0, -0.5, 0.25, 0.1])
KS = jnp.stack([THICK, WEAK])
DS = jnp.array([1.0, 2.0])
KL = jnp.stack([jnp.stack([THICK, WEAK, THICK]), jnp.stack([WEAK, WEAK, THICK])])
DL = jnp.array([[1.0, 0.5, 2.0], [0.3, 1.0, 1.5]])
EL = jnp.stack([D, 2 * D, -D])
NUS = jnp.geomspace(0.1, 10.0, 8)
WRAPS = {"eager": lambda f: f, "jit": jax.jit}
TOL = 1e-13


def ref_slab(S, eps, K, ds):
    A = jnp.zeros((8, 8)).at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4))
    E = jax.scipy.linalg.expm(A)
    return E[:4, :4] @ S + (E[:4, 4:] * ds) @ eps


def ref_los(S, eps_s, K_s, ds):
    ds = jnp.broadcast_to(ds, eps_s.shape[:1])
    body = lambda S, x: (ref_slab(S, *x), None)  # noqa: E731
    return jax.lax.scan(body, S, (eps_s, K_s, ds))[0]


def _members(slab, los, scale):
    """name -> (member(*batched), batched args): the loss is the sum over vmap."""
    e = D * scale
    return {
        "slab K": (lambda K: W @ slab(S0, e, K, 1.0), (KS,)),
        "slab ds": (lambda d: W @ slab(S0, e, THICK, d), (DS,)),
        "slab K ds": (lambda K, d: W @ slab(S0, e, K, d), (KS, DS)),
        "los K": (lambda K: W @ los(S0, EL * scale, K, 0.7), (KL,)),
        "los ds": (lambda d: W @ los(S0, EL * scale, KL[0], d), (DL,)),
        "los K ds": (lambda K, d: W @ los(S0, EL * scale, K, d), (KL, DL)),
    }


def _summed(member):
    return lambda *a: jnp.sum(jax.vmap(member)(*a))


def _pattern(x):
    return jnp.cos(jnp.arange(x.size, dtype=x.dtype)).reshape(x.shape)


def _transforms(member, args, forward):
    """Five reverse-mode transforms of the batched member (``forward=True``:
    the same quantity from forward-mode autodiff, the reference)."""
    argnums = tuple(range(len(args)))
    loss, vec = _summed(member), jax.vmap(member)
    first = jax.jacfwd if forward else jax.grad
    jac = jax.jacfwd if forward else jax.jacrev
    cot = jnp.arange(1.0, args[0].shape[0] + 1)
    tangents = tuple(_pattern(a) for a in args)

    def vjp(*a):
        if forward:
            J = jax.jacfwd(vec, argnums)(*a)
            return tuple(jnp.tensordot(cot, j, axes=1) for j in J)
        return jax.vjp(vec, *a)[1](cot)

    def hvp(*a):
        return jax.jvp(lambda *b: first(loss, argnums)(*b), a, tangents)[1]

    hess = (
        (lambda f: jax.jacfwd(jax.jacfwd(f, argnums), argnums))
        if forward
        else (lambda f: jax.hessian(f, argnums))
    )
    return {
        "grad": lambda *a: first(loss, argnums)(*a),
        "vjp": vjp,
        "jacrev": lambda *a: jac(vec, argnums)(*a),
        "hessian": lambda *a: hess(loss)(*a),
        "hvp": hvp,
    }


def _close(got, want, tol=TOL):
    got, want = jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        g, w = np.asarray(g), np.asarray(w)
        assert g.shape == w.shape and np.all(np.isfinite(g))
        scale = np.max(np.abs(w))
        assert np.max(np.abs(g - w)) <= tol * scale, np.max(np.abs(g - w)) / scale


CASES = list(_members(None, None, 1.0))
KINDS = ["grad", "vjp", "jacrev", "hessian", "hvp"]


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
@pytest.mark.parametrize("scale", (1.0, 1e306))
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", CASES)
def test_reverse_mode_of_a_batched_k_or_ds_matches_the_reference(
    case, kind, scale, wrap
):
    member, args = _members(transfer_slab, transfer_los, scale)[case]
    ref_member, _ = _members(ref_slab, ref_los, scale)[case]
    got = WRAPS[wrap](_transforms(member, args, forward=False)[kind])(*args)
    _close(got, _transforms(ref_member, args, forward=True)[kind](*args))


@pytest.mark.parametrize("case", CASES)
def test_grad_of_the_sum_over_vmap_equals_vmap_of_grad(case):
    member, args = _members(transfer_slab, transfer_los, 1.0)[case]
    argnums = tuple(range(len(args)))
    got = jax.grad(_summed(member), argnums)(*args)
    _close(got, jax.vmap(jax.grad(member, argnums))(*args))


# v0.2.0 (jax 0.10.0; jax 0.10.2 agrees to 3e-15): (sum(g), sum(g cos(arange)))
# per block of grad(sum over vmap) at order-1 scale.
V020 = {
    "slab K": [(-1.2040036201258293, 1.2836282134215042)],
    "slab ds": [(-0.04141990676190667, -0.041054507889702975)],
    "slab K ds": [
        (-2.5368220622750037, 2.787536423177907),
        (0.30202844739028156, 0.1445114298053418),
    ],
    "los K": [(-0.04116797679256949, 0.04425615121936753)],
    "los ds": [(-0.014707336004173347, -0.0035985343826769474)],
    "los K ds": [
        (0.07024549387469585, 0.06304468179343416),
        (-0.016847684133670986, -0.006969502735630139),
    ],
}


@pytest.mark.parametrize("case", CASES)
def test_grad_of_the_sum_over_vmap_matches_v020(case):
    member, args = _members(transfer_slab, transfer_los, 1.0)[case]
    grads = jax.grad(_summed(member), tuple(range(len(args))))(*args)
    for g, (total, weighted) in zip(grads, V020[case]):
        np.testing.assert_allclose(float(jnp.sum(g)), total, rtol=1e-12)
        np.testing.assert_allclose(
            float(jnp.sum(g * _pattern(g))), weighted, rtol=1e-12
        )


def _spectrum(M0, L):
    """Stokes I summed over 8 frequencies: ``K`` and ``eps`` batched by ``nu``."""

    def at(nu):
        return moment_driven_slab(nu, 3.0, 1.0, M0, 1.0, 1.0, 0.5, L)[0]

    return jnp.sum(jax.vmap(at)(NUS))


# v0.2.0 (jax 0.10.0, DEV): grad and Hessian of _spectrum in (M0, L) at (1, 1).
V020_SPECTRUM_GRAD = (3.730769672720127, 3.1593318645996473)
V020_SPECTRUM_HESSIAN = (
    (0.4772341623943672, 3.0908624541144842),
    (3.0908624541144842, -0.4913579969358053),
)


@pytest.mark.parametrize("wrap", WRAPS, ids=list(WRAPS))
def test_spectrum_of_moment_driven_slabs_reverse_mode(wrap):
    """critic q15: ``grad`` of a sum over ``vmap(moment_driven_slab)`` raised
    in round 4; v0.2.0 gives 3.7308. Reverse, forward and v0.2.0 agree."""
    x = (jnp.asarray(1.0), jnp.asarray(1.0))
    g = WRAPS[wrap](jax.grad(_spectrum, (0, 1)))(*x)
    _close(g, jax.jacfwd(_spectrum, (0, 1))(*x))
    np.testing.assert_allclose(np.asarray(g), V020_SPECTRUM_GRAD, rtol=1e-12, atol=0)
    H = WRAPS[wrap](jax.hessian(_spectrum, (0, 1)))(*x)
    _close(H, jax.jacfwd(jax.jacfwd(_spectrum, (0, 1)), (0, 1))(*x))
    np.testing.assert_allclose(np.asarray(H), V020_SPECTRUM_HESSIAN, rtol=1e-12)
