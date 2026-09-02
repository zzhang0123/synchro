"""Design tests: JAX/Equinox jit + differentiability of the model layers.

These check the *design* half of the deliverable (not the physics, which the
validate_*.py scripts cover):
  1. CumulantExpansion.__call__ compiles and runs under eqx.filter_jit.
  2. __call__ is differentiable w.r.t. the moments (mu, cov) via jax.grad.
  3. transfer_los compiles and runs under jax.jit (path-ordered LOS chain).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from synchro.expansion import build_expansion
from synchro.transfer import mueller_matrix, transfer_los


def test_cumulant_expansion_jit():
    exp = build_expansion([1, 2, 3, 5, 10], 5.0, np.pi / 4, np.pi / 3)
    mu = jnp.zeros(3)
    cov = jnp.diag(jnp.array([0.1, 0.01, 0.0]))
    out = eqx.filter_jit(exp.__call__)(mu, cov)
    assert out.shape == (5, 3)
    assert jnp.all(jnp.isfinite(out))
    return float(jnp.sum(out))


def test_cumulant_expansion_grad():
    exp = build_expansion([1, 2, 3, 5, 10], 5.0, np.pi / 4, np.pi / 3)
    mu = jnp.zeros(3)
    cov = jnp.diag(jnp.array([0.1, 0.01, 0.0]))

    def loss_mu(m):
        return jnp.sum(exp(m, cov))

    g = jax.grad(loss_mu)(mu)
    assert g.shape == (3,)
    assert jnp.all(jnp.isfinite(g))
    return float(jnp.max(jnp.abs(g)))


def test_transfer_los_jit():
    K = mueller_matrix(0.1, 0.0, 0.0, 0.0, 0.0, 0.05, 0.2)
    N = 8
    K_s = jnp.stack([K] * N)
    eps_s = jnp.stack([jnp.array([1.0, 0.2, 0.0, 0.05])] * N)
    S0 = jnp.zeros(4)
    S = jax.jit(transfer_los)(S0, eps_s, K_s, 0.1)
    assert S.shape == (4,)
    assert jnp.all(jnp.isfinite(S))
    return float(jnp.sum(S))


if __name__ == "__main__":
    s1 = test_cumulant_expansion_jit()
    s2 = test_cumulant_expansion_grad()
    s3 = test_transfer_los_jit()
    print(f"CumulantExpansion filter_jit: ok (sum={s1:.4f})")
    print(f"CumulantExpansion grad: ok (max|grad|={s2:.4f})")
    print(f"transfer_los jit: ok (sum={s3:.4f})")
    print("ALL DESIGN TESTS PASSED")
