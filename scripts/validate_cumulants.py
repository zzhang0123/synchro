"""Check Bell algebra and distinguish statistical order K from kernel degree N.

A Gaussian has zero cumulants above2, but its kernel Taylor series still has
higher terms. The exponential example uses N=14 and approaches the analytic
Gaussian expectation; it is not a proof that a finite-degree Gaussian average
is exact. The same moments at the same degree give identical answers.
"""

from __future__ import annotations

import numpy as np

from synchro.cumulants import (
    raw_moments_from_cumulants,
    cumulant_expansion,
    gaussian_cumulants,
)


def test_gaussian_moments_vs_cumulants():
    mu, sigma = 0.3, 0.5
    kappa = gaussian_cumulants(mu, sigma)  # [mu, sigma^2], kappa_{k>=3} = 0
    m = raw_moments_from_cumulants(kappa, 4)
    # analytic Gaussian raw moments (nonzero to all orders)
    m3 = mu**3 + 3 * mu * sigma**2
    m4 = mu**4 + 6 * mu**2 * sigma**2 + 3 * sigma**4
    assert abs(m[3] - m3) < 1e-12
    assert abs(m[4] - m4) < 1e-12
    return m[3], m[4]


def test_gaussian_exponential_finite_taylor():
    mu, sigma = 0.3, 0.5
    N = 14
    # reference point p0 = 0, so the deviation is dp = gamma itself, and its
    # cumulants are kappa = [mu, sigma^2].  For S(gamma) = exp(gamma):
    #   S^(n)(0) = 1 for every n.
    S_derivs = [1.0] * (N + 1)
    kappa = gaussian_cumulants(mu, sigma)  # [mu, sigma^2]
    approx = cumulant_expansion(S_derivs, kappa)  # cumulant expansion (K=2)
    exact = np.exp(mu + sigma**2 / 2.0)  # <exp(gamma)>_Gaussian
    raw2 = 1.0 + mu + 0.5 * (sigma**2 + mu**2)  # raw Taylor-2 (drops m_3, m_4)
    return approx, exact, raw2


def test_vector_cumulant():
    """Vector (multi-parameter) cumulant expansion: scalar reduction + jit."""
    import math
    import jax
    import jax.numpy as jnp
    from synchro.cumulants import vector_cumulant_expansion

    # P=1 reduction must match the scalar Bell sum to Taylor order 4
    k1 = jnp.array([0.3])
    k2 = jnp.array([[0.25]])
    k3 = jnp.array([[[0.1]]])
    k4 = jnp.array([[[[0.05]]]])
    S0 = jnp.array(1.0)
    g = jnp.array([1.0])
    H = jnp.array([[1.0]])
    T = jnp.array([[[1.0]]])
    F4 = jnp.array([[[[1.0]]]])
    vec = vector_cumulant_expansion([S0, g, H, T, F4], [k1, k2, k3, k4])
    m = raw_moments_from_cumulants([0.3, 0.25, 0.1, 0.05], 4)
    scalar = sum(m[n] / math.factorial(n) for n in range(5))
    assert abs(float(vec) - scalar) < 1e-12

    # 2-parameter case: jittable
    p0 = jnp.array([1.0, 2.0])
    grad = jnp.array([2 * p0[0] * p0[1] ** 3, 3 * p0[0] ** 2 * p0[1] ** 2])
    hess = jnp.array(
        [
            [2 * p0[1] ** 3, 6 * p0[0] * p0[1] ** 2],
            [6 * p0[0] * p0[1] ** 2, 6 * p0[0] ** 2 * p0[1]],
        ]
    )
    third = jnp.zeros((2, 2, 2))
    S0v = p0[0] ** 2 * p0[1] ** 3
    kappa1 = jnp.array([0.1, 0.2])
    kappa2 = jnp.array([[0.5, 0.0], [0.0, 0.3]])

    def fn(a, b):
        return vector_cumulant_expansion([S0v, grad, hess, third], [a, b, third])

    out = jax.jit(fn)(kappa1, kappa2)
    assert bool(jnp.isfinite(out))
    return float(out)


if __name__ == "__main__":
    print("=== 1. Gaussian: raw moments vs cumulants ===")
    m3, m4 = test_gaussian_moments_vs_cumulants()
    print(f"  raw moments  m_3 = {m3:.4f}, m_4 = {m4:.4f}  (NONZERO)")
    print("  cumulants    kappa_3 = 0, kappa_4 = 0        (vanish exactly)")

    print("=== 2. <exp(gamma)> under a Gaussian ===")
    approx, exact, raw2 = test_gaussian_exponential_finite_taylor()
    print(f"  exact <e^gamma>        = {exact:.8f}")
    print(
        f"  Bell sum K=2, N=14    = {approx:.8f}  (rel err {abs(approx-exact)/exact:.2e})"
    )
    print(
        f"  raw Taylor-2           = {raw2:.8f}  (rel err {abs(raw2-exact)/exact:.2e})"
    )
    assert abs(approx - exact) / exact < 1e-8
    assert abs(raw2 - exact) / exact > 1e-2
    print(
        "  Gaussian K=2 closes its statistics; N=14 and N=2 retain different kernel terms."
    )

    print("=== 3. Vector cumulant expansion ===")
    v = test_vector_cumulant()
    print(f"  2-param K=3 output = {v:.6f} (scalar reduction + jit ok)")
    print("ALL CUMULANT CHECKS PASSED")
