"""Validate the strict cumulant expansion (vs raw-moment truncation).

Key point: for a Gaussian, the higher RAW moments (m_3, m_4, ...) do NOT
vanish, but the higher CUMULANTS (kappa_3, kappa_4, ...) DO.  The cumulant
expansion (kappa_k = 0 for k >= 3) is therefore exact to all Taylor orders,
while a raw-moment Taylor-2 truncation silently drops non-vanishing terms.
"""

from __future__ import annotations

import numpy as np

from synchro.cumulants import (
    raw_moments_from_cumulants, cumulant_expansion,
    gaussian_cumulants, raw_moments_gaussian,
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


def test_cumulant_expansion_exact_for_gaussian():
    mu, sigma = 0.3, 0.5
    N = 14
    # reference point p0 = 0, so the deviation is dp = gamma itself, and its
    # cumulants are kappa = [mu, sigma^2].  For S(gamma) = exp(gamma):
    #   S^(n)(0) = 1 for every n.
    S_derivs = [1.0] * (N + 1)
    kappa = gaussian_cumulants(mu, sigma)            # [mu, sigma^2]
    approx = cumulant_expansion(S_derivs, kappa)     # cumulant expansion (K=2)
    exact = np.exp(mu + sigma**2 / 2.0)              # <exp(gamma)>_Gaussian
    raw2 = 1.0 + mu + 0.5 * (sigma**2 + mu**2)       # raw Taylor-2 (drops m_3, m_4)
    return approx, exact, raw2


if __name__ == "__main__":
    print("=== 1. Gaussian: raw moments vs cumulants ===")
    m3, m4 = test_gaussian_moments_vs_cumulants()
    print(f"  raw moments  m_3 = {m3:.4f}, m_4 = {m4:.4f}  (NONZERO)")
    print(f"  cumulants    kappa_3 = 0, kappa_4 = 0        (vanish exactly)")

    print("=== 2. <exp(gamma)> under a Gaussian ===")
    approx, exact, raw2 = test_cumulant_expansion_exact_for_gaussian()
    print(f"  exact <e^gamma>        = {exact:.8f}")
    print(f"  cumulant expansion K=2 = {approx:.8f}  (rel err {abs(approx-exact)/exact:.2e})")
    print(f"  raw Taylor-2           = {raw2:.8f}  (rel err {abs(raw2-exact)/exact:.2e})")
    assert abs(approx - exact) / exact < 1e-8
    assert abs(raw2 - exact) / exact > 1e-2
    print("  -> cumulant expansion is exact for a Gaussian; raw-2 truncation is not.")
    print("ALL CUMULANT CHECKS PASSED")
