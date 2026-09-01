"""Strict cumulant expansion of the ensemble average.

The exact ensemble average of S(p) = S(p0 + dp) over a distribution with
cumulants kappa_1, kappa_2, ... is

    <S> = [ exp( sum_{k>=1} (1/k!) kappa_k d^k/dp^k ) S(p) ]_{p=p0} .

Truncating the *cumulant hierarchy* at order K (kappa_k = 0 for k > K) is
principled, unlike truncating the raw-moment Taylor series:

  * a Gaussian is exactly kappa_k = 0 for k >= 3, so the K = 2 cumulant
    expansion is EXACT (to all Taylor orders) for a Gaussian;
  * the raw moments m_n = <dp^n> = B_n(kappa_1, ..., kappa_n) (complete Bell
    polynomials) are in general non-zero to arbitrarily high order for a
    Gaussian, so a raw-moment Taylor truncation silently drops non-vanishing
    terms;
  * for independent parameters the cross-cumulants vanish (additivity).

Numerically we evaluate the cumulant expansion by converting cumulants to raw
moments with the Bell recursion and summing the Taylor series to order N >= K:

    m_0 = 1,   m_n = sum_{j=1}^{min(n,K)} C(n-1, j-1) kappa_j m_{n-j},
    <S> = sum_{n=0}^{N} S^(n)(p0) m_n / n! .
"""

from __future__ import annotations

import math

import numpy as np


def raw_moments_from_cumulants(kappa, N):
    """Raw moments m_0..m_N from cumulants kappa_1..kappa_K (K = len(kappa)).

    Complete Bell polynomial recursion, with kappa_j = 0 for j > K.  Returns a
    list of length N+1 with m_n = <dp^n>.
    """
    K = len(kappa)
    m = [1.0]  # m_0 = 1
    for n in range(1, N + 1):
        mn = sum(
            math.comb(n - 1, j - 1) * kappa[j - 1] * m[n - j]
            for j in range(1, min(n, K) + 1)
        )
        m.append(mn)
    return m


def cumulant_expansion(S_derivs, kappa):
    """Ensemble average <S> from derivative spectra and cumulants.

    ``S_derivs`` : list [S(p0), S'(p0), ..., S^(N)(p0)] — derivatives of S at
        the reference point p0.
    ``kappa``    : list [kappa_1, ..., kappa_K] — cumulants of the *deviation*
        dp = p - p0 (so kappa_1 = <dp>, kappa_2 = Var(dp), ...).

    Returns sum_n S^(n) m_n / n! with m_n = <dp^n> from the Bell recursion
    (cumulant hierarchy truncated at order K, Taylor series truncated at N).
    """
    N = len(S_derivs) - 1
    m = raw_moments_from_cumulants(kappa, N)
    return float(sum(S_derivs[n] * m[n] / math.factorial(n) for n in range(N + 1)))


def gaussian_cumulants(mu, sigma):
    """Cumulants of a Gaussian variable ~ N(mu, sigma^2).

    Use as the deviation cumulants with reference point p0 = 0 (i.e. dp = p):
    kappa_1 = mu, kappa_2 = sigma^2, kappa_{k>=3} = 0.  For a reference at the
    mean p0 = mu, the deviation cumulants are instead [0, sigma^2].
    """
    return np.array([mu, sigma**2])


def raw_moments_gaussian(mu, sigma, N):
    """Raw Gaussian moments m_0..m_N (nonzero to all orders)."""
    return raw_moments_from_cumulants(gaussian_cumulants(mu, sigma), N)
