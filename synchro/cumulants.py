"""Finite Bell/moment algebra and finite kernel Taylor contraction.

First K moments and cumulants are invertible coordinate descriptions of the
same statistics, assuming the moments exist. The scalar helper can optionally
set unprovided higher cumulants to zero while constructing moments through N;
this is an additional closure, not a general positive-PDF reconstruction.
It must not be identified with a legal polynomial log-characteristic function:
Marcinkiewicz excludes non-Gaussian exact finite polynomial log CFs of degree>2.

Gaussian K=2 specifies the full Gaussian distribution, but evaluating a kernel
from only N derivatives still leaves its Taylor remainder. Equality with an
infinite differential-operator series requires convergence and interchange
conditions. For a given finite polynomial and the same supplied information,
moment and cumulant contractions are identical.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def raw_moments_from_cumulants(kappa, N):
    """Raw moments m_0..m_N from cumulants kappa_1..kappa_K (K = len(kappa)).

    Complete Bell polynomial recursion, with kappa_j = 0 for j > K.  Returns a
    list of length N+1 with m_n = <dp^n>.
    """
    if not isinstance(N, int) or N < 0:
        raise ValueError("N must be a nonnegative static integer")
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
    return jnp.asarray(
        sum(S_derivs[n] * m[n] / math.factorial(n) for n in range(N + 1))
    )


def gaussian_cumulants(mu, sigma):
    """Cumulants of a Gaussian variable ~ N(mu, sigma^2).

    Use as the deviation cumulants with reference point p0 = 0 (i.e. dp = p):
    kappa_1 = mu, kappa_2 = sigma^2, kappa_{k>=3} = 0.  For a reference at the
    mean p0 = mu, the deviation cumulants are instead [0, sigma^2].
    """
    return jnp.asarray([mu, sigma**2])


def raw_moments_gaussian(mu, sigma, N):
    """Raw Gaussian moments m_0..m_N (nonzero to all orders)."""
    return raw_moments_from_cumulants(gaussian_cumulants(mu, sigma), N)


def vector_cumulant_expansion(S_derivs, kappas):
    """Vector (multi-parameter) cumulant expansion, truncated at Taylor degree K <= 4.

    Generalises :class:`synchro.expansion.CumulantExpansion` beyond second
    order by adding the non-Gaussian cumulants kappa_3, kappa_4.  Parameters:

    ``S_derivs`` : list of derivative tensors of S at p0,
        [S0, grad(P), hess(P,P), third(P,P,P), fourth(P,P,P,P)].
    ``kappas``   : list of cumulant tensors of the deviations dp = p - p0,
        [kappa1(P), kappa2(P,P), kappa3(P,P,P), kappa4(P,P,P,P)] (up to the
        desired order K = len(kappas)).

    Returns the degree-K kernel average expressed through cumulants:

        K=1: kappa_i S^(i)
        K=2: (1/2) kappa_ij S^(ij) + (1/2) kappa_i kappa_j S^(ij)
        K=3: (1/6) kappa_ijk S^(ijk) + (1/2) kappa_ij kappa_k S^(ijk)
             + (1/6) kappa_i kappa_j kappa_k S^(ijk)
        K=4: (1/24) kappa_ijkl S^(ijkl) + (1/6) kappa_ijk kappa_l S^(ijkl)
             + (1/8) kappa_ij kappa_kl S^(ijkl)
             + (1/4) kappa_ij kappa_k kappa_l S^(ijkl)
             + (1/24) kappa_i kappa_j kappa_k kappa_l S^(ijkl)

    (indices summed implicitly).  For K=2 this coincides with the second-order
    moment expansion; the coefficients are the multivariate Faà di Bruno
    (complete Bell) weights.  JAX-native, so it is jittable/differentiable.
    """
    K = len(kappas)
    if K > 4:
        raise ValueError("vector_cumulant_expansion supports at most four orders")
    if len(S_derivs) < K + 1:
        raise ValueError("need derivative tensors through the requested Taylor order")
    out = S_derivs[0]
    if K < 1:
        return out
    k1 = kappas[0]
    out = out + jnp.einsum("i,i->", k1, S_derivs[1])
    if K < 2:
        return out
    k2 = kappas[1]
    out = out + 0.5 * jnp.einsum("ij,ij->", k2, S_derivs[2])
    out = out + 0.5 * jnp.einsum("i,j,ij->", k1, k1, S_derivs[2])
    if K < 3:
        return out
    k3 = kappas[2]
    out = out + (1.0 / 6.0) * jnp.einsum("ijk,ijk->", k3, S_derivs[3])
    out = out + 0.5 * jnp.einsum("ij,k,ijk->", k2, k1, S_derivs[3])
    out = out + (1.0 / 6.0) * jnp.einsum("i,j,k,ijk->", k1, k1, k1, S_derivs[3])
    if K < 4:
        return out
    k4 = kappas[3]
    out = out + (1.0 / 24.0) * jnp.einsum("ijkl,ijkl->", k4, S_derivs[4])
    out = out + (1.0 / 6.0) * jnp.einsum("ijk,l,ijkl->", k3, k1, S_derivs[4])
    out = out + (1.0 / 8.0) * jnp.einsum("ij,kl,ijkl->", k2, k2, S_derivs[4])
    out = out + (1.0 / 4.0) * jnp.einsum("ij,k,l,ijkl->", k2, k1, k1, S_derivs[4])
    out = out + (1.0 / 24.0) * jnp.einsum("i,j,k,l,ijkl->", k1, k1, k1, k1, S_derivs[4])
    return out
