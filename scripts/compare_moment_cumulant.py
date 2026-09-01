"""Compare raw-moment (Taylor-N) vs cumulant (cumulant-K) truncation.

Demonstrates the truncation-philosophy point of Sec 4.2 with a NON-Gaussian
distribution: a Gamma(k, theta) whose cumulants kappa_n = (n-1)! k theta^n are
all non-zero.  For S(dp) = exp(dp) the exact answer is the moment-generating
function (1 - theta)^-k, so both series are exact when summed to infinity; the
comparison is which *truncation* converges faster.

  raw-moment Taylor-N : <S> ~ sum_{n=0}^N S^(n)(0) m_n / n!       (true m_n)
  cumulant-K          : <S> ~ sum_{n=0}^M S^(n)(0) m_n^(K) / n!   (m_n^(K) from
                          Bell(kappa_1..kappa_K, 0, ...), M large so the Taylor
                          sum is converged; only the cumulant hierarchy is cut)

Generates figures/moment_vs_cumulant.pdf.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc

from synchro.cumulants import raw_moments_from_cumulants

rc("text", usetex=False)
rc("font", family="serif", size=11)
rc("mathtext", fontset="cm")


def gamma_cumulants(k, theta, K):
    """Cumulants of a Gamma(k, theta): kappa_n = (n-1)! k theta^n."""
    return [math.factorial(n - 1) * k * theta**n for n in range(1, K + 1)]


def gamma_raw_moments(k, theta, N):
    """Raw moments m_n = theta^n Gamma(k+n)/Gamma(k)."""
    return [theta**n * math.gamma(k + n) / math.gamma(k) for n in range(N + 1)]


def exact_mgf(k, theta):
    """<exp(dp)> = (1 - theta)^-k for dp ~ Gamma(k, theta), theta < 1."""
    return (1.0 - theta) ** (-k)


def main():
    k, theta = 5.0, 0.2  # mean 1, var 0.2, skew ~0.89 (clearly non-Gaussian)
    exact = exact_mgf(k, theta)

    # cumulant-K error (Taylor sum converged at M=40)
    M = 40
    Ks = range(1, 9)
    cumul_err = []
    for K in Ks:
        kappa = gamma_cumulants(k, theta, K)
        mK = raw_moments_from_cumulants(kappa, M)
        approx = sum(mK[n] / math.factorial(n) for n in range(M + 1))  # S^(n)(0)=1
        cumul_err.append(abs(approx - exact) / exact)

    # raw-moment Taylor-N error (true raw moments)
    Ns = range(1, 21)
    raw_err = []
    for N in Ns:
        m = gamma_raw_moments(k, theta, N)
        approx = sum(m[n] / math.factorial(n) for n in range(N + 1))
        raw_err.append(abs(approx - exact) / exact)

    # summary
    print(f"Gamma(k={k}, theta={theta}), exact <e^dp> = {exact:.6f}")
    print("cumulant-K truncation:")
    for K, e in zip(Ks, cumul_err):
        print(f"  K={K}: rel err = {e:.3e}")
    print("raw-moment Taylor-N truncation:")
    for N in [1, 2, 3, 4, 6, 8, 10, 15, 20]:
        print(f"  N={N:2d}: rel err = {raw_err[N-1]:.3e}")

    # figure
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogy(list(Ns), raw_err, "C0-o", lw=1.4, ms=4,
                label="raw-moment Taylor-$N$")
    ax.semilogy(list(Ks), cumul_err, "C1-s", lw=1.4, ms=4,
                label="cumulant $K$")
    ax.set_xlabel("Truncation order ($N$ or $K$)")
    ax.set_ylabel(r"relative error in $\langle e^{\delta p}\rangle$")
    ax.set_title(rf"Gamma($k={k}$, $\theta={theta}$): moment vs cumulant truncation")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()

    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "figures", "moment_vs_cumulant.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # assert cumulant converges at least as fast as raw-moment (at K=N)
    for K in range(1, 9):
        assert cumul_err[K - 1] <= raw_err[K - 1] * 1.01
    print("cumulant-K converges no slower than raw Taylor-N at equal order")


if __name__ == "__main__":
    main()
