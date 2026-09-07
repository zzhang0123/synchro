"""Validation of the PySync model against exact/known results.

All heavy harmonic sums use SciPy Bessel (fast, and the *formula* — not the
JAX kernels — is what the total-power check exercises).  The JAX-specific
paths (pointwise match, F/G, autodiff derivatives, moment expansion) are
checked separately with jit/vmap where appropriate.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv, jvp, kv as scipy_kv
from scipy.integrate import quad

import jax
import jax.numpy as jnp

from synchro.stokes import stokes_harmonic
from synchro.derivatives import derivative_spectra, _stokes_stack
from synchro.expansion import build_expansion
from synchro.ultrarel import F, G


# --- SciPy reference of the (corrected) harmonic Stokes ---------------------
def stokes_scipy(n, gamma, alpha, theta):
    beta = np.sqrt(1 - 1 / gamma**2)
    bp = beta * np.cos(alpha)
    bt = beta * np.sin(alpha)
    st, ct = np.sin(theta), np.cos(theta)
    D = 1 - bp * ct
    x = n * bt * st / D
    Jn = jv(n, x)
    Jnp = jvp(n, x, 1)
    Ppar = n**2 * (ct - bp) ** 2 / st**2 * Jn**2 / D**3
    Pperp = n**2 * bt**2 * Jnp**2 / D**3
    I = Ppar + Pperp
    Q = Ppar - Pperp
    V = 2.0 * n**2 * bt * (ct - bp) / (st * D**3) * Jn * Jnp
    return I, Q, V


def test_larmor(gamma=2.0, alpha=np.pi / 4, nmax=400, ntheta=3000):
    thetas = np.linspace(1e-3, np.pi - 1e-3, ntheta)
    dth = thetas[1] - thetas[0]
    beta = np.sqrt(1 - 1 / gamma**2)
    total = 0.0
    for n in range(1, nmax + 1):
        I = np.array([stokes_scipy(n, gamma, alpha, th)[0] for th in thetas])
        total += np.sum(I * np.sin(thetas)) * dth
    total *= 2 * np.pi
    expected = (4 * np.pi / 3) * gamma**4 * (beta * np.sin(alpha)) ** 2
    return total, expected, total / expected


def test_jax_matches_scipy(gamma=3.0, alpha=np.pi / 4, theta=np.pi / 3, nmax=50):
    errs = []
    for n in range(1, nmax + 1):
        a = stokes_harmonic(n, gamma, alpha, theta)
        b = stokes_scipy(n, gamma, alpha, theta)
        errs.append(max(abs(float(a[k]) - b[k]) for k in range(3)))
    return max(errs)


def test_full_polarisation(n=7, gamma=3.0, alpha=np.pi / 4, theta=np.pi / 3):
    I, Q, V = stokes_scipy(n, gamma, alpha, theta)
    return float(I**2 - (Q**2 + V**2))


def test_V_isotropy(gamma=5.0, theta=np.pi / 3, nmax=20, nalpha=3000):
    alphas = np.linspace(1e-4, np.pi - 1e-4, nalpha)
    da = alphas[1] - alphas[0]
    Vsum = Isum = 0.0
    for n in range(1, nmax + 1):
        for a in alphas:
            I, _, V = stokes_scipy(n, gamma, a, theta)
            Vsum += V * np.sin(a) * da
            Isum += I * np.sin(a) * da
    return Vsum, Isum, Vsum / Isum


def test_G():
    xs = np.array([0.05, 0.1, 0.5, 1.0, 3.0, 10.0])
    g = jax.jit(G)
    errs = {}
    for x in xs:
        ref = x * scipy_kv(2.0 / 3.0, x)
        errs[x] = abs(float(g(jnp.asarray(x))) - ref) / ref
    return errs


def test_F():
    xs = np.array([0.05, 0.1, 0.5, 1.0, 3.0, 10.0])
    f = jax.jit(F)
    errs = {}
    for x in xs:
        ref = x * quad(lambda xi: scipy_kv(5.0 / 3.0, xi), x, np.inf)[0]
        errs[x] = abs(float(f(jnp.asarray(x))) - ref) / ref
    return errs


def test_derivatives(n=10, gamma0=5.0, alpha0=np.pi / 4, theta0=np.pi / 3, h=1e-4):
    val, grad, hess = derivative_spectra(n, gamma0, alpha0, theta0)
    p = jnp.array([gamma0, alpha0, theta0])
    f = lambda pp: _stokes_stack(n, pp)  # noqa: E731
    fd = []
    for i in range(3):
        pp = p.at[i].add(h)
        pm = p.at[i].add(-h)
        fd.append((f(pp) - f(pm)) / (2 * h))
    fd = jnp.stack(fd, axis=1)
    gerr = float(jnp.max(jnp.abs(grad - fd) / (1e-12 + jnp.abs(fd))))
    return val, grad, gerr


def test_cumulant_expansion(
    ns=(1, 2, 3, 5, 10, 20),
    gamma0=5.0,
    alpha0=np.pi / 4,
    theta0=np.pi / 3,
    sigma_g=0.3,
    sigma_a=0.1,
    n_samples=200000,
):
    exp = build_expansion(ns, gamma0, alpha0, theta0)
    mu = jnp.zeros(3)
    cov = jnp.diag(jnp.array([sigma_g**2, sigma_a**2, 0.0]))
    approx = exp(mu, cov)

    rng = np.random.default_rng(0)
    gs = rng.normal(gamma0, sigma_g, n_samples)
    als = rng.normal(alpha0, sigma_a, n_samples)
    exact = np.zeros((len(ns), 3))
    for k, n in enumerate(ns):
        acc = np.zeros(3)
        for g, a in zip(gs, als):
            I, Q, V = stokes_scipy(n, g, a, theta0)
            acc += (I, Q, V)
        exact[k] = acc / n_samples
    rel = np.abs(np.asarray(approx) - exact) / (1e-12 + np.abs(exact))
    return np.asarray(approx), exact, rel


if __name__ == "__main__":
    print("=== 1. Larmor total power (sum over harmonics + solid angle) ===")
    for g, a in [(2.0, np.pi / 4), (5.0, np.pi / 4), (2.0, np.pi / 2)]:
        tot, expv, ratio = test_larmor(gamma=g, alpha=a)
        print(f"  gamma={g:4.1f} alpha={a:.3f}: ratio={ratio:.6f}")

    print("=== 2. JAX kernels match SciPy reference (pointwise) ===")
    print(f"  max abs diff over n=1..50: {test_jax_matches_scipy():.2e}")

    print("=== 3. Full polarisation I^2 = Q^2 + V^2 ===")
    print(f"  residual = {test_full_polarisation():.3e}")

    print("=== 4. Isotropic pitch-angle average of V (theta=60 deg) ===")
    Vs, Is, frac = test_V_isotropy()
    print(f"  <V>={Vs:+.4e}  <I>={Is:.4e}  V/I={frac:+.4f}  (nonzero)")

    print("=== 5. Ultra-relativistic G(x), F(x) vs scipy ===")
    print("  G rel-err:", {f"{k:.2f}": f"{v:.1e}" for k, v in test_G().items()})
    print("  F rel-err:", {f"{k:.2f}": f"{v:.1e}" for k, v in test_F().items()})

    print("=== 6. Derivative spectra: autodiff vs finite difference ===")
    val, grad, gerr = test_derivatives()
    print(f"  S(n=10) = {np.asarray(val)}")
    print(f"  grad rel err = {gerr:.2e}")

    print("=== 7. Moment expansion vs direct Gaussian ensemble ===")
    approx, exact, rel = test_cumulant_expansion()
    for k, n in enumerate((1, 2, 3, 5, 10, 20)):
        print(f"    n={n:2d}: I={rel[k,0]:.2e}  Q={rel[k,1]:.2e}  V={rel[k,2]:.2e}")
