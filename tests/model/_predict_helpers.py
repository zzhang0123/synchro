"""Populations, oracles and stub inputs shared by the ``predict`` tests (not collected).

Oracles are NumPy sums over discrete populations of the ``PolynomialTestKernel``
(``eq: finite joint response`` with the main-text cutoff ``r + s + b <= N``).
"""

import math

import numpy as np
from scipy.special import eval_legendre

from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.channels import Channels
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.kernels import PolynomialTestKernel, required_m_max
from syncmoments.model.moments import PopulationSamples, Reference, Support

GAMMA0, B0 = 20.0, 2.0
LINE_NU = np.array([1.5e8, 4.0e8, 9.0e8])
TAU = 2 * (C_SI_M / LINE_NU) ** 2
S_DEPTH = 1.0 / TAU[0]
DEPTH_REF = 0.7 * S_DEPTH


# -- populations and oracles ---------------------------------------------------


def nine_atoms(seed=11, constant_depth=True):
    """Nine correlated atoms: every variable is a function of one latent ``t``."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, 9) + 0.05 * rng.standard_normal(9)
    depth = np.full(9, DEPTH_REF)
    if not constant_depth:
        depth = DEPTH_REF + S_DEPTH * (0.3 * t + 0.1 * t**2)
    return dict(
        gamma=GAMMA0 * (1 + 0.15 * t + 0.05 * t**2),
        B=B0 * (1 + 0.1 * t - 0.08 * t**2),
        mu=np.clip(0.7 * t + 0.1 * rng.standard_normal(9), -0.95, 0.95),
        eta=np.clip(-0.4 * t + 0.3 * t**2, -0.95, 0.95),
        phi=0.3 + 1.4 * t,
        depth=depth,
        w=rng.uniform(0.5, 2.0, 9),
    )


def samples_of(pop):
    return PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=pop["w"],
    )


def reference():
    return Reference(GAMMA0, B0, DEPTH_REF, scales=(GAMMA0, B0, S_DEPTH))


def support(truncated=None):
    return Support(
        gamma=(GAMMA0 * 0.7, GAMMA0 * 1.3),
        B=(B0 * 0.7, B0 * 1.3),
        depth=(DEPTH_REF - S_DEPTH, DEPTH_REF + S_DEPTH),
        truncated=truncated,
    )


def polynomial_kernel(seed=5, N=2, L=2, n_ch=3):
    """Random coefficients with total degree ``r + s <= N`` (exact at ``Truncation(L, L, N)``)."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((3, n_ch, L + 1, L + 1, N + 1, N + 1))
    for r in range(N + 1):
        for s in range(N + 1):
            if r + s > N:
                c[..., r, s] = 0.0
    for l in range(L + 1):
        for k in range(L + 1):  # eq: angular parity: I, Q even, V odd in l + k
            c[[0, 1] if (l + k) % 2 else [2], :, l, k] = 0.0
    kernel = PolynomialTestKernel(
        c, LINE_NU[:n_ch], gamma0=GAMMA0, B0=B0, s_gamma=GAMMA0, s_B=B0
    )
    return kernel, c


def polynomial_direct(pop, c, amplitude, total_degree=None):
    """NumPy direct average of the polynomial kernel with the per-line phase.

    With ``total_degree=N`` the polarised kernel is replaced by the retained
    sum ``sum_{r+s+b<=N} c_rs z_g^r z_B^s (i tau s_d z_d)^b / b!`` times
    ``exp(i tau depth_ref)`` (``eq: finite joint response``, main-text cutoff).
    """
    w = pop["w"] / pop["w"].sum()
    L, N = c.shape[2] - 1, c.shape[4] - 1
    tau = TAU[: c.shape[1]]
    out = np.zeros((c.shape[1], 4))
    for n in range(w.size):
        zg, zB = (pop["gamma"][n] - GAMMA0) / GAMMA0, (pop["B"][n] - B0) / B0
        pl = eval_legendre(np.arange(L + 1), pop["mu"][n])
        pk = eval_legendre(np.arange(L + 1), pop["eta"][n])
        pg, pb = zg ** np.arange(N + 1), zB ** np.arange(N + 1)
        K = np.einsum("sjlkrt,l,k,r,t->sj", c, pl, pk, pg, pb)
        zd = (pop["depth"][n] - DEPTH_REF) / S_DEPTH
        if total_degree is None:
            KP = K[1] * np.exp(1j * tau * pop["depth"][n])
        else:
            KP = np.zeros(c.shape[1], dtype=complex)
            for r in range(N + 1):
                for s in range(N + 1):
                    for b in range(total_degree + 1 - r - s):
                        if r + s > total_degree:
                            continue
                        KP += (
                            np.einsum("jlk,l,k->j", c[1, :, :, :, r, s], pl, pk)
                            * zg**r
                            * zB**s
                            * (1j * tau * S_DEPTH * zd) ** b
                            / math.factorial(b)
                        )
            KP = KP * np.exp(1j * tau * DEPTH_REF)
        P = np.exp(2j * pop["phi"][n]) * KP
        out += w[n] * np.stack([K[0], P.real, P.imag, K[2]], axis=1)
    return amplitude * out


def full_inputs(basis, n_ch=3):
    """Every optional budget input supplied (zeros): closes the budget slots."""
    index = basis.index
    return dict(
        errors=RemainderInputs(
            rho_ang=np.zeros((n_ch, 3)),
            H=np.zeros((n_ch, 3, index.n_lk)),
            absolute_moments=np.zeros((2, index.n_lk)),
        ),
        statistical_input=np.zeros(index.n_real),
        amplitude_uncertainty=0.0,
        depth_model=ErrorTerm.declared_zero(
            "depth exact in this test", shape=(n_ch, 4)
        ),
    )


# -- assumption triple, part (2), on the harmonic kernel -------------------------------


H_GAMMA0, H_B0 = 5.0, 1.0
H_NU_STAR = E_ESU * H_B0 / (2 * np.pi * H_GAMMA0 * M_E * C_CGS)
H_TAU_STAR = 2 * (C_SI_M / H_NU_STAR) ** 2
H_S_DEPTH = 1.0 / H_TAU_STAR
H_DEPTH_REF = 0.5 * H_S_DEPTH


def harmonic_setup():
    channels = Channels.bump(
        np.array([2.0, 4.0]) * H_NU_STAR, 0.5 * np.array([2.0, 4.0]) * H_NU_STAR
    )
    supp = Support(
        gamma=(4.0, 6.0), B=(0.8, 1.2), depth=(0.0, H_S_DEPTH), truncated=False
    )
    m_max = required_m_max(supp, channels)
    assert m_max <= 40
    kernel = HarmonicKernel(m_max, n_outer=16, n_inner=16)
    ref = Reference(H_GAMMA0, H_B0, H_DEPTH_REF, scales=(H_GAMMA0, H_B0, H_S_DEPTH))
    rng = np.random.default_rng(3)
    t = np.linspace(-1, 1, 9)
    pop = dict(
        gamma=H_GAMMA0 * (1 + 0.12 * t + 0.03 * t**2),
        B=H_B0 * (1 + 0.1 * t - 0.05 * t**2),
        mu=np.clip(0.6 * t + 0.1 * rng.standard_normal(9), -0.9, 0.9),
        eta=np.clip(0.3 - 0.5 * t, -0.9, 0.9),
        phi=0.4 + 1.1 * t,
        depth=H_DEPTH_REF + H_S_DEPTH * (0.25 * t + 0.1 * t**2),
        w=rng.uniform(0.5, 2.0, 9),
    )
    return kernel, channels, supp, ref, pop
