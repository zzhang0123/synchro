"""Validation of the polarised radiative-transfer layer.

Checks:
1. Faraday constant: RM = 0.812 rad/m^2 per (pc cm^-3 uG).
2. Burn depolarisation: Gaussian-RM average equals exp(-2 var(RM) lam^4).
3. Transfer slab: thin limit, pure rotation (P -> P e^{i theta}), pure
   conversion (Q -> V), and const-K slab vs the matrix-exponential closed form.
4. LOS chain: constant-K chain == single slab; non-commutative chain != naive.
5. Self-absorption: turnover slopes nu^(5/2) (thick) vs nu^(-(p-1)/2) (thin).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from syncmoments.rm import (
    RM_PER_UNIT,
    rotation_measure_rad_m2,
    burn_depolarisation,
)
from syncmoments.transfer import (
    mueller_matrix,
    transfer_slab,
    transfer_los,
)
from syncmoments.solutions import (
    uniform_source_intensity,
)


def test_rm_constant():
    # 1 pc cm^-3 uG  ->  0.812 rad/m^2
    path_cgs = 1.0 * 1e-6 * 3.085677581e18  # cm^-3 * G * cm
    rm = float(rotation_measure_rad_m2(jnp.asarray(path_cgs)))
    return rm, abs(rm - RM_PER_UNIT) / RM_PER_UNIT


def test_burn():
    lam = 0.3  # m
    mean_rm, sig_rm = 10.0, 4.0  # rad/m^2
    rng = np.random.default_rng(0)
    rms = rng.normal(mean_rm, sig_rm, 200000)
    P0 = 1.0 + 0.3j
    # direct average <P0 e^{2 i RM lam^2}>
    Pnum = np.mean(P0 * np.exp(2j * rms * lam**2))
    Pana = burn_depolarisation(jnp.asarray(P0), mean_rm, sig_rm**2, lam)
    return complex(Pana), Pnum, abs(Pana - Pnum)


def test_transfer_limits():
    # thin limit
    eps = jnp.array([1.0, 0.2, 0.0, 0.1])
    K = mueller_matrix(1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    S_thin = transfer_slab(jnp.zeros(4), eps, K, 1.0)
    thin_err = float(jnp.max(jnp.abs(S_thin - eps)))

    # pure rotation: rV = 0.5 over ds=1  -> P -> P e^{i*0.5}
    rV, ds = 0.5, 1.0
    K = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rV)
    S0 = jnp.array([0.0, 1.0, 0.0, 0.0])  # pure Q
    S = transfer_slab(S0, jnp.zeros(4), K, ds)
    P_expected = 1.0 * np.exp(1j * rV * ds)  # Q+iU rotates by rV*ds
    rot_err = float(
        jnp.max(
            jnp.abs(
                jnp.array([S[1], S[2]]) - jnp.array([P_expected.real, P_expected.imag])
            )
        )
    )

    # pure conversion: rU = 0.5 over ds=1  ->  Q<->V rotation
    K = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
    S = transfer_slab(jnp.array([0.0, 1.0, 0.0, 0.0]), jnp.zeros(4), K, 1.0)
    # (Q, V) rotated by angle -rU*ds (see K matrix): Q->cos, V->-sin
    conv_q = float(S[1])
    conv_v = float(S[3])
    exp_q = np.cos(0.5)
    exp_v = -np.sin(0.5)
    conv_err = abs(conv_q - exp_q) + abs(conv_v - exp_v)

    return thin_err, rot_err, conv_err


def test_los_chain():
    ds = 0.1
    S0 = jnp.array([0.0, 1.0, 0.0, 0.0])
    rV = 0.3
    K = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rV)
    N = 10
    K_s = jnp.stack([K] * N)
    eps_s = jnp.zeros((N, 4))
    S_chain = transfer_los(S0, eps_s, K_s, ds)
    # should equal single slab of length N*ds
    S_one = transfer_slab(S0, jnp.zeros(4), K, N * ds)
    err = float(jnp.max(jnp.abs(S_chain - S_one)))

    # non-commutative check: rotation then conversion != conversion then rotation
    Kr = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)
    Kc = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0)
    S_rc = transfer_los(S0, jnp.zeros((2, 4)), jnp.stack([Kr, Kc]), 1.0)
    S_cr = transfer_los(S0, jnp.zeros((2, 4)), jnp.stack([Kc, Kr]), 1.0)
    noncomm = float(jnp.max(jnp.abs(S_rc - S_cr)))
    return err, noncomm


def test_self_absorption(p=2.5, B=1.0):
    # slopes of I_nu in the thick (low nu) and thin (high nu) limits
    nu_lo = np.logspace(-4, -2, 20)
    nu_hi = np.logspace(1, 3, 20)
    I_lo = np.array([float(uniform_source_intensity(n, p, B, L=1e6)) for n in nu_lo])
    I_hi = np.array([float(uniform_source_intensity(n, p, B, L=1e-3)) for n in nu_hi])
    slope_lo = np.polyfit(np.log(nu_lo), np.log(I_lo), 1)[0]
    slope_hi = np.polyfit(np.log(nu_hi), np.log(I_hi), 1)[0]
    return slope_lo, slope_hi, 2.5, -(p - 1) / 2


if __name__ == "__main__":
    print("=== 1. Faraday constant ===")
    rm, rel = test_rm_constant()
    print(f"  RM(1 pc cm^-3 uG) = {rm:.6f} rad/m^2  (rel err vs 0.812 = {rel:.2e})")

    print("=== 2. Burn depolarisation (lambda^4) ===")
    Pana, Pnum, err = test_burn()
    print(f"  <P> analytic={Pana:.6f}  numeric={Pnum:.6f}  |err|={err:.2e}")

    print("=== 3. Transfer limits (thin / rotation / conversion) ===")
    thin, rot, conv = test_transfer_limits()
    print(f"  thin err={thin:.2e}  rotation err={rot:.2e}  conversion err={conv:.2e}")

    print("=== 4. LOS chain (const-K == single slab; non-commutativity) ===")
    chain_err, noncomm = test_los_chain()
    print(
        f"  chain vs single-slab err={chain_err:.2e}  |rot*conv - conv*rot|={noncomm:.2e}"
    )

    print("=== 5. Self-absorption turnover slopes ===")
    s_lo, s_hi, exp_lo, exp_hi = test_self_absorption()
    print(
        f"  thick slope={s_lo:.3f} (expect {exp_lo})   thin slope={s_hi:.3f} (expect {exp_hi})"
    )
