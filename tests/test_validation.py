"""Pytest layer: assert the physics and design checks (imported from scripts/).

This complements scripts/run_all.py (which just runs each script and checks
the exit code). Here each check becomes an assert, so a regression fails loudly
with a clear test name.  The check functions are imported as *modules* (not as
``test_*`` names) so pytest neither collides nor re-collects them.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import validate_model
import validate_transfer
import validate_kirchhoff
import validate_rt_moments
import validate_sed
import validate_cumulants
import test_design
from synchro.sed import power_law_emissivity_rel


# --- emissivity side -------------------------------------------------------

def test_larmor_total_power():
    _, _, ratio = validate_model.test_larmor(gamma=2.0, alpha=np.pi / 4)
    assert abs(ratio - 1.0) < 1e-4
    _, _, ratio = validate_model.test_larmor(gamma=5.0, alpha=np.pi / 4)
    assert abs(ratio - 1.0) < 0.02  # finite harmonic sum


def test_jax_kernels_match_scipy():
    assert validate_model.test_jax_matches_scipy() < 1e-10


def test_full_polarisation_identity():
    assert abs(validate_model.test_full_polarisation()) < 1e-12


def test_circular_polarisation_nonzero_for_isotropic_pitch():
    _, _, frac = validate_model.test_V_isotropy()
    assert abs(frac) > 0.1  # nonzero away from theta = pi/2


def test_ultrarelativistic_F_and_G_match_scipy():
    assert max(validate_model.test_G().values()) < 1e-9   # G(x) = x K_{2/3}(x)
    assert max(validate_model.test_F().values()) < 1e-9   # F(x) = x int_x^inf K_{5/3}


def test_derivative_spectra_match_finite_difference():
    _, _, gerr = validate_model.test_derivatives()
    assert gerr < 1e-5  # limited by the FD step, not by the autodiff


def test_cumulant_expansion_matches_gaussian_ensemble():
    # reduced sample count keeps the suite fast; see scripts/ for the full run
    _, _, rel = validate_model.test_cumulant_expansion(
        ns=(1, 2, 3), sigma_g=0.3, sigma_a=0.1, n_samples=20000)
    assert float(np.max(rel)) < 0.02  # 2nd-order truncation at low harmonics


# --- radiative transfer ----------------------------------------------------

def test_faraday_constant():
    _, rel = validate_transfer.test_rm_constant()
    assert rel < 1e-3  # 0.8119 vs 0.812


def test_transfer_slab_limits():
    thin, rot, conv = validate_transfer.test_transfer_limits()
    assert thin < 1e-4
    assert rot < 1e-12
    assert conv < 1e-12


def test_burn_depolarisation_matches_direct_average():
    _, _, err = validate_transfer.test_burn()
    assert err < 5e-3  # Monte-Carlo noise of the direct <P0 exp(2i RM lam^2)>


def test_los_chain_and_noncommutativity():
    err, noncomm = validate_transfer.test_los_chain()
    assert err < 1e-12    # constant-K chain == one slab of the total length
    assert noncomm > 1e-3  # rotation and conversion genuinely do not commute


def test_self_absorption_turnover_slopes():
    s_lo, s_hi, e_lo, e_hi = validate_transfer.test_self_absorption()
    assert abs(s_lo - e_lo) < 0.01
    assert abs(s_hi - e_hi) < 0.01


# --- Kirchhoff / absorption ------------------------------------------------

def test_powerlaw_slopes_exact():
    sj, sa, ss, e_j, e_a, e_s = validate_kirchhoff.test_powerlaw_slopes()
    assert abs(sj - e_j) < 0.01
    assert abs(sa - e_a) < 0.01
    assert abs(ss - e_s) < 0.01


def test_polarisation_fractions_thin_and_thick():
    for _, pt, pt_th, pk, pk_th in validate_kirchhoff.test_polarisation_fraction():
        assert abs(pt - pt_th) < 1e-3   # (p+1)/(p+7/3)
        assert abs(pk - pk_th) < 1e-3   # -3/(6p+13)


def test_source_function_independent_of_amplitude():
    _, _, rel = validate_kirchhoff.test_closure()
    assert rel < 1e-10  # S = j/alpha is amplitude-free by construction


def test_kirchhoff_moment_expansion_vs_quadrature():
    _, _, _, _, rel_j, rel_a = validate_kirchhoff.test_moment_expansion()
    assert rel_j < 1e-3
    assert rel_a < 1e-2  # absorption probes the derivative-weighted N, so noisier


def test_absorption_moments_are_linear_functional_of_emissivity_moments():
    exact_err, approx_err = validate_kirchhoff.test_closure_form()
    assert max(exact_err) < 1e-3   # D_k from (M_k, <1/gamma>) is exact
    assert approx_err < 1e-2       # geometric series for <1/gamma> is approximate


def test_absorbed_intensity_from_moments_end_to_end():
    _, _, _, rel = validate_kirchhoff.test_end_to_end()
    assert float(np.max(rel)) < 2e-2


# --- Magnus / conversion / cross-cumulants ---------------------------------

def test_magnus_convergence():
    e1, e2 = validate_rt_moments.test_magnus()
    assert e2 < e1          # order-2 beats order-1
    assert e2 < 1e-4


def test_conversion_scaling():
    _, _, _, rs = validate_rt_moments.test_conversion()
    assert abs(rs - 0.5) < 1e-3  # rho_Q ~ nu^-3 vs rho_V ~ nu^-2


def test_emissivity_rotation_cross_cumulant_is_nonnegligible():
    P_corr, P_unc, dP = validate_rt_moments.test_cross_cumulant()
    # the same field drives emission and rotation, so this term cannot be
    # set to zero a priori (RT_FRAMEWORK section 7.2)
    assert dP > 1e-3
    assert P_corr != P_unc


def test_moment_driven_slab_thin_absorbed_and_rotated():
    thin_err, abs_err, rot_err = validate_rt_moments.test_moment_slab()
    assert thin_err < 1e-5  # L -> 0 gives I = j L
    assert abs_err < 1e-8   # matches the Kirchhoff closed form
    assert rot_err < 1e-5   # (Q, U) rotated by rho_V L


# --- SED -------------------------------------------------------------------

def test_spectral_index_exact():
    for _, a, t in validate_sed.test_spectral_index():
        assert abs(a - t) < 1e-3


def test_curvature_is_var_p_over_4():
    c, t = validate_sed.test_curvature()
    assert abs(c - t) / t < 0.03  # Gamma-prefactor correction is O(Var(p)^2)


def test_emissivity_differentiable_in_p():
    g = jax.grad(lambda p: power_law_emissivity_rel(1.0, p))(jnp.asarray(2.5))
    assert bool(jnp.isfinite(g))


def test_relative_emissivity_matches_double_integral():
    for _, _, _, ratio in validate_sed.test_rel_emissivity():
        assert abs(ratio - 1.0) < 1e-3


def test_absolute_emissivity_matches_relative_form():
    _, _, ratio = validate_sed.test_abs_emissivity()
    assert abs(ratio - 1.0) < 1e-8  # algebraic identity


def test_los_spectral_index_cumulants():
    (lin, quad, cub), (c_lin, c_quad, c_cub) = validate_sed.test_los_cumulants()
    assert abs(lin - c_lin) < 1e-3                    # -<alpha_s>
    assert abs(quad - c_quad) / abs(quad) < 0.05      # Var(alpha_s)/2
    assert abs(cub - c_cub) < 1e-3                    # -Skew(alpha_s)/6


def test_running_spectral_index_log_parabola():
    slope, pred_slope, a_num, a_pred = validate_sed.test_running_index()
    assert abs(slope - pred_slope) / pred_slope < 0.1   # finite nu grid
    assert abs(a_num - a_pred) / a_pred < 0.05


# --- strict cumulant expansion ---------------------------------------------

def test_cumulant_gaussian_exactness():
    approx, exact, raw2 = validate_cumulants.test_cumulant_expansion_exact_for_gaussian()
    assert abs(approx - exact) / exact < 1e-8   # cumulant K=2 is exact
    assert abs(raw2 - exact) / exact > 1e-2     # raw Taylor-2 is not


def test_cumulant_gaussian_moments_nonzero():
    m3, m4 = validate_cumulants.test_gaussian_moments_vs_cumulants()
    assert abs(m3) > 1e-3 and abs(m4) > 1e-3    # raw moments nonzero
    # (the Gaussian cumulants kappa_3, kappa_4 vanish exactly by construction)


def test_vector_cumulant_expansion_reduces_and_jits():
    # P=1 reduction to the scalar Bell sum and jit-ability are asserted inside
    out = validate_cumulants.test_vector_cumulant()
    assert np.isfinite(out)


# --- design (jit / autodiff) ----------------------------------------------

def test_design_cumulant_expansion_jit():
    test_design.test_cumulant_expansion_jit()


def test_design_cumulant_expansion_grad():
    test_design.test_cumulant_expansion_grad()


def test_design_transfer_los_jit():
    test_design.test_transfer_los_jit()


# --- public API surface ----------------------------------------------------
# The validate_*.py checks above exercise the physics *paths*; the tests below
# pin the remaining exported functions, which is where signature-level defects
# hide (they are invisible to a physics check that never calls them).

def test_bessel_kernels_match_scipy():
    from scipy.special import jv, jvp
    from synchro.bessel import bessel_jn, bessel_jn_prime, bessel_jn_and_prime
    for n in (0, 1, 5, 20):
        for x in (1e-3, 0.5, 3.0, 18.0):
            assert abs(float(bessel_jn(n, x)) - jv(n, x)) < 1e-12
            assert abs(float(bessel_jn_prime(n, x)) - jvp(n, x)) < 1e-12
    j, jp = bessel_jn_and_prime(5, 3.0)
    assert abs(float(j) - jv(5, 3.0)) < 1e-12
    assert abs(float(jp) - jvp(5, 3.0)) < 1e-12


def test_larmor_power_and_dimensional_stokes_scale_as_B_squared():
    from synchro.stokes import larmor_power, stokes_harmonic
    p1 = float(larmor_power(5.0, np.pi / 4, 1e-6))
    p2 = float(larmor_power(5.0, np.pi / 4, 2e-6))
    assert abs(p2 / p1 - 4.0) < 1e-10          # P ~ omega_B^2 ~ B^2
    I1, _, _ = stokes_harmonic(3, 5.0, np.pi / 4, np.pi / 3, B=1e-6)
    I2, _, _ = stokes_harmonic(3, 5.0, np.pi / 4, np.pi / 3, B=2e-6)
    assert abs(float(I2) / float(I1) - 4.0) < 1e-10
    I0, _, _ = stokes_harmonic(3, 5.0, np.pi / 4, np.pi / 3)  # B=None: dimensionless
    assert float(I0) != float(I1)


def test_apply_B_is_exact_second_moment():
    from synchro.expansion import build_expansion
    exp = build_expansion([1, 2, 3], 5.0, np.pi / 4, np.pi / 3)
    S = exp(jnp.zeros(3), jnp.zeros((3, 3)))
    B0, mu_B, sig_B = 1e-6, 2e-7, 3e-7
    got = exp.apply_B(S, B0, mu_B, sig_B**2)
    rng = np.random.default_rng(0)
    Bs = rng.normal(B0 + mu_B, sig_B, 400000)
    expected = np.asarray(S) * np.mean((Bs / B0) ** 2)   # <B^2> is exact
    assert np.max(np.abs(np.asarray(got) - expected) / np.abs(expected)) < 5e-3


def test_rm_utilities_are_mutually_consistent():
    from synchro.rm import (rotation_measure_practical, rotation_angle,
                            gaussian_rm_cumulants, burn_depolarisation)
    s = np.linspace(0.0, 1.0, 501)                      # pc
    rm = rotation_measure_practical(np.ones_like(s), np.ones_like(s), s)
    assert abs(float(rm) - 0.812) < 1e-6                 # 1 pc cm^-3 uG
    # chi = RM lam^2 : doubling lambda quadruples the angle
    chi1 = float(rotation_angle(10.0, 1e-6))
    chi2 = float(rotation_angle(20.0, 1e-6))
    assert abs(chi2 / chi1 - 4.0) < 1e-10
    rng = np.random.default_rng(1)
    samples = rng.normal(12.0, 3.0, 200000)
    mean, var = gaussian_rm_cumulants(samples)
    assert abs(float(mean) - 12.0) < 0.05
    assert abs(float(var) - 9.0) < 0.15
    # zero variance => pure rotation, no depolarisation
    P = burn_depolarisation(1.0 + 0j, float(mean), 0.0, 0.3)
    assert abs(abs(complex(P)) - 1.0) < 1e-12


def test_limiting_solutions():
    from synchro.solutions import (thin_limit, self_absorbed_source_function,
                                   uniform_source_intensity,
                                   thin_rotation_depolarisation)
    assert float(thin_limit(jnp.asarray(2.0), 0.5)) == 2.0 * 0.5
    s1 = float(self_absorbed_source_function(1.0, 2.5, 1.0))
    s2 = float(self_absorbed_source_function(2.0, 2.5, 1.0))
    assert abs(s2 / s1 - 2.0 ** 2.5) < 1e-10             # S ~ nu^(5/2)
    s3 = float(self_absorbed_source_function(1.0, 2.5, 4.0))
    assert abs(s3 / s1 - 4.0 ** -0.5) < 1e-10            # S ~ B^(-1/2)
    # optically thin -> I -> eps L ; optically thick -> I -> S (saturates).
    # The thin end is a boundary case: tau here is ~1e-21, where the naive
    # 1-exp(-tau) underflows to exactly zero (hence -expm1 in solutions.py).
    from synchro.solutions import power_law_emissivity
    L = 1e-8
    thin = float(uniform_source_intensity(1e4, 2.5, 1.0, L))
    eps_L = float(power_law_emissivity(1e4, 2.5, 1.0)) * L
    assert thin > 0.0
    assert abs(thin / eps_L - 1.0) < 1e-9
    thick = float(uniform_source_intensity(1e-4, 2.5, 1.0, 1e12))
    sat = float(self_absorbed_source_function(1e-4, 2.5, 1.0))
    assert abs(thick / sat - 1.0) < 1e-6
    # Burn wrapper agrees with the complex form
    Q, U = thin_rotation_depolarisation(1.0, 0.0, 10.0, 4.0, 0.3)
    assert np.isfinite(float(Q)) and np.isfinite(float(U))
    assert float(Q) ** 2 + float(U) ** 2 < 1.0           # depolarised


def test_transfer_variable_step_and_conversion_matrix():
    from synchro.transfer import (mueller_matrix, transfer_los,
                                  conversion_matrix, faraday_rotation_matrix)
    K = mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4)
    S0 = jnp.array([0.0, 1.0, 0.0, 0.0])
    N = 6
    K_s = jnp.stack([K] * N)
    eps_s = jnp.zeros((N, 4))
    uniform = transfer_los(S0, eps_s, K_s, 0.1)
    per_slab = transfer_los(S0, eps_s, K_s, jnp.full(N, 0.1))  # array ds branch
    assert float(jnp.max(jnp.abs(uniform - per_slab))) < 1e-12
    # conversion mixes Q and V but conserves Q^2 + V^2
    M = conversion_matrix(0.7)
    S = M @ jnp.array([1.0, 0.6, 0.0, 0.3])
    assert abs(float(S[1] ** 2 + S[3] ** 2) - (0.6 ** 2 + 0.3 ** 2)) < 1e-12
    # rotation mixes Q and U but conserves Q^2 + U^2
    R = faraday_rotation_matrix(0.7)
    S = R @ jnp.array([1.0, 0.6, 0.2, 0.0])
    assert abs(float(S[1] ** 2 + S[2] ** 2) - (0.6 ** 2 + 0.2 ** 2)) < 1e-12


def test_absorbed_polarisation_fraction_thin_and_thick_limits():
    """The moments-based polarised RT, checked at both dispatch boundaries.

    scripts/validate_kirchhoff.py checks the power-law fractions through the
    direct gamma-quadrature; here the *moment* path is pinned at its two limits
    (kirchhoff.absorbed_polarisation_fraction is the quantity the paper claims).
    The sign flip to -3/(6p+13) is a power-law result and is NOT expected for
    the narrow Gaussian used here.
    """
    from synchro.kirchhoff import (absorbed_polarisation_fraction,
                                   emissivity_from_moments,
                                   emissivity_Q_from_moments,
                                   absorption_from_moments,
                                   absorption_Q_from_moments)
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - 30.0) ** 2 / 3.0 ** 2)
    dg = g - 30.0
    M0 = np.trapezoid(N, g)
    M1 = np.trapezoid(N * dg, g)
    M2 = np.trapezoid(N * dg ** 2, g)
    ig = np.trapezoid(N / g, g)
    args = (100.0, 30.0, 1.0, M0, M1, M2)

    jF = float(emissivity_from_moments(*args))
    jG = float(emissivity_Q_from_moments(*args))
    aF = float(absorption_from_moments(*args, ig))
    aG = float(absorption_Q_from_moments(*args, ig))
    S_perp = (jF + jG) / (aF + aG)
    S_par = (jF - jG) / (aF - aG)

    # thin boundary (tau ~ 1e-20: the -expm1 branch); Pi -> jG/jF
    pi_thin = float(absorbed_polarisation_fraction(*args, 1e-14, ig))
    assert abs(pi_thin - jG / jF) < 1e-12

    # thick boundary; Pi -> (S_perp - S_par)/(S_perp + S_par)
    pi_thick = float(absorbed_polarisation_fraction(*args, 1e12, ig))
    assert abs(pi_thick - (S_perp - S_par) / (S_perp + S_par)) < 1e-12

    # self-absorption depolarises
    assert abs(pi_thick) < abs(pi_thin)
    assert abs(pi_thin) < 1.0


def test_absorbed_intensity_thin_limit_is_j_times_L():
    """Boundary test for the tau -> 0 branch of absorbed_intensity_from_moments.

    With the naive 1-exp(-tau) this returned exactly 0.0 for tau < 1e-16.
    """
    from synchro.kirchhoff import (absorbed_intensity_from_moments,
                                   emissivity_from_moments)
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - 30.0) ** 2 / 3.0 ** 2)
    dg = g - 30.0
    M0 = np.trapezoid(N, g)
    M1 = np.trapezoid(N * dg, g)
    M2 = np.trapezoid(N * dg ** 2, g)
    ig = np.trapezoid(N / g, g)
    j = float(emissivity_from_moments(100.0, 30.0, 1.0, M0, M1, M2))
    for L in (1e-20, 1e-14, 1e-8):
        I = float(absorbed_intensity_from_moments(100.0, 30.0, 1.0,
                                                  M0, M1, M2, L, ig))
        assert I > 0.0
        assert abs(I / (j * L) - 1.0) < 1e-9


def _gaussian_moment_set(gamma0=30.0, sigma=3.0):
    g = np.linspace(1.0, 200.0, 2000)
    N = np.exp(-0.5 * (g - gamma0) ** 2 / sigma ** 2)
    dg = g - gamma0
    return (np.trapezoid(N, g), np.trapezoid(N * dg, g),
            np.trapezoid(N * dg ** 2, g), np.trapezoid(N / g, g))


def test_moment_slab_polarised_absorption_matches_kirchhoff():
    """Wired alpha_Q: the slab Q/I equals the analytic Kirchhoff fraction.

    Boundary sweep over eight decades of optical depth, spanning the thin
    (Pi -> jG/jF) and thick (Pi -> (S_perp-S_par)/(S_perp+S_par)) limits.
    """
    from synchro.los_moments import moment_driven_slab
    from synchro.kirchhoff import absorbed_polarisation_fraction
    M0, M1, M2, ig = _gaussian_moment_set()
    args = (100.0, 30.0, 1.0, M0, M1, M2)
    for L in (1e-2, 1e2, 1e4, 1e6, 1e10):
        S = moment_driven_slab(*args, ig, L=L, n_slabs=256,
                               polarised_absorption=True)
        qi = float(S[1]) / float(S[0])
        pi = float(absorbed_polarisation_fraction(*args, L, ig))
        assert np.isfinite(qi)
        assert abs(qi - pi) / abs(pi) < 1e-10


def test_moment_slab_rejects_imposed_polarisation_with_kirchhoff_absorption():
    from synchro.los_moments import moment_driven_slab
    M0, M1, M2, ig = _gaussian_moment_set()
    import pytest
    with pytest.raises(ValueError):
        moment_driven_slab(100.0, 30.0, 1.0, M0, M1, M2, ig, L=1.0,
                           Q0=0.2, polarised_absorption=True)


def test_transfer_slab_bright_optically_thick_slab_is_finite():
    """Regression: the augmented expm used to overflow to NaN.

    When |eps| ds and |K| ds differ by many orders of magnitude the
    scaling-and-squaring in expm produced NaN; the source column is now
    normalised (transfer_slab).  The saturated intensity must be eps/aI.
    """
    from synchro.transfer import mueller_matrix, transfer_slab
    aI, eps_I = 3.5e-5, 6.27
    K = mueller_matrix(aI, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    eps = jnp.array([eps_I, 0.0, 0.0, 0.0])
    for ds in (1e2, 1e6, 1e8):        # tau = 3.5e-3 ... 3.5e3
        S = transfer_slab(jnp.zeros(4), eps, K, ds)
        assert bool(jnp.all(jnp.isfinite(S)))
    S_sat = transfer_slab(jnp.zeros(4), eps, K, 1e8)
    assert abs(float(S_sat[0]) / (eps_I / aI) - 1.0) < 1e-10
    # zero emissivity must still work (scale guard)
    S0 = transfer_slab(jnp.array([1.0, 0.0, 0.0, 0.0]), jnp.zeros(4), K, 1.0)
    assert bool(jnp.all(jnp.isfinite(S0)))


def test_magnus_orders_on_a_two_region_path():
    """Omega1 converges as psi^2 and Omega1+Omega2 as psi^3.

    Stronger than `test_magnus_convergence` (which only asserts e2 < e1).
    The input Stokes vector must be generic: the leading commutator
    [K_conv, K_rot] generates a U-V mixing that annihilates a pure-Q input,
    which hides the Omega2 gain and makes both orders look identical.
    """
    from synchro.transfer import mueller_matrix, transfer_los
    from synchro.magnus import magnus_S
    S0 = jnp.array([1.0, 0.6, 0.3, 0.2])
    ds = 1.0
    depths = np.logspace(-2.0, -0.7, 12)
    e1, e2 = [], []
    for a in depths:
        K_s = jnp.stack([
            mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a),   # rotation
            mueller_matrix(0.0, 0.0, 0.0, 0.0, 0.0, a, 0.0),   # conversion
        ])
        S_ex = transfer_los(S0, jnp.zeros((2, 4)), K_s, ds)
        e1.append(float(jnp.max(jnp.abs(S_ex - magnus_S(S0, K_s, ds, order=1)))))
        e2.append(float(jnp.max(jnp.abs(S_ex - magnus_S(S0, K_s, ds, order=2)))))
    s1 = np.polyfit(np.log(depths), np.log(e1), 1)[0]
    s2 = np.polyfit(np.log(depths), np.log(e2), 1)[0]
    assert abs(s1 - 2.0) < 0.15
    assert abs(s2 - 3.0) < 0.15
    assert e1[0] / e2[0] > 20.0   # order gain, not just a constant factor


def _gauss_hermite_ensemble(n, gamma0, alpha0, theta0, sig_g=0.0, sig_a=0.0,
                            n_nodes=21):
    """Exact Gaussian ensemble average of I_n by Gauss-Hermite quadrature.

    Deterministic (no Monte-Carlo noise), which is what makes the fourth-order
    convergence law measurable.  n_nodes=21 reaches +-7.85 sigma, so gamma stays
    above 1 for sigma_gamma/gamma0 < 0.12 -- a Gaussian in gamma is unphysical
    in its tails beyond that.
    """
    from synchro.stokes import stokes_harmonic
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    W = w / np.sum(w)
    vals = np.array([float(stokes_harmonic(n, gamma0 + sig_g * xi,
                                           alpha0 + sig_a * xi, theta0)[0])
                     for xi in x])
    return float(np.sum(W * vals))


def test_convergence_is_fourth_order_in_the_width():
    """Sec. 4.7: for a Gaussian the leading neglected term is 4th order.

    Pins the exponent quoted in the paper (q = 4.01).
    """
    from synchro.expansion import build_expansion
    g0, a0, t0 = 20.0, np.pi / 4, np.pi / 3
    ns = (1, 10, 50)
    fracs = np.array([0.01, 0.02, 0.04])
    exp = build_expansion(ns, g0, a0, t0)
    errs = []
    for f in fracs:
        sig = f * g0
        ap = np.asarray(exp(jnp.zeros(3),
                            jnp.diag(jnp.array([sig ** 2, 0.0, 0.0]))))[:, 0]
        ex = np.array([_gauss_hermite_ensemble(n, g0, a0, t0, sig_g=sig)
                       for n in ns])
        errs.append(np.abs(ap - ex) / np.abs(ex))
    errs = np.array(errs)
    for i in range(len(ns)):
        q = np.polyfit(np.log(fracs), np.log(errs[:, i]), 1)[0]
        assert abs(q - 4.0) < 0.15, f"n={ns[i]}: exponent {q}"


def test_pitch_angle_is_the_binding_direction():
    """Sec. 4.7: the pitch-angle width constrains the expansion, not the energy.

    The paper previously attributed a ~10% error at n=20 to a 6% energy spread;
    it is in fact the 0.1 rad pitch-angle spread. This pins the separation.
    """
    from synchro.expansion import build_expansion
    g0, a0, t0 = 5.0, np.pi / 4, np.pi / 3
    ns = (1, 20)
    exp = build_expansion(ns, g0, a0, t0)

    sig_g = 0.06 * g0
    ap_g = np.asarray(exp(jnp.zeros(3),
                          jnp.diag(jnp.array([sig_g ** 2, 0.0, 0.0]))))[:, 0]
    ex_g = np.array([_gauss_hermite_ensemble(n, g0, a0, t0, sig_g=sig_g)
                     for n in ns])
    err_g = np.abs(ap_g - ex_g) / np.abs(ex_g)

    sig_a = 0.1
    ap_a = np.asarray(exp(jnp.zeros(3),
                          jnp.diag(jnp.array([0.0, sig_a ** 2, 0.0]))))[:, 0]
    ex_a = np.array([_gauss_hermite_ensemble(n, g0, a0, t0, sig_a=sig_a)
                     for n in ns])
    err_a = np.abs(ap_a - ex_a) / np.abs(ex_a)

    assert err_g.max() < 1e-4          # 6% energy spread is harmless
    assert err_a[1] > 1e-2             # 0.1 rad pitch spread is not, at n=20
    assert err_a[1] / err_g[1] > 100   # the two directions differ by >2 decades


def test_mildly_relativistic_claims_of_section_5_4():
    """Pins the numbers quoted in Sec. 5.4 (isotropic pitch angles, theta=pi/3).

    These were previously stated for an unspecified pitch angle and did not
    reproduce; V_n/I_n at a single harmonic is fixed by geometry (full
    polarisation), so the meaningful statement is about the ensemble.
    """
    from synchro.stokes import stokes_harmonic
    th = np.pi / 3
    mus = np.linspace(-0.995, 0.995, 61)      # coarse but adequate for a pin
    out = {}
    for g, nmax in ((2.0, 200), (5.0, 400)):
        I = np.zeros(nmax - 1)
        V = np.zeros(nmax - 1)
        for mu in mus:
            al = float(np.arccos(mu))
            for k, n in enumerate(range(1, nmax)):
                i, _, v = [float(x) for x in stokes_harmonic(n, g, al, th)]
                I[k] += i
                V[k] += v
        tot = I.sum()
        c = np.cumsum(I)
        out[g] = (V.sum() / tot, I[0] / tot,
                  int(np.searchsorted(c, 0.9 * tot)) + 1)

    v2, f2, n90_2 = out[2.0]
    v5, f5, _ = out[5.0]
    assert abs(v2 - 0.46) < 0.05          # <V>/<I> at gamma=2
    assert abs(v5 - 0.20) < 0.03          # matches the gamma=5 value of Sec. 5.3
    assert abs(2.0 * v2 - 1.0) < 0.15     # the 1/gamma law, unit coefficient
    assert abs(5.0 * v5 - 1.0) < 0.15
    assert abs(f2 - 0.046) < 0.01         # fundamental carries ~4.6% at gamma=2
    assert 15 <= n90_2 <= 30              # 90% of the power below n ~ 22
