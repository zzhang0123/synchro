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


# --- radiative transfer ----------------------------------------------------

def test_faraday_constant():
    _, rel = validate_transfer.test_rm_constant()
    assert rel < 1e-3  # 0.8119 vs 0.812


def test_transfer_slab_limits():
    thin, rot, conv = validate_transfer.test_transfer_limits()
    assert thin < 1e-4
    assert rot < 1e-12
    assert conv < 1e-12


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


# --- Magnus / conversion / cross-cumulants ---------------------------------

def test_magnus_convergence():
    e1, e2 = validate_rt_moments.test_magnus()
    assert e2 < e1          # order-2 beats order-1
    assert e2 < 1e-4


def test_conversion_scaling():
    _, _, _, rs = validate_rt_moments.test_conversion()
    assert abs(rs - 0.5) < 1e-3  # rho_Q ~ nu^-3 vs rho_V ~ nu^-2


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


# --- design (jit / autodiff) ----------------------------------------------

def test_design_moment_expansion_jit():
    test_design.test_moment_expansion_jit()


def test_design_moment_expansion_grad():
    test_design.test_moment_expansion_grad()


def test_design_transfer_los_jit():
    test_design.test_transfer_los_jit()
