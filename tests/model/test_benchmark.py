"""Manuscript Section 5.3.1 benchmark (``fig: full channel response``) with the package.

Configuration (main.tex ``eq: channel toy population``, FINAL_DESIGN Section 11):
``gamma0 = 20``, ``B0 = 1`` G, three bump channels at ``y_j = 2, 4, 8`` with
65 % half-widths, ``HarmonicKernel(m_max=40)`` (``required_m_max`` is 40, so
the harmonic-truncation term is a zero bound), ``Truncation(L, L, N)``,
``predict`` with amplitude 1, units converted by ``e^2 Omega_0^2 / (2 pi c)``.
The pins are the saved ``finite_IQUV`` / ``direct_IQUV`` rows of
``validation/full_response_results.json`` (moments on 16 x 16 latent and
256 x 256 angular nodes; basis on 128 product nodes with a 5e-4 five-point
step; direct reference on the same population grid).

Two variants:

* default (fast, about 60 s): ``L in {2, 4}``, the kernel's default 64 x 64
  product nodes, moments on 8 x 8 latent and 64 x 64 angular nodes, direct
  average on 4 x 4 x 64 x 64. Tolerances follow the manuscript's refinement
  table and the measurements recorded here: the moment quadrature changed by
  7.7e-15 between (160, 12) and (256, 16) (8 x 8 x 64 x 64 already agrees
  with the saved moments to 1e-14), the basis grid by 4.3e-8 (96 to 128
  nodes) and the derivative step by 8.3e-9; with 64 product nodes the finite
  response differs from the saved table by 1.9e-6 of channel ``I``
  (tolerance 5e-6, inside the 1e-5 design pin). The direct reference changed
  by 6.4e-7 between (160, 12) and (256, 16); the 64-node angular grid was
  measured at 2.4e-4 (``w = 0.5``) and 9.2e-5 (``w = 1``) here, so its
  tolerance is 5e-4.
* ``-m slow`` or ``SYNCMOMENTS_RUN_SLOW=1`` (marked ``slow``, several minutes): ``L in {2, 4, 8}``
  with the manuscript's 128 x 128 product nodes, moments on 16 x 16 x 256 x 256
  (``SYNCMOMENTS_BENCH_ANGULAR`` overrides the angular count, e.g. 160 when memory
  is short), finite response within 1e-8 (measured 9e-15 at ``N = 0``, 1e-12
  at ``N = 1`` and 5.5e-10 at ``N = 2``, the AD-versus-five-point-difference
  residual), direct average on 8 x 8 x 96 x 96 within 2e-5 (measured 6.8e-6),
  the mixed-term deletion change at ``N = 2``, ``L = 8`` within 5 % of the
  saved 1.02e-3 and 4.09e-3, and the angular trend.

Every discrepancy is recorded with ``record_property``. These are finite
comparisons against one saved population; they certify neither a global
remainder nor the Galactic harmonic regime (manuscript Section 5.3.1).
"""

import os

import numpy as np
import pytest

from syncmoments.model.basis import build_basis
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import required_m_max
from syncmoments.model.predict import direct_channel_average, predict

import _benchmark_helpers as H
from _harmonic_oracles import BENCH_SUPPORT, benchmark_channels

# ``slow`` tests run with ``-m slow`` or ``SYNCMOMENTS_RUN_SLOW=1`` (tests/conftest.py)
slow = pytest.mark.slow
FULL_ANGULAR = int(os.environ.get("SYNCMOMENTS_BENCH_ANGULAR", "256"))

FINITE_TOL = 1e-5  # |finite - saved finite| / saved channel I (the design pin)
FAST_FINITE_TOL = 5e-6  # 64 product nodes vs the manuscript's 128 (measured 1.9e-6)
FULL_FINITE_TOL = 1e-8  # 128 product nodes: AD vs five-point FD (measured 5.5e-10)
FAST_DIRECT_TOL = 5e-4  # 4x4x64x64 direct average (measured 2.4e-4 and 9.2e-5)
FULL_DIRECT_TOL = 2e-5  # 8x8x96x96 direct average (measured 6.8e-6)
MIXED_TERM_MIN = 1e-3  # manuscript: 1.0e-3 I and 4.1e-3 I at N=2, L=8
FAST_NODES = 64  # HarmonicKernel default product nodes (n_outer = n_inner)
FULL_NODES = 128  # the manuscript's finest basis grid
WIDTHS = H.WIDTHS
DEGREES = (0, 1, 2)


def kernel(nodes=FAST_NODES):
    return HarmonicKernel(40, n_outer=nodes, n_inner=nodes)


def build(L, N, nodes=FAST_NODES):
    return build_basis(
        kernel(nodes),
        benchmark_channels(),
        Truncation(L, L, N),
        H.reference(),
        support=BENCH_SUPPORT,
        convergence=False,
    )


def finite(basis, moments):
    """Package prediction in manuscript units, ``(n_ch, 4)``."""
    pred = predict(basis, H.restrict_moments(moments, basis.index), amplitude=1.0)
    assert pred.budget.harmonic_truncation.kind == "bound"
    assert float(np.max(np.abs(np.asarray(pred.budget.harmonic_truncation.value)))) == 0
    return H.to_manuscript_units(pred.stokes)


def direct(width, latent, angular):
    samples = H.full_population(width, latent, angular)
    pred = direct_channel_average(
        samples,
        kernel(),
        benchmark_channels(),
        amplitude=1.0,
        reference=H.reference(),
        support=BENCH_SUPPORT,
        batch_size=1024,
    )
    assert pred.budget.harmonic_truncation.kind == "bound"
    return H.to_manuscript_units(pred.stokes)


# -- fixtures --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def results():
    return H.load_results()


@pytest.fixture(scope="module")
def fast_bases():
    return {(L, N): build(L, N) for L in (2, 4) for N in DEGREES}


@pytest.fixture(scope="module")
def fast_moments():
    index = MomentIndex.build(Truncation(4, 4, 2))
    return {w: H.toy_moments(w, 8, 64, index) for w in WIDTHS}


@pytest.fixture(scope="module")
def full_bases():
    return {(L, N): build(L, N, FULL_NODES) for L in (2, 4, 8) for N in DEGREES}


@pytest.fixture(scope="module")
def full_moments():
    index = MomentIndex.build(Truncation(8, 8, 2))
    return {w: H.toy_moments(w, 16, FULL_ANGULAR, index) for w in WIDTHS}


# -- the saved table itself ----------------------------------------------------------------


def test_saved_table_is_the_manuscript_configuration(results):
    config = results["configuration"]
    assert config["gamma0"] == 20.0 and config["zeta0"] == 4.0
    assert config["channel_centres"] == [2.0, 4.0, 8.0]
    assert config["channel_half_widths"] == [1.3, 2.6, 5.2]
    assert config["nmax"] == 40 and config["quick"] is False
    assert config["grids"] == [[160, 12], [256, 16]]
    assert config["basis_product_grids"] == [96, 128]
    assert len(results["rows"]) == 18
    # Numbers quoted in main.tex Section 5.3.1.
    assert (
        abs(
            H.saved_row(results, 0.5, 2, 8)["max_abs_stokes_error_over_channel_I"]
            - 1.64e-4
        )
        < 1e-6
    )
    assert (
        abs(
            H.saved_row(results, 1.0, 2, 8)["max_abs_stokes_error_over_channel_I"]
            - 1.75e-3
        )
        < 1e-5
    )
    for width in WIDTHS:
        row = H.saved_row(results, width, 2, 8)
        assert row["angular_only_error_over_channel_I"] < 2.7e-6
        assert (
            H.saved_row(results, width, 0, 2)["angular_only_error_over_channel_I"]
            > 6.8e-2
        )
        stability = H.saved_stability(results, width)
        assert stability["reference_refinement"][-1] < 6.5e-7
        assert stability["moment_quadrature_change"] < 1e-14
        assert stability["derivative_angular_grid_change"] < 4.4e-8
        assert stability["derivative_step_change"] < 8.3e-9
        assert stability["mixed_term_deletion_change"] > MIXED_TERM_MIN


def test_no_harmonic_tail_in_this_example():
    """``m < 39.6`` on the support (manuscript), so ``m_max = 40`` is complete."""
    assert required_m_max(BENCH_SUPPORT, benchmark_channels()) == 40
    # The population reaches gamma = 24 and B/B0 = 0.8 at most (w = 1).
    assert 20.0 * 1.2 <= BENCH_SUPPORT.gamma_max()
    assert 0.8 >= BENCH_SUPPORT.B_min()


# -- fast variant (default run) --------------------------------------------------------------


@pytest.mark.parametrize("N", DEGREES)
@pytest.mark.parametrize("L", (2, 4))
@pytest.mark.parametrize("width", WIDTHS)
def test_fast_finite_response_matches_saved_table(
    width, L, N, fast_bases, fast_moments, results, record_property
):
    row = H.saved_row(results, width, N, L)
    got = finite(fast_bases[L, N], fast_moments[width])
    saved_finite = H.saved_stokes(row, "finite_IQUV")
    saved_direct = H.saved_stokes(row, "direct_IQUV")
    discrepancy = H.error_over_channel_I(got, saved_finite)
    error = H.error_over_channel_I(got, saved_direct)
    record_property("finite_vs_saved_over_I", discrepancy)
    record_property("finite_vs_direct_over_I", error)
    assert discrepancy < FAST_FINITE_TOL < FINITE_TOL
    assert abs(error - row["max_abs_stokes_error_over_channel_I"]) < FAST_FINITE_TOL


@pytest.mark.parametrize("width", WIDTHS)
def test_fast_direct_average_matches_saved_reference(width, results, record_property):
    got = direct(width, 4, 64)
    saved = H.saved_stokes(H.saved_row(results, width, 2, 8), "direct_IQUV")
    discrepancy = H.error_over_channel_I(got, saved)
    record_property("direct_vs_saved_over_I", discrepancy)
    assert discrepancy < FAST_DIRECT_TOL
    # The saved reference itself moved by < 6.5e-7 under its last refinement.
    assert discrepancy > H.saved_stability(results, width)["reference_refinement"][-1]


@pytest.mark.parametrize("width", WIDTHS)
def test_fast_error_trend_in_L_and_N(width, fast_bases, fast_moments, results):
    """``L = 4`` beats ``L = 2`` at every ``N``; ``N`` helps only once ``L = 4``."""
    error = {}
    for (L, N), basis in fast_bases.items():
        saved_direct = H.saved_stokes(H.saved_row(results, width, N, L), "direct_IQUV")
        error[L, N] = H.error_over_channel_I(
            finite(basis, fast_moments[width]), saved_direct
        )
    for N in DEGREES:
        assert error[4, N] < error[2, N]
    assert error[4, 2] < error[4, 0]
    assert (
        error[2, 2] > 0.5 * error[2, 0]
    )  # L = 2 leaves the angular structure unresolved


@pytest.mark.parametrize("width", WIDTHS)
def test_fast_mixed_terms_matter(
    width, fast_bases, fast_moments, results, record_property
):
    basis = fast_bases[4, 2]
    moments = H.restrict_moments(fast_moments[width], basis.index)
    full = H.to_manuscript_units(predict(basis, moments, amplitude=1.0).stokes)
    dropped = H.to_manuscript_units(
        predict(basis, H.drop_mixed(moments), amplitude=1.0).stokes
    )
    saved_direct = H.saved_stokes(H.saved_row(results, width, 2, 4), "direct_IQUV")
    change = H.error_over_channel_I(dropped, full) * full[0, 0] / saved_direct[0, 0]
    record_property("mixed_term_deletion_change_L4", change)
    assert np.count_nonzero(H.mixed_row_mask(basis.index)) > 0
    assert (
        change > 0.5 * MIXED_TERM_MIN
    )  # L = 4 carries most of the L = 8 value (1.0e-3, 4.1e-3)


def test_budget_and_provenance_of_the_benchmark(fast_bases, fast_moments):
    basis = fast_bases[2, 0]
    pred = predict(
        basis, H.restrict_moments(fast_moments[1.0], basis.index), amplitude=1.0
    )
    budget = pred.budget
    assert budget.harmonic_truncation.kind == "bound"
    assert budget.basis_remainder.kind == "unbounded"  # no RemainderInputs supplied
    assert (
        budget.physical_kernel.kind == "unbounded"
    )  # vacuum helical orbit not declared
    assert budget.total().kind == "unbounded"
    d = pred.to_dict()
    assert d["budget"]["total"]["value"] is None
    assert "harmonic" in str(pred.provenance.kernel)
    assert dict(pred.provenance.truncation)["L_mu"] == 2
    assert dict(pred.provenance.truncation)["N"] == 0


# -- full variant (SYNCMOMENTS_RUN_SLOW=1) ------------------------------------------------------


@slow
@pytest.mark.parametrize("N", DEGREES)
@pytest.mark.parametrize("L", (2, 4, 8))
@pytest.mark.parametrize("width", WIDTHS)
def test_full_finite_response_matches_saved_table(
    width, L, N, full_bases, full_moments, results, record_property
):
    row = H.saved_row(results, width, N, L)
    got = finite(full_bases[L, N], full_moments[width])
    discrepancy = H.error_over_channel_I(got, H.saved_stokes(row, "finite_IQUV"))
    error = H.error_over_channel_I(got, H.saved_stokes(row, "direct_IQUV"))
    record_property("finite_vs_saved_over_I", discrepancy)
    record_property("finite_vs_direct_over_I", error)
    assert discrepancy < FULL_FINITE_TOL < FINITE_TOL
    assert abs(error - row["max_abs_stokes_error_over_channel_I"]) < FULL_FINITE_TOL


@slow
@pytest.mark.parametrize("width", WIDTHS)
def test_full_direct_average_matches_saved_reference(width, results, record_property):
    got = direct(width, 8, 96)
    saved = H.saved_stokes(H.saved_row(results, width, 2, 8), "direct_IQUV")
    discrepancy = H.error_over_channel_I(got, saved)
    record_property("direct_vs_saved_over_I", discrepancy)
    assert discrepancy < FULL_DIRECT_TOL


@slow
@pytest.mark.parametrize("width", WIDTHS)
def test_full_mixed_term_deletion_change(
    width, full_bases, full_moments, results, record_property
):
    basis = full_bases[8, 2]
    moments = H.restrict_moments(full_moments[width], basis.index)
    full = H.to_manuscript_units(predict(basis, moments, amplitude=1.0).stokes)
    dropped = H.to_manuscript_units(
        predict(basis, H.drop_mixed(moments), amplitude=1.0).stokes
    )
    saved_direct = H.saved_stokes(H.saved_row(results, width, 2, 8), "direct_IQUV")
    change = float(np.max(np.abs(dropped - full) / saved_direct[:, [0]]))
    saved_change = H.saved_stability(results, width)["mixed_term_deletion_change"]
    record_property("mixed_term_deletion_change", change)
    assert change > MIXED_TERM_MIN
    assert abs(change - saved_change) < 0.05 * saved_change


@slow
@pytest.mark.parametrize("width", WIDTHS)
def test_full_angular_trend(width, full_bases, full_moments, results):
    """Angular resolution dominates: ``L = 8`` beats ``L = 2`` at every ``N``."""
    error = {}
    for (L, N), basis in full_bases.items():
        saved_direct = H.saved_stokes(H.saved_row(results, width, N, L), "direct_IQUV")
        error[L, N] = H.error_over_channel_I(
            finite(basis, full_moments[width]), saved_direct
        )
    for N in DEGREES:
        assert error[8, N] < error[4, N] < error[2, N]
    assert error[8, 2] < 0.1 * error[8, 0]
    # The saved angular-only column (exact q dependence, angular truncation only).
    angular_only = {
        L: H.saved_row(results, width, 2, L)["angular_only_error_over_channel_I"]
        for L in (2, 4, 8)
    }
    assert angular_only[8] < angular_only[4] < angular_only[2]
    assert error[8, 2] > angular_only[8]  # Taylor truncation is what remains at L = 8
