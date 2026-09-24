"""Tests of ``synchro.model.basis.build_basis`` validation: refusals on both
sides of ``required_m_max`` and the smoothness threshold, the continuum kernel's
closure, phase-degree checks and screen terms, input validation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.constants import C_SI_M
from synchro.model.basis import (
    SpectralBasis,
    build_basis,
)
from synchro.model.channels import Channels
from synchro.model.errors import ErrorTerm
from synchro.model.harmonic import HarmonicKernel
from synchro.model.index import MomentIndex, Truncation
from synchro.model.kernels import ContinuumKernel
from synchro.model.moments import Reference, Support
from synchro.model.phase import CumulantScreen, GaussianScreen, TaylorPhase

from _basis_fixtures import (
    SMALL_SUPPORT,
    polynomial_setup,
)
from _harmonic_oracles import (
    B0,
    BENCH_SUPPORT,
    GAMMA0,
    NU_STAR,
    S_DEPTH,
    benchmark_channels,
)

# -- refusals on both sides of every threshold -----------------------------------------


def test_required_m_max_refusal_both_sides():
    channels = benchmark_channels()
    reference = Reference(GAMMA0, B0, scales=(GAMMA0, B0, S_DEPTH))
    truncation = Truncation(0, 0, 0)
    common = dict(support=BENCH_SUPPORT, convergence=False)
    with pytest.raises(ValueError, match="ContinuumKernel"):
        build_basis(
            HarmonicKernel(39, n_outer=8, n_inner=8),
            channels,
            truncation,
            reference,
            **common,
        )
    ok = build_basis(
        HarmonicKernel(40, n_outer=8, n_inner=8),
        channels,
        truncation,
        reference,
        **common,
    )
    assert ok.kernel_terms.harmonic_truncation.kind == "bound"
    truncated = build_basis(
        HarmonicKernel(39, n_outer=8, n_inner=8),
        channels,
        truncation,
        reference,
        allow_truncated=True,
        **common,
    )
    assert truncated.kernel_terms.harmonic_truncation.kind == "unbounded"
    probed = build_basis(
        HarmonicKernel(39, n_outer=8, n_inner=8, tail_probe=2),
        channels,
        truncation,
        reference,
        allow_truncated=True,
        **common,
    )
    assert probed.kernel_terms.harmonic_truncation.kind == "estimate"
    # Galactic regime: refused even with allow_truncated (names the continuum kernel).
    galactic = Support(gamma=(1e3, 1e4), B=(5e-6, 1e-5), depth=(0.0, 1.0))
    ghz = Channels.bump([1e9], [1e8])
    with pytest.raises(ValueError, match="ContinuumKernel"):
        build_basis(
            HarmonicKernel(300, n_outer=8, n_inner=8),
            ghz,
            truncation,
            Reference(5e3, 7e-6),
            support=galactic,
            allow_truncated=True,
            convergence=False,
        )


def test_smoothness_refusal_both_sides():
    reference = Reference(GAMMA0, B0, scales=(GAMMA0, B0, S_DEPTH))
    kernel = HarmonicKernel(10, n_outer=8, n_inner=8)
    support = SMALL_SUPPORT
    raised = Channels.raised_cosine([2.0 * NU_STAR], [0.5 * 2.0 * NU_STAR])
    assert raised.smoothness == 1
    ok = build_basis(
        kernel,
        raised,
        Truncation(0, 0, 0),
        reference,
        support=support,
        convergence=False,
    )
    assert not any("smoothness" in note for note in ok.provenance.notes)
    with pytest.raises(ValueError, match="smoothness"):
        build_basis(
            kernel,
            raised,
            Truncation(0, 0, 1),
            reference,
            support=support,
            convergence=False,
        )
    allowed = build_basis(
        kernel,
        raised,
        Truncation(0, 0, 1),
        reference,
        support=support,
        convergence=False,
        allow_nonsmooth=True,
    )
    assert any("smoothness" in note for note in allowed.provenance.notes)
    # The continuum kernel integrates the response; no smoothness is needed.
    tophat = Channels.tophat([1e9], [1e8])
    continuum = build_basis(
        ContinuumKernel(),
        tophat,
        Truncation(0, 2, 1),
        Reference(1e3, 1e-5),
        support=Support(gamma=(5e2, 5e3), B=(5e-6, 2e-5), depth=(0.0, 1.0)),
        convergence=False,
    )
    assert continuum.kernel_terms.harmonic_truncation.kind == "not_applicable"


def test_continuum_kernel_requires_isotropic_pitch_and_drops_v_rows():
    kernel = ContinuumKernel()
    channels = Channels.bump([1e9, 2e9], [2e8, 4e8])
    reference = Reference(1e3, 1e-5, scales=(1e3, 1e-5, 1.0))
    support = Support(gamma=(5e2, 5e3), B=(5e-6, 2e-5), depth=(0.0, 1.0))
    with pytest.raises(ValueError, match="uniform_mu"):
        build_basis(kernel, channels, Truncation(1, 2, 1), reference, support=support)
    basis = build_basis(
        kernel,
        channels,
        Truncation(0, 2, 1),
        reference,
        support=support,
        convergence=False,
    )
    index = basis.index
    assert index.components == ("I", "Q")
    assert all((l + k) % 2 == 0 for l, k, *_ in index.h0)
    assert np.all(np.asarray(basis.V_basis) == 0)
    assert np.max(np.abs(np.asarray(basis.I_basis))) > 0
    names = [a.name for a in basis.provenance.assumptions]
    assert "isotropic_pitch" in names
    assert any("V not modelled" in note for note in basis.provenance.notes)
    assert basis.kernel_terms.physical_kernel.kind == "unbounded"
    C = basis.response_matrix()
    assert C.shape == (8, index.n_real)


def test_phase_degree_validation_and_screen_terms():
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    common = dict(support=support, convergence=False)
    with pytest.raises(ValueError, match="max_b"):
        build_basis(
            kernel,
            channels,
            Truncation(1, 1, 2),
            reference,
            phase=TaylorPhase(1),
            **common,
        )
    with pytest.raises(ValueError, match="max_b"):
        build_basis(
            kernel,
            channels,
            Truncation(1, 1, 1),
            reference,
            phase=GaussianScreen(1.0, 0.3),
            **common,
        )
    gaussian = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1, depth_degree=0),
        reference,
        phase=GaussianScreen(1.0, 0.3),
        **common,
    )
    assert gaussian.kernel_terms.screen_exponent.kind == "bound"
    assert np.all(np.asarray(gaussian.kernel_terms.screen_exponent.value) == 0)
    names = [a.name for a in gaussian.provenance.assumptions]
    assert names == ["independent_screen", "gaussian_screen"]
    tau = 2 * (C_SI_M / line_nu) ** 2
    expected = np.exp(1j * tau * 1.0 - 0.5 * (tau * 0.3) ** 2)
    pos = gaussian.index.position(2, 0, 0, 0, 0, 0) - gaussian.index.n0  # h2 row
    assert_allclose(
        np.asarray(gaussian.P_basis[:, pos]), coeff[1, :, 0, 0, 0, 0] * expected
    )
    kappa = jnp.array([1.0, 0.09, 0.0, 0.0])
    cumulant = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1, depth_degree=0),
        reference,
        phase=CumulantScreen(kappa),
        **common,
    )
    assert cumulant.kernel_terms.screen_exponent.kind == "unbounded"
    bounded = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1, depth_degree=0),
        reference,
        phase=CumulantScreen(kappa, g5_bound=1e-3),
        **common,
    )
    term = bounded.kernel_terms.screen_exponent
    assert term.kind == "estimate" and term.value.shape == (2, 4)
    value = np.asarray(term.value)
    assert np.all(value[:, [0, 3]] == 0) and np.all(value[:, [1, 2]] > 0)


def test_input_validation():
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    truncation = Truncation(1, 1, 1)
    with pytest.raises(ValueError, match="Truncation"):
        build_basis(
            kernel, channels, MomentIndex.build(truncation), reference, support=support
        )
    with pytest.raises(ValueError, match="Support"):
        build_basis(kernel, channels, truncation, reference, support=None)
    with pytest.raises(ValueError, match="Reference"):
        build_basis(
            kernel, channels, truncation, SimpleNamespace(gamma0=20.0), support=support
        )
    with pytest.raises(ValueError, match="convergence"):
        build_basis(
            kernel,
            channels,
            truncation,
            reference,
            support=support,
            convergence="maybe",
        )
    with pytest.raises(ValueError, match="E_phys"):
        build_basis(
            kernel, channels, truncation, reference, support=support, E_phys=0.0
        )
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda g: build_basis(
                kernel, channels, truncation, Reference(g, 2.0), support=support
            ).I_basis
        )(20.0)
    outside = Reference(20.0, 2.0, scales=(2.0, 0.2, 0.5))
    narrow = Support(gamma=(21.0, 30.0), B=(1.0, 3.0), depth=(0.0, 3.0))
    basis = build_basis(
        kernel, channels, truncation, outside, support=narrow, convergence=False
    )
    assert any("outside" in note for note in basis.provenance.notes)
    declared = ErrorTerm(jnp.zeros((2, 4)), "bound", "declared")
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, E_phys=declared
    )
    assert basis.kernel_terms.physical_kernel is declared
    with pytest.raises(ValueError, match="shape"):
        SpectralBasis(
            index=basis.index,
            truncation=truncation,
            channels=channels,
            reference=reference,
            support=support,
            I_basis=jnp.zeros((3, basis.index.n0)),
            V_basis=basis.V_basis,
            P_basis=basis.P_basis,
            kernel_terms=basis.kernel_terms,
            provenance=basis.provenance,
        )
