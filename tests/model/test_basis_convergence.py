"""Tests of the jitted ``_build_core`` (JIT, ``jacfwd`` with respect to the
reference), provenance JSON round trips and the two-quadrature ``numerical``
estimate of ``basis_convergence``."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
from syncmoments.model.basis import (
    _build_core,
    basis_convergence,
    build_basis,
)
from syncmoments.model.channels import Channels
from syncmoments.model.errors import Provenance
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import Reference, Support
from syncmoments.model.phase import TaylorPhase

from _basis_fixtures import (
    polynomial_setup,
    small_harmonic,
)
from _harmonic_oracles import (
    B0,
    GAMMA0,
    S_DEPTH,
)

# -- JIT and autodiff of the core ------------------------------------------------------


def test_build_core_jit_and_jacfwd_wrt_gamma0():
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    truncation = Truncation(2, 2, 2)
    index = MomentIndex.build(truncation)
    phase = TaylorPhase(2)

    def core(gamma0):
        ref = Reference(gamma0, 2.0, depth_ref=1.5, scales=(2.0, 0.2, 0.5))
        return _build_core(kernel, channels, ref, phase, index=index, quadrature=None)

    eager = core(20.0)
    jitted = jax.jit(core)(20.0)
    for a, b in zip(eager, jitted):
        assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13)
    jac = jax.jacfwd(lambda g: core(g)[0])(20.0)
    # d/dgamma0 of the coefficient of z^r at fixed scale: shifting the reference by
    # dgamma0 re-expands the polynomial: c_r(gamma0) = sum_{p>=r} c_p C(p,r) (dz)^{p-r},
    # so dc_r/dgamma0 = (r+1) c_{r+1} / s_gamma.
    jac = np.asarray(jac)
    for pos, (l, k, r, s, b) in enumerate(index.h0):
        even = (l + k) % 2 == 0
        expected = (r + 1) * coeff[0, :, l, k, r + 1, s] / 2.0 if r < 2 else 0.0
        assert_allclose(
            jac[:, pos], np.asarray(expected) * even, rtol=1e-11, atol=1e-13
        )
    assert np.all(np.isfinite(jac))


def test_harmonic_core_jacfwd_wrt_gamma0_is_finite_and_matches_fd():
    kernel, channels, reference, support = small_harmonic()
    index = MomentIndex.build(Truncation(1, 1, 1))
    phase = TaylorPhase(1)

    def core(gamma0):
        ref = Reference(
            gamma0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
        )
        return jnp.concatenate(
            [
                a.ravel()
                for a in _build_core(
                    kernel, channels, ref, phase, index=index, quadrature=None
                )[:2]
            ]
        )

    jac = np.asarray(jax.jit(jax.jacfwd(core))(GAMMA0))
    h = 2e-2
    d1 = (np.asarray(core(GAMMA0 + h)) - np.asarray(core(GAMMA0 - h))) / (2 * h)
    d2 = (np.asarray(core(GAMMA0 + h / 2)) - np.asarray(core(GAMMA0 - h / 2))) / h
    fd = (4 * d2 - d1) / 3
    assert np.all(np.isfinite(jac))
    assert np.max(np.abs(jac - fd)) < 1e-6 * np.max(np.abs(fd))


# -- provenance and convergence --------------------------------------------------------


def test_provenance_json_round_trip():
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1),
        reference,
        support=support,
        convergence=False,
    )
    prov = basis.provenance
    assert isinstance(prov, Provenance)
    d = prov.to_dict()
    text = json.dumps(d)
    back = json.loads(text)
    assert back["kernel"]["name"] == "harmonic" and back["kernel"]["m_max"] == 10
    assert back["channels"]["family"] == "bump"
    assert back["truncation"] == {
        "L_mu": 1,
        "L_eta": 1,
        "N": 1,
        "depth_degree": None,
        "max_orders": None,
        "parity": True,
        "components": ["I", "Q", "V"],
        "n0": 12,
        "n2": 8,
        "n_real": 28,
    }
    assert back["reference"]["gamma0"] == 20.0
    assert back["support"]["truncated"] is None
    assert (
        back["phase_route"]["route"] == "taylor" and back["phase_route"]["degree"] == 1
    )
    assert back["quadrature"]["route"] == "product"
    assert set(back["numerics"]) >= {"n_nodes", "m_range", "width_ratio_min", "cells"}
    assert back["numerics"]["n_nodes"] == 128
    assert back["certified_orders"] == [0, 1, 2, 3]
    assert back["assumptions"] == []
    hash(prov)


@pytest.mark.parametrize("convergence", ["angular", "full"])
def test_numerical_term_is_an_estimate_with_a_value(convergence):
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1),
        reference,
        support=support,
        convergence=convergence,
    )
    term = basis.kernel_terms.numerical
    # R20: a per-column envelope |C_2x - C| (4 n_ch, n_real) of the primary route,
    # contracted with |m| by predict (was a unit-moment (n_ch, 4) contraction).
    assert term.kind == "estimate" and term.value.shape == (4, basis.index.n_real)
    value = np.asarray(term.value)
    assert np.all(np.isfinite(value)) and np.all(value >= 0) and np.max(value) > 0
    assert term.manuscript_term == "E_num"
    check = dict(basis.provenance.quadrature)
    # NEW-9: the product-vs-tensor route check runs only with "full" (opt-in).
    expected_check = "tensor" if convergence == "full" else None
    assert check["route"] == "product" and check["check_route"] == expected_check
    assert check["numerical_route"].startswith("product route at 2x angular nodes")
    assert check["convergence"] == convergence
    assert basis.provenance.finite_checks
    again = basis_convergence(basis, kernel, factor=2, full=(convergence == "full"))
    assert_allclose(np.asarray(again.value), value, rtol=1e-12)
    # Column by column the refinement changes C by less than C itself.
    C = np.abs(np.asarray(basis.response_matrix()))
    assert np.all(np.max(value, axis=0) <= np.max(C, axis=0))


def test_tensor_route_basis_checks_against_product():
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 0),
        reference,
        support=support,
        quadrature="tensor",
        convergence="angular",
        cross_route=True,
    )
    check = dict(basis.provenance.quadrature)
    assert check["route"] == "tensor" and check["check_route"] == "product"
    assert basis.kernel_terms.numerical.kind == "estimate"


def test_continuum_convergence_doubles_channel_nodes():
    kernel = ContinuumKernel()
    channels = Channels.bump([1e9], [2e8], n_nu=24)
    reference = Reference(1e3, 1e-5, scales=(1e3, 1e-5, 1.0))
    support = Support(gamma=(5e2, 5e3), B=(5e-6, 2e-5), depth=(0.0, 1.0))
    basis = build_basis(
        kernel, channels, Truncation(0, 1, 1), reference, support=support
    )
    term = basis.kernel_terms.numerical
    assert term.kind == "estimate" and term.value.shape == (4, basis.index.n_real)
    value = np.asarray(term.value).reshape(1, 4, -1)
    assert np.all(np.isfinite(value)) and np.all(value[:, 3] == 0)
    assert "n_nu" in term.note and "tail" in term.note
    numerics = dict(basis.provenance.numerics)
    assert numerics["m_range"] is None and numerics["cells"] == 0
    assert basis.channels.n_nu == 24  # the basis keeps the caller's channels


def test_polynomial_kernel_has_no_quadrature_to_check():
    kernel, coeff, line_nu, reference, channels, support = polynomial_setup()
    basis = build_basis(
        kernel, channels, Truncation(1, 1, 1), reference, support=support
    )
    assert basis.kernel_terms.numerical.kind == "not_applicable"
