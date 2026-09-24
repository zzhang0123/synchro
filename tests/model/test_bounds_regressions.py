"""Regressions of the review fixes in ``synchro.model.bounds`` (R04, R06, R07, R14, R30).

R04: a line-kernel basis with channel smoothness below ``N + 1`` has an
unbounded ``basis_remainder``. R06: the screen factorisation bound with
per-line inputs covers a channel holding several lines. R07: the modulus of
the phased ``b = 0`` channel coefficient gives an ``estimate``; the supplied
absolute line sum gives a ``bound`` that covers a two-line channel. R14: a
screen phase route has no depth-Taylor term. R30: the derivative probe
refuses an order outside ``basis.certified_orders``.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import synchro  # noqa: F401
from synchro.model._kernel_helpers import phase_coordinate, phase_weights
from synchro.model._polynomial import PolynomialTestKernel
from synchro.model.basis import build_basis
from synchro.model.bounds import RemainderInputs, screen_factorisation_bound
from synchro.model.channels import Channels
from synchro.model.index import Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support
from synchro.model.phase import GaussianScreen, TaylorPhase
from synchro.model.predict import direct_channel_average, predict

from _basis_stub import build_stub_basis
from _predict_helpers import (
    B0,
    DEPTH_REF,
    GAMMA0,
    LINE_NU,
    S_DEPTH,
    harmonic_setup,
    nine_atoms,
    polynomial_kernel,
    reference,
    samples_of,
    support,
)


def _zero_inputs(basis, n_ch, **extra):
    n_lk = basis.index.n_lk
    return RemainderInputs(
        rho_ang=np.zeros((n_ch, 3)),
        H=np.zeros((n_ch, 3, n_lk)),
        absolute_moments=np.zeros((2, n_lk)),
        kind="bound",
        **extra,
    )


def _polarised(stokes):
    stokes = np.asarray(stokes)
    return stokes[:, 1] + 1j * stokes[:, 2]


# -- R04: nonsmooth channels of a line kernel ---------------------------------------


def test_nonsmooth_line_basis_remainder_is_unbounded():
    kernel, channels, supp, ref, _ = harmonic_setup()
    tophat = Channels.tophat(channels.centres_hz, channels.widths_hz)
    basis = build_basis(
        kernel,
        tophat,
        Truncation(1, 1, 1),
        ref,
        support=supp,
        convergence=False,
        allow_nonsmooth=True,
    )
    assert any(
        n.startswith("basis_remainder unbounded") for n in basis.provenance.notes
    )
    term = _zero_inputs(basis, tophat.n_ch).basis_remainder(basis, 1.0)
    assert term.kind == "unbounded" and term.value is None
    assert "smoothness 0 < N + 1 = 2" in term.note


def test_line_kernel_record_with_low_smoothness_is_unbounded_without_note():
    kernel, _ = polynomial_kernel(N=1, L=1)
    channels = Channels.tophat(LINE_NU, 0.1 * LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(1, 1, 1), reference(), support()
    )
    record = dataclasses.replace(basis.provenance, kernel=(("name", "harmonic"),))
    basis = dataclasses.replace(basis, provenance=record)
    term = _zero_inputs(basis, 3).basis_remainder(basis, 1.0)
    assert term.kind == "unbounded"
    assert "smoothness 0 < N + 1 = 2" in term.note


def test_smooth_non_line_basis_keeps_valued_remainder():
    kernel, _ = polynomial_kernel(N=1, L=1)
    channels = Channels.tophat(LINE_NU, 0.1 * LINE_NU)
    basis = build_stub_basis(
        kernel, channels, Truncation(1, 1, 1), reference(), support()
    )
    term = _zero_inputs(basis, 3).basis_remainder(basis, 1.0)
    assert term.kind == "bound"


# -- R06: several lines in one channel -----------------------------------------------


def test_screen_factorisation_per_line_inputs_cover_two_line_channel():
    # Lines at tau = 1 and 3; two equal rays with RM = 0 and pi/2, each ray
    # puts unit incident power on a different line.
    tau = np.array([1.0, 3.0])
    rm = np.array([0.0, np.pi / 2])
    P_in = np.array([[1.0, 0.0], [0.0, 1.0]])  # (ray, line)
    phase = np.exp(1j * rm[:, None] * tau[None, :])
    observed = np.mean(np.sum(P_in * phase, axis=1))
    factorised = np.sum(P_in.mean(0) * phase.mean(0))
    true = abs(observed - factorised)
    assert true == pytest.approx(0.5)
    sigma_line = np.sqrt(np.mean(np.abs(P_in - P_in.mean(0)) ** 2, axis=0))
    phi_line = np.abs(phase.mean(0))
    term = screen_factorisation_bound(sigma_line[None, :], phi_line[None, :])
    assert term.kind == "bound" and term.value.shape == (1, 4)
    value = np.asarray(term.value)[0]
    assert value[1] == pytest.approx(np.sum(sigma_line * np.sqrt(1 - phi_line**2)))
    assert value[1] >= true and value[2] == value[1] and value[0] == value[3] == 0
    # Channel-level inputs: the sum of the per-line sigmas with min |Phi| covers it.
    summed = screen_factorisation_bound(sigma_line.sum()[None], phi_line.min()[None])
    assert np.asarray(summed.value)[0, 1] >= true


# -- R07: phased channel coefficient vs the absolute line sum ------------------------

NU1, NU2 = 1.5e8, 1.6e8
T1, T2 = (float(phase_coordinate(n)) for n in (NU1, NU2))
TWO_DREF = np.pi / (T1 - T2)  # exp(i T1 d) + exp(i T2 d) = 0
TWO_SD = 1.0 / T1


class TwoLine(PolynomialTestKernel):
    """Two equal lines inside one channel; the polarised mode is the phased line sum."""

    def _w(self, phase, depth_ref, s_depth):
        return phase_weights(phase, jnp.asarray([T1]), depth_ref, s_depth) + (
            phase_weights(phase, jnp.asarray([T2]), depth_ref, s_depth)
        )

    def channel_modes(self, channels, gamma, B, mu, eta, *, phase=None, **kw):
        m = PolynomialTestKernel.channel_modes(self, channels, gamma, B, mu, eta, **kw)
        w = self._w(phase, kw.get("depth_ref", 0.0), kw.get("s_depth", 1.0))
        return type(m)(I=m.I, V=m.V, P=m.P[0][None, :] * w)

    def angular_projection(self, channels, gamma, B, *, phase=None, truncation, **kw):
        kw.pop("quadrature", None)
        m = PolynomialTestKernel.angular_projection(
            self, channels, gamma, B, truncation=truncation, **kw
        )
        w = self._w(phase, kw.get("depth_ref", 0.0), kw.get("s_depth", 1.0))
        return type(m)(I=m.I, V=m.V, P=m.P[0][None] * w[:, :, None])


def _two_line_case(dz):
    c = np.zeros((3, 1, 1, 1, 1, 1))
    c[0, 0, 0, 0, 0, 0] = c[1, 0, 0, 0, 0, 0] = 1.0
    kernel = TwoLine(c, np.array([NU1]), gamma0=GAMMA0, B0=B0, s_gamma=GAMMA0, s_B=B0)
    channels = Channels.bump(np.array([1.55e8]), np.array([0.2e8]))
    ref = Reference(GAMMA0, B0, TWO_DREF, scales=(GAMMA0, B0, TWO_SD))
    supp = Support(
        gamma=(GAMMA0 * 0.7, GAMMA0 * 1.3),
        B=(B0 * 0.7, B0 * 1.3),
        depth=(TWO_DREF - 2 * TWO_SD, TWO_DREF + 2 * TWO_SD),
        truncated=False,
    )
    basis = build_basis(
        kernel,
        channels,
        Truncation(0, 0, 0, depth_degree=0),
        ref,
        support=supp,
        phase=TaylorPhase(0),
        convergence=False,
    )
    samples = PopulationSamples(
        np.array([GAMMA0]),
        np.array([B0]),
        np.array([0.3]),
        np.array([0.1]),
        np.array([0.0]),
        np.array([TWO_DREF + dz * TWO_SD]),
    )
    moments = JointMoments.from_samples(samples, basis.index, ref)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=ref, phase=TaylorPhase(0)
    )
    return basis, moments, direct


@pytest.mark.parametrize("dz", [0.5, 1.0])
def test_phased_depth_coefficients_give_estimate(dz):
    basis, moments, direct = _two_line_case(dz)
    tail = np.array([abs(dz * TWO_SD)])
    inputs = _zero_inputs(basis, 1, depth_tail=tail, intrinsic_residual=np.zeros(1))
    prediction = predict(basis, moments, amplitude=1.0, errors=inputs)
    term = prediction.budget.basis_remainder
    error = np.abs(_polarised(prediction.stokes) - _polarised(direct.stokes))
    assert error[0] > 0.05 and np.asarray(term.value)[0, 1] < error[0]
    assert term.kind == "estimate"
    assert "absolute_depth_coefficients" in term.note


@pytest.mark.parametrize("dz", [0.5, 1.0])
def test_absolute_depth_coefficients_bound_covers_two_line_channel(dz):
    basis, moments, direct = _two_line_case(dz)
    tail = np.array([abs(dz * TWO_SD)])
    absolute = np.array([[2.0]])  # sum_m |c_{a,m}|: two unit lines
    inputs = _zero_inputs(
        basis,
        1,
        depth_tail=tail,
        intrinsic_residual=np.zeros(1),
        absolute_depth_coefficients=absolute,
    )
    prediction = predict(basis, moments, amplitude=1.0, errors=inputs)
    term = prediction.budget.basis_remainder
    error = np.abs(_polarised(prediction.stokes) - _polarised(direct.stokes))
    assert term.kind == "bound"
    assert np.asarray(term.value)[0, 1] >= error[0]
    with pytest.raises(ValueError, match="absolute_depth_coefficients"):
        _zero_inputs(
            basis,
            1,
            depth_tail=tail,
            intrinsic_residual=np.zeros(1),
            absolute_depth_coefficients=np.ones((2, 1)),
        ).basis_remainder(basis, 1.0, depth_coefficients=basis.P_basis)


# -- R14: screen routes carry no depth-Taylor term -----------------------------------


def test_screen_route_has_no_depth_taylor_term():
    kernel, _ = polynomial_kernel()
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    screen = GaussianScreen(mean=DEPTH_REF, sigma=0.3 * S_DEPTH)
    basis = build_basis(
        kernel,
        channels,
        Truncation(2, 2, 2, depth_degree=0),
        reference(),
        support=support(False),
        phase=screen,
    )
    pop = nine_atoms(constant_depth=False)
    pop["depth"] = pop["depth"] + 2.0 * S_DEPTH
    samples = samples_of(pop)
    inputs = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, phase=screen, angular_residual="probe"
    )
    moments = JointMoments.from_samples(samples, basis.index, reference())
    term = predict(basis, moments, amplitude=1.0, errors=inputs).budget.basis_remainder
    P = np.asarray(term.value)[:, 1]
    np.testing.assert_allclose(P, np.asarray(inputs.intrinsic_residual), rtol=1e-12)
    assert np.all(P < 1e-10)
    assert "independent_screen" in term.note
    # Supplied inputs on a screen basis need neither depth_tail nor depth_coefficients.
    supplied = _zero_inputs(basis, 3, intrinsic_residual=np.zeros(3))
    assert supplied.basis_remainder(basis, 1.0).kind == "bound"


# -- R30: probe order must be a validated order --------------------------------------


def test_probe_refuses_order_outside_certified_orders():
    kernel, _ = polynomial_kernel(N=1, L=1)
    channels = Channels.bump(LINE_NU, 0.1 * LINE_NU)
    samples = samples_of(nine_atoms())
    args = (kernel, channels, Truncation(1, 1, 1), reference(), support())
    beyond = build_stub_basis(*args, certified_orders=(0, 1))
    with pytest.raises(ValueError, match="certified_orders"):
        RemainderInputs.from_samples(samples, beyond, kernel=kernel)
    # Supplied envelopes need no derivative probe: no refusal.
    n_lk = beyond.index.n_lk
    RemainderInputs.from_samples(
        samples, beyond, derivative_envelope=np.zeros((3, 3, n_lk))
    )
    inside = build_stub_basis(*args, certified_orders=(0, 1, 2))
    inputs = RemainderInputs.from_samples(samples, inside, kernel=kernel)
    assert inputs.kind == "estimate"
