"""Boundary sweeps of the Faraday-phase routes and the truncation grid.

``depth_ref in {0, 1e2, 1e4}`` (reference phases up to ``9e10`` rad),
``tau Delta_depth in {1e-3, 1, 10, 100}`` across ``depth_degree in {0, 1, 2, 4}``
(the measured Taylor error grows as ``|tau|^{N+1}`` and is covered by the
``eq: local response remainder`` bound, by the package's ``basis_remainder`` in
both index layouts where the probe order is certified, and the
``GaussianScreen`` stays exact at ``tau sigma = 100`` where the Taylor route
diverges), and the ``N in {0..3}`` x ``L in {0, 4, 8}`` grid on the
polynomial oracle (exact) and on the harmonic kernel (shared columns agree
across truncations). Both sides of every switch are evaluated directly.
"""

import itertools
import math

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

import synchro  # noqa: F401
from synchro.model.basis import build_basis
from synchro.model.bounds import RemainderInputs
from synchro.model.channels import Channels
from synchro.model.index import Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference, Support
from synchro.model.phase import EmpiricalScreen, GaussianScreen, TaylorPhase
from synchro.model.predict import direct_channel_average, predict

from _basis_fixtures import polynomial_setup, small_harmonic
from _boundary_oracles import (
    DEPTH_DEGREES,
    DEPTH_REFS,
    L_VALUES,
    N_VALUES,
    TAU_DELTA,
    constant_kernel,
    depth_reference,
    depth_support,
    gaussian_grid,
    slow,
    tau_of,
    taylor_remainder,
)
from _harmonic_oracles import NU_STAR

KERNEL, LINE = constant_kernel(1)
TAU = float(tau_of(LINE[0]))  # 17.98 m^2 at 1e8 Hz
CHANNELS = Channels.bump(LINE, 0.1 * LINE)
C_Q = 0.7
PHI = 0.4
SKY = complex(np.exp(2j * PHI))


def _polarisation(prediction):
    s = np.asarray(prediction.stokes)[0]
    return complex(s[1] + 1j * s[2])


def _asymmetric_population(depth_ref, delta, n=41):
    """Depths ``depth_ref + delta u`` with weights ``w (1 + 0.5 u)``: odd moments nonzero."""
    u, w = np.polynomial.legendre.leggauss(n)
    w = w * (1 + 0.5 * u)
    ones = np.ones(n)
    samples = PopulationSamples(
        5 * ones,
        2 * ones,
        0.3 * ones,
        -0.2 * ones,
        PHI * ones,
        depth_ref + delta * u,
        weights=w,
    )
    absolute = {p: float(np.sum(w * np.abs(u) ** p) / np.sum(w)) for p in range(1, 7)}
    # sky factor exp(2 i phi) at phi = 0.4 multiplies every P prediction
    exact = SKY * complex(
        np.sum(w * np.exp(1j * TAU * (depth_ref + delta * u))) / np.sum(w)
    )
    return samples, absolute, exact


# -- depth_ref extremes --------------------------------------------------------------------


@pytest.mark.parametrize("depth_ref", DEPTH_REFS)
@pytest.mark.parametrize("nu", [1.0e8, NU_STAR])
def test_reference_phase_extremes(depth_ref, nu):
    """``w_0 = exp(i tau depth_ref)`` has unit modulus, is finite and equals
    ``numpy.exp`` of the same float argument (up to ``9e10`` rad); the
    Gaussian screen with ``sigma = 0`` gives the same weight."""
    tau = jnp.asarray(tau_of(nu))
    w = np.asarray(TaylorPhase(2)(tau, depth_ref=depth_ref, s_depth=1.0))
    assert np.all(np.isfinite(w))
    assert_allclose(abs(w[0]), 1.0, rtol=1e-14)
    argument = float(tau) * depth_ref
    assert_allclose(w[0], np.exp(1j * argument), rtol=1e-12)
    assert_allclose(w[1], 1j * float(tau) * np.exp(1j * argument), rtol=1e-12)
    g = np.asarray(GaussianScreen(jnp.asarray(depth_ref), jnp.asarray(0.0))(tau))
    assert_allclose(g[0], w[0], rtol=1e-12)


@pytest.mark.parametrize("depth_ref", DEPTH_REFS)
def test_constant_depth_prediction_exact_at_every_reference(depth_ref):
    """A population at ``depth = depth_ref`` has zero remainder: ``predict``
    equals ``direct_channel_average`` to roundoff for every reference phase."""
    samples, _, exact = _asymmetric_population(depth_ref, 0.0)
    ref = depth_reference(depth_ref, 1.0)
    basis = build_basis(
        KERNEL,
        CHANNELS,
        Truncation(0, 0, 2),
        ref,
        support=depth_support(depth_ref, 1.0),
        convergence=False,
    )
    moments = JointMoments.from_samples(samples, basis.index, ref)
    pred = predict(basis, moments, amplitude=1.0)
    direct = direct_channel_average(
        samples, KERNEL, CHANNELS, amplitude=1.0, reference=ref
    )
    # the phase argument tau depth_ref reaches 1.8e5 rad: one ulp of tau is 2e-11 rad
    assert_allclose(_polarisation(pred), C_Q * exact, rtol=1e-9)
    assert_allclose(_polarisation(direct), C_Q * exact, rtol=1e-9)


# -- tau Delta_depth x depth_degree ----------------------------------------------------------


def _finite_and_direct(samples, ref, delta, truncation, degree):
    basis = build_basis(
        KERNEL,
        CHANNELS,
        truncation,
        ref,
        support=depth_support(ref.depth_ref, delta),
        phase=TaylorPhase(degree),
        convergence=False,
    )
    moments = JointMoments.from_samples(samples, basis.index, ref)
    pred = predict(basis, moments, amplitude=1.0)
    direct = direct_channel_average(
        samples, KERNEL, CHANNELS, amplitude=1.0, reference=ref
    )
    return basis, moments, pred, direct


@pytest.mark.parametrize("degree", DEPTH_DEGREES)
def test_taylor_remainder_covers_direct_error_and_grows_as_x_to_N_plus_1(
    degree, record_property
):
    """For every ``x = tau Delta``: the finite response is finite, its error
    against the direct average is at most the remainder bound
    ``|c_Q| x^{D+1}/(D+1)! <|u|^{D+1}>`` (equality up to ``O(x^{D+2})`` at
    small ``x``), the ``app: depth moments`` layout ``depth_degree = D`` and the
    main-text layout ``N = D`` agree, and the measured growth exponent
    between ``x = 1e-3`` and ``x = 1`` is ``D + 1`` within 0.15."""
    errors = {}
    for x in TAU_DELTA:
        delta = x / TAU
        samples, absolute, exact = _asymmetric_population(2.0, delta)
        ref = depth_reference(2.0, delta)
        outputs = [
            _finite_and_direct(
                samples, ref, delta, Truncation(0, 0, 0, depth_degree=degree), degree
            ),
            _finite_and_direct(samples, ref, delta, Truncation(0, 0, degree), degree),
        ]
        values = [_polarisation(o[2]) for o in outputs]
        direct = _polarisation(outputs[0][3])
        assert np.isfinite([abs(v) for v in values]).all() and np.isfinite(abs(direct))
        assert_allclose(direct, C_Q * exact, rtol=1e-11)
        assert_allclose(values[1], values[0], rtol=1e-11, atol=1e-13)
        bound = C_Q * taylor_remainder(x, degree, absolute[degree + 1])
        error = abs(values[0] - direct)
        errors[x] = error
        record_property(f"x_{x}_error_over_bound", error / bound)
        # Floor 1e-14: at x = 1e-3, D = 4 the bound is 1.2e-18 and the measured
        # error 5e-16 is the roundoff of the O(1) phase sums (|P| = 0.7).
        assert error <= bound * (1 + 1e-6) + 1e-14, (x, error, bound)
        # the finite value cannot exceed the exact one by more than the bound
        assert abs(values[0]) <= abs(direct) + bound * (1 + 1e-6) + 1e-14
    # D = 4 is at roundoff at x = 1e-3: use (1, 10) there (measured 5.05; lower
    # degrees are not asymptotic at x = 10: 0.68, 1.08, 2.94).
    lo, hi = (1e-3, 1.0) if errors[1e-3] > 1e-12 else (1.0, 10.0)
    exponent = math.log10(errors[hi] / errors[lo]) / math.log10(hi / lo)
    record_property("growth_exponent", exponent)
    assert abs(exponent - (degree + 1)) < 0.15, exponent


@pytest.mark.parametrize("degree", [0, 1, 2])
@pytest.mark.parametrize("x", [1e-3, 1.0, 10.0])
def test_package_basis_remainder_covers_direct_error_in_both_layouts(degree, x):
    """``RemainderInputs.from_samples`` (probe order ``D + 1 <= 3``, certified)
    gives a ``basis_remainder`` that covers the measured error in both
    layouts; the main-text layout reproduces the oracle bound to ``1e-9`` and
    the ``app: depth moments`` layout is looser by ``(nu_c/nu_lo)^{2(D+1)}`` (``tau_max``).
    """
    delta = x / TAU
    samples, absolute, _ = _asymmetric_population(2.0, delta)
    ref = depth_reference(2.0, delta)
    oracle = C_Q * taylor_remainder(x, degree, absolute[degree + 1])
    looser = (float(LINE[0]) / float(CHANNELS.support[0, 0])) ** (2 * (degree + 1))
    for layout, truncation in (
        ("main", Truncation(0, 0, degree)),
        ("appendix", Truncation(0, 0, 0, depth_degree=degree)),
    ):
        basis, moments, pred, direct = _finite_and_direct(
            samples, ref, delta, truncation, degree
        )
        inputs = RemainderInputs.from_samples(
            samples, basis, kernel=KERNEL, angular_residual="probe"
        )
        term = predict(
            basis, moments, amplitude=1.0, errors=inputs
        ).budget.basis_remainder
        assert term.kind == "estimate" and term.value is not None
        value = float(np.asarray(term.value)[0, 1])
        error = abs(_polarisation(pred) - _polarisation(direct))
        assert np.isfinite(value) and error <= value * (1 + 1e-6) + 1e-14, (
            layout,
            error,
            value,
        )
        if layout == "main":
            assert_allclose(value, oracle, rtol=1e-9)
        else:
            assert_allclose(value, oracle * looser, rtol=1e-9)
        assert float(np.asarray(term.value)[0, 0]) == 0.0  # I has no depth remainder


@pytest.mark.parametrize("x", TAU_DELTA)
def test_gaussian_screen_exact_where_taylor_diverges(x):
    """``GaussianScreen`` equals a fine discrete Gaussian screen (``EmpiricalScreen``
    and the direct per-emitter average) to ``1e-10`` at every ``tau sigma``,
    including 100 where the degree-4 Taylor route is off by ``~x^4/24``."""
    sigma = x / TAU
    mean = 3.0
    depths, weights = gaussian_grid(mean, sigma)
    tau = jnp.asarray(TAU)
    gaussian = complex(
        np.asarray(GaussianScreen(jnp.asarray(mean), jnp.asarray(sigma))(tau))[0]
    )
    empirical = complex(
        np.asarray(EmpiricalScreen(jnp.asarray(depths), jnp.asarray(weights))(tau))[0]
    )
    exact = complex(np.exp(1j * TAU * mean - 0.5 * (TAU * sigma) ** 2))
    assert np.isfinite(abs(gaussian)) and abs(gaussian - exact) < 1e-14
    assert abs(empirical - gaussian) < 1e-10
    n = depths.size
    ones = np.ones(n)
    samples = PopulationSamples(
        5 * ones, 2 * ones, 0.3 * ones, -0.2 * ones, PHI * ones, depths, weights=weights
    )
    ref = Reference(5.0, 2.0, mean, scales=(1.0, 1.0, max(sigma, 1e-12)))
    direct = direct_channel_average(
        samples, KERNEL, CHANNELS, amplitude=1.0, reference=ref
    )
    screened = direct_channel_average(
        samples,
        KERNEL,
        CHANNELS,
        amplitude=1.0,
        reference=ref,
        phase=GaussianScreen(jnp.asarray(mean), jnp.asarray(sigma)),
    )
    assert abs(_polarisation(screened) - SKY * C_Q * gaussian) < 1e-12
    assert abs(_polarisation(direct) - _polarisation(screened)) < 1e-10 * C_Q
    # both sides of the route switch at x = 100: Taylor degree 4 diverges, the screen does not
    if x == 100.0:
        basis = build_basis(
            KERNEL,
            CHANNELS,
            Truncation(0, 0, 0, depth_degree=4),
            ref,
            support=Support(
                gamma=(2, 9), B=(0.5, 4), depth=(mean - 8 * sigma, mean + 8 * sigma)
            ),
            phase=TaylorPhase(4),
            convergence=False,
        )
        taylor = predict(
            basis, JointMoments.from_samples(samples, basis.index, ref), amplitude=1.0
        )
        assert abs(_polarisation(taylor)) > 1e5 * C_Q  # ~ x^4 <u^4>/24 with <u^4> = 3
        assert np.isfinite(abs(_polarisation(taylor)))


# -- N x L grid ----------------------------------------------------------------------------


def _nine_atoms(reference, seed=2):
    rng = np.random.default_rng(seed)
    n = 9
    return PopulationSamples(
        float(reference.gamma0) * (1 + 0.1 * rng.uniform(-1, 1, n)),
        float(reference.B0) * (1 + 0.1 * rng.uniform(-1, 1, n)),
        rng.uniform(-0.9, 0.9, n),
        rng.uniform(-0.9, 0.9, n),
        rng.uniform(0, 2 * np.pi, n),
        np.full(n, float(reference.depth_ref)),
        weights=rng.uniform(0.5, 2.0, n),
    )


@pytest.mark.parametrize("N,L", list(itertools.product(N_VALUES, L_VALUES)))
def test_polynomial_kernel_exact_on_the_N_L_grid(N, L):
    """Polynomial oracle of degrees ``<= (N, L)`` at constant depth: the finite
    response equals the direct average to ``1e-10`` for every ``(N, L)``."""
    kernel, _, _, reference, channels, support = polynomial_setup(
        seed=N * 10 + L, n_ch=2, L=L, N=N
    )
    basis = build_basis(
        kernel,
        channels,
        Truncation(L, L, N),
        reference,
        support=support,
        convergence=False,
    )
    samples = _nine_atoms(reference)
    moments = JointMoments.from_samples(samples, basis.index, reference)
    pred = predict(basis, moments, amplitude=1.0)
    direct = direct_channel_average(
        samples, kernel, channels, amplitude=1.0, reference=reference
    )
    scale = float(np.max(np.abs(np.asarray(direct.stokes))))
    assert np.all(np.isfinite(np.asarray(pred.stokes)))
    assert_allclose(
        np.asarray(pred.stokes), np.asarray(direct.stokes), rtol=0, atol=1e-10 * scale
    )
    assert basis.index.n_real == moments.to_vector().shape[0]


HARMONIC_GRID = [
    (N, L) for N, L in itertools.product(N_VALUES, L_VALUES) if not (N == 3 and L > 0)
]
HARMONIC_SLOW = [(3, 4), (3, 8)]


def _shared_columns(a, b):
    """Rows and pairs common to two bases: every shared column must agree."""
    rows_a, rows_b = {r: i for i, r in enumerate(a.index.h0)}, {
        r: i for i, r in enumerate(b.index.h0)
    }
    common = [(rows_a[r], rows_b[r]) for r in rows_a if r in rows_b]
    ia, ib = zip(*common)
    h2a, h2b = {r: i for i, r in enumerate(a.index.h2)}, {
        r: i for i, r in enumerate(b.index.h2)
    }
    common2 = [(h2a[r], h2b[r]) for r in h2a if r in h2b]
    ja, jb = zip(*common2)
    return (
        np.asarray(a.I_basis)[:, list(ia)],
        np.asarray(b.I_basis)[:, list(ib)],
        np.asarray(a.V_basis)[:, list(ia)],
        np.asarray(b.V_basis)[:, list(ib)],
        np.asarray(a.P_basis)[:, list(ja)],
        np.asarray(b.P_basis)[:, list(jb)],
    )


def _harmonic_grid_check(cells):
    kernel, channels, reference, support = small_harmonic()
    bases = {}
    for N, L in cells:
        basis = build_basis(
            kernel,
            channels,
            Truncation(L, L, N),
            reference,
            support=support,
            convergence=False,
        )
        for name in ("I_basis", "V_basis", "P_basis"):
            assert np.all(np.isfinite(np.asarray(getattr(basis, name)))), (N, L, name)
        bases[N, L] = basis
    base = bases[min(cells)]
    for key, basis in bases.items():
        for a, b in zip(
            _shared_columns(base, basis)[::2], _shared_columns(base, basis)[1::2]
        ):
            scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
            assert_allclose(a, b, rtol=0, atol=1e-12 * scale, err_msg=str(key))
    return bases


def test_harmonic_shared_columns_agree_across_the_N_L_grid():
    """Columns shared by two truncations (rows ``r + s <= min N``, pairs
    ``l, k <= min L``) are identical: both sides of every ``N`` and ``L`` switch."""
    bases = _harmonic_grid_check(HARMONIC_GRID)
    for (N, L), basis in bases.items():
        assert basis.index.truncation == Truncation(L, L, N)
        assert basis.kernel_terms.harmonic_truncation.kind == "bound"


@slow
@pytest.mark.slow
def test_harmonic_shared_columns_agree_at_N_3_large_L():
    _harmonic_grid_check([(0, 0)] + HARMONIC_SLOW)
