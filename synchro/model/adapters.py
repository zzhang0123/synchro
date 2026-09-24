"""Adapters between the finite joint response and the existing package routines
(``synchro.model.adapters``).

LABEL: ``eq: joint screen response`` and ``eq: joint screen remainder``
(:func:`to_joint_faraday`: the ``app: depth moments`` contraction of
``synchro.faraday.joint_faraday_average``), ``extra eq: channel kernel``
against the fixed-harmonic quadratic average of ``synchro.expansion``
(:func:`fixed_harmonic_comparison`; two different observables, not an
equality), ``eq: independent gaussian screen`` against
``synchro.rm.burn_depolarisation`` (:func:`burn_screen_check`).

Shapes: channel quantities are ``(n_ch, 4)`` or ``(n_ch,)``; the ``app: depth moments``
contraction uses ``(n_ch, n_a)`` coefficients and an ``(n_a, L+1)`` matrix.
Units follow ``synchro.model.predict``: channel Stokes are the amplitude
times the per-electron channel kernel; wavelengths are in metres; depths in
rad/m^2. ``to_joint_faraday`` evaluates the nominal phase coordinate at each
channel centre, so the equality with ``predict`` holds up to the variation
of ``tau = 2 (c/nu)^2`` across the channel support (exact for a delta-like
channel; the relative residual grows like the depth degree times the
fractional channel width). Nothing here certifies the physics of either side
of a comparison; the returned differences are measured on the supplied
population only.
"""

from __future__ import annotations

from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C_SI_M
from ..expansion import mixed_moments
from ..rm import burn_depolarisation
from ..stokes import stokes_harmonic
from ._direct import direct_channel_average
from ._remainder_probe import depth_rows, depth_tail
from .errors import ErrorTerm
from .moments import JointMoments
from .phase import GaussianScreen
from .predict import predict


def joint_faraday_rows(index) -> tuple:
    """``app: depth moments`` row order ``a = (l, k, r, s)``: the ``h2`` rows with ``b = 0``.

    Dimensionless bookkeeping (``app: depth moments``); assumes the ``app: depth moments``
    layout of the index; nothing certified.
    """
    return tuple(row[:4] for row in depth_rows(index))


def to_joint_faraday(basis, moments, *, samples=None, absolute_next=None):
    """``(coefficients (n_ch, n_a), M (n_a, L+1), absolute_next (n_a,) | None)``.

    ``basis`` must use the ``app: depth moments`` layout (``depth_degree = L``);
    ``coefficients[j, a]`` is the ``b = 0`` column of ``P_basis`` divided by
    ``exp(i tau_j depth_ref)`` with ``tau_j = 2 (c / centre_j)^2`` (the
    nominal wavelength ``lam_j = C_SI_M / centre_j``); ``M[a, b]`` is
    ``M2[l k; r s b] s_depth^b`` (raw depth displacements, ``z`` for
    ``r, s``). ``absolute_next`` is the supplied envelope, or
    ``<|chi_a| |depth - depth_ref|^{L+1}>`` computed on ``samples``, else
    ``None`` (``joint_faraday_average`` requires it). With
    ``joint_faraday_average(coefficients, M, lam, absolute_next=..., source_error=0,
    reference_depth=basis.reference.depth_ref, source_column=amplitude)`` the
    polarised channel prediction equals ``predict`` up to the channel-width
    residual named in the module docstring. Units: coefficients in the basis
    units, ``M`` dimensionless in ``z`` for ``r, s`` and rad/m^2 powers for
    ``b``. Assumes the channel is narrow enough for one nominal ``tau_j``;
    not certified: that width residual (measured in the tests, not bounded).
    """
    index = basis.index
    L = index.truncation.depth_degree
    if L is None:
        raise ValueError(
            "to_joint_faraday needs the ``app: depth moments`` layout (depth_degree=L)"
        )
    if moments.index != index:
        raise ValueError("moments.index must equal basis.index")
    rows = joint_faraday_rows(index)
    n0 = index.n0
    s_depth = basis.reference.scales[2]
    columns = np.array(
        [[index.position(2, *row, b) - n0 for b in range(L + 1)] for row in rows],
        dtype=int,
    ).reshape(len(rows), L + 1)
    powers = s_depth ** jnp.arange(L + 1, dtype=float)
    M = moments.m2[jnp.asarray(columns)] * powers
    tau = 2.0 * (C_SI_M / jnp.asarray(basis.channels.centres_hz)) ** 2
    phase = jnp.exp(-1j * tau * basis.reference.depth_ref)
    coefficients = basis.P_basis[:, jnp.asarray(columns[:, 0])] * phase[:, None]
    if absolute_next is not None:
        absolute_next = jnp.asarray(absolute_next, dtype=float)
        if absolute_next.shape != (len(rows),):
            raise ValueError(f"absolute_next must have shape ({len(rows)},)")
    elif samples is not None:
        absolute_next = depth_tail(samples, basis)
    return coefficients, M, absolute_next


class HarmonicComparison(eqx.Module):
    """Channel prediction versus the fixed-harmonic quadratic average.

    ``channel_finite``/``channel_direct`` ``(n_ch, 4)`` (``channel_direct``
    is ``None`` without a kernel); ``fixed_finite``/``fixed_direct``
    ``(n_harmonics, 4)`` in sky Stokes order at fixed harmonic number and
    physical ``B``; the two ``remainder`` terms are measured differences on
    the supplied population (``channel_remainder`` is the basis-remainder
    term of the budget when no kernel is given). The observables differ
    (fixed channel versus fixed harmonic index), so no equality is asserted.
    Units: channel entries in the basis units, fixed-harmonic entries in
    erg/s/sr per electron. Assumes the vacuum helical-orbit model on both
    sides; both remainders are measured on the supplied population and
    certify nothing for other populations. ``[extension]``.
    """

    LABEL: ClassVar[str] = "[extension] fixed-harmonic comparison"
    channel_finite: jax.Array
    channel_direct: jax.Array | None
    channel_remainder: ErrorTerm
    fixed_finite: jax.Array
    fixed_direct: jax.Array
    fixed_remainder: ErrorTerm
    harmonics: tuple = eqx.field(static=True)
    note: str = eqx.field(static=True)


def _fixed_harmonic_direct(samples, harmonics, amplitude):
    """``A sum_n w_n S_m(gamma_n, alpha_n, theta_n; B_n)`` in sky Stokes, ``(n_harm, 4)``."""
    w = samples.normalised_weights()
    alpha, theta = jnp.arccos(samples.mu), jnp.arccos(samples.eta)
    rows = []
    for m in harmonics:
        I, Q, V = jax.vmap(lambda g, a, t, B, m=m: stokes_harmonic(m, g, a, t, B=B))(
            samples.gamma, alpha, theta, samples.B
        )
        P = jnp.exp(2j * samples.phi) * Q
        rows.append(jnp.stack([w @ I, w @ jnp.real(P), w @ jnp.imag(P), w @ V]))
    return amplitude * jnp.stack(rows)


def fixed_harmonic_comparison(
    basis, expansion, samples, *, alpha0, theta0, kernel=None, amplitude=1.0
) -> HarmonicComparison:
    """Compare ``predict`` on ``basis`` with ``expansion.mixed_average`` on ``samples``.

    ``expansion`` is a ``synchro.expansion.CumulantExpansion`` built at
    ``(basis.reference.gamma0, alpha0, theta0, B=basis.reference.B0)``;
    its offsets are ``(gamma - gamma0, alpha - alpha0, theta - theta0)`` with
    ``alpha = arccos(mu)``, ``theta = arccos(eta)`` and the exact field
    factors ``(B/B0)^2`` and ``(B/B0)^2 exp(2 i phi)``. The fixed-harmonic
    direct reference is ``synchro.stokes.stokes_harmonic`` at every sample.
    ``kernel`` (the basis kernel) enables the channel direct average.
    Units: see :class:`HarmonicComparison`. Assumes the same vacuum
    helical-orbit model on both sides; nothing certified beyond the
    measured differences on ``samples``.
    """
    reference = basis.reference
    moments = JointMoments.from_samples(samples, basis.index, reference)
    channel = predict(basis, moments, amplitude=amplitude)
    if kernel is None:
        channel_direct, channel_remainder = None, channel.budget.basis_remainder
    else:
        direct = direct_channel_average(
            samples,
            kernel,
            basis.channels,
            amplitude=amplitude,
            reference=reference,
            support=basis.support,
        )
        channel_direct = direct.stokes
        channel_remainder = ErrorTerm(
            jnp.abs(channel.stokes - direct.stokes),
            "measured",
            "|predict - direct_channel_average| on the supplied population",
            "basis remainder",
        )
    offsets = jnp.stack(
        [
            samples.gamma - reference.gamma0,
            jnp.arccos(samples.mu) - jnp.asarray(alpha0, dtype=float),
            jnp.arccos(samples.eta) - jnp.asarray(theta0, dtype=float),
        ],
        axis=1,
    )
    factor = (samples.B / reference.B0) ** 2
    field = mixed_moments(offsets, factor, samples.weights)
    phase = mixed_moments(offsets, factor * jnp.exp(2j * samples.phi), samples.weights)
    fixed_finite, _ = expansion.mixed_average(field, phase, absolute_error=0.0)
    fixed_finite = jnp.asarray(amplitude) * fixed_finite
    harmonics = tuple(int(m) for m in expansion.harmonics)
    fixed_direct = _fixed_harmonic_direct(samples, harmonics, jnp.asarray(amplitude))
    fixed_remainder = ErrorTerm(
        jnp.abs(fixed_finite - fixed_direct),
        "measured",
        "|mixed_average - direct fixed-harmonic average| on the supplied population "
        "(quadratic kernel in (gamma, alpha, theta), exact B^2 and sky factors)",
        "fixed-harmonic Taylor remainder",
    )
    return HarmonicComparison(
        channel_finite=channel.stokes,
        channel_direct=channel_direct,
        channel_remainder=channel_remainder,
        fixed_finite=fixed_finite,
        fixed_direct=fixed_direct,
        fixed_remainder=fixed_remainder,
        harmonics=harmonics,
        note=(
            "fixed-channel and fixed-harmonic observables differ; the two remainders "
            "are measured on the supplied population and are not comparable bounds"
        ),
    )


class BurnCheck(NamedTuple):
    """``screened`` and ``nominal`` polarised channel kernels ``(n_ch,)`` complex,
    their ``difference`` (``ErrorTerm`` ``(n_ch, 4)``, measured) and the nominal
    wavelengths ``(n_ch,)`` in metres. Kernels in ``amplitude`` times
    erg/s/sr per electron. Assumes the Gaussian screen of
    ``eq: independent gaussian screen``; the difference is a measurement
    for the supplied channel and viewing point, not a bound."""

    LABEL = "eq: independent gaussian screen"
    screened: jax.Array
    nominal: jax.Array
    difference: ErrorTerm
    wavelengths_m: jax.Array


def burn_screen_check(
    kernel, channels, gamma, B, mu, eta, mean, sigma, *, amplitude=1.0
) -> BurnCheck:
    """``GaussianScreen`` applied line by line versus Burn rotation at the channel centre.

    ``screened = A sum_m Q_m R_j(nu_m) w_0(tau_m)`` with
    ``w_0 = exp(i tau mean - tau^2 sigma^2 / 2)`` (``eq: independent gaussian
    screen``); ``nominal = burn_depolarisation(P0_j, mean, sigma^2, lam_j)``
    with ``P0_j = A sum_m Q_m R_j(nu_m)`` and ``lam_j = C_SI_M / centre_j``.
    The two agree for a delta-like channel; their difference is the error of
    rotating an already integrated channel at one nominal wavelength.
    Units: ``amplitude`` times erg/s/sr per electron; ``mean`` in rad/m^2,
    ``sigma`` in rad/m^2, ``B`` in Gauss. Assumes an independent Gaussian
    screen; the difference is measured at one viewing point and is not a
    bound over a population.
    """
    screen = GaussianScreen(
        jnp.asarray(mean, dtype=float), jnp.asarray(sigma, dtype=float)
    )
    amplitude = jnp.asarray(amplitude, dtype=float)
    screened = (
        amplitude * kernel.channel_modes(channels, gamma, B, mu, eta, phase=screen).P[0]
    )
    P0 = amplitude * kernel.channel_modes(channels, gamma, B, mu, eta, phase=None).P[0]
    lam = C_SI_M / jnp.asarray(channels.centres_hz)
    nominal = burn_depolarisation(P0, screen.mean, screen.sigma**2, lam)
    diff = jnp.abs(screened - nominal)
    zero = jnp.zeros_like(diff)
    difference = ErrorTerm(
        jnp.stack([zero, diff, diff, zero], axis=1),
        "measured",
        "|line-by-line Gaussian screen - Burn rotation at the channel centre|",
        "E_phys",
    )
    return BurnCheck(screened, nominal, difference, lam)


__all__ = [
    "joint_faraday_rows",
    "to_joint_faraday",
    "HarmonicComparison",
    "fixed_harmonic_comparison",
    "BurnCheck",
    "burn_screen_check",
]
