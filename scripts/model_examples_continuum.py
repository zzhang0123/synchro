"""Continuum examples (b) and (d) of ``scripts/model_examples.py``.

(b) the continuum kernel in the Galactic regime (``gamma0 = 3000``,
    ``B0 = 5 uG``, bump channels between 0.1 and 3 GHz), where
    ``isotropic_pitch`` is required and ``physical_kernel`` stays unbounded;
(d) ``assumption_allowances`` on a screen route: two rays whose field
    strength and Faraday depth are correlated, an ``EmpiricalScreen`` of their
    depths, and ``bounds.screen_factorisation_bound`` as the caller's
    allowance for ``independent_screen``.

Run through ``scripts/model_examples.py``; nothing printed is a certificate.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

import syncmoments  # noqa: F401  (enables float64)
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.assumptions import isotropic_pitch
from syncmoments.model.basis import build_basis
from syncmoments.model.bounds import RemainderInputs, screen_factorisation_bound
from syncmoments.model.channels import Channels
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel, required_m_max
from syncmoments.model.moments import (
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)
from syncmoments.model.phase import EmpiricalScreen, GaussianScreen
from syncmoments.model.predict import direct_channel_average, predict


def banner(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def galactic_setup(n_ch=8):
    """Local Galactic reference: gamma0 = 3000, B0 = 5 uG, screen mean 30 rad/m^2."""
    centres = np.geomspace(0.1e9, 3.0e9, n_ch)  # Hz
    channels = Channels.bump(centres_hz=centres, widths_hz=0.3 * centres)
    reference = Reference(
        gamma0=3000.0, B0=5e-6, depth_ref=30.0, scales=(300.0, 5e-7, 5.0)
    )
    support = Support(gamma=(2700.0, 3300.0), B=(4.5e-6, 5.5e-6), depth=(0.0, 60.0))
    screen = GaussianScreen(mean=30.0, sigma=5.0)  # rad/m^2, Burn depolarisation
    return channels, reference, support, screen


def galactic_population(n_gamma=32, n_B=8, n_mu=16):
    """Power-law energies on (2700, 3300), field within +/-10 %, uniform pitch, one
    viewing angle (60 degrees) and one sky azimuth; a product measure, so the
    ``isotropic_pitch`` factorisation is exact on it. The depth column is a
    placeholder: with a screen phase route the prediction uses the screen's
    characteristic function, not the sample depths."""
    x, w = leggauss(n_gamma)
    gamma = 3000.0 + 300.0 * x
    w_gamma = w * gamma ** (-2.5)  # N(gamma) ~ gamma^-2.5
    xb, wb = leggauss(n_B)
    B = 5e-6 * (1.0 + 0.1 * xb)
    xa, wa = leggauss(n_mu)
    return PopulationSamples.product(
        gamma=(gamma, w_gamma),
        B=(B, wb),
        mu=(xa, wa),
        eta=(np.array([np.cos(np.pi / 3)]), np.array([1.0])),
        phi=(np.array([0.3]), np.array([1.0])),
        depth=(np.array([30.0]), np.array([1.0])),
    )


def example_b():
    banner("(b) continuum kernel, Galactic regime, isotropic_pitch required")
    t0 = time.time()
    channels, reference, support, screen = galactic_setup()
    print("channels (GHz):", np.round(np.asarray(channels.centres_hz) / 1e9, 3))
    print("required_m_max(support, channels) =", required_m_max(support, channels))
    # eq: directional continuum: x = nu / (a_B gamma^2), a_B = 3 e B_perp / (4 pi m_e c),
    # at the population's viewing angle (eta = cos 60 deg) and B0 = 5 uG.
    a_B = 3 * E_ESU * 5e-6 * np.sin(np.pi / 3) / (4 * np.pi * M_E * C_CGS)
    print(
        "critical frequency a_B gamma^2 (GHz) at gamma = 2700, 3000, 3300:",
        np.round(a_B * np.array([2700.0, 3000.0, 3300.0]) ** 2 / 1e9, 3),
    )
    lam = C_SI_M / np.asarray(channels.centres_hz)
    burn = np.exp(-2 * float(screen.sigma) ** 2 * lam**4)
    print("Burn factor exp(-2 sigma^2 lambda^4) at the channel centres:")
    print(" ", np.array2string(burn, precision=4))
    try:
        build_basis(
            HarmonicKernel(m_max=300),
            channels,
            Truncation(0, 2, 2),
            reference,
            support=support,
            allow_truncated=True,
        )
    except ValueError as err:
        print("HarmonicKernel refused:", err)

    truncation = Truncation(
        L_mu=0, L_eta=2, N=2, depth_degree=0
    )  # screen route: max b = 0
    kernel = ContinuumKernel()
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, phase=screen
    )
    index = basis.index
    print(
        f"index: components={index.components} n0={index.n0} n2={index.n2} "
        f"n_real={index.n_real}  build time {time.time() - t0:.1f} s"
    )
    print("forced assumptions:", [r.name for r in basis.provenance.assumptions])
    print("notes:", basis.provenance.notes)
    try:
        build_basis(
            kernel,
            channels,
            Truncation(1, 2, 2, depth_degree=0),
            reference,
            support=support,
            phase=screen,
        )
    except ValueError as err:
        print("L_mu > 0 refused:", err)

    samples = galactic_population()
    print("population size:", samples.size)
    moments = JointMoments.from_samples(
        samples,
        index,
        reference,
        parameter_map=isotropic_pitch(index),
        discrepancy="measured",
    )
    t1 = time.time()
    probe = galactic_population(n_gamma=2, n_B=2, n_mu=2)  # 8 corner samples
    remainder = RemainderInputs.from_samples(
        probe,
        basis,
        kernel=kernel,
        phase=screen,
        segment_points=(1.0,),
        angular_residual="probe",
    )
    print(
        f"remainder probe on {probe.size} samples: {time.time() - t1:.1f} s, "
        f"kind {remainder.kind}"
    )
    pred = predict(
        basis,
        moments,
        amplitude=1e20,
        errors=remainder,
        statistical_input=np.zeros(index.n_real),
    )  # electrons per cm^2; exact moments
    print(pred.summary())
    print("budget.total().kind =", pred.budget.total().kind)
    print("budget.unbounded() =", pred.budget.unbounded())
    for name, term in pred.budget.assumption:
        value = (
            "-"
            if term.value is None
            else f"{float(np.max(np.asarray(term.value))):.3e}"
        )
        print(f"assumption '{name}': kind {term.kind}, max value {value}")
    stokes = np.asarray(pred.stokes)
    print(
        "polarisation fraction |P|/I per channel:",
        np.round(np.hypot(stokes[:, 1], stokes[:, 2]) / stokes[:, 0], 4),
    )
    direct = direct_channel_average(
        samples,
        kernel,
        channels,
        amplitude=1e20,
        reference=reference,
        phase=screen,
        support=support,
    )
    gap = np.abs(stokes - np.asarray(direct.stokes)).max(axis=1) / stokes[:, 0]
    print("max |pred - direct| / I per channel:", np.array2string(gap, precision=2))
    valued = sum(
        np.broadcast_to(np.asarray(term.value), stokes.shape)
        for _, term in pred.budget.terms()
        if term.value is not None
    )
    print(
        "sum of the valued terms / I per channel:",
        np.array2string(valued.max(axis=1) / stokes[:, 0], precision=2),
    )
    print(
        "valued terms cover |pred - direct|:",
        bool(np.all(np.abs(stokes - np.asarray(direct.stokes)) <= valued)),
    )
    print(f"total time {time.time() - t0:.1f} s")
    return basis, samples, moments, pred


def two_ray_population(fields=(4.7e-6, 5.3e-6), depths=(25.0, 35.0)):
    """Two equal-mass rays, each a product measure (power-law energies, uniform
    pitch, viewing angle 60 degrees); the field strength and the Faraday depth
    change together from ray to ray, so emission and screen are correlated."""
    x, w = leggauss(32)
    gamma, mu = 3000.0 + 300.0 * x, leggauss(16)
    rays = [
        PopulationSamples.product(
            gamma=(gamma, w * gamma ** (-2.5)),
            B=np.array([B]),
            mu=mu,
            eta=np.array([np.cos(np.pi / 3)]),
            phi=np.array([0.3]),
            depth=np.array([depth]),
        )
        for B, depth in zip(fields, depths)
    ]
    join = lambda name: jnp.concatenate([getattr(r, name) for r in rays])  # noqa: E731
    weights = jnp.concatenate([0.5 * r.normalised_weights() for r in rays])
    columns = ("gamma", "B", "mu", "eta", "phi", "depth")
    return rays, PopulationSamples(*(join(v) for v in columns), weights=weights)


def incident_polarisation(kernel, channels, ray):
    """``<K_Q exp(2 i phi)>`` of one ray at the channel quadrature nodes ``(n_ch, n_nu)``.

    The samples go through ``jax.lax.map`` in batches of
    ``kernel.samples_per_step(channels, 256)``, so one step holds at most the
    kernel's ``chunk_budget`` ``F``/``G`` values (summation order only)."""
    nodes = jnp.asarray(channels.nodes)

    def one(leaf):
        gamma, B, eta, phi = leaf
        return kernel.kernels(nodes, gamma, B, eta)[1] * jnp.exp(2j * phi)

    batch = kernel.samples_per_step(channels, 256)
    values = jax.lax.map(one, (ray.gamma, ray.B, ray.eta, ray.phi), batch_size=batch)
    return jnp.einsum("s,scn->cn", ray.normalised_weights(), values)


def screen_allowance(kernel, channels, rays, depths, amplitude):
    """``screen_factorisation_bound`` in its per-line form, with the channel
    quadrature nodes as the lines: ``sigma_{j,n} = w_n |R_j(nu_n)| sigma_P(nu_n)``
    (ray standard deviation of the incident polarisation at node ``n``) and
    ``Phi_{j,n} = |<exp(i tau_n depth)>|`` over the rays."""
    nodes = np.asarray(channels.nodes)
    response = np.diagonal(np.asarray(channels(jnp.asarray(nodes))), axis1=0, axis2=1).T
    P = np.stack([np.asarray(incident_polarisation(kernel, channels, r)) for r in rays])
    sigma_P = np.sqrt(np.mean(np.abs(P - P.mean(axis=0)) ** 2, axis=0))  # equal masses
    sigma = np.asarray(channels.weights) * np.abs(response) * sigma_P
    tau = 2.0 * (C_SI_M / nodes) ** 2
    phi = np.abs(np.mean(np.exp(1j * tau[..., None] * np.asarray(depths)), axis=-1))
    return screen_factorisation_bound(sigma, np.minimum(phi, 1.0), amplitude=amplitude)


def example_d():
    banner("(d) assumption_allowances: screen_factorisation_bound on a screen route")
    t0 = time.time()
    channels, reference, support, _ = galactic_setup()
    depths, amplitude = (25.0, 35.0), 1e20
    rays, samples = two_ray_population(depths=depths)
    screen = EmpiricalScreen(jnp.asarray(depths), jnp.asarray([0.5, 0.5]))
    kernel = ContinuumKernel()
    truncation = Truncation(L_mu=0, L_eta=2, N=2, depth_degree=0)
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, phase=screen
    )
    moments = JointMoments.from_samples(samples, basis.index, reference)
    inputs = dict(
        amplitude=amplitude, statistical_input=0.0, amplitude_uncertainty=0.0
    )  # exact moments of the measure, amplitude declared exact
    allowances = {
        "independent_screen": screen_allowance(
            kernel, channels, rays, depths, amplitude
        ),
        "isotropic_pitch": ErrorTerm.declared_zero(
            "each ray is uniform in mu by construction", shape=(channels.n_ch, 4)
        ),
    }
    before = predict(basis, moments, **inputs)
    after = predict(basis, moments, **inputs, assumption_allowances=allowances)
    print("forced assumptions:", [r.name for r in basis.provenance.assumptions])
    print("without allowances: budget.unbounded() =", before.budget.unbounded())
    print("with allowances:    budget.unbounded() =", after.budget.unbounded())
    for name, term in after.budget.assumption:
        print(f"  assumption:{name}: {term.kind}; note: {term.note[:60]}...")
    exact = direct_channel_average(samples, kernel, channels, amplitude=amplitude,
                                   reference=reference)  # per-emitter phase  # fmt: skip
    factorised = direct_channel_average(
        samples, kernel, channels, amplitude=amplitude, reference=reference, phase=screen,
        assumption_allowances={"independent_screen": allowances["independent_screen"]},
    )  # fmt: skip
    error = np.abs(np.asarray(exact.stokes) - np.asarray(factorised.stokes))
    bound = np.asarray(allowances["independent_screen"].value)
    scale = np.asarray(exact.stokes)[:, :1]
    print("screen error |direct exact - direct screen| / I:",
          np.array2string(error.max(axis=1) / scale[:, 0], precision=3))  # fmt: skip
    print("screen_factorisation_bound / I:",
          np.array2string(bound.max(axis=1) / scale[:, 0], precision=3))  # fmt: skip
    # Two rays attain Cauchy-Schwarz at each node: allow roundoff (relative 1e-12).
    print("bound covers the screen error:", bool(np.all(error <= bound * (1 + 1e-12))))
    print("direct screen route: assumption:independent_screen is",
          dict(factorised.budget.assumption)["independent_screen"].kind)  # fmt: skip
    print(f"total time {time.time() - t0:.1f} s")
    return after
