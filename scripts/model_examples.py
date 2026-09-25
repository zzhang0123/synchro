"""Executed examples of the README section "Finite joint response and spectral fits".

Run from the package root with ``PYTHONPATH=. python scripts/model_examples.py``
(optionally ``a``, ``b``, ``c`` or ``d`` to select examples). The printed
numbers are pasted into ``README.md``; rerun this script after any change to
``syncmoments.model`` and update the README when the output changes.

(a) the manuscript's smooth-channel benchmark regime (main.tex Section 5.3.1)
    with the harmonic kernel, ``independent_screen`` declared and
    ``discrepancy="measured"`` (this file);
(b) the continuum kernel in the Galactic regime (``gamma0 = 3000``,
    ``B0 = 5 uG``, bump channels between 0.1 and 3 GHz), where
    ``isotropic_pitch`` is required and ``physical_kernel`` stays unbounded
    (``model_examples_continuum.py``);
(c) ``fit_linear`` on synthetic channel data with identifiability and
    feasibility output (``model_examples_fit.py``);
(d) ``assumption_allowances`` on a screen route, with
    ``bounds.screen_factorisation_bound`` for ``independent_screen``
    (``model_examples_continuum.py``).

Nothing printed here is a certificate: ``E_phys`` and ``E_tail`` are inputs,
the refined-rule ``numerical`` term and the remainder probe are estimates,
the measured assumption term holds for the supplied population only, and an
allowance is the caller's declaration.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import syncmoments  # noqa: F401  (enables float64)
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.assumptions import independent_screen
from syncmoments.model.basis import build_basis
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.channels import Channels
from syncmoments.model.errors import ErrorTerm
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import required_m_max
from syncmoments.model.moments import (
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)
from syncmoments.model.predict import predict

from model_examples_continuum import banner, example_b, example_d
from model_examples_fit import example_c

ROOT = Path(__file__).resolve().parents[1]
# Verbatim, hash-checked copy of the manuscript's saved benchmark results
# (tests/model/reference/README.md); the script needs no manuscript checkout.
MANUSCRIPT_RESULTS = (
    ROOT / "tests" / "model" / "reference" / "validation" / "full_response_results.json"
)

GAMMA0, B0 = 20.0, 1.0  # manuscript benchmark: gamma0 = 20, B0 = 1 G
NU_STAR = E_ESU * B0 / (2 * np.pi * GAMMA0 * M_E * C_CGS)  # Hz
S_DEPTH = 1.0 / (2 * (C_SI_M / NU_STAR) ** 2)  # rad/m^2 per unit zeta
UNITS = E_ESU**2 * (E_ESU * B0 / (M_E * C_CGS)) ** 2 / (2 * np.pi * C_CGS)


def toy_population(width=1.0, latent_nodes=16, angular_nodes=64) -> PopulationSamples:
    """Discrete measure of ``eq: channel toy population`` (validation/full_response.py)."""
    nodes, weights = leggauss(angular_nodes)
    latent, lw = leggauss(latent_nodes)
    u = latent[:, None, None, None]
    v = latent[None, :, None, None]
    mu = nodes[None, None, :, None]
    eta = nodes[None, None, None, :]
    rho = np.exp((2 + u) * mu + (1 + 0.5 * v) * eta + 0.75 * mu * eta)
    measure = weights[None, None, :, None] * weights[None, None, None, :] * rho
    measure = measure / measure.sum(axis=(2, 3), keepdims=True)
    w = lw[:, None, None, None] * lw[None, :, None, None] / 4 * measure
    shape = (latent_nodes, latent_nodes, angular_nodes, angular_nodes)
    full = lambda x: np.broadcast_to(x, shape).ravel()  # noqa: E731
    return PopulationSamples(
        gamma=full(GAMMA0 * (1 + width * 0.2 * u)),
        B=full(B0 * (1 + width * 0.2 * (0.6 * u + 0.4 * v))),
        mu=full(mu),
        eta=full(eta),
        phi=full(0.2 + 0.4 * mu + 0.2 * v),
        depth=full((4 + width * (2 * u + v)) * S_DEPTH),
        weights=w.ravel(),
    )


def manuscript_row(width, N, L):
    """``(direct, finite)`` in erg/s/sr per electron from the manuscript's saved results."""
    with open(MANUSCRIPT_RESULTS) as handle:
        rows = json.load(handle)["rows"]
    for row in rows:
        if row["width_scale"] == width and row["N"] == N and row["L_mu"] == L:
            direct = np.asarray(row["direct_IQUV"]).T * UNITS  # (n_ch, 4)
            finite = np.asarray(row["finite_IQUV"]).T * UNITS
            return direct, finite
    raise KeyError((width, N, L))


def example_a():
    banner(
        "(a) harmonic kernel, manuscript benchmark regime, independent_screen measured"
    )
    t0 = time.time()
    channels = Channels.bump(
        centres_hz=np.array([2.0, 4.0, 8.0]) * NU_STAR,
        widths_hz=0.65 * np.array([2.0, 4.0, 8.0]) * NU_STAR,
    )
    reference = Reference(
        GAMMA0, B0, depth_ref=4.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    support = Support(
        gamma=(16.0, 24.0), B=(0.8, 1.2), depth=(0.0, 10.0 * S_DEPTH), truncated=False
    )
    truncation = Truncation(L_mu=8, L_eta=8, N=2)
    kernel = HarmonicKernel(m_max=40)
    print("required_m_max(support, channels) =", required_m_max(support, channels))

    basis = build_basis(kernel, channels, truncation, reference, support=support)
    index = basis.index
    print(
        f"index: n0={index.n0} n2={index.n2} n_real={index.n_real}  "
        f"build time {time.time() - t0:.1f} s"
    )
    print("numerics:", dict(basis.provenance.numerics))

    samples = toy_population(width=1.0)
    print("population size:", samples.size)
    joint = JointMoments.from_samples(samples, index, reference)
    screened = JointMoments.from_samples(
        samples,
        index,
        reference,
        parameter_map=independent_screen(index),
        discrepancy="measured",
    )
    direct, finite = manuscript_row(1.0, 2, 8)

    pred_joint = predict(basis, joint, amplitude=1.0)
    scale = np.abs(direct[:, 0])[:, None]
    print(
        "joint moments:  max |pred - manuscript finite| / I_direct per channel:",
        np.abs(np.asarray(pred_joint.stokes) - finite).max(axis=1) / scale[:, 0],
    )
    print(
        "joint moments:  max |pred - manuscript direct| / I_direct per channel:",
        np.abs(np.asarray(pred_joint.stokes) - direct).max(axis=1) / scale[:, 0],
    )

    pred = predict(basis, screened, amplitude=1.0)
    print(pred.summary())
    print("budget.total().kind =", pred.budget.total().kind)
    print("budget.unbounded() =", pred.budget.unbounded())
    name, term = pred.budget.assumption[0]
    gap = np.abs(np.asarray(pred.stokes) - np.asarray(pred_joint.stokes))
    print(
        f"assumption '{name}': kind {term.kind}; max |pred_screen - pred_joint| / I = "
        f"{(gap / scale).max():.3e}; max term / I = {(np.asarray(term.value) / scale).max():.3e}; "
        f"covered: {bool(np.all(gap <= np.asarray(term.value) * (1 + 1e-9) + 1e-30))}"
    )

    # Declared inputs: the synthetic population is complete, its moments are exact
    # and the harmonic reference is the truth. basis_remainder needs a probe.
    t1 = time.time()
    # The order-3 derivative envelope H is probed at the reference point only
    # (segment_points=()): one nested jacfwd of the L = 8 projection. The angular
    # residual rho_ang is measured on a separate 2-node (16-atom) Gauss-Legendre
    # discretisation of the same toy model, not a subset of the population.
    probe = toy_population(width=1.0, latent_nodes=2, angular_nodes=2)  # 16 atoms
    probed = RemainderInputs.from_samples(
        probe, basis, kernel=kernel, segment_points=(), angular_residual="probe"
    )
    # The absolute moments <|P_l P_k| ||z||^3> must belong to the predicted
    # population: recompute them on `samples` (weighted sums, no jacfwd) with
    # H and rho_ang supplied from the probe, and keep the probe's kind.
    on_population = RemainderInputs.from_samples(
        samples, basis, derivative_envelope=probed.H, angular_residual=probed.rho_ang
    )
    remainder = RemainderInputs(
        rho_ang=probed.rho_ang,
        H=probed.H,
        absolute_moments=on_population.absolute_moments,
        kind=probed.kind,
    )
    print(
        f"remainder probe (H at the reference, rho_ang on the {probe.size}-atom "
        f"discretisation, absolute moments on the {samples.size} population "
        f"samples): {time.time() - t1:.1f} s, kind {remainder.kind}"
    )
    ratio = np.asarray(remainder.absolute_moments).sum(axis=1) / np.asarray(
        probed.absolute_moments
    ).sum(axis=1)
    print(
        "absolute-moment row sums, population / 16-atom probe (z_2, z_3):",
        np.array2string(ratio, precision=3),
    )
    declared = build_basis(
        kernel,
        channels,
        truncation,
        reference,
        support=support,
        E_phys=ErrorTerm.declared_zero(
            "synthetic population: harmonic reference is the truth"
        ),
    )
    pred2 = predict(
        declared,
        screened,
        amplitude=1.0,
        errors=remainder,
        statistical_input=np.zeros(index.n_real),  # exact moments of the measure
        amplitude_uncertainty=0.0,
        depth_model=ErrorTerm.declared_zero("depths are the exact sample values"),
    )
    print(pred2.summary())
    print("budget.total().kind =", pred2.budget.total().kind)
    print("budget.unbounded() =", pred2.budget.unbounded())
    env = np.asarray(pred2.budget.total().value)
    print(
        "envelope covers |pred_screen - manuscript direct|:",
        bool(np.all(np.abs(np.asarray(pred2.stokes) - direct) <= env)),
    )
    print(f"total time {time.time() - t0:.1f} s")
    return pred2


def main(argv):
    which = set(argv[1:]) or {"a", "b", "c", "d"}
    if "a" in which:
        example_a()
    if "b" in which:
        example_b()
    if "c" in which:
        example_c()
    if "d" in which:
        example_d()


if __name__ == "__main__":
    main(sys.argv)
