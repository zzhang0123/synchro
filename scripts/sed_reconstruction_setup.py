"""Configuration and helpers of the SED reconstruction example (public API only).

Constants of main.tex Section 5.3 (noise, seed, precision threshold, the
manuscript representatives), the grid sizes of the full and ``--quick`` runs,
and the basis, truth, mock-data and fit helpers shared by
``sed_reconstruction_example.py`` and ``sed_reconstruction_checks.py``.
Units: package units for Stokes data (``NORM`` times the direct SED).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import syncmoments  # noqa: F401  (enables float64)
from syncmoments.model import (
    Channels,
    ContinuumKernel,
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
    TaylorPhase,
    Truncation,
    build_basis,
)
from syncmoments.model.fit import StokesData, fit_combinations

import sed_reconstruction_direct as direct

NOISE_FRACTION = 0.01  # sigma = 1 % of direct I on I, Q and U
SEED = 20260925
MAX_SIGMA = 0.1  # illustrative precision criterion 1/s_i <= 0.1, not a recommendation
KAPPA = 1.15  # continuum rescaling B -> kappa B, gamma -> gamma/sqrt(kappa)
AMPLITUDE_BOUND = 2.0  # illustrative declared bound on the source-column ratio
MANUSCRIPT_REPRESENTATIVES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 2, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 0, 2),
)  # q_0, q_g, q_BB, q_d, q_gd, q_dd: reproduces the manuscript T
STEM = "sed_moment_reconstruction"
QUICK_NOTE = "quick configuration: reduced grids, not comparable with history"


@dataclass(frozen=True)
class Config:
    """Grid sizes of one run; ``direct`` is ``(n_eta, n_nu)``, ``basis`` ``(n_eta, n_nu, n_nodes_F)``."""

    n_centres: int
    n_fit: int
    nodes: int
    coarse_nodes: int
    direct: tuple
    coarse_direct: tuple
    basis: tuple
    refined: tuple
    mc_draws: int
    quick: bool


FULL = Config(
    72, 24, 16, 12, (96, 24), (64, 16), (64, 24, 128), (96, 32, 192), 10000, False
)
QUICK = Config(18, 6, 8, 6, (32, 12), (24, 8), (24, 12, 64), (32, 16, 96), 500, True)


def item(value, label, note=""):
    """One validation entry; ``label`` is finite check, measured or statistical."""
    return {"value": value, "label": label, "note": note}


def make_basis(centres, n_eta, n_nu, n_f):
    return build_basis(
        ContinuumKernel(n_eta=n_eta, n_nodes_F=n_f),
        Channels.bump(
            centres,
            direct.HALF_WIDTH * centres,
            normalisation="unit_integral",
            n_nu=n_nu,
        ),
        Truncation(0, 2, 2),
        Reference(direct.GAMMA0, direct.B0, direct.DEPTH0, scales=direct.SCALES),
        support=Support(**direct.SUPPORT),
        phase=TaylorPhase(2),
        convergence=False,
    )


def truth(layout, reference, nodes):
    """``(a_true, closed_form, moments)``: package route and closed-form route."""
    grid = direct.sample_grid(nodes)
    samples = PopulationSamples(
        grid["gamma"],
        grid["B"],
        grid["mu"],
        grid["eta"],
        grid["phi"],
        grid["depth"],
        weights=grid["weights"],
    )
    moments = JointMoments.from_samples(samples, layout.index, reference)
    a_true = layout.full_vector(moments, amplitude=1.0)
    return a_true, direct.moment_vector(layout.entries, nodes), moments


def mock_data(sed, fit_ch):
    """``(StokesData, noise_full, sigma_kept)`` in package units (``NORM`` times ``sed``)."""
    n_ch = sed.shape[0]
    sigma_ch = NOISE_FRACTION * sed[:, 0] * direct.NORM
    kept = np.array(sorted(4 * ch + comp for ch in fit_ch for comp in range(3)))
    sigma_kept = sigma_ch[kept // 4]
    noise = np.zeros(4 * n_ch)
    noise[kept] = np.random.default_rng(SEED).standard_normal(kept.size) * sigma_kept
    stokes = np.zeros((n_ch, 4))
    stokes[:, :3] = sed * direct.NORM
    mask = np.zeros((n_ch, 4), bool)
    mask[fit_ch, :3] = True
    data = StokesData(
        stokes + noise.reshape(n_ch, 4), np.repeat(sigma_ch**2, 4), mask=mask
    )
    return data, noise, sigma_kept


def combinations_fit(basis, data, reduction, bound=None):
    return fit_combinations(
        basis, data, reduction=reduction, max_sigma=MAX_SIGMA, coefficient_bound=bound
    )
