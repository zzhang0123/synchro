"""Helpers of the manuscript Section 5.3.1 benchmark (not collected).

The population is ``eq: channel toy population`` on the latent Gauss-Legendre
grid of ``validation/full_response.py``: for every latent node pair ``(u, v)``
the conditional angular measure is normalised to one on its own tensor
Gauss-Legendre grid, and the latent weights ``lw_u lw_v / 4`` sum to one.
Moments are accumulated block by block over latent pairs so that the
``16 x 16 x 256 x 256`` population never has to be materialised at once.

Units: the package returns erg/s/sr per electron; the manuscript's saved
numbers are in ``e^2 Omega_0^2 / (2 pi c)`` with ``Omega_0 = e B_0 / (m_e c)``
(``UNITS`` of ``_harmonic_oracles``, ``B_0 = 1`` G).
"""

import json
from pathlib import Path

import equinox as eqx
import numpy as np
from numpy.polynomial.legendre import leggauss

from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference

from _harmonic_oracles import B0, GAMMA0, MANUSCRIPT, S_DEPTH, UNITS

RESULTS = Path(MANUSCRIPT) / "validation" / "full_response_results.json"
WIDTHS = (0.5, 1.0)
STOKES = ("I", "Q", "U", "V")


def reference() -> Reference:
    """``Reference(20, B0, 4 s_depth, (20, B0, s_depth))`` of FINAL_DESIGN Section 11."""
    return Reference(GAMMA0, B0, depth_ref=4.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH))


# -- saved manuscript numbers -------------------------------------------------------------


def load_results() -> dict:
    with open(RESULTS) as handle:
        return json.load(handle)


def saved_row(results, width, N, L) -> dict:
    for row in results["rows"]:
        if row["width_scale"] == width and row["N"] == N and row["L_mu"] == L:
            assert row["L_eta"] == L
            return row
    raise KeyError((width, N, L))


def saved_stokes(row, key) -> np.ndarray:
    """``(n_ch, 4)`` Stokes ``I, Q, U, V`` from the JSON ``[stokes][channel]`` layout."""
    return np.array(row[key], dtype=float).T


def saved_stability(results, width) -> dict:
    for entry in results["stability"]:
        if entry["width_scale"] == width:
            return entry
    raise KeyError(width)


def error_over_channel_I(stokes, reference_stokes) -> float:
    """``max |S - S_ref| / I_ref`` over Stokes components and channels."""
    return float(np.max(np.abs(stokes - reference_stokes) / reference_stokes[:, [0]]))


def to_manuscript_units(stokes) -> np.ndarray:
    return np.asarray(stokes, dtype=float) / UNITS


# -- the toy population, block by block over latent nodes --------------------------------


def latent_grid(latent_nodes):
    latent, lw = leggauss(latent_nodes)
    pairs = [(iu, iv) for iu in range(latent_nodes) for iv in range(latent_nodes)]
    return latent, lw, pairs


def block_samples(width, latent, lw, pairs, angular_nodes):
    """``(PopulationSamples, mass)`` of the latent pairs ``pairs``.

    Weights are ``lw_u lw_v / 4`` times the per-pair normalised angular measure;
    ``mass`` is their total, so that ``mass * (block average)`` is the block's
    contribution to the population average.
    """
    nodes, weights = leggauss(angular_nodes)
    u = latent[[iu for iu, _ in pairs]][:, None, None]
    v = latent[[iv for _, iv in pairs]][:, None, None]
    mass_uv = np.array([lw[iu] * lw[iv] / 4 for iu, iv in pairs])[:, None, None]
    mu = nodes[None, :, None]
    eta = nodes[None, None, :]
    rho = np.exp((2 + u) * mu + (1 + 0.5 * v) * eta + 0.75 * mu * eta)
    measure = weights[None, :, None] * weights[None, None, :] * rho
    measure = measure / measure.sum(axis=(1, 2), keepdims=True)
    q_gamma = width * 0.2 * u
    q_B = width * 0.2 * (0.6 * u + 0.4 * v)
    zeta = 4.0 + width * (2 * u + v)
    shape = (len(pairs), angular_nodes, angular_nodes)

    def full(x):
        return np.ascontiguousarray(np.broadcast_to(x, shape).ravel())

    samples = PopulationSamples(
        full(GAMMA0 * (1 + q_gamma)),
        full(B0 * (1 + q_B)),
        full(mu),
        full(eta),
        full(0.2 + 0.4 * mu + 0.2 * v),
        full(zeta * S_DEPTH),
        weights=full(mass_uv * measure),
    )
    return samples, float(mass_uv.sum())


def full_population(width, latent_nodes, angular_nodes) -> PopulationSamples:
    """All latent pairs in one ``PopulationSamples`` (for the direct average)."""
    latent, lw, pairs = latent_grid(latent_nodes)
    samples, mass = block_samples(width, latent, lw, pairs, angular_nodes)
    assert abs(mass - 1.0) < 1e-13
    return samples


def toy_moments(width, latent_nodes, angular_nodes, index, *, pairs_per_block=None):
    """``JointMoments`` of the toy population on ``index`` (block accumulation).

    The result equals ``JointMoments.from_samples`` on the whole population up
    to the summation order of the blocks; ``m0_ext`` is accumulated as well.
    """
    latent, lw, pairs = latent_grid(latent_nodes)
    if pairs_per_block is None:
        pairs_per_block = max(1, (1 << 20) // (angular_nodes * angular_nodes))
    ref = reference()

    @eqx.filter_jit
    def block_moments(samples):
        moments = JointMoments.from_samples(samples, index, ref)
        return moments.to_vector(), moments.m0_ext

    vector, ext = None, None
    for start in range(0, len(pairs), pairs_per_block):
        block = pairs[start : start + pairs_per_block]
        samples, mass = block_samples(width, latent, lw, block, angular_nodes)
        m, m_ext = block_moments(samples)
        m, m_ext = mass * np.asarray(m), mass * np.asarray(m_ext)
        vector = m if vector is None else vector + m
        ext = m_ext if ext is None else ext + m_ext
    moments = JointMoments.from_vector(index, vector, ref, tol=1e-12)
    return eqx.tree_at(lambda x: x.m0_ext, moments, ext, is_leaf=lambda x: x is None)


# -- restriction of a moment tensor to a smaller truncation --------------------------------


def restrict_moments(moments: JointMoments, index: MomentIndex) -> JointMoments:
    """The rows of ``index`` read from a moment tensor on a larger truncation.

    Exact (the same population and reference); this is how the manuscript's
    ``contract`` slices one coefficient/moment tensor for every ``(N, L)``.
    """
    big = moments.index
    vector = np.asarray(moments.to_vector())
    out = np.zeros(index.n_real)
    for pos, (l, k, r, s, b) in enumerate(index.h0):
        out[pos] = vector[big.position(0, l, k, r, s, b)]
    for pos, (l, k, r, s, b) in enumerate(index.h2):
        slot = big.position(2, l, k, r, s, b)
        out[index.n0 + pos] = vector[slot]
        out[index.n0 + index.n2 + pos] = vector[slot + big.n2]
    return JointMoments.from_vector(index, out, moments.reference, tol=1e-12)


def mixed_row_mask(index: MomentIndex) -> np.ndarray:
    """``True`` on the slots of ``m`` whose ``(r, s, b)`` has two or more nonzero
    exponents (the manuscript's ``drop_mixed`` rows), over ``h0``, ``Re h2``, ``Im h2``.
    """
    mixed_h0 = [sum(x > 0 for x in (r, s, b)) > 1 for (_, _, r, s, b) in index.h0]
    mixed_h2 = [sum(x > 0 for x in (r, s, b)) > 1 for (_, _, r, s, b) in index.h2]
    return np.array(mixed_h0 + mixed_h2 + mixed_h2, dtype=bool)


def drop_mixed(moments: JointMoments) -> JointMoments:
    vector = np.asarray(moments.to_vector()) * ~mixed_row_mask(moments.index)
    return JointMoments.from_vector(moments.index, vector, moments.reference, tol=1e-12)


def truncation(L, N) -> Truncation:
    return Truncation(L, L, N)
