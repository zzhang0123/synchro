"""Fixtures and NumPy oracles of the T-006 reduction and combination tests (not collected).

* ``small_continuum_basis``: a cached ``ContinuumKernel`` basis (8 channels,
  ``n_eta=16``, ``n_nodes_F=64``, ``n_nu=8``, ``convergence=False``) at
  fractional ``(gamma, B)`` scales.
* ``harmonic_basis`` and ``polynomial_basis``: out-of-scope kernels.
* ``MatrixBasis``: a duck-typed basis around a hand-made response ``C``.
* ``continuum_jets``: an independent oracle for the continuum columns. Each
  row of every ``(h, part, l, k, b)`` block is the ``z``-Taylor jet of
  ``B Phi(B gamma^2)`` for a random polynomial ``Phi``, from binomial
  coefficients (no package code).
* ``manuscript_rows``: the six spec rows of ``T`` at 20 %/20 % scales.
* ``stokes_data``: diagonal or dense noise, masks and an observing response.
"""

from __future__ import annotations

import functools
from math import comb

import equinox as eqx
import jax
import numpy as np

from syncmoments.model.basis import build_basis
from syncmoments.model.channels import Channels
from syncmoments.model.errors import Provenance
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import Reference, Support
from syncmoments.model.phase import GaussianScreen, TaylorPhase

GAMMA0, B0, DEPTH0, S_DEPTH = 1.0e4, 5.0e-6, 2.0, 0.3
SUPPORT = Support(gamma=(8000.0, 12000.0), B=(4e-6, 6e-6), depth=(1.7, 2.3))
MANUSCRIPT_REPRESENTATIVES = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 2, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 0, 2),
)


def reference(eps_g=0.2, eps_B=0.2):
    return Reference(GAMMA0, B0, DEPTH0, scales=(eps_g * GAMMA0, eps_B * B0, S_DEPTH))


@functools.lru_cache(maxsize=None)
def small_continuum_basis(
    scales=(0.2, 0.2),
    truncation=(0, 2, 2, None, None),
    phase="taylor",
    n_ch=8,
    quad=(16, 64, 8),
):
    """Cached continuum basis; ``truncation = (L_mu, L_eta, N, depth_degree, caps)``,
    ``quad = (n_eta, n_nodes_F, n_nu)``."""
    L_mu, L_eta, N, depth_degree, caps = truncation
    trunc = Truncation(L_mu, L_eta, N, depth_degree, max_orders=caps)
    if phase == "taylor":
        route = TaylorPhase(trunc.max_b())
    elif phase == "gaussian":
        route = GaussianScreen(2.0, 0.1)
    else:
        raise ValueError(phase)
    centres = np.geomspace(0.4e9, 3.0e9, n_ch)
    n_eta, n_nodes_F, n_nu = quad
    channels = Channels.bump(
        centres, 0.04 * centres, normalisation="unit_integral", n_nu=n_nu
    )
    return build_basis(
        ContinuumKernel(n_eta=n_eta, n_nodes_F=n_nodes_F),
        channels,
        trunc,
        reference(*scales),
        support=SUPPORT,
        phase=route,
        convergence=False,
    )


@functools.lru_cache(maxsize=None)
def harmonic_basis():
    from _basis_fixtures import small_harmonic

    kernel, channels, ref, support = small_harmonic()
    return build_basis(kernel, channels, Truncation(0, 2, 2), ref, support=support)


@functools.lru_cache(maxsize=None)
def polynomial_basis():
    from _basis_fixtures import polynomial_setup

    kernel, _, _, ref, channels, support = polynomial_setup()
    return build_basis(
        kernel, channels, Truncation(2, 2, 2), ref, support=support, convergence=False
    )


def provenance(kernel_name):
    kernel = () if kernel_name is None else (("name", kernel_name),)
    return Provenance(
        "test", kernel, (), (), (), (), (), (), (), "test units", (), (), (), ()
    )


class MatrixBasis(eqx.Module):
    """Duck-typed basis: ``response_matrix()``, ``index``, ``reference``, ``provenance``."""

    index: MomentIndex = eqx.field(static=True)
    reference: Reference
    provenance: Provenance = eqx.field(static=True)
    C: jax.Array

    def response_matrix(self):
        return self.C


def matrix_basis(C, *, index=None, kernel=None, ref=None):
    """``MatrixBasis`` of ``C (4 n_ch, n_real)``; the default index is ``(0, 2, 2)`` I,Q."""
    if index is None:
        index = MomentIndex.build(Truncation(0, 2, 2), components=("I", "Q"))
    C = np.asarray(C, dtype=float)
    assert C.shape[1] == index.n_real and C.shape[0] % 4 == 0
    return MatrixBasis(
        index, reference() if ref is None else ref, provenance(kernel), C
    )


def random_columns(n_ch=6, seed=0, index=None):
    """Generic ``C`` on the default index plus hand-made dependencies (see body)."""
    index = index or MomentIndex.build(Truncation(0, 2, 2), components=("I", "Q"))
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((4 * n_ch, index.n_real))
    C[3::4] = 0.0  # V rows unused by an (I, Q) index
    C[:, 3] = 0.0  # a zero column
    C[:, 5] = C[:, 4]  # identical
    C[:, 7] = -2.5 * C[:, 6]  # proportional
    C[:, 10] = C[:, 8] - 0.5 * C[:, 9]  # three-column dependency
    return index, C


# -- continuum jet oracle -------------------------------------------------------------


def _jet(p, eps_g, eps_B, cells):
    """Taylor coefficients in ``z`` of ``B Phi(B gamma^2)``, ``Phi(y) = sum_j p_j y^j``."""
    return np.array(
        [
            sum(
                pj * comb(j + 1, s) * eps_B**s * comb(2 * j, r) * eps_g**r
                for j, pj in enumerate(p)
            )
            for r, s in cells
        ]
    )


def continuum_jets(index, eps_g, eps_B, n_rows, seed=0):
    """``(n_rows, n_real)`` rows of random continuum jets, independently per block."""
    rng = np.random.default_rng(seed)
    blocks = {}
    for entry in index.entries():
        key = (entry.h, entry.part, entry.l, entry.k, entry.b)
        blocks.setdefault(key, []).append(((entry.r, entry.s), entry.slot))
    C = np.zeros((n_rows, index.n_real))
    degree = index.truncation.N + 3
    for cells in blocks.values():
        rs = [c for c, _ in cells]
        slots = [s for _, s in cells]
        for row in range(n_rows):
            p = rng.standard_normal(degree) / np.array(
                [np.prod(np.arange(1, j + 1)) for j in range(degree)]
            )
            C[row, slots] = _jet(p, eps_g, eps_B, rs)
    return C


# -- manuscript T -----------------------------------------------------------------------

_GROUPS = (
    ((0, 0, 0), {(0, 0, 0): 1.0, (0, 1, 0): 0.2}),
    ((1, 0, 0), {(1, 0, 0): 1.0, (0, 1, 0): 0.5, (2, 0, 0): -0.3}),
    ((0, 2, 0), {(0, 2, 0): 1.0, (1, 1, 0): 4.0, (2, 0, 0): 4.0}),
    ((0, 0, 1), {(0, 0, 1): 1.0, (0, 1, 1): 0.2}),
    ((1, 0, 1), {(1, 0, 1): 1.0, (0, 1, 1): 0.5}),
    ((0, 0, 2), {(0, 0, 2): 1.0}),
)


def manuscript_rows(index):
    """``{representative slot: row (60,)}`` from the six spec rows (``(0, 2, 2)`` index)."""
    n_full = index.n_real + len(index.h0_ext)
    rows = {}
    for h, offset in ((0, 0), (2, 0), (2, index.n2)):
        for k in (0, 2):
            for source, terms in _GROUPS[:3] if h == 0 else _GROUPS:
                row = np.zeros(n_full)
                for powers, weight in terms.items():
                    row[index.position(h, 0, k, *powers) + offset] = weight
                rows[index.position(h, 0, k, *source) + offset] = row
    return rows


# -- observations -----------------------------------------------------------------------


def stokes_data(stokes, *, sigma=0.05, dense=False, mask=None, response=None, **kw):
    """``StokesData`` with variances ``sigma^2`` or a dense covariance with correlations."""
    stokes = np.asarray(stokes, dtype=float)
    n = stokes.size if response is None else np.asarray(response).shape[0]
    if dense:
        rng = np.random.default_rng(11)
        A = rng.standard_normal((n, n)) * 0.2 * sigma
        noise = A @ A.T / n + np.diag(np.full(n, sigma**2))
    else:
        noise = np.full(n, sigma**2)
    return StokesData(stokes, noise, mask=mask, response=response, **kw)


# -- NumPy oracle of the combination fit ------------------------------------------------


def oracle_fit(C, data, *, J=None, a_off=None, M=None, max_sigma, rank_tol=1e-8):
    """Ungrouped NumPy solution on the full ``C``: ``dict(s, K, Pi, Cov, x_hat, n_R)``.

    Explicit Cholesky whitening of the kept covariance, the observing
    response, the metric pull-back ``J^T M J = R^T R`` and an SVD of
    ``L^-1 R C J R^-1``; independent of ``reduce_response`` and of the
    package's SVD code.
    """
    C = np.asarray(C, dtype=float)
    n_full = C.shape[1]
    J = np.eye(n_full) if J is None else np.asarray(J, dtype=float)
    a_off = np.zeros(n_full) if a_off is None else np.asarray(a_off, dtype=float)
    M = np.eye(n_full) if M is None else np.asarray(M, dtype=float)
    Rm = np.linalg.cholesky(J.T @ M @ J).T
    Rm_inv = np.linalg.inv(Rm)
    Rresp = np.eye(C.shape[0]) if data.response is None else np.asarray(data.response)
    kept = list(data.kept_rows())
    noise = np.asarray(data.noise)
    cov = (np.diag(noise) if noise.ndim == 1 else noise)[np.ix_(kept, kept)]
    Lw = np.linalg.inv(np.linalg.cholesky(cov))
    X = Lw @ (Rresp @ C @ J @ Rm_inv)[kept]
    d = np.asarray(data.data_vector())
    y = Lw @ (d - Rresp @ C @ a_off)[kept]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = (s >= 1.0 / max_sigma) & (s > rank_tol * s[0])
    V, D, Uk = Vt[keep].T, s[keep], U[:, keep]
    W = Rm_inv @ V
    K = (W / D) @ Uk.T @ Lw
    return {
        "s": s[s > rank_tol * s[0]],
        "K": K,
        "Pi": W @ V.T @ Rm,
        "Cov": (W / D**2) @ W.T,
        "x_hat": W @ ((Uk.T @ y) / D),
        "n_R": int(keep.sum()),
    }


def continuum_problem(
    n_ch=8, *, noise_frac=1e-2, seed=0, dense=False, scales=(0.2, 0.2)
):
    """``(basis, a_true, clean, data)`` on the small continuum basis, I, Q, U kept."""
    basis = small_continuum_basis(scales, (0, 2, 2, None, None), "taylor", n_ch)
    C = np.asarray(basis.response_matrix())
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(C.shape[1] + len(basis.index.h0_ext))
    a[0] = 1.0
    clean = (C @ a[: C.shape[1]]).reshape(-1, 4)
    sigma = noise_frac * np.max(np.abs(clean[:, 0]))
    noisy = clean + sigma * rng.standard_normal(clean.shape) * np.array([1, 1, 1, 0])
    data = stokes_data(noisy, sigma=sigma, dense=dense, mask=np.array([1, 1, 1, 0]))
    return basis, a, clean, data


__all__ = [
    "oracle_fit",
    "continuum_problem",
    "GAMMA0",
    "B0",
    "DEPTH0",
    "S_DEPTH",
    "SUPPORT",
    "MANUSCRIPT_REPRESENTATIVES",
    "reference",
    "small_continuum_basis",
    "harmonic_basis",
    "polynomial_basis",
    "provenance",
    "MatrixBasis",
    "matrix_basis",
    "random_columns",
    "continuum_jets",
    "manuscript_rows",
    "stokes_data",
]
