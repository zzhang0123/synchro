"""Setup and NumPy oracles for ``test_fit_integration*.py`` (not collected).

The pipeline is the harmonic kernel (``m_max = required_m_max <= 40``) on 16
bump channels at ``Truncation(1, 1, 1)``: 64 data rows, ``n_real = 28``
(masking the V rows makes the six odd-parity moments a null space).
Populations are discrete atoms whose variables are all functions of one
latent ``t`` (correlated), optionally extended by a Gauss-Legendre pitch
grid so that the pitch angle is exactly isotropic and independent.

``fit_linear``, ``fisher`` and ``FitResult`` are the package's
(``syncmoments.model.fit.linear`` and ``syncmoments.model.fit.result``).
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

import syncmoments  # noqa: F401
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncmoments.model.assumptions import Parameters
from syncmoments.model.channels import Channels
from syncmoments.model.fit.linear import fisher, fit_linear
from syncmoments.model.fit.observation import StokesData
from syncmoments.model.fit.result import FitResult
from syncmoments.model.harmonic import HarmonicKernel
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import required_m_max
from syncmoments.model.moments import PopulationSamples, Reference, Support

GAMMA0, B0 = 5.0, 1.0
NU_STAR = E_ESU * B0 / (2 * np.pi * GAMMA0 * M_E * C_CGS)
TAU_STAR = 2 * (C_SI_M / NU_STAR) ** 2
S_DEPTH = 1.0 / TAU_STAR
DEPTH_REF = 0.5 * S_DEPTH
N_CH = 16
TRUNCATION = Truncation(1, 1, 1)
AMPLITUDE = 2.0e17  # brings the per-electron powers (~1e-17) to order one
NOISE_FRACTION = 2e-3  # sigma relative to the largest channel I


def setup():
    """``(kernel, channels, support, reference)`` with ``m_max <= 40``."""
    centres = np.geomspace(1.5, 6.0, N_CH) * NU_STAR
    channels = Channels.bump(centres, 0.18 * centres)
    support = Support(
        gamma=(4.0, 6.0), B=(0.8, 1.2), depth=(0.0, S_DEPTH), truncated=False
    )
    m_max = required_m_max(support, channels)
    assert m_max <= 40
    kernel = HarmonicKernel(m_max, n_outer=16, n_inner=16)
    reference = Reference(GAMMA0, B0, DEPTH_REF, scales=(GAMMA0, B0, S_DEPTH))
    return kernel, channels, support, reference


@functools.lru_cache(maxsize=None)
def cached_basis(truncation=TRUNCATION):
    """One ``SpectralBasis`` per truncation for the whole session (about 4 s each)."""
    from syncmoments.model.basis import build_basis

    kernel, channels, support, reference = setup()
    basis = build_basis(
        kernel, channels, truncation, reference, support=support, convergence=False
    )
    return kernel, basis


def correlated_population(n=12, seed=3):
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, n)
    return dict(
        gamma=GAMMA0 * (1 + 0.12 * t + 0.03 * t**2),
        B=B0 * (1 + 0.1 * t - 0.05 * t**2),
        mu=np.clip(0.6 * t + 0.1 * rng.standard_normal(n), -0.9, 0.9),
        eta=np.clip(0.3 - 0.5 * t, -0.9, 0.9),
        phi=0.4 + 1.1 * t,
        depth=DEPTH_REF + S_DEPTH * (0.25 * t + 0.1 * t**2),
        w=rng.uniform(0.5, 2.0, n),
    )


def isotropic_extension(pop, n_mu=8):
    """Outer product of ``pop`` with a Gauss-Legendre pitch grid (uniform ``mu``)."""
    x, w = np.polynomial.legendre.leggauss(n_mu)
    n = pop["w"].size
    out = {k: np.repeat(v, n_mu) for k, v in pop.items() if k not in ("mu", "w")}
    out["mu"] = np.tile(x, n)
    out["w"] = np.repeat(pop["w"], n_mu) * np.tile(w / 2.0, n)
    return out


def samples_of(pop, *, weights=True):
    return PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=pop["w"] if weights else None,
    )


def make_data(basis, m, amplitude, seed, *, noise=NOISE_FRACTION, **kwargs):
    """``StokesData`` with ``d = A C m + eta``, ``eta ~ N(0, sigma^2)``, uniform
    ``sigma = noise * max_j I_j``; ``seed=None`` leaves the data exact (the
    declared sigma is kept). Extra keywords go to ``StokesData``."""
    C = np.asarray(basis.response_matrix())
    d = amplitude * C @ np.asarray(m)
    sigma = noise * np.max(np.abs(d.reshape(-1, 4)[:, 0]))
    if seed is not None:
        d = d + sigma * np.random.default_rng(seed).standard_normal(d.size)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, sigma**2), **kwargs)
    return data, sigma


def whitened_design(basis, data, parameter_map):
    """NumPy ``(L^-1 G, L^-1 d)`` with ``G = C [c | P]`` (no response, diagonal noise)."""
    P, c = parameter_map.affine_pieces(reference=basis.reference)
    C = np.asarray(basis.response_matrix())
    G = C @ np.column_stack([np.asarray(c), np.asarray(P)])
    sigma = np.sqrt(np.asarray(data.noise))
    return G / sigma[:, None], np.asarray(data.data_vector()) / sigma


def model_of(basis, parameter_map, u):
    """``C [c | P] u`` (``(4 n_ch,)`` channel-major) for unknowns ``u = (A, A theta)``."""
    P, c = parameter_map.affine_pieces(reference=basis.reference)
    C = np.asarray(basis.response_matrix())
    return C @ np.column_stack([np.asarray(c), np.asarray(P)]) @ np.asarray(u)


def unknowns(parameter_map, theta, amplitude):
    """``u = (A, A theta)`` in :meth:`ParameterMap.flatten` order."""
    return np.concatenate(
        [[amplitude], amplitude * np.asarray(parameter_map.flatten(theta))]
    )


def with_amplitude(theta, amplitude):
    return Parameters(
        log_amplitude=jnp.log(jnp.asarray(amplitude, float)),
        tables=theta.tables,
        hyper=theta.hyper,
        logits=theta.logits,
    )


__all__ = [
    "AMPLITUDE",
    "DEPTH_REF",
    "FitResult",
    "GAMMA0",
    "N_CH",
    "NOISE_FRACTION",
    "S_DEPTH",
    "TRUNCATION",
    "cached_basis",
    "correlated_population",
    "fisher",
    "fit_linear",
    "isotropic_extension",
    "make_data",
    "model_of",
    "samples_of",
    "setup",
    "unknowns",
    "whitened_design",
    "with_amplitude",
]
