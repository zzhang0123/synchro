"""Jitted core and eager helpers of ``syncmoments.model.basis`` (private).

Split out of ``basis.py`` to keep that file under 400 lines. ``build_core``
evaluates ``kernel.angular_projection`` at the reference and its derivatives
in ``(z_gamma, z_B)`` through the total Taylor degree ``N`` (``eq: channel
derivative coefficients``; nested ``jax.jacfwd``, or ``angular_taylor`` of
``HarmonicKernel(derivatives="analytic")``), divides by ``r! s!``
and scatters the results into the ``I``, ``V`` and ``P`` basis columns of
the ``MomentIndex`` rows. The depth index ``b`` selects the phase weight
``w_b`` (which carries its own ``1/b!``). Everything else here is eager
bookkeeping: the refined kernels of the ``numerical`` estimate and the
route check, the per-column envelope helpers, the active-harmonic record at
the reference and the
``screen_exponent`` term. Units and shapes follow ``syncmoments.model.basis``;
nothing here certifies quadrature accuracy.
"""

from __future__ import annotations

import dataclasses
import math

import jax.numpy as jnp
import numpy as np

from ..constants import C_CGS, C_SI_M, E_ESU, M_E
from ._basis_taylor import columns, projection_in_z, taylor_blocks, taylor_tensors
from ._channel_shapes import legendre_rule
from ._continuum import ContinuumKernel
from ._harmonic_cells import auto_nodes
from .errors import ErrorTerm
from .harmonic import HarmonicKernel
from .index import Truncation

# Beyond this required harmonic no HarmonicKernel can be built (auto_nodes > 65536).
M_REGIME = (65536 - 64) // 4


def package_version() -> str:
    """This package's ``syncmoments.__version__`` (single source, see pyproject.toml).

    Installed-distribution metadata is not used: ``syncmoments`` on PyPI is an
    unrelated project, so ``importlib.metadata.version("syncmoments")`` can name
    the wrong package.
    """
    from .. import __version__

    return __version__


def concrete(value, name) -> float:
    """Eager float of a scalar; ``ValueError`` for traced or non-numeric input."""
    try:
        out = np.asarray(value, dtype=float)
    except Exception as exc:
        raise ValueError(f"{name} must be a concrete number, got {value!r}") from exc
    if out.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {out.shape}")
    return float(out)


def build_core(kernel, channels, reference, phase, *, index, quadrature):
    """``(I_basis, V_basis, P_basis)`` of shapes ``(n_ch, n0)``, ``(n_ch, n0)``, ``(n_ch, n2)``.

    Traced in the kernel, channel and reference leaves; ``index`` and
    ``quadrature`` are static. ``jax.jacfwd`` with respect to the reference
    leaves differentiates the whole construction. Parity masks are applied to
    the ``I`` and ``V`` columns (``eq: angular parity``). The derivative
    blocks come from :func:`taylor_blocks`.
    """
    blocks = taylor_blocks(kernel, channels, reference, phase, index, quadrature)
    n_ch = channels.n_ch
    I0, _, P0 = blocks[(0, 0)]
    real = jnp.result_type(I0, 1.0)
    I_basis = columns(
        index.h0, index, lambda r, s: blocks[(r, s)][0], n_ch, real, False
    )
    V_basis = columns(
        index.h0, index, lambda r, s: blocks[(r, s)][1], n_ch, real, False
    )
    cplx = jnp.result_type(P0, 1j)
    P_basis = columns(index.h2, index, lambda r, s: blocks[(r, s)][2], n_ch, cplx, True)
    parity_I = jnp.asarray(index.parity_I(), dtype=real)
    parity_V = jnp.asarray(index.parity_V(), dtype=real)
    return I_basis * parity_I, V_basis * parity_V, P_basis


# -- refined configurations for the numerical estimate and the route check -------------


def refined_channels(channels, factor):
    """Same channels with ``factor`` times more Gauss-Legendre nodes per support."""
    x, w = legendre_rule(factor * channels.n_nu)
    lo, hi = channels.support[:, 0], channels.support[:, 1]
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return dataclasses.replace(
        channels,
        nodes=mid[:, None] + half[:, None] * jnp.asarray(x)[None, :],
        weights=half[:, None] * jnp.asarray(w)[None, :],
        n_nu=factor * channels.n_nu,
    )


def refined(kernel, channels, route, factor, full, *, cross=False):
    """``(kernel', channels', route', description)`` for a convergence rebuild.

    Harmonic: the same angular route (``cross=True``: the other route) at
    ``factor`` times the node counts (``full`` also multiplies the Bessel
    resolution). Continuum: ``factor`` times ``n_nu`` and ``n_eta`` (``full``
    also ``n_nodes_F``). Both keep the caller's ``chunk_budget``. The continuum
    kernel has no second route, so ``cross=True`` gives
    ``None``, as do kernels without quadrature controls.
    """
    if isinstance(kernel, HarmonicKernel):
        used = kernel.quadrature if route is None else route
        target = ("tensor" if used == "product" else "product") if cross else used
        n_nodes = factor * kernel.resolution() if full else kernel.n_nodes
        alt = HarmonicKernel(
            kernel.m_max,
            n_nodes=n_nodes,
            n_outer=factor * kernel.n_outer,
            n_inner=factor * kernel.n_inner,
            n_mu=factor * kernel.n_mu,
            n_eta=factor * kernel.n_eta,
            m_chunk=kernel.m_chunk,
            chunk_budget=kernel.chunk_budget,
            tail_probe=kernel.tail_probe,
            E_phys=kernel.E_phys,
            quadrature=target,
            derivatives=kernel.derivatives,
        )
        text = f"{target} route at {factor}x angular nodes" + (
            f" and {factor}x Bessel nodes" if full else ""
        )
        return alt, channels, target, text
    if isinstance(kernel, ContinuumKernel) and not cross:
        alt = ContinuumKernel(
            n_nodes_F=factor * kernel.n_nodes_F if full else kernel.n_nodes_F,
            tail_cutoff=kernel.tail_cutoff,
            x_min=kernel.x_min,
            E_phys=kernel.E_phys,
            n_eta=factor * kernel.n_eta,
            chunk_budget=kernel.chunk_budget,
        )
        text = f"{factor}x n_nu and n_eta" + (
            f" and {factor}x n_nodes_F" if full else ""
        )
        return alt, refined_channels(channels, factor), route, text
    return None


def continuum_tail(kernel, channels, reference) -> ErrorTerm | None:
    """``F``/``G`` finite-tail bound at the reference for continuum kernels."""
    if not isinstance(kernel, ContinuumKernel):
        return None
    return kernel.tail_bound(channels, reference.gamma0, reference.B0, 0.0)


# -- response matrix and column envelopes ---------------------------------------------


def response_matrix(index, n_ch, I_basis, V_basis, P_basis):
    """``C`` of shape ``(4 n_ch, n_real)``; layout in ``syncmoments.model.basis``."""
    n0, n2 = index.n0, index.n2
    real = jnp.result_type(I_basis, 1.0)
    C = jnp.zeros((n_ch, 4, index.n_real), dtype=real)
    C = C.at[:, 0, :n0].set(I_basis)
    C = C.at[:, 3, :n0].set(V_basis)
    re, im = jnp.real(P_basis), jnp.imag(P_basis)
    C = C.at[:, 1, n0 : n0 + n2].set(re)
    C = C.at[:, 1, n0 + n2 :].set(-im)
    C = C.at[:, 2, n0 : n0 + n2].set(im)
    C = C.at[:, 2, n0 + n2 :].set(re)
    return C.reshape(4 * n_ch, index.n_real)


def on_mass_column(envelope, index, value):
    """Add a per-electron ``(n_ch, 4)`` term to the ``(0,0,0,0,0)`` column of an
    ``(4 n_ch, n_real)`` envelope; the contraction multiplies it by ``|m_0|``."""
    n_ch = value.shape[0]
    mass = index.position(0, 0, 0, 0, 0, 0)
    out = envelope.reshape(n_ch, 4, index.n_real)
    return out.at[:, :, mass].add(value).reshape(envelope.shape)


def column_ratio(delta, C) -> float:
    """``max_a max_j |dC_ja| / max_j |C_ja|`` over nonzero columns (eager).

    Invariant under ``Reference.scales``, which rescale whole columns.
    """
    delta, C = np.abs(np.asarray(delta)), np.abs(np.asarray(C))
    top = np.max(C, axis=0)
    keep = top > 0
    if not np.any(keep):
        return 0.0
    return float(np.max(np.max(delta, axis=0)[keep] / top[keep]))


# -- numerics record --------------------------------------------------------------------


def harmonic_activity(kernel, channels, reference):
    """``(m_range, cells)`` of the product cells at the reference (eager NumPy).

    ``m_range`` is ``(m_min, m_max_active)`` over the harmonics whose line can
    meet a channel support at ``(gamma0, B0)`` for some ``(mu, eta)``, or
    ``None`` when no line does (or the kernel has no harmonics); ``cells``
    counts the non-empty ``(m, channel, sign)`` cells.
    """
    if not isinstance(kernel, HarmonicKernel):
        return None, 0
    gamma0 = concrete(reference.gamma0, "reference.gamma0")
    B0 = concrete(reference.B0, "reference.B0")
    beta = math.sqrt(1.0 - 1.0 / gamma0**2)
    nu_B = E_ESU * B0 / (2.0 * math.pi * gamma0 * M_E * C_CGS)
    support = np.asarray(channels.support, dtype=float)
    m = np.arange(1, kernel.m_max + 1, dtype=float)[:, None]
    lower = np.maximum(-1.0, (1.0 - m * nu_B / support[None, :, 0]) / beta)
    upper = np.minimum(1.0, (1.0 - m * nu_B / support[None, :, 1]) / beta)
    lo_pos, hi_pos = np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
    lo_neg, hi_neg = np.clip(-upper, 0.0, 1.0), np.clip(-lower, 0.0, 1.0)
    filled = np.stack([hi_pos > lo_pos, hi_neg > lo_neg])  # (2, m, n_ch)
    cells = int(np.sum(filled))
    active = np.flatnonzero(np.any(filled, axis=(0, 2)))
    if active.size == 0:
        return None, 0
    return (int(active[0]) + 1, int(active[-1]) + 1), cells


def resolution(kernel):
    if isinstance(kernel, HarmonicKernel):
        return kernel.resolution()
    if isinstance(kernel, ContinuumKernel):
        return kernel.n_nodes_F
    return None


def numerics_record(kernel, channels, reference, route) -> tuple:
    m_range, cells = harmonic_activity(kernel, channels, reference)
    return (
        ("n_nodes", resolution(kernel)),
        ("m_range", m_range),
        ("width_ratio_min", float(np.min(channels.width_ratio()))),
        ("cells", cells),
        ("route", route),
    )


# -- screen exponent ----------------------------------------------------------------------


def screen_exponent_term(kernel, channels, reference, phase, quadrature) -> ErrorTerm:
    """Per-electron ``screen_exponent`` slot of ``eq: channel error budget``.

    Taylor route: ``not_applicable``. Gaussian and empirical screens: a zero
    ``bound`` (their characteristic functions are exact within the declared
    screen). Cumulant screen: ``unbounded`` without ``g5_bound``; otherwise the
    ``estimate`` ``<H_I>_{angles} max|w_0| (e^{eps5,j} - 1)`` in the ``Q, U``
    columns, with ``eps5,j = tau_max^5 g5/5!`` at the channel's lower support
    edge, ``<H_I>`` the angular average of the ``I`` channel kernel at the
    reference (an upper bound on ``sum_m |Q_m R_j|`` since ``|Q_m| <= I_m``,
    ``G <= F``) and ``max|w_0|`` over the channel quadrature nodes. It is an
    estimate because it is evaluated at the reference point, not averaged
    over the population.
    """
    name = getattr(phase, "name", "taylor")
    n_ch = channels.n_ch
    if name == "taylor":
        return ErrorTerm.not_applicable(
            "Taylor phase route: no screen characteristic function is truncated"
        )
    if name != "cumulant_screen":
        return ErrorTerm(
            jnp.zeros((n_ch, 4)),
            "bound",
            f"{name}: exact characteristic function of the declared screen; no "
            "exponent remainder (eq: screen exponent error does not arise)",
            "eq: screen exponent error",
        )
    tau_max = 2.0 * (C_SI_M / jnp.asarray(channels.support)[:, 0]) ** 2
    eps = phase.exponent_remainder(tau_max)
    if eps is None:
        return ErrorTerm.unbounded(
            "cumulant screen without g5_bound: exponent remainder R_5 unknown",
            "eq: screen exponent error",
        )
    average = kernel.angular_projection(
        channels,
        reference.gamma0,
        reference.B0,
        phase=None,
        truncation=Truncation(0, 0, 0),
        quadrature=quadrature,
    ).I[:, 0]
    tau_nodes = 2.0 * (C_SI_M / jnp.asarray(channels.nodes)) ** 2
    w_max = jnp.max(jnp.abs(phase(tau_nodes, depth_ref=0.0, s_depth=1.0)[0]), axis=-1)
    polarised = jnp.abs(average) * w_max * jnp.expm1(eps)
    zero = jnp.zeros(n_ch)
    return ErrorTerm(
        jnp.stack([zero, polarised, polarised, zero], axis=1),
        "estimate",
        "eq: screen exponent error at the reference point: <H_I>_angles max|w_0| "
        "(exp(tau_max^5 g5/5!) - 1) per channel; not a population average",
        "eq: screen exponent error",
    )


def bessel_nodes_for(m_max) -> int:
    return auto_nodes(m_max)


__all__ = [
    "M_REGIME",
    "package_version",
    "concrete",
    "projection_in_z",
    "taylor_tensors",
    "taylor_blocks",
    "build_core",
    "refined",
    "refined_channels",
    "continuum_tail",
    "response_matrix",
    "on_mass_column",
    "column_ratio",
    "harmonic_activity",
    "numerics_record",
    "screen_exponent_term",
]
