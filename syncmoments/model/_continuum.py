"""Ultra-relativistic continuum kernel (private; re-exported by ``kernels``).

LABEL: ``eq: directional continuum``, ``extra eq: channel basis integration``.
See :class:`ContinuumKernel` for units, assumptions and what is not certified.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..bessel import _modified_tail_bound
from ..constants import C_CGS, E_ESU, M_E
from ..ultrarel import F, F_tail_bound, G
from ._harmonic_cells import CHUNK_BUDGET, static_int, sum_point_blocks
from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    legendre_table,
    phase_coordinate,
    phase_weights,
    resolve_index,
    sine_from_cosine,
)
from .errors import ErrorTerm

# Above this argument F and G are below 1e-300 (F ~ sqrt(x) exp(-x)); clamping
# keeps the K-integral finite when B_perp -> 0 sends x -> inf.
_X_CLAMP = 1e4


class ContinuumKernel(eqx.Module):
    """Ultra-relativistic isotropic-pitch continuum kernel (``eq: directional continuum``).

    ``K_I(nu) = A_theta F(x)``, ``K_Q(nu) = -A_theta G(x)``, ``K_V = 0``, with
    ``B_perp = B sqrt(1 - eta^2)``, ``A_theta = sqrt(3) e^3 B_perp / (4 pi m_e c^2)``
    [erg/s/Hz/sr per electron], ``x = nu / (a_B gamma^2)``,
    ``a_B = 3 e B_perp / (4 pi m_e c)``. ``mu`` is ignored (the kernel is
    already pitch-averaged), so ``components = ("I", "Q")``,
    ``required_closures = ("uniform_mu",)``, and the ``l > 0`` Legendre
    projections vanish. ``channel_modes`` sums ``w_n R_j(nu_n) K_S(nu_n)
    w_b(tau_n)`` over the channel's Gauss-Legendre nodes (``extra eq: channel
    basis integration``); units erg/s/sr per electron for ``unit_peak``.

    ``x < x_min`` at a node with ``B_perp > 0`` is an ``equinox.error_if``
    (``F`` has a divergent slope at zero); ``B_perp = 0`` gives zero kernel.
    ``x`` is clamped at 1e4 where ``F, G < 1e-300``. ``n_nodes_F`` and
    ``tail_cutoff`` are the static ``F``/``G`` quadrature controls of
    ``syncmoments.ultrarel``; :meth:`tail_bound` reports the finite-tail bound of
    that quadrature for the ``numerical`` slot. ``n_eta`` (static, keyword)
    is the Gauss-Legendre count of the ``eta`` projection.

    Memory: ``chunk_budget`` (static, keyword, default ``CHUNK_BUDGET =
    2^20``) bounds the ``F``/``G`` cosh-integral values of one ``lax.map``
    step: ``angular_projection`` runs its ``eta`` nodes in blocks of
    :meth:`eta_block` (``2 n_ch n_nu n_nodes_F`` values per node), and
    :meth:`samples_per_step` caps the samples of a caller that vmaps
    ``channel_modes`` (``direct_channel_average``). One ``eta`` node or one
    sample is the floor. Forward tangents multiply the working set; nested
    ``jacfwd`` callers divide the budget with :meth:`for_tangents`
    (``build_basis`` does). Blocking changes only the summation order.

    Not certified: the discrepancy from the harmonic reference (``physical_error``
    is ``unbounded`` unless ``E_phys`` is supplied; main.tex sec: ultrarelativistic
    states no uniform bound), the channel quadrature error (``n_nu`` of the
    channels) and the ``F``/``G`` discretisation error (only the tail is bounded).
    """

    n_nodes_F: int = eqx.field(static=True)
    tail_cutoff: float = eqx.field(static=True)
    x_min: float = eqx.field(static=True)
    E_phys: ErrorTerm | None
    n_eta: int = eqx.field(static=True)
    chunk_budget: int = eqx.field(static=True)

    name: ClassVar[str] = "continuum"
    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: directional continuum",
        "extra eq: channel basis integration",
    )
    components: ClassVar[tuple[str, ...]] = ("I", "Q")
    required_closures: ClassVar[tuple[str, ...]] = ("uniform_mu",)

    def __init__(
        self,
        *,
        n_nodes_F=128,
        tail_cutoff=50.0,
        x_min=1e-6,
        E_phys=None,
        n_eta=48,
        chunk_budget=CHUNK_BUDGET,
    ):
        for name, value in (("n_nodes_F", n_nodes_F), ("n_eta", n_eta)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be a static integer")
        if n_nodes_F < 16 or n_eta < 2:
            raise ValueError("n_nodes_F >= 16 and n_eta >= 2 are required")
        if not (float(tail_cutoff) > 0.0) or not (float(x_min) > 0.0):
            raise ValueError("tail_cutoff and x_min must be positive")
        if E_phys is not None and not isinstance(E_phys, ErrorTerm):
            raise ValueError("E_phys must be an ErrorTerm or None")
        self.n_nodes_F = int(n_nodes_F)
        self.tail_cutoff = float(tail_cutoff)
        self.x_min = float(x_min)
        self.E_phys = E_phys
        self.n_eta = int(n_eta)
        self.chunk_budget = static_int(chunk_budget, "chunk_budget", 1)

    # -- working-set budget -----------------------------------------------------

    def _values_per_point(self, channels) -> int:
        """``F`` and ``G`` integrand values of one ``(gamma, B, eta)`` point."""
        return 2 * int(np.prod(np.shape(channels.nodes))) * self.n_nodes_F

    def eta_block(self, channels) -> int:
        """``eta`` nodes per ``lax.map`` step of ``angular_projection`` (static)."""
        per_eta = self._values_per_point(channels)
        return max(1, min(self.n_eta, self.chunk_budget // per_eta))

    def samples_per_step(self, channels, requested: int) -> int:
        """Samples per step of a ``vmap`` of ``channel_modes`` within ``chunk_budget``."""
        per_sample = self._values_per_point(channels)
        return max(1, min(int(requested), self.chunk_budget // per_sample))

    def for_tangents(self, tangents: int) -> "ContinuumKernel":
        """Copy with ``chunk_budget // tangents`` (at least one) for nested
        ``jacfwd`` callers; summation order only."""
        budget = max(1, self.chunk_budget // max(1, int(tangents)))
        return dataclasses.replace(self, chunk_budget=budget)

    # -- kernel -----------------------------------------------------------------

    def _geometry(self, gamma, B, eta):
        b_perp = B * sine_from_cosine(eta)
        positive = b_perp > 0.0
        safe = jnp.where(positive, b_perp, 1.0)
        amplitude = jnp.sqrt(3.0) * E_ESU**3 * safe / (4.0 * jnp.pi * M_E * C_CGS**2)
        a_B = 3.0 * E_ESU * safe / (4.0 * jnp.pi * M_E * C_CGS)
        return positive, amplitude, a_B * gamma**2

    def _argument(self, nu, positive, scale):
        x = jnp.asarray(nu) / scale
        x = eqx.error_if(
            x,
            jnp.any(positive & (x < self.x_min)),
            "continuum kernel argument x = nu/(a_B gamma^2) below x_min",
        )
        return jnp.where(positive, jnp.minimum(x, _X_CLAMP), 1.0)

    def kernels(self, nu, gamma, B, eta) -> tuple[jax.Array, jax.Array]:
        """``(K_I, K_Q)`` [erg/s/Hz/sr per electron] at frequencies ``nu`` (Hz)."""
        positive, amplitude, scale = self._geometry(gamma, B, eta)
        x = self._argument(nu, positive, scale)
        f = F(x, n_nodes=self.n_nodes_F, tail_cutoff=self.tail_cutoff)
        g = G(x, n_nodes=self.n_nodes_F, tail_cutoff=self.tail_cutoff)
        zero = jnp.zeros_like(x)
        return (
            jnp.where(positive, amplitude * f, zero),
            jnp.where(positive, -amplitude * g, zero),
        )

    def _response(self, channels):
        nodes = jnp.asarray(channels.nodes)
        full = channels(nodes)  # (n_ch, n_ch, n_nu)
        own = jnp.diagonal(full, axis1=0, axis2=1).T
        return nodes, jnp.asarray(channels.weights), own

    def channel_modes(
        self, channels, gamma, B, mu, eta, *, phase=None, depth_ref=0.0, s_depth=1.0
    ):
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        nodes, weights, response = self._response(channels)
        k_I, k_Q = self.kernels(nodes, gamma, B, eta)
        measure = weights * response
        w = phase_weights(phase, phase_coordinate(nodes), depth_ref, s_depth)
        return Modes(
            I=jnp.sum(measure * k_I, axis=-1),
            V=jnp.zeros(channels.n_ch, dtype=measure.dtype),
            P=jnp.einsum("cn,wcn->wc", measure * k_Q, w),
        )

    def angular_projection(
        self,
        channels,
        gamma,
        B,
        *,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        truncation,
        quadrature=None,
    ):
        if quadrature not in (None, "tensor", "product"):
            raise ValueError("quadrature must be None, 'tensor' or 'product'")
        index = resolve_index(truncation, self.components)
        gamma, B, _, _ = check_scalar_point(gamma, B, 0.0, 0.0)
        eta_nodes, eta_weights = np.polynomial.legendre.leggauss(self.n_eta)
        eta_nodes, eta_weights = jnp.asarray(eta_nodes), jnp.asarray(eta_weights)
        nodes, weights, response = self._response(channels)
        measure = weights * response  # (n_ch, n_nu)
        w = phase_weights(phase, phase_coordinate(nodes), depth_ref, s_depth)

        def at_eta(eta):
            k_I, k_Q = self.kernels(nodes, gamma, B, eta)
            return jnp.sum(measure * k_I, axis=-1), jnp.einsum(
                "cn,wcn->wc", measure * k_Q, w
            )

        L_mu, L_eta = index.truncation.L_mu, index.truncation.L_eta
        legendre = legendre_table(eta_nodes, L_eta)  # (n_eta, L_eta+1)
        # (2k+1)/2 int K P_k deta; the mu projection is delta_{l0} (kernel is mu-free).
        k_weights = eta_weights[:, None] * legendre * (2 * jnp.arange(L_eta + 1) + 1)
        k_weights = k_weights / 2.0

        def block(args, mask):  # eta nodes of one lax.map step, reduced over eta
            I_eta, P_eta = jax.vmap(at_eta)(args[0])  # (b, n_ch), (b, n_w, n_ch)
            kw = args[1] * mask[:, None]
            return jnp.einsum("ec,ek->ck", I_eta, kw), jnp.einsum(
                "ewc,ek->wck", P_eta, kw
            )

        I_k, P_k = sum_point_blocks(
            block, (eta_nodes, k_weights), self.eta_block(channels)
        )
        l_delta = (jnp.arange(L_mu + 1) == 0).astype(I_k.dtype)
        grid_I = I_k[:, None, :] * l_delta[None, :, None]
        grid_P = P_k[:, :, None, :] * l_delta[None, None, :, None]
        I_proj = gather_pairs(grid_I, index.pairs)
        return ProjectedModes(
            I=I_proj, V=jnp.zeros_like(I_proj), P=gather_pairs(grid_P, index.pairs)
        )

    def tail_bound(self, channels, gamma, B, eta) -> ErrorTerm:
        """Finite-tail bound of the ``F``/``G`` quadrature, ``(n_ch, 4)`` per electron.

        Columns ``I, Q, U, V`` carry ``sum_n w_n |R_j| A F_tail``, the ``G``
        analogue twice and zero. Excludes discretisation and roundoff error.
        """
        gamma, B, _, eta = check_scalar_point(gamma, B, 0.0, eta)
        nodes, weights, response = self._response(channels)
        positive, amplitude, scale = self._geometry(gamma, B, eta)
        x = self._argument(nodes, positive, scale)
        f_tail = F_tail_bound(x, tail_cutoff=self.tail_cutoff)
        g_tail = x * _modified_tail_bound(2.0 / 3.0, x, tail_cutoff=self.tail_cutoff)
        measure = weights * jnp.abs(response) * amplitude
        f_col = jnp.where(positive, jnp.sum(measure * f_tail, axis=-1), 0.0)
        g_col = jnp.where(positive, jnp.sum(measure * g_tail, axis=-1), 0.0)
        value = jnp.stack([f_col, g_col, g_col, jnp.zeros_like(f_col)], axis=-1)
        return ErrorTerm(
            value,
            "bound",
            "finite upper limit of the F/G cosh integrals (syncmoments.ultrarel); "
            "excludes quadrature discretisation and roundoff",
            "E_num",
        )

    def physical_error(self, channels) -> ErrorTerm:
        if self.E_phys is not None:
            return self.E_phys
        return ErrorTerm.unbounded(
            "ultra-relativistic isotropic-pitch continuum replaces the harmonic "
            "reference; no uniform discrepancy bound (main.tex sec: ultrarelativistic)",
            "E_phys",
        )

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ):
        return ErrorTerm.not_applicable(
            "continuum kernel has no harmonic sum; the F/G tail bound is reported "
            "by tail_bound()"
        )

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("label", self.LABEL),
            ("n_nodes_F", self.n_nodes_F),
            ("tail_cutoff", self.tail_cutoff),
            ("x_min", self.x_min),
            ("n_eta", self.n_eta),
            ("chunk_budget", self.chunk_budget),
            ("E_phys", None if self.E_phys is None else self.E_phys.kind),
        )


__all__ = ["ContinuumKernel"]
