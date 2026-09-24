"""Polynomial test kernel (private; re-exported by ``kernels``). ``[extension, test oracle]``."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    legendre_table,
    monomials,
    phase_coordinate,
    phase_weights,
    resolve_index,
)
from .errors import ErrorTerm


class PolynomialTestKernel(eqx.Module):
    """``[extension, test oracle]`` finite Legendre-times-polynomial kernel.

    ``K_S(gamma, B, mu, eta) = sum c[S, j, l, k, r, s] z_gamma^r z_B^s P_l(mu)
    P_k(eta)`` with ``z = ((gamma - gamma0)/s_gamma, (B - B0)/s_B)``, ``S`` in
    ``(I, Q, V)``, and ``P`` weighted by ``w_b(tau_j)`` at the fixed line
    ``line_nu_hz[j]``. ``channels`` is ignored apart from its channel count.
    The Legendre projection is exact (``angular_projection`` returns the
    coefficient polynomial in ``z``), so it tests index bookkeeping, factorials
    and the ``C`` layout independently of Bessel numerics. Units are whatever the
    coefficients carry. Not a physical model and not certified as one; ``physical_error`` is a declared
    zero and ``truncation_error`` is ``not_applicable``.
    """

    coefficients: jax.Array
    line_nu_hz: jax.Array
    gamma0: jax.Array
    B0: jax.Array
    s_gamma: jax.Array
    s_B: jax.Array

    name: ClassVar[str] = "polynomial_test"
    LABEL: ClassVar[tuple[str, ...]] = ("[extension]", "test oracle")
    components: ClassVar[tuple[str, ...]] = ("I", "Q", "V")
    required_closures: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self, coefficients, line_nu_hz, *, gamma0=1.0, B0=1.0, s_gamma=1.0, s_B=1.0
    ):
        coefficients = jnp.asarray(coefficients, dtype=float)
        line_nu_hz = jnp.asarray(line_nu_hz, dtype=float)
        if coefficients.ndim != 6 or coefficients.shape[0] != 3:
            raise ValueError(
                "coefficients must have shape (3, n_ch, L_mu+1, L_eta+1, N_g+1, N_B+1)"
            )
        if line_nu_hz.shape != (coefficients.shape[1],):
            raise ValueError("line_nu_hz must have shape (n_ch,)")
        self.coefficients = coefficients
        self.line_nu_hz = line_nu_hz
        self.gamma0 = jnp.asarray(gamma0, dtype=float)
        self.B0 = jnp.asarray(B0, dtype=float)
        self.s_gamma = jnp.asarray(s_gamma, dtype=float)
        self.s_B = jnp.asarray(s_B, dtype=float)

    @property
    def n_ch(self) -> int:
        return self.coefficients.shape[1]

    def _check(self, channels):
        if channels is not None and getattr(channels, "n_ch", self.n_ch) != self.n_ch:
            raise ValueError("channels.n_ch must match the coefficient table")

    def _powers(self, gamma, B):
        z_g = (gamma - self.gamma0) / self.s_gamma
        z_B = (B - self.B0) / self.s_B
        n_g, n_B = self.coefficients.shape[4], self.coefficients.shape[5]
        return monomials(z_g, n_g), monomials(z_B, n_B)

    def channel_modes(
        self, channels, gamma, B, mu, eta, *, phase=None, depth_ref=0.0, s_depth=1.0
    ):
        self._check(channels)
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        pg, pb = self._powers(gamma, B)
        p_l = legendre_table(mu, self.coefficients.shape[2] - 1)
        p_k = legendre_table(eta, self.coefficients.shape[3] - 1)
        values = jnp.einsum("sjlkrt,l,k,r,t->sj", self.coefficients, p_l, p_k, pg, pb)
        w = phase_weights(phase, phase_coordinate(self.line_nu_hz), depth_ref, s_depth)
        return Modes(I=values[0], V=values[2], P=values[1][None, :] * w)

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
        self._check(channels)
        index = resolve_index(truncation, self.components)
        gamma, B, _, _ = check_scalar_point(gamma, B, 0.0, 0.0)
        pg, pb = self._powers(gamma, B)
        poly = jnp.einsum("sjlkrt,r,t->sjlk", self.coefficients, pg, pb)
        L_mu, L_eta = index.truncation.L_mu, index.truncation.L_eta
        lm, le = min(poly.shape[2], L_mu + 1), min(poly.shape[3], L_eta + 1)
        grid = jnp.zeros((3, self.n_ch, L_mu + 1, L_eta + 1), dtype=poly.dtype)
        grid = grid.at[:, :, :lm, :le].set(poly[:, :, :lm, :le])
        w = phase_weights(phase, phase_coordinate(self.line_nu_hz), depth_ref, s_depth)
        modes = gather_pairs(grid, index.pairs)  # (3, n_ch, n_lk)
        return ProjectedModes(I=modes[0], V=modes[2], P=modes[1][None] * w[:, :, None])

    def physical_error(self, channels) -> ErrorTerm:
        return ErrorTerm.declared_zero(
            "polynomial test kernel is its own reference [extension]",
            shape=(self.n_ch, 4),
        )

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ):
        return ErrorTerm.not_applicable("polynomial test kernel has no harmonic sum")

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("label", self.LABEL),
            ("shape", tuple(int(n) for n in self.coefficients.shape)),
        )


__all__ = ["PolynomialTestKernel"]
