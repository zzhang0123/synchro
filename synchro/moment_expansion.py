"""Moment expansion of the ensemble Stokes parameters (JAX + Equinox).

The single-particle Stokes S_n(p) are Taylor-expanded around a reference point
p0.  The electron/field population then enters only through its moments.  With
p = (gamma, alpha, theta) and B, phi treated separately (B is a multiplicative
factor w_B^2 ~ B^2; phi only rotates Q,U on the sky), the *corrected*
second-order expansion in raw moments is

    <S_n> = S_n(p0)
          + sum_i  S_n^{(i)}       mu_i
          + (1/2) sum_ij S_n^{(ij)} (Sigma_ij + mu_i mu_j) + ...

where mu_i = <dp_i> and Sigma_ij = <dp_i dp_j> - mu_i mu_j are the first
moments and covariance of the distribution, and S_n^{(i)}, S_n^{(ij)} are the
derivative spectra precomputed by :mod:`synchro.derivatives`.

The derivative spectra depend only on p0 (not on the moments), so they are
precomputed once and bound into the module as *traced, non-differentiated*
array leaves.  The online evaluation is a cheap tensor contraction, jittable
and differentiable w.r.t. the moments.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .derivatives import N_PARAM, STOKES, derivative_spectra
from .stokes import E_ESU, M_E, C_CGS


class MomentExpansion(eqx.Module):
    """Ensemble Stokes <S_n> from moments of (gamma, alpha, theta).

    Attributes
    ----------
    harmonics : tuple of int (static)
    S0 : (N, 3) array -- zeroth-order Stokes at the reference point
        (traced, not differentiated)
    dS : (N, 3, P) array -- first derivative spectra
    ddS : (N, 3, P, P) array -- second derivative spectra

    The reference point (gamma0, alpha0, theta0) is implicit in S0/dS/ddS; it is
    not stored as a field (so it never enters the jit cache or the gradient).
    """

    harmonics: tuple = eqx.field(static=True)
    S0: jax.Array
    dS: jax.Array
    ddS: jax.Array

    def __call__(self, mu, cov):
        """Evaluate <S> at harmonic grid.

        Parameters
        ----------
        mu : (P,) array
            First moments <dp_i>.
        cov : (P, P) array
            Covariance <dp_i dp_j> - mu_i mu_j (must be symmetric).

        Returns
        -------
        (N, 3) array of ensemble Stokes (I, Q, V) per harmonic.
        """
        raw = cov + mu[:, None] * mu[None, :]  # Sigma_ij + mu_i mu_j
        t1 = jnp.einsum("nsi,i->ns", self.dS, mu)
        t2 = 0.5 * jnp.einsum("nsij,ij->ns", self.ddS, raw)
        return self.S0 + t1 + t2

    def apply_B(self, S, B0, mu_B=0.0, var_B=0.0):
        """Fold in the magnetic-field magnitude moments exactly.

        Since S_n ~ B^2, S_n(B) = S_n(B0) (B/B0)^2 and the ensemble average is
        exact at every order::

            <S> = S(B0) * [1 + 2 mu_B/B0 + (var_B + mu_B^2)/B0^2].

        ``S``: (N, 3) Stokes array from ``__call__``; ``mu_B``/``var_B`` are
        the mean and variance of d B = B - B0.
        """
        factor = 1.0 + 2.0 * mu_B / B0 + (var_B + mu_B**2) / B0**2
        return S * factor


def build_expansion(harmonics, gamma0, alpha0, theta0):
    """Precompute derivative spectra and return a :class:`MomentExpansion`."""
    harmonics = tuple(int(n) for n in harmonics)
    S0s, dSs, ddSs = [], [], []
    for n in harmonics:
        val, g, h = derivative_spectra(n, gamma0, alpha0, theta0)
        S0s.append(val)
        dSs.append(g)
        ddSs.append(h)
    return MomentExpansion(
        harmonics=harmonics,
        S0=jnp.stack(S0s),
        dS=jnp.stack(dSs),
        ddS=jnp.stack(ddSs),
    )
