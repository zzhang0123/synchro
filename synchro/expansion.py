"""Ensemble Stokes expansion: raw moments (2nd order) and cumulants (general).

The single-particle Stokes S_n(p) are Taylor-expanded around a reference point
p0.  The electron/field population then enters through its distribution's
statistics.  Two equivalent-to-second-order but differently-truncated forms
are implemented:

* the *raw-moment* (Taylor) expansion (Sec 4.1 of the paper), and
* the *cumulant* expansion (Sec 4.2), via :mod:`synchro.cumulants`.

This module implements the second-order form, which the two coincide on:

    <S_n> = S_n(p0)
          + sum_i  S_n^{(i)}       kappa_i
          + (1/2) sum_ij S_n^{(ij)} (kappa_ij + kappa_i kappa_j) + ...

with kappa_i = <dp_i>, kappa_ij = <dp_i dp_j> - <dp_i><dp_j> the first two
cumulants of dp = p - p0 (equivalently the mean and covariance), and
S_n^{(i)}, S_n^{(ij)} the derivative spectra from :mod:`synchro.derivatives`.
The general-order cumulant truncation (where the two formalisms differ) lives
in :mod:`synchro.cumulants`; a systematic moment-vs-cumulant comparison is the
natural follow-up.

The derivative spectra depend only on p0 (not on the statistics), so they are
precomputed once and bound into the module as *traced, non-differentiated*
array leaves.  The online evaluation is a cheap tensor contraction, jittable
and differentiable w.r.t. the cumulants.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .derivatives import N_PARAM, STOKES, derivative_spectra
from .stokes import E_ESU, M_E, C_CGS


class CumulantExpansion(eqx.Module):
    """Ensemble Stokes <S_n> from the statistics of (gamma, alpha, theta).

    The interface is raw moments -- ``__call__(mu, cov)`` takes the mean and
    covariance, which is what an observer measures -- but the expansion is
    *organised* in cumulants, which is what makes the truncation principled:
    for a Gaussian the K=2 cumulant truncation is exact to all Taylor orders,
    whereas the raw-moment series is not.  Hence the name.

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

    def __call__(self, kappa1, kappa2):
        """Evaluate <S> at the harmonic grid via the strict cumulant expansion.

        This is the cumulant expansion truncated at order 2 (kappa_k = 0 for
        k >= 3), which is EXACT for a Gaussian joint distribution of the
        parameters and for independent parameters (whose cross-cumulants
        vanish) — see synchro.cumulants for the general-order machinery and the
        Gaussian-exactness demonstration.

        Parameters
        ----------
        kappa1 : (P,) array
            First cumulants <dp_i> (the deviation means; zero if the reference
            point is the mean).
        kappa2 : (P, P) array
            Second cumulants <dp_i dp_j> - <dp_i><dp_j> (the covariance).

        Returns
        -------
        (N, 3) array of ensemble Stokes (I, Q, V) per harmonic.
        """
        # <S> = S0 + S^(i) kappa1_i + (1/2) S^(ij) (kappa2_ij + kappa1_i kappa1_j)
        raw2 = kappa2 + kappa1[:, None] * kappa1[None, :]
        t1 = jnp.einsum("nsi,i->ns", self.dS, kappa1)
        t2 = 0.5 * jnp.einsum("nsij,ij->ns", self.ddS, raw2)
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
    """Precompute derivative spectra and return a :class:`CumulantExpansion`."""
    harmonics = tuple(int(n) for n in harmonics)
    S0s, dSs, ddSs = [], [], []
    for n in harmonics:
        val, g, h = derivative_spectra(n, gamma0, alpha0, theta0)
        S0s.append(val)
        dSs.append(g)
        ddSs.append(h)
    return CumulantExpansion(
        harmonics=harmonics,
        S0=jnp.stack(S0s),
        dS=jnp.stack(dSs),
        ddS=jnp.stack(ddSs),
    )
