"""Second-degree Taylor response, supplied with deviation mean and covariance.

The first two cumulants are equivalent coordinates for the same raw moments:
<S> = S0 + dS kappa1 + ddS:(kappa2+kappa1*kappa1)/2. The omitted kernel
Taylor remainder is separate from any PDF closure; even Gaussian statistics
do not make this finite-degree kernel approximation exact. General Bell
algebra is in ``cumulants``; neither module reconstructs a positive PDF.

Derivative arrays are precomputed and stored as ordinary Equinox array leaves.
They are traced under filter_jit. They are frozen only when the caller's
partition/gradient arguments exclude them; the module does not enforce that.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .derivatives import derivative_spectra


class CumulantExpansion(eqx.Module):
    """Ensemble Stokes <S_n> from the statistics of (gamma, alpha, theta).

    Historical class name retained for compatibility. The inputs are first
    and second cumulants, and the kernel Taylor degree is two. Moments or
    cumulants with the same information give identical predictions. Gaussian
    higher cumulants vanish, but fourth and higher kernel terms remain.

    Attributes
    ----------
    harmonics : tuple of int (static)
    S0 : (N, 3) array -- zeroth-order Stokes at the reference point
        (ordinary array leaf; exclude from the differentiation arguments)
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
        """Evaluate the quadratic kernel average from mean and covariance.

        These statistics alone do not specify a physical PDF. This function
        neither checks its support nor bounds the omitted kernel terms.

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

        At fixed harmonic number, for a physical spectrum at B0, independent
        B magnitude (or fixed remaining parameters) permits the factor::

            <S> = S(B0) * [1 + 2 mu_B/B0 + (var_B + mu_B^2)/B0^2].

        ``S`` is physical Stokes evaluated at B0. A dimensionless S remains a
        dimensionless rescaled quantity; this operation does not add physical
        units. ``mu_B``/``var_B`` are statistics of B-B0. For correlated B and
        other parameters, apply conditional moments before the remaining
        average. At fixed observing frequency B also shifts the spectrum, so
        this multiplicative reduction is not generally valid.
        """
        factor = 1.0 + 2.0 * mu_B / B0 + (var_B + mu_B**2) / B0**2
        return S * factor


def build_expansion(harmonics, gamma0, alpha0, theta0, *, B=None):
    """Build a quadratic response; B [G] selects physical fixed-B derivatives.

    B=None preserves the legacy dimensionless output. The reference point and
    physical units are construction-time choices; changing either requires a
    new expansion. The public inputs are (gamma, alpha, theta); the two angles are in radians.
    """
    harmonics = tuple(harmonics)
    if not harmonics or any(int(n) != n or n < 1 for n in harmonics):
        raise ValueError("harmonics must be a nonempty sequence of positive integers")
    harmonics = tuple(int(n) for n in harmonics)
    S0s, dSs, ddSs = [], [], []
    for n in harmonics:
        val, g, h = derivative_spectra(n, gamma0, alpha0, theta0, B=B)
        S0s.append(val)
        dSs.append(g)
        ddSs.append(h)
    return CumulantExpansion(
        harmonics=harmonics,
        S0=jnp.stack(S0s),
        dS=jnp.stack(dSs),
        ddS=jnp.stack(ddSs),
    )
