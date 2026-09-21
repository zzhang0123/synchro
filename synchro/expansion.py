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

    def checked_average(self, mean, covariance, *, absolute_error):
        """Validate statistics and return ``(prediction, absolute_error)``.

        This opt-in fitting boundary accepts real finite (P,) means and (P,P)
        covariances. Symmetry and PSD checks use a relative floating-point
        tolerance of 32*P*eps in correlation coordinates; negative variances
        and nonzero covariance with a zero-variance coordinate are rejected
        outright. No eigenvalues are clipped. Zero covariance is valid. The low-level
        ``__call__`` remains available without this eigensolver overhead.

        ``absolute_error`` is a required, finite nonnegative envelope in the
        output's units, broadcastable to (N,3). It is supplied by the caller,
        NOT inferred from covariance validity. It must cover the omitted
        kernel terms and any other model errors the caller claims to control.
        Physical support and the validity of this envelope are not certified
        here; zero is justified only for an exact quadratic response or other
        independently established zero remainder. Numerical roundoff is not
        bounded by this routine. For a later linear map A, propagate this
        componentwise envelope as abs(A) @ envelope.
        """
        mean, covariance = jnp.asarray(mean), jnp.asarray(covariance)
        p = self.dS.shape[-1]
        if mean.shape != (p,) or covariance.shape != (p, p):
            raise ValueError("mean and covariance must have shapes (P,) and (P,P)")
        if jnp.iscomplexobj(mean) or jnp.iscomplexobj(covariance):
            raise ValueError("mean and covariance must be real")
        dtype = jnp.result_type(mean, covariance, 1.0)
        mean, covariance = mean.astype(dtype), covariance.astype(dtype)
        mean = eqx.error_if(mean, jnp.any(~jnp.isfinite(mean)), "mean must be finite")
        covariance = eqx.error_if(
            covariance, jnp.any(~jnp.isfinite(covariance)), "covariance must be finite"
        )
        diagonal = jnp.diag(covariance)
        zero_pair = (diagonal[:, None] == 0) | (diagonal[None, :] == 0)
        covariance = eqx.error_if(
            covariance,
            jnp.any(diagonal < 0) | jnp.any(zero_pair & (covariance != 0)),
            "covariance requires nonnegative variances and zero covariance for zero variance",
        )
        # Congruence scaling makes the check insensitive to parameter units.
        # A large independent variance must not hide an indefinite small block.
        std = jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1.0))
        scaled = covariance / std[:, None] / std[None, :]
        tol = 32 * p * jnp.finfo(dtype).eps
        scaled = eqx.error_if(
            scaled,
            jnp.any(~jnp.isfinite(scaled)) | jnp.any(jnp.abs(scaled) > 1 + tol),
            "covariance violates the Cauchy-Schwarz bound",
        )
        invalid = (jnp.max(jnp.abs(scaled - scaled.T)) > tol) | (
            jnp.min(jnp.linalg.eigvalsh((scaled + scaled.T) / 2)) < -tol
        )
        covariance = eqx.error_if(
            covariance, invalid, "covariance must be symmetric positive semidefinite"
        )
        prediction = self(mean, covariance)
        prediction = eqx.error_if(
            prediction,
            jnp.any(~jnp.isfinite(prediction)),
            "quadratic response overflowed",
        )
        error = jnp.asarray(absolute_error)
        if jnp.iscomplexobj(error):
            raise ValueError("absolute_error must be real")
        error = jnp.broadcast_to(error, prediction.shape)
        error = eqx.error_if(
            error,
            jnp.any(~jnp.isfinite(error)) | jnp.any(error < 0),
            "absolute_error must be finite and nonnegative",
        )
        return prediction, error


# Descriptive name without breaking existing Equinox trees or imports.
QuadraticTaylorExpansion = CumulantExpansion


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
