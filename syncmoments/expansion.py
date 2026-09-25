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
from .rm import _normalised_weights


def mixed_moments(offsets, factor, weights=None):
    """Return E[W], E[W*dq], E[W*dq*dq] for a supplied joint population.

    ``offsets`` is a nonempty real (samples, P) array of deviations from the
    kernel reference; ``factor`` is a matching real or complex (samples,)
    array W. Optional nonnegative relative electron-number weights are
    normalized as in ``syncmoments.rm`` (in float64 by exact powers of two; a
    largest weight below 2^-1022 is refused), but W is NOT normalized. For fixed-harmonic field averaging,
    use W=(B/B0)^2 and W=(B/B0)^2*exp(2j*phi), with B0>0 and phi in radians.
    Keep every sample's q, B and phi paired to preserve their correlations.
    The result is raw mixed moments, not a mean/covariance under W weights.

    Shapes are (), (P,), (P,P). JIT/autodiff are supported. The routine checks
    finite arrays and weights, not physical support or sampling/quadrature
    accuracy. These finite statistics do not infer a joint PDF or a remainder.
    """
    offsets, factor = jnp.asarray(offsets), jnp.asarray(factor)
    if offsets.ndim != 2 or min(offsets.shape) == 0:
        raise ValueError("offsets must have nonempty shape (samples,P)")
    if factor.shape != (offsets.shape[0],):
        raise ValueError("factor must have shape (samples,)")
    if jnp.iscomplexobj(offsets):
        raise ValueError("offsets must be real")
    offsets = offsets.astype(jnp.result_type(offsets, 1.0))
    factor = factor.astype(jnp.result_type(factor, 1.0))
    offsets = eqx.error_if(
        offsets, jnp.any(~jnp.isfinite(offsets)), "offsets must be finite"
    )
    factor = eqx.error_if(
        factor, jnp.any(~jnp.isfinite(factor)), "factor must be finite"
    )
    if weights is None:
        w = jnp.ones(offsets.shape[0], dtype=offsets.dtype) / offsets.shape[0]
    else:
        w = jnp.asarray(weights)
        if w.shape != factor.shape or jnp.iscomplexobj(w):
            raise ValueError("weights must be real with shape (samples,)")
        w = w.astype(jnp.result_type(w, offsets, 1.0))
        # Shared validation and normalisation (float64, tangent headroom).
        w = _normalised_weights(w).astype(w.dtype)
    wf = w * factor
    moments = (
        jnp.sum(wf),
        jnp.einsum("n,ni->i", wf, offsets),
        jnp.einsum("n,ni,nj->ij", wf, offsets, offsets),
    )
    return tuple(
        eqx.error_if(m, jnp.any(~jnp.isfinite(m)), "mixed moment arithmetic overflowed")
        for m in moments
    )


class CumulantExpansion(eqx.Module):
    """Ensemble Stokes <S_n> from the statistics of (gamma, alpha, theta).

    Historical class name retained for compatibility. The inputs are first
    and second cumulants, and the kernel Taylor degree is two. Moments or
    cumulants with the same information give identical predictions. Gaussian
    higher cumulants vanish, but fourth and higher kernel terms remain.

    Shapes: ``N`` harmonics, three Stokes components, ``P`` parameters.

    Attributes
    ----------
    harmonics : tuple of int
        Harmonic numbers (static field).
    S0 : jax.Array
        Zeroth-order Stokes at the reference point, shape ``(N, 3)``
        (ordinary array leaf; exclude from the differentiation arguments).
    dS : jax.Array
        First derivative spectra, shape ``(N, 3, P)``.
    ddS : jax.Array
        Second derivative spectra, shape ``(N, 3, P, P)``.

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
        other parameters, use ``mixed_average`` with joint mixed moments, or
        explicitly integrate conditional moments before the remaining average.
        At fixed observing frequency B also shifts the spectrum, so
        this multiplicative reduction is not generally valid.
        """
        factor = 1.0 + 2.0 * mu_B / B0 + (var_B + mu_B**2) / B0**2
        return S * factor

    def mixed_average(self, field_moments, phase_moments, *, absolute_error):
        """Average correlated B/q/phi using exact factors and a quadratic kernel.

        Build the natural-basis response at fixed physical B0>0. Its q
        coordinates are (gamma, alpha, theta), NOT (gamma, cos(alpha), theta).
        ``field_moments`` and ``phase_moments`` each contain three raw mixed
        moments with shapes (), (P,), (P,P): E[W], E[W*dq], E[W*dq*dq].
        Use real W=(B/B0)^2 for field_moments, and complex
        W=(B/B0)^2*exp(2j*phi) for phase_moments. Both use the same normalized
        electron-number distribution; ``mixed_moments`` can compute them from
        paired samples. Zero field is allowed. Do not divide by E[W], which
        would discard amplitude and can be zero for the complex factor.

        Returns (prediction, absolute_error), both (N,4), in sky (I,Q,U,V)
        order. Exact field/phase handling preserves correlations, but only the
        remaining q-kernel is quadratic. In all original variables the inputs
        include higher mixed orders, not just a global covariance. Physical
        units come from the reference response, never from this contraction.

        The caller must supply a componentwise nonnegative error envelope.
        For a pointwise natural-kernel remainder ``|R_s(q)| <= rho_s(q)``, the I,V
        errors are bounded by E[(B/B0)^2*rho_s]; the same Q-kernel bound applies
        separately to sky Q and U. Uncertain mixed moments add the absolute
        coefficient contraction with their error envelopes. This API checks
        shapes/finiteness, NOT joint-moment realizability, physical support,
        envelope validity, sampling error or roundoff. It neither differentiates
        unknown conditional moments nor infers them. A caller-created
        dimensionless response stays dimensionless. Fixed-frequency channels
        need their moving-line kernel and cannot use this B^2 factorisation.
        """
        p = self.dS.shape[-1]

        def contract(moments, real):
            if len(moments) != 3:
                raise ValueError("supply three mixed moments of degree 0,1,2")
            arrays = tuple(jnp.asarray(m) for m in moments)
            if tuple(m.shape for m in arrays) != ((), (p,), (p, p)):
                raise ValueError("mixed moment shapes must be (), (P,), (P,P)")
            if real and any(jnp.iscomplexobj(m) for m in arrays):
                raise ValueError("field moments must be real")
            arrays = tuple(
                eqx.error_if(
                    m, jnp.any(~jnp.isfinite(m)), "mixed moments must be finite"
                )
                for m in arrays
            )
            m0, m1, m2 = arrays
            return (
                self.S0 * m0
                + jnp.einsum("nsi,i->ns", self.dS, m1)
                + 0.5 * jnp.einsum("nsij,ij->ns", self.ddS, m2)
            )

        natural = contract(field_moments, real=True)
        linear = contract(phase_moments, real=False)[:, 1]
        prediction = jnp.stack(
            [natural[:, 0], linear.real, linear.imag, natural[:, 2]], axis=-1
        )
        prediction = eqx.error_if(
            prediction, jnp.any(~jnp.isfinite(prediction)), "mixed response overflowed"
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
