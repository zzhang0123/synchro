"""Pure Faraday rotation of joint emitting populations, including mixed media.

The positive measure weights source electrons, not complex polarised flux.
Each emitting element carries its intervening depth to the observer. Absorption,
scattering, conversion and incident background light are excluded. The cold-
plasma lambda-squared phase law is assumed. Geometry enters through the paired
emission/depth inputs; neither independence nor a common intrinsic spectrum is
assumed. Channel integration must act on the combined emitted/rotated spectrum.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .rm import _screen_samples


def _finite(value, name):
    value = jnp.asarray(value)
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite")


def _real(value, name, *, scalar=False, nonnegative=False):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value) or (scalar and value.ndim != 0):
        raise ValueError(f"{name} must be real" + (" and scalar" if scalar else ""))
    value = _finite(value.astype(jnp.result_type(value, 1.0)), name)
    if nonnegative:
        value = eqx.error_if(value, jnp.any(value < 0), f"{name} must be nonnegative")
    return value


def _phase_coordinate(lam):
    lam = _real(lam, "wavelengths")
    return _finite(2 * lam**2, "Faraday phase coordinate")


def emission_polarisation(
    emission: ArrayLike,
    depths: ArrayLike,
    lam: ArrayLike,
    weights: ArrayLike | None = None,
    *,
    source_column: ArrayLike = 1.0,
) -> jax.Array:
    """Return ``source_column * E[emission * exp(2j*depth*lam**2)]``.

    ``depths``: nonempty real ``(n,)`` intervening depths in rad/m^2;
    ``lam``: real wavelengths in metres, any shape;
    ``emission``: observer-basis per-source complex Q+iU, either scalar,
    ``(n,)`` (independent of frequency), or exactly ``(n, *lam.shape)``.
    The last form retains spectral variation correlated with depth. Output
    shape is ``lam.shape``. Supply the true source column as a finite nonnegative
    scalar; the default gives the per-source average. Nonnegative relative
    electron-number/quadrature weights are normalized internally. They must
    have positive total mass, even when source_column=0. No emissivity or complex
    phase factor is absorbed into or renormalized as a probability weight.

    Inputs must be finite. Supports JIT and differentiation. A scan avoids an
    extra emitter-by-frequency phase array; reverse AD can retain intermediates.
    The supplied discrete sum has no statistical truncation, but population/path
    quadrature, numerical and physical-model errors remain external inputs.
    In particular, finite phases need not have accurate argument reduction.
    For fixed measure and column N, paired perturbations obey::

        |delta P| <= N*(E[|delta emission|] + 2*lam^2*E[|emission|*|delta depth|]),

    using the reference emission in the second term. Source-column and weight
    uncertainties require their additional normalization terms.
    """
    depths, w = _screen_samples(depths, weights, sample_name="Faraday-depth")
    t = _phase_coordinate(lam)
    emission = _finite(emission, "emission")
    if emission.ndim == 0:
        emission = jnp.broadcast_to(emission, depths.shape)
    if emission.shape not in (depths.shape, (depths.size, *t.shape)):
        raise ValueError("emission must be scalar, (n,), or (n, *lam.shape)")
    column = _real(source_column, "source_column", scalar=True, nonnegative=True)
    dtype = jnp.result_type(emission, depths, t, 1j)

    def accumulate(total, element):
        depth, amplitude, weight = element
        phase = _finite(t * depth, "Faraday phase")
        return total + weight * amplitude * jnp.exp(1j * phase), None

    average, _ = jax.lax.scan(
        accumulate, jnp.zeros(t.shape, dtype=dtype), (depths, emission, w)
    )
    return _finite(column * average, "emission-depth average")


def joint_faraday_moments(
    basis_values: ArrayLike,
    depths: ArrayLike,
    degree: int,
    weights: ArrayLike | None = None,
    *,
    reference_depth: ArrayLike = 0.0,
) -> tuple[jax.Array, jax.Array]:
    """Collect ``M[a,b]=E[psi_a*(depth-reference_depth)**b]``, b=0..degree.

    ``basis_values`` is finite ``(n, n_basis)`` (real or complex), with at least
    one basis function. Basis functions have no frequency dependence. Depths
    and relative electron-number weights follow ``emission_polarisation``.
    ``degree`` is a nonnegative static integer; ``reference_depth`` is a finite
    real scalar in rad/m^2. Use filter_jit or mark degree static in jax.jit.

    Returns the ``(n_basis, degree+1)`` moments and ``(n_basis,)`` absolute next
    moments ``E[|psi_a|*|depth-reference_depth|**(degree+1)]``. A complex zeroth
    moment may vanish; it is never divided out. These are statistics of the
    supplied discrete measure, not a continuous-population certificate. The
    next absolute moment is additional information, not inferred from the
    retained moments or a Gaussian/independent closure. JIT/AD are supported.
    """
    if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
        raise ValueError("degree must be a nonnegative static integer")
    depths, w = _screen_samples(depths, weights, sample_name="Faraday-depth")
    basis = _finite(basis_values, "basis_values")
    if basis.ndim != 2 or basis.shape[0] != depths.size or basis.shape[1] == 0:
        raise ValueError("basis_values must have shape (n, n_basis) with n_basis>0")
    reference = _real(reference_depth, "reference_depth", scalar=True)
    delta = _finite(depths - reference, "depth deviations")
    weighted_basis = w[:, None] * basis

    def collect(power, _):
        moment = jnp.sum(weighted_basis * power[:, None], axis=0)
        return power * delta, moment

    next_power, moments = jax.lax.scan(
        collect, jnp.ones_like(delta), None, length=degree + 1
    )
    absolute_next = jnp.sum(
        w[:, None] * jnp.abs(basis) * jnp.abs(next_power[:, None]), axis=0
    )
    return _finite(moments.T, "joint moments"), _finite(
        absolute_next, "absolute next moments"
    )


def joint_faraday_average(
    coefficients: ArrayLike,
    moments: ArrayLike,
    lam: ArrayLike,
    *,
    absolute_next: ArrayLike,
    source_error: ArrayLike,
    reference_depth: ArrayLike = 0.0,
    source_column: ArrayLike = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Finite joint source/depth response and an absolute truncation envelope.

    Represent the per-source intrinsic kernel as sum_a c_a(lam)*psi_a + r.
    ``coefficients`` must have shape ``(*lam.shape, n_basis)``; ``moments`` has
    shape ``(n_basis, degree+1)`` and is normalized as in joint_faraday_moments.
    It may be fitted directly. No samples or conditional coefficient functions
    are needed in this contraction. The same moments serve every wavelength.

    Required error inputs: ``absolute_next`` is a nonnegative ``(n_basis,)``
    upper envelope for ``E[|psi_a|*|depth-reference_depth|**(degree+1)]``;
    ``source_error`` is a nonnegative scalar or ``lam.shape`` envelope for ``E|r|``,
    both per source under the same positive measure. They must be externally
    justified for the represented population; setting source_error=0 asserts
    an exact intrinsic basis. Return (prediction, error), both ``lam.shape``::

        P = N*exp(it*reference)*sum_ab c_a*(it)^b/b!*M_ab,
        error = N*(source_error + |t|^(degree+1)/(degree+1)!
                   * sum_a |c_a|*absolute_next_a), t=2*lam^2.

    This is a Taylor approximation to the real Faraday phase, not a truncated
    cumulant exponent. Large phases can require high degree or direct averaging.
    Moment realizability, support/tail assumptions, statistical-input uncertainty,
    numerical error and physical-model discrepancy are not certified here.
    Integrate the emitted/rotated prediction and the absolute-response-weighted
    envelope together for a finite channel; rotating a channel average is not
    equivalent. JIT and differentiation are supported; degree is set by shape.
    """
    t = _phase_coordinate(lam)
    reference = _real(reference_depth, "reference_depth", scalar=True)
    column = _real(source_column, "source_column", scalar=True, nonnegative=True)
    moments = _finite(moments, "moments")
    if moments.ndim != 2 or min(moments.shape) == 0:
        raise ValueError("moments must have nonempty shape (n_basis, degree+1)")
    coefficients = _finite(coefficients, "coefficients")
    if coefficients.shape != (*t.shape, moments.shape[0]):
        raise ValueError("coefficients must have shape (*lam.shape, n_basis)")
    absolute_next = _real(absolute_next, "absolute_next", nonnegative=True)
    if absolute_next.shape != (moments.shape[0],):
        raise ValueError("absolute_next must have shape (n_basis,)")
    source_error = _real(source_error, "source_error", nonnegative=True)
    if source_error.ndim != 0 and source_error.shape != t.shape:
        raise ValueError("source_error must be scalar or have lam.shape")

    def phase_term(value, index):
        return value * (1j * t) / (index + 1), value

    next_term, phase_terms = jax.lax.scan(
        phase_term,
        jnp.ones(t.shape, dtype=jnp.result_type(t, 1j)),
        jnp.arange(moments.shape[1]),
    )
    polynomial = jnp.sum(
        (coefficients @ moments) * jnp.moveaxis(phase_terms, 0, -1), axis=-1
    )
    phase = _finite(t * reference, "reference Faraday phase")
    prediction = column * jnp.exp(1j * phase) * polynomial
    error = column * (
        source_error + jnp.abs(next_term) * (jnp.abs(coefficients) @ absolute_next)
    )
    return _finite(prediction, "finite Faraday response"), _finite(
        error, "Faraday error envelope"
    )
