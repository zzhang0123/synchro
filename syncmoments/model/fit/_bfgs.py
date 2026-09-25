"""BFGS helpers of :func:`syncmoments.model.fit.nonlinear.fit_bfgs` (private).

Split out of ``nonlinear.py`` to keep that file under 400 lines, which
re-exports every name here. :func:`gauss_newton_scale` is the diagonal
preconditioner, :func:`at_precision_floor` the relative stopping test and
:func:`total_objective` the minimised ``-log density`` made total: a trial
point the density cannot evaluate (a non-finite coordinate, a log slot
whose ``exp`` would overflow, or a non-finite density) gives ``+inf``, which
the line search rejects, instead of reaching the value checks
(``equinox.error_if``) of the map. Nothing here certifies the optimum.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

PRECISION_FLOOR = 1e-12
# Curvature floor of the preconditioner, relative to the largest diagonal
# entry: caps the scale ratio at 1e3, so zero-curvature slots at a degenerate
# start (all tables zero, say) do not get steps 1e6 times the smallest.
SCALE_FLOOR = 1e-6
# ``exp(700) ~ 1e304`` is below the float64 maximum; a log slot beyond it
# would give an infinite constrained value (an infinite sigma, say).
LOG_SLOT_LIMIT = 700.0


def gauss_newton_scale(logdensity, z0):
    """Diagonal preconditioner ``s`` with ``diag(J^T J) s^2 ~ 1`` at ``z0``.

    ``J`` is the Jacobian of the whitened residual (``jax.jacfwd``); columns
    with curvature below ``SCALE_FLOOR`` of the largest are clamped there, so
    unidentified parameters keep a finite scale at most ``1e3`` times the
    smallest.
    """
    J = jax.jacfwd(logdensity.whitened_residual)(z0)
    diag = jnp.sum(J * J, axis=0)
    floor = SCALE_FLOOR * jnp.maximum(jnp.max(diag), 1e-300)
    return 1.0 / jnp.sqrt(jnp.maximum(diag, floor))


def at_precision_floor(result) -> bool:
    """Line search stopped with no resolvable decrease left.

    True when the run ended in a line-search failure (status ``>= 2``) and
    the quasi-Newton predicted decrease ``0.5 g^T H g`` is below
    ``PRECISION_FLOOR * max(1, |f|)``: the objective is converged to that
    relative level under the BFGS curvature model.
    """
    if int(result.status) < 2:
        return False
    g, H = jnp.asarray(result.jac), jnp.asarray(result.hess_inv)
    decrease = 0.5 * float(g @ (H @ g))
    return decrease <= PRECISION_FLOOR * max(1.0, abs(float(result.fun)))


def total_objective(logdensity, scale):
    """``y -> -logdensity(scale * y)``, ``+inf`` where it cannot be evaluated.

    The density is evaluated at a finite, clipped copy of ``z = scale * y``
    (non-finite entries set to zero, log slots clipped to
    ``|z| <= LOG_SLOT_LIMIT``), so no value check fires; the result is
    ``+inf`` unless ``z`` needed no change and the density is finite. Inside
    the evaluable region it equals ``-logdensity(z)`` exactly.
    """
    is_log = np.asarray(logdensity.transform.kinds) == "log"

    def objective(y):
        z = scale * y
        finite = jnp.isfinite(z)
        in_range = jnp.where(is_log, jnp.abs(z) <= LOG_SLOT_LIMIT, True)
        z_safe = jnp.where(finite, z, 0.0)
        z_safe = jnp.where(
            is_log, jnp.clip(z_safe, -LOG_SLOT_LIMIT, LOG_SLOT_LIMIT), z_safe
        )
        value = -logdensity(z_safe)
        ok = jnp.all(finite & in_range) & jnp.isfinite(value)
        return jnp.where(ok, value, jnp.inf)

    return objective


def accepted_point(objective, y_start, result):
    """The point kept after one ``minimize`` run.

    ``result.x`` when the run reports success. Otherwise the lowest finite
    objective among ``result.x``, the pre-step point ``result.x + H g`` and
    the run's start ``y_start`` (ties go to that order; ``y_start`` when
    none is finite): when its zoom line search fails, JAX's BFGS still
    applies the unit step ``-H g`` and reports the objective of the
    pre-step point, so ``result.x`` can be a rejected (``+inf``) trial point.
    """
    x = jnp.asarray(result.x)
    if bool(result.success):
        return x
    back = x + jnp.asarray(result.hess_inv) @ jnp.asarray(result.jac)
    best, best_value = y_start, float(objective(y_start))
    for candidate in (back, x):
        value = float(objective(candidate))
        if np.isfinite(value) and not value > best_value:
            best, best_value = candidate, value
    return best


__all__ = [
    "PRECISION_FLOOR",
    "SCALE_FLOOR",
    "LOG_SLOT_LIMIT",
    "gauss_newton_scale",
    "at_precision_floor",
    "total_objective",
    "accepted_point",
]
