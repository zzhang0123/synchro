"""Unconstrained coordinates of a ``ParameterMap`` (``syncmoments.model.fit._transform``).

Private helper of :mod:`syncmoments.model.fit.nonlinear`, which re-exports
:class:`Transform`. See that module's docstring for the coordinate
conventions (identity for ``log_amplitude`` and tables, log for positive
hyper-parameters, tanh box for bounded slots, identity for nodal logits
with the additive-log-ratio Jacobian in :meth:`Transform.log_det`).
"""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..assumptions import ParameterMap, Parameters

POSITIVE_HYPER = ("sigma", "kappa_2")
AMPLITUDE_LABEL = "log_amplitude"
KINDS = ("identity", "log", "box", "logit")


def _is_positive_hyper(label):
    return label.startswith("hyper:") and label.rsplit(":", 1)[-1] in POSITIVE_HYPER


def _check_bounds(bounds, labels):
    """Return ``{label: (lo, hi)}`` with finite ``lo < hi`` (``ValueError`` otherwise)."""
    if bounds is None:
        return {}
    items = bounds.items() if isinstance(bounds, dict) else tuple(bounds)
    out = {}
    for label, pair in items:
        if label not in labels:
            raise ValueError(f"bounds names an unknown parameter {label!r}")
        try:
            lo, hi = (float(v) for v in pair)
        except (TypeError, ValueError):
            raise ValueError(f"bounds[{label!r}] must be a (lo, hi) pair") from None
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"bounds[{label!r}] needs finite lo < hi")
        out[label] = (lo, hi)
    return out


def _flatten(parameter_map, theta):
    """``ParameterMap.flatten`` with static (NumPy) masks, safe under ``jit``.

    Same layout as :meth:`ParameterMap.flatten`: per free group the real
    parts, then the imaginary parts of the ``e^{2i phi}`` entries; then the
    fitted hyper arrays; nodal maps give the logits. The public method
    indexes with a traced boolean mask, which ``jax.jit`` rejects.
    """
    parameter_map.flatten  # noqa: B018  (raises ValueError on a bad theta)
    if parameter_map.nodes is not None:
        return parameter_map.flatten(theta)
    parts = []
    for j, g in enumerate(parameter_map.free_groups()):
        table = jnp.asarray(theta.tables[j])
        parts.append(table.real)
        mask = np.asarray(parameter_map._complex_mask(g), dtype=bool)
        if mask.any():
            parts.append(table.imag[np.flatnonzero(mask)])
    parts.extend(jnp.asarray(h, dtype=float) for h in theta.hyper)
    return jnp.concatenate(parts) if parts else jnp.zeros(0)


class Transform(eqx.Module):
    """Unconstrained coordinates ``z`` of a parameter map (see the module docstring).

    ``z`` is ``(n_params,)`` with ``n_params = n_free + 1`` when ``amplitude``
    (slot 0 is ``log A``), followed by the map's :meth:`ParameterMap.flatten`
    order. ``kinds`` and ``bounds`` are static per-slot tuples; ``labels``
    names every slot. ``forward`` needs ``theta.log_amplitude`` when the
    transform carries an amplitude (``ValueError`` otherwise). ``z`` is
    dimensionless (logs of positive quantities, tanh boxes, identity
    elsewhere). Assumes only the positivity/box declarations it was built
    with; nothing certified about the fit that uses it.
    """

    LABEL: ClassVar[str] = "extra eq: finite fit model"
    parameter_map: ParameterMap
    amplitude: bool = eqx.field(static=True)
    kinds: tuple[str, ...] = eqx.field(static=True)
    bounds: tuple = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, parameter_map, *, amplitude=True, bounds=None, positive=None):
        if not isinstance(parameter_map, ParameterMap):
            raise ValueError("parameter_map must be a ParameterMap")
        if not isinstance(amplitude, bool):
            raise ValueError("amplitude must be a bool")
        labels = tuple(parameter_map.labels())
        if amplitude:
            labels = (AMPLITUDE_LABEL, *labels)
        boxed = _check_bounds(bounds, labels)
        if positive is None:
            positive = tuple(lab for lab in labels if _is_positive_hyper(lab))
        positive = tuple(positive)
        unknown = [lab for lab in positive if lab not in labels]
        if unknown:
            raise ValueError(f"positive names unknown parameters {unknown}")
        kinds, pairs = [], []
        nodal_map = parameter_map.nodes is not None
        for lab in labels:
            if lab in positive and lab in boxed:
                raise ValueError(f"{lab!r} cannot be both positive (log) and boxed")
            if lab in positive:
                kinds.append("log")
            elif lab in boxed:
                kinds.append("box")
            elif nodal_map and lab.startswith("logit["):
                kinds.append("logit")
            else:
                kinds.append("identity")
            pairs.append(boxed.get(lab))
        self.parameter_map = parameter_map
        self.amplitude = amplitude
        self.kinds = tuple(kinds)
        self.bounds = tuple(pairs)
        self.labels = labels

    @property
    def n_params(self) -> int:
        return len(self.labels)

    def _masks(self):
        """Static NumPy masks ``(is_log, is_box, is_logit, lo, hi)`` per slot."""
        kinds = np.asarray(self.kinds)
        lo = np.array([0.0 if p is None else p[0] for p in self.bounds])
        hi = np.array([1.0 if p is None else p[1] for p in self.bounds])
        return kinds == "log", kinds == "box", kinds == "logit", lo, hi

    def _check_z(self, z):
        z = jnp.asarray(z)
        if jnp.iscomplexobj(z) or z.shape != (self.n_params,):
            raise ValueError(f"z must be a real vector of shape ({self.n_params},)")
        return z.astype(jnp.result_type(z, 1.0))

    def forward(self, theta: Parameters) -> jax.Array:
        """``theta -> z`` (``(n_params,)``); ``error_if`` on infeasible values."""
        v = _flatten(self.parameter_map, theta)
        if self.amplitude:
            if theta.log_amplitude is None:
                raise ValueError("theta.log_amplitude is required by this transform")
            v = jnp.concatenate([jnp.reshape(theta.log_amplitude, (1,)), v])
        is_log, is_box, _, lo, hi = self._masks()
        v = eqx.error_if(
            v, jnp.any(is_log & (v <= 0.0)), "log-transformed parameters must be > 0"
        )
        v = eqx.error_if(
            v,
            jnp.any(is_box & ((v <= lo) | (v >= hi))),
            "boxed parameters must lie strictly inside their bounds",
        )
        safe_log = jnp.log(jnp.where(is_log, v, 1.0))
        u = jnp.clip((v - lo) / (hi - lo), 1e-300, 1.0)
        safe_box = jnp.arctanh(jnp.where(is_box, 2.0 * u - 1.0, 0.0))
        return jnp.where(is_log, safe_log, jnp.where(is_box, safe_box, v))

    def constrained(self, z) -> jax.Array:
        """The constrained coordinates ``(A, tables, positive hyper, ...)`` of ``z``.

        Identity and logit slots are returned as is (the simplex weights are
        not materialised; their Jacobian enters :meth:`log_det` only).
        """
        z = self._check_z(z)
        is_log, is_box, _, lo, hi = self._masks()
        box = lo + (hi - lo) * 0.5 * (1.0 + jnp.tanh(z))
        return jnp.where(is_log, jnp.exp(z), jnp.where(is_box, box, z))

    def inverse(self, z) -> Parameters:
        """``z -> theta``."""
        v = self.constrained(z)
        z = self._check_z(z)
        log_amplitude = None
        if self.amplitude:
            log_amplitude, v = z[0], v[1:]
        theta = self.parameter_map.unflatten(v)
        return Parameters(
            log_amplitude=log_amplitude,
            tables=theta.tables,
            hyper=theta.hyper,
            logits=theta.logits,
        )

    def log_det(self, z) -> jax.Array:
        """``log |d(constrained)/dz|`` (scalar); see the module docstring."""
        z = self._check_z(z)
        is_log, is_box, is_logit, lo, hi = self._masks()
        box = jnp.log(0.5 * (hi - lo)) + 2.0 * jnp.log(2.0) - 2.0 * jnp.logaddexp(z, -z)
        per_slot = jnp.where(is_log, z, jnp.where(is_box, box, 0.0))
        total = jnp.sum(per_slot)
        if "logit" in self.kinds:
            logits = z[is_logit]
            total = total + jnp.sum(jax.nn.log_softmax(logits))
        return total


__all__ = ["Transform", "POSITIVE_HYPER", "AMPLITUDE_LABEL", "KINDS"]
