"""Membership of concrete population samples in a declared ``Support`` (private).

``required_m_max`` (``kernels.py``) and hence the zero harmonic-tail bound of
``HarmonicKernel.truncation_error`` assume that every emitting electron has
``gamma <= support.gamma[1]`` and ``B >= support.B[0]``;
``Support.truncated=False`` declares the population complete inside all three
intervals. :func:`outside_support` compares concrete samples with those
intervals. Samples with zero normalised weight or ``B = 0`` (no emission)
are ignored. A relative slack of ``REL_TOL`` of the interval scale absorbs
roundoff on the edges; anything beyond it counts as outside. Traced samples
or a traced support cannot be checked eagerly: the result is then ``None``
and callers must state the hypothesis as unchecked (``UNCHECKED`` for the
harmonic tail, ``UNCHECKED_COMPLETE`` for a declared-complete excluded
tail). Units: ``gamma``
dimensionless, ``B`` Gauss, ``depth`` rad/m^2.
"""

from __future__ import annotations

import jax
import numpy as np

REL_TOL = 1e-12
UNCHECKED = (
    "conditional on every emitting electron lying inside the declared Support "
    "(gamma <= support.gamma[1], B >= support.B[0]); not checked"
)
UNCHECKED_COMPLETE = (
    "conditional on every emitting electron lying inside all three declared "
    "Support intervals (gamma, B, depth); not checked (traced samples or support)"
)


def _concrete(value):
    """Float NumPy array of a concrete value, ``None`` for a tracer."""
    if isinstance(value, jax.core.Tracer):
        return None
    return np.asarray(value, dtype=float)


def _interval(pair):
    lo, hi = (_concrete(v) for v in pair)
    if lo is None or hi is None:
        return None
    return float(lo), float(hi)


def _emitting(samples):
    """Mask of samples with positive normalised weight and ``B > 0`` (or ``None``)."""
    weights = _concrete(samples.normalised_weights())
    B = _concrete(samples.B)
    if weights is None or B is None:
        return None
    return (weights > 0) & (B > 0)


def outside_support(samples, support, *, harmonic_only=False):
    """Descriptions of the variables whose emitting samples leave ``support``.

    Returns ``None`` when the samples or the support are traced (unchecked),
    ``()`` when every emitting sample lies inside, and otherwise one string
    per violating variable, ``"<name> in [min, max] vs declared [lo, hi]"``
    over the emitting samples. ``harmonic_only=True`` checks only the two
    sides that enter ``required_m_max`` (``gamma`` above ``hi``, ``B`` below
    ``lo``); otherwise ``gamma``, ``B`` and ``depth`` are checked on both
    sides.
    """
    if samples is None or support is None:
        return None
    mask = _emitting(samples)
    if mask is None:
        return None
    if not np.any(mask):
        return ()
    checks = (("gamma", False, True), ("B", True, False))
    if not harmonic_only:
        checks = (("gamma", True, True), ("B", True, True), ("depth", True, True))
    found = []
    for name, check_lo, check_hi in checks:
        values = _concrete(getattr(samples, name))
        bounds = _interval(getattr(support, name))
        if values is None or bounds is None:
            return None
        lo, hi = bounds
        values = values[mask]
        slack = REL_TOL * max(abs(lo), abs(hi))
        low = check_lo and bool(np.any(values < lo - slack))
        high = check_hi and bool(np.any(values > hi + slack))
        if low or high:
            found.append(
                f"{name} in [{np.min(values):.6g}, {np.max(values):.6g}] vs "
                f"declared [{lo:.6g}, {hi:.6g}]"
            )
    return tuple(found)


def hypothesis_note(samples, outside):
    """The support hypothesis behind a zero harmonic-tail bound, as note text."""
    if samples is None:
        return UNCHECKED + " here (no samples supplied)"
    if outside is None:
        return UNCHECKED + " (traced samples or support)"
    return "the supplied emitting samples were checked inside the declared Support"


__all__ = [
    "REL_TOL",
    "UNCHECKED",
    "UNCHECKED_COMPLETE",
    "outside_support",
    "hypothesis_note",
]
