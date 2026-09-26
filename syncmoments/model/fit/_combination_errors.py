"""Error terms of ``fit_combinations`` and the observable and truth builders (private).

With ``d = R C a + delta + n`` and the fitted model ``R H T a``, the
estimator satisfies ``x_hat - Pi x = K (delta + R delta_C a) + K n`` exactly.
Four terms are kept apart (design Section 7): the declared discrepancy
(``bias = |K| delta``), unresolved directions (``|I - Pi| x_bound`` under a
declared coefficient bound, else ``unbounded``, ``not_applicable`` when
``Pi = I``), the approximate reduction (``|K| (|R| |delta_C| a_bound)_kept``)
and, in synthetic studies only, the measured bias. None is reported as zero
when it is unconstrained. Nothing here is public API.
"""

from __future__ import annotations

import numpy as np

from ..errors import WEAKEST_ORDER, ErrorTerm
from ._discrepancy import data_discrepancy

PI_RTOL = 1e-12


def discrepancy_term(data, stokes_hat):
    """``delta`` over the kept rows as an ``ErrorTerm`` (``unbounded`` with the reason)."""
    delta, kind, note, reason = data_discrepancy(data, stokes_hat)
    if delta is None:
        return ErrorTerm.unbounded(reason, "data discrepancy delta (kept rows)")
    return ErrorTerm(
        np.asarray(delta), kind, note, "data discrepancy delta (kept rows)"
    )


def propagated(gain, term, target, what):
    """``|gain| value`` with the term's kind; ``unbounded``/``not_applicable`` pass through."""
    if term.kind == "not_applicable":
        return ErrorTerm.not_applicable(term.note)
    if term.value is None:
        return ErrorTerm.unbounded(f"{what}: {term.note}", target)
    value = np.abs(np.asarray(gain)) @ np.asarray(term.value)
    return ErrorTerm(value, term.kind, f"|gain| {what}: {term.note}", target)


def column_selection(J):
    """Selected slots when every column of ``J`` is a distinct unit vector, else ``None``."""
    J = np.asarray(J)
    slots = []
    for col in J.T:
        nz = np.flatnonzero(col)
        if nz.size != 1 or col[nz[0]] != 1.0:
            return None
        slots.append(int(nz[0]))
    return slots if len(set(slots)) == len(slots) else None


def unresolved_term(Jx, Pi_complement, J, bound, target, *, check_scale=None):
    """``|Jx (I - Pi)| x_bound``; ``not_applicable`` when ``Jx (I - Pi)`` vanishes (numerical check)."""
    D = np.asarray(Jx) @ Pi_complement
    scale = (
        float(np.max(np.abs(Jx), initial=0.0)) if check_scale is None else check_scale
    )
    if float(np.max(np.abs(D), initial=0.0)) <= PI_RTOL * max(scale, 1e-300):
        return ErrorTerm.not_applicable(
            "every direction it depends on is resolved (numerical check |J (I - Pi)| "
            f"<= {PI_RTOL:g} max|J|)"
        )
    if bound is None or bound.value is None:
        why = (
            "no coefficient_bound"
            if bound is None
            else f"coefficient_bound {bound.kind}"
        )
        return ErrorTerm.unbounded(
            f"unresolved directions are not constrained by the data and {why} was "
            "given: this part of the error is unbounded, not zero",
            target,
        )
    slots = column_selection(J)
    if slots is None:
        return ErrorTerm.unbounded(
            "the coordinates x are not a selection of full slots, so the coefficient "
            "bound does not bound x: unresolved part unbounded",
            target,
        )
    x_bound = np.asarray(bound.value)[slots]
    return ErrorTerm(
        np.abs(D) @ x_bound,
        bound.kind,
        f"|J (I - Pi)| x_bound under the declared coefficient bound ({bound.note}); "
        "not a data constraint",
        target,
    )


def reduction_envelope(reduction, data, bound):
    """``(|R| |delta_C| a_bound)`` on the kept rows, or ``not_applicable``/``unbounded``."""
    target = "approximate-reduction discrepancy (kept rows)"
    if reduction.exact:
        return ErrorTerm.not_applicable(
            "exact reduction (structural, analytic and declared relations): "
            f"max|delta_C|/max|C| = {reduction.max_delta_C:.3g} is a finite rounding "
            "check, not propagated; declared relations are caller statements"
        )
    if bound is None or bound.value is None:
        return ErrorTerm.unbounded(
            "approximate relations without a coefficient_bound: R delta_C a is not "
            "bounded",
            target,
        )
    env = np.abs(np.asarray(reduction.delta_C)) @ np.asarray(bound.value)
    response = getattr(data, "response", None)
    if response is not None:
        env = np.abs(np.asarray(response)) @ env
    return ErrorTerm(
        np.asarray(data.select(env)),
        bound.kind,
        f"|R| |delta_C| a_bound under the declared coefficient bound ({bound.note})",
        target,
    )


def combine(terms, target):
    """Sum of the valued terms with the weakest kind; ``unbounded`` if any term is."""
    live = [t for t in terms if t.kind != "not_applicable"]
    if not live:
        return ErrorTerm.not_applicable("every term is not_applicable")
    missing = [t for t in live if t.value is None]
    if missing:
        return ErrorTerm.unbounded(
            "at least one term is unbounded: " + "; ".join(t.note for t in missing),
            target,
        )
    kind = max((t.kind for t in live), key=WEAKEST_ORDER.index)
    value = sum(np.asarray(t.value) for t in live)
    return ErrorTerm(value, kind, "sum of " + ", ".join(t.kind for t in live), target)


def data_space(fit_data, value, name):
    """Kept rows of a Stokes ``(n_ch, 4)`` array or a data vector ``(n_data,)`` (mapped by ``R``)."""
    value = np.asarray(value, dtype=float)
    n_ch = fit_data.n_ch
    response = getattr(fit_data, "response", None)
    if value.shape == (n_ch, 4):
        flat = value.reshape(-1)
        full = flat if response is None else np.asarray(response) @ flat
    elif value.shape == (fit_data.n_data(),):
        full = value
    elif value.shape == (fit_data.n_kept(),):
        return value
    else:
        raise ValueError(
            f"{name} must be ({n_ch}, 4), ({fit_data.n_data()},) or "
            f"({fit_data.n_kept()},), got {value.shape}"
        )
    return np.asarray(fit_data.select(full))


__all__ = [
    "discrepancy_term",
    "propagated",
    "column_selection",
    "unresolved_term",
    "reduction_envelope",
    "combine",
    "data_space",
]
