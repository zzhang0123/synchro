"""Option checks, relation validation and text of ``reduce_response`` (private).

Split out of :mod:`syncmoments.model.fit.reduction` to keep that file under
400 lines. Nothing here is public API; the contracts are stated in the
``reduction`` module docstring.
"""

from __future__ import annotations

import math

import numpy as np

from . import _relations as rel

RELATIONS = ("auto", "continuum", "none")
CALLER_SOURCES = ("declared", "approximate")
OUT_OF_SCOPE = (
    "the continuum scaling identity K(gamma/sqrt(lambda), lambda B) = lambda K "
    "holds only for ContinuumKernel (K = B Phi(B gamma^2)); for this kernel no "
    "analytic relation is claimed"
)


def validate_relation(relation, n_full, *, caller, relation_type):
    if not isinstance(relation, relation_type):
        raise ValueError("declared relations must be LinearRelation instances")
    allowed = CALLER_SOURCES if caller else rel.SOURCES
    if relation.source not in allowed:
        raise ValueError(f"relation source must be one of {allowed}")
    if not isinstance(relation.scope, str) or not relation.scope.strip():
        raise ValueError("a relation needs a non-empty scope statement")
    try:
        slots = [j for j, _ in relation.weights]
        weights = np.asarray([w for _, w in relation.weights], dtype=float)
    except (TypeError, ValueError):
        raise ValueError("relation weights must be ((slot, weight), ...)") from None
    if not slots or any(
        isinstance(j, bool)
        or not isinstance(j, (int, np.integer))
        or not 0 <= j < n_full
        for j in slots
    ):
        raise ValueError(f"relation slots must be integers in [0, {n_full})")
    if len(set(slots)) != len(slots):
        raise ValueError("relation slots must be unique")
    if not np.all(np.isfinite(weights)) or not np.any(weights != 0.0):
        raise ValueError("relation weights must be finite and not all zero")
    return tuple((int(j), float(w)) for j, w in relation.weights), relation.source


def check_options(relations, representatives, include_ext, check_rtol):
    if relations not in RELATIONS:
        raise ValueError(f"relations must be one of {RELATIONS}")
    if not isinstance(include_ext, bool):
        raise ValueError("include_ext must be a bool")
    try:
        check_rtol = float(check_rtol)
    except (TypeError, ValueError):
        raise ValueError("check_rtol must be a float") from None
    if not (math.isfinite(check_rtol) and check_rtol >= 0.0):
        raise ValueError("check_rtol must be finite and nonnegative")
    if representatives != "gamma":
        triples = tuple(representatives) if isinstance(representatives, tuple) else None
        ok = triples is not None and all(
            isinstance(t, tuple)
            and len(t) == 3
            and all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in t
            )
            for t in triples
        )
        if not ok or len(set(triples)) != len(triples):
            raise ValueError(
                "representatives must be 'gamma' or a tuple of distinct (r, s, b) triples"
            )
    return check_rtol


def record(source):
    return {
        "structural": "identically zero column, checked exactly",
        "analytic": "derived (continuum scaling identity), checked to check_rtol",
        "declared": "declared by caller, checked to check_rtol, then treated as exact",
        "approximate": "approximate: delta_C kept and propagated",
    }[source]


def generated(layout, relations, kernel, notes):
    out = rel.structural_relations(layout)
    if relations == "none":
        return out
    if kernel == "continuum":
        return out + rel.continuum_relations(layout)
    reason = f"kernel {kernel!r}: {OUT_OF_SCOPE}"
    if relations == "continuum":
        raise ValueError(f"relations='continuum' refused: {reason}")
    notes.append(f"relations='auto': structural zeros only; {reason}")
    return out


def formula(row, labels):
    terms = [(w, labels[j]) for j, w in enumerate(row) if w != 0.0]
    text = " + ".join(
        f"{w:.6g} a[{name}]" if w != 1.0 else f"a[{name}]" for w, name in terms
    )
    return text.replace("+ -", "- ")


def scope_text(layout, kernel, applied):
    sources = sorted({r.source for r in applied})
    text = (
        f"kernel {kernel!r}; sources {sources}. Structural zeros are exact; analytic "
        "relations follow from the continuum scaling identity for ContinuumKernel only"
    )
    if "analytic" in sources:
        eps_g, eps_B = rel.fractional_scales(layout)
        text += f" (eps_g = {eps_g:.6g}, eps_B = {eps_B:.6g})"
    return (
        text + "; declared relations are caller statements; approximate relations "
        "keep delta_C. The numerical rank and the practical recoverability are "
        "decided later, by fit_combinations, for a stated noise model and metric."
    )


__all__ = [
    "RELATIONS",
    "CALLER_SOURCES",
    "OUT_OF_SCOPE",
    "validate_relation",
    "check_options",
    "record",
    "generated",
    "formula",
    "scope_text",
]
