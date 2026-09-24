"""Caller-supplied assumption allowances (private helper of ``synchro.model.predict``).

``assumption_allowances={name: ErrorTerm}`` replaces the budget term of a
named assumption record, basis-forced (phase-route screens, the continuum
kernel's ``isotropic_pitch``) or moment-level, in ``eq: channel error
budget``. An allowance is in Stokes units (the amplitude included, as
``bounds.screen_factorisation_bound(..., amplitude=N_src)`` returns it), with
a value that broadcasts to ``(n_ch, 4)``, and of a valued kind or
``unbounded``; the term keeps the supplied kind and value, and its note
records that the caller supplied it. An unknown name, a non-``ErrorTerm``, a
wrong shape or the kind ``not_applicable`` (the assumption does arise on
this route, so it cannot be dropped from the budget) raises ``ValueError``. Nothing here checks that the allowance holds: the inputs of
the bound that produced it are the caller's declaration.
"""

from __future__ import annotations

from collections.abc import Mapping

from ._budget_terms import stokes_shape
from .errors import ErrorTerm

NOTE = "allowance supplied by caller"


def check_allowances(allowances, names, n_ch):
    """Validated ``{name: ErrorTerm}`` for the assumption ``names`` of this budget."""
    if allowances is None:
        return {}
    if not isinstance(allowances, Mapping):
        raise ValueError("assumption_allowances must be a mapping {name: ErrorTerm}")
    known = tuple(names)
    out = {}
    for name, term in allowances.items():
        if name not in known:
            raise ValueError(
                f"unknown assumption {name!r} in assumption_allowances; the "
                f"assumptions of this budget are {known}"
            )
        if not isinstance(term, ErrorTerm):
            raise ValueError(
                f"assumption_allowances[{name!r}] must be an ErrorTerm in Stokes "
                f"units, got {type(term).__name__}"
            )
        if term.kind == "not_applicable":
            raise ValueError(
                f"assumption_allowances[{name!r}] has kind 'not_applicable'; the "
                "assumption is made on this route, so its allowance must be a "
                "valued ErrorTerm or 'unbounded'"
            )
        if term.value is not None:
            stokes_shape(term.value, n_ch, f"assumption_allowances[{name!r}]")
        note = f"{NOTE}: {term.note}" if term.note else NOTE
        out[name] = ErrorTerm(
            term.value, term.kind, note, term.manuscript_term or "E_phys"
        )
    return out


def split_assumption_terms(pairs, allowances, amplitude):
    """``(budget_pairs, per_electron_pairs, stokes_pairs)`` for ``pairs`` at unit amplitude.

    ``budget_pairs`` keeps the record order: an allowance where one was
    supplied, else the per-electron term times ``amplitude``. The other two
    feed the amplitude cross term: per-electron terms, and Stokes-unit
    allowances (named ``assumption:<name>``).
    """
    budget, per_electron, stokes = [], [], []
    for name, term in pairs:
        if name in allowances:
            budget.append((name, allowances[name]))
            stokes.append((f"assumption:{name}", allowances[name]))
        else:
            budget.append((name, term.scaled(amplitude)))
            per_electron.append((f"assumption:{name}", term))
    return tuple(budget), tuple(per_electron), tuple(stokes)


def allowance_notes(allowances):
    """Provenance notes, one per supplied allowance."""
    return tuple(
        f"assumption '{name}': {NOTE} (kind '{term.kind}') replaces its term"
        for name, term in allowances.items()
    )


__all__ = ["NOTE", "check_allowances", "split_assumption_terms", "allowance_notes"]
