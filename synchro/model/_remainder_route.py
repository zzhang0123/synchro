"""Basis-route checks of ``RemainderInputs.basis_remainder`` (private helper of ``bounds``).

``nonsmooth_reason`` returns why the Taylor remainder of a basis is
unconstrained (a line kernel whose channels have smoothness below
``N + 1``; ``_basis_checks.check_kernel`` records the same condition as a
provenance note). ``is_screen_route`` tells whether the basis phase route is
a screen, whose finite response does not Taylor-expand the depth.
"""

from __future__ import annotations

from .phase import TaylorPhase

NONSMOOTH_PREFIX = "basis_remainder unbounded"
LINE_KERNELS = ("harmonic",)


def _record(pairs) -> dict:
    try:
        return dict(pairs)
    except (TypeError, ValueError):
        return {}


def nonsmooth_reason(basis) -> str | None:
    """The unbounded-remainder reason of ``basis``, or ``None`` when none applies."""
    provenance = getattr(basis, "provenance", None)
    notes = getattr(provenance, "notes", ()) or ()
    for note in notes:
        if note.startswith(NONSMOOTH_PREFIX):
            return note
    kernel = _record(getattr(provenance, "kernel", ()) or ())
    if kernel.get("name") not in LINE_KERNELS:
        return None
    smoothness = basis.channels.smoothness
    N = basis.index.truncation.N
    if smoothness < N + 1:
        return (
            f"{NONSMOOTH_PREFIX}: channel smoothness {smoothness} < N + 1 = {N + 1}: "
            "derivatives of a Dirac-line kernel through the channel edges are not "
            "defined and the basis remainder is unbounded"
        )
    return None


def is_screen_route(basis) -> bool:
    """``True`` when the basis phase route is a screen (not ``TaylorPhase``).

    Reads ``basis.phase`` when set, else the ``route`` entry of
    ``provenance.phase_route``; an unrecorded route counts as Taylor, which
    keeps the depth-Taylor term.
    """
    phase = getattr(basis, "phase", None)
    if phase is not None:
        return not isinstance(phase, TaylorPhase)
    provenance = getattr(basis, "provenance", None)
    route = _record(getattr(provenance, "phase_route", ()) or ()).get("route")
    return route is not None and route != TaylorPhase.name


__all__ = ["nonsmooth_reason", "is_screen_route", "NONSMOOTH_PREFIX"]
