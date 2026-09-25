"""Common refinement of parameter maps for ``ParameterMap.assume``.

Private helper of :mod:`syncmoments.model.assumptions`, split out of
``_project.py`` to keep both files under 400 lines. ``refine`` merges the
partitions and closures of several maps; a map's moment-level discrepancy
is carried over only when the other maps add no constraint, otherwise the
combined assumption has no allowance (``unbounded``). Nothing here
certifies that a population satisfies the combined assumption. ``record``
builds the map's ``AssumptionRecord`` (split out of ``assumptions.py``).
"""

from __future__ import annotations

from . import _factor as _f
from ._project import record_hyper, record_kinds, same_closure


def _adds_no_constraint(pm, groups, closures):
    """True when the refined ``(groups, closures)`` are the map's own.

    Only then does the map's moment-level discrepancy describe the combined
    assumption; a finer partition or an extra closure is a further
    constraint that the allowance was not measured or bounded for.
    """
    if tuple(pm.factorisation.groups) != tuple(groups):
        return False
    if len(pm.closures) != len(closures):
        return False
    own = {_f.closure_key(c): c for c in pm.closures}
    keys = [_f.closure_key(c) for c in closures]
    return all(k in own and same_closure(own[k], c) for k, c in zip(keys, closures))


def refine(maps):
    """Common refinement of the maps' partitions and merged closures.

    Returns ``(groups, closures, free_hyper, discrepancy)``; raises
    ``ValueError`` for a nodal map, mismatched indices, a closed group split
    by the refinement, conflicting closures or two discrepancies. The
    discrepancy of an input map is kept only when the other maps add no
    constraint (the refinement equals that map's own groups and closures);
    otherwise it is ``None``, so the combined assumption is ``unbounded``
    until the caller supplies an allowance for it. Symmetry closures (empty
    group) are never split and merge by kind.
    """
    if any(m.nodes is not None for m in maps):
        raise ValueError("a nodal map cannot be refined")
    if any(m.index != maps[0].index for m in maps):
        raise ValueError("all maps must share one MomentIndex")
    blocks = [set(_f.VARS)]
    for m in maps:
        blocks = [
            b & set(g) for b in blocks for g in m.factorisation.groups if b & set(g)
        ]
    groups = _f.canonical_partition(tuple(tuple(b) for b in blocks))
    closures, free_hyper = [], []
    for m in maps:
        for i, closure in enumerate(m.closures):
            symmetry = closure.kind in _f.SYMMETRY_KINDS
            if not symmetry and closure.group not in groups:
                raise ValueError(
                    f"closure on {closure.group} is split by the refinement"
                )
            key = _f.closure_key(closure)
            match = [c for c in closures if _f.closure_key(c) == key]
            if match:
                if not same_closure(match[0], closure):
                    raise ValueError(f"conflicting closures on {closure.group}")
                continue
            if i in m.free_hyper:
                free_hyper.append(len(closures))
            closures.append(closure)
    discrepancies = [m.discrepancy for m in maps if m.discrepancy is not None]
    if len(discrepancies) > 1:
        raise ValueError(
            "combine moment-level discrepancies explicitly before assume()"
        )
    closures = tuple(closures)
    discrepancy = None
    for m in maps:
        if m.discrepancy is not None and _adds_no_constraint(m, groups, closures):
            discrepancy = m.discrepancy
    return groups, closures, tuple(free_hyper), discrepancy


def record(pm, AssumptionRecord):
    """``AssumptionRecord`` of ``pm`` (see ``ParameterMap.record``)."""
    return AssumptionRecord(
        name=pm.name,
        label=pm.LABEL,
        groups=pm.factorisation.groups,
        closure_kind=record_kinds(pm),
        hyper=record_hyper(pm.closures, pm.free_hyper),
        n_free=pm.n_free(),
        discrepancy_kind="unbounded" if pm.discrepancy is None else pm.discrepancy.kind,
    )


__all__ = ["refine", "record"]
