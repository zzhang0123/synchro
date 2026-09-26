"""Full coefficient coordinates of a response reduction (``syncmoments.model.fit.layout``).

LABEL: ``extra eq: finite fit model`` (the real layout ``m = concat(M0[h0],
Re M2[h2], Im M2[h2])`` of ``C``) and ``eq: explicit joint moments`` (the
``b >= 1`` rows of ``M0`` in ``MomentIndex.h0_ext``).

A :class:`CoefficientLayout` extends the ``n_real`` public slots of ``m`` by
the ``len(index.h0_ext)`` real ``M0`` rows with ``b >= 1`` (``include_ext``),
giving the ``n_full`` coordinates ``a = A (m, m0_ext)`` with ``A = N_src/N_*``
the source-column ratio. Slots ``0 .. n_real-1`` are ``index.entries()``
unchanged; slot ``n_real + i`` is ``h0_ext[i]``. The extension slots and every
row whose basis vanishes by ``eq: angular parity`` (an ``h0`` row with
neither ``I`` nor ``V`` modelled, an ``h2`` row with ``l + k`` odd) have an
identically zero response column; they are listed in ``structural_zero`` with
a reason each. :meth:`CoefficientLayout.pad` appends zero columns to a
``(n_rows, n_real)`` response, giving ``(n_rows, n_full)``.

Units: ``a`` is dimensionless (``z``-moments times the dimensionless source
ratio); :meth:`CoefficientLayout.to_raw` multiplies slot ``(r, s, b)`` by
``s_gamma^r s_B^s s_depth^b`` (the raw displacement moments of the
manuscript, in the units of those scales).

:func:`coefficient_bounds` returns the envelope ``|a_j| <= A_max z_g^r z_B^s
z_d^b`` over ``(n_full,)``, with ``z_v`` the largest scaled distance of the
declared ``Support`` from the reference. It holds because every moment is a
weighted average of ``P_l(mu) P_k(eta) z^...`` (times ``exp(2 i phi)`` for
``M2``) with ``|P_l| <= 1``. Not certified: that the population lies inside
the declared ``Support`` (the excluded tail is not covered) or that
``amplitude_bound`` bounds the actual source-column ratio; both are caller
declarations. Everything here is eager NumPy on concrete values.
"""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import numpy as np

from .._provenance import strict_json
from ..errors import ErrorTerm
from ..index import Entry, MomentIndex

LABEL = ("extra eq: finite fit model", "eq: explicit joint moments")
REFERENCE_KEYS = ("gamma0", "B0", "depth_ref", "s_gamma", "s_B", "s_depth")
_EXT_REASON = (
    "M0 row with b >= 1: I and V use b = 0 only (eq: finite joint response), so the "
    "response column is identically zero"
)


def reference_record(reference) -> tuple[tuple[str, float], ...]:
    """Concrete ``(name, float)`` pairs of a ``Reference`` (``REFERENCE_KEYS`` order)."""
    values = (
        float(reference.gamma0),
        float(reference.B0),
        float(reference.depth_ref),
        *(float(s) for s in reference.scales),
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("the reference point and scales must be finite")
    return tuple(zip(REFERENCE_KEYS, values))


def _structural(index, components):
    """``(slots, reasons)`` of the rows of ``m`` whose basis vanishes by parity."""
    with_V = "V" in components
    slots, reasons = [], []
    for entry in index.entries():
        odd = (entry.l + entry.k) % 2 == 1
        if entry.h == 0 and odd and not with_V:
            slots.append(entry.slot)
            reasons.append(
                "M0 row with l+k odd: the I basis is parity-masked and V is not "
                "modelled (eq: angular parity)"
            )
        elif entry.h == 2 and odd:
            slots.append(entry.slot)
            reasons.append(
                "M2 row with l+k odd: the P basis vanishes (eq: angular parity)"
            )
    return slots, reasons


class CoefficientLayout(eqx.Module):
    """Full coordinates ``a = A (m, m0_ext)`` of a reduction; all fields static.

    ``index`` (``MomentIndex``), ``include_ext``, ``entries`` (one
    ``Entry`` per slot, ``n_full``), ``structural_zero`` (slots with an
    identically zero response) with ``structural_reasons``, ``raw_scales``
    (``s_gamma^r s_B^s s_depth^b`` per slot) and ``reference`` (concrete
    ``(name, value)`` pairs). Build with :meth:`build`. Shapes, units and what
    is not certified: see the module docstring.
    """

    LABEL: ClassVar[tuple[str, ...]] = LABEL
    index: MomentIndex = eqx.field(static=True)
    include_ext: bool = eqx.field(static=True)
    entries: tuple[Entry, ...] = eqx.field(static=True)
    structural_zero: tuple[int, ...] = eqx.field(static=True)
    structural_reasons: tuple[str, ...] = eqx.field(static=True)
    raw_scales: tuple[float, ...] = eqx.field(static=True)
    reference: tuple[tuple[str, float], ...] = eqx.field(static=True)

    @classmethod
    def build(
        cls, index, reference, *, components=None, include_ext=True
    ) -> "CoefficientLayout":
        """Layout of ``index`` at ``reference``; ``components`` defaults to ``index.components``."""
        if not isinstance(index, MomentIndex):
            raise ValueError("index must be a MomentIndex")
        if not isinstance(include_ext, bool):
            raise ValueError("include_ext must be a bool")
        components = index.components if components is None else tuple(components)
        record = reference_record(reference)
        entries = list(index.entries())
        slots, reasons = _structural(index, components)
        if include_ext:
            n_real = index.n_real
            for i, row in enumerate(index.h0_ext):
                label = "M0[l={},k={};r={},s={},b={}]".format(*row)
                entries.append(Entry(0, *row, "real", n_real + i, label))
                slots.append(n_real + i)
                reasons.append(_EXT_REASON)
        s_g, s_B, s_d = (v for _, v in record[3:])
        raw = tuple(float(s_g**e.r * s_B**e.s * s_d**e.b) for e in entries)
        return cls(
            index=index,
            include_ext=include_ext,
            entries=tuple(entries),
            structural_zero=tuple(slots),
            structural_reasons=tuple(reasons),
            raw_scales=raw,
            reference=record,
        )

    @property
    def n_full(self) -> int:
        """Number of full coordinates, ``n_real (+ len(h0_ext))``."""
        return len(self.entries)

    @property
    def n_active(self) -> int:
        """Number of public slots of ``m``, ``index.n_real``."""
        return self.index.n_real

    def value(self, name) -> float:
        """One concrete reference value (``REFERENCE_KEYS``)."""
        return dict(self.reference)[name]

    def labels(self) -> tuple[str, ...]:
        """Printed slot names, ``index.labels()`` then the ``h0_ext`` rows."""
        return tuple(e.label for e in self.entries)

    def pad(self, C) -> np.ndarray:
        """``(n_rows, n_real)`` response to ``(n_rows, n_full)`` with zero columns."""
        C = np.asarray(C)
        if C.ndim != 2 or C.shape[1] != self.n_active:
            raise ValueError(
                f"the response must be (n_rows, {self.n_active}), got {C.shape}"
            )
        return np.pad(C, ((0, 0), (0, self.n_full - self.n_active)))

    def full_vector(self, moments, *, amplitude=1.0) -> np.ndarray:
        """``a = A (m, m0_ext)`` of a ``JointMoments`` on this index and reference."""
        if getattr(moments, "index", None) != self.index:
            raise ValueError("moments.index must equal the layout's index")
        other = reference_record(moments.reference)
        for (name, a), (_, b) in zip(other, self.reference):
            if abs(a - b) > 1e-12 * (abs(a) + abs(b)):
                raise ValueError(
                    f"moments and layout use different references ({name}: {a} vs {b})"
                )
        A = float(np.asarray(amplitude))
        if not (math.isfinite(A) and A > 0.0):
            raise ValueError("amplitude must be a positive finite scalar")
        parts = [np.asarray(moments.to_vector(), dtype=float)]
        if self.include_ext:
            if moments.m0_ext is None:
                raise ValueError(
                    "include_ext needs moments.m0_ext (the b >= 1 rows of M0); build "
                    "the moments with JointMoments.from_samples"
                )
            parts.append(np.asarray(moments.m0_ext, dtype=float))
        a = A * np.concatenate(parts)
        if not np.all(np.isfinite(a)):
            raise ValueError("the coefficient vector must be finite")
        return a

    def to_raw(self, a) -> np.ndarray:
        """``a * raw_scales``: the raw displacement moments ``(n_full,)`` or ``(n_full, k)``."""
        a = np.asarray(a, dtype=float)
        if a.shape[0] != self.n_full:
            raise ValueError(f"expected a leading axis of length {self.n_full}")
        scales = np.asarray(self.raw_scales)
        return a * scales.reshape((-1,) + (1,) * (a.ndim - 1))

    def to_dict(self) -> dict:
        """Strict-JSON summary: sizes, labels, structural slots, scales, reference."""
        t = self.index.truncation
        return strict_json(
            {
                "truncation": {
                    "L_mu": t.L_mu,
                    "L_eta": t.L_eta,
                    "N": t.N,
                    "depth_degree": t.depth_degree,
                    "max_orders": t.max_orders,
                },
                "n_full": self.n_full,
                "n_active": self.n_active,
                "include_ext": self.include_ext,
                "labels": list(self.labels()),
                "structural_zero": list(self.structural_zero),
                "structural_reasons": list(self.structural_reasons),
                "raw_scales": list(self.raw_scales),
                "reference": dict(self.reference),
            }
        )


def _scaled_extent(interval, centre, scale):
    lo, hi = (float(v) for v in interval)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return max(abs(lo - centre), abs(hi - centre)) / scale


def coefficient_bounds(layout, support, *, amplitude_bound) -> ErrorTerm:
    """``|a_j| <= A_max z_g^r z_B^s z_d^b`` over ``(n_full,)``, a ``bound`` (module docstring).

    ``support`` is a ``Support`` (``None`` or a missing or infinite interval
    gives ``unbounded``); ``amplitude_bound`` is the declared maximum of the
    source-column ratio ``A`` (not a positive finite float: ``unbounded``).
    Holds for electrons inside the declared support only; not certified:
    the excluded tail and both declarations.
    """
    if not isinstance(layout, CoefficientLayout):
        raise ValueError("layout must be a CoefficientLayout")
    target = "coefficient bound |a|"
    try:
        A = float(amplitude_bound)
    except (TypeError, ValueError):
        A = float("nan")
    if not (math.isfinite(A) and A > 0.0):
        return ErrorTerm.unbounded(
            "amplitude_bound is not a positive finite float: |a| is not bounded", target
        )
    extents = []
    for name, ref, scale in (
        ("gamma", "gamma0", "s_gamma"),
        ("B", "B0", "s_B"),
        ("depth", "depth_ref", "s_depth"),
    ):
        interval = getattr(support, name, None) if support is not None else None
        z = None
        if interval is not None:
            z = _scaled_extent(interval, layout.value(ref), layout.value(scale))
        if z is None:
            return ErrorTerm.unbounded(
                f"the {name} support is missing or infinite: |a| is not bounded", target
            )
        extents.append(z)
    zg, zB, zd = extents
    value = np.array([A * zg**e.r * zB**e.s * zd**e.b for e in layout.entries])
    return ErrorTerm(
        value,
        "bound",
        "|a_j| <= A_max z_g^r z_B^s z_d^b from the declared Support (|P_l| <= 1, "
        f"|exp(2 i phi)| = 1; z = ({zg:.6g}, {zB:.6g}, {zd:.6g}), A_max = {A:.6g}); "
        "holds for electrons inside the declared Support only, the excluded tail is "
        "not covered, and the Support and amplitude_bound are caller declarations",
        target,
    )


__all__ = ["CoefficientLayout", "coefficient_bounds", "LABEL", "REFERENCE_KEYS"]
