"""Index layout of the finite joint response (``synchro.model.index``).

LABEL: ``eq: explicit joint moments`` (rows ``(l, k, r, s, b)`` of the moment
tensors ``M^{(0)}``, ``M^{(2)}``), ``eq: finite joint response`` (cutoffs
``r+s <= N`` for ``I, V`` and ``r+s+b <= N`` for ``P``), ``eq: angular
parity`` (``P`` rows with ``l+k`` odd dropped), ``app: depth moments``
(``depth_degree=L`` layout ``r+s <= N, b <= L``), ``extra eq: finite fit
model`` (real flattening ``m = concat(M0[h0], Re M2[h2], Im M2[h2])``).

Everything here is static bookkeeping: tuples of integers, hashable, with no
array leaves, so a ``MomentIndex`` can be a static field of an
``equinox.Module`` or a static argument of ``jax.jit``. Moments themselves
live in ``synchro.model.moments``; nothing in this module is a physical
quantity, and no units enter. This module provides only the enumeration and
slot arithmetic of the retained rows (exact integer bookkeeping, validated by
the package tests); it does not certify that a moment vector
laid out this way is feasible for any population.
"""

from __future__ import annotations

import itertools
from math import comb
from typing import ClassVar, NamedTuple

import equinox as eqx
import numpy as np

_COMPONENTS = ("I", "Q", "U", "V")
Row = tuple[int, int, int, int, int]


def _static_int(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a static integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return int(value)


class Truncation(eqx.Module):
    """Static truncation orders ``(L_mu, L_eta, N, depth_degree)``.

    ``L_mu``, ``L_eta``: Legendre degrees in ``mu = cos(alpha)`` and
    ``eta = cos(theta)``; ``N``: total Taylor degree in the displacements
    ``z = (z_gamma, z_B, z_depth)``. ``depth_degree=None`` uses the main-text
    cutoff ``r+s+b <= N`` for the ``P`` block (``eq: finite joint response``);
    ``depth_degree=L`` uses the ``app: depth moments`` layout ``r+s <= N, b <= L``
    (``app: depth moments``). ``I, V`` rows always use ``r+s <= N, b = 0``.
    All fields are static and hashable; the module has no array leaves.
    Dimensionless. Assumes nothing about the population; the orders are
    declared cutoffs, and nothing here certifies that they are sufficient
    (the remainder terms of ``eq: channel error budget`` do that).
    """

    LABEL: ClassVar[str] = "eq: finite joint response"
    L_mu: int = eqx.field(static=True)
    L_eta: int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    depth_degree: int | None = eqx.field(static=True, default=None)

    def __init__(self, L_mu, L_eta, N, depth_degree=None):
        self.L_mu = _static_int(L_mu, "L_mu")
        self.L_eta = _static_int(L_eta, "L_eta")
        self.N = _static_int(N, "N")
        self.depth_degree = (
            None if depth_degree is None else _static_int(depth_degree, "depth_degree")
        )

    def max_b(self) -> int:
        """Largest retained depth power: ``N`` or ``depth_degree``."""
        return self.N if self.depth_degree is None else self.depth_degree


class Entry(NamedTuple):
    """One real slot of the flattened moment vector ``m``.

    ``h in {0, 2}`` is the azimuthal Fourier weight, ``part`` is ``"real"``
    for ``M0`` rows and ``"re"``/``"im"`` for the two real slots of an ``M2``
    row, ``slot`` is the position in ``m`` and ``label`` the printed name.
    Dimensionless bookkeeping (``extra eq: finite fit model``); no physical
    content and nothing certified.
    """

    LABEL = "extra eq: finite fit model"

    h: int
    l: int
    k: int
    r: int
    s: int
    b: int
    part: str
    slot: int
    label: str


def _label(h, l, k, r, s, b, part):
    if h == 0:
        return f"M0[l={l},k={k};r={r},s={s}]"
    prefix = "Re" if part == "re" else "Im"
    return f"{prefix} M2[l={l},k={k};r={r},s={s},b={b}]"


def _rs_rows(N):
    return [(r, s) for r in range(N + 1) for s in range(N + 1 - r)]


def _rsb_rows(truncation):
    N = truncation.N
    if truncation.depth_degree is None:
        return [(r, s, b) for r, s in _rs_rows(N) for b in range(N + 1 - r - s)]
    return [(r, s, b) for r, s in _rs_rows(N) for b in range(truncation.max_b() + 1)]


class MomentIndex(eqx.Module):
    """Enumeration of the retained moment rows and their slots in ``m``.

    Rows are 5-tuples ``(l, k, r, s, b)`` in lexicographic order:

    * ``h0``: ``M0`` rows, all ``(l, k)`` with ``l <= L_mu, k <= L_eta``,
      ``r+s <= N``, ``b = 0``; slot 0 is ``M0[l=0,k=0;r=0,s=0] = 1``.
      Without ``"V"`` in ``components`` the ``l+k`` odd rows are dropped
      (they weight only the ``V`` basis, ``eq: angular parity``).
    * ``h2``: ``M2`` rows (complex), ``l+k`` even when ``parity=True``
      (default; the odd bases vanish identically), all pairs when
      ``parity=False`` ``[extension]``; ``r+s+b <= N`` or the ``app: depth moments``
      cutoff.
    * ``h0_ext``: real ``M0`` rows with ``b >= 1`` over the ``h2`` pair set
      and cutoff. Not part of ``m``; used by parameter maps (azimuth
      separability needs ``M0`` with ``b > 0``) and by sample moments.

    ``m = concat(M0[h0], Re M2[h2], Im M2[h2])`` has length
    ``n_real = n0 + 2 n2``. ``pairs`` lists the ``(l, k)`` pairs that appear
    in ``h0`` or ``h2``; ``n_lk = len(pairs)``. Sizes (``d = 1`` iff both
    Legendre limits are even): ``n0 = (L_mu+1)(L_eta+1) C(N+2,2)`` when
    ``V`` is retained, ``n2 = n_+ C(N+3,3)`` with
    ``n_+ = [(L_mu+1)(L_eta+1) + d]/2`` in the main-text layout.

    All fields are static and hashable; there are no array leaves. Build with
    :meth:`build` (or the constructor, which takes the same keywords).
    Dimensionless bookkeeping. Assumes the parity rule ``eq: angular parity``
    when ``parity=True``; nothing here certifies that the retained rows
    suffice for a given population (see ``bounds.RemainderInputs``).
    """

    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: explicit joint moments",
        "extra eq: finite fit model",
    )
    truncation: Truncation = eqx.field(static=True)
    parity: bool = eqx.field(static=True)
    h0: tuple[Row, ...] = eqx.field(static=True)
    h0_ext: tuple[Row, ...] = eqx.field(static=True)
    h2: tuple[Row, ...] = eqx.field(static=True)
    n0: int = eqx.field(static=True)
    n2: int = eqx.field(static=True)
    n_real: int = eqx.field(static=True)
    n_lk: int = eqx.field(static=True)
    pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, truncation, *, parity=True, components=("I", "Q", "V")):
        if not isinstance(truncation, Truncation):
            raise ValueError("truncation must be a Truncation")
        if not isinstance(parity, bool):
            raise ValueError("parity must be a bool")
        components = _check_components(components)
        with_V = "V" in components
        with_P = "Q" in components or "U" in components
        L_mu, L_eta, N = truncation.L_mu, truncation.L_eta, truncation.N
        all_pairs = list(itertools.product(range(L_mu + 1), range(L_eta + 1)))
        even = [(l, k) for l, k in all_pairs if (l + k) % 2 == 0]
        h0_pairs = all_pairs if with_V else even
        h2_pairs = (all_pairs if not parity else even) if with_P else []
        rsb = _rsb_rows(truncation)
        self.truncation = truncation
        self.parity = parity
        self.components = components
        self.h0 = tuple((l, k, r, s, 0) for l, k in h0_pairs for r, s in _rs_rows(N))
        self.h2 = tuple((l, k, r, s, b) for l, k in h2_pairs for r, s, b in rsb)
        self.h0_ext = tuple(row for row in self.h2 if row[4] >= 1)
        self.n0 = len(self.h0)
        self.n2 = len(self.h2)
        self.n_real = self.n0 + 2 * self.n2
        self.pairs = tuple(sorted({(row[0], row[1]) for row in self.h0 + self.h2}))
        self.n_lk = len(self.pairs)

    @classmethod
    def build(cls, truncation, *, parity=True, components=("I", "Q", "V")):
        """Enumerate the index for ``truncation``; see the class docstring."""
        return cls(truncation, parity=parity, components=components)

    # -- lookups ---------------------------------------------------------

    def position(self, h, l, k, r, s, b) -> int:
        """Slot of a row in ``m``: ``M0`` rows directly, ``M2`` rows give the
        ``Re`` slot ``n0 + i``; the ``Im`` slot is ``position + n2``.
        Raises ``ValueError`` for rows outside ``m`` (including ``h0_ext``)."""
        row = (int(l), int(k), int(r), int(s), int(b))
        if h == 0:
            table, offset = self.h0, 0
        elif h == 2:
            table, offset = self.h2, self.n0
        else:
            raise ValueError(f"h must be 0 or 2, got {h!r}")
        try:
            return offset + table.index(row)
        except ValueError:
            raise ValueError(
                f"row h={h}, (l,k,r,s,b)={row} is not retained in m"
            ) from None

    def ext_position(self, l, k, r, s, b) -> int:
        """Index of a real ``b >= 1`` row in ``h0_ext`` (not a slot of ``m``)."""
        row = (int(l), int(k), int(r), int(s), int(b))
        try:
            return self.h0_ext.index(row)
        except ValueError:
            raise ValueError(f"row (l,k,r,s,b)={row} is not in h0_ext") from None

    def entries(self) -> tuple[Entry, ...]:
        """All ``n_real`` slots in order: ``h0`` real, ``h2`` re, ``h2`` im."""
        out = [
            Entry(0, *row, "real", i, _label(0, *row, "real"))
            for i, row in enumerate(self.h0)
        ]
        for part, offset in (("re", self.n0), ("im", self.n0 + self.n2)):
            out.extend(
                Entry(2, *row, part, offset + i, _label(2, *row, part))
                for i, row in enumerate(self.h2)
            )
        return tuple(out)

    def labels(self) -> tuple[str, ...]:
        """Printed names, e.g. ``"M0[l=1,k=2;r=1,s=0]"``, ``"Im M2[l=0,k=2;r=0,s=1,b=1]"``."""
        return tuple(entry.label for entry in self.entries())

    # -- masks and maps ----------------------------------------------------

    def parity_I(self) -> tuple[bool, ...]:
        """Over ``h0``: ``l+k`` even (rows that weight the ``I`` basis)."""
        return tuple((row[0] + row[1]) % 2 == 0 for row in self.h0)

    def parity_V(self) -> tuple[bool, ...]:
        """Over ``h0``: ``l+k`` odd (rows that weight the ``V`` basis)."""
        return tuple((row[0] + row[1]) % 2 == 1 for row in self.h0)

    def h0_pairs(self) -> tuple[int, ...]:
        """Index into ``pairs`` for each ``h0`` row."""
        lookup = {pair: i for i, pair in enumerate(self.pairs)}
        return tuple(lookup[(row[0], row[1])] for row in self.h0)

    def h2_pairs(self) -> tuple[int, ...]:
        """Index into ``pairs`` for each ``h2`` row."""
        lookup = {pair: i for i, pair in enumerate(self.pairs)}
        return tuple(lookup[(row[0], row[1])] for row in self.h2)

    @staticmethod
    def exponents(rows) -> np.ndarray:
        """NumPy ``(n_rows, 5)`` int array of ``(l, k, r, s, b)`` for ``rows``."""
        rows = tuple(rows)
        if not rows:
            return np.zeros((0, 5), dtype=int)
        arr = np.asarray(rows, dtype=int)
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError("rows must be a sequence of 5-tuples (l, k, r, s, b)")
        return arr


def _check_components(components):
    if not isinstance(components, tuple) or not components:
        raise ValueError("components must be a nonempty tuple of Stokes names")
    if any(not isinstance(c, str) or c not in _COMPONENTS for c in components):
        raise ValueError(
            f"components must be drawn from {_COMPONENTS}, got {components!r}"
        )
    if len(set(components)) != len(components):
        raise ValueError("components must not repeat")
    if "I" not in components:
        raise ValueError(
            'components must include "I" (slot 0 is M0[l=0,k=0;r=0,s=0] = 1)'
        )
    return tuple(components)


def n_plus(L_mu: int, L_eta: int) -> int:
    """``[(L_mu+1)(L_eta+1) + d]/2`` retained even-parity pairs, ``d = 1`` iff both even.

    Dimensionless count; ``eq: angular parity``. No assumptions, nothing certified.
    """
    d = 1 if (L_mu % 2 == 0 and L_eta % 2 == 0) else 0
    return ((L_mu + 1) * (L_eta + 1) + d) // 2


def main_text_sizes(truncation: Truncation) -> tuple[int, int, int]:
    """Closed-form ``(n0, n2, n_real)`` for the default index (all components).

    Dimensionless counts for ``parity=True`` and the main-text cutoff; a
    cross-check of the :class:`MomentIndex` sizes (a finite check, not a
    certificate).
    """
    L_mu, L_eta, N = truncation.L_mu, truncation.L_eta, truncation.N
    n0 = (L_mu + 1) * (L_eta + 1) * comb(N + 2, 2)
    if truncation.depth_degree is None:
        n2 = n_plus(L_mu, L_eta) * comb(N + 3, 3)
    else:
        n2 = n_plus(L_mu, L_eta) * comb(N + 2, 2) * (truncation.depth_degree + 1)
    return n0, n2, n0 + 2 * n2


__all__ = ["Truncation", "Entry", "MomentIndex", "n_plus", "main_text_sizes"]
