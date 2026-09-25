"""Provenance records for ``syncmoments.model`` (private helper of ``errors.py``).

Split out of ``syncmoments/model/errors.py`` to keep that file under 400 lines;
import these names from ``syncmoments.model.errors``. Both classes are frozen
dataclasses of hashables so that a record can be a static field of an
``equinox.Module``. They document configuration and declared assumptions;
they certify nothing about accuracy.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np


def _static_str(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str, got {value!r}")
    return value


def strict_json(value):
    """Recursively render ``value`` as strict JSON (``allow_nan=False`` safe).

    Dicts keep their (stringified) keys, tuples and lists become lists,
    NumPy scalars and arrays become Python values, and non-finite floats
    (the NaN placeholders of fitted hyper-parameters, a non-finite fit
    statistic) become ``None``. Shared by every ``to_dict`` of
    ``syncmoments.model`` so the output survives a JSON round trip unchanged.
    """
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if not isinstance(value, (np.ndarray, str, bytes)) and hasattr(value, "__array__"):
        value = np.asarray(value)  # jax.Array and other array-likes
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _tuple_of(values, check, name):
    values = tuple(values)
    for v in values:
        if not check(v):
            raise ValueError(f"{name} has an invalid entry {v!r}")
    return values


@dataclasses.dataclass(frozen=True)
class AssumptionRecord:
    """A declared factorisation or closure, as recorded in provenance.

    ``groups`` is the partition of ``(gamma, B, mu, eta, phi, depth)``;
    ``closure_kind`` names the closure (``"none"``, ``"uniform_mu"``,
    ``"gaussian_depth"``, ...; a fitted closure is ``"<kind> (fitted)"``);
    ``hyper`` its fixed hyper-parameters, or NaN placeholders for fitted
    ones (the fitted values are in the ``FitResult`` provenance notes and
    in the provenance of its prediction; :meth:`to_dict` renders the
    placeholders as ``null``); ``n_free`` the free-parameter count of the
    map; ``discrepancy_kind`` the kind of the discrepancy allowance that
    was supplied. Hashable. ``hyper`` values are in the closure's units
    (rad/m^2 for depth closures). The record documents a declared assumption; it does not
    certify that the population satisfies it (the discrepancy allowance of
    ``eq: channel error budget`` is a separate input). ``[extension]``.
    """

    LABEL: ClassVar[str] = "[extension] assumption record"
    name: str
    label: str
    groups: tuple[tuple[str, ...], ...]
    closure_kind: str
    hyper: tuple[float, ...]
    n_free: int
    discrepancy_kind: str

    def __post_init__(self):
        for field in ("name", "label", "closure_kind", "discrepancy_kind"):
            _static_str(getattr(self, field), field)
        groups = tuple(
            _tuple_of(g, lambda v: isinstance(v, str), "groups") for g in self.groups
        )
        hyper = _tuple_of(
            self.hyper,
            lambda v: isinstance(v, (int, float, np.floating, np.integer))
            and not isinstance(v, bool),
            "hyper",
        )
        if (
            isinstance(self.n_free, bool)
            or not isinstance(self.n_free, int)
            or self.n_free < 0
        ):
            raise ValueError("n_free must be a nonnegative int")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "hyper", tuple(float(h) for h in hyper))

    def to_dict(self) -> dict:
        """Strict-JSON dict; NaN placeholders in ``hyper`` become ``None``."""
        return strict_json(
            {
                "name": self.name,
                "label": self.label,
                "groups": [list(g) for g in self.groups],
                "closure_kind": self.closure_kind,
                "hyper": list(self.hyper),
                "n_free": self.n_free,
                "discrepancy_kind": self.discrepancy_kind,
            }
        )


def _is_pair(item):
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)


def _plain(value):
    """Convert nested tuples to strict-JSON lists/dicts (pair tuples become dicts)."""
    if isinstance(value, AssumptionRecord):
        return value.to_dict()
    if isinstance(value, tuple):
        if value and all(_is_pair(item) for item in value):
            return {key: _plain(item) for key, item in value}
        return [_plain(item) for item in value]
    return strict_json(value)


def _hashable(value):
    try:
        hash(value)
    except TypeError:
        return False
    return True


_STR_FIELDS = ("package_version", "units")


@dataclasses.dataclass(frozen=True)
class Provenance:
    """Hashable record of what produced a basis or prediction.

    Every field is a str, an int, a tuple of ``(key, value)`` pairs (rendered
    as a dict by :meth:`to_dict`) or a tuple of hashables; arrays are not
    allowed (record concrete floats). ``assumptions`` holds
    :class:`AssumptionRecord` entries. ``notes`` carries statements such as
    ``"incident polarisation not modelled"``. Usable as a static field of an
    ``equinox.Module``. The record documents configuration; it is not a
    certificate of accuracy. ``certified_orders`` (name kept for
    compatibility) lists the derivative orders validated by finite-difference
    tests (finite checks, not certificates). Units are stated in the ``units`` field;
    ``reference`` and ``support`` carry Gauss, Hz and rad/m^2 as concrete
    floats. ``[extension]``.
    """

    LABEL: ClassVar[str] = "[extension] provenance record"
    package_version: str
    kernel: tuple
    channels: tuple
    truncation: tuple
    reference: tuple
    support: tuple
    phase_route: tuple
    quadrature: tuple
    assumptions: tuple[AssumptionRecord, ...]
    units: str
    numerics: tuple
    certified_orders: tuple[int, ...]
    finite_checks: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name in _STR_FIELDS:
                _static_str(value, field.name)
            elif not isinstance(value, tuple):
                raise ValueError(
                    f"{field.name} must be a tuple, got {type(value).__name__}"
                )
            if not _hashable(value):
                raise ValueError(f"{field.name} must be hashable")
        _tuple_of(
            self.assumptions, lambda v: isinstance(v, AssumptionRecord), "assumptions"
        )
        _tuple_of(
            self.certified_orders,
            lambda v: isinstance(v, int) and not isinstance(v, bool),
            "certified_orders",
        )
        _tuple_of(self.finite_checks, lambda v: isinstance(v, str), "finite_checks")
        _tuple_of(self.notes, lambda v: isinstance(v, str), "notes")

    def to_dict(self) -> dict:
        """Strict-JSON dict; ``(key, value)`` pair tuples become dicts, NaN ``None``."""
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def with_notes(self, *notes) -> "Provenance":
        """New record with ``notes`` appended."""
        return dataclasses.replace(self, notes=self.notes + tuple(notes))

    def with_assumptions(self, *records) -> "Provenance":
        """New record with ``AssumptionRecord`` entries appended."""
        return dataclasses.replace(self, assumptions=self.assumptions + tuple(records))


__all__ = ["AssumptionRecord", "Provenance", "strict_json"]
