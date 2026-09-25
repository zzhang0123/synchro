"""Error terms, budgets and provenance (``syncmoments.model.errors``).

LABEL: ``eq: channel error budget`` (the nine slots of :class:`ErrorBudget`),
``eq: local response remainder`` (``basis_remainder``), ``eq: screen exponent
error`` (``screen_exponent``), ``eq: joint screen remainder`` (``depth_model``
in the ``app: depth moments`` layout), ``extra eq: data error propagation`` (consumers of
:meth:`ErrorBudget.envelope`).

An :class:`ErrorTerm` is a nonnegative envelope ``value`` in the units of the
quantity it bounds (channel Stokes, ``(n_ch, 4)`` or broadcastable) with a
static ``kind``: ``"bound"`` (a proven envelope under stated hypotheses),
``"measured"`` (an actual difference on a finite population),
``"estimate"`` (a heuristic such as a two-quadrature difference),
``"unbounded"`` (no information; ``value=None``) or ``"not_applicable"``
(the term does not arise; ``value=None``). A missing input is ``unbounded``,
never zero. ``ErrorBudget.total()`` is ``unbounded`` when any term is,
otherwise the sum of the valued terms with the weakest kind present
(``bound < measured < estimate``). ``not_applicable`` terms are skipped.

Nothing here certifies a value: a ``bound`` is only as good as the inputs the
producing module documents, and the manuscript states that ``E_phys`` and
``E_tail`` are never bounded by the package itself.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._provenance import AssumptionRecord, Provenance

KINDS = ("bound", "estimate", "measured", "unbounded", "not_applicable")
WEAKEST_ORDER = ("bound", "measured", "estimate")
_VALUELESS = ("unbounded", "not_applicable")

MANUSCRIPT_TERMS = {
    "basis_remainder": "N_src <rho_nu>",
    "statistical_input": "N_src sum_a |S_nu a| Delta_a",
    "physical_kernel": "E_phys",
    "harmonic_truncation": "E_num",
    "excluded_tail": "E_tail",
    "numerical": "E_num",
    "screen_exponent": "eq: screen exponent error",
    "depth_model": "N_src tau <|K_P| |delta varphi|>",
    "amplitude": "amplitude uncertainty",
    "assumption": "E_phys",
}


def _static_str(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a str, got {value!r}")
    return value


class ErrorTerm(eqx.Module):
    """One error envelope: ``value`` (nonnegative, real, finite) and a static kind.

    ``value`` is ``None`` iff ``kind in {"unbounded", "not_applicable"}``;
    otherwise it is a float array, usually ``(n_ch, 4)`` in channel Stokes
    units, or any shape that broadcasts to it. ``note`` says where the number
    came from and under which hypotheses; ``manuscript_term`` names the term
    of ``eq: channel error budget`` it belongs to (``"E_num"``, ...).
    Value checks use ``equinox.error_if`` (JIT and AD safe).

    Assumes the caller's envelope holds under the hypotheses of its note;
    the term records that claim and its kind, it does not verify it. Not
    certified: any ``bound`` whose inputs (``E_phys``, ``E_tail``, supplied
    envelopes) are external declarations.
    """

    LABEL: ClassVar[str] = "eq: channel error budget"
    value: jax.Array | None
    kind: str = eqx.field(static=True)
    note: str = eqx.field(static=True, default="")
    manuscript_term: str = eqx.field(static=True, default="")

    def __init__(self, value, kind, note="", manuscript_term=""):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.kind = kind
        self.note = _static_str(note, "note")
        self.manuscript_term = _static_str(manuscript_term, "manuscript_term")
        if kind in _VALUELESS:
            if value is not None:
                raise ValueError(f"kind {kind!r} carries no value")
            self.value = None
            return
        if value is None:
            raise ValueError(f"kind {kind!r} requires a value")
        value = jnp.asarray(value)
        if jnp.iscomplexobj(value):
            raise ValueError("error values must be real magnitudes")
        value = value.astype(jnp.result_type(value, 1.0))
        value = eqx.error_if(
            value, jnp.any(~jnp.isfinite(value)), "error value must be finite"
        )
        self.value = eqx.error_if(
            value, jnp.any(value < 0), "error value must be nonnegative"
        )

    @classmethod
    def unbounded(cls, note="", manuscript_term=""):
        """No information: ``value=None``, ``kind="unbounded"``."""
        return cls(None, "unbounded", note, manuscript_term)

    @classmethod
    def not_applicable(cls, note=""):
        """The term does not arise for this route (skipped in ``total()``)."""
        return cls(None, "not_applicable", note)

    @classmethod
    def declared_zero(cls, reason, shape=(1, 4)):
        """A zero ``bound`` with the declaration recorded in ``note``."""
        return cls(jnp.zeros(shape), "bound", f"declared zero: {reason}")

    def scaled(self, factor):
        """Multiply ``value`` by a nonnegative ``factor`` (broadcast); keep kind and note."""
        if self.value is None:
            return self
        factor = jnp.asarray(factor)
        factor = eqx.error_if(
            factor, jnp.any(factor < 0), "scale factor must be nonnegative"
        )
        return ErrorTerm(
            self.value * factor, self.kind, self.note, self.manuscript_term
        )

    def to_dict(self) -> dict:
        """``{"value": nested list | None, "kind", "note", "manuscript_term"}`` (concrete values only)."""
        value = None if self.value is None else np.asarray(self.value).tolist()
        return {
            "value": value,
            "kind": self.kind,
            "note": self.note,
            "manuscript_term": self.manuscript_term,
        }


def _check_term(term, name):
    if not isinstance(term, ErrorTerm):
        raise ValueError(f"{name} must be an ErrorTerm, got {type(term).__name__}")
    return term


class ErrorBudget(eqx.Module):
    """The nine slots of ``eq: channel error budget`` plus named assumption terms.

    Slots (all :class:`ErrorTerm`, channel Stokes units ``(n_ch, 4)`` or
    broadcastable): ``basis_remainder`` (``eq: local response remainder``),
    ``statistical_input`` (``A sum_a |S_a| Delta_a``), ``physical_kernel``
    (``E_phys``), ``harmonic_truncation`` and ``numerical`` (both part of the
    manuscript's ``E_num``), ``excluded_tail`` (``E_tail``),
    ``screen_exponent``, ``depth_model``, ``amplitude``. ``assumption`` is a
    tuple of ``(name, ErrorTerm)`` pairs, one per declared factorisation or
    closure; the names are plain strings inside a pytree, so a budget with
    assumptions crosses a JIT boundary through ``equinox.filter_jit`` (plain
    ``jax.jit`` rejects string leaves). Values are in channel Stokes units,
    ``(n_ch, 4)`` or broadcastable. The budget assumes nothing itself; it
    aggregates declared terms, and it does not certify any of them (``E_phys``
    and ``E_tail`` are inputs; ``estimate`` terms are not bounds).
    """

    basis_remainder: ErrorTerm
    statistical_input: ErrorTerm
    physical_kernel: ErrorTerm
    harmonic_truncation: ErrorTerm
    excluded_tail: ErrorTerm
    numerical: ErrorTerm
    screen_exponent: ErrorTerm
    depth_model: ErrorTerm
    amplitude: ErrorTerm
    assumption: tuple[tuple[str, ErrorTerm], ...] = ()

    SLOTS: ClassVar[tuple[str, ...]] = (
        "basis_remainder",
        "statistical_input",
        "physical_kernel",
        "harmonic_truncation",
        "excluded_tail",
        "numerical",
        "screen_exponent",
        "depth_model",
        "amplitude",
    )
    LABEL: ClassVar[str] = "eq: channel error budget"

    def __check_init__(self):
        for slot in self.SLOTS:
            _check_term(getattr(self, slot), slot)
        if not isinstance(self.assumption, tuple):
            raise ValueError("assumption must be a tuple of (name, ErrorTerm) pairs")
        names = []
        for pair in self.assumption:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], str)
            ):
                raise ValueError(
                    "assumption entries must be (name: str, ErrorTerm) pairs"
                )
            _check_term(pair[1], f"assumption:{pair[0]}")
            names.append(pair[0])
        if len(set(names)) != len(names):
            raise ValueError("assumption names must be unique")

    @classmethod
    def all_unbounded(cls, note=""):
        """Every slot ``unbounded`` with ``note``; no assumptions."""
        return cls(**{slot: ErrorTerm.unbounded(note) for slot in cls.SLOTS})

    def terms(self) -> tuple[tuple[str, ErrorTerm], ...]:
        """The nine slots in order, then ``("assumption:<name>", term)`` pairs."""
        slots = tuple((slot, getattr(self, slot)) for slot in self.SLOTS)
        return slots + tuple(
            (f"assumption:{name}", term) for name, term in self.assumption
        )

    def unbounded(self) -> tuple[str, ...]:
        """Names of the terms with ``kind == "unbounded"``."""
        return tuple(name for name, term in self.terms() if term.kind == "unbounded")

    def total(self) -> ErrorTerm:
        """Sum of the valued terms with the weakest kind; ``unbounded`` if any term is.

        Values broadcast to a common shape (``ValueError`` when they cannot).
        ``not_applicable`` terms are skipped; with no valued term the total is
        a zero ``bound`` of shape ``(1, 4)``.
        """
        missing = self.unbounded()
        if missing:
            return ErrorTerm.unbounded(
                "unbounded terms: " + ", ".join(missing), manuscript_term=self.LABEL
            )
        valued = [(name, term) for name, term in self.terms() if term.value is not None]
        if not valued:
            return ErrorTerm(
                jnp.zeros((1, 4)), "bound", "no applicable terms", self.LABEL
            )
        shape = jnp.broadcast_shapes(*(term.value.shape for _, term in valued))
        value = sum(jnp.broadcast_to(term.value, shape) for _, term in valued)
        rank = max(WEAKEST_ORDER.index(term.kind) for _, term in valued)
        kind = WEAKEST_ORDER[rank]
        weakest = [name for name, term in valued if term.kind == kind]
        note = f"sum of {len(valued)} terms; kind {kind} from " + ", ".join(weakest)
        return ErrorTerm(value, kind, note, self.LABEL)

    def envelope(self) -> jax.Array | None:
        """``total().value`` (``None`` when unbounded)."""
        return self.total().value

    def to_dict(self) -> dict:
        """Slot dicts (with the manuscript term filled in), assumptions and total."""
        out = {}
        for slot in self.SLOTS:
            out[slot] = _term_dict(getattr(self, slot), MANUSCRIPT_TERMS[slot])
        out["assumption"] = {
            name: _term_dict(term, MANUSCRIPT_TERMS["assumption"])
            for name, term in self.assumption
        }
        total = self.total().to_dict()
        total["unbounded"] = list(self.unbounded())
        out["total"] = total
        return out


def _term_dict(term, default_manuscript_term):
    d = term.to_dict()
    if not d["manuscript_term"]:
        d["manuscript_term"] = default_manuscript_term
    return d


__all__ = [
    "KINDS",
    "WEAKEST_ORDER",
    "MANUSCRIPT_TERMS",
    "ErrorTerm",
    "ErrorBudget",
    "AssumptionRecord",
    "Provenance",
]
