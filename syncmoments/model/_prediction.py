"""The ``Prediction`` container (private helper of ``syncmoments.model.predict``).

Split out so that ``predict.py`` and ``_direct.py`` share it without a
circular import; see ``syncmoments.model.predict`` for the contract, units and
what is not certified.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np

from . import _budget_terms as terms
from .channels import Channels
from ._provenance import strict_json
from .errors import ErrorBudget, ErrorTerm, Provenance
from .moments import JointMoments

LABEL = "eq: finite joint response"


class Prediction(eqx.Module):
    """Channel Stokes ``stokes`` ``(n_ch, 4)`` with amplitude, moments, budget and provenance.

    ``propagate(R, response_uncertainty=)`` applies a linear observing
    response ``R`` ``(n_data, 4 n_ch)`` acting on the channel-major flattened
    ``(I, Q, U, V)`` vector: ``|R| @ envelope + |delta R| @ |stokes|``
    (``extra eq: data error propagation``); the second term is ``unbounded``
    unless ``response_uncertainty`` is supplied (``0`` declares the response
    exact). ``to_dict`` is JSON-ready (``total.value`` is ``null`` when any
    term is unbounded); ``summary`` is a short text report. Both are eager.
    Units: ``stokes`` and every budget term in ``provenance.units``;
    ``amplitude`` is the source column in the basis amplitude units.
    Assumes what ``provenance.assumptions`` lists. Not certified: any
    ``unbounded`` slot, and the truth of declared inputs (``E_phys``,
    ``E_tail``, supplied envelopes).
    """

    stokes: jax.Array
    amplitude: jax.Array
    moments: JointMoments
    budget: ErrorBudget
    provenance: Provenance = eqx.field(static=True)
    channels: Channels
    LABEL: str = eqx.field(static=True, default=LABEL)

    def __check_init__(self):
        if self.stokes.ndim != 2 or self.stokes.shape[1] != 4:
            raise ValueError(
                f"stokes must have shape (n_ch, 4), got {self.stokes.shape}"
            )
        if not isinstance(self.budget, ErrorBudget):
            raise ValueError("budget must be an ErrorBudget")
        if not isinstance(self.provenance, Provenance):
            raise ValueError("provenance must be a Provenance")

    @property
    def n_ch(self) -> int:
        return int(self.stokes.shape[0])

    def propagate(self, R, *, response_uncertainty=None) -> ErrorTerm:
        """``(n_data,)`` envelope of ``R @ stokes.ravel()`` (see the class docstring)."""
        return terms.propagate_envelope(
            self.budget.total(), self.stokes, R, response_uncertainty
        )

    def to_dict(self) -> dict:
        """Strict-JSON dict: ``stokes``, ``units``, ``amplitude``, ``moments``, ``budget``, ``provenance``.

        Non-finite floats are written as ``None`` (``strict_json``).
        """
        return strict_json(
            {
                "stokes": np.asarray(self.stokes).tolist(),
                "stokes_order": ["I", "Q", "U", "V"],
                "units": self.provenance.units,
                "amplitude": float(np.asarray(self.amplitude)),
                "moments": self.moments.to_dict(),
                "budget": self.budget.to_dict(),
                "provenance": self.provenance.to_dict(),
            }
        )

    def summary(self) -> str:
        """Text report: Stokes per channel, envelope or unbounded terms, assumptions."""
        stokes = np.asarray(self.stokes)
        lines = [
            f"Prediction ({self.LABEL}): {self.n_ch} channels, units {self.provenance.units}, "
            f"amplitude {float(np.asarray(self.amplitude)):.6g}",
        ]
        for j, row in enumerate(stokes):
            lines.append(
                f"  channel {j}: I={row[0]:.6e} Q={row[1]:.6e} U={row[2]:.6e} V={row[3]:.6e}"
            )
        total = self.budget.total()
        if total.value is None:
            lines.append(
                "  error envelope unbounded; unbounded terms: "
                + ", ".join(self.budget.unbounded())
            )
        else:
            envelope = np.broadcast_to(np.asarray(total.value), stokes.shape)
            scale = np.max(np.abs(stokes[:, 0])) if np.any(stokes[:, 0] != 0) else 1.0
            lines.append(
                f"  error envelope ({total.kind}): max {np.max(envelope):.3e}, "
                f"{np.max(envelope) / scale:.3e} of the largest channel I"
            )
        for name, term in self.budget.terms():
            value = (
                "-"
                if term.value is None
                else f"{float(np.max(np.asarray(term.value))):.3e}"
            )
            lines.append(f"    {name}: {term.kind} {value}")
        if self.provenance.assumptions:
            lines.append(
                "  assumptions: "
                + ", ".join(r.name for r in self.provenance.assumptions)
            )
        lines.extend(f"  note: {note}" for note in self.provenance.notes)
        return "\n".join(lines)


__all__ = ["Prediction", "LABEL"]
