"""Declared data discrepancy on the kept rows, as ``fit_linear`` composes it (private).

``data_discrepancy`` reproduces the composition inside
``_linear_core.bias_bound`` without editing it: ``data.discrepancy_vector()``
plus, with an observing response, ``_linear_core._response_part`` (the
``|delta R| (|S_hat| + E)`` term of ``extra eq: data error propagation``,
which makes the result an ``estimate``). A drift test
(``tests/model/test_combinations_errors.py``) checks it against
``fit_linear``. Nothing here is public API.
"""

from __future__ import annotations

from ._linear_core import _response_part


def data_discrepancy(data, stokes_hat):
    """``(delta | None, kind, note, unbounded_reason | None)`` over the kept rows."""
    term = getattr(data, "discrepancy", None)
    if term is None or term.value is None:
        reason = (
            "no data.discrepancy declared"
            if term is None
            else f"data.discrepancy is {term.kind}"
        )
        return None, "unbounded", "", reason + ": the bias is not bounded"
    kind, extra = term.kind, ""
    delta = data.discrepancy_vector()
    if getattr(data, "response", None) is not None:
        if getattr(data, "response_uncertainty", None) is None:
            return (
                None,
                "unbounded",
                "",
                "data.response is set without response_uncertainty (delta R): the "
                "|delta R S_hat| term of extra eq: data error propagation is not "
                "bounded (pass zeros to declare the response exact)",
            )
        part, extra = _response_part(data, stokes_hat)
        if part is not None:
            delta, kind = delta + part, "estimate"
    return delta, kind, f"declared discrepancy: {term.note}{extra}", None


__all__ = ["data_discrepancy"]
