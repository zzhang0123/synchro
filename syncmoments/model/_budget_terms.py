"""Slot fillers of ``eq: channel error budget`` shared by ``predict`` and the
direct average (private helper of ``syncmoments.model.predict``).

Every function returns an ``ErrorTerm`` of value shape ``(n_ch, 4)`` in the
units of the prediction (amplitude times the per-electron kernel). A missing
input gives ``unbounded`` with the reason in the note, never a zero; a
declared value keeps its declared kind. Nothing here certifies the truth of
a supplied envelope.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .errors import WEAKEST_ORDER, ErrorTerm

_PROPAGATION = "extra eq: data error propagation"
_STATISTICAL = "N_src sum_a |S_a| Delta_a"


def as_amplitude(amplitude):
    """Nonnegative finite scalar amplitude (``N_src``)."""
    from .bounds import _amplitude

    return _amplitude(amplitude)


def stokes_shape(value, n_ch, name):
    """Broadcast ``value`` to ``(n_ch, 4)`` or raise ``ValueError``."""
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        return jnp.broadcast_to(value.astype(jnp.result_type(value, 1.0)), (n_ch, 4))
    except ValueError:
        raise ValueError(
            f"{name} must broadcast to ({n_ch}, 4), got shape {value.shape}"
        ) from None


def term_or_value(value, n_ch, name, *, kind="bound", note="", manuscript_term=""):
    """Accept an ``ErrorTerm`` (returned as is) or an array (a ``kind`` term)."""
    if isinstance(value, ErrorTerm):
        if value.value is not None:
            stokes_shape(value.value, n_ch, name)
        return value
    return ErrorTerm(stokes_shape(value, n_ch, name), kind, note, manuscript_term)


def contract_moment_error(absC, delta, amplitude, n_ch):
    """``amplitude (|C| @ Delta)`` reshaped to ``(n_ch, 4)``.

    ``Delta`` is ``(n_real,)`` or a scalar, which is broadcast to every moment
    (``statistical_input=0`` declares exact moments); other shapes raise.
    """
    delta = jnp.asarray(delta)
    if jnp.iscomplexobj(delta):
        raise ValueError("moment discrepancies must be real magnitudes")
    if delta.ndim == 0:
        delta = jnp.broadcast_to(
            delta.astype(jnp.result_type(delta, 1.0)), (absC.shape[1],)
        )
    if delta.shape != (absC.shape[1],):
        raise ValueError(
            f"moment discrepancy must have shape ({absC.shape[1]},), got {delta.shape}"
        )
    return amplitude * (absC @ jnp.abs(delta)).reshape(n_ch, 4)


def _weaker(*kinds):
    """Weakest of valued kinds (``bound < measured < estimate``)."""
    return max(kinds, key=WEAKEST_ORDER.index)


def statistical_term(statistical_input, moments, absC, amplitude, n_ch):
    """``N_src sum_a |S_a| Delta_a`` from an explicit ``Delta``, an unattributed
    moment-level discrepancy, or both (their sum, with the weaker kind)."""
    discrepancy = getattr(moments, "discrepancy", None)
    if discrepancy is not None and moments.assumptions:
        discrepancy = None  # attributed: carried by the assumption terms
    parts = []  # (Delta, kind, note)
    if isinstance(statistical_input, ErrorTerm):
        if statistical_input.value is None:
            return statistical_input
        parts.append(
            (statistical_input.value, statistical_input.kind, statistical_input.note)
        )
    elif statistical_input is not None:
        parts.append((statistical_input, "bound", "supplied statistical input Delta_a"))
    if discrepancy is not None:
        if discrepancy.value is None:
            return discrepancy
        parts.append((discrepancy.value, discrepancy.kind, discrepancy.note))
    if not parts:
        return ErrorTerm.unbounded(
            "no statistical input Delta_a supplied (pass statistical_input=zeros((n_real,)) "
            "for exact moments of the declared population)",
            _STATISTICAL,
        )
    value = sum(contract_moment_error(absC, d, amplitude, n_ch) for d, _, _ in parts)
    note = "A |C| Delta: " + "; plus unattributed moment-level discrepancy: ".join(
        n for _, _, n in parts
    )
    return ErrorTerm(value, _weaker(*(k for _, k, _ in parts)), note, _STATISTICAL)


def assumption_terms(records, discrepancy, absC, amplitude, n_ch):
    """``(name, term)`` pairs, one term per record. The moment-level
    discrepancy is the allowance of the first record only; every further
    record is ``unbounded`` (no allowance of its own is established)."""
    out = []
    for i, record in enumerate(records):
        if i > 0:
            term = ErrorTerm.unbounded(
                f"assumption '{record.name}' ({record.label}): the moment-level "
                f"discrepancy is attributed to '{records[0].name}'; no allowance of "
                "its own was supplied",
                "E_phys",
            )
        elif discrepancy is None or discrepancy.value is None:
            term = ErrorTerm.unbounded(
                f"assumption '{record.name}' ({record.label}) declared without a "
                "discrepancy allowance",
                "E_phys",
            )
        else:
            value = contract_moment_error(absC, discrepancy.value, amplitude, n_ch)
            term = ErrorTerm(
                value, discrepancy.kind, f"A |C| Delta: {discrepancy.note}", "E_phys"
            )
        out.append((record.name, term))
    return out


def _forced_term(record, replaced=None):
    note = (
        f"assumption '{record.name}' forced by the basis ({record.label}); "
        "no discrepancy allowance"
    )
    if replaced is not None:
        note += (
            f"; the moment-level term of kind '{replaced.kind}' does not bound it "
            "(the retained moments do not resolve the variables it constrains)"
        )
    return ErrorTerm.unbounded(note, "E_phys")


def forced_assumption_terms(records, present):
    """Unbounded terms for basis-forced records not already named in ``present``."""
    return [(r.name, _forced_term(r)) for r in records if r.name not in present]


def merge_forced_terms(moment_terms, records):
    """Moment-level ``(name, term)`` pairs plus an ``unbounded`` term for every
    basis-forced record. A forced record overrides a moment-level term of the
    same name: the phase route or kernel imposes it on variables the retained
    moments do not resolve (``b = 0`` rows for screens, ``L_mu = 0`` for the
    continuum kernel), so a moment-level discrepancy cannot bound it."""
    forced = {r.name: r for r in records}
    out = [
        (name, _forced_term(forced[name], term) if name in forced else term)
        for name, term in moment_terms
    ]
    return out + forced_assumption_terms(records, {name for name, _ in out})


def excluded_tail_term(support, excluded_tail, n_ch):
    """``E_tail`` from the ``Support.truncated`` tri-state and an optional input."""
    kind = "unbounded" if support is None else support.tail_kind()
    if excluded_tail is not None:
        term = term_or_value(
            excluded_tail,
            n_ch,
            "excluded_tail",
            note="supplied excluded-tail term",
            manuscript_term="E_tail",
        )
        if kind == "declared_zero":
            return (
                ErrorTerm(
                    term.value,
                    term.kind,
                    "supplied excluded-tail term (support declared complete)",
                    "E_tail",
                )
                if term.value is not None
                else term
            )
        return term
    if kind == "declared_zero":
        return ErrorTerm.declared_zero(
            "support declared complete (Support.truncated=False)", shape=(n_ch, 4)
        )
    if kind == "required_input":
        return ErrorTerm.unbounded(
            "support declared truncated (Support.truncated=True): excluded_tail is a "
            "required input and the amplitude counts the retained column only",
            "E_tail",
        )
    return ErrorTerm.unbounded(
        "Support.truncated is None: excluded population not declared", "E_tail"
    )


def _stokes_per_electron(terms, amplitude, delta):
    """``value / amplitude`` of Stokes-unit ``(name, term)`` pairs; raises for
    ``amplitude = 0 < delta``."""
    amplitude = jnp.asarray(amplitude)
    safe = jnp.where(amplitude > 0, amplitude, 1.0)
    out = []
    for name, term in terms:
        if term.value is None:
            out.append((name, term))
            continue
        value = eqx.error_if(
            term.value / safe,
            (amplitude <= 0) & (delta > 0),
            f"amplitude_uncertainty > 0 at amplitude 0: the Stokes-unit {name} "
            "term cannot be converted to per-electron units",
        )
        out.append((name, ErrorTerm(value, term.kind, term.note, term.manuscript_term)))
    return out


def _concrete_zero(value):
    """``True`` only for a concrete (untraced) value equal to zero."""
    if isinstance(value, jax.core.Tracer):
        return False
    return bool(np.asarray(value) == 0)


def _checked_delta(amplitude_uncertainty):
    delta = jnp.asarray(amplitude_uncertainty)
    if delta.ndim != 0:
        raise ValueError("amplitude_uncertainty must be a scalar")
    if jnp.iscomplexobj(delta):
        raise ValueError("amplitude_uncertainty must be real")
    delta = delta.astype(jnp.result_type(delta, 1.0))
    return eqx.error_if(
        delta,
        jnp.logical_not(delta >= 0),
        "amplitude_uncertainty must be a nonnegative number (negative or NaN given)",
    )


def amplitude_term(
    amplitude_uncertainty,
    per_electron,
    n_ch,
    *,
    per_electron_errors=(),
    stokes_errors=(),
    amplitude=None,
):
    """``delta A (|C m| + e)`` with ``e`` the valued per-electron error slots.

    ``(A + dA) s_true - A C m = A (s_true - C m) + dA s_true``, so the
    amplitude slot needs ``dA |s_true| <= dA (|C m| + e)``.
    ``per_electron_errors`` are ``(name, term)`` pairs at unit amplitude;
    ``stokes_errors`` are ``(name, term)`` pairs in Stokes units proportional
    to ``amplitude`` (divided by it).
    Unbounded contributions are named in the note: their cross term is
    unbounded with them and the total stays unbounded. A supplied
    ``ErrorTerm`` is returned as the complete slot. A negative (or NaN)
    ``delta A`` raises through ``equinox.error_if``. A concrete ``delta A = 0``
    declares the amplitude exact: a zero ``bound``, since ``0 |s_true| = 0``
    for any finite ``s_true``. The kind is static, so a traced ``delta A``
    gets the weakest of ``bound`` and the valued slot kinds even when its
    value is zero.
    """
    if amplitude_uncertainty is None:
        return ErrorTerm.unbounded(
            "amplitude_uncertainty not supplied", "amplitude uncertainty"
        )
    if isinstance(amplitude_uncertainty, ErrorTerm):
        return amplitude_uncertainty
    delta = _checked_delta(amplitude_uncertainty)
    if _concrete_zero(amplitude_uncertainty):
        return ErrorTerm(
            jnp.zeros((n_ch, 4)),
            "bound",
            "amplitude declared exact (amplitude_uncertainty = 0)",
            "amplitude uncertainty",
        )
    errors = list(per_electron_errors)
    if stokes_errors:
        if amplitude is None:
            raise ValueError("stokes_errors need the amplitude")
        errors += _stokes_per_electron(stokes_errors, amplitude, delta)
    valued = [t for _, t in errors if t.value is not None]
    missing = [name for name, t in errors if t.kind == "unbounded"]
    envelope = jnp.abs(per_electron)
    for term in valued:
        envelope = envelope + jnp.broadcast_to(term.value, (n_ch, 4))
    note = "delta A (|C m| + valued per-electron error slots)"
    if missing:
        note += "; cross terms with unbounded slots omitted: " + ", ".join(missing)
    return ErrorTerm(
        envelope * delta,
        _weaker("bound", *(t.kind for t in valued)),
        note,
        "amplitude uncertainty",
    )


def depth_model_term(depth_model, n_ch):
    if depth_model is None:
        return ErrorTerm.unbounded(
            "depth-model term not supplied (use bounds.depth_error_bound)",
            "N_src tau <|K_P| |delta varphi|>",
        )
    return term_or_value(
        depth_model,
        n_ch,
        "depth_model",
        note="supplied depth-model term",
        manuscript_term="N_src tau <|K_P| |delta varphi|>",
    )


def scaled_kernel_term(term, amplitude, m, n_ch):
    """Per-electron basis term times ``amplitude``; a ``(4 n_ch, n_real)``
    per-column envelope is contracted with ``|m|`` first."""
    if term.value is None:
        return term
    if m is not None and term.value.shape == (4 * n_ch, m.shape[0]):
        value = amplitude * (jnp.abs(term.value) @ jnp.abs(m)).reshape(n_ch, 4)
        return ErrorTerm(
            value, term.kind, term.note + " (contracted with |m|)", term.manuscript_term
        )
    return term.scaled(amplitude)


def propagate_envelope(total, stokes, R, response_uncertainty):
    """``|R| @ envelope + |delta R| @ |stokes|`` per data element (``extra eq: data error propagation``)."""
    n_ch = stokes.shape[0]
    R = jnp.asarray(R)
    if R.ndim != 2 or R.shape[1] != 4 * n_ch:
        raise ValueError(f"R must have shape (n_data, {4 * n_ch}), got {R.shape}")
    if total.value is None:
        return ErrorTerm.unbounded(
            f"budget total is {total.kind}: {total.note}", _PROPAGATION
        )
    envelope = jnp.broadcast_to(total.value, (n_ch, 4)).ravel()
    first = jnp.abs(R) @ envelope
    if response_uncertainty is None:
        return ErrorTerm.unbounded(
            "response_uncertainty (delta R) not supplied; pass 0 to declare the "
            "observing response exact",
            _PROPAGATION,
        )
    flat = jnp.abs(stokes).ravel()
    dR = jnp.asarray(response_uncertainty)
    if dR.ndim == 0:
        second = jnp.abs(dR) * jnp.sum(flat) * jnp.ones(R.shape[0])
        note = "scalar |delta R| bound (0 declares the observing response exact)"
    elif dR.shape == R.shape:
        second = jnp.abs(dR) @ flat
        note = "supplied |delta R|"
    else:
        raise ValueError("response_uncertainty must be a scalar or have R's shape")
    return ErrorTerm(
        first + second,
        total.kind,
        f"|R| @ envelope ({total.kind}: {total.note}) + |delta R| @ |stokes| ({note})",
        _PROPAGATION,
    )


__all__ = [
    "as_amplitude",
    "stokes_shape",
    "term_or_value",
    "contract_moment_error",
    "statistical_term",
    "assumption_terms",
    "forced_assumption_terms",
    "merge_forced_terms",
    "excluded_tail_term",
    "amplitude_term",
    "depth_model_term",
    "scaled_kernel_term",
    "propagate_envelope",
]
