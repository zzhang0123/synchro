"""Channel predictions with their error budget (``synchro.model.predict``).

LABEL: ``eq: finite joint response`` and ``extra eq: finite fit model``
(``stokes = N_src reshape(C m)`` with the response matrix ``C`` of the basis
and the real moment vector ``m``), ``eq: channel error budget`` (the
:class:`ErrorBudget` of every :class:`Prediction`), ``eq: local response
remainder`` (``basis_remainder`` through ``RemainderInputs``), ``extra eq:
data error propagation`` (:meth:`Prediction.propagate`), ``eq: smooth
channel kernel`` and ``extra eq: faraday channel kernel``
(:func:`direct_channel_average`: the discrete-population average with the
per-emitter, per-line Faraday phase and the sky factor ``exp(2 i phi)``).

Units: ``stokes`` ``(n_ch, 4)`` in Stokes order ``I, Q, U, V`` is the
amplitude ``N_src`` times the per-electron channel kernel (erg/s/sr per
electron for ``unit_peak`` channels, per Hz for ``unit_integral``); every
budget term is in the same units, shape ``(n_ch, 4)``. ``amplitude`` is a
nonnegative scalar. Moments are in the ``z`` coordinates of the basis
reference; ``predict`` raises ``ValueError`` for an index mismatch and
``equinox.error_if`` for a reference mismatch.

Budget rules (FINAL_DESIGN Section 8). Missing inputs are ``unbounded``,
never zero: ``basis_remainder`` needs ``errors`` (a ``RemainderInputs``) and
is ``unbounded`` on a nonsmooth line-kernel basis (the ``build_basis``
note); ``statistical_input`` is ``N_src |C| Delta`` from
``statistical_input`` (``Delta_a`` over ``m``) plus any moment-level
discrepancy that is not attributed to an assumption (their sum, weaker
kind); the kernel terms come from ``basis.kernel_terms`` scaled by the
amplitude (a ``(4 n_ch, n_real)`` per-column envelope is contracted with
``|m|``); ``excluded_tail`` follows ``Support.truncated`` (``None``:
unbounded, ``False``: declared zero, ``True``: ``excluded_tail`` is a
required input); ``depth_model`` needs its input; ``amplitude`` is
``delta N_src (|C m| + e)`` with ``e`` the valued per-electron slots
(``depth_model`` divided by ``N_src``; ``excluded_tail`` counts electrons
outside ``N_src`` and is not rescaled; Stokes-unit assumption allowances
are divided by ``N_src``); a concrete ``delta N_src = 0`` is a zero
``bound`` (amplitude declared exact) and a negative one raises. The first
``AssumptionRecord`` of the moments gets ``N_src |C| Delta`` from ``moments.discrepancy`` or is
``unbounded``; every further record is ``unbounded``. Assumptions forced by
the basis (phase-route screens, the continuum kernel's isotropic pitch) are
``unbounded`` and override a moment-level term of the same name, since the
retained moments do not resolve what they constrain. ``assumption_allowances``
(keyword addition, ``{name: ErrorTerm}`` in Stokes units) replaces the term
of a named assumption, forced or moment-level, by a caller-supplied
allowance (for example ``bounds.screen_factorisation_bound`` for
``independent_screen``); its kind is kept (``not_applicable`` raises) and the
note records the source.
``samples`` with ``kernel`` (keyword addition) probe the remainder inputs on
the basis phase route, run the Support check of ``kernel.truncation_error``
(harmonic tail) and of a declared-complete ``excluded_tail`` (samples outside
the Support make both ``unbounded``, as in :func:`direct_channel_average`;
traced samples keep the declared zero and its note says it was not checked),
and leave ``basis_remainder`` ``unbounded`` when the order-``N+1``
derivative probe is not in ``basis.certified_orders``.

Not certified: physical adequacy of the kernel (``physical_kernel`` is
whatever the basis declares), feasibility of the supplied moments, the truth
of any supplied envelope, and quadrature error beyond the ``numerical`` term
the basis reports. ``to_dict``/``summary`` need concrete values (eager);
``predict`` itself is ``equinox.filter_jit``- and ``grad``-safe in the
amplitude and the moments (assumption names are string leaves).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from . import _budget_terms as terms
from ._allowances import allowance_notes, check_allowances, split_assumption_terms
from ._direct import direct_channel_average, excluded_tail_term, truncation_term
from ._prediction import LABEL, Prediction
from .bounds import RemainderInputs
from .errors import ErrorBudget, ErrorTerm


def _check_reference(m, moments_reference, basis_reference):
    """``error_if`` on any difference between the two references (relative 1e-12)."""
    pairs = []
    for name in ("gamma0", "B0", "depth_ref"):
        pairs.append((getattr(moments_reference, name), getattr(basis_reference, name)))
    pairs.extend(zip(moments_reference.scales, basis_reference.scales))
    a = jnp.stack([jnp.asarray(x, dtype=float) for x, _ in pairs])
    b = jnp.stack([jnp.asarray(y, dtype=float) for _, y in pairs])
    mismatch = jnp.any(jnp.abs(a - b) > 1e-12 * (jnp.abs(a) + jnp.abs(b)))
    return eqx.error_if(m, mismatch, "moments and basis use different references")


_NONSMOOTH = "basis_remainder unbounded"


def _basis_remainder(errors, basis, amplitude):
    nonsmooth = [n for n in basis.provenance.notes if n.startswith(_NONSMOOTH)]
    if nonsmooth:  # build_basis(allow_nonsmooth=True) on a line kernel
        return ErrorTerm.unbounded(nonsmooth[0], "N_src <rho_nu>")
    if isinstance(errors, ErrorTerm):  # the sample probe was refused
        return errors
    if errors is None:
        return ErrorTerm.unbounded(
            "no RemainderInputs supplied (eq: local response remainder)",
            "N_src <rho_nu>",
        )
    if not isinstance(errors, RemainderInputs):
        raise ValueError("errors must be a RemainderInputs")
    depth_coefficients = None
    if basis.truncation.depth_degree is not None:
        columns = [i for i, row in enumerate(basis.index.h2) if row[4] == 0]
        depth_coefficients = basis.P_basis[:, jnp.asarray(columns, dtype=int)]
    return errors.basis_remainder(
        basis, amplitude, depth_coefficients=depth_coefficients
    )


def _probed_errors(samples, basis, kernel):
    """``RemainderInputs.from_samples`` on the basis phase route, or an
    ``unbounded`` term when the order-``N+1`` probe is not validated."""
    N = basis.index.truncation.N
    validated = tuple(int(o) for o in basis.certified_orders)
    if N + 1 not in validated:
        return ErrorTerm.unbounded(
            f"the derivative probe of order N+1={N + 1} is outside "
            f"basis.certified_orders={validated} (orders validated against finite "
            "differences), so predict(samples=...) cannot probe the remainder; pass "
            "errors=RemainderInputs(...) or errors=RemainderInputs.from_samples(..., "
            "derivative_envelope=<array>)",
            "N_src <rho_nu>",
        )
    return RemainderInputs.from_samples(
        samples,
        basis,
        kernel=kernel,
        angular_residual="probe",
        phase=getattr(basis, "phase", None),
    )


def _discrepancy_note(basis, moments, allowances):
    """Where the moment-level discrepancy entered the budget, if anywhere."""
    discrepancy = moments.discrepancy
    if discrepancy is None:
        return None
    forced = {record.name for record in basis.provenance.assumptions}
    if moments.assumptions and moments.assumptions[0].name in allowances:
        return (
            f"moment-level discrepancy of kind '{discrepancy.kind}' not propagated: "
            f"the term of assumption '{moments.assumptions[0].name}' is the "
            "allowance supplied by caller"
        )
    if moments.assumptions and moments.assumptions[0].name in forced:
        return (
            f"moment-level discrepancy of kind '{discrepancy.kind}' not propagated: "
            f"assumption '{moments.assumptions[0].name}' is forced by the basis and "
            "stays unbounded"
        )
    slot = "assumption" if moments.assumptions else "statistical_input"
    return (
        f"moment-level discrepancy of kind '{discrepancy.kind}' propagated as "
        f"N_src |C| Delta ({slot})"
    )


def _merge_provenance(basis, moments, allowances):
    present = {record.name for record in basis.provenance.assumptions}
    new = tuple(r for r in moments.assumptions if r.name not in present)
    provenance = basis.provenance.with_assumptions(*new)
    note = _discrepancy_note(basis, moments, allowances)
    notes = allowance_notes(allowances) + (() if note is None else (note,))
    return provenance.with_notes(*notes) if notes else provenance


def _unit_terms(basis, moments, m, absC, errors, statistical_input, truncation):
    """Per-electron (unit-amplitude) slots and assumption pairs."""
    n_ch, kernel_terms = basis.n_ch, basis.kernel_terms
    slots = {
        "basis_remainder": _basis_remainder(errors, basis, 1.0),
        "statistical_input": terms.statistical_term(
            statistical_input, moments, absC, 1.0, n_ch
        ),
        "physical_kernel": kernel_terms.physical_kernel,
        "harmonic_truncation": truncation,
        "numerical": kernel_terms.numerical,
        "screen_exponent": kernel_terms.screen_exponent,
    }
    for name in (
        "physical_kernel",
        "harmonic_truncation",
        "numerical",
        "screen_exponent",
    ):
        slots[name] = terms.scaled_kernel_term(slots[name], 1.0, m, n_ch)
    moment_terms = terms.assumption_terms(
        moments.assumptions, moments.discrepancy, absC, 1.0, n_ch
    )
    assumption = terms.merge_forced_terms(moment_terms, basis.provenance.assumptions)
    return slots, assumption


def predict(
    basis,
    moments,
    *,
    amplitude,
    errors=None,
    statistical_input=None,
    amplitude_uncertainty=None,
    excluded_tail=None,
    depth_model=None,
    samples=None,
    kernel=None,
    assumption_allowances=None,
) -> Prediction:
    """Finite joint response ``N_src C m`` with its budget (module docstring).

    ``basis`` is a ``SpectralBasis``; ``moments`` a ``JointMoments`` on the
    same index and reference. ``errors``: ``RemainderInputs`` (or probed from
    ``samples`` with ``kernel``). ``statistical_input``: ``Delta_a``
    ``(n_real,)`` or an ``ErrorTerm`` over ``m`` (added to an unattributed
    moment-level discrepancy). ``amplitude_uncertainty``: scalar
    ``delta N_src`` (``equinox.error_if`` when ``amplitude=0`` and a valued
    ``depth_model`` must be rescaled), or an ``ErrorTerm`` taken as the
    whole slot. ``excluded_tail``/``depth_model``: ``ErrorTerm``
    or ``(n_ch, 4)`` arrays (``bound``). ``samples`` (with ``kernel``) also
    replace the harmonic-truncation term by ``kernel.truncation_error`` on
    the samples (Support check, tail estimate) and check a declared-complete
    excluded tail against them. ``assumption_allowances``: ``{name:
    ErrorTerm}`` in Stokes units for named assumptions (``ValueError`` for an
    unknown name or the kind ``not_applicable``). Units and what is not certified: module
    docstring; the returned budget lists every missing input as
    ``unbounded`` rather than assuming it zero.
    """
    index = basis.index
    if moments.index != index:
        raise ValueError("moments.index must equal basis.index")
    if samples is not None and kernel is None:
        raise ValueError("samples need kernel= (the kernel that built the basis)")
    amplitude = terms.as_amplitude(amplitude)
    n_ch = basis.n_ch
    C = basis.response_matrix()
    if C.shape != (4 * n_ch, index.n_real):
        raise ValueError("basis.response_matrix() must have shape (4 n_ch, n_real)")
    m = _check_reference(moments.to_vector(), moments.reference, basis.reference)
    per_electron = (C @ m).reshape(n_ch, 4)
    stokes = amplitude * per_electron
    absC = jnp.abs(C)
    if errors is not None and not isinstance(errors, RemainderInputs):
        raise ValueError("errors must be a RemainderInputs")
    if errors is None and samples is not None:
        errors = _probed_errors(samples, basis, kernel)
    truncation = basis.kernel_terms.harmonic_truncation
    if samples is not None:
        truncation = truncation_term(
            kernel,
            basis.support,
            basis.channels,
            samples,
            getattr(basis, "phase", None),
            reference=basis.reference,
        )
    slots, pairs = _unit_terms(
        basis, moments, m, absC, errors, statistical_input, truncation
    )
    allowances = check_allowances(
        assumption_allowances, [name for name, _ in pairs], n_ch
    )
    assumption, per_electron_assumptions, stokes_assumptions = split_assumption_terms(
        pairs, allowances, amplitude
    )
    depth = terms.depth_model_term(depth_model, n_ch)
    amplitude_slot = terms.amplitude_term(
        amplitude_uncertainty,
        per_electron,
        n_ch,
        per_electron_errors=tuple(slots.items()) + per_electron_assumptions,
        stokes_errors=(("depth_model", depth),) + stokes_assumptions,
        amplitude=amplitude,
    )
    budget = ErrorBudget(
        **{name: term.scaled(amplitude) for name, term in slots.items()},
        excluded_tail=excluded_tail_term(basis.support, excluded_tail, n_ch, samples),
        depth_model=depth,
        amplitude=amplitude_slot,
        assumption=assumption,
    )
    return Prediction(
        stokes=stokes,
        amplitude=amplitude,
        moments=moments,
        budget=budget,
        provenance=_merge_provenance(basis, moments, allowances),
        channels=basis.channels,
    )


__all__ = ["Prediction", "predict", "direct_channel_average", "LABEL"]
