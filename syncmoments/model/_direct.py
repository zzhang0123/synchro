"""Direct discrete-population channel average (private helper of ``predict``).

LABEL: ``eq: smooth channel kernel``, ``extra eq: faraday channel kernel``,
``eq: joint emission depth``. ``direct_channel_average`` sums the kernel's
channel modes over a ``PopulationSamples`` measure with the sky factor
``exp(2 i phi_n)`` on ``P`` and the exact per-emitter, per-line phase
``exp(i tau_m depth_n)`` (a ``TaylorPhase`` of degree 0 evaluated at each
sample's own depth); a screen route replaces that phase by the screen's
characteristic function (the sample depths are then ignored, and the screen
assumptions are recorded). A kernel requiring ``uniform_mu``
(``ContinuumKernel``, which ignores ``mu``) records ``isotropic_pitch`` as
``build_basis`` does, with an ``unbounded`` term. No Legendre or Taylor truncation enters, so
``basis_remainder`` is ``not_applicable`` and ``statistical_input`` is a
declared zero (exact average over the supplied measure). Units follow
``syncmoments.model.predict``. The population/quadrature error of the discrete
measure relative to a continuous population is not assessed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import _budget_terms as terms
from ._allowances import allowance_notes, check_allowances, split_assumption_terms
from ._prediction import Prediction
from ._bound_helpers import _tau_max
from ._basis_checks import ISOTROPIC
from ._basis_core import package_version
from ._support_check import UNCHECKED_COMPLETE, outside_support
from .errors import ErrorBudget, ErrorTerm, Provenance
from .index import MomentIndex, Truncation
from .moments import JointMoments, Reference
from .phase import CumulantScreen, TaylorPhase

_DIRECT_INDEX = MomentIndex.build(Truncation(0, 0, 0))
_LEAVES = ("gamma", "B", "mu", "eta", "phi", "depth")


def _route(phase):
    """``(route, use_sample_depth)``: exact per-emitter phase for Taylor routes."""
    if phase is None or isinstance(phase, TaylorPhase):
        return TaylorPhase(0), True
    return phase, False


def _forced_records(route, kernel):
    """Assumptions the direct average makes: the phase route's, plus
    ``isotropic_pitch`` for a kernel requiring ``uniform_mu`` (as ``build_basis``)."""
    forced = tuple(route.forced_assumptions())
    if "uniform_mu" in getattr(kernel, "required_closures", ()):
        forced += (ISOTROPIC,)
    return forced


def sample_batch(kernel, channels, requested, size) -> int:
    """Samples per ``lax.map`` step: ``requested``, capped by ``size`` and by the
    kernel's ``samples_per_step(channels, requested)`` when it has one (its
    ``chunk_budget`` over the per-sample ``channel_modes`` work). At least one."""
    batch = max(1, min(int(requested), int(size)))
    per_step = getattr(kernel, "samples_per_step", None)
    return batch if per_step is None else max(1, min(batch, per_step(channels, batch)))


def _per_sample(samples, kernel, channels, route, use_sample_depth, depth_ref, batch):
    """``(S, n_ch, 4)`` per-electron channel Stokes of every sample (sky basis)."""

    def one(leaf):
        gamma, B, mu, eta, phi, depth = leaf
        ref = depth if use_sample_depth else depth_ref
        modes = kernel.channel_modes(
            channels,
            gamma,
            B,
            mu,
            eta,
            phase=route,
            depth_ref=ref,
            s_depth=jnp.ones(()),
        )
        P = jnp.exp(2j * phi) * modes.P[0]
        return jnp.stack([modes.I, jnp.real(P), jnp.imag(P), modes.V], axis=-1)

    leaves = tuple(getattr(samples, name) for name in _LEAVES)
    batch = sample_batch(kernel, channels, batch, samples.size)
    return jax.lax.map(one, leaves, batch_size=batch)


def _default_reference(samples):
    """Weighted means (eager); the ``(0, 0, 0)`` moments do not depend on it."""
    w = np.asarray(samples.normalised_weights())
    gamma0 = max(float(w @ np.asarray(samples.gamma)), 1.0 + 1e-9)
    B0 = max(float(w @ np.asarray(samples.B)), 1e-300)
    return Reference(gamma0, B0, float(w @ np.asarray(samples.depth)))


def truncation_term(kernel, support, channels, samples, route, reference=None):
    """``kernel.truncation_error`` with the samples (so its Support check runs)."""
    kwargs = dict(samples=samples, reference=reference, phase=route)
    if support is None:
        try:
            return kernel.truncation_error(None, channels, **kwargs)
        except AttributeError:
            return ErrorTerm.unbounded(
                "harmonic-tail term needs a Support (required_m_max); none supplied",
                "E_num",
            )
    return kernel.truncation_error(support, channels, **kwargs)


def excluded_tail_term(support, excluded_tail, n_ch, samples):
    """``E_tail``; a ``Support.truncated=False`` declaration contradicted by the
    concrete samples (``_support_check.outside_support``) becomes ``unbounded``.
    Supplied samples that cannot be checked (traced) keep the declared zero,
    and its note states the unchecked hypothesis."""
    term = terms.excluded_tail_term(support, excluded_tail, n_ch)
    if excluded_tail is not None or support is None:
        return term
    if support.tail_kind() != "declared_zero":
        return term
    outside = outside_support(samples, support)
    if outside is None and samples is not None:
        return ErrorTerm(
            term.value,
            term.kind,
            f"{term.note}; {UNCHECKED_COMPLETE}",
            term.manuscript_term,
        )
    if not outside:
        return term
    return ErrorTerm.unbounded(
        "Support.truncated=False declares the population complete inside the "
        f"Support, but emitting samples lie outside it ({'; '.join(outside)}); "
        "the declared-zero excluded tail does not apply",
        "E_tail",
    )


def _screen_term(route, per_sample, w, channels, amplitude):
    """``eq: screen exponent error`` for a cumulant screen: ``A <|P_j|> (e^eps5 - 1)`` (estimate)."""
    if not isinstance(route, CumulantScreen):
        return ErrorTerm.not_applicable("no cumulant screen on the direct average")
    eps = route.exponent_remainder(_tau_max(channels))
    if eps is None:
        return ErrorTerm.unbounded(
            "cumulant screen without g5_bound: exponent remainder unbounded",
            "eq: screen exponent error",
        )
    modulus = jnp.hypot(per_sample[..., 1], per_sample[..., 2])  # (S, n_ch)
    P = amplitude * (w @ modulus) * jnp.expm1(eps)
    value = jnp.stack([jnp.zeros_like(P), P, P, jnp.zeros_like(P)], axis=1)
    return ErrorTerm(
        value,
        "estimate",
        "A <|P_j|> (exp(eps5) - 1) with eps5 = |tau_max|^5 g5/5! at the lower support "
        "edge and |P_j| the modulus of the phased line sum (below the absolute sum)",
        "eq: screen exponent error",
    )


def _provenance(
    kernel,
    channels,
    reference,
    support,
    route,
    samples,
    use_sample_depth,
    notes,
    forced,
):
    phase_route = route.describe()
    if use_sample_depth:
        phase_route = (("route", "exact per-emitter phase exp(i tau_m depth_n)"),)
    return Provenance(
        package_version=package_version(),
        kernel=kernel.describe(),
        channels=channels.describe(),
        truncation=(("route", "direct average: no Legendre or Taylor truncation"),),
        reference=reference.describe(),
        support=() if support is None else support.describe(),
        phase_route=phase_route,
        quadrature=(("route", "discrete population sum"), ("n_samples", samples.size)),
        assumptions=forced,
        units=(
            "erg/s/sr per electron times amplitude"
            if channels.normalisation == "unit_peak"
            else "erg/s/sr/Hz per electron times amplitude"
        ),
        numerics=(("kernel", kernel.describe()),),
        certified_orders=(),
        finite_checks=(),
        notes=(
            "incident polarisation not modelled",
            "direct discrete-population average; sampling error of the measure not assessed",
        )
        + tuple(notes),
    )


def direct_channel_average(
    samples,
    kernel,
    channels,
    *,
    amplitude,
    reference=None,
    phase=None,
    support=None,
    excluded_tail=None,
    amplitude_uncertainty=None,
    depth_model=None,
    batch_size=256,
    assumption_allowances=None,
) -> Prediction:
    """Exact channel average of ``samples`` under ``kernel`` (module docstring).

    ``phase=None`` or any ``TaylorPhase`` gives the exact per-emitter phase;
    a screen (``GaussianScreen``, ``CumulantScreen``, ``EmpiricalScreen``)
    is applied as the screen's characteristic function at ``reference.depth_ref``
    (sample depths ignored). ``support`` enables the harmonic-tail term
    (``bound`` zero when ``m_max >= required_m_max`` and no emitting sample
    leaves the support) and the excluded-tail tri-state (a declared-complete
    support contradicted by the samples gives ``unbounded``); ``reference`` (default: sample-weighted means, eager only)
    labels the returned ``(0, 0, 0)`` moments. Keyword additions:
    ``excluded_tail``, ``amplitude_uncertainty``, ``depth_model`` as in
    ``predict``, ``assumption_allowances`` (``{name: ErrorTerm}`` in Stokes
    units, replacing the unbounded term of a forced assumption, a screen
    route's or the ``isotropic_pitch`` of a ``uniform_mu`` kernel, as in
    ``predict``); ``batch_size >= 1`` samples per ``lax.map`` step, an upper
    cap: the kernel's ``samples_per_step`` lowers it so that one vmapped
    step stays within the kernel's ``chunk_budget`` (summation order only).
    The ``numerical`` term is unbounded (no two-quadrature estimate here).
    Provenance is built eagerly (``describe`` calls need concrete values).
    Units: ``amplitude`` times the per-electron channel Stokes of the
    kernel. Assumes the discrete measure is the population; not certified:
    sampling error against a continuous population and ``physical_kernel``.
    """
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    amplitude = terms.as_amplitude(amplitude)
    route, use_sample_depth = _route(phase)
    if reference is None:
        reference = _default_reference(samples)
    per_sample = _per_sample(
        samples,
        kernel,
        channels,
        route,
        use_sample_depth,
        reference.depth_ref,
        batch_size,
    )
    w = samples.normalised_weights()
    per_electron = jnp.einsum("s,scq->cq", w, per_sample)
    n_ch = channels.n_ch
    moments = JointMoments.from_samples(samples, _DIRECT_INDEX, reference)
    forced = _forced_records(route, kernel)
    allowances = check_allowances(assumption_allowances, [r.name for r in forced], n_ch)
    slots = {  # per electron
        "basis_remainder": ErrorTerm.not_applicable(
            "direct average: no Legendre or Taylor truncation"
        ),
        "statistical_input": ErrorTerm.declared_zero(
            "exact average over the supplied discrete measure", shape=(n_ch, 4)
        ),
        "physical_kernel": kernel.physical_error(channels),
        "harmonic_truncation": truncation_term(
            kernel, support, channels, samples, route
        ),
        "numerical": ErrorTerm.unbounded(
            "direct average: Bessel/channel quadrature and roundoff not estimated",
            "E_num",
        ),
        "screen_exponent": _screen_term(route, per_sample, w, channels, 1.0),
    }
    assumption, per_electron_assumptions, stokes_assumptions = split_assumption_terms(
        terms.forced_assumption_terms(forced, set()), allowances, amplitude
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
        excluded_tail=excluded_tail_term(support, excluded_tail, n_ch, samples),
        depth_model=depth,
        amplitude=amplitude_slot,
        assumption=assumption,
    )
    return Prediction(
        stokes=amplitude * per_electron,
        amplitude=amplitude,
        moments=moments,
        budget=budget,
        provenance=_provenance(
            kernel,
            channels,
            reference,
            support,
            route,
            samples,
            use_sample_depth,
            allowance_notes(allowances),
            forced,
        ),
        channels=channels,
    )


__all__ = [
    "direct_channel_average",
    "excluded_tail_term",
    "sample_batch",
    "truncation_term",
]
