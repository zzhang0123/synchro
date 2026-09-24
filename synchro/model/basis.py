"""Spectral basis of the finite joint response (``synchro.model.basis``).

LABEL: ``eq: channel derivative coefficients`` (the columns
``X~_{j;lk;rsb} = d_z^r d_z^s [w_b H_{X,j;lk}] / (r! s!)`` at the reference,
``b!`` inside the phase weight), ``eq: finite joint response`` (the
contraction ``S_j = A sum_a C_{ja} m_a`` that :meth:`SpectralBasis.response_matrix`
prepares), ``extra eq: finite fit model`` (the real layout of ``C``), ``eq:
angular parity`` (``I`` columns on ``l+k`` even, ``V`` on odd rows).

:func:`build_basis` is an eager function: it validates concrete inputs with
NumPy, records :class:`~synchro.model.errors.Provenance`, and calls the
jitted :func:`_build_core`, which evaluates ``kernel.angular_projection`` at
``(gamma0, B0)`` and its nested ``jax.jacfwd`` derivatives in the
dimensionless displacements ``z = ((gamma - gamma0)/s_gamma, (B - B0)/s_B)``
through the total degree ``N``. ``jax.jacfwd`` with respect to the reference
leaves applies to :func:`_build_core`, not to :func:`build_basis`.

Units: the bases are per-electron channel Stokes responses per unit
``z``-moment, erg/s/sr for ``unit_peak`` channels (band-integrated) and
erg/s/sr/Hz for ``unit_integral``; the polynomial test kernel carries the
units of its coefficients. ``C`` has shape ``(4 n_ch, n_real)``, channel-major
with Stokes order ``I, Q, U, V``: ``C[I_j, h0] = I~``, ``C[V_j, h0] = V~``,
and with ``P~ = a + i b``: ``C[Q_j, Re] = a``, ``C[Q_j, Im] = -b``,
``C[U_j, Re] = b``, ``C[U_j, Im] = a``.

Validated against finite differences / exact checks here (finite checks, not
certificates): the algebra of the columns (exact on the polynomial test
kernel), nested derivatives of orders 0..3 against finite differences on the
harmonic kernel (``certified_orders``, a name kept for compatibility), and the
manuscript Section 5.3.1 coefficients to 1e-6 of the largest coefficient per
Stokes. Not certified:
quadrature error of the angular projection (the ``numerical`` term, the
route refined against itself, is an estimate, not a bound), derivative orders above 3, the physical
adequacy of the kernel (``physical_kernel``), the Taylor remainder of the
basis (``bounds.RemainderInputs``, which needs ``smoothness >= N+1``), and
harmonic tails when ``allow_truncated=True``.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from . import _basis_core as _core
from .channels import Channels
from .errors import ErrorTerm, Provenance
from .index import MomentIndex, Truncation
from .moments import Reference, Support
from .phase import PhaseWeights, TaylorPhase
from ._basis_checks import (
    CERTIFIED_ORDERS,
    CONVERGENCE,
    ISOTROPIC,
    UNITS,
    check_kernel,
    check_phase,
    check_reference,
    check_types,
    resolve_cross_route,
    truncation_record,
)

_build_core = eqx.filter_jit(_core.build_core)
"""Jitted ``(I_basis, V_basis, P_basis)`` at the reference; see ``_basis_core.build_core``."""
_TERMS = ("physical_kernel", "harmonic_truncation", "numerical", "screen_exponent")


class KernelTerms(eqx.Module):
    """Per-electron kernel-side slots of ``eq: channel error budget``.

    ``physical_kernel`` (``E_phys``), ``harmonic_truncation`` (omitted lines,
    ``E_num``), ``numerical`` (refined-rule estimate plus, for the continuum
    kernel, the ``F``/``G`` tail bound; ``E_num``) and ``screen_exponent``
    (``eq: screen exponent error``). Values are ``(n_ch, 4)`` per electron,
    except ``numerical``: a per-column envelope ``(4 n_ch, n_real)`` that
    ``predict`` contracts with ``|m|``; ``predict`` scales all by the amplitude.
    Units: the per-electron channel Stokes units of the basis. Each term
    carries its own kind; ``physical_kernel`` is a declaration, never a
    package certificate.
    """

    LABEL: ClassVar[str] = "eq: channel error budget"
    physical_kernel: ErrorTerm
    harmonic_truncation: ErrorTerm
    numerical: ErrorTerm
    screen_exponent: ErrorTerm

    def __check_init__(self):
        for name in _TERMS:
            if not isinstance(getattr(self, name), ErrorTerm):
                raise ValueError(f"{name} must be an ErrorTerm")


class SpectralBasis(eqx.Module):
    """The response of every retained moment row in every channel.

    ``I_basis``, ``V_basis`` are real ``(n_ch, n0)`` over ``index.h0`` (zero
    on the parity-forbidden rows); ``P_basis`` is complex ``(n_ch, n2)`` over
    ``index.h2`` and includes the phase weight ``w_b``. ``kernel_terms`` holds
    the per-electron kernel-side error slots; ``provenance`` is the static
    hashable record; ``certified_orders`` (name kept for compatibility) lists
    the orders validated by finite-difference tests (finite checks, not
    certificates); ``phase`` (keyword
    addition) is the phase route used, kept so that :func:`basis_convergence`
    can rebuild the basis. Shape errors raise ``ValueError``. Units: per-
    electron channel Stokes per unit ``z``-moment (module docstring).
    Assumes what ``provenance.assumptions`` lists (phase-route screens,
    ``isotropic_pitch`` for the continuum kernel). Not certified: the
    quadrature error beyond the ``numerical`` estimate, derivative orders
    outside ``certified_orders``, and the kernel's physical adequacy.
    """

    index: MomentIndex = eqx.field(static=True)
    truncation: Truncation = eqx.field(static=True)
    channels: Channels
    reference: Reference
    support: Support
    I_basis: jax.Array
    V_basis: jax.Array
    P_basis: jax.Array
    kernel_terms: KernelTerms
    provenance: Provenance = eqx.field(static=True)
    certified_orders: tuple[int, ...] = eqx.field(static=True, default=(0, 1, 2))
    phase: PhaseWeights | None = None
    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: channel derivative coefficients",
        "eq: finite joint response",
        "extra eq: finite fit model",
    )

    def __check_init__(self):
        n_ch, n0, n2 = self.channels.n_ch, self.index.n0, self.index.n2
        for name, shape in (
            ("I_basis", (n_ch, n0)),
            ("V_basis", (n_ch, n0)),
            ("P_basis", (n_ch, n2)),
        ):
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f"{name} must have shape {shape}, got {got}")
        if jnp.iscomplexobj(self.I_basis) or jnp.iscomplexobj(self.V_basis):
            raise ValueError("I_basis and V_basis must be real")
        if self.truncation != self.index.truncation:
            raise ValueError("truncation must be the index's truncation")

    @property
    def n_ch(self) -> int:
        return self.channels.n_ch

    def response_matrix(self) -> jax.Array:
        """``C`` of shape ``(4 n_ch, n_real)``; see the module docstring for the layout."""
        return _core.response_matrix(
            self.index, self.n_ch, self.I_basis, self.V_basis, self.P_basis
        )


# -- construction ---------------------------------------------------------------------------


def build_basis(
    kernel,
    channels,
    truncation,
    reference,
    *,
    support,
    phase=None,
    quadrature=None,
    allow_truncated=False,
    convergence="angular",
    E_phys=None,
    allow_nonsmooth=False,
    cross_route=None,
) -> SpectralBasis:
    """Build the :class:`SpectralBasis` of ``kernel`` on ``channels`` at ``reference``.

    Eager: every input must be concrete. Validation (``ValueError``): types;
    ``gamma0 > 1``, ``B0 > 0``; support ordering; ``phase.max_b()`` equal to
    ``truncation.max_b()`` (``phase=None`` selects ``TaylorPhase(max_b)``);
    ``ContinuumKernel`` with ``L_mu > 0``; for Dirac-line kernels a channel
    ``smoothness < N + 1`` unless ``allow_nonsmooth=True`` (keyword addition;
    the remainder is then unbounded and noted); ``m_max < required_m_max``
    unless ``allow_truncated=True``, and any ``required_m_max`` beyond the
    harmonic design regime. Kernels without ``V`` drop the odd-parity rows
    (``MomentIndex.build(components=kernel.components)``) and record
    ``isotropic_pitch`` when they require ``uniform_mu``.

    ``quadrature`` (``None`` = the kernel default) selects the angular route.
    ``convergence="angular"`` (default) fills ``kernel_terms.numerical`` with
    the per-column envelope of :func:`basis_convergence` (same route at 2x
    nodes for the harmonic kernel, 2x ``n_nu``/``n_eta`` for the continuum);
    ``"full"`` also doubles the Bessel or ``F``/``G`` resolution, ``False``
    leaves it ``unbounded``. ``cross_route`` (keyword addition; ``None`` =
    ``convergence == "full"``) also rebuilds the harmonic basis on the other
    angular route at 2x nodes and records the product-vs-tensor comparison
    as the named finite check ``"route check ..."`` (``check_route`` in
    ``provenance.quadrature``); it is not part of the budget. ``E_phys`` (an
    ``ErrorTerm``) replaces ``kernel.physical_error``. Cost: one nested-
    ``jacfwd`` evaluation of the projection through order ``N``, one
    convergence rebuild, and one more with the route check. Units and
    what is not certified: see :class:`SpectralBasis` and the module
    docstring; ``E_phys`` and ``allow_truncated`` are caller declarations.
    """
    check_types(kernel, channels, truncation, reference, support)
    if convergence not in CONVERGENCE:
        raise ValueError(f"convergence must be one of {CONVERGENCE}")
    cross_route = resolve_cross_route(cross_route, convergence)
    if E_phys is not None and not isinstance(E_phys, ErrorTerm):
        raise ValueError("E_phys must be an ErrorTerm or None")
    phase = TaylorPhase(truncation.max_b()) if phase is None else phase
    check_phase(phase, truncation)
    notes: list[str] = []
    check_reference(reference, support, notes)
    check_kernel(
        kernel, channels, truncation, support, (allow_truncated, allow_nonsmooth), notes
    )
    index = MomentIndex.build(truncation, components=tuple(kernel.components))
    assumptions = tuple(phase.forced_assumptions())
    if "uniform_mu" in kernel.required_closures:
        assumptions += (ISOTROPIC,)
    if "V" not in index.components:
        notes.append(f"V not modelled [{type(kernel).__name__}]")
    notes.append("incident polarisation not modelled")
    route = getattr(kernel, "quadrature", None) if quadrature is None else quadrature
    I_basis, V_basis, P_basis = _build_core(
        kernel, channels, reference, phase, index=index, quadrature=quadrature
    )
    terms = KernelTerms(
        physical_kernel=(kernel.physical_error(channels) if E_phys is None else E_phys),
        harmonic_truncation=kernel.truncation_error(
            support, channels, reference=reference, phase=phase
        ),
        numerical=ErrorTerm.unbounded(
            "convergence=False: no two-quadrature estimate", "E_num"
        ),
        screen_exponent=_core.screen_exponent_term(
            kernel, channels, reference, phase, quadrature
        ),
    )
    provenance = Provenance(
        package_version=_core.package_version(),
        kernel=tuple(kernel.describe()),
        channels=tuple(channels.describe()),
        truncation=truncation_record(index),
        reference=tuple(reference.describe()),
        support=tuple(support.describe()),
        phase_route=tuple(phase.describe()),
        quadrature=(
            ("route", route),
            ("convergence", convergence),
            ("numerical_route", None),
            ("check_route", None),
        ),
        assumptions=assumptions,
        units=UNITS,
        numerics=_core.numerics_record(kernel, channels, reference, route),
        certified_orders=CERTIFIED_ORDERS,
        finite_checks=(),
        notes=tuple(notes),
    )
    basis = SpectralBasis(
        index=index,
        truncation=truncation,
        channels=channels,
        reference=reference,
        support=support,
        I_basis=I_basis,
        V_basis=V_basis,
        P_basis=P_basis,
        kernel_terms=terms,
        provenance=provenance,
        certified_orders=CERTIFIED_ORDERS,
        phase=phase,
    )
    if convergence is False:
        return basis
    return _with_convergence(
        basis, kernel, full=(convergence == "full"), cross_route=cross_route
    )


def _with_convergence(basis, kernel, *, full, cross_route) -> SpectralBasis:
    numerical = basis_convergence(basis, kernel, factor=2, full=full)
    terms = KernelTerms(
        physical_kernel=basis.kernel_terms.physical_kernel,
        harmonic_truncation=basis.kernel_terms.harmonic_truncation,
        numerical=numerical,
        screen_exponent=basis.kernel_terms.screen_exponent,
    )
    route = dict(basis.provenance.quadrature)
    own = _core.refined(kernel, basis.channels, route["route"], 2, full)
    other = (
        _core.refined(kernel, basis.channels, route["route"], 2, full, cross=True)
        if cross_route
        else None
    )
    checks = (f"numerical: {numerical.note}",) if numerical.value is not None else ()
    if other is not None:
        delta = basis_convergence(basis, kernel, factor=2, full=full, cross_route=True)
        ratio = _core.column_ratio(delta.value, basis.response_matrix())
        checks += (
            f"route check ({other[3]} vs the basis): max_a max_j |dC_ja| / "
            f"max_j |C_ja| = {ratio:.3g}; scale-invariant, not part of the budget",
        )
    provenance = Provenance(
        **{
            **basis.provenance.__dict__,
            "quadrature": (
                ("route", route["route"]),
                ("convergence", route["convergence"]),
                ("numerical_route", None if own is None else own[3]),
                ("check_route", None if other is None else other[2]),
            ),
            "finite_checks": basis.provenance.finite_checks + checks,
        }
    )
    return SpectralBasis(
        index=basis.index,
        truncation=basis.truncation,
        channels=basis.channels,
        reference=basis.reference,
        support=basis.support,
        I_basis=basis.I_basis,
        V_basis=basis.V_basis,
        P_basis=basis.P_basis,
        kernel_terms=terms,
        provenance=provenance,
        certified_orders=basis.certified_orders,
        phase=basis.phase,
    )


def basis_convergence(
    basis, kernel, *, factor=2, full=False, cross_route=False
) -> ErrorTerm:
    """Per-column ``numerical`` envelope ``|C' - C|``, per electron, ``(4 n_ch, n_real)``.

    ``C'`` is the basis rebuilt by :func:`_basis_core.refined`: the same
    angular route at ``factor`` times the node counts for the harmonic
    kernel (``cross_route=True``, keyword addition: the other route, the
    finite route check), ``factor`` times ``n_nu`` and ``n_eta`` for the
    continuum kernel; ``full`` also multiplies the Bessel or ``F``/``G``
    resolution. ``predict`` contracts the envelope with ``|m|``, which is
    invariant under ``Reference.scales`` and bounds ``|(C' - C) m|``. The
    continuum kernel adds its ``F``/``G`` tail bound at ``(gamma0, B0,
    eta = 0)`` on the mass column ``(0,0,0,0,0)`` (so it is multiplied by
    ``|m_0|``). An ``estimate``: the refined rule's own error and
    non-monotone convergence are not bounded. Kernels without quadrature
    controls (or without a second route) give ``not_applicable``.
    ``basis.phase`` must be set.
    """
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise ValueError("factor must be an integer >= 2")
    if basis.phase is None:
        raise ValueError("basis_convergence needs basis.phase (set by build_basis)")
    route = dict(basis.provenance.quadrature).get("route")
    check = _core.refined(
        kernel, basis.channels, route, factor, full, cross=bool(cross_route)
    )
    if check is None:
        return ErrorTerm.not_applicable(
            f"{type(kernel).__name__} has no quadrature controls to refine"
            + (" (or no second route)" if cross_route else "")
        )
    alt_kernel, alt_channels, alt_route, text = check
    alt = _build_core(
        alt_kernel,
        alt_channels,
        basis.reference,
        basis.phase,
        index=basis.index,
        quadrature=alt_route,
    )
    refined_C = _core.response_matrix(basis.index, basis.n_ch, *alt)
    value = jnp.abs(refined_C - basis.response_matrix())
    note = (
        f"|C({text}) - C| per column, (4 n_ch, n_real); contracted with |m| "
        "(invariant under Reference.scales)"
    )
    tail = _core.continuum_tail(kernel, basis.channels, basis.reference)
    if tail is not None:
        value = _core.on_mass_column(value, basis.index, jnp.asarray(tail.value))
        note += (
            "; plus the F/G finite-tail bound at (gamma0, B0, eta=0) on the mass "
            "column (0,0,0,0,0), times |m_0|"
        )
    return ErrorTerm(value, "estimate", note, "E_num")


__all__ = [
    "KernelTerms",
    "SpectralBasis",
    "build_basis",
    "basis_convergence",
    "CERTIFIED_ORDERS",
]
