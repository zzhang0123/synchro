"""Eager validation and provenance records of ``build_basis`` (private).

Split out of ``syncmoments/model/basis.py`` to keep that file under 400 lines.
Every check here runs on concrete inputs with NumPy and raises ``ValueError``
at call time; nothing here is traced. See ``syncmoments.model.basis`` for the
contract these checks enforce and what it does not certify.
"""

from __future__ import annotations

from . import _basis_core as _core
from .channels import Channels
from .errors import AssumptionRecord
from .index import Truncation
from .kernels import KernelModel, required_m_max
from .moments import Reference, Support
from .phase import PhaseWeights

CONVERGENCE = ("angular", "full", False)
CERTIFIED_ORDERS = (0, 1, 2, 3)
"""Derivative orders validated by finite-difference tests (finite checks, not
certificates); the name is kept for API compatibility."""
UNITS = (
    "per-electron channel Stokes per unit z-moment: erg/s/sr (unit_peak) or "
    "erg/s/sr/Hz (unit_integral); z = ((gamma-gamma0)/s_gamma, (B-B0)/s_B, "
    "(depth-depth_ref)/s_depth), Gauss, Hz, rad/m^2"
)
ISOTROPIC = AssumptionRecord(
    "isotropic_pitch",
    "extra eq: pitch angle expansion [required by ContinuumKernel]",
    (("gamma", "B", "eta", "phi", "depth"), ("mu",)),
    "uniform_mu",
    (),
    0,
    "unbounded",
)

# -- validation ------------------------------------------------------------------------------


def check_types(kernel, channels, truncation, reference, support):
    if not isinstance(kernel, KernelModel):
        raise ValueError("kernel must implement the KernelModel protocol")
    if not isinstance(channels, Channels):
        raise ValueError("channels must be a Channels")
    if not isinstance(truncation, Truncation):
        raise ValueError("truncation must be a Truncation (not a MomentIndex)")
    if not isinstance(reference, Reference):
        raise ValueError("reference must be a Reference")
    if not isinstance(support, Support):
        raise ValueError("support must be a Support")


def check_reference(reference, support, notes):
    gamma0 = _core.concrete(reference.gamma0, "reference.gamma0")
    B0 = _core.concrete(reference.B0, "reference.B0")
    if not gamma0 > 1.0 or not B0 > 0.0:
        raise ValueError("reference needs gamma0 > 1 and B0 > 0")
    bounds = {
        "gamma": tuple(_core.concrete(v, "support.gamma") for v in support.gamma),
        "B": tuple(_core.concrete(v, "support.B") for v in support.B),
        "depth": tuple(_core.concrete(v, "support.depth") for v in support.depth),
    }
    for name in ("gamma", "B"):
        if not bounds[name][0] < bounds[name][1]:
            raise ValueError(f"support.{name} must satisfy lo < hi")
    if not bounds["depth"][0] <= bounds["depth"][1]:
        raise ValueError("support.depth must satisfy lo <= hi")
    depth_ref = _core.concrete(reference.depth_ref, "reference.depth_ref")
    for name, value in (("gamma0", gamma0), ("B0", B0), ("depth_ref", depth_ref)):
        lo, hi = bounds[name.rstrip("0").replace("depth_ref", "depth")]
        if not lo <= value <= hi:
            notes.append(
                f"reference {name}={value:g} lies outside the declared support "
                f"[{lo:g}, {hi:g}]; the remainder bound covers reference-to-support "
                "segments"
            )


def resolve_cross_route(cross_route, convergence) -> bool:
    """Whether ``build_basis`` runs the product-vs-tensor route check.

    ``None`` (default) runs it only with ``convergence="full"``; ``True`` or
    ``False`` decide explicitly. ``ValueError`` for a non-boolean value or
    ``cross_route=True`` with ``convergence=False``.
    """
    if cross_route is None:
        return convergence == "full"
    if not isinstance(cross_route, bool):
        raise ValueError(
            f"cross_route must be None, True or False, got {cross_route!r}"
        )
    if cross_route and convergence is False:
        raise ValueError("cross_route=True needs convergence='angular' or 'full'")
    return cross_route


def check_phase(phase, truncation):
    if not isinstance(phase, PhaseWeights):
        raise ValueError("phase must implement PhaseWeights")
    if phase.max_b() != truncation.max_b():
        raise ValueError(
            f"phase.max_b()={phase.max_b()} must equal truncation.max_b()="
            f"{truncation.max_b()} (use depth_degree=0 for screen routes)"
        )
    if phase.n_weights != truncation.max_b() + 1:
        raise ValueError("phase.n_weights must equal truncation.max_b() + 1")


def check_kernel(kernel, channels, truncation, support, allow, notes):
    allow_truncated, allow_nonsmooth = allow
    if "uniform_mu" in kernel.required_closures and truncation.L_mu > 0:
        raise ValueError(
            f"{type(kernel).__name__} requires the uniform_mu closure "
            "(isotropic pitch): use L_mu = 0"
        )
    line_kernel = hasattr(kernel, "line_frequencies")
    if line_kernel and channels.smoothness < truncation.N + 1:
        message = (
            f"channel smoothness {channels.smoothness} < N + 1 = {truncation.N + 1}: "
            "derivatives of a Dirac-line kernel through the channel edges are not "
            "defined and the basis remainder is unbounded"
        )
        if not allow_nonsmooth:
            raise ValueError(message + " (pass allow_nonsmooth=True to proceed)")
        notes.append("basis_remainder unbounded: " + message)
    if hasattr(kernel, "m_max"):
        required = required_m_max(support, channels)
        if required > _core.M_REGIME:
            raise ValueError(
                f"required_m_max={required} lies outside the harmonic design regime "
                f"(m <~ {_core.M_REGIME}); use ContinuumKernel with isotropic_pitch"
            )
        if kernel.m_max < required and not allow_truncated:
            raise ValueError(
                f"m_max={kernel.m_max} < required_m_max={required}: lines above m_max "
                "meet a channel on the declared support; raise m_max, pass "
                "allow_truncated=True (tail term unbounded or estimated), or use "
                "ContinuumKernel"
            )
        if kernel.m_max < required:
            notes.append(
                f"harmonic sum truncated at m_max={kernel.m_max} < required_m_max="
                f"{required} (allow_truncated=True)"
            )


def truncation_record(index) -> tuple:
    """Hashable ``(key, value)`` record of the index layout for ``Provenance``.

    ``max_orders`` is ``Truncation.max_orders``: ``None`` for the uncapped
    cutoff (non-binding caps normalise to ``None``) or the binding caps
    ``(N_gamma, N_B, N_depth)`` with ``None`` for an uncapped variable, so a
    capped (lower-set) layout is identifiable from the record alone; in
    strict JSON it is ``null`` or a three-entry list.
    """
    t = index.truncation
    return (
        ("L_mu", t.L_mu),
        ("L_eta", t.L_eta),
        ("N", t.N),
        ("depth_degree", t.depth_degree),
        ("max_orders", t.max_orders),
        ("parity", index.parity),
        ("components", index.components),
        ("n0", index.n0),
        ("n2", index.n2),
        ("n_real", index.n_real),
    )


__all__ = [
    "CERTIFIED_ORDERS",
    "CONVERGENCE",
    "ISOTROPIC",
    "UNITS",
    "check_types",
    "resolve_cross_route",
    "check_reference",
    "check_phase",
    "check_kernel",
    "truncation_record",
]
