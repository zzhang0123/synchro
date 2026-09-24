"""Manuscript error-bound helpers for the channel budget (``synchro.model.bounds``).

LABEL: ``eq: local response remainder`` (``RemainderInputs.basis_remainder``,
main-text layout), ``eq: joint screen remainder`` (``app: depth moments`` layout inputs
``depth_tail``, ``intrinsic_residual``), ``detail-eq: screen factorisation
error`` (``screen_factorisation_bound``), ``detail-eq: angular factorisation
error`` (``azimuth_factorisation_bound``) and the depth-model term of
``eq: channel error budget`` (``depth_error_bound``, implemented in
``_depth_bound.py`` and re-exported here).

Every helper returns an ``ErrorTerm`` with value shape ``(n_ch, 4)`` in the
Stokes order ``I, Q, U, V`` and the units of the amplitude times the
per-electron kernel (erg/s/sr per electron when ``amplitude=1``). A missing
input gives ``kind="unbounded"`` with the missing names in the note, never a
zero. The polarised bounds control ``|P| = |Q + iU|`` and are placed in both
the ``Q`` and ``U`` columns; ``I, V`` columns are zero for terms that act on
the polarised channel only.

Not certified: the truth of the supplied envelopes (``rho_ang``, ``H``,
absolute moments, ``sigma``, ``|Phi|``), which are external inputs; sampling
error of a discrete population; and, for ``depth_error_bound`` without an
absolute channel sum, the gap between ``|H_P|`` and ``sum_m |Q_m R_j|``
(kind ``estimate``); for the ``app: depth moments`` layout without
``absolute_depth_coefficients``, the gap between ``|c_a|`` and the absolute
line sum (kind ``estimate``). ``RemainderInputs.from_samples`` probes
envelopes at finitely many points (``estimate``), not a supremum over the
reference-to-support segments. A line-kernel basis whose channel smoothness
is below ``N + 1`` has an unbounded ``basis_remainder``.
"""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from . import _remainder_probe as probe
from ._bound_helpers import (
    C_SI_M,
    _amplitude,
    _as_float,
    _nonnegative,
    _stokes,
    _tau_max,
)
from ._depth_bound import depth_error_bound
from ._factorisation_bounds import (
    azimuth_factorisation_bound,
    screen_factorisation_bound,
)
from ._remainder_route import is_screen_route, nonsmooth_reason
from .errors import ErrorTerm

_KINDS = ("bound", "estimate")


class RemainderInputs(eqx.Module):
    """External inputs of ``eq: local response remainder``.

    ``rho_ang`` ``(n_ch, 3)``: population-averaged angular residual bounds
    per natural component ``I, Q, V``. ``H`` ``(n_ch, 3, n_lk)``: operator
    norms of the order-``N+1`` derivative tensor in ``z`` of each projected
    channel kernel (``exp(i tau depth) K_Q`` for the polarised component),
    per retained ``(l, k)`` pair in ``index.pairs`` order.
    ``absolute_moments`` ``(2, n_lk)``: row 0 is ``<|P_l P_k| ||z_2||^{N+1}>``
    over ``(z_gamma, z_B)`` (used by ``I, V``), row 1 the same with
    ``z_3 = (z_gamma, z_B, z_depth)`` (used by ``P``). ``depth_tail``
    ``(n_a,)`` and ``intrinsic_residual`` ``(n_ch,)`` are the ``app: depth moments``
    inputs ``<|chi_a| |depth - depth_ref|^{L+1}>`` and ``<|r_in|>`` of
    ``eq: joint screen remainder``. ``kind`` is ``"bound"`` or
    ``"estimate"`` and labels the returned term. ``absolute_depth_coefficients``
    ``(n_ch, n_a)`` (keyword addition) is ``sum_m |c_{a,m} R_j(nu_m)|`` over
    the lines of each channel (``int |R_j c_a(nu)| dnu`` for a continuum);
    without it the depth term uses the modulus of the phased channel
    coefficient and is an ``estimate``. All arrays are nonnegative.
    Units: ``rho_ang``, ``H`` and ``absolute_depth_coefficients`` in the
    per-electron channel Stokes units of the basis (erg/s/sr or
    erg/s/sr/Hz), moments dimensionless. Assumes the
    supplied envelopes dominate the true remainder on the reference-to-sample
    segments; the package does not verify supplied arrays, and probed
    envelopes (``from_samples``) are estimates at finitely many points.
    """

    rho_ang: jax.Array | None = None
    H: jax.Array | None = None
    absolute_moments: jax.Array | None = None
    depth_tail: jax.Array | None = None
    intrinsic_residual: jax.Array | None = None
    absolute_depth_coefficients: jax.Array | None = None
    kind: str = eqx.field(static=True, default="bound")
    LABEL: ClassVar[str] = "eq: local response remainder"

    def __init__(
        self,
        rho_ang=None,
        H=None,
        absolute_moments=None,
        depth_tail=None,
        intrinsic_residual=None,
        kind="bound",
        *,
        absolute_depth_coefficients=None,
    ):
        if kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
        fields = {
            "rho_ang": (rho_ang, 2),
            "H": (H, 3),
            "absolute_moments": (absolute_moments, 2),
            "depth_tail": (depth_tail, 1),
            "intrinsic_residual": (intrinsic_residual, 1),
            "absolute_depth_coefficients": (absolute_depth_coefficients, 2),
        }
        for name, (value, ndim) in fields.items():
            if value is not None:
                value = _as_float(value, name)
                if value.ndim != ndim:
                    raise ValueError(
                        f"{name} must have {ndim} dimensions, got {value.ndim}"
                    )
                value = _nonnegative(value, name)
            setattr(self, name, value)
        self.kind = kind

    @classmethod
    def from_samples(
        cls,
        samples,
        basis,
        *,
        kernel=None,
        derivative_envelope="probe",
        angular_residual=None,
        phase=None,
        segment_points=(0.5, 1.0),
        batch_size=64,
        absolute_depth_coefficients=None,
    ) -> "RemainderInputs":
        """Probe the remainder inputs of ``basis`` on ``samples`` (kind ``estimate``).

        ``kernel`` (keyword addition; the kernel that built ``basis``) is
        required for the ``"probe"`` envelopes. ``derivative_envelope``:
        ``"probe"`` takes the maximum over the probe points ``z = 0`` and
        ``t z_n`` (``t`` in ``segment_points``) of the Frobenius norm of the
        order-``N+1`` nested-``jacfwd`` tensor of each projected channel kernel
        in ``z_3 = (z_gamma, z_B, z_depth)`` (``exp(i tau depth) K_Q`` for
        ``P``); an array ``(n_ch, 3, n_lk)`` is used as supplied.
        ``angular_residual``: ``None`` leaves ``rho_ang`` unset (the returned
        ``basis_remainder`` is then unbounded), ``"probe"`` measures
        ``<|K_X - K_{X,L}|>`` on the samples with exact ``(gamma, B)``
        dependence, an array ``(n_ch, 3)`` is used as supplied. ``phase`` is
        the phase route the probe describes; ``None`` (default) means
        ``basis.phase``, the route the basis uses (a basis without a ``phase``
        field falls back to the Taylor route). A ``TaylorPhase`` means the
        exact per-emitter phase ``exp(i tau depth_n)``; a screen is used as is
        (no depth dependence). Passing another route than ``basis.phase``
        makes the probed ``P`` inputs describe that route, not the basis.
        ``absolute_moments`` are exact weighted sums; with ``depth_degree``
        the ``app: depth moments`` ``depth_tail`` and (when ``rho_ang`` is known)
        ``intrinsic_residual`` are filled (``basis_remainder`` ignores
        ``depth_tail`` on a screen route). ``absolute_depth_coefficients``
        is passed through. The ``"probe"`` envelope applies nested
        ``jacfwd`` of order ``N + 1``; it is refused (``ValueError``) unless
        ``N + 1`` is in ``basis.certified_orders``, the orders validated
        by finite-difference tests (0..3 on the harmonic kernel, so
        ``N <= 2``; finite checks, not certificates). A supplied ``derivative_envelope`` needs no probe and is
        not refused. ``kind`` is ``"bound"`` only when both envelopes are
        supplied arrays.
        Frobenius norms at finitely many points are not a supremum over the
        reference-to-support segments; sampling error is not assessed.
        """
        index = basis.index
        N = index.truncation.N
        if phase is None:
            phase = getattr(basis, "phase", None)
        certified = tuple(int(o) for o in basis.certified_orders)
        if isinstance(derivative_envelope, str) and N + 1 not in certified:
            raise ValueError(
                f"the derivative probe of order N+1={N + 1} is outside "
                f"basis.certified_orders={certified} (orders validated against "
                "finite differences); supply derivative_envelope as an array"
            )
        if samples.size < 1:
            raise ValueError("samples must be nonempty")
        needs_kernel = isinstance(derivative_envelope, str) or isinstance(
            angular_residual, str
        )
        if needs_kernel and kernel is None:
            raise ValueError("from_samples needs kernel= for a probe")
        n_ch, n_lk = basis.channels.n_ch, index.n_lk
        H_2d = None
        if isinstance(derivative_envelope, str):
            if derivative_envelope != "probe":
                raise ValueError("derivative_envelope must be 'probe' or an array")
            H, H_2d = probe.derivative_envelope(
                samples, basis, kernel, phase=phase, segment_points=segment_points
            )
        else:
            H = _as_float(derivative_envelope, "derivative_envelope")
            if H.shape != (n_ch, 3, n_lk):
                raise ValueError(
                    f"derivative_envelope must have shape ({n_ch}, 3, {n_lk})"
                )
        if isinstance(angular_residual, str):
            if angular_residual != "probe":
                raise ValueError("angular_residual must be None, 'probe' or an array")
            rho = probe.angular_residual(
                samples, basis, kernel, phase=phase, batch_size=batch_size
            )
        elif angular_residual is None:
            rho = None
        else:
            rho = _as_float(angular_residual, "angular_residual")
            if rho.shape != (n_ch, 3):
                raise ValueError(f"angular_residual must have shape ({n_ch}, 3)")
        absolute = probe.absolute_moments(samples, basis)
        tail = residual = None
        if index.truncation.depth_degree is not None:
            tail = probe.depth_tail(samples, basis)
            if rho is not None and H_2d is not None:
                residual = probe.intrinsic_residual(rho, H_2d, absolute, N)
        supplied_H = not isinstance(derivative_envelope, str)
        supplied_rho = rho is not None and not isinstance(angular_residual, str)
        kind = "bound" if supplied_H and supplied_rho else "estimate"
        return cls(
            rho_ang=rho,
            H=H,
            absolute_moments=absolute,
            depth_tail=tail,
            intrinsic_residual=residual,
            kind=kind,
            absolute_depth_coefficients=absolute_depth_coefficients,
        )

    def _missing(self, names):
        return [name for name in names if getattr(self, name) is None]

    def basis_remainder(
        self, basis, amplitude, *, depth_coefficients=None
    ) -> ErrorTerm:
        """``(n_ch, 4)`` remainder envelope of the finite response.

        Main-text layout (``depth_degree is None``): every column is
        ``amplitude [rho_ang + sum_lk H_lk absolute_lk / (N+1)!]`` with
        absolute-moment row 0 for ``I, V`` and row 1 for ``P``. ``app: depth
        moments`` layout: ``I, V`` as above; on a Taylor phase route ``P`` is
        ``amplitude [intrinsic_residual + tau_max^{L+1}/(L+1)! sum_a A_a
        depth_tail_a]`` with ``tau_max = 2 (c/nu_lo)^2`` per channel and
        ``A`` either ``absolute_depth_coefficients`` or ``|c_a|`` from
        ``depth_coefficients`` ``(n_ch, n_a)``, the complex ``b = 0``
        coefficients (keyword addition). ``|c_a|`` is the modulus of the phased
        line sum, which can be below the absolute line sum; that route returns
        kind ``estimate``. On a screen route the depth is not Taylor-expanded:
        ``P = amplitude intrinsic_residual`` and the emitter-depth/screen
        mismatch belongs to the ``independent_screen`` assumption term.
        A line-kernel basis with channel smoothness below ``N + 1`` returns
        an unbounded term with the reason. Missing inputs give an unbounded
        term naming them.
        """
        reason = nonsmooth_reason(basis)
        if reason is not None:
            return ErrorTerm.unbounded(
                note=f"{reason} (eq: local response remainder)",
                manuscript_term="basis remainder",
            )
        index, n_ch = basis.index, basis.channels.n_ch
        n_lk, N, L = index.n_lk, index.truncation.N, index.truncation.depth_degree
        amplitude = _amplitude(amplitude)
        screen = L is not None and is_screen_route(basis)
        names = ["rho_ang", "H", "absolute_moments"]
        if L is not None:
            names += (
                ["intrinsic_residual"]
                if screen
                else ["depth_tail", "intrinsic_residual"]
            )
        missing = self._missing(names)
        if L is not None and not screen and depth_coefficients is None:
            missing.append("depth_coefficients")
        if missing:
            return ErrorTerm.unbounded(
                note=f"basis remainder inputs missing: {', '.join(missing)} "
                "(eq: local response remainder)"
            )
        rho, H, absolute = self.rho_ang, self.H, self.absolute_moments
        if rho.shape != (n_ch, 3):
            raise ValueError(f"rho_ang must have shape ({n_ch}, 3), got {rho.shape}")
        if H.shape != (n_ch, 3, n_lk):
            raise ValueError(f"H must have shape ({n_ch}, 3, {n_lk}), got {H.shape}")
        if absolute.shape != (2, n_lk):
            raise ValueError(
                f"absolute_moments must have shape (2, {n_lk}), got {absolute.shape}"
            )
        taylor = 1.0 / math.factorial(N + 1)
        I = rho[:, 0] + taylor * (H[:, 0] @ absolute[0])
        V = rho[:, 2] + taylor * (H[:, 2] @ absolute[0])
        note = "eq: local response remainder: <rho_ang> + sum_lk H_lk <|P_l P_k| ||z||^{N+1}>/(N+1)!"
        kind = self.kind
        if L is None:
            P = rho[:, 1] + taylor * (H[:, 1] @ absolute[1])
            if self.depth_tail is not None or self.intrinsic_residual is not None:
                note += (
                    "; depth_tail/intrinsic_residual unused in the total-degree layout"
                )
        elif screen:
            P = self._residual(n_ch)
            note += (
                "; P = intrinsic_residual: the screen route does not Taylor-expand "
                "the depth; the emitter-depth/screen mismatch is the "
                "independent_screen assumption term"
            )
        else:
            P, note, kind = self._appendix_c(basis, depth_coefficients, note)
        value = amplitude * _stokes(n_ch, I=I, P=P, V=V)
        return ErrorTerm(
            value=value, kind=kind, note=note, manuscript_term="basis remainder"
        )

    def _residual(self, n_ch):
        residual = self.intrinsic_residual
        if residual.shape != (n_ch,):
            raise ValueError(
                f"intrinsic_residual must have shape ({n_ch},), got {residual.shape}"
            )
        return residual

    def _appendix_c(self, basis, depth_coefficients, note):
        n_ch = basis.channels.n_ch
        L = basis.index.truncation.depth_degree
        tail = self.depth_tail
        shape = (n_ch, tail.shape[0])
        coefficients = jnp.asarray(depth_coefficients)
        if coefficients.shape != shape:
            raise ValueError(
                f"depth_coefficients must have shape {shape}, got {coefficients.shape}"
            )
        residual = self._residual(n_ch)
        supplied = self.absolute_depth_coefficients
        if supplied is None:
            weights, kind = jnp.abs(coefficients), "estimate"
            source = (
                "|c_a| the modulus of the phased b = 0 channel coefficient (below "
                "the absolute line sum; supply absolute_depth_coefficients for a "
                "bound): estimate"
            )
        else:
            if supplied.shape != shape:
                raise ValueError(
                    f"absolute_depth_coefficients must have shape {shape}, "
                    f"got {supplied.shape}"
                )
            weights, kind = supplied, self.kind
            source = "the supplied absolute line sums absolute_depth_coefficients"
        tau = _tau_max(basis.channels)
        P = residual + tau ** (L + 1) / math.factorial(L + 1) * (weights @ tail)
        note += (
            f"; P from eq: joint screen remainder with tau_max^{L + 1}/({L + 1})! "
            f"sum_a A_a depth_tail_a (tau_max at the channel's lower support edge), "
            f"A = {source}"
        )
        return P, note, kind


__all__ = [
    "RemainderInputs",
    "screen_factorisation_bound",
    "azimuth_factorisation_bound",
    "depth_error_bound",
    "C_SI_M",
]
