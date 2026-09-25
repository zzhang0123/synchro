"""Basis-route checks of ``RemainderInputs.basis_remainder`` (private helper of ``bounds``).

``nonsmooth_reason`` returns why the Taylor remainder of a basis is
unconstrained (a line kernel whose channels have smoothness below
``N + 1``; ``_basis_checks.check_kernel`` records the same condition as a
provenance note). ``is_screen_route`` tells whether the basis phase route is
a screen, whose finite response does not Taylor-expand the depth.
``lower_set_margin`` is the margin ``S(Lambda)`` of the lower-set Taylor
remainder used for capped truncations (re-exported by ``syncmoments.model.index``);
``margin_columns`` evaluates that form from the probed or supplied inputs
and ``appendix_c`` the ``eq: joint screen remainder`` column of ``P``.
Units follow ``syncmoments.model.bounds``; nothing here verifies the supplied
envelopes.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from ._bound_helpers import _tau_max
from .phase import TaylorPhase

NONSMOOTH_PREFIX = "basis_remainder unbounded"
LINE_KERNELS = ("harmonic",)


def _record(pairs) -> dict:
    try:
        return dict(pairs)
    except (TypeError, ValueError):
        return {}


def nonsmooth_reason(basis) -> str | None:
    """The unbounded-remainder reason of ``basis``, or ``None`` when none applies."""
    provenance = getattr(basis, "provenance", None)
    notes = getattr(provenance, "notes", ()) or ()
    for note in notes:
        if note.startswith(NONSMOOTH_PREFIX):
            return note
    kernel = _record(getattr(provenance, "kernel", ()) or ())
    if kernel.get("name") not in LINE_KERNELS:
        return None
    smoothness = basis.channels.smoothness
    N = basis.index.truncation.N
    if smoothness < N + 1:
        return (
            f"{NONSMOOTH_PREFIX}: channel smoothness {smoothness} < N + 1 = {N + 1}: "
            "derivatives of a Dirac-line kernel through the channel edges are not "
            "defined and the basis remainder is unbounded"
        )
    return None


def is_screen_route(basis) -> bool:
    """``True`` when the basis phase route is a screen (not ``TaylorPhase``).

    Reads ``basis.phase`` when set, else the ``route`` entry of
    ``provenance.phase_route``; an unrecorded route counts as Taylor, which
    keeps the depth-Taylor term.
    """
    phase = getattr(basis, "phase", None)
    if phase is not None:
        return not isinstance(phase, TaylorPhase)
    provenance = getattr(basis, "provenance", None)
    route = _record(getattr(provenance, "phase_route", ()) or ()).get("route")
    return route is not None and route != TaylorPhase.name


def lower_set_margin(lower_set) -> tuple[tuple[int, ...], ...]:
    """Telescoped margin ``S(Lambda)`` of a finite lower set of ``d``-tuples.

    Result (``[extension]`` of ``eq: local response remainder``): for ``f``
    of class ``C^{|beta|}`` on the box spanned by ``0`` and ``z``,

        ``|f(z) - T_Lambda f(z)| <= sum_{beta in S} |z^beta| / beta!
        sup_{t in [0,1]} |d^beta f(p_k(t))|``,

    ``T_Lambda f = sum_{beta in Lambda} d^beta f(0) z^beta / beta!``,
    ``k`` the first nonzero coordinate of ``beta`` and ``p_k(t) = (z_1, ...,
    z_{k-1}, t z_k, 0, ..., 0)`` (inside the box, so the box sup also bounds).
    Derivation, by induction on ``d``: write ``Lambda = union_j Lambda_j x {j}``
    over the last coordinate, ``j <= J`` (sections ``Lambda_j`` are lower
    sets). The one-dimensional Taylor step in ``z_d`` at fixed ``z'`` gives
    ``f = sum_{j<=J} z_d^j/j! h_j(z') + R_d`` with ``h_j = d_d^j f(., 0)``
    and ``|R_d| <= |z_d|^{J+1}/(J+1)! sup_t |d_d^{J+1} f(z', t z_d)|``
    (integral form, complex ``f``); since ``T_Lambda f = sum_j z_d^j/j!
    T_{Lambda_j} h_j``, ``f - T_Lambda f = R_d + sum_j z_d^j/j! (h_j -
    T_{Lambda_j} h_j)`` and the induction hypothesis on each ``h_j``
    telescopes the rest. Hence ``S = {(0,..,0,J+1)} union_j S(Lambda_j) x {j}``
    and ``S({0..n}) = {n+1}`` in one dimension. For total degree ``|beta| <= N``
    this is the shell ``|beta| = N + 1``, the manuscript's multi-index form.
    ``S`` contains every minimal element of the complement and can contain
    more (for ``{0,1}^2``: ``(2,1)``); the minimal elements alone do not bound
    (``tests/model/test_lower_set_truncation.py`` shows a counterexample).
    Every ``beta`` in ``S`` has a predecessor ``beta - e_i`` in ``Lambda``.

    ``lower_set``: nonempty iterable of equal-length int tuples, closed
    under ``beta -> beta - e_i`` (``ValueError`` otherwise). Returns the
    sorted tuple ``S``. Dimensionless combinatorics; exact integer
    bookkeeping validated by tests. The derivative sup is an input: nothing
    here verifies it, and a probed sup is an estimate.
    """
    lam = {tuple(int(v) for v in beta) for beta in lower_set}
    if not lam or len({len(beta) for beta in lam}) != 1:
        raise ValueError("lower_set must be a nonempty set of equal-length tuples")
    for beta in lam:
        for i, value in enumerate(beta):
            if value < 0 or (
                value and beta[:i] + (value - 1,) + beta[i + 1 :] not in lam
            ):
                raise ValueError(f"{sorted(lam)} is not a lower (downward-closed) set")
    return tuple(sorted(_margin(lam)))


def _margin(lam):
    if len(next(iter(lam))) == 1:
        return {(max(beta[0] for beta in lam) + 1,)}
    J = max(beta[-1] for beta in lam)
    out = {(0,) * (len(next(iter(lam))) - 1) + (J + 1,)}
    for j in range(J + 1):
        section = {beta[:-1] for beta in lam if beta[-1] == j}
        out |= {alpha + (j,) for alpha in _margin(section)}
    return out


def margin_columns(rho, margin_H, margin_moments, basis):
    """``(n_ch, 3)`` in ``I, P, V`` order: ``rho + sum_lk sum_beta H A / beta!``.

    ``rho`` ``(n_ch, 3)``, ``margin_H`` ``(n_ch, 3, n_lk, n_m)``,
    ``margin_moments`` ``(n_lk, n_m)`` over ``truncation.margin_rows()``;
    ``ValueError`` on a shape mismatch.
    """
    rows = basis.index.truncation.margin_rows()
    n_ch, n_lk, n_m = basis.channels.n_ch, basis.index.n_lk, len(rows)
    for name, value, shape in (
        ("rho_ang", rho, (n_ch, 3)),
        ("margin_H", margin_H, (n_ch, 3, n_lk, n_m)),
        ("margin_moments", margin_moments, (n_lk, n_m)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    weights = jnp.asarray([1.0 / math.prod(map(math.factorial, b)) for b in rows])
    return rho + jnp.einsum("cxlm,lm,m->cx", margin_H, margin_moments, weights)


MARGIN_NOTE = (
    "lower-set margin form of eq: local response remainder (capped truncation): "
    "<rho_ang> + sum_lk sum_{beta in S} H_lk,beta <|P_l P_k| |z^beta|> / beta!, "
    "S = truncation.margin_rows()"
)


def appendix_c(inputs, basis, depth_coefficients, residual, note):
    """``(P, note, kind)`` of ``eq: joint screen remainder`` on the Taylor route.

    ``P = residual + tau_max^{L+1}/(L+1)! sum_a A_a depth_tail_a`` with ``A``
    the supplied absolute line sums (kind of ``inputs``) or ``|c_a|`` of the
    phased ``b = 0`` coefficients ``depth_coefficients`` ``(n_ch, n_a)``
    (kind ``estimate``).
    """
    n_ch = basis.channels.n_ch
    L = basis.index.truncation.max_b()
    tail = inputs.depth_tail
    shape = (n_ch, tail.shape[0])
    coefficients = jnp.asarray(depth_coefficients)
    if coefficients.shape != shape:
        raise ValueError(
            f"depth_coefficients must have shape {shape}, got {coefficients.shape}"
        )
    supplied = inputs.absolute_depth_coefficients
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
        weights, kind = supplied, inputs.kind
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
    "margin_columns",
    "MARGIN_NOTE",
    "appendix_c",
    "nonsmooth_reason",
    "is_screen_route",
    "lower_set_margin",
    "NONSMOOTH_PREFIX",
]
