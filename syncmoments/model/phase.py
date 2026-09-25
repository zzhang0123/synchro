"""Faraday-phase routes: per-line weights ``w_b(tau)`` multiplying ``Q_m R_j(nu_m)``.

LABEL ``eq: smooth channel kernel`` (the phase ``exp(2 i (c/nu_m)^2 varphi)`` is
applied line by line inside the channel sum), ``eq: channel derivative
coefficients`` (Taylor route, ``b = 0..degree``), ``eq: independent gaussian
screen`` and ``eq: independent screen response`` (screen routes),
``eq: screen exponent error`` (``CumulantScreen.exponent_remainder``).

The phase coordinate is ``tau = 2 lambda^2`` with ``lambda = C_SI_M / nu`` in
metres, so ``tau`` has units m^2 and ``tau * varphi`` is in radians for a depth
``varphi`` in rad/m^2. ``phase_coordinate(nu_hz)`` computes it; the harmonic
kernel evaluates it at every line ``nu_m`` and the continuum kernel at every
frequency node, never at a nominal channel frequency (main.tex: "the channel
phase cannot be factored at a central frequency").

Routes (all satisfy :class:`PhaseWeights`; ``__call__`` returns a complex
array ``(n_weights, *tau.shape)`` and is ``jit``/``grad`` safe):

* :class:`TaylorPhase` (default): ``w_b = exp(i tau depth_ref) (i tau s_depth)^b / b!``
  for ``b = 0..degree``, the phase expanded about the reference depth in the
  dimensionless displacement ``z_depth = (varphi - depth_ref)/s_depth``. The
  Taylor remainder is bounded by ``build_basis``/``bounds.py`` through
  ``|tau s_depth|^{N+1}/(N+1)!`` times the absolute depth moment; it is not
  bounded here.
* :class:`GaussianScreen`: ``w_0 = exp(i tau mean - tau^2 sigma^2 / 2)`` (Burn);
  exact within the independent Gaussian-screen model, which is recorded as
  two forced assumptions (``independent_screen``, ``gaussian_screen``) whose
  discrepancy is unbounded. ``depth_ref``/``s_depth`` are accepted and ignored.
* :class:`CumulantScreen`: ``w_0 = exp(g_4)`` with
  ``g_4 = i tau k1 - tau^2 k2/2 - i tau^3 k3/6 + tau^4 k4/24``. The zero-free
  interval and the continuous logarithm of the manuscript are assumed, not
  verifiable from ``kappa``; the exponent remainder ``|R_5| <= |tau|^5 g5/5!``
  needs the supplied ``g5_bound`` (``exponent_remainder``), else it is unbounded.
* :class:`EmpiricalScreen`: ``w_0 = sum_i w_i exp(i tau depth_i)``, exact for
  the supplied discrete independent screen; sampling error is external.

Not certified: independence of the screen from the emitting parameters (a
physical assumption recorded by ``forced_assumptions``), the Taylor remainder,
the cumulant zero-free condition, argument reduction of large phases
(``tau * depth`` must be finite; accuracy degrades as it grows) and any
frequency dependence of the depth (the rotation law is ``lambda^2``).
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..constants import C_SI_M
from ..rm import _screen_samples
from .errors import AssumptionRecord

SCREEN_GROUPS = (("gamma", "B", "mu", "eta", "phi"), ("depth",))
INDEPENDENT_LABEL = "eq: independent screen moments"
GAUSSIAN_LABEL = "eq: independent gaussian screen"


def _real(value, name, *, scalar=False):
    value = jnp.asarray(value)
    if jnp.iscomplexobj(value) or (scalar and value.ndim != 0):
        raise ValueError(f"{name} must be real" + (" and scalar" if scalar else ""))
    value = value.astype(jnp.result_type(value, 1.0))
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} must be finite")


def _finite(value, name):
    return eqx.error_if(value, jnp.any(~jnp.isfinite(value)), f"{name} overflowed")


def phase_coordinate(nu_hz) -> jax.Array:
    """``tau = 2 (C_SI_M / nu)^2`` in m^2 for positive finite ``nu_hz`` (any shape).

    ``nu_hz`` in Hz; value errors use ``equinox.error_if``. No assumptions
    beyond the pure-rotation phase ``2 lambda^2 depth``; nothing certified.
    """
    nu = _real(nu_hz, "frequencies")
    nu = eqx.error_if(nu, jnp.any(nu <= 0), "frequencies must be positive")
    return _finite(2.0 * (C_SI_M / nu) ** 2, "phase coordinate")


def _independent_record(closure_kind, hyper, discrepancy_kind="unbounded"):
    return AssumptionRecord(
        "independent_screen",
        INDEPENDENT_LABEL,
        SCREEN_GROUPS,
        closure_kind,
        tuple(hyper),
        0,
        discrepancy_kind,
    )


@runtime_checkable
class PhaseWeights(Protocol):
    """Per-line phase weights (``eq: smooth channel kernel``); contract in the module docstring.

    ``__call__(tau, depth_ref=, s_depth=)`` returns ``(n_weights, *tau.shape)``
    complex weights for ``tau = 2 lambda^2`` in m^2 and depths in rad/m^2.
    ``forced_assumptions`` lists what the route assumes (nothing for the
    Taylor route; an independent screen for the screen routes). Nothing in a
    route certifies that the screen model holds for the population.
    """

    n_weights: int
    LABEL: str
    name: str

    def __call__(self, tau: jax.Array, *, depth_ref, s_depth) -> jax.Array: ...

    def describe(self) -> tuple: ...

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]: ...

    def max_b(self) -> int: ...


class TaylorPhase(eqx.Module):
    """``w_b(tau) = exp(i tau depth_ref) (i tau s_depth)^b / b!``, ``b = 0..degree``.

    ``degree`` (static, >= 0) must equal the index's maximum ``b``;
    ``build_basis`` validates that. ``depth_ref`` [rad/m^2] and ``s_depth``
    (positive scale) are read from ``Reference`` by the caller. No screen
    assumption is forced. ``tau * depth_ref`` must be finite (``eqx.error_if``).
    ``tau`` is in m^2, so ``tau * depth`` is a phase in radians. Not
    certified: the truncation of the phase series at ``degree``; its
    remainder is the depth part of ``eq: local response remainder``.
    """

    degree: int = eqx.field(static=True)
    LABEL: ClassVar[str] = "eq: channel derivative coefficients"
    name: ClassVar[str] = "taylor"

    def __check_init__(self):
        if (
            not isinstance(self.degree, int)
            or isinstance(self.degree, bool)
            or self.degree < 0
        ):
            raise ValueError("degree must be a nonnegative static integer")

    @property
    def n_weights(self) -> int:
        return self.degree + 1

    def __call__(self, tau, *, depth_ref, s_depth) -> jax.Array:
        tau = _real(tau, "tau")
        reference = _real(depth_ref, "depth_ref", scalar=True)
        scale = _real(s_depth, "s_depth", scalar=True)
        scale = eqx.error_if(scale, scale <= 0, "s_depth must be positive")
        base = jnp.exp(1j * _finite(tau * reference, "reference Faraday phase"))
        step = 1j * _finite(tau * scale, "scaled phase coordinate")
        terms = [jnp.ones_like(step)]
        for b in range(1, self.degree + 1):
            terms.append(terms[-1] * step / b)
        return _finite(base * jnp.stack(terms), "Taylor phase weights")

    def describe(self) -> tuple:
        return (("route", self.name), ("degree", self.degree), ("label", self.LABEL))

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        return ()

    def max_b(self) -> int:
        return self.degree


class GaussianScreen(eqx.Module):
    """``w_0 = exp(i tau mean - tau^2 sigma^2 / 2)``; ``mean`` [rad/m^2], ``sigma >= 0``.

    Equals ``syncmoments.rm.burn_depolarisation(1, mean, sigma**2, lam)`` at
    ``lam = C_SI_M / nu``. ``depth_ref`` and ``s_depth`` are ignored: the
    screen is not expanded about a reference. Forces ``max_b() == 0``.
    ``tau`` in m^2. Assumes a Faraday screen independent of the emitters with
    a Gaussian depth distribution (records ``independent_screen`` and
    ``gaussian_screen``); neither is verified here, and the discrepancy of
    a real population from this model is an input of the budget.
    """

    mean: jax.Array
    sigma: jax.Array
    n_weights: ClassVar[int] = 1
    LABEL: ClassVar[str] = GAUSSIAN_LABEL
    name: ClassVar[str] = "gaussian_screen"

    def __call__(self, tau, *, depth_ref=None, s_depth=None) -> jax.Array:
        tau = _real(tau, "tau")
        mean = _real(self.mean, "mean", scalar=True)
        sigma = _real(self.sigma, "sigma", scalar=True)
        sigma = eqx.error_if(sigma, sigma < 0, "sigma must be nonnegative")
        exponent = 1j * _finite(tau * mean, "screen phase") - 0.5 * (tau * sigma) ** 2
        return _finite(jnp.exp(exponent), "Gaussian screen weight")[None]

    def _hyper(self) -> tuple[float, float]:
        return (float(np.asarray(self.mean)), float(np.asarray(self.sigma)))

    def describe(self) -> tuple:
        return (
            ("route", self.name),
            ("mean", self._hyper()[0]),
            ("sigma", self._hyper()[1]),
        )

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        """Independent screen and Gaussian closure, both with unbounded discrepancy."""
        hyper = self._hyper()
        return (
            _independent_record("gaussian_depth", hyper),
            AssumptionRecord(
                "gaussian_screen",
                GAUSSIAN_LABEL,
                SCREEN_GROUPS,
                "gaussian_depth",
                hyper,
                0,
                "unbounded",
            ),
        )

    def max_b(self) -> int:
        return 0


class CumulantScreen(eqx.Module):
    """``w_0 = exp(g_4(tau))`` from the first four depth cumulants ``kappa`` (4,).

    ``kappa = (k1, k2, k3, k4)`` in rad/m^2 powers; ``k2 >= 0``. ``g5_bound``
    is ``sup |g^(5)|`` on the zero-free interval (rad^5/m^10 units); with it,
    ``exponent_remainder(tau) = |tau|^5 g5_bound / 5!`` bounds ``|R_5|``,
    otherwise the exponent remainder is unbounded. The zero-free interval is
    assumed. ``depth_ref``/``s_depth`` are ignored. Forces ``max_b() == 0``.
    A positive ``k4`` makes ``|w_0|`` grow as ``exp(tau^4 k4/24)``; overflow
    raises through ``eqx.error_if``.
    ``tau`` in m^2, cumulants in powers of rad/m^2. Assumes an independent
    screen whose characteristic function has no zero on the used ``tau``
    interval; not certified: that assumption, and the truncation of the
    cumulant series beyond the supplied ``g5_bound``.
    """

    kappa: jax.Array
    g5_bound: jax.Array | None = None
    n_weights: ClassVar[int] = 1
    LABEL: ClassVar[str] = GAUSSIAN_LABEL
    name: ClassVar[str] = "cumulant_screen"

    def __check_init__(self):
        if jnp.shape(self.kappa) != (4,):
            raise ValueError("kappa must have shape (4,): (k1, k2, k3, k4)")
        if self.g5_bound is not None and jnp.ndim(self.g5_bound) != 0:
            raise ValueError("g5_bound must be a scalar")

    def __call__(self, tau, *, depth_ref=None, s_depth=None) -> jax.Array:
        tau = _real(tau, "tau")
        kappa = _real(self.kappa, "kappa")
        kappa = eqx.error_if(
            kappa, kappa[1] < 0, "kappa[1] (variance) must be nonnegative"
        )
        k1, k2, k3, k4 = kappa
        exponent = (
            1j * _finite(tau * k1, "screen phase")
            - tau**2 * k2 / 2
            - 1j * tau**3 * k3 / 6
            + tau**4 * k4 / 24
        )
        return _finite(jnp.exp(exponent), "cumulant screen weight")[None]

    def exponent_remainder(self, tau) -> jax.Array | None:
        """``epsilon_5(tau) = |tau|^5 g5_bound / 120`` (``eq: screen exponent error``), or None."""
        if self.g5_bound is None:
            return None
        g5 = _real(self.g5_bound, "g5_bound", scalar=True)
        g5 = eqx.error_if(g5, g5 < 0, "g5_bound must be nonnegative")
        return jnp.abs(_real(tau, "tau")) ** 5 * g5 / 120.0

    def _hyper(self) -> tuple[float, ...]:
        hyper = tuple(float(v) for v in np.asarray(self.kappa))
        if self.g5_bound is not None:
            hyper = hyper + (float(np.asarray(self.g5_bound)),)
        return hyper

    def describe(self) -> tuple:
        return (
            ("route", self.name),
            ("kappa", self._hyper()[:4]),
            ("g5_bound", self._hyper()[4:]),
        )

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        """Independent screen and the fourth-order cumulant closure; the
        exponent remainder is ``bound`` only when ``g5_bound`` is supplied."""
        hyper = self._hyper()
        kind = "unbounded" if self.g5_bound is None else "bound"
        return (
            _independent_record("cumulant_depth", hyper),
            AssumptionRecord(
                "cumulant_screen",
                GAUSSIAN_LABEL,
                SCREEN_GROUPS,
                "cumulant_depth",
                hyper,
                0,
                kind,
            ),
        )

    def max_b(self) -> int:
        return 0


class EmpiricalScreen(eqx.Module):
    """``w_0 = sum_i w_i exp(i tau depth_i)`` over a discrete independent screen.

    ``depths`` (n,) in rad/m^2; ``weights`` (n,) nonnegative relative masses
    (normalised as in ``syncmoments.rm.screen_polarisation``; None means uniform).
    Exact for the supplied screen; equals ``rm.screen_polarisation(1, depths,
    lam, weights)``. A scan keeps O(tau.size) working storage.
    ``depth_ref``/``s_depth`` are ignored. Forces ``max_b() == 0``.
    ``tau`` in m^2. Assumes a screen independent of the emitters (records
    ``independent_screen``); exactness holds for the supplied discrete
    screen only, and nothing certifies that the discrete screen represents
    a continuous one (sampling error is an input).
    """

    depths: jax.Array
    weights: jax.Array | None = None
    n_weights: ClassVar[int] = 1
    LABEL: ClassVar[str] = "eq: independent screen response"
    name: ClassVar[str] = "empirical_screen"

    def __check_init__(self):
        if jnp.ndim(self.depths) != 1 or jnp.size(self.depths) == 0:
            raise ValueError("depths must be a nonempty 1D array")
        if self.weights is not None and jnp.shape(self.weights) != jnp.shape(
            self.depths
        ):
            raise ValueError("weights must match the 1D depths")

    def __call__(self, tau, *, depth_ref=None, s_depth=None) -> jax.Array:
        tau = _real(tau, "tau")
        depths, w = _screen_samples(
            self.depths, self.weights, sample_name="Faraday-depth"
        )

        def accumulate(total, element):
            depth, weight = element
            phase = _finite(tau * depth, "screen phase")
            return total + weight * jnp.exp(1j * phase), None

        total, _ = jax.lax.scan(
            accumulate,
            jnp.zeros(tau.shape, dtype=jnp.result_type(tau, 1j)),
            (depths, w),
        )
        return _finite(total, "empirical screen weight")[None]

    def describe(self) -> tuple:
        return (("route", self.name), ("n_depths", int(jnp.size(self.depths))))

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        """Independent screen with a fixed discrete table (exact for that table)."""
        return (_independent_record("fixed_table", (float(jnp.size(self.depths)),)),)

    def max_b(self) -> int:
        return 0


__all__ = [
    "PhaseWeights",
    "TaylorPhase",
    "GaussianScreen",
    "CumulantScreen",
    "EmpiricalScreen",
    "phase_coordinate",
    "SCREEN_GROUPS",
]
