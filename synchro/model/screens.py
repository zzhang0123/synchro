"""Non-Gaussian independent Faraday screens with exact characteristic functions.

LABEL ``eq: illustrated screens`` (main text, independent-screen limit), used
through ``eq: independent screen response``: for a depth distribution that is
statistically independent of the emitting population, the observed complex
polarisation is the emitted one times ``C_scr(tau) = <exp(i tau depth)>``,
applied per line at ``tau_m = 2 (c/nu_m)^2`` inside the channel sum.

Two screen shapes are provided, parametrised by their depth mean ``mean``
[rad/m^2] and standard deviation ``sigma`` [rad/m^2] so that they can be
compared with ``GaussianScreen`` at identical first two moments:

* ``LaplaceScreen``: symmetric, excess kurtosis 3,
  ``C = exp(i tau mean) / (1 + (tau sigma)^2 / 2)``.
* ``GammaScreen``: shifted Gamma of shape ``k``, skewness ``sign * 2/sqrt(k)``,
  excess kurtosis ``6/k``;
  ``C = exp(i tau mean) exp(-i s t sqrt(k)) (1 - i s t / sqrt(k))^(-k)``,
  ``t = tau sigma``, ``s = sign``. At ``k = 4``, ``s = +1`` this is the
  manuscript's ``exp(-2 i t) (1 - i t/2)^(-4)``.

Both are routes of ``build_basis``/``direct_channel_average`` with
``depth_degree = 0``, exactly like ``GaussianScreen``. Each records the
assumptions ``independent_screen`` and ``<shape>_screen`` with unbounded
discrepancy: independence of screen and emitters and the screen shape are
declared model inputs. Nothing here certifies either for a real sightline;
the difference between a real population and the declared screen belongs in
``E_phys`` (main text, combined spectral error). Zero covariance between depth
and emission does not justify the factorisation.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .errors import AssumptionRecord
from .phase import SCREEN_GROUPS, _finite, _independent_record, _real

ILLUSTRATED_LABEL = "eq: illustrated screens"

__all__ = ["LaplaceScreen", "GammaScreen", "ILLUSTRATED_LABEL"]


def _moments(mean, sigma):
    mean = _real(mean, "mean", scalar=True)
    sigma = _real(sigma, "sigma", scalar=True)
    return mean, eqx.error_if(sigma, sigma < 0, "sigma must be nonnegative")


def _shape_record(name, closure_kind, hyper) -> AssumptionRecord:
    return AssumptionRecord(
        name, ILLUSTRATED_LABEL, SCREEN_GROUPS, closure_kind, hyper, 0, "unbounded"
    )


class LaplaceScreen(eqx.Module):
    """Laplace depth density with mean ``mean`` and standard deviation ``sigma``.

    ``w_0(tau) = exp(i tau mean) / (1 + (tau sigma)^2 / 2)``, shape ``(1, *tau.shape)``,
    ``tau`` in m^2. The density is ``exp(-sqrt(2)|x|)/sqrt(2)`` in
    ``x = (depth - mean)/sigma``. ``depth_ref``/``s_depth`` are ignored; forces
    ``max_b() == 0``. Records ``independent_screen`` and ``laplace_screen``
    (discrepancy unbounded). The shape is a declared input, not inferred.
    Not certified: independence of screen and emitters, and the Laplace shape
    for any real sightline (see the module docstring).
    """

    mean: jax.Array
    sigma: jax.Array
    n_weights: ClassVar[int] = 1
    LABEL: ClassVar[str] = ILLUSTRATED_LABEL
    name: ClassVar[str] = "laplace_screen"

    def __call__(self, tau, *, depth_ref=None, s_depth=None) -> jax.Array:
        tau = _real(tau, "tau")
        mean, sigma = _moments(self.mean, self.sigma)
        phase = jnp.exp(1j * _finite(tau * mean, "screen phase"))
        weight = phase / (1.0 + 0.5 * (tau * sigma) ** 2)
        return _finite(weight, "Laplace screen weight")[None]

    def _hyper(self) -> tuple[float, float]:
        return (float(np.asarray(self.mean)), float(np.asarray(self.sigma)))

    def describe(self) -> tuple:
        mean, sigma = self._hyper()
        return (("route", self.name), ("mean", mean), ("sigma", sigma))

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        """Independent screen and Laplace depth shape, both unbounded."""
        hyper = self._hyper()
        return (
            _independent_record("laplace_depth", hyper),
            _shape_record("laplace_screen", "laplace_depth", hyper),
        )

    def max_b(self) -> int:
        return 0


class GammaScreen(eqx.Module):
    """Shifted Gamma depth density: ``depth = mean + sign sigma (Y - k)/sqrt(k)``.

    ``Y ~ Gamma(k, 1)``; ``shape = k > 0`` and ``sign = +1`` or ``-1`` are
    static. Mean ``mean``, standard deviation ``sigma`` [rad/m^2]; skewness
    ``sign 2/sqrt(k)``, excess kurtosis ``6/k``.
    ``w_0 = exp(i tau mean) exp(-i s t sqrt(k)) (1 - i s t/sqrt(k))^(-k)``,
    ``t = tau sigma``, principal logarithm (``Re(1 - i s t/sqrt(k)) = 1 > 0``,
    so the branch is continuous in ``tau``). Shape ``(1, *tau.shape)``.
    Records ``independent_screen`` and ``gamma_screen`` (discrepancy
    unbounded). ``depth_ref``/``s_depth`` are ignored; forces ``max_b() == 0``.
    Not certified: independence of screen and emitters, and the shifted-Gamma
    shape for any real sightline (see the module docstring).
    """

    mean: jax.Array
    sigma: jax.Array
    shape: float = eqx.field(static=True, default=4.0)
    sign: int = eqx.field(static=True, default=1)
    n_weights: ClassVar[int] = 1
    LABEL: ClassVar[str] = ILLUSTRATED_LABEL
    name: ClassVar[str] = "gamma_screen"

    def __check_init__(self):
        if not (np.isfinite(self.shape) and self.shape > 0):
            raise ValueError("shape must be a finite positive number")
        if self.sign not in (1, -1):
            raise ValueError("sign must be +1 or -1")

    def __call__(self, tau, *, depth_ref=None, s_depth=None) -> jax.Array:
        tau = _real(tau, "tau")
        mean, sigma = _moments(self.mean, self.sigma)
        root_k = np.sqrt(self.shape)
        t = self.sign * tau * sigma
        exponent = (
            1j * _finite(tau * mean, "screen phase")
            - 1j * t * root_k
            - self.shape * jnp.log(1.0 - 1j * t / root_k)
        )
        return _finite(jnp.exp(exponent), "Gamma screen weight")[None]

    def _hyper(self) -> tuple[float, ...]:
        return (
            float(np.asarray(self.mean)),
            float(np.asarray(self.sigma)),
            float(self.shape),
            float(self.sign),
        )

    def describe(self) -> tuple:
        mean, sigma, shape, sign = self._hyper()
        return (
            ("route", self.name),
            ("mean", mean),
            ("sigma", sigma),
            ("shape", shape),
            ("sign", sign),
        )

    def forced_assumptions(self) -> tuple[AssumptionRecord, ...]:
        """Independent screen and shifted-Gamma depth shape, both unbounded."""
        hyper = self._hyper()
        return (
            _independent_record("gamma_depth", hyper),
            _shape_record("gamma_screen", "gamma_depth", hyper),
        )

    def max_b(self) -> int:
        return 0
