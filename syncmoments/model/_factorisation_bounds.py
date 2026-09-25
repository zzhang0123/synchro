"""Factorisation bounds of the channel budget (private helper of ``bounds``).

``screen_factorisation_bound`` (``detail-eq: screen factorisation error``)
and ``azimuth_factorisation_bound`` (``detail-eq: angular factorisation
error``); re-exported by ``syncmoments.model.bounds``, whose module docstring
states the units and what is not certified.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from ._bound_helpers import _amplitude, _as_float, _nonnegative, _stokes
from .errors import ErrorTerm


def _variance_bound(sigma, modulus, sigma_name, modulus_name, *, per_line=False):
    """``sum_m sigma_m sqrt(1 - |modulus_m|^2)`` over the last axis of ``(n_ch, n_line)``.

    A 1-D ``sigma`` ``(n_ch,)`` gives ``sigma sqrt(1 - |modulus|^2)``; a 2-D
    ``sigma`` is accepted only with ``per_line``. ``modulus`` is a scalar or
    matches ``sigma``.
    """
    sigma = _as_float(sigma, sigma_name)
    allowed = (1, 2) if per_line else (1,)
    if sigma.ndim not in allowed:
        shapes = "(n_ch,) or (n_ch, n_line)" if per_line else "(n_ch,)"
        raise ValueError(f"{sigma_name} must have shape {shapes}")
    modulus = _as_float(modulus, modulus_name)
    if modulus.ndim == 0:
        modulus = jnp.broadcast_to(modulus, sigma.shape)
    if modulus.shape != sigma.shape:
        raise ValueError(f"{modulus_name} must be a scalar or match {sigma_name}")
    sigma = _nonnegative(sigma, sigma_name)
    modulus = _nonnegative(modulus, modulus_name)
    modulus = eqx.error_if(
        modulus, jnp.any(modulus > 1), f"{modulus_name} must be <= 1"
    )
    value = sigma * jnp.sqrt(jnp.clip(1.0 - modulus**2, 0.0, None))
    return value if value.ndim == 1 else jnp.sum(value, axis=1)


def screen_factorisation_bound(
    sigma_P_in, Phi_R_abs_min, *, amplitude=1.0
) -> ErrorTerm:
    """Bound on ``|C_scr|`` per channel (``detail-eq: screen factorisation error``).

    The channel error is ``sum_m Cov_ray(R_j(nu_m) P_m, exp(i tau_m R))`` over
    the lines ``m`` inside channel ``j``, and the source inequality holds per
    line. Two input forms:

    * per line, ``sigma_P_in`` ``(n_ch, n_line)`` the ray standard deviation
      of the channel-weighted incident line polarisation
      ``|R_j(nu_m)| sigma_{P_m}`` and ``Phi_R_abs_min`` ``(n_ch, n_line)`` (or
      a scalar) ``|<exp(i tau_m R)>|``; returns ``sum_m sigma_m sqrt(1 -
      |Phi_m|^2)`` (pad absent lines with ``sigma = 0``);
    * per channel, ``sigma_P_in`` ``(n_ch,)`` the SUM over the lines of the
      per-line standard deviations, ``sum_m |R_j(nu_m)| sigma_{P_m}`` (for a
      continuum, ``int |R_j| sigma_{P(nu)} dnu``), and ``Phi_R_abs_min``
      ``(n_ch,)`` the smallest ``|Phi_R|`` over the lines; returns
      ``sigma sqrt(1 - |Phi|^2)``.

    The standard deviation of the channel-summed incident polarisation is not
    a valid ``sigma_P_in``: covariances of different lines can cancel in it
    while the channel error does not. The bound applies to ``|P|`` only
    (zeros in ``I, V``); it can be loose and is an assumption term of the
    budget. Units: those of ``sigma_P_in`` times ``amplitude`` (channel
    Stokes). The inputs are the caller's; nothing here certifies them.
    """
    value = _variance_bound(
        sigma_P_in, Phi_R_abs_min, "sigma_P_in", "Phi_R_abs_min", per_line=True
    )
    value = _amplitude(amplitude) * _stokes(value.shape[0], P=value)
    return ErrorTerm(
        value=value,
        kind="bound",
        note="detail-eq: screen factorisation error: sum over lines of "
        "sigma_{P_m} sqrt(1-|Phi_R(tau_m)|^2), independent-screen "
        "factorisation of the polarised channel",
        manuscript_term="E_phys",
    )


def azimuth_factorisation_bound(sigma_F, m_phi_abs, *, amplitude=1.0) -> ErrorTerm:
    """``sigma_F sqrt(1 - |m_phi|^2)`` per channel (``detail-eq: angular factorisation error``).

    ``sigma_F`` ``(n_ch,)`` is the standard deviation of the natural-basis
    polarised channel kernel under the population measure; ``m_phi_abs`` is
    ``|<exp(2 i phi)>|`` (scalar or ``(n_ch,)``). Zeros in ``I, V``.
    Units: those of ``sigma_F`` times ``amplitude``. Assumes the
    constant-conditional-circular-moment model of that equation; the
    inputs are the caller's and are not certified here.
    """
    value = _variance_bound(sigma_F, m_phi_abs, "sigma_F", "m_phi_abs")
    value = _amplitude(amplitude) * _stokes(value.shape[0], P=value)
    return ErrorTerm(
        value=value,
        kind="bound",
        note="detail-eq: angular factorisation error: sigma_F sqrt(1-|m_phi|^2), "
        "azimuth-separable approximation of the polarised channel",
        manuscript_term="E_phys",
    )


__all__ = ["screen_factorisation_bound", "azimuth_factorisation_bound"]
