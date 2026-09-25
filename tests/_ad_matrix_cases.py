"""Entry points of the AD-transform matrix and their independent references.

Every public entry point whose value passes through a custom derivative rule
(``custom_jvp``), custom batching (``custom_vmap``) or ``equinox.filter_jit``:

* ``transfer`` (``_slab``, ``_linear``, ``_finite_primal``; ``_expm.any_member``
  and ``batched_only``; ``filter_jit`` cores): ``transfer_slab``,
  ``transfer_los`` and through them ``los_moments.moment_driven_slab(_cgs)``.
  Reference: ``Phi S + G eps`` from a plain Taylor exponential of the 8x8
  ``[[-K ds, I], [0, 0]]`` (8 squarings, 18 terms; no branch on values).
* ``rm`` weight normalisation (``_relative_weights_core``, ``_divide_by_sum``,
  ``_sum_tangent``; ``filter_jit``): ``gaussian_rm_cumulants``,
  ``screen_polarisation``, ``expansion.mixed_moments``,
  ``faraday.emission_polarisation``, ``faraday.joint_faraday_moments``,
  ``model.PopulationSamples`` (and ``.product``), ``JointMoments.from_samples``
  and ``model.phase.EmpiricalScreen`` (the last three in
  ``tests/_ad_matrix_model_cases.py``). Reference: ``w / sum(w)`` and the sums
  written out in ``jnp``.
* ``model._bessel_recurrence`` (``bessel_band``, ``neighbours_recurrence``,
  and their consumer ``model.harmonic.harmonic_lines``;
  ``tests/_ad_matrix_model_cases.py``). Reference: autodiff of the contour
  quadrature of ``syncmoments.bessel`` (``harmonic_lines(..., bessel="autodiff")``).

Values are in the normal floating-point range (the extreme-scale behaviour is
covered by ``tests/test_transfer_*`` and ``tests/test_weight_normalisation_*``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from _ad_matrix import Group
from syncmoments import expansion, faraday, rm
from syncmoments.conversion import mueller_conversion, mueller_rotation
from syncmoments.kirchhoff import (
    absorption_from_moments,
    absorption_Q_from_moments,
    cgs_coefficients_from_moments,
    emissivity_from_moments,
    emissivity_Q_from_moments,
)
from syncmoments.los_moments import moment_driven_slab, moment_driven_slab_cgs
from syncmoments.transfer import mueller_matrix, transfer_los, transfer_slab

A = jnp.asarray


def expm_taylor(M, squarings=8, terms=18):
    """``exp(M)``: degree-``terms`` Taylor polynomial of ``M 2^-squarings``, squared."""
    B = M / 2.0**squarings
    term = E = jnp.eye(M.shape[0], dtype=M.dtype)
    for k in range(1, terms + 1):
        term = term @ B / k
        E = E + term
    for _ in range(squarings):
        E = E @ E
    return E


def slab_ref(S, eps, K, ds):
    """``Phi S + G eps`` with ``[[Phi, G / ds], [0, I]] = exp([[-K ds, I], [0, 0]])``."""
    M = jnp.zeros((8, 8)).at[:4, :4].set(-K * ds).at[:4, 4:].set(jnp.eye(4))
    E = expm_taylor(M)
    return E[:4, :4] @ S + (E[:4, 4:] * ds) @ eps


def los_ref(S0, eps_s, K_s, ds):
    lengths = jnp.broadcast_to(ds, eps_s.shape[:1])
    S = S0
    for i in range(eps_s.shape[0]):
        S = slab_ref(S, eps_s[i], K_s[i], lengths[i])
    return S


def _groups(name, f, ref, x, keys, positive=()):
    """One ``Group`` per key tuple (``"all"``: every argument together)."""
    out = []
    for group in keys:
        names = tuple(x) if group == "all" else group
        sub = {k: A(x[k]) for k in names}

        def bound(fun):
            return lambda y: fun({**{k: A(v) for k, v in x.items()}, **y})

        label = f"{name}[{'all' if group == 'all' else ','.join(group)}]"
        out.append(Group(label, names, bound(f), bound(ref), sub, frozenset(positive)))
    return out


def _singles(x):
    """One group per argument of ``x``, then ``"all"``."""
    return (*((k,) for k in x), "all")


def _complex_parts(z):
    z = jnp.ravel(z)
    return jnp.concatenate([jnp.real(z), jnp.imag(z)])


# -- transfer --------------------------------------------------------------------

K1 = mueller_matrix(0.8, 0.1, -0.05, 0.02, 0.3, -0.2, 1.1)
K2 = mueller_matrix(0.3, -0.05, 0.1, 0.0, -0.4, 0.25, -2.0)
K3 = mueller_matrix(1.5, 0.2, 0.0, -0.1, 0.0, 0.1, 0.5)
SLAB = dict(S=[1.0, 0.2, -0.1, 0.05], eps=[2.0, 0.3, -0.2, 0.1], K=K1, ds=0.7)
LOS = dict(
    S0=[1.0, 0.1, 0.0, -0.05],
    eps_s=[[2.0, 0.3, -0.2, 0.1], [0.5, -0.1, 0.2, 0.0], [1.0, 0.0, 0.1, 0.05]],
    K_s=jnp.stack([K1, K2, K3]),
    ds=[0.7, 0.4, 1.1],
)


def _slab(p):
    return transfer_slab(p["S"], p["eps"], p["K"], p["ds"])


def _slab_ref(p):
    return slab_ref(A(p["S"]), A(p["eps"]), A(p["K"]), A(p["ds"]))


def _los(p):
    return transfer_los(p["S0"], p["eps_s"], p["K_s"], p["ds"])


def _los_ref(p):
    return los_ref(A(p["S0"]), A(p["eps_s"]), A(p["K_s"]), A(p["ds"]))


MDS = dict(nu=1.3, gamma0=3.0, nu_c_ref=1.0, M0=1.0, M1=0.8, M2=1.2)
MDS.update(inv_gamma=0.5, L=1.0, Q0=0.1, U0=-0.05)
MDS_KEYS = ("nu", "gamma0", "nu_c_ref", "M0", "M1", "M2", "inv_gamma")


def _mds(p, polarised=False):
    args = [p[k] for k in MDS_KEYS]
    if polarised:
        return moment_driven_slab(*args, p["L"], polarised_absorption=True)
    return moment_driven_slab(*args, p["L"], Q0=p["Q0"], U0=p["U0"])


def _mds_ref(p, polarised=False):
    args = [A(p[k]) for k in MDS_KEYS[:6]]
    j = emissivity_from_moments(*args)
    a = absorption_from_moments(*args, A(p["inv_gamma"]))
    if polarised:
        q, u = emissivity_Q_from_moments(*args), 0.0
        aq = absorption_Q_from_moments(*args, A(p["inv_gamma"]))
    else:
        q, u, aq = A(p["Q0"]) * j, A(p["U0"]) * j, 0.0
    K = mueller_matrix(a, aq, 0.0, 0.0, 0.0, 0.0, 0.0)
    return slab_ref(jnp.zeros(4), jnp.stack([j, q, u, 0.0 * j]), K, A(p["L"]))


CGS = dict(nu_hz=1.2e8, gamma0=2500.0, B_perp=5e-6, M0=1.0, M1=0.9, M2=1.1)
CGS.update(inv_gamma=0.5, L_cm=1e10, n_e=0.03, B_par=3e-6, phi=0.3)
CGS.update(S0=[1e-20, 2e-21, -1e-21, 5e-22])
CGS_KEYS = ("nu_hz", "gamma0", "B_perp", "M0", "M1", "M2", "inv_gamma", "L_cm")


def _cgs(p):
    args = [p[k] for k in CGS_KEYS]
    extra = dict(n_e=p["n_e"], B_par=p["B_par"], phi=p["phi"], S0=p["S0"])
    return moment_driven_slab_cgs(*args, **extra)


def _cgs_ref(p):
    nu, g0, Bp, M0, M1, M2, ig, L = (A(p[k]) for k in CGS_KEYS)
    jI, jQ, aI, aQ = cgs_coefficients_from_moments(nu, g0, Bp, M0, M1, M2, ig)
    co, si = jnp.cos(2 * A(p["phi"])), jnp.sin(2 * A(p["phi"]))
    rV = mueller_rotation(nu, A(p["n_e"]), A(p["B_par"]))
    rQ = mueller_conversion(nu, A(p["n_e"]), Bp)
    eps = jnp.stack([jI, jQ * co, jQ * si, 0.0 * jI])
    K = mueller_matrix(aI, aQ * co, aQ * si, 0.0, rQ * co, rQ * si, rV)
    return slab_ref(A(p["S0"]), eps, K, L)


def transfer_groups():
    los_scalar = {**LOS, "ds": 0.8}
    return [
        *_groups("transfer_slab", _slab, _slab_ref, SLAB, _singles(SLAB), ["ds"]),
        *_groups("transfer_los", _los, _los_ref, LOS, _singles(LOS), ["ds"]),
        *_groups("transfer_los_scalar_ds", _los, _los_ref, los_scalar, (("ds",),)),
        *_groups(
            "moment_driven_slab",
            _mds,
            _mds_ref,
            MDS,
            (("M0", "M1", "M2"), ("L",), ("nu", "gamma0", "nu_c_ref", "inv_gamma"))
            + (("Q0", "U0"), "all"),
            ["L", "nu", "gamma0", "nu_c_ref", "M0", "M1", "M2", "inv_gamma"],
        ),
        *_groups(
            "moment_driven_slab_polarised",
            lambda p: _mds(p, True),
            lambda p: _mds_ref(p, True),
            {k: v for k, v in MDS.items() if k not in ("Q0", "U0")},
            ("all",),
            ["L", "nu", "gamma0", "nu_c_ref", "M0", "M1", "M2", "inv_gamma"],
        ),
        *_groups(
            "moment_driven_slab_cgs",
            _cgs,
            _cgs_ref,
            CGS,
            (("M0", "M1", "M2"), ("L_cm",), ("n_e", "B_par", "phi"))
            + (("nu_hz", "gamma0", "B_perp", "inv_gamma"), ("S0",), "all"),
            ["nu_hz", "gamma0", "B_perp", "M0", "M1", "M2", "inv_gamma", "L_cm"]
            + ["n_e"],
        ),
    ]


# -- weights ---------------------------------------------------------------------

RMS = [3.0, -1.0, 0.5, 2.2, -0.7, 1.4]
WEIGHTS = [1.0, 0.5, 2.0, 0.3, 1.2, 0.8]
LAM = [0.1, 0.2, 0.35]


def _p(w):
    w = A(w)
    return w / jnp.sum(w)


def _cumulants_ref(p):
    w, x = _p(p["weights"]), A(p["rms"])
    mean = jnp.sum(w * x)
    return jnp.stack([mean, jnp.sum(w * (x - mean) ** 2)])


def _screen_ref(p):
    w, lam2 = _p(p["weights"]), A(p["lam"]) ** 2
    phase = jnp.exp(2j * A(p["rms"])[:, None] * lam2[None, :])
    return _complex_parts(jnp.sum((w * A(p["P0"]))[:, None] * phase, axis=0))


OFFSETS = np.array([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.2], [0.1, 0.6], [0.3, -0.5]])
MIXED = dict(offsets=OFFSETS, factor=[1.1, 0.7, 1.3, 0.9, 1.0])
MIXED["weights"] = WEIGHTS[:5]


def _mixed(p):
    return jnp.concatenate(
        [jnp.ravel(m) for m in expansion.mixed_moments(**p)]  # noqa: C416
    )


def _mixed_ref(p):
    wf, q = _p(p["weights"]) * A(p["factor"]), A(p["offsets"])
    parts = (jnp.sum(wf), wf @ q, (wf[:, None] * q).T @ q)
    return jnp.concatenate([jnp.ravel(m) for m in parts])


EMISSION = dict(emission=[1.0, 0.6, -0.3, 0.8, 0.2, 1.1], depths=RMS, lam=LAM)
EMISSION.update(weights=WEIGHTS, source_column=2.5)


def _emission(p):
    return _complex_parts(
        faraday.emission_polarisation(
            p["emission"], p["depths"], p["lam"], p["weights"],
            source_column=p["source_column"],
        )
    )  # fmt: skip


def _emission_ref(p):
    w, t = _p(p["weights"]), 2 * A(p["lam"]) ** 2
    phase = jnp.exp(1j * A(p["depths"])[:, None] * t[None, :])
    total = jnp.sum((w * A(p["emission"]))[:, None] * phase, axis=0)
    return _complex_parts(A(p["source_column"]) * total)


BASIS = np.array([[1.0, 0.2], [0.5, -0.3], [0.8, 0.1], [1.2, 0.4], [0.3, -0.2]])
JOINT = dict(basis_values=BASIS, depths=RMS[:5], weights=WEIGHTS[:5])
JOINT["reference_depth"] = 0.25


def _joint(p):
    moments, nxt = faraday.joint_faraday_moments(
        p["basis_values"], p["depths"], 3, p["weights"],
        reference_depth=p["reference_depth"],
    )  # fmt: skip
    return jnp.concatenate([jnp.ravel(moments), nxt])


def _joint_ref(p):
    w, psi = _p(p["weights"]), A(p["basis_values"])
    delta = A(p["depths"]) - A(p["reference_depth"])
    powers = jnp.stack([delta**b for b in range(4)], axis=1)
    moments = (w[:, None] * psi).T @ powers
    nxt = (w[:, None] * jnp.abs(psi)).T @ jnp.abs(delta) ** 4
    return jnp.concatenate([jnp.ravel(moments), nxt])


def rm_groups():
    cum = dict(rms=RMS, weights=WEIGHTS)
    screen = dict(P0=[1.0, 0.8, 0.9, 1.1, 0.7, 1.2], rms=RMS, lam=LAM, weights=WEIGHTS)
    return [
        *_groups(
            "gaussian_rm_cumulants",
            lambda p: jnp.stack(rm.gaussian_rm_cumulants(p["rms"], p["weights"])),
            _cumulants_ref, cum, _singles(cum), ["weights"],
        ),
        *_groups(
            "gaussian_rm_cumulants_unweighted",
            lambda p: jnp.stack(rm.gaussian_rm_cumulants(p["rms"])),
            lambda p: _cumulants_ref({**p, "weights": jnp.ones(6)}),
            dict(rms=RMS), (("rms",),),
        ),
        *_groups(
            "screen_polarisation",
            lambda p: _complex_parts(rm.screen_polarisation(**p)),
            _screen_ref, screen, _singles(screen), ["weights"],
        ),
        *_groups("mixed_moments", _mixed, _mixed_ref, MIXED, _singles(MIXED), ["weights"]),
        *_groups(
            "emission_polarisation", _emission, _emission_ref, EMISSION,
            _singles(EMISSION), ["weights"],
        ),
        *_groups("joint_faraday_moments", _joint, _joint_ref, JOINT, _singles(JOINT),
                 ["weights"]),
    ]  # fmt: skip


def all_groups():
    from _ad_matrix_model_cases import model_groups  # it imports this module

    return [*transfer_groups(), *rm_groups(), *model_groups()]
