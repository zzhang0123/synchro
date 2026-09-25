"""Model-layer entry points of the AD-transform matrix (see ``_ad_matrix_cases``).

``PopulationSamples`` (normalised weights, ``product``), ``JointMoments.from_samples``
and ``EmpiricalScreen`` use ``rm``'s weight normalisation; ``bessel_band`` and
``neighbours_recurrence`` carry the Bessel order-recurrence ``custom_jvp``, and
``model.harmonic.harmonic_lines`` (the kernel's line powers) is its consumer:
reference ``bessel="autodiff"`` (plain autodiff of the same contour quadrature),
outputs divided by fixed per-output scales (``HL_SCALES``, not by a
data-dependent maximum, whose derivative at ties differs between ``jit`` and
eager evaluation of the reference).
"""

from __future__ import annotations

import jax.numpy as jnp

from _ad_matrix_cases import RMS, WEIGHTS, A, _complex_parts, _groups, _p
from syncmoments.bessel import bessel_jn, bessel_jn_neighbours
from syncmoments.model._bessel_recurrence import bessel_band, neighbours_recurrence
from syncmoments.model.harmonic import harmonic_lines
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference
from syncmoments.model.phase import EmpiricalScreen

INDEX = MomentIndex.build(Truncation(1, 1, 1))
N_NODES = 64

# -- model layer -----------------------------------------------------------------

POP = dict(
    gamma=[2.0, 3.5, 2.8, 4.1, 3.0],
    B=[1.0, 1.4, 0.8, 1.1, 1.3],
    mu=[0.2, -0.5, 0.7, 0.1, -0.3],
    eta=[0.4, 0.1, -0.6, 0.3, 0.0],
    phi=[0.3, 1.2, -0.4, 2.0, 0.8],
    depth=[0.5, -1.0, 0.2, 1.5, -0.3],
    weights=WEIGHTS[:5],
)
POP.update(gamma0=3.0, B0=1.1, depth_ref=0.1, s_gamma=0.8, s_B=0.4, s_depth=1.2)
_SAMPLE_KEYS = ("gamma", "B", "mu", "eta", "phi", "depth")
_REF_KEYS = ("gamma0", "B0", "depth_ref", "s_gamma", "s_B", "s_depth")


def _population(p):
    return PopulationSamples(*(p[k] for k in _SAMPLE_KEYS), weights=p["weights"])


def _joint_moments(p):
    ref = Reference(
        p["gamma0"], p["B0"], p["depth_ref"], (p["s_gamma"], p["s_B"], p["s_depth"])
    )  # noqa: E501
    m = JointMoments.from_samples(_population(p), INDEX, ref)
    return jnp.concatenate([m.m0, m.m0_ext, jnp.real(m.m2), jnp.imag(m.m2)])


def _legendre(x, n):
    return (jnp.ones_like(x), x, (3 * x**2 - 1) / 2)[n]


def _joint_moments_ref(p):
    q = {k: A(p[k]) for k in (*_SAMPLE_KEYS, *_REF_KEYS)}
    z = (
        (q["gamma"] - q["gamma0"]) / q["s_gamma"],
        (q["B"] - q["B0"]) / q["s_B"],
        (q["depth"] - q["depth_ref"]) / q["s_depth"],
    )
    w = _p(p["weights"])

    def row(l, k, r, s, b):
        return (
            _legendre(q["mu"], l)
            * _legendre(q["eta"], k)
            * z[0] ** r
            * z[1] ** s
            * z[2] ** b
        )  # noqa: E501

    real = [jnp.sum(w * row(*r)) for r in INDEX.exponents((*INDEX.h0, *INDEX.h0_ext))]
    e2 = jnp.exp(2j * q["phi"])
    cplx = jnp.stack([jnp.sum(w * e2 * row(*r)) for r in INDEX.exponents(INDEX.h2)])
    return jnp.concatenate([jnp.stack(real), jnp.real(cplx), jnp.imag(cplx)])


def _product_weights(p):
    marginals = {k: A(POP[k][:2]) for k in _SAMPLE_KEYS}
    marginals["gamma"] = (A(POP["gamma"][:3]), p["wg"])
    marginals["depth"] = (A(POP["depth"][:2]), p["wd"])
    return PopulationSamples.product(**marginals).normalised_weights()


def _product_weights_ref(p):
    return jnp.kron(jnp.kron(_p(p["wg"]), jnp.full(16, 1 / 16)), _p(p["wd"]))


SCREEN = dict(depths=RMS, weights=WEIGHTS, tau=[0.05, 0.3, 0.9])


def _screen_model(p):
    return _complex_parts(EmpiricalScreen(p["depths"], p["weights"])(p["tau"]))


def _screen_model_ref(p):
    phase = jnp.exp(1j * A(p["depths"])[:, None] * A(p["tau"])[None, :])
    return _complex_parts(jnp.sum(_p(p["weights"])[:, None] * phase, axis=0))


BESSEL = dict(m=[3.0, 5.0, 8.0, 12.0], x=[1.2, 4.0, 9.5, 11.0])


def _band_ref(p):
    return jnp.ravel(
        jnp.stack(
            [
                bessel_jn(A(p["m"]) + k, A(p["x"]), n_nodes=N_NODES)
                for k in range(-2, 3)
            ],
            axis=-1,
        )
    )


HL_M = [1.0, 2.0, 3.0, 5.0]  # harmonics; Bessel arguments x from 0.8 to 3.5
HL = dict(gamma=[2.0, 3.0, 1.5, 2.5], B=[1.0, 0.5, 2.0, 1.2])
HL.update(mu=[0.3, -0.2, 0.6, 0.1], eta=[0.5, 0.1, -0.4, 0.7])
HL_SCALES = (3e-17, 1.3e-17, 3e-17, 1e7)  # about max|I|, |Q|, |V|, |nu| at HL


def _lines(rule=None):
    """``harmonic_lines`` at ``HL_M`` with Bessel rule ``rule``, scaled, flat."""
    extra = {} if rule is None else {"bessel": rule}  # v0.2.0 has no ``bessel``

    def lines(p):
        args = (p[k] for k in ("gamma", "B", "mu", "eta"))
        out = harmonic_lines(A(HL_M), *args, n_nodes=N_NODES, **extra)
        return jnp.concatenate([a / c for a, c in zip(out, HL_SCALES)])

    return lines


def _fixed_m(fun):
    """``fun(m, x)`` at the fixed orders ``BESSEL["m"]`` as a function of ``{"x"}``."""
    return lambda p: fun(A(BESSEL["m"]), p["x"])


def model_groups():
    pop_keys = (
        ("weights",),
        ("gamma", "B", "depth"),
        ("mu", "eta", "phi"),
        _REF_KEYS,
        "all",
    )  # noqa: E501
    sample_only = {k: POP[k] for k in (*_SAMPLE_KEYS, "weights")}
    return [
        *_groups(
            "PopulationSamples.normalised_weights",
            lambda p: _population(p).normalised_weights(),
            lambda p: _p(p["weights"]), sample_only, (("weights",),), ["weights"],
        ),
        *_groups(
            "PopulationSamples.product", _product_weights, _product_weights_ref,
            dict(wg=[1.0, 2.0, 0.5], wd=[0.7, 1.3]), (("wg", "wd"),), ["wg", "wd"],
        ),
        *_groups(
            "JointMoments.from_samples", _joint_moments, _joint_moments_ref, POP,
            pop_keys, ["weights", "gamma", "B", "gamma0", "B0", *_REF_KEYS[3:]],
        ),
        *_groups(
            "EmpiricalScreen", _screen_model, _screen_model_ref, SCREEN,
            (("depths",), ("weights",), ("tau",), "all"), ["weights"],
        ),
        *_groups(
            "neighbours_recurrence",
            _fixed_m(lambda m, x: jnp.ravel(jnp.stack(neighbours_recurrence(m, x, N_NODES)))),  # noqa: E501
            _fixed_m(lambda m, x: jnp.ravel(jnp.stack(bessel_jn_neighbours(m, x, n_nodes=N_NODES)))),  # noqa: E501
            {"x": BESSEL["x"]}, (("x",),),
        ),
        *_groups(
            "bessel_band",
            _fixed_m(lambda m, x: jnp.ravel(bessel_band(m, x, 2, N_NODES))),
            lambda p: _band_ref({"m": BESSEL["m"], **p}),
            {"x": BESSEL["x"]}, (("x",),),
        ),
        *_groups(
            "harmonic_lines", _lines(), _lines("autodiff"), HL,
            (("gamma", "B"), ("mu", "eta"), "all"), ["gamma", "B"],
        ),
    ]  # fmt: skip
