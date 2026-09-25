"""Exact oracles and targets for second derivatives in the weights (no tests).

Shared by ``test_weight_normalisation_hessian.py`` and
``test_weight_normalisation_second_order.py``. For ``f(w) = F(p)`` with
``p = w / W`` and ``W = sum(w)``::

    H_w[a, b] = sum_ij F''_ij J_ia J_jb + sum_i F'_i Hp_i[a, b],
    J_ia = (delta_ia - p_i) / W,  Hp_i[a, b] = (2 p_i - delta_ia - delta_ib) / W^2.

For ``w = exp(theta)`` (log-weights) the same sum holds with the softmax
derivatives ``J_ia = p_i (delta_ia - p_a)`` and
``Hp_i[a, b] = p_i (delta_ia - p_a)(delta_ib - p_b) - p_i p_a (delta_ab - p_b)``.
All of it is evaluated in ``Fraction`` from the float64 inputs actually used,
so the oracle shares no arithmetic with the package. ``scale`` is the sum of
the magnitudes of all terms of an entry; ``classify`` accepts a float64 result
within ``64 eps scale + |exact| 1e-12`` of the exact value.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.expansion import mixed_moments
from syncmoments.model import (
    JointMoments,
    MomentIndex,
    PopulationSamples,
    Reference,
    Truncation,
)
from syncmoments.rm import gaussian_rm_cumulants

EPS = Fraction(np.finfo(float).eps)
BIG = Fraction(np.finfo(float).max)
PATTERN = np.array([1.0, 2.0, 3.0, 4.0]) / 4  # largest weight 1
X = np.array([1.0, 2.0, 3.0, 4.0])
OFFSETS = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, -1.0], [0.4, 0.5]])
FACTOR = np.ones(4)
SAMPLES = {
    "gamma": [2.0, 3.0, 4.0, 5.0],
    "B": [1.0, 2.0, 1.5, 0.5],
    "mu": [0.1, -0.3, 0.5, 0.2],
    "eta": [0.2, 0.1, -0.4, 0.3],
    "phi": [0.1, 0.2, 0.3, 0.4],
    "depth": [0.0, 1.0, 2.0, 3.0],
}
INDEX, REFERENCE, ROW = MomentIndex.build(Truncation(1, 1, 1)), Reference(3.0, 1.0), 3


def fractions(values):
    return [Fraction(float(v)) for v in np.ravel(values)]


def linear(a):
    """``F(p) = sum p_i a_i``."""
    a = fractions(a)
    zero = [[Fraction(0)] * len(a) for _ in a]
    return lambda p: (a, zero, [abs(v) for v in a])


def variance(x):
    """``F(p) = sum p_i x_i^2 - (sum p_i x_i)^2``."""
    x = fractions(x)

    def derivatives(p):
        mu = sum(pi * xi for pi, xi in zip(p, x))
        first = [xi * xi - 2 * mu * xi for xi in x]
        second = [[-2 * xi * xj for xj in x] for xi in x]
        return first, second, [xi * xi + 2 * abs(mu * xi) for xi in x]

    return derivatives


def _assemble(p, J, Ja, Hp, Hpa, derivatives):
    first, second, magnitude = derivatives(p)
    n = range(len(p))
    K = [[sum(second[i][j] * J[j][b] for j in n) for b in n] for i in n]
    Ka = [[sum(abs(second[i][j]) * Ja[j][b] for j in n) for b in n] for i in n]
    H = [
        [sum(J[i][a] * K[i][b] + first[i] * Hp[i][a][b] for i in n) for b in n]
        for a in n
    ]
    S = [
        [sum(Ja[i][a] * Ka[i][b] + magnitude[i] * Hpa[i][a][b] for i in n) for b in n]
        for a in n
    ]
    return H, S


def hessian_w(w, derivatives):
    """Exact ``(H, scale)`` of ``F(w / sum w)`` in ``w``."""
    w = fractions(w)
    n, W = len(w), sum(w)
    p = [wi / W for wi in w]
    d = [[int(i == a) for a in range(n)] for i in range(n)]
    J = [[(d[i][a] - p[i]) / W for a in range(n)] for i in range(n)]
    Ja = [[(d[i][a] + p[i]) / W for a in range(n)] for i in range(n)]
    Hp = [
        [[(2 * p[i] - d[i][a] - d[i][b]) / W**2 for b in range(n)] for a in range(n)]
        for i in range(n)
    ]
    Hpa = [
        [[(2 * p[i] + d[i][a] + d[i][b]) / W**2 for b in range(n)] for a in range(n)]
        for i in range(n)
    ]
    return _assemble(p, J, Ja, Hp, Hpa, derivatives)


def hessian_theta(w, derivatives):
    """Exact ``(H, scale)`` of ``F(softmax(theta))`` at ``w = exp(theta)``."""
    w = fractions(w)
    n, W = len(w), sum(w)
    p = [wi / W for wi in w]
    d = [[int(i == a) for a in range(n)] for i in range(n)]

    def hp(i, a, b, s):  # s = -1: exact entry; s = +1: bound on its terms
        first = p[i] * (d[i][a] + s * p[a]) * (d[i][b] + s * p[b])
        return first + s * p[i] * p[a] * (d[a][b] + s * p[b])

    J = [[p[i] * (d[i][a] - p[a]) for a in range(n)] for i in range(n)]
    Ja = [[p[i] * (d[i][a] + p[a]) for a in range(n)] for i in range(n)]
    Hp = [[[hp(i, a, b, -1) for b in range(n)] for a in range(n)] for i in range(n)]
    Hpa = [[[hp(i, a, b, 1) for b in range(n)] for a in range(n)] for i in range(n)]
    return _assemble(p, J, Ja, Hp, Hpa, derivatives)


def classify(got, exact, scale):
    """``'ok'`` or a failure label: ``nan``; ``finite`` (exact value exceeds
    float64 but a finite number came back); ``sign`` (infinite, wrong sign);
    ``inf`` (representable exact value, infinite result); ``err``."""
    got = float(got)
    if np.isnan(got):
        return "nan"
    tol = 64 * EPS * scale + abs(exact) / 10**12
    if abs(exact) > max(BIG, tol):
        if not np.isinf(got):
            return "finite"
        return "ok" if np.sign(got) == (1 if exact > 0 else -1) else "sign"
    if tol > BIG:  # representable, but its rounding bound is not
        return "ok" if np.isinf(got) or abs(Fraction(got) - exact) <= tol else "err"
    if not np.isfinite(got):
        return "inf"
    return "ok" if abs(Fraction(got) - exact) <= tol else "err"


def failures(got, oracle):
    """Map ``label -> count`` of the entries of ``got`` that are not ``ok``."""
    H, S = oracle
    got = np.asarray(got)
    out = {}
    for a in range(len(H)):
        for b in range(len(H)):
            label = classify(got[a, b], H[a][b], S[a][b])
            if label != "ok":
                out[label] = out.get(label, 0) + 1
    return out


def population_moment(w):
    pop = PopulationSamples(
        **{k: jnp.asarray(v) for k, v in SAMPLES.items()}, weights=w
    )
    return JointMoments.from_samples(pop, INDEX, REFERENCE).m0[ROW]


def _population_values():  # the moment of each single sample (p = e_i)
    return [float(population_moment(jnp.eye(4)[i])) for i in range(4)]


TARGETS = {
    "mixed_moments": (
        lambda w: mixed_moments(OFFSETS, FACTOR, w)[1][0],
        linear(OFFSETS[:, 0] * FACTOR),
    ),
    "rm_variance": (lambda w: gaussian_rm_cumulants(X, w)[1], variance(X)),
    "population": (population_moment, linear(_population_values())),
}


def _hvp_rows(f):
    def rows(w):
        basis = jnp.eye(w.shape[0], dtype=w.dtype)
        return jax.vmap(lambda e: jax.jvp(jax.grad(f), (w,), (e,))[1])(basis)

    return rows


MODES = {
    "fwd-rev": jax.hessian,
    "fwd-fwd": lambda f: jax.jacfwd(jax.jacfwd(f)),
    "rev-rev": lambda f: jax.jacrev(jax.jacrev(f)),
    "rev-fwd": lambda f: jax.jacrev(jax.jacfwd(f)),
    "hvp": _hvp_rows,
}
