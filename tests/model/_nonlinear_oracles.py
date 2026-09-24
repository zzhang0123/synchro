"""NumPy oracles and fixtures shared by ``test_nonlinear*.py`` (not collected).

The response matrix of a ``PolynomialTestKernel`` is written down directly
from its coefficients (channel-major rows ``4 j + s``, ``s`` in ``I, Q, U,
V``; ``P = sum c w_b M2`` split into real and imaginary parts), the node
features from ``scipy.special.eval_legendre``, and gradients from central
differences with one Richardson step. When ``synchro.model.basis`` is
importable the basis comes from ``build_basis``; otherwise a private stub
with the same three attributes (``index``, ``reference``,
``response_matrix``) stands in.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy.special import eval_legendre

from synchro.constants import C_SI_M
from synchro.model.channels import Channels
from synchro.model.kernels import PolynomialTestKernel
from synchro.model.moments import PopulationSamples, Reference, Support
from synchro.model.phase import TaylorPhase

try:  # phase 2 module (E3); the stub below is used until it lands
    from synchro.model.basis import build_basis
except ImportError:  # pragma: no cover - depends on the build order
    build_basis = None

REF = (5.2, 2.1, 0.3, (1.3, 0.7, 0.9))


def reference_of(ref=REF):
    gamma0, B0, depth_ref, scales = ref
    return Reference(gamma0, B0, depth_ref, scales=scales)


def taylor_weights(tau, degree, depth_ref, s_depth):
    """``w_b = exp(i tau depth_ref) (i tau s_depth)^b / b!`` for ``b <= degree``."""
    return np.array(
        [
            np.exp(1j * tau * depth_ref) * (1j * tau * s_depth) ** b / math.factorial(b)
            for b in range(degree + 1)
        ]
    )


def polynomial_response(coefficients, line_nu, index, ref=REF):
    """Channel-major ``(4 n_ch, n_real)`` response of a polynomial kernel."""
    _, _, depth_ref, (_, _, s_depth) = ref
    n_ch = coefficients.shape[1]
    n0, n2 = index.n0, index.n2
    C = np.zeros((4 * n_ch, index.n_real))
    degree = index.truncation.max_b()
    for a, (l, k, r, s, _) in enumerate(index.h0):
        for j in range(n_ch):  # eq: angular parity: I on l+k even, V on odd
            if (l + k) % 2 == 0:
                C[4 * j + 0, a] = coefficients[0, j, l, k, r, s]
            else:
                C[4 * j + 3, a] = coefficients[2, j, l, k, r, s]
    for i, (l, k, r, s, b) in enumerate(index.h2):
        for j in range(n_ch):
            tau = 2.0 * (C_SI_M / line_nu[j]) ** 2
            w = taylor_weights(tau, degree, depth_ref, s_depth)[b]
            c = coefficients[1, j, l, k, r, s]
            C[4 * j + 1, n0 + i] = c * w.real
            C[4 * j + 1, n0 + n2 + i] = -c * w.imag
            C[4 * j + 2, n0 + i] = c * w.imag
            C[4 * j + 2, n0 + n2 + i] = c * w.real
    return C


class StubBasis:
    """Minimal stand-in for ``SpectralBasis`` (index, reference, response_matrix)."""

    def __init__(self, index, reference, C):
        self.index = index
        self.reference = reference
        self._C = jnp.asarray(C)

    def response_matrix(self):
        return self._C


def polynomial_basis(index, n_ch, seed=0, ref=REF, *, force_stub=False):
    """``(basis, C_oracle)`` for random polynomial coefficients on ``index``."""
    rng = np.random.default_rng(seed)
    t = index.truncation
    coefficients = rng.standard_normal(
        (3, n_ch, t.L_mu + 1, t.L_eta + 1, t.N + 1, t.N + 1)
    )
    line_nu = 1.0e8 * (1.0 + 0.5 * rng.uniform(size=n_ch))
    C = polynomial_response(coefficients, line_nu, index, ref)
    reference = reference_of(ref)
    if build_basis is None or force_stub:
        return StubBasis(index, reference, C), C
    gamma0, B0, _, (s_gamma, s_B, _) = ref
    kernel = PolynomialTestKernel(
        coefficients, line_nu, gamma0=gamma0, B0=B0, s_gamma=s_gamma, s_B=s_B
    )
    channels = Channels.bump(line_nu, 0.1 * line_nu, n_nu=16)
    support = Support(
        gamma=(2.0, 9.0), B=(0.5, 4.0), depth=(-3.0, 3.0), truncated=False
    )
    basis = build_basis(
        kernel,
        channels,
        t,
        reference,
        support=support,
        phase=TaylorPhase(t.max_b()),
        convergence=False,
    )
    return basis, C


def random_nodes(seed, n):
    rng = np.random.default_rng(seed)
    return dict(
        gamma=4.0 + 3.0 * rng.uniform(size=n),
        B=1.0 + 2.5 * rng.uniform(size=n),
        mu=rng.uniform(-0.9, 0.9, n),
        eta=rng.uniform(-0.9, 0.9, n),
        phi=rng.uniform(0, 2 * np.pi, n),
        depth=rng.uniform(-2.0, 2.0, n),
    )


def samples_of(pop, weights=None):
    return PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=weights,
    )


def node_features(pop, index, ref=REF):
    """NumPy ``(S, n_real)`` features in the ``to_vector`` layout."""
    gamma0, B0, depth_ref, (sg, sB, sd) = ref
    zg = (pop["gamma"] - gamma0) / sg
    zB = (pop["B"] - B0) / sB
    zd = (pop["depth"] - depth_ref) / sd

    def row(l, k, r, s, b):
        return (
            eval_legendre(l, pop["mu"])
            * eval_legendre(k, pop["eta"])
            * zg**r
            * zB**s
            * zd**b
        )

    real = np.stack([row(*r) for r in index.h0], axis=1)
    cplx = (
        np.stack([row(*r) for r in index.h2], axis=1) * np.exp(2j * pop["phi"])[:, None]
    )
    return np.concatenate([real, cplx.real, cplx.imag], axis=1)


def fd_gradient(f, z, h=1e-4):
    """Central differences with one Richardson step (error ``O(h^4)``)."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    for i in range(z.size):
        e = np.zeros_like(z)
        e[i] = 1.0

        def d(step):
            return (f(z + step * e) - f(z - step * e)) / (2 * step)

        out[i] = (4.0 * d(h / 2) - d(h)) / 3.0
    return out


__all__ = [
    "REF",
    "reference_of",
    "taylor_weights",
    "polynomial_response",
    "StubBasis",
    "polynomial_basis",
    "random_nodes",
    "samples_of",
    "node_features",
    "fd_gradient",
    "build_basis",
]
