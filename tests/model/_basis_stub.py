"""Minimal ``SpectralBasis`` stand-in for the ``predict``/``adapters`` tests (not collected).

Builds the frozen ``SpectralBasis`` field set of INTERFACES.md from a kernel's
``angular_projection`` by nested ``jax.jacfwd`` in ``z = (z_gamma, z_B)`` at
the reference, so that the tests of ``synchro.model.predict`` do not depend
on ``synchro.model.basis`` (phase 2, another engineer). Provenance and the
kernel terms follow FINAL_DESIGN Sections 6 and 8; ``numerical`` is left
unbounded (no two-quadrature estimate here).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from synchro.model.errors import ErrorTerm, Provenance
from synchro.model.index import MomentIndex
from synchro.model.phase import CumulantScreen, TaylorPhase


class KernelTerms(eqx.Module):
    physical_kernel: ErrorTerm
    harmonic_truncation: ErrorTerm
    numerical: ErrorTerm
    screen_exponent: ErrorTerm


class StubBasis(eqx.Module):
    index: MomentIndex = eqx.field(static=True)
    truncation: object = eqx.field(static=True)
    channels: object
    reference: object
    support: object
    I_basis: jax.Array
    V_basis: jax.Array
    P_basis: jax.Array
    kernel_terms: KernelTerms
    provenance: Provenance = eqx.field(static=True)
    certified_orders: tuple = eqx.field(static=True, default=(0, 1, 2))

    @property
    def n_ch(self) -> int:
        return self.channels.n_ch

    def response_matrix(self) -> jax.Array:
        """``(4 n_ch, n_real)`` per FINAL_DESIGN Section 3 (channel-major I, Q, U, V)."""
        index, n_ch = self.index, self.n_ch
        n0, n2 = index.n0, index.n2
        par_I = jnp.asarray(index.parity_I(), dtype=float)
        par_V = jnp.asarray(index.parity_V(), dtype=float)
        C = jnp.zeros((n_ch, 4, index.n_real))
        C = C.at[:, 0, :n0].set(self.I_basis * par_I)
        C = C.at[:, 3, :n0].set(self.V_basis * par_V)
        a, b = jnp.real(self.P_basis), jnp.imag(self.P_basis)
        C = C.at[:, 1, n0 : n0 + n2].set(a)
        C = C.at[:, 1, n0 + n2 :].set(-b)
        C = C.at[:, 2, n0 : n0 + n2].set(b)
        C = C.at[:, 2, n0 + n2 :].set(a)
        return C.reshape(4 * n_ch, index.n_real)


def _taylor_coefficients(fn, order):
    """``{(r, s): d^r_0 d^s_1 fn / (r! s!)}`` at ``z = 0`` for ``r + s <= order``."""
    derivatives = [fn]
    for _ in range(order):
        derivatives.append(jax.jacfwd(derivatives[-1]))
    zero = jnp.zeros(2)
    values = [jax.jit(d)(zero) for d in derivatives]
    out = {}
    for r in range(order + 1):
        for s in range(order + 1 - r):
            tensor = values[r + s]
            for i in [0] * r + [1] * s:
                tensor = jax.tree.map(lambda t, i=i: t[..., i], tensor)
            factor = 1.0 / (math.factorial(r) * math.factorial(s))
            out[(r, s)] = jax.tree.map(lambda t: t * factor, tensor)
    return out


def build_stub_basis(
    kernel,
    channels,
    truncation,
    reference,
    support,
    *,
    phase=None,
    quadrature=None,
    certified_orders=(0, 1, 2),
    numerical=None,
):
    index = MomentIndex.build(truncation, components=tuple(kernel.components))
    phase = TaylorPhase(truncation.max_b()) if phase is None else phase
    s_gamma, s_B, s_depth = reference.scales

    def projected(z):
        modes = kernel.angular_projection(
            channels,
            reference.gamma0 + s_gamma * z[0],
            reference.B0 + s_B * z[1],
            phase=phase,
            depth_ref=reference.depth_ref,
            s_depth=s_depth,
            truncation=index,
            quadrature=quadrature,
        )
        return modes.I, modes.V, modes.P

    coefficients = _taylor_coefficients(projected, truncation.N)
    pair_of = {pair: i for i, pair in enumerate(index.pairs)}
    n_ch = channels.n_ch
    I = np.zeros((n_ch, index.n0))
    V = np.zeros((n_ch, index.n0))
    P = np.zeros((n_ch, index.n2), dtype=complex)
    for i, (l, k, r, s, _) in enumerate(index.h0):
        cI, cV, _ = coefficients[(r, s)]
        I[:, i] = np.asarray(cI)[:, pair_of[(l, k)]]
        V[:, i] = np.asarray(cV)[:, pair_of[(l, k)]]
    for i, (l, k, r, s, b) in enumerate(index.h2):
        _, _, cP = coefficients[(r, s)]
        P[:, i] = np.asarray(cP)[b, :, pair_of[(l, k)]]
    screen = ErrorTerm.not_applicable("Taylor route: no screen exponent")
    if isinstance(phase, CumulantScreen):
        screen = ErrorTerm.unbounded(
            "cumulant screen exponent not evaluated by the stub"
        )
    terms = KernelTerms(
        physical_kernel=kernel.physical_error(channels),
        harmonic_truncation=kernel.truncation_error(support, channels),
        numerical=(
            ErrorTerm.unbounded("stub basis: no two-quadrature estimate")
            if numerical is None
            else numerical
        ),
        screen_exponent=screen,
    )
    provenance = Provenance(
        package_version="stub",
        kernel=kernel.describe(),
        channels=channels.describe(),
        truncation=(
            ("L_mu", truncation.L_mu),
            ("L_eta", truncation.L_eta),
            ("N", truncation.N),
            ("depth_degree", truncation.depth_degree),
        ),
        reference=reference.describe(),
        support=support.describe(),
        phase_route=phase.describe(),
        quadrature=(("route", "stub nested jacfwd"),),
        assumptions=tuple(phase.forced_assumptions()),
        units="erg/s/sr per electron times amplitude",
        numerics=(),
        certified_orders=tuple(certified_orders),
        finite_checks=(),
        notes=("incident polarisation not modelled",),
    )
    return StubBasis(
        index=index,
        truncation=truncation,
        channels=channels,
        reference=reference,
        support=support,
        I_basis=jnp.asarray(I),
        V_basis=jnp.asarray(V),
        P_basis=jnp.asarray(P),
        kernel_terms=terms,
        provenance=provenance,
        certified_orders=tuple(certified_orders),
    )


def response_oracle(basis, m):
    """NumPy contraction of the response matrix layout with a moment vector."""
    C = np.asarray(basis.response_matrix())
    return (C @ np.asarray(m)).reshape(basis.n_ch, 4)
