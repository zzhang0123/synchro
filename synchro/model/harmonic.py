"""Harmonic (Dirac-line) channel kernel (``synchro.model.harmonic``).

LABEL: ``eq: smooth channel kernel`` (``H_{P,j} = sum_m Q_m R_j(nu_m)
exp(2 i (c/nu_m)^2 varphi)``, ``nu_m = m nu_B / D``; ``I, V`` without phase),
``eq: channel derivative coefficients`` (``build_basis`` differentiates
:meth:`HarmonicKernel.angular_projection`), ``extra eq: channel kernel`` and
``extra eq: faraday channel kernel``. The natural-basis line powers
``(I_m, Q_m, V_m)`` are the recurrence used by ``synchro.stokes``
(``J_m/sin theta = b_perp (J_{m-1} + J_{m+1})/(2D)``), in erg/s/sr per
electron, at a traced float harmonic index with a static Bessel resolution
``n_nodes`` (default ``2^ceil(log2(max(128, 4 m_max + 64)))``, the automatic
count of ``synchro.bessel``). Harmonics are batched by ``vmap`` inside
``lax.map`` chunks (``m_chunk``; for the angular projections the chunk is
reduced so that one step holds at most ~2^26 Bessel integrand values).

Angular projection, primary route ``quadrature="product"`` (a JAX port of the
manuscript's ``validation/full_response_product.py``, see
``_harmonic_cells.product_cells``): for each harmonic, channel and sign a cell
in ``t = mu eta`` bounded by the channel support, ``t = sign v^4`` on
Gauss-Legendre nodes, ``log mu`` on Gauss-Legendre nodes, and the ``mu < 0``
half through the parity factors ``(1 +/- (-1)^(l+k))``. No Python branch
depends on a traced value. The response vanishes to all orders at the support
edges, so the moving cell boundaries contribute no boundary terms. AD
differentiates the rule itself except in cells with ``0 < lo < 0.1 hi``
(a channel edge near a line at ``t = 0``), whose nodes follow the bounds
affinely in ``t`` (``product_cells``); either way a derivative carries a
quadrature error of its own (estimated by ``numerical``, not bounded).
Secondary route ``quadrature="tensor"``: ``n_mu x n_eta``
Gauss-Legendre in ``(mu, eta)``; it converges slowly near the ``mu eta = 0``
resonance and serves the finite route check of ``basis.py``.

Shapes: scalar ``(gamma, B, mu, eta)``; ``Modes`` ``(n_ch,)``/``(n_weights, n_ch)``;
``ProjectedModes`` over ``MomentIndex.pairs``. Units: Gauss, Hz, m^2 for ``tau``.
Regime: ``gamma <~ 50``, ``m_max <~ 1e3``; Galactic harmonic numbers
(``m ~ 1e11``) are outside the design and refused by ``build_basis``.

Not certified: quadrature error of either angular route (the refined-rule
``numerical`` estimate and the route check live in ``basis.py``), Bessel roundoff beyond the
finite resolution-class checks in the tests, and the physics of the vacuum
helical-orbit, pure-rotation reference (``physical_error`` is ``unbounded``
unless ``E_phys`` is declared).
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._harmonic_cells import (
    auto_nodes,
    harmonic_lines,
    leggauss,
    product_cells,
    static_int,
    sum_harmonics,
)
from ._harmonic_tail import probe_estimate
from ._kernel_helpers import (
    Modes,
    ProjectedModes,
    check_scalar_point,
    gather_pairs,
    gyrofrequency_hz,
    legendre_norm,
    legendre_table,
    phase_coordinate,
    phase_weights,
    resolve_index,
)
from ._support_check import hypothesis_note, outside_support
from .errors import ErrorTerm
from .kernels import required_m_max

_MAX_NODES = 65536
_CHUNK_BUDGET = 1 << 26  # Bessel integrand values per lax.map step
QUADRATURES = ("product", "tensor")


class HarmonicKernel(eqx.Module):
    """Dirac-line harmonic channel kernel; see the module docstring for the contract.

    Static fields: ``m_max`` (harmonics ``1..m_max``), ``n_nodes`` (Bessel
    resolution, ``None`` for the automatic count; an explicit count must be
    ``>= 4 m_max + 2``), ``n_outer``/``n_inner`` (product-cell Gauss-Legendre
    counts), ``n_mu``/``n_eta`` (tensor route), ``m_chunk`` (harmonics
    batched by ``vmap`` inside one ``lax.map`` step), ``tail_probe``
    (harmonics beyond ``m_max`` summed for the truncation estimate, or
    ``None``), ``quadrature`` (default route). ``E_phys`` is an optional
    declared physical-error term (leaf).
    Units: per-electron channel Stokes; ``B`` in
    Gauss, Hz. Assumes vacuum helical-orbit radiation with a pure-rotation
    phase; not certified: that model (``physical_error`` unbounded unless
    ``E_phys`` is declared) and the quadrature beyond the ``numerical`` estimate.
    """

    m_max: int = eqx.field(static=True)
    n_nodes: int | None = eqx.field(static=True)
    n_outer: int = eqx.field(static=True)
    n_inner: int = eqx.field(static=True)
    n_mu: int = eqx.field(static=True)
    n_eta: int = eqx.field(static=True)
    m_chunk: int = eqx.field(static=True)
    tail_probe: int | None = eqx.field(static=True)
    E_phys: ErrorTerm | None
    quadrature: str = eqx.field(static=True)

    name: ClassVar[str] = "harmonic"
    LABEL: ClassVar[tuple[str, ...]] = (
        "eq: smooth channel kernel",
        "eq: channel derivative coefficients",
        "extra eq: channel kernel",
        "extra eq: faraday channel kernel",
    )
    components: ClassVar[tuple[str, ...]] = ("I", "Q", "V")
    required_closures: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        m_max,
        *,
        n_nodes=None,
        n_outer=64,
        n_inner=64,
        n_mu=48,
        n_eta=48,
        m_chunk=64,
        tail_probe=None,
        E_phys=None,
        quadrature="product",
    ):
        m_max = static_int(m_max, "m_max", 1)
        if n_nodes is not None:
            n_nodes = static_int(n_nodes, "n_nodes", 16)
            if n_nodes < 4 * m_max + 2:
                raise ValueError(
                    f"n_nodes={n_nodes} cannot resolve m_max={m_max}: "
                    f"need >= {4 * m_max + 2}"
                )
        elif auto_nodes(m_max) > _MAX_NODES:
            raise ValueError(
                "m_max beyond the validated harmonic regime (n_nodes > 65536)"
            )
        if quadrature not in QUADRATURES:
            raise ValueError(f"quadrature must be one of {QUADRATURES}")
        if E_phys is not None and not isinstance(E_phys, ErrorTerm):
            raise ValueError("E_phys must be an ErrorTerm or None")
        self.m_max = m_max
        self.n_nodes = n_nodes
        self.n_outer = static_int(n_outer, "n_outer", 2)
        self.n_inner = static_int(n_inner, "n_inner", 2)
        self.n_mu = static_int(n_mu, "n_mu", 2)
        self.n_eta = static_int(n_eta, "n_eta", 2)
        self.m_chunk = static_int(m_chunk, "m_chunk", 1)
        self.tail_probe = (
            None if tail_probe is None else static_int(tail_probe, "tail_probe", 1)
        )
        self.E_phys = E_phys
        self.quadrature = quadrature

    # -- harmonic batching -----------------------------------------------------

    def resolution(self) -> int:
        """The static Bessel node count in use."""
        return auto_nodes(self.m_max) if self.n_nodes is None else self.n_nodes

    def harmonics(self) -> jax.Array:
        """``1..m_max`` as floats, shape ``(m_max,)``."""
        return jnp.arange(1, self.m_max + 1, dtype=float)

    def _projection_chunk(self, points: int) -> int:
        """Bound the vmapped Bessel work per ``lax.map`` step to ``_CHUNK_BUDGET``."""
        per_harmonic = 3 * points * self.resolution()
        return max(1, min(self.m_chunk, _CHUNK_BUDGET // max(per_harmonic, 1)))

    # -- protocol ---------------------------------------------------------------

    def line_frequencies(self, gamma, B, mu, eta) -> jax.Array:
        """``nu_m = m nu_B / (1 - beta mu eta)`` [Hz], shape ``(m_max,)``."""
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        beta = jnp.sqrt(1.0 - 1.0 / gamma**2)
        return self.harmonics() * gyrofrequency_hz(gamma, B) / (1.0 - beta * mu * eta)

    def channel_modes(
        self, channels, gamma, B, mu, eta, *, phase=None, depth_ref=0.0, s_depth=1.0
    ):
        gamma, B, mu, eta = check_scalar_point(gamma, B, mu, eta)
        n_nodes = self.resolution()
        n_ch = channels.n_ch
        active = (B > 0.0).astype(float)

        def one(m):
            I, Q, V, nu = harmonic_lines(m, gamma, B, mu, eta, n_nodes=n_nodes)
            nu_safe = jnp.where(nu > 0.0, nu, 1.0)
            response = channels(nu_safe) * active  # (n_ch,)
            w = phase_weights(phase, phase_coordinate(nu_safe), depth_ref, s_depth)
            return response * I, response * V, (response * Q)[None, :] * w[:, None]

        I, V, P = sum_harmonics(one, self.harmonics(), self.m_chunk)
        return Modes(I=I.reshape(n_ch), V=V.reshape(n_ch), P=P)

    def angular_projection(
        self,
        channels,
        gamma,
        B,
        *,
        phase=None,
        depth_ref=0.0,
        s_depth=1.0,
        truncation,
        quadrature=None,
    ):
        quadrature = self.quadrature if quadrature is None else quadrature
        if quadrature not in QUADRATURES:
            raise ValueError(f"quadrature must be one of {QUADRATURES}")
        index = resolve_index(truncation, self.components)
        gamma, B, _, _ = check_scalar_point(gamma, B, 0.0, 0.0)
        degrees = (index.truncation.L_mu, index.truncation.L_eta)
        route = (
            self._project_product if quadrature == "product" else self._project_tensor
        )
        grid_I, grid_V, grid_P = route(
            channels, gamma, B, phase, depth_ref, s_depth, degrees
        )
        norm = legendre_norm(index.pairs)
        return ProjectedModes(
            I=gather_pairs(grid_I, index.pairs) * norm,
            V=gather_pairs(grid_V, index.pairs) * norm,
            P=gather_pairs(grid_P, index.pairs) * norm,
        )

    def _project_product(self, channels, gamma, B, phase, depth_ref, s_depth, degrees):
        L_mu, L_eta = degrees
        n_nodes = self.resolution()
        support = jnp.asarray(channels.support, dtype=float)
        active = (B > 0.0).astype(float)
        chunk = self._projection_chunk(2 * self.n_outer * self.n_inner)
        ell = np.arange(L_mu + 1)[:, None] + np.arange(L_eta + 1)[None, :]
        parity = jnp.asarray((-1.0) ** ell)

        def per_channel(carry, args):
            nu_lo, nu_hi, j = args

            def one(m):
                mu, eta, measure = product_cells(
                    m, gamma, B, nu_lo, nu_hi, self.n_outer, self.n_inner
                )
                I, Q, V, nu = harmonic_lines(m, gamma, B, mu, eta, n_nodes=n_nodes)
                nu_safe = jnp.where(nu > 0.0, nu, 1.0)
                response = jnp.take(channels(nu_safe), j, axis=0) * active
                w = phase_weights(phase, phase_coordinate(nu_safe), depth_ref, s_depth)
                p_l = legendre_table(mu, L_mu)
                p_k = legendre_table(eta, L_eta)
                meas = measure * response
                proj_I = jnp.einsum("sab,sabl,sabk->lk", meas * I, p_l, p_k)
                proj_V = jnp.einsum("sab,sabl,sabk->lk", meas * V, p_l, p_k)
                proj_P = jnp.einsum(
                    "wsab,sabl,sabk->wlk", (meas * Q)[None] * w, p_l, p_k
                )
                return proj_I, proj_V, proj_P

            return carry, sum_harmonics(one, self.harmonics(), chunk)

        _, (I, V, P) = jax.lax.scan(
            per_channel,
            None,
            (support[:, 0], support[:, 1], jnp.arange(channels.n_ch)),
        )
        # mu < 0 half: I, Q even and V odd under (mu, eta) -> (-mu, -eta).
        return (
            I * (1.0 + parity),
            V * (1.0 - parity),
            jnp.moveaxis(P, 0, 1) * (1.0 + parity),
        )

    def _project_tensor(self, channels, gamma, B, phase, depth_ref, s_depth, degrees):
        L_mu, L_eta = degrees
        n_nodes = self.resolution()
        x_mu, w_mu = leggauss(self.n_mu)
        x_eta, w_eta = leggauss(self.n_eta)
        MU, ETA = jnp.meshgrid(x_mu, x_eta, indexing="ij")
        p_l = legendre_table(x_mu, L_mu) * w_mu[:, None]
        p_k = legendre_table(x_eta, L_eta) * w_eta[:, None]
        active = (B > 0.0).astype(float)
        chunk = self._projection_chunk(self.n_mu * self.n_eta)

        def one(m):
            I, Q, V, nu = harmonic_lines(m, gamma, B, MU, ETA, n_nodes=n_nodes)
            nu_safe = jnp.where(nu > 0.0, nu, 1.0)
            response = channels(nu_safe) * active  # (n_ch, n_mu, n_eta)
            w = phase_weights(phase, phase_coordinate(nu_safe), depth_ref, s_depth)
            proj_I = jnp.einsum("cab,al,bk->clk", response * I, p_l, p_k)
            proj_V = jnp.einsum("cab,al,bk->clk", response * V, p_l, p_k)
            proj_P = jnp.einsum("cab,wab,al,bk->wclk", response * Q, w, p_l, p_k)
            return proj_I, proj_V, proj_P

        return sum_harmonics(one, self.harmonics(), chunk)

    def physical_error(self, channels) -> ErrorTerm:
        if self.E_phys is not None:
            return self.E_phys
        return ErrorTerm.unbounded(
            "vacuum helical-orbit, pure-rotation reference (main.tex sec: stokes "
            "harmonic, sec: screen model); zero only if declared",
            "E_phys",
        )

    def truncation_error(
        self, support, channels, *, samples=None, reference=None, phase=None
    ):
        """Omitted-harmonic term ``sum_{m > m_max} |S_m R_j|`` per electron, ``(n_ch, 4)``.

        ``bound`` zero when ``m_max >= required_m_max(support, channels)``,
        every channel support is finite and positive and no emitting sample
        leaves the support (``_support_check``; the note says whether that
        was checked). ``estimate`` from the probe harmonics (``_harmonic_tail``)
        over ``samples`` or the angular grid at ``reference``; ``unbounded``
        otherwise, naming any violating variable and its range. Columns are
        ``I, Q, U, V`` with the natural ``|Q_m|`` sum in both ``Q`` and ``U``.
        """
        n_ch = channels.n_ch
        required = required_m_max(support, channels)
        supp = np.asarray(channels.support, dtype=float)
        compact = bool(np.all(np.isfinite(supp)) and np.all(supp[:, 0] > 0))
        outside = outside_support(samples, support, harmonic_only=True)
        if self.m_max >= required and compact and not outside:
            return ErrorTerm(
                jnp.zeros((n_ch, 4)),
                "bound",
                f"m_max={self.m_max} >= required_m_max={required}: no line above "
                "m_max meets a channel on the declared support; "
                + hypothesis_note(samples, outside),
                "E_num",
            )
        reason = f"m_max={self.m_max} < required_m_max={required}"
        if self.m_max >= required and not compact:
            reason = "a channel support is not finite with a positive lower edge"
        if outside:
            reason = (
                f"samples outside the declared Support ({'; '.join(outside)}), so "
                f"required_m_max={required} does not cover them"
            )
        if self.tail_probe is None or (samples is None and reference is None):
            return ErrorTerm.unbounded(
                f"{reason}; set tail_probe and supply samples or a reference for "
                "an estimate, or raise m_max (or widen the Support)",
                "E_num",
            )
        return probe_estimate(self, channels, samples, reference, f"{reason}: ")

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("label", self.LABEL),
            ("m_max", self.m_max),
            ("n_nodes", self.resolution()),
            ("quadrature", self.quadrature),
            ("n_outer", self.n_outer),
            ("n_inner", self.n_inner),
            ("n_mu", self.n_mu),
            ("n_eta", self.n_eta),
            ("m_chunk", self.m_chunk),
            ("tail_probe", self.tail_probe),
            ("E_phys", None if self.E_phys is None else self.E_phys.kind),
        )


__all__ = ["HarmonicKernel", "harmonic_lines", "product_cells", "auto_nodes"]
