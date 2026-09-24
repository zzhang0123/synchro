"""Population statistics of the finite joint response (``synchro.model.moments``).

LABEL: ``eq: explicit joint moments`` (``JointMoments``), the dimensionless
displacement coordinates of ``eq: local response remainder`` (``Reference.z``),
the declared population support of ``eq: channel error budget``
(``Support``) and the discrete electron-number measure of
``eq: joint emission depth`` (``PopulationSamples``).

Coordinates. Moments are stored in the dimensionless displacements
``z = ((gamma - gamma0)/s_gamma, (B - B0)/s_B, (depth - depth_ref)/s_depth)``
with the Legendre cosines ``mu = cos(alpha)``, ``eta = cos(theta)`` and the
sky azimuth ``phi``. Every row ``(l, k, r, s, b)`` of the index is the
weighted average of ``P_l(mu) P_k(eta) z_gamma^r z_B^s z_depth^b`` (``M0``)
or of the same product times ``exp(2 i phi)`` (``M2``). The raw-displacement
form of the manuscript is ``M_raw = s_gamma^r s_B^s s_depth^b M_z``
(``to_raw_displacements``).

Shapes: sample columns are ``(S,)``; ``m0`` is ``(n0,)``, ``m2`` ``(n2,)``
complex, ``m0_ext`` ``(n0_ext,)``; ``to_vector()`` is ``(n_real,)``.
Units: ``gamma`` dimensionless (``> 1``), ``B`` Gauss (``> 0``), ``depth``
rad/m^2, ``mu, eta`` in ``[-1, 1]``, ``phi`` radians; scales carry the units
of their variable. Weights are relative electron-number masses, normalised as
in ``synchro.rm._screen_samples``.

Exact by construction: weighted sums over the supplied discrete measure (to
floating-point rounding); JIT and autodiff through sample values, weights and
reference are validated by the package tests (finite checks, not
certificates). Not certified:
sampling or quadrature error of the discrete measure relative to a continuous
population, feasibility of a vector supplied to ``from_vector``, and any
factorised construction beyond the supplied samples (``parameter_map``,
``discrepancy="measured"`` measure the factorisation error on those samples only).
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._provenance import strict_json
from .errors import ErrorTerm
from ._population import (  # noqa: F401  (re-exported)
    VARIABLES,
    PopulationSamples,
    Reference,
    Support,
    _as_float,
    _finite,
)

LABEL = "eq: explicit joint moments"
_CHUNK = 4096


def _legendre_table(x, degree):
    """``(..., degree + 1)`` Legendre values by the three-term recurrence."""
    columns = [jnp.ones_like(x)]
    if degree >= 1:
        columns.append(x)
    for n in range(1, degree):
        columns.append(((2 * n + 1) * x * columns[n] - n * columns[n - 1]) / (n + 1))
    return jnp.stack(columns, axis=-1)


def _power_table(x, degree):
    columns = [jnp.ones_like(x)]
    for _ in range(degree):
        columns.append(columns[-1] * x)
    return jnp.stack(columns, axis=-1)


def _row_products(mu, eta, z, rows, maxima):
    """``(chunk, n_rows)`` products ``P_l P_k z_g^r z_B^s z_d^b`` for static rows."""
    tables = (
        _legendre_table(mu, maxima[0]),
        _legendre_table(eta, maxima[1]),
        _power_table(z[..., 0], maxima[2]),
        _power_table(z[..., 1], maxima[3]),
        _power_table(z[..., 2], maxima[4]),
    )
    out = jnp.ones(mu.shape + (rows.shape[0],), dtype=mu.dtype)
    for table, exponent in zip(tables, rows.T):
        out = out * table[..., exponent]
    return out


def _weighted_sums(samples, reference, rows_real, rows_complex):
    """Weighted sums of the row monomials over chunks of samples."""
    rows = np.concatenate([rows_real, rows_complex], axis=0)
    maxima = tuple(int(m) for m in rows.max(axis=0)) if rows.size else (0,) * 5
    n_real = rows_real.shape[0]
    w = samples.normalised_weights()
    z = reference.z(samples.gamma, samples.B, samples.depth)
    size = samples.size
    chunk = min(size, _CHUNK)
    pad = (-size) % chunk

    def padded(value, mode):
        return jnp.pad(value, [(0, pad)] + [(0, 0)] * (value.ndim - 1), mode=mode)

    leaves = (
        padded(w, "constant").reshape(-1, chunk),
        padded(samples.mu, "edge").reshape(-1, chunk),
        padded(samples.eta, "edge").reshape(-1, chunk),
        padded(z, "edge").reshape(-1, chunk, 3),
        padded(samples.phi, "edge").reshape(-1, chunk),
    )

    def accumulate(carry, leaf):
        w_c, mu_c, eta_c, z_c, phi_c = leaf
        chi = _row_products(mu_c, eta_c, z_c, rows, maxima)
        real = carry[0] + w_c @ chi[:, :n_real]
        cplx = carry[1] + (w_c * jnp.exp(2j * phi_c)) @ chi[:, n_real:]
        return (real, cplx), None

    init = (
        jnp.zeros(n_real, dtype=w.dtype),
        jnp.zeros(rows_complex.shape[0], dtype=jnp.result_type(w, 1j)),
    )
    (real, cplx), _ = jax.lax.scan(accumulate, init, leaves)
    return _finite(real, "M0 moments"), _finite(cplx, "M2 moments")


UNIT_MOMENT_TOL = 1e-12  # default |m[0] - 1| accepted by JointMoments.from_vector


class JointMoments(eqx.Module):
    """Retained joint moments ``M0[h0]``, ``M2[h2]`` (``eq: explicit joint moments``).

    ``m0`` is real ``(n0,)`` in ``index.h0`` order, ``m2`` complex ``(n2,)``
    in ``index.h2`` order, ``m0_ext`` real ``(n0_ext,)`` over ``index.h0_ext``
    (``b >= 1`` rows outside ``m``; ``None`` when built from a vector).
    ``to_vector`` is ``concat(m0, Re m2, Im m2)`` (``extra eq: finite fit
    model``). ``assumptions`` records the parameter maps that produced the
    moments; ``discrepancy`` is a moment-level ``Delta_a`` (``n_real,``).
    Shape mismatches raise ``ValueError``; nothing here checks feasibility.
    Units: dimensionless (moments of ``z`` displacements and Legendre
    polynomials); ``to_raw_displacements`` restores ``s_gamma^r s_B^s
    s_depth^b``. Assumes nothing unless ``assumptions`` is nonempty. Not
    certified: feasibility (``fit.diagnostics.feasibility_checks`` gives
    necessary conditions only) and the accuracy of moments supplied through
    ``from_vector``.
    """

    index: object = eqx.field(static=True)
    m0: jax.Array
    m2: jax.Array
    reference: Reference
    m0_ext: jax.Array | None = None
    assumptions: tuple = eqx.field(static=True, default=())
    discrepancy: object | None = None
    LABEL: ClassVar[str] = LABEL

    def __check_init__(self):
        index = self.index
        if self.m0.shape != (index.n0,):
            raise ValueError(f"m0 must have shape ({index.n0},), got {self.m0.shape}")
        if self.m2.shape != (index.n2,):
            raise ValueError(f"m2 must have shape ({index.n2},), got {self.m2.shape}")
        n_ext = len(index.h0_ext)
        if self.m0_ext is not None and self.m0_ext.shape != (n_ext,):
            raise ValueError(
                f"m0_ext must have shape ({n_ext},), got {self.m0_ext.shape}"
            )

    @classmethod
    def from_samples(
        cls, samples, index, reference, *, parameter_map=None, discrepancy=None
    ) -> "JointMoments":
        """Exact moments of a discrete population for every retained row.

        Computes ``h0``, ``h0_ext`` and ``h2`` by chunked weighted sums
        (JIT and autodiff safe). With ``parameter_map`` the joint tensor is
        projected onto the map and the factorised tensor is returned (with
        the map's record and discrepancy); ``discrepancy="measured"`` stores
        ``|m_joint - m_fac|`` on these samples as a ``measured`` term, which
        is not a bound for any other population.
        """
        if discrepancy not in (None, "measured"):
            raise ValueError("discrepancy must be None or 'measured'")
        if discrepancy == "measured" and parameter_map is None:
            raise ValueError("discrepancy='measured' needs a parameter_map")
        if parameter_map is not None and not (
            callable(parameter_map) and hasattr(parameter_map, "project")
        ):
            raise ValueError("parameter_map must be a ParameterMap")
        rows_real = index.exponents(tuple(index.h0) + tuple(index.h0_ext))
        rows_complex = index.exponents(index.h2)
        real, cplx = _weighted_sums(samples, reference, rows_real, rows_complex)
        joint = cls(
            index=index,
            m0=real[: index.n0],
            m2=cplx,
            reference=reference,
            m0_ext=real[index.n0 :],
        )
        if parameter_map is None:
            return joint
        factorised = parameter_map(parameter_map.project(joint), reference)
        if discrepancy != "measured":
            return factorised
        delta = jnp.abs(joint.to_vector() - factorised.to_vector())
        term = ErrorTerm(
            delta,
            "measured",
            note=(
                f"|m_joint - m_fac| for '{factorised.assumptions[0].name}' on the "
                f"{samples.size} supplied samples; not a bound for other populations"
            ),
            manuscript_term="E_phys",
        )
        return cls(
            index=index,
            m0=factorised.m0,
            m2=factorised.m2,
            reference=reference,
            m0_ext=factorised.m0_ext,
            assumptions=factorised.assumptions,
            discrepancy=term,
        )

    @classmethod
    def from_vector(
        cls, index, m, reference, *, assumptions=(), tol=UNIT_MOMENT_TOL
    ) -> "JointMoments":
        """Unpack a real ``(n_real,)`` vector; ``m[0]`` must equal 1 within ``tol``.

        ``tol`` (default ``UNIT_MOMENT_TOL = 1e-12``) is the allowed
        ``|m[0] - 1|``; since the target is 1 it is also the relative
        tolerance. The default admits the roundoff of normalised weights
        (``from_samples`` gives ``m[0] = 1 - O(1e-16)``); ``tol=0.0`` asks for
        exact equality. Checked with ``eqx.error_if`` (raises at run time,
        also under JIT); the vector is not otherwise checked for feasibility.
        """
        m = _as_float(m, "m")
        if m.shape != (index.n_real,):
            raise ValueError(f"m must have shape ({index.n_real},), got {m.shape}")
        m = _finite(m, "m")
        m = eqx.error_if(
            m, jnp.abs(m[0] - 1.0) > tol, "m[0] must equal 1 (M0[l=0,k=0;r=0,s=0])"
        )
        n0, n2 = index.n0, index.n2
        return cls(
            index=index,
            m0=m[:n0],
            m2=m[n0 : n0 + n2] + 1j * m[n0 + n2 :],
            reference=reference,
            assumptions=tuple(assumptions),
        )

    def to_vector(self) -> jax.Array:
        return jnp.concatenate([self.m0, jnp.real(self.m2), jnp.imag(self.m2)])

    def get(self, h, l, k, r, s, b) -> jax.Array:
        """One moment; ``h=0, b>=1`` reads ``m0_ext``; complex for ``h=2``."""
        row = (int(l), int(k), int(r), int(s), int(b))
        if h == 2:
            table, values = self.index.h2, self.m2
        elif h == 0 and row[4] == 0:
            table, values = self.index.h0, self.m0
        elif h == 0:
            if self.m0_ext is None:
                raise ValueError(
                    "b >= 1 rows of M0 need m0_ext (built by from_samples)"
                )
            table, values = self.index.h0_ext, self.m0_ext
        else:
            raise ValueError(f"h must be 0 or 2, got {h!r}")
        try:
            return values[table.index(row)]
        except ValueError:
            raise ValueError(f"row h={h}, (l,k,r,s,b)={row} is not retained") from None

    def to_raw_displacements(self) -> "JointMoments":
        """Moments of ``(gamma-gamma0)^r (B-B0)^s (depth-depth_ref)^b`` (unit scales)."""
        s_gamma, s_B, s_depth = self.reference.scales

        def factor(rows):
            e = self.index.exponents(rows)
            return s_gamma ** e[:, 2] * s_B ** e[:, 3] * s_depth ** e[:, 4]

        f0, f2 = factor(self.index.h0), factor(self.index.h2)
        discrepancy = self.discrepancy
        if discrepancy is not None and getattr(discrepancy, "value", None) is not None:
            discrepancy = discrepancy.scaled(jnp.concatenate([f0, f2, f2]))
        reference = Reference(
            self.reference.gamma0, self.reference.B0, self.reference.depth_ref
        )
        return JointMoments(
            index=self.index,
            m0=self.m0 * f0,
            m2=self.m2 * f2,
            reference=reference,
            m0_ext=(
                None if self.m0_ext is None else self.m0_ext * factor(self.index.h0_ext)
            ),
            assumptions=self.assumptions,
            discrepancy=discrepancy,
        )

    def to_dict(self) -> dict:
        """Strict-JSON summary (eager use only; non-finite floats become ``None``)."""
        t = self.index.truncation
        return strict_json(
            {
                "index": {
                    "L_mu": t.L_mu,
                    "L_eta": t.L_eta,
                    "N": t.N,
                    "depth_degree": t.depth_degree,
                    "parity": self.index.parity,
                    "n0": self.index.n0,
                    "n2": self.index.n2,
                    "n_real": self.index.n_real,
                },
                "vector": np.asarray(self.to_vector()).tolist(),
                "labels": list(self.index.labels()),
                "reference": dict(self.reference.describe()),
                "m0_ext": (
                    None if self.m0_ext is None else np.asarray(self.m0_ext).tolist()
                ),
                "assumptions": [a.to_dict() for a in self.assumptions],
                "discrepancy": (
                    None if self.discrepancy is None else self.discrepancy.to_dict()
                ),
            }
        )


__all__ = [
    "Reference",
    "Support",
    "PopulationSamples",
    "JointMoments",
    "VARIABLES",
    "UNIT_MOMENT_TOL",
]
