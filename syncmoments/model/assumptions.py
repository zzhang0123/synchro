"""Assumptions as parametrisations of the moment vector (``syncmoments.model.assumptions``).

LABEL: ``eq: independent screen moments`` (``{depth} | rest``), ``eq:
independent gaussian screen`` (Gaussian depth closure on the Taylor route),
``detail-eq: angular factorisation error`` (``{phi} | rest``, constant
conditional circular moment), ``extra eq: pitch angle expansion``
(``uniform_mu`` zeroes every ``l >= 1`` coefficient), ``extra eq: nodal
distribution`` (weights on fixed nodes); other partitions and the symmetry
closures ``pitch_symmetric``, ``field_reversal_symmetric`` are ``[extension]``.

A ``ParameterMap`` is a map ``theta -> m`` from free parameters to the
``(n_real,)`` moment vector of ``eq: explicit joint moments``; it never switches the
physics. Under a factorisation (partition of ``VARS``) every retained moment
is a product of group marginals; closures replace a marginal by a fixed
table. Moments are stored in the displacements ``z`` of ``eq: local response
remainder``; depth closures take rad/m^2 and convert with the reference.
Free counts at ``(2, 2, 2)``: 153, 115, 113, 88, 75, 57, 12, 115, 129 for the
named constructors in order. Free tables come from the rows of ``m`` (``h0``,
``h2``); ``h0_ext`` is reconstructed when covered (``ext_ok``), else ``None``.

Not certified: that an assumption holds for a population (a moment-level
``discrepancy`` or a measured value from ``JointMoments.from_samples`` is the
only allowance; otherwise budgets list it ``unbounded``) and feasibility of
``theta``. JIT and autodiff work through ``__call__`` in ``theta`` and the
reference; ``record()`` is eager bookkeeping.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from . import _factor as _f
from . import _project as _p
from ._refine import record as _record, refine as _refine
from ._factor import VARS
from .errors import AssumptionRecord, ErrorTerm
from .index import MomentIndex
from .moments import JointMoments, PopulationSamples

_REST = {v: tuple(u for u in VARS if u != v) for v in VARS}


class Factorisation(eqx.Module):
    """Static partition of ``VARS`` into independent groups, with a name.

    ``groups``: disjoint tuples covering ``VARS``. Declares a product measure
    over the groups (dimensionless); nothing verifies it, and its discrepancy
    is an input of the budget. ``[extension]``.
    """

    LABEL: ClassVar[str] = "[extension] product factorisation"
    groups: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    name: str = eqx.field(static=True, default="factorisation")


class Closure(eqx.Module):
    """Fixed marginal of one group. ``kind`` in ``CLOSURE_KINDS``.

    ``hyper``: ``()`` for ``uniform_mu``; ``(mean, sigma)`` rad/m^2 for
    ``gaussian_depth``; ``(kappa (K,),)`` rad/m^2 cumulants for
    ``cumulant_depth``; one raw value per group variable for ``delta``;
    ``(table,)`` over the group's nonconstant entries for ``fixed_table``
    (as without symmetries; a symmetry ignores the entries it sets to zero).
    The symmetry kinds ``pitch_symmetric`` and ``field_reversal_symmetric``
    take the empty group ``()`` and no hyper: they constrain the joint
    measure and drop rows without closing a marginal. ``hyper`` are leaves.
    A declared marginal or symmetry; nothing certifies it holds.
    """

    LABEL: ClassVar[str] = "[extension] closed marginal"
    group: tuple[str, ...] = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    hyper: tuple = ()


class Parameters(eqx.Module):
    """Free parameters: one table per free group (complex when it carries
    ``e^{2i phi}``), one packed hyper array per fitted closure, softmax logits
    for a nodal map, and an optional ``log_amplitude`` (not part of ``m``).

    Tables dimensionless, ``hyper`` in the closure's units, ``log_amplitude``
    the log source column. Not checked for feasibility.
    """

    LABEL: ClassVar[str] = "extra eq: finite fit model"
    log_amplitude: jax.Array | None = None
    tables: tuple = ()
    hyper: tuple = ()
    logits: jax.Array | None = None


class ParameterMap(eqx.Module):
    """``theta -> JointMoments`` as a product of static gathers.

    Build with :meth:`build` or a named constructor. ``tables_spec[g]`` lists
    the nonconstant projections of every row of ``m`` onto group ``g``;
    ``gathers[g]`` indexes ``[1, *table_g]`` for every retained row in
    ``(h0, h0_ext, h2)`` order (``-1``: dropped by a zero closed factor or
    not covered); ``keep`` is the static zero-row mask; ``ext_ok`` says
    whether ``m0_ext`` is reconstructed. ``free_hyper`` names the closures
    whose hyper-parameters come from ``theta.hyper``; ``nodes`` makes the
    map nodal (``theta.logits``); ``symmetrised`` holds the trims of fixed
    tables under a symmetry (``_factor``). Only ``uniform_mu`` and the symmetry
    closures drop rows statically; value-dependent zeros of other closures
    are kept. Complex free tables keep only the real part of entries
    without ``e^{2i phi}``.

    Moments are dimensionless in ``z``; depth closures take rad/m^2 and
    convert with the ``reference`` of ``__call__``. Assumes its factorisation
    and closures and records them in every ``JointMoments`` it returns; not
    certified: that the population satisfies them (``discrepancy`` is an
    input or measured on supplied samples) or feasibility.
    """

    index: MomentIndex = eqx.field(static=True)
    factorisation: Factorisation = eqx.field(static=True)
    closures: tuple[Closure, ...]
    tables_spec: tuple = eqx.field(static=True)
    gathers: tuple = eqx.field(static=True)
    discrepancy: ErrorTerm | None = None
    name: str = eqx.field(static=True, default="")
    LABEL: str = eqx.field(static=True, default="[extension]")
    keep: tuple = eqx.field(static=True, default=())
    ext_ok: bool = eqx.field(static=True, default=True)
    free_hyper: tuple[int, ...] = eqx.field(static=True, default=())
    nodes: PopulationSamples | None = None
    symmetrised: tuple = eqx.field(static=True, default=())

    @classmethod
    def build(
        cls,
        index,
        factorisation,
        closures=(),
        *,
        discrepancy=None,
        name=None,
        label="[extension]",
        free_hyper=(),
        nodes=None,
    ):
        """Validate the partition and closures and precompute the static tables."""
        if not isinstance(index, MomentIndex):
            raise ValueError("index must be a MomentIndex")
        groups = _f.canonical_partition(factorisation.groups)
        closures = tuple(closures)
        free_hyper = tuple(int(i) for i in free_hyper)
        spec, gathers, keep, ext_ok, trims = _p.prepare(
            index, groups, closures, free_hyper, nodes, discrepancy, ErrorTerm
        )
        factorisation = Factorisation(groups, factorisation.name)
        name = name or factorisation.name
        args = (closures, spec, gathers, discrepancy, name, label, keep, ext_ok)
        return cls(index, factorisation, *args, free_hyper, nodes, trims)

    def _closure_index(self, g):
        group = self.factorisation.groups[g]
        match = [i for i, c in enumerate(self.closures) if c.group == group]
        return match[0] if match else None

    def free_groups(self) -> tuple[int, ...]:
        """Indices of groups with a free (unclosed, nonempty) table."""
        specs = enumerate(self.tables_spec)
        return tuple(g for g, spec in specs if spec and self._closure_index(g) is None)

    def _complex_mask(self, g):
        group = self.factorisation.groups[g]
        return tuple(_f.entry_is_complex(group, p) for p in self.tables_spec[g])

    def _n_hyper(self, i):
        return int(_f.pack_hyper(self.closures[i]).shape[0])

    def n_free(self) -> int:
        """Real free parameters: table entries (two per complex entry), fitted
        hyper-parameters and nodal logits; the unit moment is excluded."""
        if self.nodes is not None:
            return int(self.nodes.size)
        free = self.free_groups()
        count = sum(len(self.tables_spec[g]) + sum(self._complex_mask(g)) for g in free)
        return count + sum(self._n_hyper(i) for i in self.free_hyper)

    def labels(self) -> tuple[str, ...]:
        """Names of the ``n_free`` real parameters in :meth:`flatten` order."""
        return _p.labels(self)

    def is_affine(self) -> bool:
        """True iff ``m`` is affine in :meth:`flatten`: at most one free group,
        no fitted hyper-parameters, not nodal."""
        return (
            len(self.free_groups()) <= 1 and not self.free_hyper and self.nodes is None
        )

    def flatten(self, theta: Parameters) -> jax.Array:
        """Real ``(n_free,)`` vector: per free group all real parts then the
        imaginary parts of complex entries; then hyper; then logits."""
        _p.check_theta(self, theta, Parameters)
        return _p.flatten(self, theta)

    def unflatten(self, vector) -> Parameters:
        """Inverse of :meth:`flatten` (``log_amplitude`` is left ``None``)."""
        vector = jnp.asarray(vector)
        if vector.shape != (self.n_free(),):
            raise ValueError(f"vector must have shape ({self.n_free()},)")
        if self.nodes is not None:
            return Parameters(logits=vector)
        tables, hyper = _p.unflatten(self, vector)
        return Parameters(tables=tables, hyper=hyper)

    def _group_tables(self, theta, reference):
        _p.check_theta(self, theta, Parameters)
        return _p.group_tables(self, theta, reference)

    def _assemble(self, m_all, reference):
        n0, n_ext = self.index.n0, len(self.index.h0_ext)
        m0_ext = m_all[n0 : n0 + n_ext].real if self.ext_ok else None
        m0, m2 = m_all[:n0].real, m_all[n0 + n_ext :].astype(complex)
        record, delta = (self.record(),), self.discrepancy
        return JointMoments(self.index, m0, m2, reference, m0_ext, record, delta)

    def __call__(self, theta: Parameters, reference) -> JointMoments:
        """Moments of ``theta`` at ``reference``; ``m0_ext`` when covered."""
        if self.nodes is not None:
            _p.check_theta(self, theta, Parameters)
            return self._assemble(_p.nodal_moments(self, theta, reference), reference)
        m_all = jnp.asarray(self.keep, dtype=float).astype(complex)
        for gather, table in zip(self.gathers, self._group_tables(theta, reference)):
            m_all = m_all * table[jnp.asarray(gather)]
        return self._assemble(m_all, reference)

    def affine_pieces(self, *, reference=None) -> tuple[jax.Array, jax.Array]:
        """``(P, c)`` with ``m = P @ flatten(theta) + c``; valid iff ``is_affine()``.
        ``reference`` is required when a closure converts rad/m^2 (depth or
        delta closures)."""
        return _p.affine_pieces(self, reference)

    def project(self, full: JointMoments, *, amplitude=None) -> Parameters:
        """Group marginals read off a full tensor (``m0_ext`` needed for depth
        marginals); fitted hyper-parameters from the depth moments. Raises
        ``ValueError`` for a nodal map or a marginal outside the index."""
        return _p.project(self, full, amplitude, Parameters)

    def constraint_residual(self, full: JointMoments) -> jax.Array:
        """Signed ``m_full - m(project(full))`` over ``m`` (``(n_real,)``)."""
        return full.to_vector() - self(self.project(full), full.reference).to_vector()

    def assume(self, *others) -> "ParameterMap":
        """Common refinement of the partitions; closures are merged and must
        agree on any shared group (``ValueError`` on conflict, on a split
        closed group, or on a nodal map)."""
        maps = (self, *others)
        groups, closures, free_hyper, discrepancy = _refine(maps)
        return ParameterMap.build(
            self.index,
            Factorisation(groups, "+".join(m.name for m in maps)),
            closures,
            discrepancy=discrepancy,
            label="; ".join(dict.fromkeys(m.LABEL for m in maps)),
            free_hyper=free_hyper,
        )

    def record(self) -> AssumptionRecord:
        """Hashable record for provenance, independent of ``theta``. A fitted
        closure is recorded with its kind suffixed ``" (fitted)"`` and NaN
        placeholders for its hyper-parameters (the values live in ``theta``;
        ``FitResult`` provenance notes quote them); traced constructor hyper
        values are NaN too. A symmetrised ``fixed_table`` keeps its full
        table and its kind names the dropped entries."""
        return _record(self, AssumptionRecord)


# -- named constructors ------------------------------------------------------------


def no_assumption(index) -> ParameterMap:
    """One group: every retained moment is free (``eq: explicit joint moments``). Declares the factorisation; nothing verifies it."""
    factorisation = Factorisation((VARS,), "no_assumption")
    return ParameterMap.build(index, factorisation, label="eq: explicit joint moments")


def independent_screen(index, *, screen=None) -> ParameterMap:
    """``{depth} | rest`` (``eq: independent screen moments``); ``screen`` is an
    optional ``Closure`` on ``('depth',)``, else the screen moments are free. Declares the factorisation; nothing verifies it.
    """
    closures = ()
    if screen is not None:
        if not isinstance(screen, Closure) or tuple(screen.group) != ("depth",):
            raise ValueError("screen must be a Closure on the group ('depth',)")
        closures = (screen,)
    factorisation = Factorisation((_REST["depth"], ("depth",)), "independent_screen")
    return ParameterMap.build(
        index, factorisation, closures, label="eq: independent screen moments"
    )


def gaussian_screen(index, mean, sigma, *, fit_hyper=False) -> ParameterMap:
    """Independent Gaussian screen on the Taylor route: depth moments from
    ``(mean, sigma)`` in rad/m^2 (``eq: independent gaussian screen``). Declares independence and Gaussianity; nothing verifies either.
    """
    closure = Closure(
        ("depth",), "gaussian_depth", (jnp.asarray(mean), jnp.asarray(sigma))
    )
    factorisation = Factorisation((_REST["depth"], ("depth",)), "gaussian_screen")
    return ParameterMap.build(
        index,
        factorisation,
        (closure,),
        label="eq: independent gaussian screen",
        free_hyper=(0,) if fit_hyper else (),
    )


def field_independent(index) -> ParameterMap:
    """``{B} | rest`` ``[extension]``: ``<P_l P_k e^{ih phi} z_gamma^r z_depth^b> <z_B^s>``. Declares the factorisation; nothing verifies it."""
    factorisation = Factorisation((_REST["B"], ("B",)), "field_independent")
    return ParameterMap.build(index, factorisation)


def azimuth_separable(index) -> ParameterMap:
    """``{phi} | rest``: ``M2 = m_phi M0_ext`` with ``m_phi = <e^{2i phi}>``
    (``detail-eq: angular factorisation error``; weaker than independence). Declares the separability; nothing verifies it.
    """
    factorisation = Factorisation((_REST["phi"], ("phi",)), "azimuth_separable")
    return ParameterMap.build(
        index, factorisation, label="detail-eq: angular factorisation error"
    )


def isotropic_pitch(index) -> ParameterMap:
    """``{mu} | rest`` with ``uniform_mu``: rows with ``l >= 1`` are zero
    (``extra eq: pitch angle expansion``). Declares isotropic pitch; nothing verifies it.
    """
    factorisation = Factorisation((_REST["mu"], ("mu",)), "isotropic_pitch")
    closure = Closure(("mu",), "uniform_mu")
    return ParameterMap.build(
        index, factorisation, (closure,), label="extra eq: pitch angle expansion"
    )


def fully_independent(index) -> ParameterMap:
    """Six singletons ``[extension]``: ``N_gamma + N_B + max_b + L_mu + L_eta + 2``
    free parameters with ``(N_gamma, N_B, max_b) = truncation.caps()`` (``2N +
    max_b + ...`` uncapped). Declares the factorisation; nothing verifies it."""
    factorisation = Factorisation(tuple((v,) for v in VARS), "fully_independent")
    return ParameterMap.build(index, factorisation)


def pitch_symmetric(index) -> ParameterMap:
    """Population invariant under ``mu -> -mu`` at fixed other variables
    ``[extension]``: every odd-``l`` row is zero and dropped (no independence
    is assumed; one group). Declares the symmetry; nothing verifies it."""
    return _symmetric(index, "pitch_symmetric")


def field_reversal_symmetric(index) -> ParameterMap:
    """Invariant under ``B -> -B`` with electrons fixed, ``(mu, eta, phi) ->
    (-mu, -eta, phi + pi)`` at fixed ``(gamma, B, depth)`` ``[extension]``:
    odd-``l + k`` rows vanish (odd ``l`` or ``k`` when the partition separates
    ``mu`` from ``eta``), so the finite ``V`` response is exactly zero.
    A local reversal; a global one would also flip the Faraday depth, which
    this does not model. Declares the symmetry; nothing verifies it."""
    return _symmetric(index, "field_reversal_symmetric")


def _symmetric(index, kind):
    return ParameterMap.build(index, Factorisation((VARS,), kind), (Closure((), kind),))


def nodal(index, nodes: PopulationSamples) -> ParameterMap:
    """Nonnegative weights ``softmax(logits)`` on fixed nodes
    (``extra eq: nodal distribution``); discretisation error is an input. Nothing certifies that the nodes resolve the population.
    """
    if not isinstance(nodes, PopulationSamples):
        raise ValueError("nodes must be a PopulationSamples")
    factorisation = Factorisation((VARS,), "nodal")
    return ParameterMap.build(
        index, factorisation, nodes=nodes, label="extra eq: nodal distribution"
    )


__all__ = [
    "VARS",
    "Factorisation",
    "Closure",
    "Parameters",
    "ParameterMap",
    "no_assumption",
    "independent_screen",
    "gaussian_screen",
    "field_independent",
    "azimuth_separable",
    "isotropic_pitch",
    "fully_independent",
    "pitch_symmetric",
    "field_reversal_symmetric",
    "nodal",
]
