"""Static table construction and closure evaluation for ``ParameterMap``.

Private helper of :mod:`synchro.model.assumptions`; nothing here is public API.

Rows are 6-tuples ``(h, l, k, r, s, b)`` with ``h in {0, 2}`` the azimuthal
weight. The variables ``VARS = (gamma, B, mu, eta, phi, depth)`` carry the
exponents ``(r, s, l, k, h, b)`` respectively, and every retained moment is
the average of the product monomial
``z_gamma^r z_B^s P_l(mu) P_k(eta) e^{i h phi} z_depth^b``
(``eq: explicit joint moments`` in the dimensionless displacements of
``eq: local response remainder``). A factorisation (partition of ``VARS``)
turns each row into a product of group marginals; this module enumerates the
distinct nonconstant projections of the rows of ``m`` (``h0`` and ``h2``)
onto every group and the static gathers that rebuild every retained row,
including the ``h0_ext`` rows when their projections are covered.

Closure tables are evaluated in the same displacement coordinates: depth
closures take their hyper-parameters in rad/m^2 and are converted with the
reference ``depth_ref`` and ``s_depth``; a ``delta`` closure takes raw values
(``gamma``, Gauss, cosines, radians, rad/m^2). Nothing here certifies that a
closure describes the population; it only evaluates the closure's moments.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp

from ..cumulants import raw_moments_from_cumulants

VARS = ("gamma", "B", "mu", "eta", "phi", "depth")
ROW_POSITION = {"phi": 0, "mu": 1, "eta": 2, "gamma": 3, "B": 4, "depth": 5}
CLOSURE_KINDS = (
    "uniform_mu",
    "gaussian_depth",
    "cumulant_depth",
    "delta",
    "fixed_table",
)


def project_row(row, group):
    """Exponents of ``row`` on the variables of ``group`` (group order)."""
    return tuple(int(row[ROW_POSITION[v]]) for v in group)


def canonical_partition(groups):
    """Validate a partition of ``VARS`` and return it in canonical order.

    Variables inside a group follow ``VARS``; groups are ordered by their
    first variable. Raises ``ValueError`` for anything that is not a
    partition of ``VARS``.
    """
    if not isinstance(groups, (tuple, list)) or not groups:
        raise ValueError("groups must be a nonempty tuple of variable tuples")
    seen = []
    ordered = []
    for group in groups:
        if not isinstance(group, (tuple, list)) or not group:
            raise ValueError("every group must be a nonempty tuple of variable names")
        for v in group:
            if v not in VARS:
                raise ValueError(f"unknown variable {v!r}; variables are {VARS}")
            if v in seen:
                raise ValueError(f"variable {v!r} appears in more than one group")
            seen.append(v)
        ordered.append(tuple(sorted(group, key=VARS.index)))
    if len(seen) != len(VARS):
        missing = tuple(v for v in VARS if v not in seen)
        raise ValueError(f"groups must cover every variable; missing {missing}")
    return tuple(sorted(ordered, key=lambda g: VARS.index(g[0])))


def retained_rows(index):
    """``(h0, h0_ext, h2)`` rows as 6-tuples ``(h, l, k, r, s, b)``."""
    h0 = tuple((0, *row) for row in index.h0)
    ext = tuple((0, *row) for row in index.h0_ext)
    h2 = tuple((2, *row) for row in index.h2)
    return h0, ext, h2


def build_tables(index, groups, closures):
    """Static tables, gathers and keep mask for a factorisation.

    Returns ``(tables_spec, gathers, keep, ext_ok)``: per group the tuple of
    distinct nonconstant projections (sorted lexicographically), per group
    the gather index for every retained row in ``(h0, h0_ext, h2)`` order
    (``0`` = constant entry, ``j >= 1`` = table entry ``j-1``, ``-1`` =
    dropped or not covered), the static keep mask (rows whose closed factor
    is exactly zero, ``l > 0`` under ``uniform_mu``, are dropped before the
    tables are formed) and whether every kept ``h0_ext`` row is covered.
    """
    h0, ext, h2 = retained_rows(index)
    all_rows = h0 + ext + h2
    n0, n_ext = len(h0), len(ext)
    keep = [True] * len(all_rows)
    for closure in closures:
        if closure.kind == "uniform_mu":
            for i, row in enumerate(all_rows):
                if row[1] > 0:
                    keep[i] = False
    in_m = [i for i in range(len(all_rows)) if i < n0 or i >= n0 + n_ext]
    specs, gathers = [], []
    for group in groups:
        projections = sorted({project_row(all_rows[i], group) for i in in_m if keep[i]})
        spec = tuple(p for p in projections if any(p))
        lookup = {p: j + 1 for j, p in enumerate(spec)}
        gather = []
        for i, row in enumerate(all_rows):
            p = project_row(row, group)
            if not keep[i]:
                gather.append(-1)
            elif not any(p):
                gather.append(0)
            else:
                gather.append(lookup.get(p, -1))
        specs.append(spec)
        gathers.append(tuple(gather))
    ext_ok = all(
        (not keep[i]) or all(g[i] >= 0 for g in gathers) for i in range(n0, n0 + n_ext)
    )
    return tuple(specs), tuple(gathers), tuple(keep), ext_ok


def entry_is_complex(group, projection):
    """True iff the projection carries the ``e^{2i phi}`` factor."""
    return "phi" in group and projection[group.index("phi")] == 2


def entry_label(group, projection):
    """Printed name of a table entry, e.g. ``"<z_gamma^2 P_1(mu) e^{2i phi}>"``."""
    parts = []
    for v, e in zip(group, projection):
        if e == 0:
            continue
        if v in ("gamma", "B", "depth"):
            parts.append(f"z_{v}" + (f"^{e}" if e > 1 else ""))
        elif v in ("mu", "eta"):
            parts.append(f"P_{e}({v})")
        else:
            parts.append(f"e^{{{e}i phi}}")
    return "<" + " ".join(parts) + ">"


def legendre(x, degree):
    """``P_degree(x)`` by the three-term recurrence (``degree`` static)."""
    x = jnp.asarray(x)
    if degree == 0:
        return jnp.ones_like(x)
    previous, current = jnp.ones_like(x), x
    for n in range(1, degree):
        previous, current = current, ((2 * n + 1) * x * current - n * previous) / (
            n + 1
        )
    return current


def variable_factor(variable, exponent, value, reference):
    """``f_v(value)^exponent`` for one variable at a raw-unit point."""
    value = jnp.asarray(value)
    if variable == "gamma":
        return ((value - reference.gamma0) / reference.scales[0]) ** exponent
    if variable == "B":
        return ((value - reference.B0) / reference.scales[1]) ** exponent
    if variable == "depth":
        return ((value - reference.depth_ref) / reference.scales[2]) ** exponent
    if variable == "phi":
        return jnp.exp(1j * exponent * value)
    return legendre(value, exponent)


def validate_closure(closure, groups, index):
    """Static checks of a closure against the partition; ``ValueError`` on failure."""
    if closure.kind not in CLOSURE_KINDS:
        raise ValueError(
            f"closure kind must be one of {CLOSURE_KINDS}, got {closure.kind!r}"
        )
    group = tuple(closure.group)
    if group not in groups:
        raise ValueError(
            f"closure group {group} is not a group of the factorisation {groups}"
        )
    n_hyper = len(closure.hyper)
    if closure.kind == "uniform_mu" and (group != ("mu",) or n_hyper != 0):
        raise ValueError(
            "uniform_mu closes the group ('mu',) and takes no hyper-parameters"
        )
    if closure.kind == "gaussian_depth" and (group != ("depth",) or n_hyper != 2):
        raise ValueError(
            "gaussian_depth closes ('depth',) with hyper=(mean, sigma) in rad/m^2"
        )
    if closure.kind == "cumulant_depth":
        if group != ("depth",) or n_hyper != 1:
            raise ValueError(
                "cumulant_depth closes ('depth',) with hyper=(kappa,) in rad/m^2"
            )
        if jnp.ndim(closure.hyper[0]) != 1 or jnp.shape(closure.hyper[0])[0] == 0:
            raise ValueError("cumulant_depth needs a nonempty 1D cumulant array")
    if closure.kind == "delta" and n_hyper != len(group):
        raise ValueError(
            "delta closure needs one raw-unit value per variable of its group"
        )
    if closure.kind == "fixed_table" and n_hyper != 1:
        raise ValueError(
            "fixed_table closure takes hyper=(table,) over the group's nonconstant entries"
        )


def pack_hyper(closure):
    """Flatten a closure's hyper-parameters into one real 1D array."""
    if not closure.hyper:
        return jnp.zeros(0)
    return jnp.concatenate([jnp.ravel(jnp.asarray(h)).real for h in closure.hyper])


def unpack_hyper(closure, flat):
    """Inverse of :func:`pack_hyper` for the shapes of ``closure.hyper``."""
    out, start = [], 0
    for h in closure.hyper:
        shape = jnp.shape(h)
        size = math.prod(shape)
        out.append(jnp.reshape(flat[start : start + size], shape))
        start += size
    return tuple(out)


def hyper_labels(closure):
    if closure.kind == "gaussian_depth":
        return ("hyper:gaussian_depth:mean", "hyper:gaussian_depth:sigma")
    if closure.kind == "cumulant_depth":
        return tuple(
            f"hyper:cumulant_depth:kappa_{n + 1}"
            for n in range(jnp.shape(closure.hyper[0])[0])
        )
    return tuple(
        f"hyper:{closure.kind}:{i}" for i in range(int(pack_hyper(closure).shape[0]))
    )


def _depth_cumulants(closure, hyper, reference):
    depth_ref, s = reference.depth_ref, reference.scales[2]
    if closure.kind == "gaussian_depth":
        mean, sigma = (jnp.asarray(h) for h in hyper)
        sigma = eqx.error_if(
            sigma,
            ~jnp.isfinite(sigma) | (sigma < 0),
            "gaussian_depth needs finite sigma >= 0",
        )
        return [(mean - depth_ref) / s, (sigma / s) ** 2]
    kappa = jnp.asarray(hyper[0])
    kappa = eqx.error_if(
        kappa, jnp.any(~jnp.isfinite(kappa)), "cumulants must be finite"
    )
    first = [(kappa[0] - depth_ref) / s]
    return first + [kappa[n] / s ** (n + 1) for n in range(1, kappa.shape[0])]


def closure_table(closure, group, spec, reference, hyper):
    """Nonconstant table entries of a closed group, in ``spec`` order.

    ``hyper`` is the tuple of hyper-parameter arrays to use (the closure's own
    or the fitted ones). Returns a 1D array of length ``len(spec)``.
    """
    n = len(spec)
    if closure.kind == "uniform_mu":
        return jnp.zeros(n)
    if closure.kind in ("gaussian_depth", "cumulant_depth"):
        kappa = _depth_cumulants(closure, hyper, reference)
        max_b = max((p[0] for p in spec), default=0)
        moments = raw_moments_from_cumulants(kappa, max_b)
        if n == 0:
            return jnp.zeros(0)
        return jnp.stack([jnp.asarray(moments[p[0]]) for p in spec])
    if closure.kind == "fixed_table":
        table = jnp.asarray(hyper[0])
        if table.shape != (n,):
            raise ValueError(
                f"fixed_table closure on {group} needs a table of shape ({n},)"
            )
        return table
    if n == 0:
        return jnp.zeros(0)
    values = [jnp.asarray(h) for h in hyper]
    entries = []
    for p in spec:
        factor = jnp.asarray(1.0 + 0j) if "phi" in group else jnp.asarray(1.0)
        for v, e, value in zip(group, p, values):
            factor = factor * variable_factor(v, e, value, reference)
        entries.append(factor)
    return jnp.stack(entries)


__all__ = [
    "VARS",
    "ROW_POSITION",
    "CLOSURE_KINDS",
    "project_row",
    "canonical_partition",
    "retained_rows",
    "build_tables",
    "entry_is_complex",
    "entry_label",
    "legendre",
    "variable_factor",
    "validate_closure",
    "pack_hyper",
    "unpack_hyper",
    "hyper_labels",
    "closure_table",
]
