"""Projection, affine pieces and record helpers for ``ParameterMap``.

Private helper of :mod:`syncmoments.model.assumptions`, split out to keep that
file under 400 lines. Functions take the map as their first argument and
use only its public fields and methods. Nothing here certifies feasibility
of a projected parameter set; ``project`` reads marginals off a tensor that
the caller vouches for.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import _factor as _f


def marginal(pm, full, group, projection):
    """Group marginal ``<prod_v f_v>`` read off the full tensor.

    The marginal is the retained row whose exponents vanish outside
    ``group``; rows with ``b >= 1`` and ``h = 0`` come from ``m0_ext``.
    Raises ``ValueError`` when the row is not retained.
    """
    row = [0] * 6
    for v, e in zip(group, projection):
        row[_f.ROW_POSITION[v]] = e
    h, l, k, r, s, b = row
    label = _f.entry_label(group, projection)
    index = pm.index
    try:
        if h == 0 and b >= 1:
            if full.m0_ext is None:
                raise ValueError("m0_ext is absent on the full tensor")
            value = full.m0_ext[index.ext_position(l, k, r, s, b)]
        elif h == 0:
            value = full.m0[index.position(0, l, k, r, s, 0)]
        else:
            value = full.m2[index.position(2, l, k, r, s, b) - index.n0]
    except ValueError as error:
        raise ValueError(
            f"marginal {label} is not a retained moment: {error}"
        ) from None
    return value if _f.entry_is_complex(group, projection) else jnp.real(value)


def project_hyper(pm, full, closure):
    """Fitted depth hyper-parameters (rad/m^2) from the depth marginals.

    Gaussian: ``(mean, sigma)`` from ``b = 1, 2``; cumulant: ``K`` cumulants
    from the raw moments through ``b = K`` (moment-to-cumulant recursion).
    A negative roundoff variance is clamped at zero.
    """
    K = 2 if closure.kind == "gaussian_depth" else int(jnp.shape(closure.hyper[0])[0])
    if pm.index.truncation.max_b() < K or full.m0_ext is None:
        raise ValueError(
            f"projecting {closure.kind} hyper needs depth moments through b={K}"
        )
    moments = [jnp.asarray(1.0)]
    moments += [marginal(pm, full, ("depth",), (b,)) for b in range(1, K + 1)]
    kappa = []
    for n in range(1, K + 1):
        tail = sum(
            math.comb(n - 1, j - 1) * kappa[j - 1] * moments[n - j] for j in range(1, n)
        )
        kappa.append(moments[n] - tail)
    ref, s = full.reference.depth_ref, full.reference.scales[2]
    if closure.kind == "gaussian_depth":
        sigma = s * jnp.sqrt(jnp.maximum(kappa[1], 0.0))
        return jnp.stack([ref + s * kappa[0], sigma])
    return jnp.stack(
        [ref + s * kappa[0]] + [s ** (n + 1) * kappa[n] for n in range(1, K)]
    )


def affine_pieces(pm, reference):
    """``(P, c)`` with ``m = P @ flatten(theta) + c`` for an affine map.

    ``ValueError`` when the map is not affine or when a depth or delta
    closure needs a reference that was not supplied.
    """
    if not pm.is_affine():
        raise ValueError(
            "affine_pieces needs an affine map (one free group, no fitted hyper)"
        )
    needs_reference = any(c.kind not in _f.REFERENCE_FREE for c in pm.closures)
    if reference is None and needs_reference:
        raise ValueError("affine_pieces needs reference= for depth or delta closures")
    free = pm.free_groups()
    n_free, n_all = pm.n_free(), len(pm.keep)
    const = jnp.asarray(pm.keep, dtype=float).astype(complex)
    tables = pm._group_tables(pm.unflatten(jnp.zeros(n_free)), reference)
    for g, (gather, table) in enumerate(zip(pm.gathers, tables)):
        if g not in free:
            const = const * table[jnp.asarray(gather)]
    if free:
        g = free[0]
        n = len(pm.tables_spec[g])
        mask = np.asarray(pm._complex_mask(g), dtype=bool)
        col_re = np.concatenate([[-1], np.arange(n)])
        col_im = np.concatenate([[-1], np.where(mask, n + np.cumsum(mask) - 1, -1)])
        fi = np.asarray(pm.gathers[g])
        safe = np.maximum(fi, 0)
        re = jnp.asarray(np.where(fi >= 0, col_re[safe], -1))
        im = jnp.asarray(np.where(fi >= 0, col_im[safe], -1))
        P = jax.nn.one_hot(re, n_free, dtype=float) * const[:, None]
        P = P + jax.nn.one_hot(im, n_free, dtype=float) * (1j * const)[:, None]
        c = jnp.where(jnp.asarray(fi) == 0, const, 0.0)
    else:
        P, c = jnp.zeros((n_all, 0), dtype=complex), const
    n0, n_ext = pm.index.n0, len(pm.index.h0_ext)
    h0, h2 = slice(0, n0), slice(n0 + n_ext, n_all)
    P_real = jnp.concatenate([P[h0].real, P[h2].real, P[h2].imag])
    c_real = jnp.concatenate([c[h0].real, c[h2].real, c[h2].imag])
    return P_real, c_real


def same_closure(a, b):
    """True when two closures have the same kind and concrete hyper values."""
    if a.kind != b.kind or len(a.hyper) != len(b.hyper):
        return False
    for x, y in zip(a.hyper, b.hyper):
        if isinstance(x, jax.core.Tracer) or isinstance(y, jax.core.Tracer):
            if x is not y:
                return False
        elif np.shape(x) != np.shape(y) or not np.allclose(
            np.asarray(x), np.asarray(y)
        ):
            return False
    return True


# One shared NaN object: tuple equality tests identity first, so records with
# placeholders compare (and hash) equal across thetas, traces and cond branches.
PLACEHOLDER_HYPER = float("nan")
FITTED_SUFFIX = " (fitted)"


def record_hyper(closures, fitted=()):
    """Record values of the closures' hyper-parameters, as floats.

    Complex values are split into (re, im). The closures listed in
    ``fitted`` (fitted from ``theta``) and traced hyper values are recorded
    as the shared ``PLACEHOLDER_HYPER`` NaN, one per element, so the record
    does not depend on ``theta`` or on tracing.
    """
    out = []
    for i, closure in enumerate(closures):
        for h in closure.hyper:
            if i in fitted or isinstance(h, jax.core.Tracer):
                per_element = 2 if jnp.iscomplexobj(h) else 1
                out.extend(
                    [PLACEHOLDER_HYPER] * (per_element * math.prod(jnp.shape(h)))
                )
                continue
            values = np.ravel(np.asarray(h))
            for v in values:
                out.append(float(np.real(v)))
                if np.iscomplexobj(values):
                    out.append(float(np.imag(v)))
    return tuple(out)


def record_kinds(pm):
    """``closure_kind`` of the record: kinds joined by ``+``, fitted ones and
    symmetrised fixed tables (with the entries they drop) suffixed."""
    kinds = "+".join(
        c.kind
        + (FITTED_SUFFIX if i in pm.free_hyper else "")
        + (_trim(pm, i) or (0, (), ""))[2]
        for i, c in enumerate(pm.closures)
    )
    if not kinds:
        kinds = "nodal" if pm.nodes is not None else "none"
    return kinds


def _trim(pm, i):
    return pm.symmetrised[i] if pm.symmetrised else None


def fitted_hyper_note(pm, theta):
    """Provenance note with the concrete fitted hyper-parameters of ``theta``.

    ``None`` when the map fits no hyper-parameter. Eager (concrete values).
    """
    if not pm.free_hyper:
        return None
    parts = []
    for k, i in enumerate(pm.free_hyper):
        values = np.ravel(np.asarray(theta.hyper[k], dtype=float))
        names = _f.hyper_labels(pm.closures[i])
        parts.extend(f"{n}={v:.10g}" for n, v in zip(names, values))
    return (
        f"fitted hyper-parameters of '{pm.name}' (the AssumptionRecord holds NaN "
        f"placeholders for them): {', '.join(parts)}"
    )


def project(pm, full, amplitude, Parameters):
    """``Parameters`` of the group marginals of ``full``; see ``ParameterMap.project``."""
    if pm.nodes is not None:
        raise ValueError(
            "project is not defined for a nodal map; fit its weights instead"
        )
    if full.index != pm.index:
        raise ValueError("full tensor must use the map's index")
    tables = []
    for g in pm.free_groups():
        group, spec = pm.factorisation.groups[g], pm.tables_spec[g]
        table = jnp.stack([marginal(pm, full, group, p) for p in spec])
        tables.append(table.astype(complex) if any(pm._complex_mask(g)) else table)
    hyper = tuple(project_hyper(pm, full, pm.closures[i]) for i in pm.free_hyper)
    log_amplitude = None if amplitude is None else jnp.log(jnp.asarray(amplitude))
    return Parameters(log_amplitude=log_amplitude, tables=tuple(tables), hyper=hyper)


def labels(pm):
    """Names of the free parameters; see ``ParameterMap.labels``."""
    if pm.nodes is not None:
        return tuple(f"logit[{j}]" for j in range(pm.nodes.size))
    out = []
    for g in pm.free_groups():
        group, spec = pm.factorisation.groups[g], pm.tables_spec[g]
        mask = pm._complex_mask(g)
        names = [_f.entry_label(group, p) for p in spec]
        out.extend(("Re " if c else "") + n for n, c in zip(names, mask))
        out.extend("Im " + n for n, c in zip(names, mask) if c)
    for i in pm.free_hyper:
        out.extend(_f.hyper_labels(pm.closures[i]))
    return tuple(out)


def prepare(index, groups, closures, free_hyper, nodes, discrepancy, ErrorTerm):
    """Static checks of ``ParameterMap.build`` plus the table construction.

    Returns ``(tables_spec, gathers, keep, ext_ok, symmetrised)``; ``ValueError``
    on any invalid closure, hyper selection, nodal configuration, discrepancy
    shape or fixed-table length (the full table under a symmetry, see
    :func:`_factor.symmetrised_tables`; the reduced one is refused on every route).
    """
    for closure in closures:
        _f.validate_closure(closure, groups, index)
    if len({_f.closure_key(c) for c in closures}) != len(closures):
        raise ValueError("at most one closure per group (and per symmetry kind)")
    fittable = ("gaussian_depth", "cumulant_depth")
    if any(
        i < 0 or i >= len(closures) or closures[i].kind not in fittable
        for i in free_hyper
    ):
        raise ValueError(
            "free_hyper may only name gaussian_depth/cumulant_depth closures"
        )
    if nodes is not None and (groups != (_f.VARS,) or closures):
        raise ValueError("a nodal map has the single group VARS and no closures")
    if discrepancy is not None:
        if not isinstance(discrepancy, ErrorTerm):
            raise ValueError("discrepancy must be an ErrorTerm over m")
        value = discrepancy.value
        if value is not None and jnp.shape(value) != (index.n_real,):
            raise ValueError(f"discrepancy.value must have shape ({index.n_real},)")
    spec, gathers, keep, ext_ok = _f.build_tables(index, groups, closures)
    trims = _f.symmetrised_tables(index, groups, closures, spec)
    for closure, trim in zip(closures, trims):
        if closure.kind == "fixed_table":
            n = len(spec[groups.index(closure.group)]) if trim is None else trim[0]
            if jnp.shape(closure.hyper[0]) != (n,):
                note = "" if trim is None else trim[2]
                raise ValueError(
                    f"fixed_table on {closure.group} needs shape ({n},){note}"
                )
    # nodal rows (m0_ext included) come from JointMoments.from_samples
    return spec, gathers, keep, ext_ok or nodes is not None, trims


def nodal_moments(pm, theta, reference):
    """All retained rows ``(h0, h0_ext, h2)`` of the softmax-weighted nodes."""
    from .moments import JointMoments, PopulationSamples

    weights = jax.nn.softmax(jnp.asarray(theta.logits, dtype=float))
    n = pm.nodes
    samples = PopulationSamples(
        n.gamma, n.B, n.mu, n.eta, n.phi, n.depth, weights=weights
    )
    joint = JointMoments.from_samples(samples, pm.index, reference)
    ext = jnp.zeros(0) if joint.m0_ext is None else joint.m0_ext
    return jnp.concatenate([joint.m0, ext, joint.m2]).astype(complex)


def check_theta(pm, theta, Parameters):
    """Shape and type checks of ``theta`` against the map (trace time)."""
    if not isinstance(theta, Parameters):
        raise ValueError("theta must be a Parameters")
    if pm.nodes is not None:
        if theta.logits is None or jnp.shape(theta.logits) != (pm.nodes.size,):
            raise ValueError(f"nodal map needs logits of shape ({pm.nodes.size},)")
        return
    free = pm.free_groups()
    if len(theta.tables) != len(free):
        raise ValueError(f"expected {len(free)} tables, got {len(theta.tables)}")
    for j, g in enumerate(free):
        n = len(pm.tables_spec[g])
        if jnp.shape(theta.tables[j]) != (n,):
            raise ValueError(f"table {j} must have shape ({n},)")
        if jnp.iscomplexobj(theta.tables[j]) and not any(pm._complex_mask(g)):
            raise ValueError(f"table {j} must be real (no e^(2i phi) entries)")
    if len(theta.hyper) != len(pm.free_hyper):
        raise ValueError(f"expected {len(pm.free_hyper)} hyper arrays")
    for k, i in enumerate(pm.free_hyper):
        if jnp.shape(theta.hyper[k]) != (pm._n_hyper(i),):
            raise ValueError(f"hyper {k} must have shape ({pm._n_hyper(i)},)")


def flatten(pm, theta):
    """Real ``(n_free,)`` vector; see ``ParameterMap.flatten``."""
    if pm.nodes is not None:
        return jnp.asarray(theta.logits, dtype=float)
    parts = []
    for j, g in enumerate(pm.free_groups()):
        table = jnp.asarray(theta.tables[j])
        parts.append(table.real)
        mask = np.asarray(pm._complex_mask(g), dtype=bool)
        if mask.any():  # static integer indices: jax.jit rejects a boolean index
            parts.append(table.imag[np.flatnonzero(mask)])
    parts.extend(jnp.asarray(h, dtype=float) for h in theta.hyper)
    return jnp.concatenate(parts) if parts else jnp.zeros(0)


def unflatten(pm, vector):
    """``(tables, hyper)`` from a flat vector; see ``ParameterMap.unflatten``."""
    tables, start = [], 0
    for g in pm.free_groups():
        n, mask = len(pm.tables_spec[g]), pm._complex_mask(g)
        table = vector[start : start + n]
        start += n
        if any(mask):
            idx = np.flatnonzero(mask)
            imag = vector[start : start + len(idx)]
            table = table.astype(complex).at[idx].add(1j * imag)
            start += len(idx)
        tables.append(table)
    hyper = []
    for i in pm.free_hyper:
        hyper.append(vector[start : start + pm._n_hyper(i)])
        start += pm._n_hyper(i)
    return tuple(tables), tuple(hyper)


def group_tables(pm, theta, reference):
    """Evaluated ``[1, *table_g]`` for every group (free from ``theta``,
    closed from the closure; fitted hyper from ``theta.hyper``)."""
    tables, j = [], 0
    for g, (group, spec) in enumerate(zip(pm.factorisation.groups, pm.tables_spec)):
        c = pm._closure_index(g)
        if c is None:
            if spec:
                table = jnp.asarray(theta.tables[j])
                j += 1
                if any(pm._complex_mask(g)):
                    mask = jnp.asarray(pm._complex_mask(g))
                    table = jnp.where(
                        mask, table.astype(complex), table.real.astype(complex)
                    )
            else:
                table = jnp.zeros(0)
        else:
            closure = pm.closures[c]
            hyper = closure.hyper
            if c in pm.free_hyper:
                hyper = _f.unpack_hyper(closure, theta.hyper[pm.free_hyper.index(c)])
            if _trim(pm, c) is not None:  # symmetrised fixed_table: kept entries
                hyper = (jnp.asarray(hyper[0])[np.asarray(_trim(pm, c)[1], int)],)
            table = _f.closure_table(closure, group, spec, reference, hyper)
        tables.append(jnp.concatenate([jnp.ones(1, dtype=table.dtype), table]))
    return tables


__all__ = [
    "marginal",
    "project_hyper",
    "affine_pieces",
    "project",
    "labels",
    "same_closure",
    "record_hyper",
    "record_kinds",
    "fitted_hyper_note",
    "PLACEHOLDER_HYPER",
    "FITTED_SUFFIX",
    "prepare",
    "nodal_moments",
    "check_theta",
    "flatten",
    "unflatten",
    "group_tables",
]
