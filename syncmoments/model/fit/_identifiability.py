"""Whitened design assembly for ``identifiability`` (private helper).

Split out of :mod:`syncmoments.model.fit.diagnostics` to keep that file short.
``whitened_design`` forms ``L^-1 R C columns`` over the kept data rows and
``parameter_columns`` forms the columns ``[c | P]`` of a ``ParameterMap``
(exact for an affine map, the Jacobian at ``theta`` otherwise) and
``equilibrate`` rescales design columns to unit norm, so that the
singular-value rank cutoff shared by ``fit_linear``, ``identifiability`` and
the Gauss-Newton Fisher summaries does not depend on the units of the
unknowns (the ``Reference`` scales). Nothing here is public API.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def kept_rows(data, n_data, n_rows):
    """Indices of the data rows that survive ``data.mask`` (``True`` = kept)."""
    mask = getattr(data, "mask", None)
    if mask is None:
        return np.arange(n_data)
    if n_data != n_rows:
        raise ValueError(
            "a mask applies to the 4 n_ch Stokes rows; it cannot be combined "
            "with a response of a different length"
        )
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), (n_rows // 4, 4))
    return np.flatnonzero(mask.ravel())


def whitened_design(data, C, columns):
    """``L^-1 R C columns`` over the kept data rows, ``(n_kept, n_u)``.

    Uses ``data.design`` (``StokesData``: response applied, kept rows
    selected) when present; otherwise applies ``data.response`` and
    ``data.mask`` (``True`` = kept) directly.
    """
    n_rows = int(C.shape[0])
    n_ch = getattr(data, "n_ch", None)
    if n_ch is not None and 4 * int(n_ch) != n_rows:
        raise ValueError(
            f"data has {n_ch} channels but the basis response matrix has "
            f"{n_rows} rows (4 n_ch)"
        )
    _, L_inv = data.whitened()
    columns = C @ columns
    design = getattr(data, "design", None)
    if callable(design):
        G = design(columns)
    else:
        response = getattr(data, "response", None)
        R = jnp.eye(n_rows) if response is None else jnp.asarray(response)
        if R.ndim != 2 or R.shape[1] != n_rows:
            raise ValueError(f"data.response must have shape (n_data, {n_rows})")
        keep = kept_rows(data, R.shape[0], n_rows)
        G = R[jnp.asarray(keep)] @ columns
    L_inv = jnp.asarray(L_inv)
    if G.shape[0] == 0:
        raise ValueError("no data rows survive the mask")
    if L_inv.shape != (G.shape[0], G.shape[0]):
        raise ValueError(
            f"data.whitened() returned L^-1 of shape {L_inv.shape}; expected "
            f"({G.shape[0]}, {G.shape[0]}) after masking"
        )
    return L_inv @ G


def equilibrate(G):
    """``(G D, d)`` with ``D = diag(d)``, ``d_i = 1 / ||G[:, i]||`` (``1`` for a zero column).

    Column equilibration: every nonzero column of ``G D`` has unit norm, so
    its singular values and right singular vectors are invariant under a
    positive rescaling of the unknowns. Only exactly zero (or non-finite)
    column norms keep the weight 1; a column that is small only because of
    the coordinate units is restored to unit norm. Eager (``d`` is concrete).
    """
    G = jnp.asarray(G)
    norms = np.linalg.norm(np.asarray(G), axis=0)
    ok = np.isfinite(norms) & (norms > 0.0)
    d = np.where(ok, 1.0 / np.where(ok, norms, 1.0), 1.0)
    d = jnp.asarray(d, dtype=G.dtype)
    return G * d[None, :], d


def parameter_columns(parameter_map, reference, theta, amplitude):
    """``(columns (n_real, n_u), labels, note)`` of the map's free directions."""
    if parameter_map.is_affine():
        P, c = parameter_map.affine_pieces(reference=reference)
        note = "exact: affine map m = P theta + c"
    else:
        if theta is None:
            raise ValueError(
                "identifiability of a nonlinear ParameterMap needs theta= (the "
                "design is the Jacobian of m at that point)"
            )
        v0 = parameter_map.flatten(theta)

        def m_of(v):
            return parameter_map(parameter_map.unflatten(v), reference).to_vector()

        P = jax.jacfwd(m_of)(v0)
        c = m_of(v0) - P @ v0
        note = "linearised: Jacobian of m at the supplied theta (local statement)"
    labels = parameter_map.labels()
    if amplitude:
        columns = jnp.concatenate([c[:, None], P], axis=1)
        labels = ("amplitude",) + tuple(labels)
    else:
        columns = P
    if columns.shape[1] == 0:
        raise ValueError("the parameter map has no free direction to resolve")
    return columns, tuple(labels), note


__all__ = ["kept_rows", "whitened_design", "equilibrate", "parameter_columns"]
