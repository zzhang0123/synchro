"""Printed summary of the SED reconstruction example and the historical table.

``print_summary(payload, historical=None)`` prints the Section 9.5 items of
the T-006 design; with the path of the historical research results JSON
(manuscript ``validation/sed_reconstruction_results.json``, read only) it
prints a side-by-side table, the top singular values, the principal angle
between the retained row spaces and the rotation-invariant norms of
``beta_hat`` per cluster (in-cluster bases differ by a rotation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import subspace_angles

HISTORICAL_KEYS = {
    "grouped_vs_ungrouped_max_coefficient_difference": "grouped_vs_ungrouped_fit_max_coefficient_error",
}


def _rows(payload):
    """``(name, new value, historical key path)`` of the printed comparison."""
    fit, res, val = payload["fit"], payload["residuals"], payload["validation"]
    red = payload["reduction"]
    rows = [
        (
            "full / active / grouped",
            f"{red['n_full']}/{red['n_active']}/{red['n_q']}",
            None,
        ),
        ("retained combinations", fit["n_retained"], ("retained_rank",)),
        (
            "numerical rank (rank_tol 1e-8)",
            fit["claims"]["numerical"]["rank"],
            ("structural_rank_rtol_1e8",),
        ),
        (
            "rank margin s_rank / (rank_tol s_max)",
            fit["claims"]["numerical"]["rank_margin"],
            None,
        ),
        ("chi2", fit["chi2"], ("fit_chi2",)),
        ("dof", fit["dof"], ("fit_dof",)),
        ("withheld RMS / I", res["withheld_rms_over_I"], ("heldout_rms_error_over_I",)),
        ("withheld max / I", res["withheld_max_over_I"], ("heldout_max_error_over_I",)),
        (
            "forward/direct max / I",
            res["max_forward_discrepancy_over_I"],
            ("max_forward_discrepancy_over_I",),
        ),
        (
            "max reconstruction / I",
            res["max_reconstruction_error_over_I"],
            ("max_reconstruction_error_over_I",),
        ),
    ]
    for name, entry in val.items():
        rows.append(
            (name, entry["value"], ("validation", HISTORICAL_KEYS.get(name, name)))
        )
    return rows


def _lookup(hist, path):
    node = hist
    for part in path or ():
        node = node.get(part) if isinstance(node, dict) else None
    return None if path is None else node


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return "-" if value is None else str(value)


def print_summary(payload, historical=None):
    """Print the Section 9.5 items; with ``historical``, a side-by-side table."""
    hist = json.loads(Path(historical).read_text()) if historical else None
    header = f"{'quantity':58s} {'new':>14s}"
    if hist is not None:
        header += f" {'historical':>14s} {'new - hist':>12s}"
    print(header)
    for name, value, key in _rows(payload):
        old = _lookup(hist, key) if hist is not None else None
        line = f"{name:58s} {_fmt(value):>14s}"
        if hist is not None:
            numeric = all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in (value, old)
            )
            diff = value - old if numeric else None
            line += f" {_fmt(old):>14s} {_fmt(diff):>12s}"
        print(line)
    comb = payload["combinations"]
    print("beta (true, fitted +- sigma, measured bias, cluster):")
    for i, row in enumerate(
        zip(
            comb["beta_true"],
            comb["beta_hat"],
            comb["beta_sigma"],
            comb["beta_bias_measured"],
            comb["cluster"],
        ),
        1,
    ):
        print(
            f"  {i:2d} {row[0]: .6f} {row[1]: .6f} +- {row[2]:.6f}  bias {row[3]: .2e}  cluster {row[4]}"
        )
    if hist is not None:
        _print_historical_modes(payload, hist)


def _print_historical_modes(payload, hist):
    """Singular values and rotation-invariant cluster norms against history."""
    s = np.asarray(payload["fit"]["singular_values"])
    s_h = np.asarray(hist["singular_values"])
    n = min(16, s.size, s_h.size)
    print(
        f"top {n} singular values: max relative difference {np.max(np.abs(s[:n] / s_h[:n] - 1)):.3g}"
    )
    rows = np.asarray(payload["combinations"]["rows"])
    rows_h = np.asarray(hist["retained_combinations"]["weights_on_full_tensor"])
    if rows.shape == rows_h.shape:
        angle = float(np.max(subspace_angles(rows.T, rows_h.T)))
        print(f"retained row space: max principal angle to history {angle:.3g} rad")
        beta = np.asarray(payload["combinations"]["beta_hat"])
        beta_h = np.asarray(hist["retained_combinations"]["reconstructed"])
        for cid in sorted(set(payload["combinations"]["cluster"])):
            members = [
                i for i, c in enumerate(payload["combinations"]["cluster"]) if c == cid
            ]
            groups = [[i] for i in members] if cid == -1 else [members]
            for g in groups:
                print(
                    f"  beta {[i + 1 for i in g]}: |beta_hat| new {np.linalg.norm(beta[g]):.6f} "
                    f"historical {np.linalg.norm(beta_h[g]):.6f}"
                )
