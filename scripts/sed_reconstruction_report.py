"""Outputs of the SED reconstruction example: payload, provenance and files.

``build_payload`` assembles the strict-JSON record (arrays as lists,
non-finite floats as strings), ``provenance`` records the package version,
git revision and status, SHA-256 of every ``syncmoments/**/*.py`` and of the
example scripts, and the environment; ``write_outputs`` writes
``<stem>.json``, ``<stem>_arrays.npz`` and ``<stem>_combinations.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

import syncmoments

import sed_reconstruction_direct as direct

SCRIPTS = Path(__file__).resolve().parent
SCRIPT_FILES = (
    "sed_reconstruction_example.py",
    "sed_reconstruction_setup.py",
    "sed_reconstruction_direct.py",
    "sed_reconstruction_checks.py",
    "sed_reconstruction_report.py",
    "sed_reconstruction_summary.py",
    "sed_reconstruction_figure.py",
)
HISTORICAL_COMMIT = "b39cf0c26b48dbbc89de03e4117f17ac5a35a72b"
LIMITS = (
    "Only the retained combinations beta are fitted numbers; the 60-entry "
    "representative carries n_retained independent numbers and is not a "
    "recovered tensor.",
    "Unresolved directions are unbounded without a declared coefficient bound; the "
    "illustrative prior is a declaration, not a data constraint.",
    "No declared discrepancy: the bias term is unbounded; the measured bias uses "
    "the synthetic truth and exists only in this example.",
    "The representative is not checked for positivity or realisability.",
    "The rescaling family B -> kappa B, gamma -> gamma/sqrt(kappa), source column "
    "/kappa gives different populations with the same direct continuum SED.",
    "The numerical rank and the retained set hold for these channels, mask, noise, "
    "metric, reference scales and thresholds only.",
    "Isotropic pitch, the finite angular family, the compact energy segment and "
    "the absence of absorption, transfer, calibration and excluded tails are "
    "declared assumptions of the example.",
)


def _plain(value):
    """Strict-JSON copy: arrays to lists, non-finite floats to strings."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "tolist"):
        return _plain(np.asarray(value).tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(repo, *args):
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    if out.returncode:
        return f"unavailable: {out.stderr.strip() or 'git error'}"
    return out.stdout.strip()


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def provenance():
    """Package version, git state, source hashes, environment and command line."""
    package = Path(syncmoments.__file__).resolve().parent
    repo = package.parent
    return {
        "syncmoments_version": syncmoments.__version__,
        "syncmoments_file": str(Path(syncmoments.__file__).resolve()),
        "git": {
            "repository": str(repo),
            "revision": _git(repo, "rev-parse", "HEAD"),
            "status_porcelain": _git(repo, "status", "--porcelain").splitlines(),
        },
        "source_sha256": {
            str(p.relative_to(package.parent)): _sha(p)
            for p in sorted(package.rglob("*.py"))
        },
        "script_sha256": {name: _sha(SCRIPTS / name) for name in SCRIPT_FILES},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: _version(name)
            for name in ("numpy", "scipy", "jax", "jaxlib", "equinox", "matplotlib")
        },
        "command": [sys.executable, *sys.argv],
        "historical_reference": {
            "path": "validation/sed_reconstruction_results.json (manuscript repository)",
            "commit": HISTORICAL_COMMIT,
            "use": "comparison only; never modified",
        },
    }


def _clusters(fit):
    """Cluster id of each retained combination (-1 when not in a cluster)."""
    retained = [i for i, c in enumerate(fit.classes) if c == "retained"]
    ids = {i: k for k, members in enumerate(fit.clusters) for i in members}
    return [ids.get(i, -1) for i in retained]


def _sed_block(r):
    fit, norm = r["fit"], direct.NORM
    sed, fwd = r["sed"], r["forward"]
    hat = np.asarray(fit.stokes_hat)[:, :3] / norm
    sig = np.asarray(fit.stokes_sigma)[:, :3] / norm
    data = np.asarray(r["data"].stokes)[:, :3] / norm
    I = sed[:, :1]
    withheld = np.ones(sed.shape[0], bool)
    withheld[r["fit_ch"]] = False
    fit_res, fwd_res = (hat - sed) / I, (fwd - sed) / I
    sed_block = {
        "units": "NORM (package units / NORM); I, Q, U",
        "frequency_hz": r["centres"],
        "fit_channel_indices": r["fit_ch"],
        "direct": sed,
        "forward": fwd,
        "stokes_hat": hat,
        "stokes_sigma": sig,
        "data_fitted": data[r["fit_ch"]],
    }
    residuals = {
        "definition": "(S - direct) / I_direct on every centre, I, Q, U",
        "withheld": withheld,
        "fit_residual_over_I": fit_res,
        "forward_residual_over_I": fwd_res,
        "withheld_rms_over_I": float(np.sqrt(np.mean(fit_res[withheld] ** 2))),
        "withheld_max_over_I": float(np.max(np.abs(fit_res[withheld]))),
        "max_forward_discrepancy_over_I": float(np.max(np.abs(fwd_res))),
        "max_reconstruction_error_over_I": float(np.max(np.abs(fit_res))),
    }
    return sed_block, residuals


def _slots(fit, comp):
    """Per-slot arrays over the full coordinates (synthetic example only)."""
    return {
        "labels": list(fit.labels),
        "truth": comp.truth,
        "projected_truth": comp.projected_truth_full,
        "representative": comp.representative,
        "unresolved_truth": comp.unresolved_truth,
        "bias_measured": comp.bias_measured,
        "noise_part": comp.noise_part,
        "noise_sigma_retained_only": fit.standard_errors(),
        "slot_status": list(fit.slot_status()),
        "note": "synthetic example only; the representative has n_retained "
        "independent numbers; a zero projection is not a physical zero",
    }


def _combinations(fit, comp):
    """The retained combinations: rows over the full slots, truth, fit, errors."""
    return {
        "rows": fit.combination_rows,
        "labels": list(fit.combination_labels()),
        "beta_true": comp.beta_true,
        "beta_hat": fit.beta_hat,
        "beta_sigma": fit.beta_sigma,
        "beta_bias_measured": comp.beta_bias_measured,
        "beta_noise_part": comp.beta_noise_part,
        "cluster": _clusters(fit),
        "note": "inside a cluster only the subspace is determined",
    }


def _prior(r):
    """The illustrative declared prior (design Section 9.4); not in the main result."""
    unresolved_p = np.asarray(r["fit_prior"].unresolved.value)
    unresolved_t = np.abs(np.asarray(r["comparison"].unresolved_truth))
    bound = np.asarray(r["bound"].value)
    return {
        "coefficient_bound": r["bound"].to_dict(),
        "unresolved": r["fit_prior"].unresolved.to_dict(),
        "covers_unresolved_truth": bool(np.all(unresolved_p >= unresolved_t)),
        "bound_covers_truth": bool(np.all(bound >= np.abs(r["a_true"]))),
        "note": "illustrative declared prior on the Support and the source-column "
        "ratio; not a data constraint",
    }


def build_payload(r, *, config):
    """The strict-JSON record of one run (design Section 9.3, step 8)."""
    fit, comp = r["fit"], r["comparison"]
    fit_dict = fit.to_dict()
    fit_dict.pop("reduction", None)  # stored once at the top level
    sed_block, residuals = _sed_block(r)
    payload = {
        "configuration": config,
        "reduction": r["reduction"].to_dict(),
        "fit": fit_dict,
        "claims": fit_dict["claims"],
        "slots": _slots(fit, comp),
        "combinations": _combinations(fit, comp),
        "sed": sed_block,
        "residuals": residuals,
        "validation": r["validation"],
        "alternative_population": r["alternative"],
        "illustrative_prior": _prior(r),
        "limits": list(LIMITS),
        "provenance": provenance(),
    }
    out = _plain(payload)
    json.dumps(out, allow_nan=False)
    return out


def output_paths(stem):
    """The three machine-readable files of ``stem``."""
    stem = Path(stem)
    return {
        "json": stem.parent / f"{stem.name}.json",
        "npz": stem.parent / f"{stem.name}_arrays.npz",
        "csv": stem.parent / f"{stem.name}_combinations.csv",
    }


def _arrays(r):
    """Operators and arrays saved to ``stem_arrays.npz``."""
    fit, red = r["fit"], r["reduction"]
    d = fit.directions
    arrays = {name: getattr(red, name) for name in ("C", "T", "H", "L", "Q")}
    arrays.update(
        K=fit.estimator,
        K_beta=fit.beta_estimator,
        Pi=fit.projector,
        Cov=fit.covariance,
        B=fit.combination_rows,
        W_retained=d.retained,
        W_weak=d.weak,
        W_numerical_null=d.numerical_null,
        W_analytic_null=d.analytic_null,
        singular_values=fit.singular_values,
        noise=r["noise"],
        sigma_kept=r["sigma_kept"],
        kept_rows=np.asarray(fit.kept_rows),
    )
    return {k: np.asarray(v) for k, v in arrays.items()}


CSV_COLUMNS = ("beta_true", "beta_hat", "beta_sigma", "beta_bias_measured", "cluster")


def write_outputs(payload, r, stem):
    """``stem.json`` (strict), ``stem_arrays.npz`` (operators), ``stem_combinations.csv``."""
    paths = output_paths(stem)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(paths["npz"], **_arrays(r))
    comb = payload["combinations"]
    with paths["csv"].open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["i", *CSV_COLUMNS, "top_terms"])
        columns = [comb[name] for name in CSV_COLUMNS] + [comb["labels"]]
        for i, row in enumerate(zip(*columns), start=1):
            writer.writerow([i, *row])
