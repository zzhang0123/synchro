"""SED fit of the smooth correlated continuum population through the public API.

Usage:
  PYTHONPATH=. python scripts/sed_reconstruction_example.py [--quick] [--out DIR]
      [--no-figure] [--historical PATH]

Ports the example of main.tex Section 5.3 / Appendix C.1 (research reference
``validation/sed_reconstruction.py`` of the manuscript, package b39cf0c):

1. Before any data: ``reduce_response`` groups the 60 full coordinates
   (52 active response columns + 8 zero-response depth rows) into 30 exact
   groups ``q = T a`` (continuum scaling identity; no data used).
2. Mock data: an independent NumPy/SciPy full-population integration
   (``sed_reconstruction_direct.py``), 72 bump channels over 0.4-3 GHz with
   4 % half-width, I, Q and U of 24 fitted centres with Gaussian noise
   ``sigma = 0.01 I_direct`` (seed 20260925); 48 centres are withheld.
3. Fit: ``fit_combinations`` with ``max_sigma = 0.1`` in the default metric
   (Euclidean in the full coordinates ``a``, amplitude fitted as ``a_0``).
   Only the retained combinations ``beta = B a`` are fitted numbers; the
   60-entry representative carries ``n_retained`` independent numbers.

Needs SciPy; Matplotlib is optional (the figure is skipped without it). No TeX.
Every number printed or saved is a finite check of one configuration and seed,
not a certificate of physical adequacy, unique moment recovery or exact rank.
``--quick`` runs a reduced configuration that is not comparable with history.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

import syncmoments  # noqa: F401  (enables float64)
from syncmoments.model import predict
from syncmoments.model.fit import coefficient_bounds, reduce_response

import sed_reconstruction_direct as direct
import sed_reconstruction_report as report
import sed_reconstruction_summary as summary
from sed_reconstruction_checks import (
    basic_checks,
    grouping_checks,
    refinement,
    rescaling,
)
from sed_reconstruction_setup import (
    AMPLITUDE_BOUND,
    FULL,
    KAPPA,
    MANUSCRIPT_REPRESENTATIVES,
    MAX_SIGMA,
    NOISE_FRACTION,
    QUICK,
    QUICK_NOTE,
    SEED,
    STEM,
    combinations_fit,
    make_basis,
    mock_data,
    truth,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help=QUICK_NOTE)
    parser.add_argument(
        "--out", default=str(Path(__file__).resolve().parents[1] / "figures")
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--historical", default=None, help="historical results JSON")
    return parser.parse_args(argv)


def prepare(cfg):
    """Before any data: basis, reductions, truth by two routes and the forward check."""
    centres = np.geomspace(0.4e9, 3e9, cfg.n_centres)
    basis = make_basis(centres, *cfg.basis)
    reduction = reduce_response(
        basis, relations="continuum", representatives=MANUSCRIPT_REPRESENTATIVES
    )
    layout = reduction.layout
    print(
        f"before any data: {layout.n_full} full -> {layout.n_active} active -> "
        f"{reduction.n_q} grouped (max|C - H T|/max|C| = {reduction.max_delta_C:.2e})",
        flush=True,
    )
    a_true, a_closed, moments = truth(layout, basis.reference, cfg.nodes)
    forward = np.asarray(predict(basis, moments, amplitude=1.0).stokes)
    gap = np.max(np.abs(forward - (np.asarray(reduction.C) @ a_true).reshape(-1, 4)))
    return {
        "cfg": cfg,
        "centres": centres,
        "fit_ch": np.rint(np.linspace(0, cfg.n_centres - 1, cfg.n_fit)).astype(int),
        "basis": basis,
        "reduction": reduction,
        "ungrouped": reduce_response(basis, relations="none"),
        "a_true": a_true,
        "a_closed": a_closed,
        "forward": forward[:, :3] / direct.NORM,
        "forward_gap": float(gap / np.max(np.abs(forward))),
    }


def simulate_and_fit(st):
    """Direct integration, mock data and the fits (grouped, ungrouped, with prior)."""
    cfg, basis, reduction = st["cfg"], st["basis"], st["reduction"]
    f_eval, f_error = direct.f_interpolator()
    sed = direct.direct_sed(st["centres"], f_eval, cfg.nodes, *cfg.direct)
    coarse = direct.direct_sed(
        st["centres"], f_eval, cfg.coarse_nodes, *cfg.coarse_direct
    )
    data, noise, sigma_kept = mock_data(sed, st["fit_ch"])
    fit = combinations_fit(basis, data, reduction)
    delta = (sed - st["forward"]) @ np.eye(3, 4) * direct.NORM  # V: zero, masked
    bound = coefficient_bounds(
        reduction.layout, basis.support, amplitude_bound=AMPLITUDE_BOUND
    )
    return {
        **st,
        "f_eval": f_eval,
        "f_error": f_error,
        "sed": sed,
        "coarse": coarse,
        "data": data,
        "noise": noise,
        "sigma_kept": sigma_kept,
        "fit": fit,
        "fit_ungrouped": combinations_fit(basis, data, st["ungrouped"]),
        "comparison": fit.against_truth(st["a_true"], discrepancy=delta, noise=noise),
        "bound": bound,
        "fit_prior": combinations_fit(basis, data, reduction, bound),
    }


def run(cfg):
    """Every step of the example; returns the arrays and scalars for the report."""
    st = simulate_and_fit(prepare(cfg))
    rescale_err, alternative = rescaling(
        cfg,
        st["centres"],
        st["f_eval"],
        st["sed"],
        st["reduction"].layout,
        st["a_true"],
        st["fit"],
    )
    val = {**basic_checks(st), **grouping_checks(st)}
    val["exact_scaling_invariance_over_I"] = rescale_err
    val.update(
        refinement(
            cfg,
            st["centres"],
            st["data"],
            st["fit"],
            st["reduction"],
            st["a_true"],
            st["sed"],
        )
    )
    return {**st, "validation": val, "alternative": alternative}


def main(argv=None):
    args = parse_args(argv)
    cfg = QUICK if args.quick else FULL
    result = run(cfg)
    payload = report.build_payload(result, config=_configuration(cfg))
    out = Path(args.out)
    report.write_outputs(payload, result, out / STEM)
    if not args.no_figure:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib not available: figure skipped")
        else:
            import sed_reconstruction_figure as figure

            figure.plot(payload, out / STEM)
    summary.print_summary(payload, args.historical)
    return payload


def _configuration(cfg):
    return {
        **asdict(cfg),
        "note": (
            QUICK_NOTE if cfg.quick else "full configuration of main.tex Section 5.3"
        ),
        "reference": {
            "gamma0": direct.GAMMA0,
            "B0_gauss": direct.B0,
            "depth0_rad_m2": direct.DEPTH0,
            "scales": list(direct.SCALES),
        },
        "support": direct.SUPPORT,
        "channels": "bump, unit_integral, fractional half-width "
        f"{direct.HALF_WIDTH}, geomspace(0.4e9, 3e9, {cfg.n_centres})",
        "truncation": [0, 2, 2],
        "phase": "TaylorPhase(2)",
        "noise_fraction_I": NOISE_FRACTION,
        "seed": SEED,
        "max_sigma": MAX_SIGMA,
        "kappa": KAPPA,
        "amplitude_bound_illustrative": AMPLITUDE_BOUND,
        "representatives": [list(r) for r in MANUSCRIPT_REPRESENTATIVES],
        "metric": "Euclidean in the full coordinates a (default)",
        "amplitude": "fitted as a_0 (source-column ratio N_src/N_* = 1 in the truth)",
        "norm_package_units": direct.NORM,
    }


if __name__ == "__main__":
    main()
